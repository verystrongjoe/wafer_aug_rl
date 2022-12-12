import logging
import os
import random
from itertools import product
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import DataLoader

from augment.child import ChildCNN
from augment.module import Controller, Objective, Notebook
from datasets.dataset import SimpleDataset
from datasets.transforms import WM811KTransform
from datasets.wm811k import WM811K
from models.advanced import AdvancedCNN
from trainer import Trainer
from utils import get_args, pre_requisite, print_metric, make_description, augment_by_policy_wapirl


class DeepAugment:
    def __init__(self, args):
        """Initializes DeepAugment object

        Does following steps:
            1. load and preprocess data
            2. create child model
            3. create controller
            4. create notebook (for recording trainings)
            5. do initial training
            6. create objective function
            7. evaluate objective function without augmentation
        """
        self.args = args
        self.iterated = 0  # keep tracks how many times optimizer iterated

        # define main objects
        self.child_model = ChildCNN(args)
        self.args.logger.info('model created.')
        self.controller = Controller(args)
        self.notebook = Notebook(args)

        if args.child_first_train_epochs > 0:
            self._do_initial_training()
        self.args.logger.info('model init trained.')
        self.child_model.trainer.save_checkpoint(0)
        self.objective_func = Objective(args, self.child_model, self.notebook)
        self._evaluate_objective_func_without_augmentation()
        self.args.logger.info('model is evaluated without agumentation for the first time.')

    def optimize(self, iterations=300):
        """Optimize objective function hyperparameters using controller and child model
        Args: iterations (int): number of optimization iterations, which the child model will be run
        Returns: pandas.DataFrame: top policies (with highest expected accuracy increase)
        """
        # iterate optimizer
        for trial_no in range(self.iterated + 1, self.iterated + iterations + 1):
            trial_hyperparams = self.controller.ask()
            f_val = self.objective_func.evaluate(trial_no, trial_hyperparams)
            self.controller.tell(trial_hyperparams, f_val)
            self.args.logger.info(f"trial_no: {trial_no}, trial_hyperparams: {trial_hyperparams}, f_val : {f_val}")
            self.args.run.log({"f_val": f_val})  # send reward to wandb
        self.iterated += iterations  # update number of previous iterations
        self.top_policies = self.notebook.get_top_policies(20)
        self.notebook.output_top_policies()
        self.args.logger.info("\ntop policies are:\n", self.top_policies)
        return self.top_policies

    # def image_generator_with_top_policies(self, images, labels, batch_size=None):
    #     """
    #     Args:
    #         images (numpy.array): array with shape (N,dim,dim,channek-size)
    #         labels (numpy.array): array with shape (N), where each eleemnt is an integer from 0 to num_classes-1
    #         batch_size (int): batch size of the generator on demand
    #     Returns:
    #         generator: generator for augmented images
    #     """
    #     if batch_size is None:
    #         batch_size = self.args.child_batch_size
    #
    #     top_policies_list = self.top_policies[
    #         ['A_aug1_type', 'A_aug1_magnitude', 'A_aug2_type', 'A_aug2_magnitude',
    #          'B_aug1_type', 'B_aug1_magnitude', 'B_aug2_type', 'B_aug2_magnitude',
    #          'C_aug1_type', 'C_aug1_magnitude', 'C_aug2_type', 'C_aug2_magnitude',
    #          'D_aug1_type', 'D_aug1_magnitude', 'D_aug2_type', 'D_aug2_magnitude',
    #          'E_aug1_type', 'E_aug1_magnitude', 'E_aug2_type', 'E_aug2_magnitude']
    #     ].to_dict(orient="records")
    #
    #     return deepaugment_image_generator(images, labels, top_policies_list, batch_size=batch_size)

    def _do_initial_training(self):
        """Do the first training without a ugmentations

        Training weights will be used as based to further child model trainings
        """
        history = self.child_model.fit(
            self.data, epochs=self.args.child_first_train_epochs, aug_yn=False
        )
        self.notebook.record(
            -1, ["first", 0.0, "first", 0.0, "first", 0.0, 0.0], 1, None, history
        )

    def _evaluate_objective_func_without_augmentation(self):
        """Find out what would be the accuracy if augmentation are not applied
        """
        no_aug_hyperparams = ["rotate", 0.0, "rotate", 0.0,
                              "rotate", 0.0, "rotate", 0.0,
                              "rotate", 0.0, "rotate", 0.0,
                              "rotate", 0.0, "rotate", 0.0,
                              "rotate", 0.0, "rotate", 0.0]
        f_val = self.objective_func.evaluate(0, no_aug_hyperparams)
        self.controller.tell(no_aug_hyperparams, f_val)


if __name__ == '__main__':

    ####################################################################################################################
    # initialize
    ####################################################################################################################
    torch.multiprocessing.set_sharing_strategy('file_system')
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    args = get_args()
    pre_requisite(args)
    args.logger.info(f"args : {args}")  # logging params

    ####################################################################################################################
    # set seed
    ####################################################################################################################
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ####################################################################################################################
    # find best Policy
    ####################################################################################################################
    deepaug = DeepAugment(args)     # declare augmenter
    best_policies = deepaug.optimize(args.opt_iterations)       # run
    args.logger.info(best_policies)     # log
    best_policies.to_csv(f'{args.notebook_path}/best_policy.csv', index=False)

    ####################################################################################################################
    # augment data
    ####################################################################################################################
    test_transform = WM811KTransform(size=(args.input_size_xy, args.input_size_xy), mode='test')
    train_set = WM811K('./data/wm811k/labeled/train/', proportion=args.label_proportion, decouple_input=args.decouple_input)
    # todo: DeepAugment에선 1000개 샘플 뽑아 개수 줄여 속도 향상 및 오버피팅 방지
    valid_set = WM811K('./data/wm811k/labeled/valid/', transform=test_transform, decouple_input=args.decouple_input)
    test_set = WM811K('./data/wm811k/labeled/test/', transform=test_transform, decouple_input=args.decouple_input)
    valid_loader = DataLoader(valid_set, args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False,
                             pin_memory=False)
    test_loader = DataLoader(test_set, args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False,
                             pin_memory=False)
    eval_policy = best_policies[
        ['A_aug1_type', 'A_aug1_magnitude', 'A_aug2_type', 'A_aug2_magnitude',
         'B_aug1_type', 'B_aug1_magnitude', 'B_aug2_type', 'B_aug2_magnitude',
         'C_aug1_type', 'C_aug1_magnitude', 'C_aug2_type', 'C_aug2_magnitude',
         'D_aug1_type', 'D_aug1_magnitude', 'D_aug2_type', 'D_aug2_magnitude',
         'E_aug1_type', 'E_aug1_magnitude', 'E_aug2_type', 'E_aug2_magnitude']
    ].to_dict(orient="records")

    ####################################################################################################################
    # create full model and populate train dataset
    ####################################################################################################################
    model = AdvancedCNN(args)
    trainer = Trainer(args, model)

    args.logger.info(f'train data of {len(train_set)} is augmented by best policy. start...')

    # generate train dataset augmented by best policy above
    if True:
        with Pool(args.num_workers) as p:
            r = p.starmap(augment_by_policy_wapirl, product(train_set.samples, [eval_policy], [args]))
    else:
        for sample, eval_policy, args in product(train_set.samples, [eval_policy], [args]):
            augment_by_policy_wapirl(sample, eval_policy, args)
    args.logger.info('train data is augmented by best policy. end...1')

    Xs, ys = [], []
    for item in r:
         Xs.extend(item['X_train'])
         ys.extend(item['y_train'])

    train_set = SimpleDataset(Xs,ys)
    train_loader = DataLoader(train_set, args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False,
                             pin_memory=False)
    args.logger.info('train data is augmented by best policy. end...2')

    best_valid_loss, best_epoch = float('inf'), 0

    for epoch in range(args.epochs):
        train_history = trainer.train_epoch(train_loader)
        valid_history = trainer.valid_epoch(valid_loader)
        print_metric(epoch, args, train_history, valid_history)   ## log into console

        if valid_history['loss'] < best_valid_loss:
            best_valid_loss = valid_history['loss']
            best_epoch = epoch
            trainer.save_checkpoint(epoch)
        args.run.log(make_description(train_history, 'train'))
        args.run.log(make_description(train_history, 'valid'))

    # load best model in all epochs
    trainer.load_checkpoint(best_epoch)
    test_history = trainer.valid_epoch(test_loader)
    args.run.log(make_description(test_history, 'test'))

