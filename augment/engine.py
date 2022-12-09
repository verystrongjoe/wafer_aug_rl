import os
import sys
from os.path import dirname, realpath
import pathlib
import logging
from argparse import Namespace
import datetime
import numpy as np
from augment.child import ChildCNN
from augment.module import Controller, Objective, Notebook
from augment.image_generator import deepaugment_image_generator
from utils import get_args, pre_requisite, print_metric, make_description
import torch.multiprocessing


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
            self.args.logger.info(f"trial_no: {trial_no}, trial_hyperparams: {trial_hyperparams}")
            f_val = self.objective_func.evaluate(trial_no, trial_hyperparams)
            self.controller.tell(trial_hyperparams, f_val)
        self.iterated += iterations  # update number of previous iterations
        self.top_policies = self.notebook.get_top_policies(20)
        self.notebook.output_top_policies()
        self.args.logger.info("\ntop policies are:\n", self.top_policies)
        return self.top_policies

    def image_generator_with_top_policies(self, images, labels, batch_size=None):
        """
        Args:
            images (numpy.array): array with shape (N,dim,dim,channek-size)
            labels (numpy.array): array with shape (N), where each eleemnt is an integer from 0 to num_classes-1
            batch_size (int): batch size of the generator on demand
        Returns:
            generator: generator for augmented images
        """
        if batch_size is None:
            batch_size = self.config["child_batch_size"]

        top_policies_list = self.top_policies[
            ['A_aug1_type', 'A_aug1_magnitude', 'A_aug2_type', 'A_aug2_magnitude',
             'B_aug1_type', 'B_aug1_magnitude', 'B_aug2_type', 'B_aug2_magnitude',
             'C_aug1_type', 'C_aug1_magnitude', 'C_aug2_type', 'C_aug2_magnitude',
             'D_aug1_type', 'D_aug1_magnitude', 'D_aug2_type', 'D_aug2_magnitude',
             'E_aug1_type', 'E_aug1_magnitude', 'E_aug2_type', 'E_aug2_magnitude']
        ].to_dict(orient="records")

        return deepaugment_image_generator(images, labels, top_policies_list, batch_size=batch_size)

    def _do_initial_training(self):
        """Do the first training without augmentations

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
    torch.multiprocessing.set_sharing_strategy('file_system')
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    args = get_args()
    run = pre_requisite(args)

    # logging params
    args.logger.info(args)

    # declare augmenter
    deepaug = DeepAugment(args)

    # run
    best_policies = deepaug.optimize(args.opt_iterations)

    # log
    args.logger.info(best_policies)

    best_policies.to_csv('best.csv', index=False)


