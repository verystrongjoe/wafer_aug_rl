import os
import sys
from os.path import dirname, realpath
import pathlib
import logging
from argparse import Namespace
import datetime
import numpy as np


class DeepAugment:
    def __init__(self, config):
        """Initializes DeepAugment object

        Does following steps:
            1. load and preprocess data
            2. create child model
            3. create controller
            4. create notebook (for recording trainings)
            5. do initial training
            6. create objective function
            7. evaluate objective function without augmentation

        Args:
            images (numpy.array/str): array with shape (n_images, dim1, dim2 , channel_size), or a string with name of keras-dataset (cifar10, fashion_mnsit)
            labels (numpy.array): labels of images, array with shape (n_images) where each element is an integer from 0 to number of classes
            config (dict): dictionary of configurations, for updating the default config which is:
        """
        self.config = vars(config)
        self.iterated = 0  # keep tracks how many times optimizer iterated
        self._load_and_preprocess_data(config.images)
        print('data prepared.')

        # define main objects
        self.child_model = ChildCNN(self.num_classes, Namespace(**self.config))
        print('model created.')
        self.controller = Controller(self.config)
        self.notebook = Notebook(self.config)

        if self.config["child_first_train_epochs"] > 0:
            self._do_initial_training()
        print('model init trained.')
        self.child_model.save_pre_aug_weights()
        self.objective_func = Objective(self.data, self.child_model, self.notebook, self.config)
        self._evaluate_objective_func_without_augmentation()
        print('model is evaluated firstly.')

    def optimize(self, iterations=300):
        """Optimize objective function hyperparameters using controller and child model
        Args: iterations (int): number of optimization iterations, which the child model will be run
        Returns: pandas.DataFrame: top policies (with highest expected accuracy increase)
        """
        # iterate optimizer
        for trial_no in range(self.iterated + 1, self.iterated + iterations + 1):
            trial_hyperparams = self.controller.ask()
            print("trial:", trial_no, "\n", trial_hyperparams)
            f_val = self.objective_func.evaluate(trial_no, trial_hyperparams)
            self.controller.tell(trial_hyperparams, f_val)
        self.iterated += iterations  # update number of previous iterations
        self.top_policies = self.notebook.get_top_policies(20)
        self.notebook.output_top_policies()
        print("\ntop policies are:\n", self.top_policies)
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

    def _load_and_preprocess_data(self, image):
        """Loads and preprocesses data
        Records `input_shape`, `data`, and `num_classes` into `self
        Args:
            images (numpy.array/str): array with shape (n_images, dim1, dim2 , channel_size), or a string with name of keras-dataset (cifar10, fashion_mnsit)
            labels (numpy.array): labels of images, array with shape (n_images) where each element is an integer from 0 to number of classes
        """
        # todo : 데이터 로드부분 wm811k 할때 변경해서 사용하자. 지금은 cifar10
        X_train, y_train, X_val, y_val, num_classes = DataOp.load(image)

        self.data = {
            "X_train": np.asarray(X_train),
            "y_train": np.asarray(y_train),
            "X_val_seed": np.asarray(X_val),
            "y_val_seed": np.asarray(y_val)
        }
        self.num_classes = num_classes

    def _do_initial_training(self):
        """Do the first training without augmentations

        Training weights will be used as based to further child model trainings
        """
        history = self.child_model.fit(
            self.data, epochs=self.config["child_first_train_epochs"]
        )
        self.notebook.record(
            -1, ["first", 0.0, "first", 0.0, "first", 0.0, 0.0], 1, None, history
        )

    @timeit
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


def main(config):
    deepaug = DeepAugment(config)
    best_policies = deepaug.optimize(config.opt_iterations)
    print(best_policies)


if __name__ == "__main__":
    config = WaPIRLConfig.parse_arguments()
    config.pre_backbone_aug_weights_path = "pre_backbone_aug_weights.h5"
    config.pre_classifier_aug_weights_path = "pre_classifier_aug_weights.h5"
    config.notebook_path = f"{EXPERIMENT_FOLDER_PATH}/notebook.csv"

    logfile = os.path.join(config.checkpoint_dir, 'main.log')
    config.logging = get_logger(stream=False, logfile=logfile)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # todo : 0번 gpu로 고정
    torch.cuda.set_device(0)  # todo : 0번 gpu로 고정
    main(config)
