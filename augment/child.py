import torch
import numpy as np
from datasets.loaders import balanced_loader
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models.basic import CNN
from trainer import Trainer
from datasets.transforms import WM811KTransformMultiple, WM811KTransform
from datasets.wm811k import WM811K


class ChildCNN:
    def __init__(self, args):
        # todo: WaPIRL 에서 사용 용도 파악 필요
        self.args = args
        self.in_channels = int(args.decouple_input) + 1
        self.num_classes = args.num_classes
        self.train_dataloader = None
        self.valid_dataloader = None
        # todo : basic.py의  CNN or CNNDeepAugmentBasic 둘다 선택되도록 변경
        self.model = CNN(args)
        self.model.to(args.num_gpu)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.trainer = Trainer(args, self.model)

    def evaluate_with_refreshed_validation_set(self, data):
        X_val_backup = data["X_val_backup"]
        y_val_backup = data["y_val_backup"]
        # FIXME if dataset is smaller than 5000, an error will occur
        ivb = np.random.choice(len(X_val_backup), 5000, False)
        X_val_backup = X_val_backup[ivb]
        y_val_backup = y_val_backup[ivb]
        scores = self.model.evaluate(X_val_backup, y_val_backup, verbose=2)
        test_loss = scores[0]
        test_acc = scores[1]
        self.args.logger.info(f"Test loss:{test_loss}, Test accuracy:{test_acc}")
        return test_loss, test_acc

    def fit(self, trial_hyperparams, epochs=None, aug_yn=True):
        if epochs is None:
            epochs = self.args.child_epochs

        test_transform = WM811KTransform(size=(self.args.input_size_xy, self.args.input_size_xy), mode='test')
        if aug_yn:
            train_transform = WM811KTransformMultiple(self.args, trial_hyperparams)
        else:
            train_transform = test_transform

        train_set = WM811K('./data/wm811k/labeled/train/',
                           transform=train_transform,
                           decouple_input=self.args.decouple_input)
        # todo: 기존 DeepAugment에서는 1000개 샘플 뽑아 개수를 줄였음. 오래 걸리게 되면 여기도 조정 필요
        valid_set = WM811K('./data/wm811k/labeled/valid/',
                           transform=test_transform,
                           decouple_input=self.args.decouple_input)

        train_loader = balanced_loader(train_set, self.args.child_batch_size, num_workers=self.args.num_workers,
                                       shuffle=False, pin_memory=False)
        valid_loader = DataLoader(valid_set, self.args.child_batch_size, num_workers=self.args.num_workers,
                                  shuffle=True, drop_last=False, pin_memory=False)

        train_results, valid_results = [], []
        for epoch in range(epochs):
            train_history = self.trainer.train_epoch(train_loader)
            valid_history = self.trainer.valid_epoch(valid_loader)
            train_results.append(train_history)
            valid_results.append(valid_history)

        return train_results, valid_results

