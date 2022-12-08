import os
import sys
import torch
import torch.nn as nn
import numpy as np
import wandb
from itertools import product
from datasets.wm811k import WM811K
from datasets.transforms import WM811KTransform
from torchsummary import summary
from torch.utils.data import DataLoader
from datasets.loaders import balanced_loader
from metrics import MultiAccuracy, MultiAUPRC, MultiF1Score, MultiRecall, MultiPrecision, TopKAccuracy
from trainer import Trainer
from utils import get_args, pre_requisite, print_metric, make_description
from models.basic import CNN

if __name__ == '__main__':
    # 1. init
    args = get_args()
    run = pre_requisite(args)

    # 2. set model
    model = CNN().to(args.num_cpu)  # todo: change model type
    summary(model, (1, 96, 96), batch_size=16)

    # 3. set model parameter and setting
    torch.manual_seed(args.seed)

    fn_loss = torch.nn.CrossEntropyLoss()  # 비용 함수에 소프트맥스 함수 포함되어져 있음.
    criterions = {
        'MultiAccuracy': MultiAccuracy(num_classes=9),
        'MultiAUPRC': MultiAUPRC(num_classes=9),
        'MultiF1Score': MultiF1Score(num_classes=9, average='macro'),
        'MultiRecall': MultiRecall(num_classes=9, average='macro'),
        'MultiPrecision': MultiPrecision(num_classes=9, average='macro'),
        'TopKAccuracy': TopKAccuracy(num_classes=9, k=3),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_transform = WM811KTransform(size=(args.input_size_xy, args.input_size_xy), mode='test')
    test_transform = WM811KTransform(size=(args.input_size_xy, args.input_size_xy), mode='test')
    train_set = WM811K('./data/wm811k/labeled/train/',
                       transform=train_transform,
                       decouple_input=args.decouple_input)
    valid_set = WM811K('./data/wm811k/labeled/valid/',
                       transform=test_transform,
                       decouple_input=args.decouple_input)
    test_set = WM811K('./data/wm811k/labeled/test/',
                      transform=test_transform,
                      decouple_input=args.decouple_input)

    train_loader = balanced_loader(train_set, args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=False)
    valid_loader = DataLoader(valid_set, args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False, pin_memory=False)
    test_loader = DataLoader(test_set, args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False, pin_memory=False)

    best_valid_loss, best_epoch = float('inf'), 0
    trainer = Trainer(args, model, optimizer, fn_loss, criterions)

    for epoch in range(args.epochs):
        wandb_history = {}
        train_history = trainer.train_epoch(train_loader)
        valid_history = trainer.valid_epoch(valid_loader)
        print_metric(epoch, args, train_history, valid_history)   ## log into console

        if valid_history['loss'] < best_valid_loss:
            best_valid_loss = valid_history['loss']
            best_epoch = epoch
            trainer.save_checkpoint(epoch)
        run.log(make_description(train_history, 'train'))
        run.log(make_description(train_history, 'valid'))

    # load best model in all epochs
    trainer.load_checkpoint(best_epoch)
    test_history = trainer.valid_epoch(test_loader)
    run.log(make_description(test_history, 'test'))
