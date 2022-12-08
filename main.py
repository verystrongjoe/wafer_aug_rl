import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from datasets.loaders import balanced_loader
from datasets.transforms import WM811KTransform
from datasets.wm811k import WM811K
from metrics import MultiAccuracy, MultiAUPRC, MultiF1Score, MultiRecall, MultiPrecision, TopKAccuracy
from models.basic import CNN
from trainer import Trainer
from utils import get_args, pre_requisite, print_metric, make_description
import os
import random
import numpy as np

if __name__ == '__main__':
    # 1. init
    args = get_args()
    run = pre_requisite(args)

    # 2. set model
    model = CNN().to(args.num_gpu)  # todo: change model type
    summary(model, (1, 96, 96), batch_size=16)

    # 3. set model parameter and setting
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    fn_loss = torch.nn.CrossEntropyLoss()  # 비용 함수에 소프트맥스 함수 포함되어져 있음.
    criterions = {
        'MultiAccuracy': MultiAccuracy(num_classes=args.num_classes),
        'MultiAUPRC': MultiAUPRC(num_classes=args.num_classes),
        'MultiF1Score': MultiF1Score(num_classes=args.num_classes, average='macro'),
        'MultiRecall': MultiRecall(num_classes=args.num_classes, average='macro'),
        'MultiPrecision': MultiPrecision(num_classes=args.num_classes, average='macro'),
        'TopKAccuracy': TopKAccuracy(num_classes=args.num_classes, k=3),
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

