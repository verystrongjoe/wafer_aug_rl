import pandas as pd
from trainer import Trainer
from utils import get_args, pre_requisite, print_metric, make_description, augment_by_policy_wapirl
from datasets.transforms import WM811KTransform
from datasets.wm811k import WM811K
from torch.utils.data import DataLoader
from datasets.dataset import SimpleDataset
import numpy as np
from models.advanced import AdvancedCNN
from multiprocessing import Pool
from itertools import product
import torch
import os
import random


if __name__ == '__main__':

    # 1. init
    args = get_args()
    run = pre_requisite(args)
    batch_size = args.child_batch_size
    best_policies = pd.read_csv('best_policy.csv')
    print(best_policies)
    test_transform = WM811KTransform(size=(args.input_size_xy, args.input_size_xy), mode='test')

    train_set = WM811K('./data/wm811k/labeled/train/', proportion=args.label_proportion, decouple_input=args.decouple_input)
    # todo: 기존 DeepAugment에서는 1000개 샘플 뽑아 개수를 줄였음. 오래 걸리게 되면 여기도 조정 필요
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

    model = AdvancedCNN(args)
    trainer = Trainer(args, model)

    # 3. set model parameter and setting
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.logger.info('train data is augmendted by best policy. start...')
    # generate train dataset augmened by best policy above
    with Pool(args.num_workers) as p:
        r = p.starmap(augment_by_policy_wapirl, product(train_set.samples, [eval_policy], [args]))

    # for sample, eval_policy, args in product(train_set.samples, [eval_policy], [args]):
    #     augment_by_policy_wapirl(sample, eval_policy, args)

    Xs, ys = [], []
    for item in r:
         Xs.extend(item['X_train'])
         ys.extend(item['y_train'])

    train_set = SimpleDataset(Xs,ys)
    train_loader = DataLoader(train_set, args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False,
                             pin_memory=False)
    args.logger.info('train data is augmendted by best policy. end...')

    best_valid_loss, best_epoch = float('inf'), 0

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