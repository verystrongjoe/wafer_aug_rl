import pandas as pd
from augment import image_generator
from trainer import Trainer
from utils import get_args, pre_requisite, print_metric, make_description
from datasets.transforms import WM811KTransform
from datasets.wm811k import WM811K
from torch.utils.data import DataLoader
import numpy as np
from models.basic import CNN

def augment_type_chooser(args):
    return np.random.choice(args.aug_types)


def deepaugment_image_generator(X, y, policy, batch_size=64, augment_chance=0.5):
    # Yields batch of images after applying random augmentations from the policy
    # Each image is augmented by 50% chance. If augmented, one of the augment-chain in the policy is applied. Which augment-chain to apply is chosen randomly.
    if type(policy) == str:
        if policy=="random":
            policy=[]
            for i in range(20):
                policy.append(
                    {
                        "aug1_type": augment_type_chooser(),
                        "aug1_magnitude":np.random.rand(),
                        "aug2_type": augment_type_chooser(),
                        "aug2_magnitude": np.random.rand(),
                        "portion":np.random.rand()
                    }
                )
        else:
            policy_df = pd.read_csv(policy)
            policy_df = policy_df[
                ["aug1_type", "aug1_magnitude", "aug2_type", "aug2_magnitude"]
            ]
            policy = policy_df.to_dict(orient="records") # todo : 파라미터 처음 봄.

    print(f"Policies are:  {policy}")

    while True:
        ix = np.arange(len(X))
        np.random.shuffle(ix)
        for i in range(len(X) // batch_size):
            _ix = ix[i * batch_size : (i + 1) * batch_size]
            _X = X[_ix]
            _y = y[_ix]

            tiny_batch_size = 4
            aug_X = _X[0:tiny_batch_size]
            aug_y = _y[0:tiny_batch_size]
            for j in range(1, len(_X) // tiny_batch_size):
                tiny_X = _X[j * tiny_batch_size : (j + 1) * tiny_batch_size]
                tiny_y = _y[j * tiny_batch_size : (j + 1) * tiny_batch_size]
                if np.random.rand() <= augment_chance:
                    aug_chain = np.random.choice(policy)
                    aug_chain[
                        "portion"
                    ] = 1.0  # last element is portion, which we want to be 1
                    hyperparams = list(aug_chain.values())
                    aug_data = augment_by_policy_wapirl(tiny_X, tiny_y, *hyperparams)

                    aug_data["X_train"] = apply_default_transformations(
                        aug_data["X_train"]
                    )
                    aug_X = np.concatenate([aug_X, aug_data["X_train"]])
                    aug_y = np.concatenate([aug_y, aug_data["y_train"]])
                else:
                    aug_X = np.concatenate([aug_X, tiny_X])
                    aug_y = np.concatenate([aug_y, tiny_y])
            yield aug_X, aug_y


if __name__ == '__main__':
    best_policies = pd.read_csv('best.csv')
    print(best_policies)

    # 1. init
    args = get_args()
    run = pre_requisite(args)

    batch_size = args.child_batch_size

    test_transform = WM811KTransform(size=(args.input_size_xy, args.input_size_xy), mode='test')
    train_set = WM811K('./data/wm811k/labeled/train/', decouple_input=args.decouple_input)
    valid_set = WM811K('./data/wm811k/labeled/valid/',
                       transform=test_transform,
                       decouple_input=args.decouple_input) # todo: 기존 DeepAugment에서는 1000개 샘플 뽑아 개수를 줄였음. 오래 걸리게 되면 여기도 조정 필요
    test_set = WM811K('./data/wm811k/labeled/test/',
                      transform=test_transform,
                      decouple_input=args.decouple_input)
    test_loader = DataLoader(test_set, args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False,
                             pin_memory=False)

    eval_policy = best_policies[
        ['A_aug1_type', 'A_aug1_magnitude', 'A_aug2_type', 'A_aug2_magnitude',
         'B_aug1_type', 'B_aug1_magnitude', 'B_aug2_type', 'B_aug2_magnitude',
         'C_aug1_type', 'C_aug1_magnitude', 'C_aug2_type', 'C_aug2_magnitude',
         'D_aug1_type', 'D_aug1_magnitude', 'D_aug2_type', 'D_aug2_magnitude',
         'E_aug1_type', 'E_aug1_magnitude', 'E_aug2_type', 'E_aug2_magnitude']
    ].to_dict(orient="records")

    ix = np.arange(len(test_set))
    np.random.shuffle(ix)

    print('start to evaluate')

    model = CNN(args).to(args.num_gpu)  # todo: change model type
    trainer = Trainer(args, model)

    test_loader =

    trainer.valid_epoch(test_loader)
    # deepaugment_image_generator(images, labels, top_policies_list, batch_size=batch_size)
    # trainer = Trainer(args, model, criterions)   # todo : epoch = 5, shuffle = False, test dataloader