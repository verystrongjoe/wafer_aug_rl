import numpy as np
import pandas as pd
import sys
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
sys.path.insert(0, dir_of_file)


def create_cutout_mask(img_height, img_width, num_channels, size):
    """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
  Args:
    img_height: Height of image cutout mask will be applied to.
    img_width: Width of image cutout mask will be applied to.
    num_channels: Number of channels in the image.
    size: Size of the zeros mask.
  Returns:
    A mask of shape `img_height` x `img_width` with all ones except for a
    square of zeros of shape `size` x `size`. This mask is meant to be
    elementwise multiplied with the original image. Additionally returns
    the `upper_coord` and `lower_coord` which specify where the cutout mask
    will be applied.
  """
    assert img_height == img_width

    # Sample center where cutout mask will be applied
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)

    # Determine upper right and lower left corners of patch
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (
        min(img_height, height_loc + size // 2),
        min(img_width, width_loc + size // 2),
    )
    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]
    assert mask_height > 0
    assert mask_width > 0

    mask = np.ones((img_height, img_width, num_channels))
    zeros = np.zeros((mask_height, mask_width, num_channels))
    mask[upper_coord[0] : lower_coord[0], upper_coord[1] : lower_coord[1], :] = zeros
    return mask, upper_coord, lower_coord


def cutout_numpy(img, size=16):
    """Apply cutout with mask of shape `size` x `size` to `img`.
  The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
  This operation applies a `size`x`size` mask of zeros to a random location
  within `img`.
  Args:
    img: Numpy image that cutout will be applied to.
    size: Height/width of the cutout mask that will be
  Returns:
    A numpy tensor that is the result of applying the cutout mask to `img`.
  """
    img_height, img_width, num_channels = (img.shape[0], img.shape[1], img.shape[2])
    assert len(img.shape) == 3
    mask, _, _ = create_cutout_mask(img_height, img_width, num_channels, size)
    return img * mask

AUG_TYPES = [
    "crop",
    "gaussian-blur",
    "rotate",
    "shear",
    "translate-x",
    "translate-y",
    "sharpen",
    "emboss",
    "additive-gaussian-noise",
    "dropout",
    "coarse-dropout",
    "gamma-contrast",
    "brighten",
    "invert",
    "fog",
    "clouds",
    "add-to-hue-and-saturation",
    "coarse-salt-pepper",
    "horizontal-flip",
    "vertical-flip",
]


def augment_type_chooser():
    """A random function to choose among augmentation types

    Returns:
        function object: np.random.choice function with AUG_TYPES input
    """
    return np.random.choice(AUG_TYPES)


def random_flip(x):
    """Flip the input x horizontally with 50% probability."""
    if np.random.rand(1)[0] > 0.5:
        return np.fliplr(x)
    return x


def zero_pad_and_crop(img, amount=4):
    """Zero pad by `amount` zero pixels on each side then take a random crop.
  Args:
    img: numpy image that will be zero padded and cropped.
    amount: amount of zeros to pad `img` with horizontally and verically.
  Returns:
    The cropped zero padded img. The returned numpy array will be of the same
    shape as `img`.
  """
    padded_img = np.zeros(
        (img.shape[0] + amount * 2, img.shape[1] + amount * 2, img.shape[2])
    )
    padded_img[amount : img.shape[0] + amount, amount : img.shape[1] + amount, :] = img
    top = np.random.randint(low=0, high=2 * amount)
    left = np.random.randint(low=0, high=2 * amount)
    new_img = padded_img[top : top + img.shape[0], left : left + img.shape[1], :]
    return new_img


def apply_default_transformations(X):
    # apply cutout
    X_aug = []
    for img in X:
        img_aug = zero_pad_and_crop(img, amount=4)
        img_aug = cutout_numpy(img_aug, size=6)
        X_aug.append(img_aug)
    return X_aug


def deepaugment_image_generator(X, y, policy, batch_size=64, augment_chance=0.5):
    """Yields batch of images after applying random augmentations from the policy

    Each image is augmented by 50% chance. If augmented, one of the augment-chain in the policy is applied.
    Which augment-chain to apply is chosen randomly.

    Args:
        X (numpy.array):
        labels (numpy.array):
        policy (list): list of dictionaries

    Returns:
    """
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
            policy = policy_df.to_dict(orient="records")

    print("Policies are:")
    print(policy)
    print()

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
                    # todo : 여기 사용하게 되면 변경 필요 
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



