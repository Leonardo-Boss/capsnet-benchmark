import math
import re
import warnings
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


class BaseDataLoader(DataLoader):
    """Custom base class for all data loaders. Inherits from PyTorch DataLoader.

    Attributes:
        shuffle (bool): Whether to shuffle the data every epoch.
        n_samples (int): Total number of samples in the dataset.
        validation_split (int | float): Fraction or amount of the data to be
            used as validation data.
        data_fraction (float): Fraction of the *training* split to actually
            use, after the validation split is taken out. 1.0 uses all of it.
        train_sampler (Sampler): Sampler for the training data.
        valid_sampler (Sampler): Sampler for the validation data.
        init_kwargs (dict): Keyword arguments for the PyTorch DataLoader
            initialization.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool,
        validation_split: int | float,
        num_workers: int,
        collate_fn: Callable = default_collate,
        data_fraction: float = 1.0,
        groups: np.ndarray | None = None,
        strata: np.ndarray | None = None,
        group_buffer: int = 1,
        group_select: str = "random",
        valid_dataset: Any | None = None,
    ):
        """Initialize loader class with the given dataset and parameters.

        Args:
            dataset (Any): The dataset to load data from.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data every epoch.
            validation_split (int | float): Fraction or amount of the data to be
                used as validation data.
            num_workers (int): Number of subprocesses to use for data loading.
            collate_fn (Callable, optional): Merges a list of samples to form a
                mini-batch. Defaults to PyTorch default_collate.
            data_fraction (float, optional): Fraction (0, 1] of the training
                split to actually train on -- e.g. 0.1 uses 10% of the
                training data. Applied *after* validation_split, so the
                validation set is unaffected and stays identical across runs
                with different data_fraction values, keeping comparisons
                fair. The subset is chosen randomly, not just the first N
                samples. Defaults to 1.0
                (use all training data).
            groups (np.ndarray | None, optional): Per-sample integer group id,
                length `len(dataset)`. When given, the train/validation split
                is made at the *group* level -- every sample of a group lands
                entirely in train or entirely in validation, never both. Use
                this whenever samples are not independent (e.g. consecutive
                frames extracted from the same video), because an iid index
                shuffle would otherwise put near-duplicates on both sides of
                the split and inflate validation scores. Group ids are
                assumed to be contiguous and ordered along the correlation
                axis (time), so id `g` and `g+1` are neighbours -- that is
                what `group_buffer` relies on. `None` keeps the original iid
                behaviour, which is correct for genuinely independent samples
                (MNIST, CIFAR-10). Defaults to None.
            strata (np.ndarray | None, optional): Per-sample class label used
                to pick validation groups stratified by class, so the
                validation split keeps the dataset's class balance instead of
                whatever the randomly drawn groups happen to contain. Only
                used when `groups` is given. Defaults to None (unstratified).
            group_buffer (int, optional): Number of neighbouring groups on
                each side of every held-out group to drop from *training*
                (they are not added to validation either -- they are simply
                discarded). This removes the residual leakage at block
                boundaries, where the last frame of a training block and the
                first frame of the adjacent validation block are consecutive
                video frames. Only used when `groups` is given. Defaults to 1.
            group_select (str, optional): How validation groups are chosen.
                "random" draws groups at random (within each stratum), giving
                validation coverage across the whole recording. "tail" takes
                the last groups of each stratum as one contiguous held-out
                segment -- a stricter, more pessimistic estimate, since the
                validation data is maximally separated in time from training.
                Only used when `groups` is given. Defaults to "random".
            valid_dataset (Any | None, optional): Alternative dataset object
                to draw validation samples from, indexed identically to
                `dataset`. Use this to serve validation images through an
                eval-time transform (resize + normalize only) while training
                images still go through the augmentation pipeline, so that
                `val_loss` -- which drives model selection -- is measured on
                clean images and is comparable across augmentation regimes.
                Defaults to None (validation reuses `dataset`).
        """
        assert 0 < data_fraction <= 1, "data_fraction must be in (0, 1]"
        assert group_select in ("random", "tail"), (
            f"Unknown group_select '{group_select}'. Expected 'random' or 'tail'."
        )
        self.data_fraction = data_fraction

        self.groups = None if groups is None else np.asarray(groups)
        self.strata = None if strata is None else np.asarray(strata)
        self.group_buffer = group_buffer
        self.group_select = group_select
        self.valid_dataset = valid_dataset

        self.shuffle = shuffle
        self.n_samples = len(dataset)

        if self.groups is not None and len(self.groups) != self.n_samples:
            raise ValueError(
                f"groups has length {len(self.groups)} but the dataset has "
                f"{self.n_samples} samples."
            )
        if self.valid_dataset is not None and len(self.valid_dataset) != self.n_samples:
            raise ValueError(
                "valid_dataset must be indexed identically to dataset "
                f"({len(self.valid_dataset)} vs {self.n_samples} samples)."
            )

        self.validation_split = validation_split
        self.train_sampler, self.valid_sampler = self._split_sampler(
            self.validation_split
        )
        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(
        self, split: float | int
    ) -> tuple[SubsetRandomSampler, SubsetRandomSampler]:
        """Split datasets and return sampler for both training and validation sets.

        Args:
            split (float | int): If float, represents the fraction of samples to
                be used for validation. If int, represents the exact number of
                samples to be used for validation.

        Returns:
            train_sampler (SubsetRandomSampler): Sampler for the training set.
            valid_sampler (SubsetRandomSampler | None): Sampler for the
                validation set, or None if no validation split and no
                data_fraction subsetting is configured.
        """
        if split == 0.0:
            len_valid = 0
        elif isinstance(split, int):
            assert split > 0, "Validation set size should be at least 1."
            assert (
                split < self.n_samples
            ), "Validation set size should be at most equal to the number of samples."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        if self.groups is None:
            idx_full = np.arange(self.n_samples)
            np.random.shuffle(idx_full)
            valid_idx = idx_full[0:len_valid]
            train_idx = np.delete(idx_full, np.arange(0, len_valid))

            # subsample the training portion only -- validation stays full and
            # identical across different data_fraction runs
            if self.data_fraction < 1.0:
                n_keep = max(1, int(len(train_idx) * self.data_fraction))
                train_idx = np.random.choice(train_idx, size=n_keep, replace=False)
        else:
            train_idx, valid_idx = self._grouped_split(len_valid)

        # a sampler is now needed whenever we're subsetting the training
        # data -- either from validation_split or data_fraction -- so
        # shuffle is turned off and DataLoader's own default-sampler path
        # is bypassed in both cases (previously this only happened for
        # validation_split != 0)
        needs_subset = len_valid > 0 or self.data_fraction < 1.0
        if not needs_subset:
            return None, None

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx) if len_valid > 0 else None

        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def _grouped_split(self, len_valid: int) -> tuple[np.ndarray, np.ndarray]:
        """Split train/validation at the group level rather than per sample.

        Whole groups are assigned to one side of the split, so correlated
        samples (e.g. consecutive video frames) can never appear on both
        sides. Groups adjacent to a held-out group are dropped entirely
        (`group_buffer`) to remove boundary leakage.

        Args:
            len_valid (int): Target number of validation *samples*. The
                realised size will land near, but rarely exactly on, this
                number, since groups are indivisible.

        Returns:
            tuple[np.ndarray, np.ndarray]: (train_idx, valid_idx).
        """
        groups = self.groups
        strata = (
            self.strata
            if self.strata is not None
            else np.zeros(self.n_samples, dtype=np.int64)
        )

        valid_groups: list[int] = []
        if len_valid > 0:
            valid_fraction = len_valid / self.n_samples
            for stratum in np.unique(strata):
                in_stratum = strata == stratum
                # unique() sorts, so group ids stay in temporal order here --
                # "tail" depends on that, "random" does not care
                s_groups = np.unique(groups[in_stratum])
                # honour the fraction within each stratum so the validation
                # split keeps the dataset's class balance
                s_target = valid_fraction * in_stratum.sum()

                order = (
                    np.random.permutation(s_groups)
                    if self.group_select == "random"
                    else s_groups[::-1]  # last groups first == contiguous tail
                )
                taken = 0
                for gid in order:
                    if taken >= s_target:
                        break
                    valid_groups.append(int(gid))
                    taken += int((groups == gid).sum())

        valid_mask = np.isin(groups, valid_groups)

        # drop the neighbours of every held-out group from training: the
        # frames either side of a block boundary are consecutive in the
        # source video, so keeping them would reintroduce exactly the
        # leakage this split exists to prevent
        blocked = set(valid_groups)
        for gid in valid_groups:
            for offset in range(1, self.group_buffer + 1):
                blocked.add(gid - offset)
                blocked.add(gid + offset)
        train_mask = ~np.isin(groups, list(blocked))

        train_idx = np.flatnonzero(train_mask)
        valid_idx = np.flatnonzero(valid_mask)

        # subsample training by group as well: pulling a random x% of
        # *frames* from a 30fps recording barely reduces the information
        # available (the discarded frames have near-identical neighbours),
        # so a per-frame data_fraction would make the data-efficiency axis
        # almost meaningless. Dropping whole blocks actually removes
        # distinct content.
        if self.data_fraction < 1.0 and len(train_idx) > 0:
            train_idx = self._subsample_by_group(train_idx, strata)

        np.random.shuffle(train_idx)
        np.random.shuffle(valid_idx)
        return train_idx, valid_idx

    def _subsample_by_group(
        self, train_idx: np.ndarray, strata: np.ndarray
    ) -> np.ndarray:
        """Keep a random `data_fraction` of whole training groups, per stratum."""
        groups = self.groups
        keep_idx: list[np.ndarray] = []

        for stratum in np.unique(strata[train_idx]):
            s_idx = train_idx[strata[train_idx] == stratum]
            s_groups = np.unique(groups[s_idx])
            target = max(1, int(round(len(s_idx) * self.data_fraction)))

            taken, kept_groups = 0, []
            for gid in np.random.permutation(s_groups):
                if taken >= target:
                    break
                kept_groups.append(gid)
                taken += int((groups[s_idx] == gid).sum())
            keep_idx.append(s_idx[np.isin(groups[s_idx], kept_groups)])

        return np.concatenate(keep_idx)

    def split_validation(self):
        """Get the validation set if configured."""
        if self.valid_sampler is None:
            return None

        kwargs = dict(self.init_kwargs)
        if self.valid_dataset is not None:
            # same indices, eval-time transform -- val_loss is then measured
            # on clean images and stays comparable across augmentation
            # regimes, which matters because it is the model-selection metric
            kwargs["dataset"] = self.valid_dataset
        return DataLoader(sampler=self.valid_sampler, **kwargs)

class MnistDataLoader(BaseDataLoader):
    """MNIST data loading class for Efficient CapsNet training.

    Attributes:
        mnist_img_size (int): The size of the MNIST images.
        dataset (Dataset): The MNIST dataset.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        shuffle: bool = True,
        validation_split: int | float = 0.0,
        num_workers: int = 1,
        training: bool = True,
    ):
        """Initializes the MnistDataLoader with the given parameters.

        The MNIST image data is transformed manually following the paper, in
        itself follows "No Routing Needed Between Capsules" by Byerly et al. The
        label data returned is one-hot encoded. Overall, the returned data will
        be in shape of (batch_size, 1, 28, 28) and (batch_size, 10).

        Args:
            data_dir (str): The directory where the MNIST data is located.
            batch_size (int): Number of samples per batch.
            shuffle (bool, optional): Whether to shuffle the data every epoch.
                Defaults to True.
            validation_split (int | float, optional): If float, represents the
                fraction of samples to be used for validation. If int, represents
                the exact number of samples to be used for validation. Defaults to 0.0.
            num_workers (int, optional): Number of subprocesses to use for data
                loading. Defaults to 1.
            training (bool, optional): Whether the data loader is for training
                data. Defaults to True.
        """
        self.mnist_img_size = 28
        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(self.random_rotate),
                transforms.Lambda(self.random_shift),
                transforms.Lambda(self.random_squish),
                transforms.Lambda(self.random_erase),
            ]
        )
        label_transform = transforms.Lambda(self.one_hot_encode)

        self.dataset = datasets.MNIST(
            data_dir,
            train=training,
            download=True,
            transform=image_transform,
            target_transform=label_transform,
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )

    def one_hot_encode(self, label: int, size: int = 10) -> torch.Tensor:
        """Transforms the given label into a one-hot encoded tensor."""
        one_hot = torch.zeros(size)
        one_hot[label] = 1
        return one_hot

    def random_rotate(self, img: torch.Tensor) -> torch.Tensor:
        """Randomly rotate the image by a small angle."""
        # random values for angle and decision
        rand_vals = torch.clamp(
            torch.normal(0, 0.33, size=(2,)), min=-0.9999, max=0.9999
        )

        if rand_vals[1] > 0:  # return original image
            return img

        else:  # return rotated image
            angle = rand_vals[0] * 30  # degrees
            rot_mat = cv2.getRotationMatrix2D(
                center=(self.mnist_img_size / 2, self.mnist_img_size / 2),
                angle=int(angle),
                scale=1.0,
            )
            new_img = cv2.warpAffine(
                src=img.squeeze().numpy(),  # 1x28x28 -> 28x28
                M=rot_mat,
                dsize=(self.mnist_img_size, self.mnist_img_size),
            )
            new_img = torch.from_numpy(new_img).float().unsqueeze(0)  # 28x28 -> 1x28x28
            return new_img

    def random_shift(self, img: torch.Tensor) -> torch.Tensor:
        """Randomly shift the image by a small amount.

        The margins of the image (the distance from the edge of the image to the
        nearest non-zero pixel) is calculated for each direction to determine
        the shift limit. Then, random value assign the actual shift amount.
        """
        img = img.view(self.mnist_img_size, self.mnist_img_size)  # 1x28x28 -> 28x28

        # find non-zero columns and rows
        nonzero_x_cols = torch.nonzero(torch.sum(img, dim=0) > 0, as_tuple=True)[0]
        nonzero_y_rows = torch.nonzero(torch.sum(img, dim=1) > 0, as_tuple=True)[0]

        # calculate margins
        left_margin = torch.min(nonzero_x_cols)
        right_margin = self.mnist_img_size - torch.max(nonzero_x_cols) - 1
        top_margin = torch.min(nonzero_y_rows)
        bot_margin = self.mnist_img_size - torch.max(nonzero_y_rows) - 1

        # generate random values for directions and decisions
        rand_dirs = torch.rand(2)
        dir_idxs = torch.floor(rand_dirs * 2).int()
        rand_vals = torch.clamp(torch.abs(torch.normal(0, 0.33, size=(2,))), max=0.9999)

        # calculate shift amounts
        x_amts = [
            torch.floor(-1.0 * rand_vals[0] * left_margin.float()),
            torch.floor(rand_vals[0] * (1 + right_margin).float()),
        ]
        y_amts = [
            torch.floor(-1.0 * rand_vals[1] * top_margin.float()),
            torch.floor(rand_vals[1] * (1 + bot_margin).float()),
        ]
        x_amt = int(x_amts[dir_idxs[1]])
        y_amt = int(y_amts[dir_idxs[0]])

        # perform shift on image
        # vertical shift
        img = img.view(self.mnist_img_size * self.mnist_img_size)  # 28x28 -> 784
        img = torch.roll(img, shifts=y_amt * self.mnist_img_size, dims=0)  # shift
        img = img.view(self.mnist_img_size, self.mnist_img_size)  # 784 -> 28x28

        # horizontal shift
        img = img.t()  # transpose
        img = img.reshape(self.mnist_img_size * self.mnist_img_size)  # 28x28 -> 784
        img = torch.roll(img, shifts=x_amt * self.mnist_img_size, dims=0)  # shift
        img = img.view(self.mnist_img_size, self.mnist_img_size)  # 784 -> 28x28
        img = img.t()  # transpose back

        return img.view(1, self.mnist_img_size, self.mnist_img_size)  # 28x28 -> 1x28x28

    def random_squish(self, img: torch.Tensor) -> torch.Tensor:
        """Randomly distorts an image by squishing it along its width.

        'Squishing' an image refers to reducing its size in one dimension, while
        keeping the other dimension the same.
        """
        rand_vals = torch.clamp(torch.abs(torch.normal(0, 0.33, size=(2,))), max=0.9999)

        # calculate width reduction and padding offset
        width_mod = int((rand_vals[0] * (self.mnist_img_size / 4)).floor() + 1)
        offset_mod = int((rand_vals[1] * 2.0).floor())  # right pad
        offset = (width_mod // 2) + offset_mod  # left pad

        # reduce width but maintain height
        img = Ft.resize(img, [self.mnist_img_size, self.mnist_img_size - width_mod])
        # pad with offset
        img = Ft.pad(img, (offset, 0, offset_mod, 0))
        # crop (fill in) to original size
        img = Ft.crop(img, 0, 0, self.mnist_img_size, self.mnist_img_size)
        return img

    def random_erase(self, img):
        """Randomly erase a 4x4 patch from the image."""
        rand_vals = torch.rand(2)
        x = int((rand_vals[0] * 19).floor() + 4)
        y = int((rand_vals[1] * 19).floor() + 4)
        patch = torch.zeros(4, 4)
        # pad the patch with 1s to make it 28x28
        mask = F.pad(
            patch,
            (x, self.mnist_img_size - x - 4, y, self.mnist_img_size - y - 4),
            mode="constant",
            value=1,
        )
        img = img * mask
        return img


class Cifar10DataLoader(BaseDataLoader):
    """CIFAR-10 data loading class, built on `torchvision.datasets.CIFAR10`.

    `data_dir` must be the *parent* directory that contains the
    `cifar-10-batches-py` folder (torchvision appends that folder name
    itself) -- e.g. if your data lives at `project/cifar-10-batches-py/`,
    pass `data_dir='project/'`.

    Attributes:
        num_classes (int): Number of classes (10).
        dataset (Dataset): The CIFAR-10 dataset.
    """

    # Per-channel mean/std computed over the CIFAR-10 training set.
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    AUGMENTATION_LEVELS = ("none", "standard", "strong")

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        shuffle: bool = True,
        validation_split: int | float = 0.0,
        num_workers: int = 1,
        training: bool = True,
        augmentation: str = "standard",
        data_fraction: float = 1.0,

    ):
        """Initializes the Cifar10DataLoader with the given parameters.

        Standard augmentation (random crop + horizontal flip) is applied for
        the training split; only normalization is applied otherwise. Labels
        are one-hot encoded, matching MnistDataLoader's output convention.

        Args:
            data_dir (str): Parent directory containing the
                `cifar-10-batches-py` folder.
            batch_size (int): Number of samples per batch.
            shuffle (bool, optional): Whether to shuffle the data every epoch.
                Defaults to True.
            validation_split (int | float, optional): If float, represents the
                fraction of samples to be used for validation. If int, represents
                the exact number of samples to be used for validation. Defaults to 0.0.
            num_workers (int, optional): Number of subprocesses to use for data
                loading. Defaults to 1.
            training (bool, optional): Whether to load the training split
                (with augmentation) or the test split. Defaults to True.
            augmentation (str, optional): Training-time augmentation regime.
                One of "none" (no augmentation), "standard" (random crop +
                horizontal flip + rotation +/-15deg), or "strong" (standard +
                RandAugment + Random Erasing). Ignored when training=False --
                the eval/test split is always just normalized. Defaults to
                "standard".
            data_fraction (float, optional): Fraction (0, 1] of the training
                data to actually use -- e.g. 0.25 trains on 25% of the
                training split. Ignored when training=False (the test/eval
                split is always used in full). Defaults to 1.0.
        """
        self.num_classes = 10

        augmentation = augmentation.lower()
        if augmentation not in self.AUGMENTATION_LEVELS:
            raise ValueError(
                f"Unknown augmentation level '{augmentation}'. "
                f"Expected one of {self.AUGMENTATION_LEVELS}."
            )

        if training:
            image_transform = self._build_train_transform(augmentation)
        else:
            image_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD),
                ]
            )

        label_transform = transforms.Lambda(self.one_hot_encode)

        self.dataset = datasets.CIFAR10(
            data_dir,
            train=training,
            download=True,
            transform=image_transform,
            target_transform=label_transform,
        )

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers,
            data_fraction=data_fraction if training else 1.0,
        )

    def _build_train_transform(self, augmentation: str) -> transforms.Compose:
        """Build the training-time augmentation pipeline for a given level."""
        ops = []

        if augmentation in ("standard", "strong"):
            ops += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),  # +/- 15 degrees
            ]

        if augmentation == "strong":
            ops.append(transforms.RandAugment())  # operates on PIL images

        ops += [
            transforms.ToTensor(),
            transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD),
        ]

        if augmentation == "strong":
            ops.append(transforms.RandomErasing())  # operates on tensors

        return transforms.Compose(ops)

    def one_hot_encode(self, label: int) -> torch.Tensor:
        """Transforms the given label into a one-hot encoded tensor."""
        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot


class FlameDataLoader(BaseDataLoader):
    """Data loader for the FLAME wildfire-image dataset (binary Fire / No_Fire).

    Expects `data_dir` to be the dataset root with this on-disk layout
    (standard `torchvision.datasets.ImageFolder` layout, one subfolder per
    class):

        <data_dir>/
            Training/
                Fire/*.jpg
                No_Fire/*.jpg
            Test/
                Fire/*.jpg
                No_Fire/*.jpg

    `training=True` loads the `Training/` split (with augmentation, and
    optionally further divided into train/validation via
    `validation_split`, same convention as the other loaders in this
    module); `training=False` loads the `Test/` split (center-cropped/
    resized only, no augmentation).

    Images are natively 254x254 RGB jpgs. `img_size` controls what they're
    resized to before augmentation/normalization -- this must match the
    `input_size` (H, W) passed to the model's `arch.args` in config.yaml,
    since e.g. EfficientCapsNet's PrimaryCaps kernel size is derived from
    that input size. Defaults to 224 (the ImageNet-standard resolution,
    also required by patch-based ViT/DeiT backbones since it must divide
    evenly by patch_size), but any value from ~32 up to the native 254 is
    supported; smaller sizes train faster (see the compute-cost note in
    model.EfficientCapsNet), and 254 itself skips resizing entirely.

    Classes are alphabetically ordered by `ImageFolder`, so the one-hot
    label layout is index 0 = "Fire", index 1 = "No_Fire" (`self.classes`
    reflects this and can be used to double check).

    Attributes:
        num_classes (int): Number of classes (2: Fire, No_Fire).
        classes (list[str]): Class names in one-hot index order.
        img_size (int): Side length images are resized to before use.
        dataset (Dataset): The underlying ImageFolder dataset.
    """

    # Standard ImageNet stats -- a reasonable default for real photographic
    # RGB images like FLAME's, absent dataset-specific computed stats.
    FLAME_MEAN = (0.485, 0.456, 0.406)
    FLAME_STD = (0.229, 0.224, 0.225)

    AUGMENTATION_LEVELS = ("none", "standard", "strong")

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        shuffle: bool = True,
        validation_split: int | float = 0.0,
        num_workers: int = 1,
        training: bool = True,
        img_size: int = 224,
        augmentation: str = "standard",
        data_fraction: float = 1.0,
        val_split_mode: str = "blocks",
        block_size: int = 150,
        block_buffer: int = 1,
        clean_validation: bool = True,
    ):
        """Initializes the FlameDataLoader with the given parameters.

        Args:
            data_dir (str): Path to the FLAME dataset root (the directory
                containing `Training/` and `Test/`).
            batch_size (int): Number of samples per batch.
            shuffle (bool, optional): Whether to shuffle the data every
                epoch. Defaults to True.
            validation_split (int | float, optional): If float, fraction of
                the `Training` split to hold out for validation. If int,
                the exact number of samples. Only meaningful when
                `training=True`. Defaults to 0.0.
            num_workers (int, optional): Number of subprocesses for data
                loading. Defaults to 1.
            training (bool, optional): If True, loads `data_dir/Training`
                (augmented). If False, loads `data_dir/Test` (no
                augmentation, `validation_split`/`data_fraction` ignored).
                Defaults to True.
            img_size (int, optional): Side length (pixels) images are
                resized to -- must match the model's configured
                `input_size`. Defaults to 224. Pass 254 to skip resizing
                and train at native resolution instead.
            augmentation (str, optional): Training-time augmentation
                regime: "none" (resize + normalize only), "standard"
                (random resized crop + horizontal flip + small rotation),
                or "strong" (standard + color jitter + RandAugment +
                Random Erasing). Ignored when `training=False`. Defaults
                to "standard".
            data_fraction (float, optional): Fraction (0, 1] of the
                training split to actually use. Ignored when
                `training=False`. Defaults to 1.0. Under the "blocks"/"tail"
                split modes this drops whole blocks rather than scattered
                individual frames -- see BaseDataLoader.
            val_split_mode (str, optional): How the validation split is
                carved out of `Training/`.

                FLAME's training frames are extracted from continuous UAV
                video, so consecutive frames are near-duplicates. An iid
                per-frame split therefore leaks: nearly every validation
                frame has an almost identical twin in training, and the model
                scores >90% within one epoch by recognising backgrounds it
                has already memorised rather than by learning what fire looks
                like. That inflated number then drives `min val_loss` model
                selection, which picks the most background-overfit
                checkpoint, and it collapses on the official `Test/` split
                (a separate recording).

                - "blocks" (default): frames are sorted into temporal order
                  and cut into contiguous blocks of `block_size`; whole
                  blocks are held out. Validation still covers the whole
                  recording, but no validation frame has a near-duplicate in
                  training.
                - "tail": holds out one contiguous segment at the end of each
                  class's frame sequence. Strictest and most pessimistic --
                  closest in spirit to the train/test separation of the
                  official split.
                - "random": the original iid per-frame behaviour. Kept only
                  so the leaky baseline can be reproduced deliberately; it
                  should not be used for reported results.

                Defaults to "blocks".
            block_size (int, optional): Frames per temporal block. At ~30fps,
                150 frames is roughly a 5-second segment. Smaller blocks give
                more independent validation units but leave adjacent blocks
                more similar; larger blocks are stricter but coarser.
                Defaults to 150.
            block_buffer (int, optional): Blocks either side of each held-out
                block that are discarded from training, so that frames
                straddling a block boundary don't leak. Defaults to 1.
            clean_validation (bool, optional): Serve validation images
                through the eval transform (resize + normalize only) instead
                of the training augmentation pipeline, so `val_loss` is
                measured on clean images. Defaults to True.

        Raises:
            ValueError: If `val_split_mode` or `augmentation` is unknown, or
                if the on-disk class folders don't match the expected layout.
        """
        valid_modes = ("blocks", "tail", "random")
        if val_split_mode not in valid_modes:
            raise ValueError(
                f"Unknown val_split_mode '{val_split_mode}'. "
                f"Expected one of {valid_modes}."
            )
        self.num_classes = 2
        self.img_size = img_size

        augmentation = augmentation.lower()
        if augmentation not in self.AUGMENTATION_LEVELS:
            raise ValueError(
                f"Unknown augmentation level '{augmentation}'. "
                f"Expected one of {self.AUGMENTATION_LEVELS}."
            )

        split_dir = Path(data_dir) / ("Training" if training else "Test")
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Expected a '{'Training' if training else 'Test'}' folder "
                f"under '{data_dir}' (with 'Fire'/'No_Fire' subfolders), "
                f"but '{split_dir}' does not exist."
            )

        eval_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.FLAME_MEAN, self.FLAME_STD),
            ]
        )
        image_transform = (
            self._build_train_transform(augmentation) if training else eval_transform
        )

        label_transform = transforms.Lambda(self.one_hot_encode)

        self.dataset = datasets.ImageFolder(
            str(split_dir),
            transform=image_transform,
            target_transform=label_transform,
        )

        valid_dataset = None
        if training and clean_validation and validation_split:
            # second view of the same directory; ImageFolder sorts
            # deterministically, so index i refers to the same file in both
            valid_dataset = datasets.ImageFolder(
                str(split_dir),
                transform=eval_transform,
                target_transform=label_transform,
            )

        self.classes = self.dataset.classes
        if self.classes != ["Fire", "No_Fire"]:
            # ImageFolder sorts subfolder names alphabetically; this should
            # always hold for the documented layout, but flag loudly if a
            # differently-named/ordered folder set sneaks in, since the
            # one-hot index convention documented above depends on it.
            raise ValueError(
                f"Expected classes ['Fire', 'No_Fire'] under '{split_dir}', "
                f"found {self.classes}."
            )

        groups = strata = None
        if training and val_split_mode != "random":
            groups, strata = self._build_temporal_blocks(self.dataset, block_size)
            self.blocks = groups
            self.n_blocks = len(np.unique(groups))

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers,
            data_fraction=data_fraction if training else 1.0,
            groups=groups,
            strata=strata,
            group_buffer=block_buffer,
            group_select="tail" if val_split_mode == "tail" else "random",
            valid_dataset=valid_dataset,
        )

    @staticmethod
    def _build_temporal_blocks(
        dataset: datasets.ImageFolder, block_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign each frame to a contiguous temporal block within its class.

        FLAME frames are named with a sequential index, so sorting a class's
        filenames numerically recovers the order they were extracted from the
        source video in. Frames are then cut into consecutive runs of
        `block_size`, and those blocks become the indivisible unit of the
        train/validation split.

        Block ids are allocated per class and increase with time, so blocks
        `g` and `g+1` really are temporal neighbours -- which is what
        BaseDataLoader's `group_buffer` assumes when it discards the
        neighbours of held-out blocks.

        Args:
            dataset (datasets.ImageFolder): Dataset whose `.samples` holds
                (path, class_index) pairs.
            block_size (int): Number of consecutive frames per block.

        Returns:
            tuple[np.ndarray, np.ndarray]: (block id per sample, class label
                per sample).
        """
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}.")

        samples = dataset.samples
        strata = np.array([label for _, label in samples], dtype=np.int64)
        groups = np.empty(len(samples), dtype=np.int64)

        def sort_key(i: int) -> tuple[int, str]:
            """Numeric-aware sort key -- 'frame_10' must follow 'frame_9'."""
            stem = Path(samples[i][0]).stem
            digits = re.findall(r"\d+", stem)
            return (int(digits[-1]) if digits else 0, stem)

        if not any(re.search(r"\d", Path(p).stem) for p, _ in samples[:64]):
            warnings.warn(
                "FLAME frame filenames contain no digits, so temporal order "
                "cannot be recovered and blocks will follow alphabetical "
                "order instead. Check that the split is still meaningful, or "
                "supply groups explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )

        next_block = 0
        for label in np.unique(strata):
            idx = np.flatnonzero(strata == label).tolist()
            idx.sort(key=sort_key)
            for position, i in enumerate(idx):
                groups[i] = next_block + position // block_size
            next_block += math.ceil(len(idx) / block_size)

        return groups, strata

    def _build_train_transform(self, augmentation: str) -> transforms.Compose:
        """Build the training-time augmentation pipeline for a given level.

        Deliberately uses a plain `Resize` rather than `RandomResizedCrop`
        for every level, including "standard"/"strong" -- a random crop
        would randomly discard part of the frame before resizing, which
        risks cutting out the very fire/smoke region a detector needs to
        see. All augmentation here preserves the full frame instead.
        """
        ops = [transforms.Resize((self.img_size, self.img_size))]

        if augmentation in ("standard", "strong"):
            ops += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),  # +/- 15 degrees
            ]

        if augmentation == "strong":
            ops.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
            ops.append(transforms.RandAugment())  # operates on PIL images

        ops += [
            transforms.ToTensor(),
            transforms.Normalize(self.FLAME_MEAN, self.FLAME_STD),
        ]

        if augmentation == "strong":
            ops.append(transforms.RandomErasing())  # operates on tensors

        return transforms.Compose(ops)

    def one_hot_encode(self, label: int) -> torch.Tensor:
        """Transforms the given label into a one-hot encoded tensor."""
        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot
