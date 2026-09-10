import inspect
import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
        """
        assert 0 < data_fraction <= 1, "data_fraction must be in (0, 1]"
        self.data_fraction = data_fraction

        self.shuffle = shuffle
        self.n_samples = len(dataset)
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
        idx_full = np.arange(self.n_samples)
        np.random.shuffle(idx_full)

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

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        # subsample the training portion only -- validation stays full and
        # identical across different data_fraction runs
        if self.data_fraction < 1.0:
            n_keep = max(1, int(len(train_idx) * self.data_fraction))
            train_idx = np.random.choice(train_idx, size=n_keep, replace=False)

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

    def split_validation(self):
        """Get the validation set if configured."""
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

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
                `training=False`. Defaults to 1.0.
        """
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

        if training:
            image_transform = self._build_train_transform(augmentation)
        else:
            image_transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.FLAME_MEAN, self.FLAME_STD),
                ]
            )

        label_transform = transforms.Lambda(self.one_hot_encode)

        self.dataset = datasets.ImageFolder(
            str(split_dir),
            transform=image_transform,
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

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers,
            data_fraction=data_fraction if training else 1.0,
        )

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


# ---------------------------------------------------------------------------
# D-Fire: a YOLO *object-detection* dataset, consumed here as *classification*
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Split-folder name aliases seen across the various D-Fire redistributions
# (original OneDrive zips, the Kaggle mirror, Roboflow exports).
_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


def _looks_like_image_dir(path: Path) -> bool:
    """True if `path` is a directory holding at least one image file."""
    if not path.is_dir():
        return False
    return any(p.suffix.lower() in IMG_EXTENSIONS for p in path.iterdir())


def _resolve_split_dirs(root: Path, split: str, _depth: int = 0):
    """Locate the (images_dir, labels_dir) pair for `split` under `root`.

    Handles the three on-disk layouts D-Fire ships in, in this order:

        1. <root>/train/images/*.jpg   + <root>/train/labels/*.txt
        2. <root>/images/train/*.jpg   + <root>/labels/train/*.txt
        3. <root>/train/*.jpg          + <root>/train/*.txt  (side by side)

    If none match and `root` contains a single subdirectory (the usual
    result of unzipping a Kaggle archive into a fresh folder), it
    descends into it and retries, up to two levels deep.

    Returns:
        tuple[Path, Path] | None: (images_dir, labels_dir), or None if no
            layout matched.
    """
    for name in _SPLIT_ALIASES[split]:
        candidates = [
            (root / name / "images", root / name / "labels"),
            (root / "images" / name, root / "labels" / name),
            (root / name, root / name),
        ]
        for images_dir, labels_dir in candidates:
            if not _looks_like_image_dir(images_dir):
                continue
            if not labels_dir.is_dir():
                # fall back to any plausible sibling annotation folder,
                # otherwise assume the .txt files sit next to the images
                for alt in ("labels", "annotations", "labels_txt"):
                    sibling = images_dir.parent / alt
                    if sibling.is_dir():
                        labels_dir = sibling
                        break
                else:
                    labels_dir = images_dir
            return images_dir, labels_dir

    if _depth < 2:
        subdirs = [p for p in sorted(root.iterdir()) if p.is_dir()] if root.is_dir() else []
        for sub in subdirs:
            found = _resolve_split_dirs(sub, split, _depth + 1)
            if found is not None:
                return found

    return None


def _resolve_yolo_class_ids(root: Path, default_smoke: int = 0, default_fire: int = 1):
    """Figure out which YOLO class id means 'smoke' and which means 'fire'.

    D-Fire's own annotations use 0 = smoke, 1 = fire, but some
    redistributions flip them, so a `data.yaml`/`data.yml` next to the
    data (Ultralytics style, with a `names:` list or dict) is trusted
    over the defaults when present.

    Returns:
        tuple[int, int]: (smoke_class_id, fire_class_id).
    """
    for pattern in ("data.yaml", "data.yml", "*/data.yaml", "*/data.yml"):
        for cfg_path in sorted(root.glob(pattern)):
            try:
                import yaml  # local import: only needed on this path

                with cfg_path.open("r", encoding="utf8") as f:
                    names = (yaml.safe_load(f) or {}).get("names")
            except Exception:  # unreadable/odd yaml -- just use the defaults
                continue

            if isinstance(names, dict):
                pairs = [(int(k), str(v)) for k, v in names.items()]
            elif isinstance(names, (list, tuple)):
                pairs = list(enumerate(str(v) for v in names))
            else:
                continue

            smoke_id = fire_id = None
            for idx, name in pairs:
                low = name.strip().lower()
                if "smoke" in low:
                    smoke_id = idx
                elif "fire" in low or "flame" in low:
                    fire_id = idx

            if smoke_id is not None and fire_id is not None:
                _logger.info(
                    "Read class ids from %s: smoke=%d, fire=%d",
                    cfg_path, smoke_id, fire_id,
                )
                return smoke_id, fire_id

    return default_smoke, default_fire


class YoloClassificationDataset(Dataset):
    """Reads a YOLO detection dataset and serves it as image classification.

    Every image gets exactly one label, derived purely from *which*
    classes appear in its annotation file -- bounding-box coordinates are
    parsed only far enough to read the leading class id, and are
    otherwise discarded. An image with no annotation file, or with an
    empty one, is treated as a true negative (YOLO's own convention for
    background images), which is how D-Fire's ~9.8k "None" images are
    picked up.

    Label schemes (`label_scheme`):
        "binary"     2 classes, index 0 = "Fire_or_Smoke" (at least one
                     box of any class), 1 = "Nothing". The default, and
                     the most balanced split of D-Fire (~11.7k positive
                     vs ~9.8k negative).
        "fire"       2 classes, 0 = "Fire" (>=1 fire box), 1 = "No_Fire".
                     Smoke-only images count as negatives. Matches
                     FlameDataLoader's task and its 0=Fire index order,
                     but is class-imbalanced here (~5.8k vs ~15.7k).
        "smoke"      2 classes, 0 = "Smoke", 1 = "No_Smoke".
        "four_class" 4 classes ordered by severity: 0 = "Nothing",
                     1 = "Smoke", 2 = "Fire", 3 = "Fire_And_Smoke" --
                     the four categories D-Fire's own paper tabulates.

    Attributes:
        samples (list[tuple[Path, int]]): (image path, class index) pairs.
        classes (list[str]): Class names in label-index order.
        class_counts (dict[str, int]): Images per class, for sanity
            checks and for computing loss weights if wanted.
        n_missing_labels (int): Images with no annotation file at all
            (counted as negatives).
    """

    LABEL_SCHEMES = ("binary", "fire", "smoke", "four_class")

    SCHEME_CLASSES = {
        "binary": ["Fire_or_Smoke", "Nothing"],
        "fire": ["Fire", "No_Fire"],
        "smoke": ["Smoke", "No_Smoke"],
        "four_class": ["Nothing", "Smoke", "Fire", "Fire_And_Smoke"],
    }

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        label_scheme: str = "binary",
        smoke_class_id: int = 0,
        fire_class_id: int = 1,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        """Indexes the split and derives one class label per image.

        Args:
            images_dir (str | Path): Directory of image files.
            labels_dir (str | Path): Directory of YOLO `.txt` annotation
                files, matched to images by stem. May be the same as
                `images_dir` for side-by-side layouts.
            label_scheme (str, optional): One of `LABEL_SCHEMES`.
                Defaults to "binary".
            smoke_class_id (int, optional): YOLO class id meaning smoke.
                Defaults to 0 (D-Fire's convention).
            fire_class_id (int, optional): YOLO class id meaning fire.
                Defaults to 1.
            transform (Callable, optional): Image transform, applied to a
                PIL image. Defaults to None.
            target_transform (Callable, optional): Label transform, e.g.
                one-hot encoding. Defaults to None.
        """
        label_scheme = label_scheme.lower()
        if label_scheme not in self.LABEL_SCHEMES:
            raise ValueError(
                f"Unknown label_scheme '{label_scheme}'. "
                f"Expected one of {self.LABEL_SCHEMES}."
            )

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.label_scheme = label_scheme
        self.smoke_class_id = smoke_class_id
        self.fire_class_id = fire_class_id
        self.transform = transform
        self.target_transform = target_transform

        self.classes = list(self.SCHEME_CLASSES[label_scheme])
        self.num_classes = len(self.classes)

        image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No image files found under '{self.images_dir}'.")

        self.samples: list[tuple[Path, int]] = []
        self.n_missing_labels = 0

        for img_path in image_paths:
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if label_path.is_file():
                present = self._read_class_ids(label_path)
            else:
                present = set()
                self.n_missing_labels += 1
            self.samples.append((img_path, self._to_class_index(present)))

        self.class_counts = {name: 0 for name in self.classes}
        for _, idx in self.samples:
            self.class_counts[self.classes[idx]] += 1

        if self.n_missing_labels:
            frac = self.n_missing_labels / len(self.samples)
            msg = (
                "%d/%d images in %s have no matching .txt in %s and were "
                "labelled as negatives."
            )
            args = (self.n_missing_labels, len(self.samples),
                    self.images_dir, self.labels_dir)
            # a handful of missing files is normal (background images are
            # sometimes shipped without empty .txt stubs); most of them
            # missing almost always means labels_dir is pointing somewhere
            # wrong, which would silently train the model on garbage
            if frac > 0.5:
                _logger.warning(msg + " That is over half the split -- check "
                                "that the labels directory is correct.", *args)
            else:
                _logger.info(msg, *args)

        _logger.info(
            "D-Fire split %s: %d images, scheme=%s, counts=%s",
            self.images_dir, len(self.samples), label_scheme, self.class_counts,
        )

    @staticmethod
    def _read_class_ids(label_path: Path) -> set[int]:
        """Returns the set of YOLO class ids annotated in one label file."""
        present: set[int] = set()
        with label_path.open("r", encoding="utf8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token = line.split()[0]
                try:
                    present.add(int(float(token)))
                except ValueError:
                    _logger.warning(
                        "Skipping unparseable line in %s: %r", label_path, line
                    )
        return present

    def _to_class_index(self, present: set[int]) -> int:
        """Maps the class ids found in an annotation file to one label."""
        has_fire = self.fire_class_id in present
        has_smoke = self.smoke_class_id in present

        if self.label_scheme == "binary":
            return 0 if present else 1
        if self.label_scheme == "fire":
            return 0 if has_fire else 1
        if self.label_scheme == "smoke":
            return 0 if has_smoke else 1
        # four_class
        if has_fire and has_smoke:
            return 3
        if has_fire:
            return 2
        if has_smoke:
            return 1
        return 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, target = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


class DFireDataLoader(BaseDataLoader):
    """D-Fire data loader: a detection dataset repurposed for classification.

    D-Fire (Venancio et al.) is 21,527 fire/smoke surveillance and
    wildfire images annotated with YOLO bounding boxes -- 1,164 fire
    only, 5,867 smoke only, 4,658 both, and 9,838 with nothing. This
    loader throws the box geometry away and keeps only *which* classes
    each image contains, turning it into a whole-image classification
    task that plugs into the same trainer, margin loss and one-hot
    target convention as MnistDataLoader / Cifar10DataLoader /
    FlameDataLoader.

    `data_dir` is the dataset root; the split folders are auto-detected,
    so all three layouts D-Fire is distributed in work unchanged:

        <data_dir>/train/images/*.jpg + <data_dir>/train/labels/*.txt
        <data_dir>/images/train/*.jpg + <data_dir>/labels/train/*.txt
        <data_dir>/train/*.jpg        + <data_dir>/train/*.txt

    plus `val`/`valid` and `test` under the same shape, and one level of
    nesting (an extra wrapper folder from unzipping the Kaggle archive).

    `training=True` loads the train split with augmentation; validation
    is carved out of it via `validation_split`, exactly like the other
    loaders here, so `data_fraction` sweeps stay comparable. Set
    `use_official_val=True` to use the dataset's own `val`/`valid`
    folder as the validation set instead (unaugmented, and unaffected by
    `data_fraction`). `training=False` loads the test split with resize
    + normalize only.

    See `YoloClassificationDataset` for the `label_scheme` options; note
    that "binary" gives 2 classes with index 0 = positive, matching
    FlameDataLoader's 0 = Fire index order, so a 2-class `arch.args`
    config carries over from FLAME with no changes.

    Attributes:
        num_classes (int): Number of classes implied by `label_scheme`.
        classes (list[str]): Class names in one-hot index order.
        class_counts (dict[str, int]): Images per class in the loaded split.
        img_size (int): Side length images are resized to.
        dataset (Dataset): The underlying YoloClassificationDataset.
    """

    # Standard ImageNet stats -- same choice as FlameDataLoader, these
    # are real photographic RGB images.
    DFIRE_MEAN = (0.485, 0.456, 0.406)
    DFIRE_STD = (0.229, 0.224, 0.225)

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
        label_scheme: str = "binary",
        use_official_val: bool = False,
        clean_validation: bool = True,
        val_split_mode: str | None = None,
        block_size: int | None = None,
        block_buffer: int | None = None,
    ):
        """Initializes the DFireDataLoader with the given parameters.

        Args:
            data_dir (str): Path to the D-Fire dataset root.
            batch_size (int): Number of samples per batch.
            shuffle (bool, optional): Whether to shuffle every epoch.
                Defaults to True.
            validation_split (int | float, optional): Fraction (float) or
                count (int) of the train split held out for validation.
                Only meaningful when `training=True`, and ignored when
                `use_official_val=True`. Defaults to 0.0.
            num_workers (int, optional): Subprocesses for data loading.
                Defaults to 1.
            training (bool, optional): Load the train split (augmented)
                or the test split (clean). Defaults to True.
            img_size (int, optional): Side length images are resized to.
                Must match the model's configured `input_size`. D-Fire
                images vary in resolution, so unlike FLAME this resize
                always happens. Defaults to 224.
            augmentation (str, optional): "none", "standard" or "strong".
                Ignored when `training=False`. Defaults to "standard".
            data_fraction (float, optional): Fraction (0, 1] of the train
                split to use. Ignored when `training=False`. Defaults to 1.0.
            label_scheme (str, optional): How box classes collapse to one
                label per image -- "binary", "fire", "smoke" or
                "four_class". Defaults to "binary".
            use_official_val (bool, optional): Use the dataset's own
                `val`/`valid` folder for validation instead of carving it
                out of train. Defaults to False.
            clean_validation (bool, optional): Evaluate the carved-out
                validation samples with the resize+normalize transform
                instead of the training augmentation, so `val_loss` is
                measured on clean images and stays comparable across
                augmentation regimes. Defaults to True.
            val_split_mode (str, optional): Forwarded to BaseDataLoader
                if that class supports it (see note below). Defaults to
                None (leave BaseDataLoader's own default alone).
            block_size (int, optional): Forwarded to BaseDataLoader if
                supported. Defaults to None.
            block_buffer (int, optional): Forwarded to BaseDataLoader if
                supported. Defaults to None.

        Note:
            `val_split_mode`/`block_size`/`block_buffer` exist so a
            D-Fire config can carry the same keys as a FLAME one. They
            are passed through to `BaseDataLoader` only if its signature
            accepts them; otherwise they are dropped with a warning
            rather than raising, since block-style splitting is a
            countermeasure against consecutive-video-frame leakage and
            D-Fire is a still-image dataset that mostly doesn't have it.
        """
        augmentation = augmentation.lower()
        if augmentation not in self.AUGMENTATION_LEVELS:
            raise ValueError(
                f"Unknown augmentation level '{augmentation}'. "
                f"Expected one of {self.AUGMENTATION_LEVELS}."
            )

        self.img_size = img_size
        self.label_scheme = label_scheme.lower()
        self.classes = list(
            YoloClassificationDataset.SCHEME_CLASSES.get(self.label_scheme, [])
        )
        if not self.classes:
            raise ValueError(
                f"Unknown label_scheme '{label_scheme}'. Expected one of "
                f"{YoloClassificationDataset.LABEL_SCHEMES}."
            )
        self.num_classes = len(self.classes)

        root = Path(data_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"D-Fire root '{data_dir}' does not exist.")

        self.smoke_class_id, self.fire_class_id = _resolve_yolo_class_ids(root)

        split = "train" if training else "test"
        dirs = _resolve_split_dirs(root, split)
        if dirs is None:
            raise FileNotFoundError(
                f"Could not find a '{split}' split under '{data_dir}'. Expected "
                f"one of <root>/{split}/images, <root>/images/{split}, or "
                f"<root>/{split} containing image files."
            )
        images_dir, labels_dir = dirs

        if training:
            image_transform = self._build_train_transform(augmentation)
        else:
            image_transform = self._build_eval_transform()

        label_transform = transforms.Lambda(self.one_hot_encode)

        self.dataset = YoloClassificationDataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            label_scheme=self.label_scheme,
            smoke_class_id=self.smoke_class_id,
            fire_class_id=self.fire_class_id,
            transform=image_transform,
            target_transform=label_transform,
        )
        self.class_counts = self.dataset.class_counts

        # optional: the dataset's own held-out val folder, used instead of
        # slicing the train split
        self._official_val_dataset = None
        if training and use_official_val:
            val_dirs = _resolve_split_dirs(root, "val")
            if val_dirs is None:
                raise FileNotFoundError(
                    f"use_official_val=True but no val/valid split was found "
                    f"under '{data_dir}'."
                )
            val_images_dir, val_labels_dir = val_dirs
            self._official_val_dataset = YoloClassificationDataset(
                images_dir=val_images_dir,
                labels_dir=val_labels_dir,
                label_scheme=self.label_scheme,
                smoke_class_id=self.smoke_class_id,
                fire_class_id=self.fire_class_id,
                transform=self._build_eval_transform(),
                target_transform=label_transform,
            )
            validation_split = 0.0  # the train split stays whole

        # a second view of the *same* train split, unaugmented -- the
        # validation sampler indexes into this one when clean_validation
        # is on, so val_loss isn't measured through RandAugment
        self._clean_dataset = None
        if training and clean_validation and self._official_val_dataset is None:
            self._clean_dataset = YoloClassificationDataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                label_scheme=self.label_scheme,
                smoke_class_id=self.smoke_class_id,
                fire_class_id=self.fire_class_id,
                transform=self._build_eval_transform(),
                target_transform=label_transform,
            )

        # keys that only exist in some versions of BaseDataLoader -- pass
        # them on where they're understood, drop them loudly where they
        # aren't, so one config schema works against either version
        extra = {
            "val_split_mode": val_split_mode,
            "block_size": block_size,
            "block_buffer": block_buffer,
        }
        accepted = inspect.signature(BaseDataLoader.__init__).parameters
        forwarded = {k: v for k, v in extra.items() if v is not None and k in accepted}
        dropped = [k for k, v in extra.items() if v is not None and k not in accepted]
        if dropped:
            _logger.warning(
                "BaseDataLoader does not accept %s -- ignoring, the validation "
                "split will be a plain random subset of the train split.",
                ", ".join(dropped),
            )

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers,
            data_fraction=data_fraction if training else 1.0,
            **forwarded,
        )

    def split_validation(self):
        """Get the validation set, preferring the dataset's own val split."""
        if self._official_val_dataset is not None:
            kwargs = dict(self.init_kwargs)
            kwargs["dataset"] = self._official_val_dataset
            kwargs["shuffle"] = False
            return DataLoader(**kwargs)

        if self.valid_sampler is not None and self._clean_dataset is not None:
            kwargs = dict(self.init_kwargs)
            kwargs["dataset"] = self._clean_dataset  # same indices, no augmentation
            return DataLoader(sampler=self.valid_sampler, **kwargs)

        return super().split_validation()

    def _build_eval_transform(self) -> transforms.Compose:
        """Resize + normalize only -- no augmentation."""
        return transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.DFIRE_MEAN, self.DFIRE_STD),
            ]
        )

    def _build_train_transform(self, augmentation: str) -> transforms.Compose:
        """Build the training-time augmentation pipeline for a given level.

        Mirrors FlameDataLoader exactly so the two fire datasets stay
        directly comparable, including its deliberate use of a plain
        `Resize` rather than `RandomResizedCrop`: a random crop could
        remove the only fire/smoke region in the frame and silently
        mislabel the sample, which matters even more here, since D-Fire's
        boxes are often small.
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
            transforms.Normalize(self.DFIRE_MEAN, self.DFIRE_STD),
        ]

        if augmentation == "strong":
            ops.append(transforms.RandomErasing())  # operates on tensors

        return transforms.Compose(ops)

    def one_hot_encode(self, label: int) -> torch.Tensor:
        """Transforms the given label into a one-hot encoded tensor."""
        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot
