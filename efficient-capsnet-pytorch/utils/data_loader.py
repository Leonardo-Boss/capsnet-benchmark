import os
import pickle
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
        """
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
            valid_sampler (SubsetRandomSampler): Sampler for the validation set.
        """
        # no split performed
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0, "Validation set size should be at least 1."
            assert (
                split < self.n_samples
            ), "Validation set size should be at most equal to the number of samples."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
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


class CIFAR100Dataset(Dataset):
    """CIFAR-100 dataset loaded directly from the raw pickled batch files.

    Expects the following directory structure::

        data_dir/
            meta
            train
            test

    Attributes:
        classes (list[str]): Human-readable names for each label, ordered by
            label index. Coarse (superclass) or fine, depending on `coarse`.
        data (np.ndarray): Image data of shape (N, 32, 32, 3), dtype uint8.
        labels (np.ndarray): Integer labels of shape (N,).
    """

    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        coarse: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        """Initialize the dataset by reading the CIFAR-100 pickle files.

        Args:
            data_dir (str): Directory containing the `meta`, `train`, and
                `test` pickle files.
            train (bool, optional): Whether to load the training split
                (`train` file) or the test split (`test` file). Defaults to True.
            coarse (bool, optional): Whether to use the 20 coarse superclass
                labels instead of the 100 fine labels. Defaults to False.
            transform (Callable, optional): Transform applied to each PIL image.
            target_transform (Callable, optional): Transform applied to each
                integer label.
        """
        self.transform = transform
        self.target_transform = target_transform
        self.coarse = coarse

        split_file = "train" if train else "test"
        data_dict = self._unpickle(os.path.join(data_dir, split_file))
        meta_dict = self._unpickle(os.path.join(data_dir, "meta"))

        label_names_key = b"coarse_label_names" if coarse else b"fine_label_names"
        self.classes = [name.decode("utf-8") for name in meta_dict[label_names_key]]

        # raw data is (N, 3072): 1024 R, 1024 G, 1024 B -> (N, 3, 32, 32) -> (N, 32, 32, 3)
        raw = data_dict[b"data"]
        self.data = raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        label_key = b"coarse_labels" if coarse else b"fine_labels"
        self.labels = np.array(data_dict[label_key])

    @staticmethod
    def _unpickle(file_path: str) -> dict:
        """Load a CIFAR-100 pickle file as a byte-keyed dictionary."""
        with open(file_path, "rb") as fo:
            return pickle.load(fo, encoding="bytes")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        img = Image.fromarray(self.data[idx])
        label = int(self.labels[idx])

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


class Cifar100DataLoader(BaseDataLoader):
    """CIFAR-100 data loading class.

    Loads CIFAR-100 from the raw pickled batch files (`meta`, `train`, `test`)
    rather than via `torchvision.datasets.CIFAR100`, so `data_dir` should point
    directly at the folder containing those three files.

    Attributes:
        num_classes (int): Number of classes (20 if `coarse`, else 100).
        dataset (Dataset): The CIFAR-100 dataset.
    """

    # Per-channel mean/std computed over the CIFAR-100 training set.
    CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
    CIFAR100_STD = (0.2673, 0.2564, 0.2762)

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        shuffle: bool = True,
        validation_split: int | float = 0.0,
        num_workers: int = 1,
        training: bool = True,
        coarse: bool = False,
    ):
        """Initializes the Cifar100DataLoader with the given parameters.

        Standard augmentation (random crop + horizontal flip) is applied for
        the training split; only normalization is applied otherwise. Labels
        are one-hot encoded, matching MnistDataLoader's output convention.

        Args:
            data_dir (str): Directory containing the `meta`, `train`, and
                `test` files (e.g. "project/cifar100").
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
            coarse (bool, optional): Whether to use the 20 coarse superclass
                labels instead of the 100 fine labels. Defaults to False.
        """
        self.num_classes = 20 if coarse else 100

        if training:
            image_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.CIFAR100_MEAN, self.CIFAR100_STD),
                ]
            )
        else:
            image_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.CIFAR100_MEAN, self.CIFAR100_STD),
                ]
            )

        label_transform = transforms.Lambda(self.one_hot_encode)

        self.dataset = CIFAR100Dataset(
            data_dir,
            train=training,
            coarse=coarse,
            transform=image_transform,
            target_transform=label_transform,
        )

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )

    def one_hot_encode(self, label: int) -> torch.Tensor:
        """Transforms the given label into a one-hot encoded tensor."""
        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot
