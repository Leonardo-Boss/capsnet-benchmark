"""Deterministic validation helpers for checkpoint selection (task A1).

Two validation views of the SAME held-out indices:

  * clean : normalize-only transform, fixed sorted order -> fully deterministic
  * aug   : the training augmentation, but with a per-sample fixed RNG seed,
            so sample i receives the identical augmentation in every epoch
            and for every model. (The old 'augmented validation' drew fresh
            random augmentations each epoch, so its val_loss was noisy and
            not comparable across epochs or architectures.)

Also used by test.py so every test image receives the same random
transformation in all three architectures.

Assumes the augmentation pipeline draws from torch's CPU RNG (true for
torchvision's RandomCrop/Flip/Rotation/RandAugment/RandomErasing).
"""
import hashlib

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class FixedAugmentDataset(Dataset):
    """Item i always gets the same random augmentation."""

    def __init__(self, dataset, base_seed: int = 0):
        self.dataset = dataset
        self.base_seed = base_seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        # fork_rng restores the CPU RNG afterwards, so the training
        # augmentation stream is not disturbed. default_generator.manual_seed
        # seeds the CPU generator only (torch.manual_seed would also reseed CUDA).
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.base_seed * 1_000_003 + int(i))
            return self.dataset[i]


def make_val_loaders(loader, aug_seed: int = 0, batch_size=None, num_workers=None):
    """Returns (clean_loader, aug_loader) over `loader`'s validation indices.

    Returns (None, None) when the loader has no carved-out validation split
    (e.g. D-Fire with use_official_val=True).

    Both loaders own a private torch.Generator, so building their iterators
    never consumes the global RNG that drives training augmentation.
    """
    if getattr(loader, "valid_sampler", None) is None:
        return None, None
    idx = sorted(int(i) for i in loader.valid_sampler.indices)
    bs = batch_size or loader.init_kwargs["batch_size"]
    nw = loader.init_kwargs["num_workers"] if num_workers is None else num_workers
    clean_ds = getattr(loader, "_clean_dataset", None) or loader.dataset

    def _dl(ds):
        return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False, num_workers=nw,
                          generator=torch.Generator().manual_seed(0))

    return _dl(clean_ds), _dl(FixedAugmentDataset(loader.dataset, aug_seed))


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device) -> dict:
    """Sample-weighted mean loss and accuracy (one-hot labels, repo interface)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out_images, out_labels = model(images, labels, mode="eval")
        loss = criterion(images, labels, out_images, out_labels)
        b = images.shape[0]
        total_loss += loss.item() * b
        correct += (out_labels.argmax(1) == labels.argmax(1)).sum().item()
        n += b
    return {"loss": total_loss / n, "accuracy": correct / n, "n": n}


def split_fingerprint(loader) -> str:
    """Short hashes of the train/val index sets, to verify in the logs that
    every architecture trained and validated on identical partitions."""
    def h(idx):
        s = ",".join(str(int(i)) for i in sorted(idx))
        return hashlib.sha256(s.encode()).hexdigest()[:12]
    tr, va = getattr(loader, "train_sampler", None), getattr(loader, "valid_sampler", None)
    return (f"train={h(tr.indices) if tr is not None else 'n/a'} "
            f"val={h(va.indices) if va is not None else 'n/a'}")