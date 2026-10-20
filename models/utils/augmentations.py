"""Single source of truth for the image pipelines used by every dataset.

Both Cifar10DataLoader and FlameDataLoader (and test.py) build their
transforms here, so the augmentation regimes cannot drift apart again.

Regimes (identical for every dataset):
    none     : [Resize] -> ToTensor -> Normalize
    standard : [Resize] -> RandomCrop(S, pad=S/8) -> HFlip -> Rotation(+-15)
               -> ToTensor -> Normalize
    strong   : standard -> RandAugment (before ToTensor)
               ... Normalize -> RandomErasing (after Normalize)

Only two things legitimately differ per dataset and are passed in explicitly:
the normalisation statistics, and whether a Resize is needed (D-Fire images
come in heterogeneous native sizes, CIFAR-10 is already 32x32).

The crop padding scales with the image side (S/8): 4 px at 32x32, which is
exactly what CIFAR-10 used before, and 28 px at 224x224, so the relative
translation range (12.5% of the side) is the same on both datasets.
"""
from torchvision import transforms

AUGMENTATION_LEVELS = ("none", "standard", "strong")
ROTATION_DEGREES = 15
CROP_PADDING_FRACTION = 1 / 8


def crop_padding(img_size: int) -> int:
    return max(1, round(img_size * CROP_PADDING_FRACTION))


def build_transform(
    augmentation: str,
    img_size: int,
    mean,
    std,
    resize: bool = False,
    extra_tensor_ops=(),
) -> transforms.Compose:
    """Build the pipeline for one augmentation regime.

    Args:
        augmentation: 'none' | 'standard' | 'strong'.
        img_size: side length of the (square) image the model sees.
        mean, std: per-channel normalisation statistics.
        resize: prepend Resize((img_size, img_size)); needed for D-Fire.
        extra_tensor_ops: callables applied to the [0, 1] tensor right after
            ToTensor and before Normalize (used for the 'unseen' corruptions
            in test.py).
    """
    augmentation = augmentation.lower()
    if augmentation not in AUGMENTATION_LEVELS:
        raise ValueError(
            f"Unknown augmentation level '{augmentation}'. "
            f"Expected one of {AUGMENTATION_LEVELS}."
        )

    ops = []
    if resize:
        ops.append(transforms.Resize((img_size, img_size)))

    if augmentation in ("standard", "strong"):
        ops += [
            transforms.RandomCrop(img_size, padding=crop_padding(img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(ROTATION_DEGREES),
        ]
    if augmentation == "strong":
        ops.append(transforms.RandAugment())  # PIL op, torchvision defaults

    ops.append(transforms.ToTensor())
    ops += [transforms.Lambda(f) for f in extra_tensor_ops]
    ops.append(transforms.Normalize(mean, std))

    if augmentation == "strong":
        ops.append(transforms.RandomErasing())  # after Normalize: fills with 0 = mean

    return transforms.Compose(ops)
