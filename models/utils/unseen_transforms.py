"""Out-of-distribution 'unseen' corruptions for robustness evaluation.

Meant to probe generalization under distribution shift the model was never
exposed to in training -- deliberately disjoint from the crop/flip/
rotation/RandAugment/erasing family used by the training augmentation
regimes. Operate on tensors in [0, 1] (i.e. right after ToTensor, before
Normalize).
"""
import torch
import torchvision.transforms.functional as TF


def gaussian_noise(img: torch.Tensor, std: float = 0.08) -> torch.Tensor:
    return torch.clamp(img + torch.randn_like(img) * std, 0.0, 1.0)


def gaussian_blur(img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5) -> torch.Tensor:
    return TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)


def brightness_shift(img: torch.Tensor, factor: float = 1.6) -> torch.Tensor:
    return torch.clamp(img * factor, 0.0, 1.0)


def contrast_shift(img: torch.Tensor, factor: float = 0.4) -> torch.Tensor:
    return TF.adjust_contrast(img, contrast_factor=factor)

def large_rotation(img: torch.Tensor, degrees: float = 90.0) -> torch.Tensor:
    """Rotates by a random angle in [-degrees, degrees].

    Deliberately outside the +/-15deg range used by the 'standard'/'strong'
    training augmentation regimes, so this tests genuinely unseen rotation
    magnitudes rather than overlapping with what the model trained under.
    """
    angle = (torch.rand(1).item() * 2 - 1) * degrees
    return TF.rotate(img, angle)

def occlusion(img: torch.Tensor, patch_frac: float = 0.25) -> torch.Tensor:
    """Blacks out a random square patch covering ~patch_frac of the image."""
    _, h, w = img.shape
    size = int((h * w * patch_frac) ** 0.5)
    top = torch.randint(0, max(1, h - size + 1), (1,)).item()
    left = torch.randint(0, max(1, w - size + 1), (1,)).item()
    img = img.clone()
    img[:, top:top + size, left:left + size] = 0.0
    return img


UNSEEN_TRANSFORMS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "brightness_shift": brightness_shift,
    "contrast_shift": contrast_shift,
    "occlusion": occlusion,
    "large_rotation": large_rotation,
}
