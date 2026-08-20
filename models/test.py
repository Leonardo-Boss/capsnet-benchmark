"""Evaluate a trained model checkpoint on the CIFAR-10 test set under a
single, explicitly toggled condition -- run once per condition you want to
test, each producing its own CSV.

Examples:
    python test.py -c config.yaml --model saved/.../model_best.pth --augmentation strong
    python test.py -c config.yaml --model saved/.../model_best.pth --unseen-transformation
    python test.py -c config.yaml --model saved/.../model_best.pth --augmentation standard --unseen-transformation
    python test.py -c config.yaml --model saved/.../model_best.pth --unseen-transformation gaussian_noise occlusion
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import model.model as module_arch
from utils.config import Config
from utils.data_loader import Cifar10DataLoader
from utils.logger import get_logger
from utils.tools import read_yaml
from utils.unseen_transforms import UNSEEN_TRANSFORMS


def build_eval_transform(augmentation: str, unseen_names: list[str] | None) -> transforms.Compose:
    """Builds the image transform for this run's single condition.

    Args:
        augmentation: 'none' | 'standard' | 'strong' -- training-style
            regime applied to test images. Not a robustness test on its
            own (the model trained under this family) -- combine with
            unseen_names for that.
        unseen_names: list of UNSEEN_TRANSFORMS keys to apply, in order,
            stacked on top of the augmentation regime. None/empty list
            means no unseen corruption is applied.
    """
    ops = []

    if augmentation in ("standard", "strong"):
        ops += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ]
    if augmentation == "strong":
        ops.append(transforms.RandAugment())

    ops.append(transforms.ToTensor())

    for name in unseen_names or []:
        ops.append(transforms.Lambda(UNSEEN_TRANSFORMS[name]))

    if augmentation == "strong":
        ops.append(transforms.RandomErasing())

    ops.append(
        transforms.Normalize(Cifar10DataLoader.CIFAR10_MEAN, Cifar10DataLoader.CIFAR10_STD)
    )
    return transforms.Compose(ops)


def load_model(cfg: Config, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Builds the architecture from `cfg` and loads trained weights into it."""
    model = cfg.init_obj("arch", module_arch)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def evaluate(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
) -> list[dict]:
    """Runs the model over `data_loader`, returns one result dict per sample."""
    rows = []
    sample_idx = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            true_labels = labels.to(device)

            _, out_labels = model(images, mode="eval")
            pred_labels = out_labels.argmax(dim=1)
            confidence = out_labels.max(dim=1).values

            for i in range(images.shape[0]):
                rows.append({
                    "sample_idx": sample_idx,
                    "true_label": true_labels[i].item(),
                    "pred_label": pred_labels[i].item(),
                    "correct": int(true_labels[i].item() == pred_labels[i].item()),
                    "confidence": confidence[i].item(),
                })
                sample_idx += 1

    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_output_name(model_name: str, augmentation: str, unseen_names: list[str] | None) -> str:
    """Builds '{model_name}_{augmentation}_{unseen}.csv', including only the
    segments that are actually toggled on. 'none' augmentation is treated
    as not toggled (same as omitting --augmentation entirely).

    When unseen transforms are enabled, their specific names are embedded
    in the filename (e.g. 'unseen-gaussian_noise-occlusion'), so the file
    is self-describing without needing the console log. Running with
    every registered transform (--unseen-transformation with no names)
    produces 'unseen-all' rather than listing every name individually.
    """
    parts = [model_name]
    if augmentation != "none":
        parts.append(augmentation)
    if unseen_names is not None:
        if set(unseen_names) == set(UNSEEN_TRANSFORMS):
            parts.append("unseen-all")
        else:
            parts.append("unseen-" + "-".join(unseen_names))
    if len(parts) == 1:
        parts.append("clean")  # nothing toggled -- plain baseline eval
    return "_".join(parts) + ".csv"

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
    parser.add_argument(
        "-c", "--config", required=True,
        help="path to the training config.yaml used to build the model architecture",
    )
    parser.add_argument("--model", required=True, help="path to a saved .pth checkpoint")
    parser.add_argument(
        "--augmentation", default="none", choices=["none", "standard", "strong"],
        help="apply this training-style augmentation regime to test images (default: none)",
    )
    parser.add_argument(
        "--unseen-transformation", dest="unseen_transformation",
        nargs="*", default=None, choices=list(UNSEEN_TRANSFORMS),
        help=(
            "enable out-of-distribution corruption testing. Pass with no "
            "values to stack ALL registered corruptions together as one "
            "condition, or specific names to stack just those, in the "
            "given order. Omit this flag entirely to disable it."
        ),
    )
    parser.add_argument("--data_dir", default=None, help="override data_dir from config")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--output_dir", default="results", help="directory to write the CSV into")
    args = parser.parse_args()

    cfg_dict = read_yaml(args.config)
    cfg = Config(cfg_dict, run_id="eval_" + Path(args.model).stem)
    logger = get_logger(name="eval", verbosity=cfg["main"]["verbosity"])

    device = torch.device(
        "cuda" if cfg["main"]["cuda"] and torch.cuda.is_available() else "cpu"
    )
    logger.info("Using device  : %s", device)

    model = load_model(cfg, args.model, device)
    logger.info("Loaded checkpoint: %s", args.model)

    unseen_enabled = args.unseen_transformation is not None
    unseen_names = list(UNSEEN_TRANSFORMS) if unseen_enabled and not args.unseen_transformation \
        else args.unseen_transformation

    logger.info("Augmentation  : %s", args.augmentation)
    logger.info(
        "Unseen        : %s",
        ", ".join(unseen_names) if unseen_enabled else "disabled",
    )

    transform = build_eval_transform(args.augmentation, unseen_names)
    data_dir = args.data_dir or cfg["data_loader"]["args"]["data_dir"]
    dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    rows = evaluate(model, loader, device)
    acc = sum(r["correct"] for r in rows) / len(rows)
    logger.info("Accuracy      : %.4f", acc)

    model_name = Path(args.model).parent.name
    out_name = build_output_name(model_name, args.augmentation, unseen_names if unseen_enabled else None)
    out_path = Path(args.output_dir) / out_name

    write_csv(rows, out_path)
    logger.info("Wrote %d rows to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
