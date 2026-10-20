#!/usr/bin/env python3
"""Builds the final experiment matrix from the sweep winners.

For every (arch, dataset) it takes the already-generated template
configs/<arch>_<dataset>_none_1_1.yaml (arch, loss, data settings), swaps in
the tuned optimizer / scheduler from sweeps/<arch>_<dataset>/best.json, sets
the final epoch budget, and writes one YAML per (augmentation, fraction, seed)
plus train_final.sh (same style as train.sh).

LR schedule: gamma = 0.98**(100/EPOCHS), the same stretching the sweep used,
so a shorter budget ends at the same fraction of the initial learning rate
as the 100-epoch runs (0.98**100). Warm-up is the tuned fraction of EPOCHS.
Old checkpoints for the same run ids are moved to saved_archive/ so train.py
cannot resume them (it would load the old optimizer state over the tuned lr).
"""
import argparse, copy, json, shutil, time
from pathlib import Path
import yaml

FINAL_LR_RATIO = 0.98 ** 100
AUGS = ("none", "standard", "strong")
FRACS = (0.33, 0.66, 1)
SEEDS = (1, 2, 3)


def warmup_epochs(frac, epochs):
    return 0 if frac == 0 else max(1, round(frac * epochs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--archs", nargs="+", default=["ecaps", "resnet18", "deit_tiny", "mobilenetv3"])
    ap.add_argument("--datasets", nargs="+", default=["cifar_10", "dfire"])
    ap.add_argument("--sweeps", default="sweeps")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--script", default="train_final.sh")
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg_dir = Path(a.configs)
    backup = Path(f"{a.configs}_backup_before_final")
    if not a.dry_run and not backup.exists():
        shutil.copytree(cfg_dir, backup)
        print(f"backed up {cfg_dir}/ -> {backup}/")

    runs, written = [], 0
    for ds in a.datasets:
        for arch in a.archs:
            tpl_p = cfg_dir / f"{arch}_{ds}_none_1_1.yaml"
            best_p = Path(a.sweeps) / f"{arch}_{ds}" / "best.json"
            for p in (tpl_p, best_p):
                if not p.exists():
                    raise SystemExit(f"missing {p}")
            tpl = yaml.safe_load(open(tpl_p))
            hp = json.load(open(best_p))["hp"]
            name = tpl["main"]["name"]
            for aug in AUGS:
                for frac in FRACS:
                    for seed in SEEDS:
                        cfg = copy.deepcopy(tpl)
                        cfg["main"]["seed"] = seed
                        dl = cfg["data_loader"]["args"]
                        dl["augmentation"], dl["data_fraction"] = aug, frac
                        cfg["optimizer"] = {"type": "AdamW", "args": {
                            "lr": hp["lr"], "weight_decay": hp["weight_decay"]}}
                        cfg["lr_scheduler"] = {"type": "ExponentialLR",
                            "args": {"gamma": FINAL_LR_RATIO ** (1.0 / a.epochs)}}
                        w = warmup_epochs(hp["warmup_frac"], a.epochs)
                        if w:
                            cfg["lr_scheduler"]["warmup_epochs"] = w
                        t = cfg["trainer"]
                        t.update(epochs=a.epochs)
                        t.pop("save_checkpoints", None)
                        run_id = f"{name}_{aug}_{frac:g}_{seed}"
                        runs.append((name, run_id))
                        if not a.dry_run:
                            (cfg_dir / f"{run_id}.yaml").write_text(
                                yaml.safe_dump(cfg, sort_keys=False))
                        written += 1
            print(f"{arch:12s} {ds:9s} lr={hp['lr']:g} wd={hp['weight_decay']:g} "
                  f"warmup={warmup_epochs(hp['warmup_frac'], a.epochs)} ep, "
                  f"gamma={FINAL_LR_RATIO ** (1.0 / a.epochs):.4f}")
    print(f"{written} configs {'(dry run, nothing written)' if a.dry_run else 'written to ' + str(cfg_dir)}")

    # move stale runs out of the way so train.py cannot resume them
    if not a.no_archive:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        moved = 0
        for name, run_id in runs:
            for kind in ("models", "log"):
                src = Path("saved") / kind / name / run_id
                if src.exists():
                    dst = Path("saved_archive") / stamp / kind / name / run_id
                    print(f"archive {src} -> {dst}")
                    if not a.dry_run:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src), str(dst))
                    moved += 1
        print(f"archived {moved} stale run dir(s)")

    lines = ["#!/bin/bash", "nohup bash -c '", "    rm -f progress", "    touch progress",
             "    model=0", "    log () { model=$(( $model + 1 )); echo \"${model}/%d\" >> progress; }" % len(runs),
             "    run () { echo \"============================\"; echo \"RUNNING $@\"; eval \"$@\"; log; }"]
    for name, run_id in runs:
        lines.append(f"    run python train.py -c {a.configs}/{run_id}.yaml -i {run_id}")
    lines.append("' > train_final.log 2>&1 &")
    if not a.dry_run:
        Path(a.script).write_text("\n".join(lines) + "\n")
        Path(a.script).chmod(0o755)
    print(f"{len(runs)} runs listed in {a.script}")


if __name__ == "__main__":
    main()