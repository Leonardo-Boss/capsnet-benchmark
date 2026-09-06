#!/usr/bin/env python3
"""
analyze_robustness.py

Builds tables and figures comparing model robustness / equivariance to unseen
geometric transformations at test time, for a paper investigating whether
capsule networks (Efficient-CapsNet) retain an advantage over conventional
architectures (ResNet-18, DeiT-Tiny) under distribution shift.

USAGE
-----
    python analyze_robustness.py --data-dir /path/to/csvs --out-dir ./results

Expects files named:
    modelname_database_augtrain_trainfrac_seed_[strong_]<condition>.csv

e.g.  ecaps_cifar_10_standard_0.66_2_unseen-all.csv
      -> model=ecaps, train_aug=standard, train_amount=0.66, seed=2,
         test_aug=none, transformation=unseen-all

where <condition> in {clean, unseen-all, unseen-large_rotation}, and an
optional "strong_" prefix indicates strong augmentation was applied to the
test-time transformation itself (i.e. a harsher version of the shift).

Multiple seeds per config are expected (currently 3). All tables/figures
report the MEAN across seeds, with the STD across seeds also reported
(in tables) and shown as error bars (in figures) wherever accuracy is
plotted, so seed-to-seed variance is never silently hidden.

The script is tolerant of an incomplete/in-progress sweep: any
(model, train_aug, train_amount, test_aug, transformation) combo that is
partially or fully missing is simply left out of the plots/tables it can't
support, and a summary of what's missing/incomplete is printed at the top
of the run so you know what's still pending. Nothing crashes because a
combo (e.g. "standard" aug or the 0.66 fraction) isn't finished training yet.

Each CSV has columns: sample_idx,true_label,pred_label,correct,confidence
(confidence is a raw, unbounded score -- NOT a softmax probability).

OUTPUTS
-------
<out-dir>/tables/*.csv, *.md   -- all summary tables
<out-dir>/figures/*.png        -- all figures (300 dpi)
"""

import argparse
import glob
import os
import re
import sys
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG -- edit these if your naming vocabulary differs
# ----------------------------------------------------------------------------

MODEL_NAMES = ["deit_tiny", "ecaps", "resnet18"]
DATABASES = ["cifar_10"]

MODEL_LABELS = {
    "deit_tiny": "DeiT-Tiny",
    "ecaps": "Efficient-CapsNet",
    "resnet18": "ResNet-18",
}
MODEL_ORDER = ["resnet18", "deit_tiny", "ecaps"]
MODEL_COLORS = {
    "resnet18": "#4C72B0",
    "deit_tiny": "#DD8452",
    "ecaps": "#55A868",
}

TRAIN_AUG_LABELS = {
    "none": "No Train-Time Aug",
    "standard": "Standard Train-Time Aug",
    "strong": "Strong Train-Time Aug",
}
TRAIN_AUG_ORDER = ["none", "standard", "strong"]

# Base transformation (ignoring whether the *test-time* application was "strong")
TRANSFORM_LABELS = {
    "clean": "Clean (no shift)",
    "unseen-large_rotation": "Unseen: Large Rotation",
    "unseen-all": "Unseen: All Transforms",
}
CONDITION_ORDER = ["clean", "unseen-large_rotation", "unseen-all"]

REQUIRED_COLS = {"sample_idx", "true_label", "pred_label", "correct", "confidence"}

GROUP_COLS = ["model", "train_aug", "train_amount", "test_aug", "transformation"]
SEED_GROUP_COLS = GROUP_COLS + ["seed"]

# ----------------------------------------------------------------------------
# PARSING & LOADING
# ----------------------------------------------------------------------------

def build_pattern():
    model_pat = "|".join(re.escape(m) for m in sorted(MODEL_NAMES, key=len, reverse=True))
    db_pat = "|".join(re.escape(d) for d in sorted(DATABASES, key=len, reverse=True))
    aug_pat = "|".join(re.escape(a) for a in sorted(TRAIN_AUG_LABELS, key=len, reverse=True))
    return re.compile(
        rf"^(?P<model>{model_pat})_(?P<database>{db_pat})_"
        rf"(?P<train_aug>{aug_pat})_(?P<train_amount>[0-9.]+)_(?P<seed>[0-9]+)_(?P<suffix>.+)\.csv$"
    )


def parse_filename(fname, pattern):
    m = pattern.match(fname)
    if not m:
        return None
    d = m.groupdict()
    suffix = d.pop("suffix")
    if suffix.startswith("strong_"):
        test_aug = "strong"
        transformation = suffix[len("strong_"):]
    else:
        test_aug = "none"
        transformation = suffix
    d["test_aug"] = test_aug
    d["transformation"] = transformation
    d["train_amount"] = float(d["train_amount"])
    d["seed"] = int(d["seed"])
    return d


def load_all(data_dir):
    pattern = build_pattern()
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames = []
    skipped = []
    for f in files:
        base = os.path.basename(f)
        meta = parse_filename(base, pattern)
        if meta is None:
            skipped.append(base)
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  ! failed to read {base}: {e}", file=sys.stderr)
            continue
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            print(f"  ! {base} missing columns {missing}, skipping", file=sys.stderr)
            continue
        for k, v in meta.items():
            df[k] = v
        df["source_file"] = base
        frames.append(df)

    if skipped:
        print(f"Note: {len(skipped)} file(s) did not match the naming pattern and were skipped:")
        for s in skipped:
            print(f"    {s}")

    if not frames:
        raise RuntimeError("No valid CSV files found/parsed in data_dir.")

    df = pd.concat(frames, ignore_index=True)
    df["correct"] = df["correct"].astype(int)
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["transformation_label"] = df["transformation"].map(TRANSFORM_LABELS).fillna(df["transformation"])
    return df


def report_missing_combos(df, out_dir):
    """Print (and save) which expected (model, train_aug, train_amount,
    test_aug, transformation) combos are absent or have fewer seeds than the
    max observed -- so an in-progress sweep (e.g. 'standard' aug / 0.66
    fraction still training) is visible rather than silently ignored."""
    models = sorted(df["model"].unique())
    train_augs = sorted(df["train_aug"].unique(), key=lambda x: TRAIN_AUG_ORDER.index(x) if x in TRAIN_AUG_ORDER else 99)
    train_amounts = sorted(df["train_amount"].unique())
    test_augs = sorted(df["test_aug"].unique())
    transformations = sorted(df["transformation"].unique(), key=lambda x: CONDITION_ORDER.index(x) if x in CONDITION_ORDER else 99)
    max_seeds = df.groupby(GROUP_COLS)["seed"].nunique().max()

    present = df.groupby(GROUP_COLS)["seed"].nunique().reset_index()
    present_keys = {tuple(r[c] for c in GROUP_COLS): r["seed"] for _, r in present.iterrows()}

    rows = []
    for combo in product(models, train_augs, train_amounts, test_augs, transformations):
        # skip nonsensical combos: 'strong' test_aug only pairs with transformations != clean
        if combo[3] == "strong" and combo[4] == "clean":
            continue
        n_seeds = present_keys.get(combo, 0)
        status = "OK" if n_seeds == max_seeds else ("MISSING" if n_seeds == 0 else "PARTIAL")
        if status != "OK":
            rows.append({
                "model": combo[0], "train_aug": combo[1], "train_amount": combo[2],
                "test_aug": combo[3], "transformation": combo[4],
                "seeds_found": n_seeds, "seeds_expected": max_seeds, "status": status,
            })
    if rows:
        missing_df = pd.DataFrame(rows)
        print(f"\nSweep completeness check (expecting {max_seeds} seed(s) per combo, "
              f"based on the max seen anywhere in the data):")
        print(f"  {len(missing_df)} combo(s) incomplete or missing -- these will simply be")
        print(f"  left out of any plot/table cell that needs them. Full list saved to")
        print(f"  <out-dir>/tables/00_incomplete_sweep_combos.csv")
        for _, r in missing_df.head(15).iterrows():
            print(f"    [{r['status']:7s}] {r['model']}, train_aug={r['train_aug']}, "
                  f"frac={r['train_amount']}, test_aug={r['test_aug']}, "
                  f"{r['transformation']}  ({r['seeds_found']}/{r['seeds_expected']} seeds)")
        if len(missing_df) > 15:
            print(f"    ... and {len(missing_df) - 15} more")
        tdir = Path(out_dir) / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        missing_df.to_csv(tdir / "00_incomplete_sweep_combos.csv", index=False)
    else:
        print("\nSweep completeness check: all expected combos present with a full seed count.")


# ----------------------------------------------------------------------------
# SUMMARY STATS (seed-level, then aggregated across seeds)
# ----------------------------------------------------------------------------

def compute_seed_level_summary(df):
    """One row per (model, train_aug, train_amount, test_aug, transformation, seed)."""
    rows = []
    for keys, g in df.groupby(SEED_GROUP_COLS):
        model, train_aug, train_amount, test_aug, transformation, seed = keys
        n = len(g)
        acc = g["correct"].mean()
        correct_conf = g.loc[g["correct"] == 1, "confidence"]
        incorrect_conf = g.loc[g["correct"] == 0, "confidence"]
        rows.append({
            "model": model, "train_aug": train_aug, "train_amount": train_amount,
            "test_aug": test_aug, "transformation": transformation, "seed": seed,
            "n": n,
            "accuracy": acc,
            "mean_confidence": g["confidence"].mean(),
            "mean_confidence_correct": correct_conf.mean() if len(correct_conf) else np.nan,
            "mean_confidence_incorrect": incorrect_conf.mean() if len(incorrect_conf) else np.nan,
            "confidence_gap": (correct_conf.mean() - incorrect_conf.mean())
                                if len(correct_conf) and len(incorrect_conf) else np.nan,
        })
    return pd.DataFrame(rows)


def aggregate_across_seeds(seed_summary):
    """Collapse the seed dimension: mean is the headline number used everywhere;
    _std / n_seeds columns are carried alongside for error bars & transparency."""
    agg = seed_summary.groupby(GROUP_COLS).agg(
        n_seeds=("seed", "nunique"),
        n=("n", "sum"),
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        mean_confidence=("mean_confidence", "mean"),
        mean_confidence_correct=("mean_confidence_correct", "mean"),
        mean_confidence_incorrect=("mean_confidence_incorrect", "mean"),
        confidence_gap=("confidence_gap", "mean"),
        confidence_gap_std=("confidence_gap", "std"),
    ).reset_index()
    agg["accuracy_std"] = agg["accuracy_std"].fillna(0.0)
    agg["confidence_gap_std"] = agg["confidence_gap_std"].fillna(0.0)
    agg["model_label"] = agg["model"].map(MODEL_LABELS).fillna(agg["model"])
    agg["transformation_label"] = agg["transformation"].map(TRANSFORM_LABELS).fillna(agg["transformation"])
    return agg


def compute_summary(df):
    seed_summary = compute_seed_level_summary(df)
    summary = aggregate_across_seeds(seed_summary)
    return summary, seed_summary


def add_robustness_metrics(summary):
    """For each (model, train_aug, train_amount, test_aug), express accuracy
    relative to that same config's 'clean' accuracy -> robustness retained (%)
    and absolute/relative equivariance gap. Operates on the seed-averaged
    summary; accuracy_std for 'clean' is carried through for reference."""
    out = []
    key_cols = ["model", "train_aug", "train_amount", "test_aug"]
    for keys, g in summary.groupby(key_cols):
        clean_rows = g[g["transformation"] == "clean"]
        clean_acc = clean_rows["accuracy"].iloc[0] if len(clean_rows) else np.nan
        g = g.copy()
        g["clean_accuracy"] = clean_acc
        g["accuracy_drop"] = clean_acc - g["accuracy"]
        g["relative_drop_pct"] = np.where(
            clean_acc > 0, 100 * (clean_acc - g["accuracy"]) / clean_acc, np.nan
        )
        g["robustness_retained_pct"] = np.where(
            clean_acc > 0, 100 * g["accuracy"] / clean_acc, np.nan
        )
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_robustness_metrics_seed_level(seed_summary):
    """Paired-by-seed version of the drop calc: for each seed, drop = that
    seed's clean accuracy minus that seed's shifted accuracy, THEN average
    across seeds. This is the statistically correct way to get an error bar
    on the equivariance gap (rather than combining marginal stds)."""
    out = []
    key_cols = ["model", "train_aug", "train_amount", "test_aug", "seed"]
    for keys, g in seed_summary.groupby(key_cols):
        clean_rows = g[g["transformation"] == "clean"]
        clean_acc = clean_rows["accuracy"].iloc[0] if len(clean_rows) else np.nan
        g = g.copy()
        g["clean_accuracy"] = clean_acc
        g["accuracy_drop"] = clean_acc - g["accuracy"]
        out.append(g)
    seed_rob = pd.concat(out, ignore_index=True)
    agg = seed_rob[seed_rob["transformation"] != "clean"].groupby(
        ["model", "train_aug", "train_amount", "test_aug", "transformation"]
    ).agg(
        n_seeds=("seed", "nunique"),
        accuracy_drop=("accuracy_drop", "mean"),
        accuracy_drop_std=("accuracy_drop", "std"),
    ).reset_index()
    agg["accuracy_drop_std"] = agg["accuracy_drop_std"].fillna(0.0)
    return agg


# ----------------------------------------------------------------------------
# TABLES
# ----------------------------------------------------------------------------

def save_table(df, out_dir, name, index=False):
    tdir = Path(out_dir) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    csv_path = tdir / f"{name}.csv"
    md_path = tdir / f"{name}.md"
    df.to_csv(csv_path, index=index)
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=index))
    print(f"  wrote {csv_path.name} / {md_path.name}")


def make_tables(summary, seed_summary, out_dir):
    print("\nBuilding tables...")

    # 1. Master results table (mean +/- std across seeds, every condition/config)
    master = summary.sort_values(["model", "train_aug", "train_amount", "test_aug", "transformation"])
    cols = ["model_label", "train_aug", "train_amount", "test_aug", "transformation_label",
            "n_seeds", "n", "accuracy", "accuracy_std",
            "mean_confidence_correct", "mean_confidence_incorrect", "confidence_gap", "confidence_gap_std"]
    save_table(master[cols].round(4), out_dir, "01_master_results")

    # 1b. Per-seed raw results, for appendix / reproducibility
    seed_cols = ["model", "train_aug", "train_amount", "test_aug", "transformation", "seed",
                 "n", "accuracy", "confidence_gap"]
    save_table(seed_summary[seed_cols].sort_values(
        ["model", "train_aug", "train_amount", "test_aug", "transformation", "seed"]
    ).round(4), out_dir, "01b_per_seed_results")

    # 2. Accuracy pivot: rows = model x train setting, cols = test condition
    base = summary[summary["test_aug"] == "none"].copy()
    base["train_setting"] = base["model_label"] + " | train_aug=" + base["train_aug"] + \
                             " | frac=" + base["train_amount"].astype(str)
    base["accuracy_fmt"] = base.apply(lambda r: f"{r['accuracy']:.3f} +/- {r['accuracy_std']:.3f}", axis=1)
    pivot_acc = base.pivot_table(index="train_setting", columns="transformation_label",
                                  values="accuracy_fmt", aggfunc="first")
    pivot_acc = pivot_acc.reindex(columns=[TRANSFORM_LABELS[c] for c in CONDITION_ORDER if TRANSFORM_LABELS[c] in pivot_acc.columns])
    save_table(pivot_acc.reset_index(), out_dir, "02_accuracy_pivot_normal_test_aug", index=False)

    # 2b. Same, but for strong test-time augmentation of the transform
    strong = summary[summary["test_aug"] == "strong"].copy()
    if len(strong):
        strong["train_setting"] = strong["model_label"] + " | train_aug=" + strong["train_aug"] + \
                                   " | frac=" + strong["train_amount"].astype(str)
        strong["accuracy_fmt"] = strong.apply(lambda r: f"{r['accuracy']:.3f} +/- {r['accuracy_std']:.3f}", axis=1)
        pivot_acc_strong = strong.pivot_table(index="train_setting", columns="transformation_label",
                                               values="accuracy_fmt", aggfunc="first")
        save_table(pivot_acc_strong.reset_index(), out_dir, "02b_accuracy_pivot_strong_test_aug", index=False)

    # 3. Robustness / equivariance-gap table (mean-of-means version, all configs)
    rob = add_robustness_metrics(summary)
    rob_cols = ["model_label", "train_aug", "train_amount", "test_aug", "transformation_label",
                "clean_accuracy", "accuracy", "accuracy_drop", "relative_drop_pct", "robustness_retained_pct"]
    rob_sorted = rob[rob["transformation"] != "clean"].sort_values(
        ["transformation", "test_aug", "model", "train_aug", "train_amount"]
    )
    save_table(rob_sorted[rob_cols].round(3), out_dir, "03_robustness_gap")

    # 3b. Paired-by-seed equivariance gap with proper std (preferred for the paper)
    rob_seed = add_robustness_metrics_seed_level(seed_summary)
    rob_seed["model_label"] = rob_seed["model"].map(MODEL_LABELS).fillna(rob_seed["model"])
    rob_seed["transformation_label"] = rob_seed["transformation"].map(TRANSFORM_LABELS).fillna(rob_seed["transformation"])
    save_table(rob_seed.sort_values(["transformation", "test_aug", "model", "train_aug", "train_amount"]).round(4),
               out_dir, "03b_robustness_gap_paired_by_seed")

    # 4. Head-to-head "best model per condition" table
    best = summary.loc[summary.groupby(["train_aug", "train_amount", "test_aug", "transformation"])["accuracy"].idxmax()]
    save_table(best[["train_aug", "train_amount", "test_aug", "transformation_label", "model_label", "accuracy", "accuracy_std"]]
               .sort_values(["transformation_label", "train_aug", "train_amount"]).round(4),
               out_dir, "04_best_model_per_condition")

    # 5. CapsNet advantage summary: ecaps accuracy minus best-of-the-rest, per condition
    adv_rows = []
    for keys, g in summary.groupby(["train_aug", "train_amount", "test_aug", "transformation"]):
        train_aug, train_amount, test_aug, transformation = keys
        ecaps_row = g[g["model"] == "ecaps"]
        others = g[g["model"] != "ecaps"]
        if len(ecaps_row) == 0 or len(others) == 0:
            continue
        ecaps_acc = ecaps_row["accuracy"].iloc[0]
        ecaps_std = ecaps_row["accuracy_std"].iloc[0]
        best_other = others.loc[others["accuracy"].idxmax()]
        adv_rows.append({
            "train_aug": train_aug, "train_amount": train_amount, "test_aug": test_aug,
            "transformation": TRANSFORM_LABELS.get(transformation, transformation),
            "ecaps_accuracy": ecaps_acc,
            "ecaps_accuracy_std": ecaps_std,
            "best_baseline": best_other["model_label"],
            "best_baseline_accuracy": best_other["accuracy"],
            "best_baseline_accuracy_std": best_other["accuracy_std"],
            "ecaps_advantage_pp": 100 * (ecaps_acc - best_other["accuracy"]),
            "ecaps_advantage_pp_std": 100 * np.sqrt(ecaps_std**2 + best_other["accuracy_std"]**2),
        })
    adv_df = pd.DataFrame(adv_rows).sort_values(["transformation", "train_aug", "train_amount"])
    save_table(adv_df.round(3), out_dir, "05_ecaps_advantage_over_best_baseline")

    # 6. Seed variability table (min/max/std across seeds per config)
    var = seed_summary.groupby(GROUP_COLS)["accuracy"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    var.columns = list(GROUP_COLS) + ["accuracy_mean", "accuracy_std", "accuracy_min", "accuracy_max", "n_seeds"]
    var["model_label"] = var["model"].map(MODEL_LABELS).fillna(var["model"])
    var["transformation_label"] = var["transformation"].map(TRANSFORM_LABELS).fillna(var["transformation"])
    save_table(var.sort_values(["model", "train_aug", "train_amount", "test_aug", "transformation"]).round(4),
               out_dir, "06_seed_variability")

    return rob, rob_seed, adv_df


# ----------------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------------

def savefig(fig, out_dir, name):
    fdir = Path(out_dir) / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    path = fdir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def fig_grouped_bar_accuracy(summary, out_dir):
    """Grouped bars (mean +/- std across seeds): x = transformation condition,
    groups = model, one panel per (train_aug, train_amount), test_aug='none'."""
    base = summary[summary["test_aug"] == "none"]
    if base.empty:
        return
    combos = base[["train_aug", "train_amount"]].drop_duplicates().sort_values(["train_aug", "train_amount"])
    n_panels = len(combos)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    conditions = [c for c in CONDITION_ORDER if c in base["transformation"].unique()]
    x = np.arange(len(conditions))
    width = 0.8 / len(MODEL_ORDER)

    for ax, (_, combo) in zip(axes, combos.iterrows()):
        sub = base[(base["train_aug"] == combo["train_aug"]) & (base["train_amount"] == combo["train_amount"])]
        for i, model in enumerate(MODEL_ORDER):
            mrow = sub[sub["model"] == model].set_index("transformation")
            vals = [mrow.loc[c, "accuracy"] if c in mrow.index else np.nan for c in conditions]
            errs = [mrow.loc[c, "accuracy_std"] if c in mrow.index else 0 for c in conditions]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, yerr=errs, capsize=3,
                   label=MODEL_LABELS[model], color=MODEL_COLORS[model])
        ax.set_xticks(x)
        ax.set_xticklabels([TRANSFORM_LABELS[c] for c in conditions], rotation=15, ha="right")
        ax.set_title(f"Train Aug: {TRAIN_AUG_LABELS.get(combo['train_aug'], combo['train_aug'])}\nTrain Data Fraction: {combo['train_amount']}")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Test Accuracy")
    axes[-1].legend(loc="upper right", frameon=True)
    fig.suptitle("Accuracy by Model Across Clean and Unseen-Transformation Test Conditions\n(error bars = std across seeds)", y=1.08, fontsize=13)
    fig.tight_layout()
    savefig(fig, out_dir, "fig01_grouped_bar_accuracy_by_condition")


def fig_equivariance_gap(rob_seed, out_dir):
    """Bar chart of accuracy drop (clean -> unseen) per model, faceted by
    transformation, for test_aug == 'none'. Uses the paired-by-seed drop with
    a proper cross-seed std as the error bar."""
    sub = rob_seed[rob_seed["test_aug"] == "none"]
    if sub.empty:
        return
    transformations = [t for t in CONDITION_ORDER if t in sub["transformation"].unique() and t != "clean"]
    fig, axes = plt.subplots(1, len(transformations), figsize=(6 * len(transformations), 4.5), sharey=True)
    if len(transformations) == 1:
        axes = [axes]

    train_settings = sub[["train_aug", "train_amount"]].drop_duplicates().sort_values(["train_aug", "train_amount"])
    labels = [f"{TRAIN_AUG_LABELS.get(r.train_aug, r.train_aug)}\nfrac={r.train_amount}" for r in train_settings.itertuples()]
    x = np.arange(len(train_settings))
    width = 0.8 / len(MODEL_ORDER)

    for ax, transformation in zip(axes, transformations):
        tsub = sub[sub["transformation"] == transformation]
        for i, model in enumerate(MODEL_ORDER):
            vals, errs = [], []
            for r in train_settings.itertuples():
                row = tsub[(tsub["model"] == model) & (tsub["train_aug"] == r.train_aug) &
                           (tsub["train_amount"] == r.train_amount)]
                vals.append(100 * row["accuracy_drop"].iloc[0] if len(row) else np.nan)
                errs.append(100 * row["accuracy_drop_std"].iloc[0] if len(row) else 0)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, yerr=errs, capsize=3,
                   label=MODEL_LABELS[model], color=MODEL_COLORS[model])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(TRANSFORM_LABELS.get(transformation, transformation))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Accuracy Drop from Clean (pp)\n(lower = more equivariant / robust)")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Equivariance Gap: Accuracy Loss Under Unseen Transformations\n(error bars = std of per-seed drop)", y=1.05, fontsize=13)
    fig.tight_layout()
    savefig(fig, out_dir, "fig02_equivariance_gap_bars")


def fig_heatmap(summary, out_dir):
    """Heatmap of mean accuracy: rows = model x train setting, cols = condition."""
    base = summary[summary["test_aug"] == "none"].copy()
    if base.empty:
        return
    base["row_label"] = base["model_label"] + " (" + base["train_aug"] + ", frac=" + base["train_amount"].astype(str) + ")"
    conditions = [c for c in CONDITION_ORDER if c in base["transformation"].unique()]
    pivot = base.pivot_table(index="row_label", columns="transformation", values="accuracy")
    pivot = pivot.reindex(columns=conditions)
    order = []
    for model in MODEL_ORDER:
        rows = sorted([r for r in pivot.index if MODEL_LABELS.get(model, model) in r])
        order.extend(rows)
    pivot = pivot.reindex([r for r in order if r in pivot.index])

    fig, ax = plt.subplots(figsize=(2.6 * len(conditions) + 2, 0.55 * len(pivot) + 2))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([TRANSFORM_LABELS[c] for c in conditions], rotation=20, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="black" if 0.3 < val < 0.8 else "white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean Accuracy (across seeds)")
    ax.set_title("Accuracy Heatmap: Model x Training Setting x Test Condition")
    fig.tight_layout()
    savefig(fig, out_dir, "fig03_accuracy_heatmap")


def fig_confidence_box(df, out_dir):
    """Boxplots of confidence (pooled across seeds), split by correct/incorrect,
    per model, for a representative slice: no train-time aug, largest available
    fraction of training data, normal test-time aug."""
    if df.empty:
        return
    rep_amount = df["train_amount"].max()
    sub = df[(df["test_aug"] == "none") & (df["train_aug"] == "none") & (df["train_amount"] == rep_amount)]
    if sub.empty:
        return
    conditions = [c for c in CONDITION_ORDER if c in sub["transformation"].unique()]
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4.5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        csub = sub[sub["transformation"] == cond]
        data, positions, colors, ticklabels = [], [], [], []
        pos = 0
        for model in MODEL_ORDER:
            for correctness, tag in [(1, "Correct"), (0, "Incorrect")]:
                vals = csub[(csub["model"] == model) & (csub["correct"] == correctness)]["confidence"]
                if len(vals) > 0:
                    data.append(vals.values)
                    positions.append(pos)
                    colors.append(MODEL_COLORS.get(model, "#888") if correctness == 1 else "#B0B0B0")
                    ticklabels.append(f"{MODEL_LABELS.get(model, model)}\n{tag}")
                pos += 1
            pos += 0.6
        if not data:
            continue
        bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(ticklabels, rotation=45, ha="right", fontsize=8)
        ax.set_title(TRANSFORM_LABELS.get(cond, cond))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Raw Confidence Score")
    fig.suptitle(f"Confidence Distributions (Correct vs Incorrect Predictions, pooled across seeds)\n"
                 f"[Slice shown: no train-time aug, frac={rep_amount}, normal test-time aug]",
                 y=1.06, fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig04_confidence_boxplots")


def fig_radar(summary, out_dir):
    """Radar/spider chart: one axis per test condition, one polygon per model
    (mean accuracy across seeds), one panel per (train_aug, train_amount)."""
    base = summary[summary["test_aug"] == "none"]
    if base.empty:
        return
    combos = base[["train_aug", "train_amount"]].drop_duplicates().sort_values(["train_aug", "train_amount"])
    conditions = [c for c in CONDITION_ORDER if c in base["transformation"].unique()]
    n = len(conditions)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(combos), figsize=(5 * len(combos), 5), subplot_kw=dict(polar=True))
    if len(combos) == 1:
        axes = [axes]

    for ax, (_, combo) in zip(axes, combos.iterrows()):
        sub = base[(base["train_aug"] == combo["train_aug"]) & (base["train_amount"] == combo["train_amount"])]
        for model in MODEL_ORDER:
            mrow = sub[sub["model"] == model].set_index("transformation")
            vals = [mrow.loc[c, "accuracy"] if c in mrow.index else 0 for c in conditions]
            vals += vals[:1]
            ax.plot(angles, vals, label=MODEL_LABELS.get(model, model), color=MODEL_COLORS.get(model, "#888"), linewidth=2)
            ax.fill(angles, vals, color=MODEL_COLORS.get(model, "#888"), alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([TRANSFORM_LABELS[c] for c in conditions], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{TRAIN_AUG_LABELS.get(combo['train_aug'], combo['train_aug'])}, frac={combo['train_amount']}", fontsize=11, y=1.1)
    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.suptitle("Robustness Profile by Model (mean accuracy across seeds)", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "fig05_radar_robustness_profile")


def fig_data_efficiency(summary, out_dir):
    """Line plot: x = train_amount, y = mean accuracy (+/- std across seeds),
    one line per model, one panel per test condition, linestyle = train_aug."""
    base = summary[summary["test_aug"] == "none"]
    if base.empty:
        return
    conditions = [c for c in CONDITION_ORDER if c in base["transformation"].unique()]
    train_augs_present = [a for a in TRAIN_AUG_ORDER if a in base["train_aug"].unique()]
    linestyles = ["--", "-", ":"]
    markers = ["o", "s", "^"]

    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4.5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        csub = base[base["transformation"] == cond]
        for model in MODEL_ORDER:
            for j, train_aug in enumerate(train_augs_present):
                msub = csub[(csub["model"] == model) & (csub["train_aug"] == train_aug)].sort_values("train_amount")
                if len(msub):
                    ax.errorbar(msub["train_amount"], msub["accuracy"], yerr=msub["accuracy_std"],
                                linestyle=linestyles[j % len(linestyles)], marker=markers[j % len(markers)],
                                capsize=3, color=MODEL_COLORS.get(model, "#888"),
                                label=f"{MODEL_LABELS.get(model, model)} ({TRAIN_AUG_LABELS.get(train_aug, train_aug)})")
        ax.set_title(TRANSFORM_LABELS.get(cond, cond))
        ax.set_xlabel("Fraction of Training Data")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Test Accuracy")
    axes[-1].legend(loc="lower right", fontsize=6.5)
    fig.suptitle("Data Efficiency: Accuracy vs. Amount of Training Data\n(error bars = std across seeds)", y=1.05, fontsize=13)
    fig.tight_layout()
    savefig(fig, out_dir, "fig06_data_efficiency_lines")


def fig_advantage_bar(adv_df, out_dir):
    """Diverging bar chart of Efficient-CapsNet's accuracy advantage (pp) over
    the best baseline, per condition/training setting, with propagated std."""
    if adv_df.empty:
        return
    adv_df = adv_df.copy()
    adv_df["setting"] = adv_df["transformation"] + " | " + adv_df["train_aug"] + \
                         " | frac=" + adv_df["train_amount"].astype(str) + \
                         " | test_aug=" + adv_df["test_aug"]
    adv_df = adv_df.sort_values("ecaps_advantage_pp")

    fig, ax = plt.subplots(figsize=(9, 0.35 * len(adv_df) + 2))
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in adv_df["ecaps_advantage_pp"]]
    ax.barh(adv_df["setting"], adv_df["ecaps_advantage_pp"], xerr=adv_df["ecaps_advantage_pp_std"],
            capsize=3, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Efficient-CapsNet Accuracy Advantage over Best Baseline (pp)")
    ax.set_title("Where Does Efficient-CapsNet Retain (or Lose) Its Edge?\n(error bars = propagated cross-seed std)")
    ax.grid(axis="x", alpha=0.3)
    legend_elems = [Patch(facecolor="#55A868", label="CapsNet ahead"),
                    Patch(facecolor="#C44E52", label="CapsNet behind")]
    ax.legend(handles=legend_elems, loc="lower right")
    fig.tight_layout()
    savefig(fig, out_dir, "fig07_ecaps_advantage_diverging_bar")


def fig_confidence_gap_scatter(summary, out_dir):
    """Scatter: x = mean accuracy, y = mean confidence_gap, colored by model,
    marker shape by condition."""
    base = summary[summary["test_aug"] == "none"]
    if base.empty:
        return
    markers = {"clean": "o", "unseen-large_rotation": "^", "unseen-all": "s"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for model in MODEL_ORDER:
        msub = base[base["model"] == model]
        for cond, marker in markers.items():
            csub = msub[msub["transformation"] == cond]
            if len(csub):
                ax.scatter(csub["accuracy"], csub["confidence_gap"], color=MODEL_COLORS.get(model, "#888"),
                           marker=marker, s=90, edgecolor="black", linewidth=0.5,
                           label=f"{MODEL_LABELS.get(model, model)} - {TRANSFORM_LABELS.get(cond, cond)}")
    ax.set_xlabel("Mean Accuracy (across seeds)")
    ax.set_ylabel("Mean Confidence Gap (correct - incorrect)")
    ax.set_title("Accuracy vs. Confidence Separation\n(Does confidence stay informative under shift?)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=1)
    fig.tight_layout()
    savefig(fig, out_dir, "fig08_confidence_gap_vs_accuracy_scatter")


def fig_per_class_heatmap(df, out_dir):
    """Per-class accuracy heatmap (pooled across seeds) for the toughest
    condition (unseen-all), comparing models at no train aug / largest
    available training fraction."""
    if df.empty:
        return
    rep_amount = df["train_amount"].max()
    sub = df[(df["test_aug"] == "none") & (df["train_aug"] == "none") &
             (df["train_amount"] == rep_amount) & (df["transformation"] == "unseen-all")]
    if sub.empty:
        return
    pivot = sub.groupby(["model", "true_label"])["correct"].mean().unstack("true_label")
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
    fig, ax = plt.subplots(figsize=(1.0 * pivot.shape[1] + 3, 0.6 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="black" if 0.3 < val < 0.8 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Per-Class Accuracy")
    ax.set_xlabel("True Class Label")
    ax.set_title(f"Per-Class Accuracy Under Unseen-All Shift (pooled across seeds)\n[no train aug, frac={rep_amount}]")
    fig.tight_layout()
    savefig(fig, out_dir, "fig09_per_class_accuracy_heatmap")


def fig_seed_consistency(seed_summary, out_dir):
    """Dot/strip plot showing each individual seed's accuracy (jittered) plus
    the mean, per model, for the hardest condition (unseen-all, test_aug=none)
    -- makes it visually obvious whether an advantage is consistent across
    seeds or driven by a single lucky run."""
    sub = seed_summary[(seed_summary["test_aug"] == "none") & (seed_summary["transformation"] == "unseen-all")]
    if sub.empty:
        return
    combos = sub[["train_aug", "train_amount"]].drop_duplicates().sort_values(["train_aug", "train_amount"])
    fig, axes = plt.subplots(1, len(combos), figsize=(4 * len(combos), 4.5), sharey=True)
    if len(combos) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)

    for ax, (_, combo) in zip(axes, combos.iterrows()):
        csub = sub[(sub["train_aug"] == combo["train_aug"]) & (sub["train_amount"] == combo["train_amount"])]
        for i, model in enumerate(MODEL_ORDER):
            msub = csub[csub["model"] == model]
            if msub.empty:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(msub))
            ax.scatter(np.full(len(msub), i) + jitter, msub["accuracy"], color=MODEL_COLORS.get(model, "#888"),
                       s=50, alpha=0.8, edgecolor="black", linewidth=0.4, zorder=3)
            ax.hlines(msub["accuracy"].mean(), i - 0.2, i + 0.2, color="black", linewidth=2, zorder=4)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in MODEL_ORDER], rotation=15, ha="right")
        ax.set_title(f"{TRAIN_AUG_LABELS.get(combo['train_aug'], combo['train_aug'])}, frac={combo['train_amount']}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Accuracy (unseen-all)")
    fig.suptitle("Per-Seed Accuracy Under the Hardest Shift (unseen-all)\ndots = individual seeds, bar = mean", y=1.06, fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig10_seed_consistency_stripplot")


def make_figures(df, summary, seed_summary, rob_seed, adv_df, out_dir):
    print("\nBuilding figures...")
    fig_grouped_bar_accuracy(summary, out_dir)
    fig_equivariance_gap(rob_seed, out_dir)
    fig_heatmap(summary, out_dir)
    fig_confidence_box(df, out_dir)
    fig_radar(summary, out_dir)
    fig_data_efficiency(summary, out_dir)
    fig_advantage_bar(adv_df, out_dir)
    fig_confidence_gap_scatter(summary, out_dir)
    fig_per_class_heatmap(df, out_dir)
    fig_seed_consistency(seed_summary, out_dir)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Directory containing the result CSVs")
    parser.add_argument("--out-dir", default="./results", help="Directory to write tables/ and figures/ into")
    args = parser.parse_args()

    print(f"Loading CSVs from {args.data_dir} ...")
    df = load_all(args.data_dir)
    print(f"Loaded {len(df):,} predictions from {df['source_file'].nunique()} files "
          f"({df['seed'].nunique()} seed(s): {sorted(df['seed'].unique())}).")

    report_missing_combos(df, args.out_dir)

    summary, seed_summary = compute_summary(df)
    rob, rob_seed, adv_df = make_tables(summary, seed_summary, args.out_dir)
    make_figures(df, summary, seed_summary, rob_seed, adv_df, args.out_dir)

    print(f"\nDone. Tables in {Path(args.out_dir) / 'tables'}, figures in {Path(args.out_dir) / 'figures'}")


if __name__ == "__main__":
    main()
