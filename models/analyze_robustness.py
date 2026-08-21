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

where <condition> in {clean, unseen-all, unseen-large_rotation}, and an
optional "strong_" prefix indicates strong augmentation was applied to the
test-time transformation itself (i.e. a harsher version of the shift).

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

TRAIN_AUG_LABELS = {"none": "No Train-Time Aug", "strong": "Strong Train-Time Aug"}

# Base transformation (ignoring whether the *test-time* application was "strong")
TRANSFORM_LABELS = {
    "clean": "Clean (no shift)",
    "unseen-large_rotation": "Unseen: Large Rotation",
    "unseen-all": "Unseen: All Transforms",
}
CONDITION_ORDER = ["clean", "unseen-large_rotation", "unseen-all"]

REQUIRED_COLS = {"sample_idx", "true_label", "pred_label", "correct", "confidence"}

# ----------------------------------------------------------------------------
# PARSING & LOADING
# ----------------------------------------------------------------------------

def build_pattern():
    model_pat = "|".join(re.escape(m) for m in sorted(MODEL_NAMES, key=len, reverse=True))
    db_pat = "|".join(re.escape(d) for d in sorted(DATABASES, key=len, reverse=True))
    return re.compile(
        rf"^(?P<model>{model_pat})_(?P<database>{db_pat})_"
        rf"(?P<train_aug>none|strong)_(?P<train_amount>[0-9.]+)_(?P<seed>[0-9]+)_(?P<suffix>.+)\.csv$"
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
    # human-readable labels
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["transformation_label"] = df["transformation"].map(TRANSFORM_LABELS).fillna(df["transformation"])
    return df


# ----------------------------------------------------------------------------
# SUMMARY STATS
# ----------------------------------------------------------------------------

GROUP_COLS = ["model", "train_aug", "train_amount", "test_aug", "transformation"]


def compute_summary(df):
    rows = []
    for keys, g in df.groupby(GROUP_COLS):
        model, train_aug, train_amount, test_aug, transformation = keys
        n = len(g)
        acc = g["correct"].mean()
        correct_conf = g.loc[g["correct"] == 1, "confidence"]
        incorrect_conf = g.loc[g["correct"] == 0, "confidence"]
        rows.append({
            "model": model,
            "train_aug": train_aug,
            "train_amount": train_amount,
            "test_aug": test_aug,
            "transformation": transformation,
            "n": n,
            "accuracy": acc,
            "mean_confidence": g["confidence"].mean(),
            "mean_confidence_correct": correct_conf.mean() if len(correct_conf) else np.nan,
            "mean_confidence_incorrect": incorrect_conf.mean() if len(incorrect_conf) else np.nan,
            "confidence_gap": (correct_conf.mean() - incorrect_conf.mean())
                                if len(correct_conf) and len(incorrect_conf) else np.nan,
            "confidence_std": g["confidence"].std(),
        })
    summary = pd.DataFrame(rows)
    summary["model_label"] = summary["model"].map(MODEL_LABELS).fillna(summary["model"])
    summary["transformation_label"] = summary["transformation"].map(TRANSFORM_LABELS).fillna(summary["transformation"])
    return summary


def add_robustness_metrics(summary):
    """For each (model, train_aug, train_amount, test_aug), express accuracy
    relative to that same config's 'clean' accuracy -> robustness retained (%)
    and absolute/relative equivariance gap."""
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


def make_tables(summary, out_dir):
    print("\nBuilding tables...")

    # 1. Master results table (every condition, every config)
    master = summary.sort_values(["model", "train_aug", "train_amount", "test_aug", "transformation"])
    cols = ["model_label", "train_aug", "train_amount", "test_aug", "transformation_label",
            "n", "accuracy", "mean_confidence_correct", "mean_confidence_incorrect", "confidence_gap"]
    save_table(master[cols].round(4), out_dir, "01_master_results")

    # 2. Accuracy pivot: rows = model x train setting, cols = test condition
    #    (use only non-"strong test-time" conditions for the primary comparison)
    base = summary[summary["test_aug"] == "none"].copy()
    base["train_setting"] = base["model_label"] + " | train_aug=" + base["train_aug"] + \
                             " | frac=" + base["train_amount"].astype(str)
    pivot_acc = base.pivot_table(index="train_setting", columns="transformation_label",
                                  values="accuracy")
    pivot_acc = pivot_acc.reindex(columns=[TRANSFORM_LABELS[c] for c in CONDITION_ORDER if TRANSFORM_LABELS[c] in pivot_acc.columns])
    save_table(pivot_acc.round(4).reset_index(), out_dir, "02_accuracy_pivot_normal_test_aug", index=False)

    # 2b. Same, but for strong test-time augmentation of the transform
    strong = summary[summary["test_aug"] == "strong"].copy()
    if len(strong):
        strong["train_setting"] = strong["model_label"] + " | train_aug=" + strong["train_aug"] + \
                                   " | frac=" + strong["train_amount"].astype(str)
        pivot_acc_strong = strong.pivot_table(index="train_setting", columns="transformation_label",
                                               values="accuracy")
        save_table(pivot_acc_strong.round(4).reset_index(), out_dir, "02b_accuracy_pivot_strong_test_aug", index=False)

    # 3. Robustness / equivariance-gap table
    rob = add_robustness_metrics(summary)
    rob_cols = ["model_label", "train_aug", "train_amount", "test_aug", "transformation_label",
                "clean_accuracy", "accuracy", "accuracy_drop", "relative_drop_pct", "robustness_retained_pct"]
    rob_sorted = rob[rob["transformation"] != "clean"].sort_values(
        ["transformation", "test_aug", "model", "train_aug", "train_amount"]
    )
    save_table(rob_sorted[rob_cols].round(3), out_dir, "03_robustness_gap")

    # 4. Head-to-head "best model per condition" table
    best = summary.loc[summary.groupby(["train_aug", "train_amount", "test_aug", "transformation"])["accuracy"].idxmax()]
    save_table(best[["train_aug", "train_amount", "test_aug", "transformation_label", "model_label", "accuracy"]]
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
        best_other = others.loc[others["accuracy"].idxmax()]
        adv_rows.append({
            "train_aug": train_aug, "train_amount": train_amount, "test_aug": test_aug,
            "transformation": TRANSFORM_LABELS.get(transformation, transformation),
            "ecaps_accuracy": ecaps_acc,
            "best_baseline": best_other["model_label"],
            "best_baseline_accuracy": best_other["accuracy"],
            "ecaps_advantage_pp": 100 * (ecaps_acc - best_other["accuracy"]),
        })
    adv_df = pd.DataFrame(adv_rows).sort_values(["transformation", "train_aug", "train_amount"])
    save_table(adv_df.round(3), out_dir, "05_ecaps_advantage_over_best_baseline")

    return rob, adv_df


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
    """Grouped bars: x = transformation condition, groups = model, one panel per
    (train_aug, train_amount), test_aug fixed to 'none'."""
    base = summary[summary["test_aug"] == "none"]
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
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   label=MODEL_LABELS[model], color=MODEL_COLORS[model])
        ax.set_xticks(x)
        ax.set_xticklabels([TRANSFORM_LABELS[c] for c in conditions], rotation=15, ha="right")
        ax.set_title(f"Train Aug: {TRAIN_AUG_LABELS[combo['train_aug']]}\nTrain Data Fraction: {combo['train_amount']}")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Test Accuracy")
    axes[-1].legend(loc="upper right", frameon=True)
    fig.suptitle("Accuracy by Model Across Clean and Unseen-Transformation Test Conditions", y=1.05, fontsize=13)
    fig.tight_layout()
    savefig(fig, out_dir, "fig01_grouped_bar_accuracy_by_condition")


def fig_equivariance_gap(rob, out_dir):
    """Bar chart of accuracy drop (clean -> unseen) per model, faceted by transformation,
    for test_aug == 'none'."""
    sub = rob[(rob["test_aug"] == "none") & (rob["transformation"] != "clean")]
    transformations = [t for t in CONDITION_ORDER if t in sub["transformation"].unique() and t != "clean"]
    fig, axes = plt.subplots(1, len(transformations), figsize=(6 * len(transformations), 4.5), sharey=True)
    if len(transformations) == 1:
        axes = [axes]

    train_settings = sub[["train_aug", "train_amount"]].drop_duplicates().sort_values(["train_aug", "train_amount"])
    labels = [f"{TRAIN_AUG_LABELS[r.train_aug]}\nfrac={r.train_amount}" for r in train_settings.itertuples()]
    x = np.arange(len(train_settings))
    width = 0.8 / len(MODEL_ORDER)

    for ax, transformation in zip(axes, transformations):
        tsub = sub[sub["transformation"] == transformation]
        for i, model in enumerate(MODEL_ORDER):
            vals = []
            for r in train_settings.itertuples():
                row = tsub[(tsub["model"] == model) & (tsub["train_aug"] == r.train_aug) &
                           (tsub["train_amount"] == r.train_amount)]
                vals.append(100 * row["accuracy_drop"].iloc[0] if len(row) else np.nan)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   label=MODEL_LABELS[model], color=MODEL_COLORS[model])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(TRANSFORM_LABELS.get(transformation, transformation))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Accuracy Drop from Clean (pp)\n(lower = more equivariant / robust)")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Equivariance Gap: Accuracy Loss Under Unseen Transformations", y=1.03, fontsize=13)
    fig.tight_layout()
    savefig(fig, out_dir, "fig02_equivariance_gap_bars")


def fig_heatmap(summary, out_dir):
    """Heatmap of accuracy: rows = model x train setting, cols = condition."""
    base = summary[summary["test_aug"] == "none"].copy()
    base["row_label"] = base["model_label"] + " (" + base["train_aug"] + ", frac=" + base["train_amount"].astype(str) + ")"
    conditions = [c for c in CONDITION_ORDER if c in base["transformation"].unique()]
    pivot = base.pivot_table(index="row_label", columns="transformation", values="accuracy")
    pivot = pivot.reindex(columns=conditions)
    # order rows by model then train setting for readability
    order = []
    for model in MODEL_ORDER:
        rows = sorted([r for r in pivot.index if MODEL_LABELS[model] in r])
        order.extend(rows)
    pivot = pivot.reindex(order)

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
    fig.colorbar(im, ax=ax, label="Accuracy")
    ax.set_title("Accuracy Heatmap: Model x Training Setting x Test Condition")
    fig.tight_layout()
    savefig(fig, out_dir, "fig03_accuracy_heatmap")


def fig_confidence_box(df, out_dir):
    """Boxplots of confidence, split by correct/incorrect, per model, for clean vs
    the two unseen conditions (test_aug == 'none', train_aug == 'none', frac == 1
    as a representative slice -- clearly labeled)."""
    sub = df[(df["test_aug"] == "none") & (df["train_aug"] == "none") & (df["train_amount"] == df["train_amount"].max())]
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
                    colors.append(MODEL_COLORS[model] if correctness == 1 else "#B0B0B0")
                    ticklabels.append(f"{MODEL_LABELS[model]}\n{tag}")
                pos += 1
            pos += 0.6
        bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(ticklabels, rotation=45, ha="right", fontsize=8)
        ax.set_title(TRANSFORM_LABELS.get(cond, cond))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Raw Confidence Score")
    fig.suptitle("Confidence Distributions (Correct vs Incorrect Predictions)\n"
                  "[Slice shown: no train-time aug, full training data, normal test-time aug]",
                  y=1.06, fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig04_confidence_boxplots")


def fig_radar(summary, out_dir):
    """Radar/spider chart: one axis per test condition, one polygon per model.
    Uses test_aug == 'none', and averages across train_aug/train_amount for a
    high-level overview panel per train setting."""
    base = summary[summary["test_aug"] == "none"]
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
            ax.plot(angles, vals, label=MODEL_LABELS[model], color=MODEL_COLORS[model], linewidth=2)
            ax.fill(angles, vals, color=MODEL_COLORS[model], alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([TRANSFORM_LABELS[c] for c in conditions], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{TRAIN_AUG_LABELS[combo['train_aug']]}, frac={combo['train_amount']}", fontsize=11, y=1.1)
    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.suptitle("Robustness Profile by Model", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "fig05_radar_robustness_profile")


def fig_data_efficiency(summary, out_dir):
    """Line plot: x = train_amount, y = accuracy, one line per model, one panel
    per (test condition), for train_aug == 'none' vs 'strong' as linestyle."""
    base = summary[summary["test_aug"] == "none"]
    conditions = [c for c in CONDITION_ORDER if c in base["transformation"].unique()]
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4.5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        csub = base[base["transformation"] == cond]
        for model in MODEL_ORDER:
            for train_aug, ls, marker in [("none", "--", "o"), ("strong", "-", "s")]:
                msub = csub[(csub["model"] == model) & (csub["train_aug"] == train_aug)].sort_values("train_amount")
                if len(msub):
                    ax.plot(msub["train_amount"], msub["accuracy"], linestyle=ls, marker=marker,
                             color=MODEL_COLORS[model],
                             label=f"{MODEL_LABELS[model]} ({TRAIN_AUG_LABELS[train_aug]})")
        ax.set_title(TRANSFORM_LABELS.get(cond, cond))
        ax.set_xlabel("Fraction of Training Data")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Test Accuracy")
    axes[-1].legend(loc="lower right", fontsize=7)
    fig.suptitle("Data Efficiency: Accuracy vs. Amount of Training Data", y=1.03, fontsize=13)
    fig.tight_layout()
    savefig(fig, out_dir, "fig06_data_efficiency_lines")


def fig_advantage_bar(adv_df, out_dir):
    """Bar chart of Efficient-CapsNet's accuracy advantage (pp) over the best
    baseline, per condition and training setting."""
    if adv_df.empty:
        return
    adv_df = adv_df.copy()
    adv_df["setting"] = adv_df["transformation"] + " | " + adv_df["train_aug"] + \
                         " | frac=" + adv_df["train_amount"].astype(str) + \
                         " | test_aug=" + adv_df["test_aug"]
    adv_df = adv_df.sort_values("ecaps_advantage_pp")

    fig, ax = plt.subplots(figsize=(9, 0.35 * len(adv_df) + 2))
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in adv_df["ecaps_advantage_pp"]]
    ax.barh(adv_df["setting"], adv_df["ecaps_advantage_pp"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Efficient-CapsNet Accuracy Advantage over Best Baseline (pp)")
    ax.set_title("Where Does Efficient-CapsNet Retain (or Lose) Its Edge?")
    ax.grid(axis="x", alpha=0.3)
    legend_elems = [Patch(facecolor="#55A868", label="CapsNet ahead"),
                    Patch(facecolor="#C44E52", label="CapsNet behind")]
    ax.legend(handles=legend_elems, loc="lower right")
    fig.tight_layout()
    savefig(fig, out_dir, "fig07_ecaps_advantage_diverging_bar")


def fig_confidence_gap_scatter(summary, out_dir):
    """Scatter: x = accuracy, y = confidence_gap (correct-mean minus incorrect-mean),
    colored by model, marker shape by condition. A well-calibrated / trustworthy
    model should sit in the upper-right (high accuracy AND a large confidence
    gap even under shift)."""
    base = summary[summary["test_aug"] == "none"]
    markers = {"clean": "o", "unseen-large_rotation": "^", "unseen-all": "s"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for model in MODEL_ORDER:
        msub = base[base["model"] == model]
        for cond, marker in markers.items():
            csub = msub[msub["transformation"] == cond]
            if len(csub):
                ax.scatter(csub["accuracy"], csub["confidence_gap"], color=MODEL_COLORS[model],
                           marker=marker, s=90, edgecolor="black", linewidth=0.5,
                           label=f"{MODEL_LABELS[model]} - {TRANSFORM_LABELS.get(cond, cond)}")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Confidence Gap (mean confidence: correct - incorrect)")
    ax.set_title("Accuracy vs. Confidence Separation\n(Does confidence stay informative under shift?)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=1)
    fig.tight_layout()
    savefig(fig, out_dir, "fig08_confidence_gap_vs_accuracy_scatter")


def fig_per_class_heatmap(df, out_dir):
    """Per-class accuracy heatmap for the toughest condition (unseen-all),
    test_aug='none', comparing models at full training data / no train aug."""
    sub = df[(df["test_aug"] == "none") & (df["train_aug"] == "none") &
             (df["train_amount"] == df["train_amount"].max()) &
             (df["transformation"] == "unseen-all")]
    if sub.empty:
        return
    pivot = sub.groupby(["model", "true_label"])["correct"].mean().unstack("true_label")
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
    fig, ax = plt.subplots(figsize=(1.0 * pivot.shape[1] + 3, 0.6 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([MODEL_LABELS[m] for m in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="black" if 0.3 < val < 0.8 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Per-Class Accuracy")
    ax.set_xlabel("True Class Label")
    ax.set_title("Per-Class Accuracy Under Unseen-All Shift\n[no train aug, full training data]")
    fig.tight_layout()
    savefig(fig, out_dir, "fig09_per_class_accuracy_heatmap")


def make_figures(df, summary, rob, adv_df, out_dir):
    print("\nBuilding figures...")
    fig_grouped_bar_accuracy(summary, out_dir)
    fig_equivariance_gap(rob, out_dir)
    fig_heatmap(summary, out_dir)
    fig_confidence_box(df, out_dir)
    fig_radar(summary, out_dir)
    fig_data_efficiency(summary, out_dir)
    fig_advantage_bar(adv_df, out_dir)
    fig_confidence_gap_scatter(summary, out_dir)
    fig_per_class_heatmap(df, out_dir)


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
    print(f"Loaded {len(df):,} predictions from {df['source_file'].nunique()} files.")

    summary = compute_summary(df)
    rob, adv_df = make_tables(summary, args.out_dir)
    make_figures(df, summary, rob, adv_df, args.out_dir)

    print(f"\nDone. Tables in {Path(args.out_dir) / 'tables'}, figures in {Path(args.out_dir) / 'figures'}")


if __name__ == "__main__":
    main()
