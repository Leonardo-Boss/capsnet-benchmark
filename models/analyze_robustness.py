#!/usr/bin/env python3
"""
analyze_robustness.py  (v2)

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
      -> model=ecaps, database=cifar_10, train_aug=standard, train_amount=0.66,
         seed=2, test_aug=none, transformation=unseen-all

where <condition> in {clean, unseen-all, unseen-large_rotation}, and an
optional "strong_" prefix indicates strong augmentation was applied to the
test-time transformation itself (a harsher version of the shift).

Each CSV has columns: sample_idx,true_label,pred_label,correct,confidence
(confidence is a raw, unbounded score -- NOT a softmax probability, and NOT
comparable in scale between architectures; see the AUROC figure).

WHAT CHANGED IN v2
------------------
Every number is now interpreted against a reference level (chance and the
majority-class baseline), because on an imbalanced binary dataset an
"accuracy of 0.59" is not a result, it is the trivial classifier. Metrics
that are confounded by clean accuracy (retention %) are still computed but
are now plotted *against* clean accuracy so the confound is visible rather
than hidden. Raw confidence boxplots were replaced with a scale-free error
detection AUROC, and a prediction-collapse diagnostic was added to catch
models that "survive" a shift only by degenerating to a constant prediction.
Figures were rebuilt for legibility: no overlapping tick labels, no 9-line
spaghetti panels, no 3-axis radar charts, no 45-row bar charts.

OUTPUTS
-------
<out-dir>/tables/*.csv, *.md   -- all summary tables
<out-dir>/figures/*.png        -- all figures (200 dpi)
<out-dir>/FINDINGS.md          -- auto-generated headline findings
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None

# ----------------------------------------------------------------------------
# CONFIG -- edit these if your naming vocabulary differs
# ----------------------------------------------------------------------------

MODEL_NAMES = ["deit_tiny", "ecaps", "resnet18"]
DATABASES = ["cifar_10", "flame"]

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
MODEL_MARKERS = {"resnet18": "o", "deit_tiny": "s", "ecaps": "D"}

DATABASE_LABELS = {"cifar_10": "CIFAR-10", "flame": "FLAME"}
CLASS_NAMES = {
    "cifar_10": ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"],
    "flame": ["Fire", "No_Fire"],
}

TRAIN_AUG_LABELS = {"none": "No aug", "standard": "Standard aug", "strong": "Strong aug"}
TRAIN_AUG_ORDER = ["none", "standard", "strong"]
TRAIN_AUG_MARKERS = {"none": "o", "standard": "s", "strong": "^"}

TRANSFORM_LABELS = {
    "clean": "Clean",
    "unseen-large_rotation": "Large rotation",
    "unseen-all": "All transforms",
}
CONDITION_ORDER = ["clean", "unseen-large_rotation", "unseen-all"]
SHIFT_ORDER = ["unseen-large_rotation", "unseen-all"]

TEST_AUG_LABELS = {"none": "normal", "strong": "strong"}

REQUIRED_COLS = {"sample_idx", "true_label", "pred_label", "correct", "confidence"}

GROUP_COLS = ["database", "model", "train_aug", "train_amount", "test_aug", "transformation"]
SEED_GROUP_COLS = GROUP_COLS + ["seed"]
CONFIG_COLS = ["database", "model", "train_aug", "train_amount"]

# Global plot style -- one place to change, so every figure stays consistent.
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def mlabel(m):
    return MODEL_LABELS.get(m, m)


def dblabel(d):
    return DATABASE_LABELS.get(d, d)


def tlabel(t):
    return TRANSFORM_LABELS.get(t, t)


def setting_label(train_aug, train_amount):
    return f"{TRAIN_AUG_LABELS.get(train_aug, train_aug)}, {int(round(100 * train_amount))}% data"


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
        test_aug, transformation = "strong", suffix[len("strong_"):]
    else:
        test_aug, transformation = "none", suffix
    d["test_aug"] = test_aug
    d["transformation"] = transformation
    d["train_amount"] = float(d["train_amount"])
    d["seed"] = int(d["seed"])
    return d


def load_all(data_dir):
    pattern = build_pattern()
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames, skipped = [], []
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
        for s in skipped[:10]:
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    if not frames:
        raise RuntimeError("No valid CSV files found/parsed in data_dir.")

    df = pd.concat(frames, ignore_index=True)
    df["correct"] = df["correct"].astype(int)
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["database_label"] = df["database"].map(DATABASE_LABELS).fillna(df["database"])
    df["transformation_label"] = df["transformation"].map(TRANSFORM_LABELS).fillna(df["transformation"])
    return df


def report_missing_combos(df, out_dir):
    """Print (and save) which expected combos are absent or short on seeds, so an
    in-progress sweep is visible rather than silently ignored."""
    databases = sorted(df["database"].unique())
    models = sorted(df["model"].unique())
    train_augs = sorted(df["train_aug"].unique(),
                        key=lambda x: TRAIN_AUG_ORDER.index(x) if x in TRAIN_AUG_ORDER else 99)
    train_amounts = sorted(df["train_amount"].unique())
    test_augs = sorted(df["test_aug"].unique())
    transformations = sorted(df["transformation"].unique(),
                             key=lambda x: CONDITION_ORDER.index(x) if x in CONDITION_ORDER else 99)
    max_seeds = int(df.groupby(GROUP_COLS)["seed"].nunique().max())

    present = df.groupby(GROUP_COLS)["seed"].nunique().reset_index()
    present_keys = {tuple(r[c] for c in GROUP_COLS): r["seed"] for _, r in present.iterrows()}

    rows = []
    for combo in product(databases, models, train_augs, train_amounts, test_augs, transformations):
        if combo[4] == "strong" and combo[5] == "clean":
            continue  # nonsensical: no strong test-time version of "clean"
        n_seeds = present_keys.get(combo, 0)
        status = "OK" if n_seeds == max_seeds else ("MISSING" if n_seeds == 0 else "PARTIAL")
        if status != "OK":
            rows.append(dict(zip(GROUP_COLS, combo)) |
                        {"seeds_found": n_seeds, "seeds_expected": max_seeds, "status": status})
    if rows:
        missing_df = pd.DataFrame(rows)
        print(f"\nSweep completeness (expecting {max_seeds} seed(s) per combo):")
        print(f"  {len(missing_df)} combo(s) incomplete or missing -> "
              f"tables/00_incomplete_sweep_combos.csv")
        for _, r in missing_df.head(10).iterrows():
            print(f"    [{r['status']:7s}] {r['database']}/{r['model']}, aug={r['train_aug']}, "
                  f"frac={r['train_amount']}, test_aug={r['test_aug']}, {r['transformation']} "
                  f"({r['seeds_found']}/{r['seeds_expected']})")
        if len(missing_df) > 10:
            print(f"    ... and {len(missing_df) - 10} more")
        tdir = Path(out_dir) / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        missing_df.to_csv(tdir / "00_incomplete_sweep_combos.csv", index=False)
    else:
        print("\nSweep completeness: all expected combos present with a full seed count.")


# ----------------------------------------------------------------------------
# METRICS (implemented directly so sklearn is not a hard dependency)
# ----------------------------------------------------------------------------

def macro_f1_score(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def balanced_accuracy_score(y_true, y_pred, classes):
    recalls = []
    for c in classes:
        support = np.sum(y_true == c)
        if support == 0:
            continue
        recalls.append(np.sum((y_pred == c) & (y_true == c)) / support)
    return float(np.mean(recalls)) if recalls else np.nan


def auroc(scores, positives):
    """AUROC via the rank / Mann-Whitney identity. `positives` is a boolean mask.

    Used here for *error detection*: can the model's own confidence score tell
    its correct predictions from its incorrect ones? This is invariant to any
    monotonic rescaling of the score, which matters a lot because a capsule
    length in [0,1] and a raw logit in [0,12] are not remotely comparable."""
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=bool)
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def normalized_prediction_entropy(y_pred, classes):
    """0 = all predictions on one class, 1 = uniform over classes. A model whose
    accuracy 'survives' a shift while this collapses toward 0 has not stayed
    robust; it has stopped discriminating and is riding the class prior."""
    k = len(classes)
    if k < 2:
        return np.nan
    counts = np.array([np.sum(y_pred == c) for c in classes], dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(k))


def dataset_reference_levels(df):
    """Chance level and majority-class baseline per database, from the ground
    truth of the test set. Everything downstream is judged against these."""
    rows = []
    for db, g in df.groupby("database"):
        one = g[g["source_file"] == g["source_file"].iloc[0]]
        classes = np.sort(one["true_label"].unique())
        counts = one["true_label"].value_counts(normalize=True).sort_index()
        rows.append({
            "database": db,
            "database_label": dblabel(db),
            "n_classes": len(classes),
            "n_test": len(one),
            "chance_level": 1.0 / len(classes),
            "majority_baseline": float(counts.max()),
            "majority_class": int(counts.idxmax()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# SUMMARY STATS (seed-level, then aggregated across seeds)
# ----------------------------------------------------------------------------

def compute_seed_level_summary(df):
    """One row per (database, model, train_aug, train_amount, test_aug,
    transformation, seed)."""
    rows = []
    for keys, g in df.groupby(SEED_GROUP_COLS):
        rec = dict(zip(SEED_GROUP_COLS, keys))
        y_true = g["true_label"].values
        y_pred = g["pred_label"].values
        classes = np.sort(np.unique(np.concatenate([y_true, y_pred])))
        correct_conf = g.loc[g["correct"] == 1, "confidence"]
        incorrect_conf = g.loc[g["correct"] == 0, "confidence"]
        rec.update({
            "n": len(g),
            "accuracy": g["correct"].mean(),
            "macro_f1": macro_f1_score(y_true, y_pred, classes),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred, classes),
            "mean_confidence": g["confidence"].mean(),
            "mean_confidence_correct": correct_conf.mean() if len(correct_conf) else np.nan,
            "mean_confidence_incorrect": incorrect_conf.mean() if len(incorrect_conf) else np.nan,
            "confidence_gap": (correct_conf.mean() - incorrect_conf.mean())
                              if len(correct_conf) and len(incorrect_conf) else np.nan,
            "error_detection_auroc": auroc(g["confidence"].values, g["correct"].values == 1),
            "pred_entropy_norm": normalized_prediction_entropy(y_pred, classes),
            "top_pred_share": float(pd.Series(y_pred).value_counts(normalize=True).max()),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_across_seeds(seed_summary):
    """Collapse the seed dimension. Mean is the headline number; _std columns
    are carried alongside for error bars and honesty."""
    mean_cols = ["accuracy", "macro_f1", "balanced_accuracy", "mean_confidence",
                 "mean_confidence_correct", "mean_confidence_incorrect", "confidence_gap",
                 "error_detection_auroc", "pred_entropy_norm", "top_pred_share"]
    agg_spec = {"n_seeds": ("seed", "nunique"), "n": ("n", "sum")}
    for c in mean_cols:
        agg_spec[c] = (c, "mean")
    for c in ["accuracy", "macro_f1", "balanced_accuracy", "confidence_gap", "error_detection_auroc"]:
        agg_spec[c + "_std"] = (c, "std")
    agg = seed_summary.groupby(GROUP_COLS).agg(**agg_spec).reset_index()
    for c in agg.columns:
        if c.endswith("_std"):
            agg[c] = agg[c].fillna(0.0)
    agg["model_label"] = agg["model"].map(MODEL_LABELS).fillna(agg["model"])
    agg["database_label"] = agg["database"].map(DATABASE_LABELS).fillna(agg["database"])
    agg["transformation_label"] = agg["transformation"].map(TRANSFORM_LABELS).fillna(agg["transformation"])
    return agg


def compute_summary(df):
    seed_summary = compute_seed_level_summary(df)
    summary = aggregate_across_seeds(seed_summary)
    return summary, seed_summary


def add_robustness_metrics(summary, refs):
    """Express shifted accuracy relative to the same config's clean accuracy.

    Three flavours, deliberately kept side by side because they disagree:
      accuracy_drop           absolute pp lost (the honest headline)
      robustness_retained_pct acc_shift / acc_clean -- systematically flatters
                              weak models, so never report it alone
      skill_retained_pct      (acc_shift - chance) / (acc_clean - chance),
                              i.e. how much *above-chance skill* survives"""
    # 'clean' is only ever evaluated at test_aug='none', so the reference has to
    # be looked up per training config and reused for the strong-test-aug rows.
    # Grouping on test_aug (as the previous version did) left every strong row
    # with a NaN clean_accuracy and therefore a NaN drop -- half the table.
    clean_ref = summary[summary["transformation"] == "clean"].set_index(
        ["database", "model", "train_aug", "train_amount"])[["accuracy", "macro_f1"]]
    out = []
    for keys, g in summary.groupby(["database", "model", "train_aug", "train_amount", "test_aug"]):
        cfg = keys[:4]
        if cfg not in clean_ref.index:
            continue
        clean_acc = clean_ref.loc[cfg, "accuracy"]
        clean_f1 = clean_ref.loc[cfg, "macro_f1"]
        g = g.copy()
        g["clean_accuracy"] = clean_acc
        g["clean_macro_f1"] = clean_f1
        g["accuracy_drop"] = clean_acc - g["accuracy"]
        g["relative_drop_pct"] = np.where(clean_acc > 0, 100 * (clean_acc - g["accuracy"]) / clean_acc, np.nan)
        g["robustness_retained_pct"] = np.where(clean_acc > 0, 100 * g["accuracy"] / clean_acc, np.nan)
        g["macro_f1_retained_pct"] = np.where(clean_f1 > 0, 100 * g["macro_f1"] / clean_f1, np.nan)
        out.append(g)
    rob = pd.concat(out, ignore_index=True)
    rob = rob.merge(refs[["database", "chance_level", "majority_baseline"]], on="database", how="left")
    denom = rob["clean_accuracy"] - rob["chance_level"]
    rob["skill_retained_pct"] = np.where(denom > 0.02, 100 * (rob["accuracy"] - rob["chance_level"]) / denom, np.nan)
    rob["above_chance_pp"] = 100 * (rob["accuracy"] - rob["chance_level"])
    rob["above_majority_pp"] = 100 * (rob["accuracy"] - rob["majority_baseline"])
    return rob


def add_robustness_metrics_seed_level(seed_summary):
    """Paired-by-seed drop: for each seed, clean minus shifted, THEN averaged.
    This is the statistically correct way to get an error bar on the gap."""
    clean_ref = seed_summary[seed_summary["transformation"] == "clean"].set_index(
        ["database", "model", "train_aug", "train_amount", "seed"])[["accuracy", "macro_f1"]]
    out = []
    for keys, g in seed_summary.groupby(["database", "model", "train_aug", "train_amount", "test_aug", "seed"]):
        cfg = keys[:4] + (keys[5],)  # same seed's own clean run, whatever the test_aug
        if cfg not in clean_ref.index:
            continue
        g = g.copy()
        g["clean_accuracy"] = clean_ref.loc[cfg, "accuracy"]
        g["clean_macro_f1"] = clean_ref.loc[cfg, "macro_f1"]
        g["accuracy_drop"] = g["clean_accuracy"] - g["accuracy"]
        g["macro_f1_drop"] = g["clean_macro_f1"] - g["macro_f1"]
        out.append(g)
    seed_rob = pd.concat(out, ignore_index=True)
    agg = seed_rob[seed_rob["transformation"] != "clean"].groupby(
        ["database", "model", "train_aug", "train_amount", "test_aug", "transformation"]
    ).agg(
        n_seeds=("seed", "nunique"),
        accuracy_drop=("accuracy_drop", "mean"),
        accuracy_drop_std=("accuracy_drop", "std"),
        macro_f1_drop=("macro_f1_drop", "mean"),
        macro_f1_drop_std=("macro_f1_drop", "std"),
    ).reset_index()
    for c in ["accuracy_drop_std", "macro_f1_drop_std"]:
        agg[c] = agg[c].fillna(0.0)
    agg["model_label"] = agg["model"].map(MODEL_LABELS).fillna(agg["model"])
    agg["database_label"] = agg["database"].map(DATABASE_LABELS).fillna(agg["database"])
    agg["transformation_label"] = agg["transformation"].map(TRANSFORM_LABELS).fillna(agg["transformation"])
    return agg


# ----------------------------------------------------------------------------
# TABLES
# ----------------------------------------------------------------------------

def save_table(df, out_dir, name, index=False):
    tdir = Path(out_dir) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tdir / f"{name}.csv", index=index)
    try:
        with open(tdir / f"{name}.md", "w") as f:
            f.write(df.to_markdown(index=index))
    except Exception:
        pass
    print(f"  wrote {name}.csv")


def paired_seed_test(a_vals, b_vals):
    """Paired comparison across seeds: mean difference, Cohen's dz, and a
    two-sided p-value. With 3 seeds this is badly underpowered -- it is
    reported so the reader can see that, not to manufacture significance."""
    a, b = np.asarray(a_vals, float), np.asarray(b_vals, float)
    n = min(len(a), len(b))
    if n < 2:
        return np.nan, np.nan, np.nan, n, np.nan
    d = a[:n] - b[:n]
    sd = float(d.std(ddof=1))
    dz = float(d.mean() / sd) if sd > 0 else np.nan
    if _scipy_stats is not None and sd > 0:
        p = float(_scipy_stats.ttest_rel(a[:n], b[:n]).pvalue)
    else:
        p = np.nan
    return float(d.mean()), sd, dz, n, p


def make_tables(df, summary, seed_summary, refs, out_dir):
    print("\nBuilding tables...")

    save_table(refs.round(4), out_dir, "00_dataset_summary")

    master_cols = ["database_label", "model_label", "train_aug", "train_amount", "test_aug",
                   "transformation_label", "n_seeds", "n", "accuracy", "accuracy_std",
                   "macro_f1", "macro_f1_std", "balanced_accuracy", "balanced_accuracy_std",
                   "error_detection_auroc", "error_detection_auroc_std",
                   "pred_entropy_norm", "top_pred_share",
                   "mean_confidence_correct", "mean_confidence_incorrect", "confidence_gap"]
    master = summary.sort_values(GROUP_COLS)
    save_table(master[master_cols].round(4), out_dir, "01_master_results")

    seed_cols = ["database", "model", "train_aug", "train_amount", "test_aug", "transformation",
                 "seed", "n", "accuracy", "macro_f1", "balanced_accuracy",
                 "error_detection_auroc", "top_pred_share", "confidence_gap"]
    save_table(seed_summary[seed_cols].sort_values(SEED_GROUP_COLS).round(4),
               out_dir, "01b_per_seed_results")

    # 2. Headline table: accuracy with the reference levels attached, so a cell
    #    can never be read without knowing what "no skill" would have scored.
    base = summary[summary["test_aug"] == "none"].merge(
        refs[["database", "chance_level", "majority_baseline"]], on="database", how="left")
    base["train_setting"] = (base["database_label"] + " | " + base["model_label"] + " | " +
                             base["train_aug"] + " | frac=" + base["train_amount"].astype(str))
    base["cell"] = base.apply(
        lambda r: f"{r['accuracy']:.3f} ± {r['accuracy_std']:.3f}"
                  f"{' *' if r['accuracy'] <= r['majority_baseline'] else ''}", axis=1)
    pivot = base.pivot_table(index="train_setting", columns="transformation_label",
                             values="cell", aggfunc="first")
    pivot = pivot.reindex(columns=[tlabel(c) for c in CONDITION_ORDER if tlabel(c) in pivot.columns])
    save_table(pivot.reset_index(), out_dir, "02_accuracy_pivot_normal_test_aug")

    b2 = summary[summary["test_aug"] == "strong"]
    if len(b2):
        b2 = b2.copy()
        b2["train_setting"] = (b2["database_label"] + " | " + b2["model_label"] + " | " +
                               b2["train_aug"] + " | frac=" + b2["train_amount"].astype(str))
        b2["cell"] = b2.apply(lambda r: f"{r['accuracy']:.3f} ± {r['accuracy_std']:.3f}", axis=1)
        save_table(b2.pivot_table(index="train_setting", columns="transformation_label",
                                  values="cell", aggfunc="first").reset_index(),
                   out_dir, "02b_accuracy_pivot_strong_test_aug")

    mf1 = summary[summary["test_aug"] == "none"].copy()
    mf1["train_setting"] = (mf1["database_label"] + " | " + mf1["model_label"] + " | " +
                            mf1["train_aug"] + " | frac=" + mf1["train_amount"].astype(str))
    mf1["cell"] = mf1.apply(lambda r: f"{r['macro_f1']:.3f} ± {r['macro_f1_std']:.3f}", axis=1)
    save_table(mf1.pivot_table(index="train_setting", columns="transformation_label",
                               values="cell", aggfunc="first").reset_index(),
               out_dir, "02c_macro_f1_pivot_normal_test_aug")

    # 3. Robustness table, now with skill-retention alongside naive retention.
    rob = add_robustness_metrics(summary, refs)
    rob_cols = ["database_label", "model_label", "train_aug", "train_amount", "test_aug",
                "transformation_label", "clean_accuracy", "accuracy", "accuracy_drop",
                "relative_drop_pct", "robustness_retained_pct", "skill_retained_pct",
                "above_chance_pp", "above_majority_pp", "macro_f1_retained_pct"]
    save_table(rob[rob["transformation"] != "clean"][rob_cols]
               .sort_values(["database_label", "transformation_label", "test_aug",
                             "model_label", "train_aug", "train_amount"]).round(3),
               out_dir, "03_robustness_gap")

    rob_seed = add_robustness_metrics_seed_level(seed_summary)
    save_table(rob_seed.sort_values(["database", "transformation", "test_aug", "model",
                                     "train_aug", "train_amount"]).round(4),
               out_dir, "03b_robustness_gap_paired_by_seed")

    # 4. Best model per condition, with the runner-up and the margin, so a "win"
    #    that is smaller than the seed noise is visible as such.
    best_rows = []
    for keys, g in summary.groupby(["database", "train_aug", "train_amount", "test_aug", "transformation"]):
        g = g.sort_values("accuracy", ascending=False)
        top, second = g.iloc[0], (g.iloc[1] if len(g) > 1 else None)
        margin = top["accuracy"] - second["accuracy"] if second is not None else np.nan
        pooled = np.sqrt(top["accuracy_std"] ** 2 + (second["accuracy_std"] ** 2 if second is not None else 0))
        best_rows.append(dict(zip(["database", "train_aug", "train_amount", "test_aug", "transformation"], keys)) | {
            "winner": top["model_label"], "accuracy": top["accuracy"], "accuracy_std": top["accuracy_std"],
            "runner_up": second["model_label"] if second is not None else None,
            "runner_up_accuracy": second["accuracy"] if second is not None else np.nan,
            "margin_pp": 100 * margin,
            "margin_exceeds_seed_noise": bool(margin > 2 * pooled) if second is not None else False,
        })
    save_table(pd.DataFrame(best_rows).round(4), out_dir, "04_best_model_per_condition")

    # 5. CapsNet advantage, paired by seed against EACH baseline (not just the
    #    best one -- "best of the rest" is a biased comparator).
    adv_rows = []
    for keys, g in seed_summary.groupby(["database", "train_aug", "train_amount", "test_aug", "transformation"]):
        e = g[g["model"] == "ecaps"].sort_values("seed")
        if e.empty:
            continue
        for other in [m for m in MODEL_ORDER if m != "ecaps"]:
            o = g[g["model"] == other].sort_values("seed")
            if o.empty:
                continue
            res = paired_seed_test(e["accuracy"].values, o["accuracy"].values)
            mean_d, sd, dz, n, p = res
            adv_rows.append(dict(zip(["database", "train_aug", "train_amount", "test_aug", "transformation"], keys)) | {
                "baseline": mlabel(other),
                "ecaps_accuracy": e["accuracy"].mean(),
                "baseline_accuracy": o["accuracy"].mean(),
                "advantage_pp": 100 * mean_d,
                "advantage_pp_sd": 100 * sd,
                "cohens_dz": dz, "n_seeds": n, "p_value_paired_t": p,
            })
    adv_pairwise = pd.DataFrame(adv_rows)
    if len(adv_pairwise):
        save_table(adv_pairwise.round(4), out_dir, "05b_ecaps_vs_each_baseline_paired")

    # 5a. Kept for backwards compatibility: vs best baseline, seed-averaged.
    adv_rows = []
    for keys, g in summary.groupby(["database", "train_aug", "train_amount", "test_aug", "transformation"]):
        e = g[g["model"] == "ecaps"]
        others = g[g["model"] != "ecaps"]
        if e.empty or others.empty:
            continue
        e0 = e.iloc[0]
        best_other = others.loc[others["accuracy"].idxmax()]
        adv_rows.append(dict(zip(["database", "train_aug", "train_amount", "test_aug", "transformation"], keys)) | {
            "ecaps_accuracy": e0["accuracy"], "ecaps_accuracy_std": e0["accuracy_std"],
            "best_baseline": best_other["model_label"],
            "best_baseline_accuracy": best_other["accuracy"],
            "best_baseline_accuracy_std": best_other["accuracy_std"],
            "ecaps_advantage_pp": 100 * (e0["accuracy"] - best_other["accuracy"]),
            "ecaps_advantage_pp_std": 100 * np.sqrt(e0["accuracy_std"] ** 2 + best_other["accuracy_std"] ** 2),
            "ecaps_macro_f1_advantage_pp": 100 * (e0["macro_f1"] - best_other["macro_f1"]),
        })
    adv_df = pd.DataFrame(adv_rows)
    save_table(adv_df.round(3), out_dir, "05_ecaps_advantage_over_best_baseline")

    var = seed_summary.groupby(GROUP_COLS)["accuracy"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    var.columns = list(GROUP_COLS) + ["accuracy_mean", "accuracy_std", "accuracy_min", "accuracy_max", "n_seeds"]
    var["spread_pp"] = 100 * (var["accuracy_max"] - var["accuracy_min"])
    var["model_label"] = var["model"].map(MODEL_LABELS).fillna(var["model"])
    save_table(var.sort_values(GROUP_COLS).round(4), out_dir, "06_seed_variability")

    if summary["database"].nunique() > 1:
        cross = rob[rob["transformation"] != "clean"][
            ["database_label", "model_label", "train_aug", "train_amount", "transformation_label",
             "test_aug", "clean_accuracy", "accuracy", "robustness_retained_pct",
             "skill_retained_pct", "macro_f1_retained_pct"]]
        save_table(cross[cross["test_aug"] == "none"].round(3), out_dir, "07_cross_dataset_retention")

    # 8. Error-detection AUROC: the scale-free replacement for raw confidence.
    ed = summary[["database_label", "model_label", "train_aug", "train_amount", "test_aug",
                  "transformation_label", "error_detection_auroc", "error_detection_auroc_std",
                  "pred_entropy_norm", "top_pred_share"]]
    save_table(ed.sort_values(["database_label", "model_label", "transformation_label"]).round(4),
               out_dir, "08_error_detection_and_collapse")

    return rob, rob_seed, adv_df, adv_pairwise


# ----------------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------------

def savefig(fig, out_dir, name, db=None):
    fdir = Path(out_dir) / "figures"
    if db is not None:
        fdir = fdir / db
    fdir.mkdir(parents=True, exist_ok=True)
    path = fdir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(Path(out_dir))}")


def model_legend(models, title=None):
    return [Line2D([0], [0], color=MODEL_COLORS.get(m, "#888"), marker=MODEL_MARKERS.get(m, "o"),
                   markersize=6, linewidth=2, label=mlabel(m)) for m in models]


def present_models(sub):
    return [m for m in MODEL_ORDER if m in sub["model"].unique()]


def present_augs(sub):
    return [a for a in TRAIN_AUG_ORDER if a in sub["train_aug"].unique()]


def present_shifts(sub):
    return [t for t in SHIFT_ORDER if t in sub["transformation"].unique()]


def add_reference_lines(ax, chance, majority, axis="x", label=True):
    fn = ax.axvline if axis == "x" else ax.axhline
    fn(chance, color="#555", linestyle="--", linewidth=1, zorder=0)
    if majority is not None and abs(majority - chance) > 1e-6:
        fn(majority, color="#555", linestyle=":", linewidth=1.2, zorder=0)
    if label:
        txt = ax.text(0.99, 0.02, "-- chance   ⋯ majority class", transform=ax.transAxes,
                      ha="right", va="bottom", fontsize=6.5, color="#555")
        return txt


# --- fig01: clean -> shifted, as an arrow per configuration ------------------

def fig_shift_dumbbell(rob, refs, out_dir, db):
    """The headline figure. One horizontal arrow per configuration running from
    its clean accuracy to its accuracy under shift, drawn against the chance and
    majority-class reference lines. Absolute level and size of the drop are
    readable in a single glance, which no bar chart of either quantity alone
    manages."""
    sub = rob[(rob["database"] == db) & (rob["test_aug"] == "none") & (rob["transformation"] != "clean")]
    if sub.empty:
        return
    ref = refs[refs["database"] == db].iloc[0]
    shifts, augs = present_shifts(sub), present_augs(sub)
    models, fracs = present_models(sub), sorted(sub["train_amount"].unique())

    fig, axes = plt.subplots(len(shifts), len(augs), figsize=(4.1 * len(augs), 0.42 * len(models) * len(fracs) * len(shifts) + 2.2),
                             sharex=True, sharey=True, squeeze=False)
    ypos, ylabels = {}, []
    i = 0
    for frac in fracs:
        for m in models:
            ypos[(frac, m)] = i
            ylabels.append(f"{mlabel(m)}  ·  {int(round(100 * frac))}%")
            i += 1

    for r, shift in enumerate(shifts):
        for c, aug in enumerate(augs):
            ax = axes[r][c]
            s = sub[(sub["transformation"] == shift) & (sub["train_aug"] == aug)]
            for _, row in s.iterrows():
                y = ypos.get((row["train_amount"], row["model"]))
                if y is None:
                    continue
                col = MODEL_COLORS.get(row["model"], "#888")
                ax.plot([row["clean_accuracy"], row["accuracy"]], [y, y], color=col, linewidth=2, alpha=0.55, zorder=2)
                ax.scatter(row["clean_accuracy"], y, facecolor="white", edgecolor=col, s=42, linewidth=1.6, zorder=3)
                ax.scatter(row["accuracy"], y, color=col, s=48, zorder=4,
                           marker=MODEL_MARKERS.get(row["model"], "o"))
            ax.axvline(ref["chance_level"], color="#555", linestyle="--", linewidth=1, zorder=0)
            if abs(ref["majority_baseline"] - ref["chance_level"]) > 1e-6:
                ax.axvline(ref["majority_baseline"], color="#C44E52", linestyle=":", linewidth=1.3, zorder=0)
            for k in range(len(models), len(ylabels), len(models)):
                ax.axhline(k - 0.5, color="#DDD", linewidth=0.8)
            if r == 0:
                ax.set_title(TRAIN_AUG_LABELS.get(aug, aug))
            if c == 0:
                ax.set_ylabel(tlabel(shift), fontsize=10, fontweight="bold")
            if r == len(shifts) - 1:
                ax.set_xlabel("Accuracy")
    axes[0][0].set_yticks(range(len(ylabels)))
    axes[0][0].set_yticklabels(ylabels)
    axes[0][0].set_ylim(len(ylabels) - 0.4, -0.6)   # first data fraction at the top
    xlo = max(0.0, min(sub["accuracy"].min(), ref["chance_level"]) - 0.06)
    xhi = min(1.0, max(sub["clean_accuracy"].max(), ref["majority_baseline"]) + 0.06)
    axes[0][0].set_xlim(xlo, xhi)
    handles = model_legend(models) + [
        Line2D([0], [0], color="#888", marker="o", markerfacecolor="white", linestyle="",
               markersize=7, label="clean accuracy"),
        Line2D([0], [0], color="#888", marker="o", linestyle="", markersize=7, label="under shift"),
        Line2D([0], [0], color="#555", linestyle="--", label="chance"),
        Line2D([0], [0], color="#C44E52", linestyle=":", label="majority class")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{dblabel(db)}: how far does each model fall under an unseen transformation?\n"
                 f"arrow = clean accuracy → accuracy under shift, mean of {int(sub['n_seeds'].max() if 'n_seeds' in sub else 3)} seeds",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig01_shift_dumbbell", db)


# --- fig02: does clean accuracy buy robustness? -----------------------------

def fig_robustness_frontier(rob, refs, out_dir, db):
    """Clean accuracy on x, shifted accuracy on y. If architecture mattered for
    equivariance, models would separate vertically at matched clean accuracy.
    The diagonal is perfect invariance; the horizontal band is chance."""
    sub = rob[(rob["database"] == db) & (rob["test_aug"] == "none") & (rob["transformation"] != "clean")]
    if sub.empty:
        return
    ref = refs[refs["database"] == db].iloc[0]
    shifts = present_shifts(sub)
    fig, axes = plt.subplots(1, len(shifts), figsize=(5.0 * len(shifts), 4.6), squeeze=False)
    for c, shift in enumerate(shifts):
        ax = axes[0][c]
        s = sub[sub["transformation"] == shift]
        lim = (0, max(1.0, s["clean_accuracy"].max() + 0.05))
        ax.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1, zorder=0)
        ax.axhline(ref["chance_level"], color="#C44E52", linestyle=":", linewidth=1.2, zorder=0)
        ax.text(0.02, ref["chance_level"] + 0.012, "chance", color="#C44E52", fontsize=7)
        for m in present_models(s):
            ms = s[s["model"] == m]
            for aug in present_augs(ms):
                a = ms[ms["train_aug"] == aug]
                ax.scatter(a["clean_accuracy"], a["accuracy"], color=MODEL_COLORS.get(m, "#888"),
                           marker=TRAIN_AUG_MARKERS.get(aug, "o"),
                           s=30 + 70 * a["train_amount"], edgecolor="white", linewidth=0.6, zorder=3)
        if len(s) > 2:
            r = np.corrcoef(s["clean_accuracy"], s["accuracy"])[0, 1]
            ax.text(0.03, 0.97, f"r(clean, shifted) = {r:+.2f}", transform=ax.transAxes,
                    va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="none"))
        ax.set_xlim(*lim)
        ax.set_ylim(0, max(0.35, s["accuracy"].max() + 0.08))
        ax.set_xlabel("Clean accuracy")
        if c == 0:
            ax.set_ylabel("Accuracy under shift")
        ax.set_title(tlabel(shift))
    handles = [Patch(facecolor=MODEL_COLORS.get(m, "#888"), label=mlabel(m)) for m in present_models(sub)]
    handles += [Line2D([0], [0], color="#555", marker=TRAIN_AUG_MARKERS[a], linestyle="", markersize=6,
                       label=TRAIN_AUG_LABELS[a]) for a in present_augs(sub)]
    handles += [Line2D([0], [0], color="#555", marker="o", linestyle="", markersize=4, label="33% data"),
                Line2D([0], [0], color="#555", marker="o", linestyle="", markersize=8, label="100% data")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(f"{dblabel(db)}: clean accuracy vs accuracy under shift\n"
                 "dashed diagonal = perfectly invariant; vertical separation between colours = an architecture effect",
                 fontsize=11.5)
    fig.tight_layout()
    savefig(fig, out_dir, "fig02_robustness_frontier", db)


# --- fig03: the retention metric is confounded ------------------------------

def fig_retention_confound(rob, out_dir, db):
    """Retention (%) plotted against clean accuracy. A downward slope means the
    metric is largely measuring how weak the clean model was, so any claim of
    the form 'model X retains more' has to be read off the absolute figure
    instead."""
    sub = rob[(rob["database"] == db) & (rob["test_aug"] == "none") & (rob["transformation"] != "clean")]
    sub = sub.dropna(subset=["robustness_retained_pct"])
    if sub.empty:
        return
    shifts = present_shifts(sub)
    fig, axes = plt.subplots(1, len(shifts), figsize=(4.9 * len(shifts), 4.3), squeeze=False)
    for c, shift in enumerate(shifts):
        ax = axes[0][c]
        s = sub[sub["transformation"] == shift]
        for m in present_models(s):
            ms = s[s["model"] == m]
            ax.scatter(ms["clean_accuracy"], ms["robustness_retained_pct"],
                       color=MODEL_COLORS.get(m, "#888"), marker=MODEL_MARKERS.get(m, "o"),
                       s=55, edgecolor="white", linewidth=0.6, zorder=3)
        if len(s) > 2:
            x, y = s["clean_accuracy"].values, s["robustness_retained_pct"].values
            r = np.corrcoef(x, y)[0, 1]
            b, a = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, a + b * xs, color="#555", linestyle="--", linewidth=1.3, zorder=2)
            ax.text(0.97, 0.95, f"r = {r:+.2f}\nslope = {b:+.0f} pp of retention\nper unit of clean accuracy",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="none"))
        ax.axhline(100, color="#999", linewidth=1)
        ax.set_xlabel("Clean accuracy")
        if c == 0:
            ax.set_ylabel("Robustness retained (%)\nacc_shift / acc_clean")
        ax.set_title(tlabel(shift))
    fig.legend(handles=model_legend(present_models(sub)), loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{dblabel(db)}: why 'robustness retained' should not be the headline metric\n"
                 "it rewards models that were bad to begin with", fontsize=11.5)
    fig.tight_layout()
    savefig(fig, out_dir, "fig03_retention_confound", db)


# --- fig04: what does train-time augmentation actually buy? -----------------

def fig_augmentation_ladder(summary, refs, out_dir, db):
    """Accuracy as train-time augmentation is escalated, one panel per test
    condition. Whether augmentation transfers to a shift it does not resemble is
    the whole question, and it is a slope, so it should be drawn as one."""
    sub = summary[(summary["database"] == db) & (summary["test_aug"] == "none")]
    if sub.empty:
        return
    ref = refs[refs["database"] == db].iloc[0]
    conds = [c for c in CONDITION_ORDER if c in sub["transformation"].unique()]
    augs = present_augs(sub)
    x = np.arange(len(augs))
    fig, axes = plt.subplots(1, len(conds), figsize=(3.7 * len(conds), 4.3), squeeze=False, sharey=True)
    for c, cond in enumerate(conds):
        ax = axes[0][c]
        s = sub[sub["transformation"] == cond]
        for m in present_models(s):
            means, sds = [], []
            for aug in augs:
                a = s[(s["model"] == m) & (s["train_aug"] == aug)]
                means.append(a["accuracy"].mean() if len(a) else np.nan)
                sds.append(a["accuracy"].std() if len(a) > 1 else 0.0)
            ax.errorbar(x, means, yerr=sds, color=MODEL_COLORS.get(m, "#888"),
                        marker=MODEL_MARKERS.get(m, "o"), markersize=6, linewidth=2, capsize=3)
            if not np.isnan(means[0]) and not np.isnan(means[-1]):
                delta = 100 * (means[-1] - means[0])
                dy = (len(present_models(s)) - 1) / 2 - list(present_models(s)).index(m)
                ax.annotate(f"{delta:+.0f} pp", (x[-1], means[-1]), textcoords="offset points",
                            xytext=(7, 9 * dy), fontsize=7.5, color=MODEL_COLORS.get(m, "#888"), va="center")
        ax.axhline(ref["chance_level"], color="#555", linestyle="--", linewidth=1)
        ax.axhline(ref["majority_baseline"], color="#555", linestyle=":", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([TRAIN_AUG_LABELS[a] for a in augs], rotation=12, ha="right")
        ax.set_title(tlabel(cond))
        ax.set_xlim(-0.3, len(augs) - 0.45)
        if c == 0:
            ax.set_ylabel("Accuracy (averaged over data fractions)")
    fig.legend(handles=model_legend(present_models(sub)) +
               [Line2D([0], [0], color="#555", linestyle="--", label="chance"),
                Line2D([0], [0], color="#555", linestyle=":", label="majority class")],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{dblabel(db)}: does stronger training augmentation transfer to unseen shifts?\n"
                 "error bars = spread across training-data fractions", fontsize=11.5)
    fig.tight_layout()
    savefig(fig, out_dir, "fig04_augmentation_ladder", db)


# --- fig05: data scaling, per condition, with honest y-ranges ---------------

def fig_data_scaling(summary, refs, out_dir, db):
    """Accuracy vs amount of training data. Rows are augmentation regimes and
    columns are test conditions, so each panel carries three lines instead of
    nine, and each column gets its own y-range instead of squashing an OOD
    panel that lives between 0.10 and 0.20 into a 0-1 axis."""
    sub = summary[(summary["database"] == db) & (summary["test_aug"] == "none")]
    if sub.empty:
        return
    ref = refs[refs["database"] == db].iloc[0]
    conds = [c for c in CONDITION_ORDER if c in sub["transformation"].unique()]
    augs = present_augs(sub)
    fig, axes = plt.subplots(len(augs), len(conds), figsize=(3.5 * len(conds), 2.7 * len(augs)),
                             squeeze=False, sharex=True)
    for c, cond in enumerate(conds):
        cs = sub[sub["transformation"] == cond]
        lo = max(0.0, cs["accuracy"].min() - 0.08)
        hi = min(1.02, cs["accuracy"].max() + 0.08)
        if ref["chance_level"] > lo - 0.05:
            lo = min(lo, ref["chance_level"] - 0.03)
        for r, aug in enumerate(augs):
            ax = axes[r][c]
            s = cs[cs["train_aug"] == aug]
            for m in present_models(s):
                ms = s[s["model"] == m].sort_values("train_amount")
                ax.errorbar(ms["train_amount"], ms["accuracy"], yerr=ms["accuracy_std"],
                            color=MODEL_COLORS.get(m, "#888"), marker=MODEL_MARKERS.get(m, "o"),
                            markersize=5, linewidth=1.8, capsize=2.5)
            ax.axhline(ref["chance_level"], color="#555", linestyle="--", linewidth=0.9)
            if abs(ref["majority_baseline"] - ref["chance_level"]) > 1e-6:
                ax.axhline(ref["majority_baseline"], color="#555", linestyle=":", linewidth=1.1)
            ax.set_ylim(lo, hi)
            if r == 0:
                ax.set_title(tlabel(cond))
            if c == 0:
                ax.set_ylabel(f"{TRAIN_AUG_LABELS[aug]}\nAccuracy")
            if r == len(augs) - 1:
                ax.set_xlabel("Fraction of training data")
            ax.set_xticks(sorted(sub["train_amount"].unique()))
    fig.legend(handles=model_legend(present_models(sub)) +
               [Line2D([0], [0], color="#555", linestyle="--", label="chance"),
                Line2D([0], [0], color="#555", linestyle=":", label="majority class")],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{dblabel(db)}: more training data helps on clean data — does it help under shift?", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig05_data_scaling", db)


# --- fig06: the capsule advantage as a signed, significance-marked grid -----

def fig_advantage_heatmap(adv_df, out_dir, db):
    """Replaces a 45-row bar chart. Rows are training settings, columns are test
    conditions, cell is Efficient-CapsNet minus the best baseline in percentage
    points. A dot marks cells where the gap is smaller than twice the pooled
    seed-to-seed standard deviation, i.e. cells that should not be interpreted."""
    sub = adv_df[adv_df["database"] == db].copy()
    if sub.empty:
        return
    sub["row"] = sub.apply(lambda r: setting_label(r["train_aug"], r["train_amount"]), axis=1)
    def _col(r):
        if r["transformation"] == "clean":
            return "Clean"
        return f"{tlabel(r['transformation'])}\n({TEST_AUG_LABELS.get(r['test_aug'], r['test_aug'])} test-time aug)"
    sub["col"] = sub.apply(_col, axis=1)
    row_order = [setting_label(a, f) for a in TRAIN_AUG_ORDER
                 for f in sorted(sub["train_amount"].unique())]
    row_order = [r for r in row_order if r in set(sub["row"])]
    col_order = ["Clean"] + [f"{tlabel(t)}\n({TEST_AUG_LABELS[ta]} test-time aug)"
                             for t in SHIFT_ORDER for ta in ["none", "strong"]]
    col_order = [c for c in col_order if c in set(sub["col"])]
    piv = sub.pivot_table(index="row", columns="col", values="ecaps_advantage_pp").reindex(index=row_order, columns=col_order)
    err = sub.pivot_table(index="row", columns="col", values="ecaps_advantage_pp_std").reindex(index=row_order, columns=col_order)

    vmax = max(1.0, np.nanmax(np.abs(piv.values)))
    fig, ax = plt.subplots(figsize=(1.55 * len(col_order) + 3.4, 0.5 * len(row_order) + 2.4))
    ax.grid(False)
    im = ax.imshow(piv.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, fontsize=8)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            e = err.values[i, j]
            ns = (not np.isnan(e)) and abs(v) < 2 * e
            ax.text(j, i, f"{v:+.1f}" + ("°" if ns else ""), ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * vmax else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Efficient-CapsNet − best baseline (pp)")
    ax.set_title(f"{dblabel(db)}: where, if anywhere, does the capsule model win?\n"
                 "red = capsule behind, blue = capsule ahead, ° = within twice the seed noise", fontsize=11)
    fig.tight_layout()
    savefig(fig, out_dir, "fig06_advantage_heatmap", db)


# --- fig07: is confidence still informative under shift? --------------------

def fig_error_detection_auroc(summary, out_dir, db):
    """Raw confidence scores are not comparable across these architectures (a
    capsule length lives in [0,1], a logit does not), so plotting them on a
    shared axis says nothing. AUROC for separating the model's own correct from
    incorrect predictions is invariant to that rescaling and answers the
    question the confidence plot was meant to: can the model tell when it is
    wrong, and does that survive the shift?"""
    sub = summary[(summary["database"] == db) & (summary["test_aug"] == "none")]
    sub = sub.dropna(subset=["error_detection_auroc"])
    if sub.empty:
        return
    conds = [c for c in CONDITION_ORDER if c in sub["transformation"].unique()]
    augs = present_augs(sub)
    models = present_models(sub)
    x = np.arange(len(conds))
    width = 0.8 / max(len(models), 1)
    fig, axes = plt.subplots(1, len(augs), figsize=(3.6 * len(augs), 4.0), squeeze=False, sharey=True)
    for c, aug in enumerate(augs):
        ax = axes[0][c]
        s = sub[sub["train_aug"] == aug]
        for i, m in enumerate(models):
            vals, errs = [], []
            for cond in conds:
                a = s[(s["model"] == m) & (s["transformation"] == cond)]
                vals.append(a["error_detection_auroc"].mean() if len(a) else np.nan)
                errs.append(a["error_detection_auroc"].std() if len(a) > 1 else 0.0)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, yerr=errs, capsize=2.5,
                   color=MODEL_COLORS.get(m, "#888"), edgecolor="white", linewidth=0.5)
        ax.axhline(0.5, color="#C44E52", linestyle="--", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([tlabel(c_) for c_ in conds], rotation=12, ha="right")
        ax.set_title(TRAIN_AUG_LABELS.get(aug, aug))
        ax.set_ylim(0.4, 1.0)
        if c == 0:
            ax.set_ylabel("AUROC of confidence for detecting its own errors")
    axes[0][-1].text(0.985, 0.505, "no better than a coin flip", color="#C44E52", fontsize=7,
                     ha="right", va="bottom", transform=axes[0][-1].get_yaxis_transform())
    fig.legend(handles=[Patch(facecolor=MODEL_COLORS.get(m, "#888"), label=mlabel(m)) for m in models],
               loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{dblabel(db)}: does the model still know when it is wrong after the shift?", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig07_error_detection_auroc", db)


# --- fig08: prediction collapse --------------------------------------------

def fig_prediction_collapse(summary, refs, out_dir, db):
    """Share of all test predictions landing on the single most-predicted class.
    A model that keeps its accuracy under shift while this rises toward 1.0 has
    not stayed robust: it has stopped discriminating and is riding the class
    prior. This is the diagnostic that separates real invariance from a
    degenerate classifier, and no accuracy-based plot can show it."""
    sub = summary[(summary["database"] == db) & (summary["test_aug"] == "none")]
    if sub.empty or "top_pred_share" not in sub:
        return
    ref = refs[refs["database"] == db].iloc[0]
    conds = [c for c in CONDITION_ORDER if c in sub["transformation"].unique()]
    augs, models = present_augs(sub), present_models(sub)
    x = np.arange(len(conds))
    width = 0.8 / max(len(models), 1)
    fig, axes = plt.subplots(1, len(augs), figsize=(3.6 * len(augs), 4.0), squeeze=False, sharey=True)
    for c, aug in enumerate(augs):
        ax = axes[0][c]
        s = sub[sub["train_aug"] == aug]
        for i, m in enumerate(models):
            vals = [s[(s["model"] == m) & (s["transformation"] == cond)]["top_pred_share"].mean()
                    for cond in conds]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   color=MODEL_COLORS.get(m, "#888"), edgecolor="white", linewidth=0.5)
        ax.axhline(ref["majority_baseline"], color="#555", linestyle=":", linewidth=1.2)
        ax.axhline(1.0 / ref["n_classes"], color="#555", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([tlabel(c_) for c_ in conds], rotation=12, ha="right")
        ax.set_title(TRAIN_AUG_LABELS.get(aug, aug))
        ax.set_ylim(0, 1.02)
        if c == 0:
            ax.set_ylabel("Share of predictions on the single\nmost-predicted class")
    fig.legend(handles=[Patch(facecolor=MODEL_COLORS.get(m, "#888"), label=mlabel(m)) for m in models] +
               [Line2D([0], [0], color="#555", linestyle="--", label="balanced (1/K)"),
                Line2D([0], [0], color="#555", linestyle=":", label="true majority class share")],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{dblabel(db)}: is the model still discriminating, or has it collapsed onto one class?", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig08_prediction_collapse", db)


# --- fig09: which classes survive the shift? -------------------------------

def fig_per_class_drop(df, out_dir, db, train_aug="none"):
    """Per-class accuracy on clean data and under each shift, plus the change.
    Rotation is not equally hard for every class, and which classes survive is
    the most direct evidence available here about what the models are actually
    keying on."""
    sub = df[(df["database"] == db) & (df["test_aug"] == "none") & (df["train_aug"] == train_aug)]
    if sub.empty:
        return
    sub = sub[sub["train_amount"] == sub["train_amount"].max()]
    conds = [c for c in CONDITION_ORDER if c in sub["transformation"].unique()]
    models = present_models(sub)
    classes = np.sort(sub["true_label"].unique())
    piv = sub.groupby(["model", "transformation", "true_label"])["correct"].mean()

    rows, labels = [], []
    for m in models:
        for cond in conds:
            vals = [piv.get((m, cond, cl), np.nan) for cl in classes]
            rows.append(vals)
            labels.append(f"{mlabel(m)} — {tlabel(cond)}")
    mat = np.array(rows, dtype=float)

    names = CLASS_NAMES.get(db)
    xticklabels = [names[int(c)] if names and int(c) < len(names) else str(c) for c in classes]
    fig, ax = plt.subplots(figsize=(0.72 * len(classes) + 4.5, 0.36 * len(rows) + 2.8))
    ax.grid(False)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(xticklabels, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.25 < v < 0.85 else "white")
    for k in range(len(conds), len(rows), len(conds)):
        ax.axhline(k - 0.5, color="black", linewidth=1.2)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Per-class accuracy")
    ax.set_xlabel("True class")
    ax.set_title(f"{dblabel(db)}: per-class accuracy by test condition\n"
                 f"[{TRAIN_AUG_LABELS.get(train_aug, train_aug)}, "
                 f"{int(round(100 * sub['train_amount'].max()))}% of training data, pooled over seeds]", fontsize=11)
    fig.tight_layout()
    savefig(fig, out_dir, "fig09_per_class_accuracy", db)


# --- fig10: is the ranking stable across seeds? ----------------------------

def fig_seed_consistency(seed_summary, out_dir, db):
    """Every individual seed, drawn. Three seeds is few, and if the seed clouds
    of two models overlap then the gap between their means is not a result. The
    line joins each seed's clean accuracy to its own shifted accuracy, so the
    drop is paired rather than a difference of two group means."""
    sub = seed_summary[(seed_summary["database"] == db) & (seed_summary["test_aug"] == "none")]
    if sub.empty:
        return
    shift = "unseen-all" if "unseen-all" in sub["transformation"].unique() else present_shifts(sub)[0]
    augs, models = present_augs(sub), present_models(sub)
    fig, axes = plt.subplots(1, len(augs), figsize=(3.5 * len(augs), 4.2), squeeze=False, sharey=True)
    rng = np.random.default_rng(0)
    for c, aug in enumerate(augs):
        ax = axes[0][c]
        s = sub[(sub["train_aug"] == aug) & (sub["train_amount"] == sub["train_amount"].max())]
        for i, m in enumerate(models):
            ms = s[s["model"] == m]
            clean = ms[ms["transformation"] == "clean"].set_index("seed")["accuracy"]
            shifted = ms[ms["transformation"] == shift].set_index("seed")["accuracy"]
            for seed in sorted(set(clean.index) & set(shifted.index)):
                j = rng.uniform(-0.10, 0.10)
                ax.plot([i + j - 0.14, i + j + 0.14], [clean[seed], shifted[seed]],
                        color=MODEL_COLORS.get(m, "#888"), alpha=0.45, linewidth=1.2, zorder=2)
                ax.scatter(i + j - 0.14, clean[seed], facecolor="white",
                           edgecolor=MODEL_COLORS.get(m, "#888"), s=34, linewidth=1.4, zorder=3)
                ax.scatter(i + j + 0.14, shifted[seed], color=MODEL_COLORS.get(m, "#888"), s=36, zorder=3)
            if len(shifted):
                ax.hlines(shifted.mean(), i + 0.02, i + 0.30, color="black", linewidth=2, zorder=4)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([mlabel(m) for m in models], rotation=15, ha="right")
        ax.set_title(TRAIN_AUG_LABELS.get(aug, aug))
        if c == 0:
            ax.set_ylabel("Accuracy")
    fig.legend(handles=[Line2D([0], [0], color="#888", marker="o", markerfacecolor="white",
                               linestyle="", markersize=7, label="clean, one seed"),
                        Line2D([0], [0], color="#888", marker="o", linestyle="", markersize=7,
                               label=f"{tlabel(shift)}, one seed"),
                        Line2D([0], [0], color="black", linewidth=2, label="mean under shift")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"{dblabel(db)}: every seed, drawn — is the gap between models bigger than the noise?\n"
                 f"[{int(round(100 * sub['train_amount'].max()))}% of training data]", fontsize=11.5)
    fig.tight_layout()
    savefig(fig, out_dir, "fig10_seed_consistency", db)


# --- fig11: both datasets in the same visual language ----------------------

def fig_cross_dataset(summary, refs, out_dir):
    """The two datasets are not comparable in raw accuracy, so they are drawn
    against their own reference levels. The point of the figure is whether the
    conclusion replicates, not whether one dataset scores higher."""
    dbs = [d for d in DATABASES if d in summary["database"].unique()]
    if len(dbs) < 2:
        return
    sub = summary[(summary["test_aug"] == "none") & (summary["train_amount"] == summary["train_amount"].max())]
    conds = [c for c in CONDITION_ORDER if c in sub["transformation"].unique()]
    models = present_models(sub)
    augs = present_augs(sub)
    fig, axes = plt.subplots(len(dbs), len(augs), figsize=(3.4 * len(augs), 3.1 * len(dbs)),
                             squeeze=False, sharex=True)
    x = np.arange(len(conds))
    width = 0.8 / max(len(models), 1)
    for r, db in enumerate(dbs):
        ref = refs[refs["database"] == db].iloc[0]
        for c, aug in enumerate(augs):
            ax = axes[r][c]
            s = sub[(sub["database"] == db) & (sub["train_aug"] == aug)]
            for i, m in enumerate(models):
                vals = [s[(s["model"] == m) & (s["transformation"] == cond)]["accuracy"].mean() for cond in conds]
                errs = [s[(s["model"] == m) & (s["transformation"] == cond)]["accuracy_std"].mean() for cond in conds]
                ax.bar(x + i * width - 0.4 + width / 2, vals, width, yerr=errs, capsize=2,
                       color=MODEL_COLORS.get(m, "#888"), edgecolor="white", linewidth=0.5)
            ax.axhline(ref["chance_level"], color="#555", linestyle="--", linewidth=1)
            ax.axhline(ref["majority_baseline"], color="#C44E52", linestyle=":", linewidth=1.4)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels([tlabel(c_) for c_ in conds], rotation=15, ha="right")
            if r == 0:
                ax.set_title(TRAIN_AUG_LABELS.get(aug, aug))
            if c == 0:
                ax.set_ylabel(f"{dblabel(db)}\nAccuracy", fontweight="bold")
    fig.legend(handles=[Patch(facecolor=MODEL_COLORS.get(m, "#888"), label=mlabel(m)) for m in models] +
               [Line2D([0], [0], color="#555", linestyle="--", label="chance"),
                Line2D([0], [0], color="#C44E52", linestyle=":", label="majority-class baseline")],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Both datasets against their own baselines\n"
                 "(a bar below the dotted line loses to a classifier that ignores the image)", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig11_cross_dataset_vs_baseline")


def make_figures(df, summary, seed_summary, rob, adv_df, refs, out_dir):
    print("\nBuilding figures...")
    for db in [d for d in DATABASES if d in summary["database"].unique()]:
        fig_shift_dumbbell(rob, refs, out_dir, db)
        fig_robustness_frontier(rob, refs, out_dir, db)
        fig_retention_confound(rob, out_dir, db)
        fig_augmentation_ladder(summary, refs, out_dir, db)
        fig_data_scaling(summary, refs, out_dir, db)
        fig_advantage_heatmap(adv_df, out_dir, db)
        fig_error_detection_auroc(summary, out_dir, db)
        fig_prediction_collapse(summary, refs, out_dir, db)
        fig_per_class_drop(df, out_dir, db)
        fig_seed_consistency(seed_summary, out_dir, db)
    fig_cross_dataset(summary, refs, out_dir)


# ----------------------------------------------------------------------------
# AUTO-GENERATED FINDINGS
# ----------------------------------------------------------------------------

def write_findings(summary, seed_summary, rob, adv_pairwise, refs, out_dir):
    """Writes the handful of numbers that decide what the paper can claim.
    Everything here is recomputed from the data, so it cannot drift out of sync
    with the tables."""
    L = ["# Auto-generated findings", "",
         "Recomputed on every run from the prediction CSVs. Numbers are means across seeds.", ""]

    for _, ref in refs.iterrows():
        db = ref["database"]
        L.append(f"## {ref['database_label']}")
        L.append("")
        L.append(f"- {int(ref['n_classes'])} classes, {int(ref['n_test'])} test images. "
                 f"Chance = {ref['chance_level']:.3f}, majority-class baseline = {ref['majority_baseline']:.3f}.")

        s = summary[(summary["database"] == db) & (summary["test_aug"] == "none")]
        clean = s[s["transformation"] == "clean"]
        below = clean[clean["accuracy"] <= ref["majority_baseline"]]
        if len(clean):
            L.append(f"- **Clean accuracy vs the trivial baseline:** {len(below)}/{len(clean)} configurations "
                     f"score at or below the majority-class baseline on clean data.")
            if len(below) > 0.3 * len(clean):
                L.append("  - With this many configurations failing to beat a constant classifier, "
                         "robustness differences on this dataset are differences between models that "
                         "have not learned the task, and should not be interpreted as evidence about "
                         "equivariance either way.")
        for shift in SHIFT_ORDER:
            sh = s[s["transformation"] == shift]
            if sh.empty:
                continue
            worst, best = sh["accuracy"].min(), sh["accuracy"].max()
            L.append(f"- **{tlabel(shift)}:** accuracy spans {worst:.3f}–{best:.3f} "
                     f"(chance {ref['chance_level']:.3f}). "
                     f"Best: {sh.loc[sh['accuracy'].idxmax(), 'model_label']} "
                     f"at {TRAIN_AUG_LABELS.get(sh.loc[sh['accuracy'].idxmax(), 'train_aug'], '')}.")

        r = rob[(rob["database"] == db) & (rob["test_aug"] == "none") & (rob["transformation"] != "clean")]
        r = r.dropna(subset=["robustness_retained_pct"])
        for shift in SHIFT_ORDER:
            rs = r[r["transformation"] == shift]
            if len(rs) > 3:
                corr = np.corrcoef(rs["clean_accuracy"], rs["robustness_retained_pct"])[0, 1]
                L.append(f"- **Retention is confounded ({tlabel(shift)}):** correlation between clean accuracy "
                         f"and retention % is r = {corr:+.2f}"
                         + (". Retention here mostly measures how weak the clean model was."
                            if corr < -0.4 else "."))

        # data scaling under shift
        for shift in SHIFT_ORDER:
            sh = s[s["transformation"] == shift]
            if sh["train_amount"].nunique() < 2:
                continue
            lo, hi = sh["train_amount"].min(), sh["train_amount"].max()
            d_shift = sh[sh["train_amount"] == hi]["accuracy"].mean() - sh[sh["train_amount"] == lo]["accuracy"].mean()
            cl = s[s["transformation"] == "clean"]
            d_clean = cl[cl["train_amount"] == hi]["accuracy"].mean() - cl[cl["train_amount"] == lo]["accuracy"].mean()
            L.append(f"- **Tripling the training data ({lo:g}→{hi:g}):** clean accuracy {100 * d_clean:+.1f} pp, "
                     f"{tlabel(shift)} accuracy {100 * d_shift:+.1f} pp.")

        # augmentation transfer
        for shift in SHIFT_ORDER:
            sh = s[s["transformation"] == shift]
            if not {"none", "strong"} <= set(sh["train_aug"].unique()):
                continue
            d = sh[sh["train_aug"] == "strong"]["accuracy"].mean() - sh[sh["train_aug"] == "none"]["accuracy"].mean()
            L.append(f"- **No aug → strong aug ({tlabel(shift)}):** {100 * d:+.1f} pp.")

        # capsule verdict
        a = adv_pairwise[adv_pairwise["database"] == db] if len(adv_pairwise) else pd.DataFrame()
        if len(a):
            wins = (a["advantage_pp"] > 0).sum()
            clear = a[(a["advantage_pp"] > 0) & (a["advantage_pp"] > 2 * a["advantage_pp_sd"])]
            L.append(f"- **Efficient-CapsNet, paired by seed against each baseline:** ahead in "
                     f"{wins}/{len(a)} comparisons; ahead by more than twice the paired seed SD in "
                     f"{len(clear)}/{len(a)}.")

        # collapse
        c = s[s["transformation"] != "clean"]
        if "top_pred_share" in c and len(c):
            worst = c.loc[c["top_pred_share"].idxmax()]
            L.append(f"- **Most collapsed model under shift:** {worst['model_label']} "
                     f"({TRAIN_AUG_LABELS.get(worst['train_aug'], '')}, {worst['transformation_label']}) puts "
                     f"{100 * worst['top_pred_share']:.0f}% of predictions on a single class.")

        # error detection
        if "error_detection_auroc" in s:
            for m in present_models(s):
                ms = s[s["model"] == m]
                cl = ms[ms["transformation"] == "clean"]["error_detection_auroc"].mean()
                shf = ms[ms["transformation"] != "clean"]["error_detection_auroc"].mean()
                if not np.isnan(cl):
                    L.append(f"- **{mlabel(m)} error-detection AUROC:** {cl:.3f} clean → {shf:.3f} under shift.")
        L.append("")

    path = Path(out_dir) / "FINDINGS.md"
    path.write_text("\n".join(L))
    print(f"\n  wrote {path.name}")
    return "\n".join(L)


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
          f"({df['seed'].nunique()} seed(s): {sorted(df['seed'].unique())}, "
          f"database(s): {sorted(df['database'].unique())}).")

    report_missing_combos(df, args.out_dir)

    refs = dataset_reference_levels(df)
    print("\nReference levels:")
    for _, r in refs.iterrows():
        print(f"  {r['database_label']}: {int(r['n_classes'])} classes, chance={r['chance_level']:.3f}, "
              f"majority baseline={r['majority_baseline']:.3f}")

    summary, seed_summary = compute_summary(df)
    rob, rob_seed, adv_df, adv_pairwise = make_tables(df, summary, seed_summary, refs, args.out_dir)
    make_figures(df, summary, seed_summary, rob, adv_df, refs, args.out_dir)
    write_findings(summary, seed_summary, rob, adv_pairwise, refs, args.out_dir)

    print(f"\nDone. Tables in {Path(args.out_dir) / 'tables'}, figures in {Path(args.out_dir) / 'figures'}")


if __name__ == "__main__":
    main()
