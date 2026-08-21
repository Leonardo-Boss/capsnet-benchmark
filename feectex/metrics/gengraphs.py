#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(".")

OUTPUT_DIR = Path("plots")

BEST_EPOCH_FILE = DATA_DIR / "bestepoch.csv"


# ============================================================
# Display names
# ============================================================

MODEL_NAMES = {
    "deit_tiny": "DeiT-Tiny",
    "ecaps": "Efficient-CapsNet",
    "resnet18": "ResNet-18",
}

AUGMENTATION_NAMES = {
    "none": "No augmentation",
    "strong": "Strong augmentation",
}


# ============================================================
# Filename parsing
# ============================================================

def parse_filename(path):
    """
    Expected filename:

        modelname_database_augmentation_datafraction_seed.csv

    Example:

        deit_tiny_cifar_10_none_0.33_1.csv

    The last three fields are:

        augmentation
        data fraction
        seed

    Everything before them is model + database.
    """

    stem = path.stem

    parts = stem.split("_")

    if len(parts) < 4:
        return None

    seed = parts[-1]
    fraction = parts[-2]
    augmentation = parts[-3]

    model_database = parts[:-3]

    if not seed.isdigit():
        return None

    try:
        float(fraction)
    except ValueError:
        return None

    if augmentation not in AUGMENTATION_NAMES:
        return None

    if len(model_database) < 2:
        return None

    known_models = sorted(
        MODEL_NAMES.keys(),
        key=len,
        reverse=True,
    )

    model = None
    database = None

    full_model_database = "_".join(model_database)

    for candidate in known_models:

        prefix = candidate + "_"

        if full_model_database.startswith(prefix):

            model = candidate

            database = full_model_database[
                len(prefix):
            ]

            break

    if model is None:

        model = model_database[0]
        database = "_".join(model_database[1:])

    return {
        "model": model,
        "model_name": MODEL_NAMES.get(
            model,
            model,
        ),
        "database": database,
        "augmentation": augmentation,
        "augmentation_name": AUGMENTATION_NAMES[
            augmentation
        ],
        "fraction": fraction,
        "seed": seed,

        # This is exactly the name used in bestepoch.csv
        "experiment_name": stem,

        "filename": path.name,
        "path": path,
    }


# ============================================================
# Load best epochs
# ============================================================

def load_best_epochs(path):
    """
    Reads bestepoch.csv.

    Expected format:

        experiment_a,experiment_b,experiment_c
        42,57,31

    Returns:

        {
            "experiment_a": 42,
            "experiment_b": 57,
            "experiment_c": 31,
        }
    """

    if not path.exists():

        print(
            f"WARNING: {path} does not exist."
        )

        return {}

    df = pd.read_csv(path)

    if df.empty:

        print(
            f"WARNING: {path} is empty."
        )

        return {}

    # Use the first row.
    row = df.iloc[0]

    best_epochs = {}

    for column in df.columns:

        value = row[column]

        if pd.isna(value):
            continue

        try:
            best_epochs[column] = int(value)

        except (ValueError, TypeError):

            print(
                f"WARNING: invalid epoch for "
                f"{column}: {value}"
            )

    return best_epochs


# ============================================================
# Load experiments
# ============================================================

def load_runs(data_dir, best_epochs):

    runs = []

    for path in sorted(data_dir.glob("*.csv")):

        # bestepoch.csv is not a training log
        if path.name == BEST_EPOCH_FILE.name:
            continue

        info = parse_filename(path)

        if info is None:

            print(
                f"Skipping unrecognized filename: "
                f"{path.name}"
            )

            continue

        df = pd.read_csv(path)

        required_columns = [
            "epoch",
            "loss",
            "accuracy",
            "val_loss",
            "val_accuracy",
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:

            print(
                f"WARNING: {path.name} is missing "
                f"{missing}"
            )

            continue

        # ----------------------------------------------------
        # Best epoch
        # ----------------------------------------------------

        experiment_name = info["experiment_name"]

        info["best_epoch"] = best_epochs.get(
            experiment_name
        )

        if info["best_epoch"] is None:

            print(
                f"WARNING: no best epoch found for "
                f"{experiment_name}"
            )

        info["df"] = df

        runs.append(info)

    return runs


# ============================================================
# Best epoch helper
# ============================================================

def get_checkpoint_epoch_point(run, metric):
    """
    Get the metric value at the epoch selected by the checkpoint.

    The checkpoint is selected according to minimum val_loss,
    so run["best_epoch"] is ALWAYS the epoch determined by
    validation loss.

    metric:
        "loss"     -> val_loss at checkpoint epoch
        "accuracy" -> val_accuracy at checkpoint epoch
    """

    best_epoch = run["best_epoch"]

    if best_epoch is None:
        return None

    df = run["df"]

    rows = df[df["epoch"] == best_epoch]

    if rows.empty:
        print(
            f"WARNING: checkpoint epoch {best_epoch} "
            f"not found in {run['filename']}"
        )
        return None

    row = rows.iloc[0]

    if metric == "loss":
        value = row["val_loss"]

    elif metric == "accuracy":
        value = row["val_accuracy"]

    else:
        raise ValueError(
            f"Unknown metric: {metric}"
        )

    return best_epoch, value


def add_best_epoch_dot(
    ax,
    run,
    metric,
    show_label=True,
):
    """
    Add a dot corresponding to the checkpoint selected
    by minimum validation loss.

    IMPORTANT:
    The epoch is the same for both plots.

    For loss:
        dot = val_loss at best checkpoint epoch

    For accuracy:
        dot = val_accuracy at best checkpoint epoch
    """

    point = get_checkpoint_epoch_point(
        run,
        metric,
    )

    if point is None:
        return

    epoch, value = point

    # --------------------------------------------------------
    # Dot
    # --------------------------------------------------------

    ax.scatter(
        epoch,
        value,
        s=10,
        zorder=10,
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    if show_label:

        ax.annotate(
            f"best epoch: {epoch}",
            xy=(epoch, value),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
        )

# ============================================================
# Labels
# ============================================================

def condition_label(run):

    return (
        f"{run['fraction']} data, "
        f"{run['augmentation_name']}"
    )


# ============================================================
# Individual runs
# ============================================================

def plot_individual_runs(runs):

    output = OUTPUT_DIR / "individual"

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for run in runs:

        df = run["df"]

        title = (
            f"{run['model_name']} — "
            f"{run['database']} — "
            f"{condition_label(run)}"
        )

        base = (
            f"{run['model']}_"
            f"{run['database']}_"
            f"{run['augmentation']}_"
            f"{run['fraction']}_"
            f"{run['seed']}"
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            df["epoch"],
            df["loss"],
            label="Training",
        )

        ax.plot(
            df["epoch"],
            df["val_loss"],
            label="Validation",
        )

        add_best_epoch_dot(
            ax,
            run,
            "loss",
        )

        ax.set_title(
            f"{title} — Loss"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            output / f"{base}_loss.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            df["epoch"],
            df["accuracy"],
            label="Training",
        )

        ax.plot(
            df["epoch"],
            df["val_accuracy"],
            label="Validation",
        )

        add_best_epoch_dot(
            ax,
            run,
            "accuracy",
        )

        ax.set_title(
            f"{title} — Accuracy"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")

        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            output / f"{base}_accuracy.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# Plot each model with all conditions
# ============================================================

def plot_by_model(runs):

    output = OUTPUT_DIR / "by_model"

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = sorted(
        set(run["model"] for run in runs)
    )

    for model in models:

        model_runs = [
            run
            for run in runs
            if run["model"] == model
        ]

        model_name = model_runs[0]["model_name"]

        # ----------------------------------------------------
        # Validation loss
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for run in model_runs:

            df = run["df"]

            ax.plot(
                df["epoch"],
                df["val_loss"],
                label=condition_label(run),
            )

            add_best_epoch_dot(
                ax,
                run,
                "loss",
            )

        ax.set_title(
            f"{model_name} — Validation Loss"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")

        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            output / f"{model}_val_loss.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Validation accuracy
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for run in model_runs:

            df = run["df"]

            ax.plot(
                df["epoch"],
                df["val_accuracy"],
                label=condition_label(run),
            )

            add_best_epoch_dot(
                ax,
                run,
                "accuracy",
            )

        ax.set_title(
            f"{model_name} — Validation Accuracy"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")

        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            output / f"{model}_val_accuracy.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# Compare models under each condition
# ============================================================

def plot_model_comparison(runs):

    output = OUTPUT_DIR / "model_comparison"

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    conditions = sorted(
        set(
            (
                run["augmentation"],
                run["fraction"],
            )
            for run in runs
        )
    )

    for augmentation, fraction in conditions:

        matching = [
            run
            for run in runs
            if (
                run["augmentation"] == augmentation
                and run["fraction"] == fraction
            )
        ]

        if not matching:
            continue

        condition = (
            f"{fraction} data — "
            f"{AUGMENTATION_NAMES[augmentation]}"
        )

        # ----------------------------------------------------
        # Validation loss
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for run in matching:

            df = run["df"]

            ax.plot(
                df["epoch"],
                df["val_loss"],
                label=run["model_name"],
            )

            add_best_epoch_dot(
                ax,
                run,
                "loss",
            )

        ax.set_title(
            f"Validation Loss — {condition}"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")

        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            output
            / f"{augmentation}_{fraction}_loss.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Validation accuracy
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for run in matching:

            df = run["df"]

            ax.plot(
                df["epoch"],
                df["val_accuracy"],
                label=run["model_name"],
            )

            add_best_epoch_dot(
                ax,
                run,
                "accuracy",
            )

        ax.set_title(
            f"Validation Accuracy — {condition}"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")

        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            output
            / f"{augmentation}_{fraction}_accuracy.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# 3 × N grid
# ============================================================

def plot_grid(runs, metric):

    output = OUTPUT_DIR / "grids"

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = sorted(
        set(run["model"] for run in runs)
    )

    conditions = sorted(
        set(
            (
                run["augmentation"],
                run["fraction"],
            )
            for run in runs
        ),
        key=lambda x: (
            x[1],
            x[0],
        ),
    )

    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=len(conditions),
        figsize=(
            16,
            4 * len(models),
        ),
        squeeze=False,
    )

    for row, model in enumerate(models):

        for col, (augmentation, fraction) in enumerate(
            conditions
        ):

            ax = axes[row][col]

            matching = [
                run
                for run in runs
                if (
                    run["model"] == model
                    and run["augmentation"] == augmentation
                    and run["fraction"] == fraction
                )
            ]

            for run in matching:

                df = run["df"]

                if metric == "loss":

                    ax.plot(
                        df["epoch"],
                        df["loss"],
                        linestyle="--",
                        label="Train",
                    )

                    ax.plot(
                        df["epoch"],
                        df["val_loss"],
                        label="Validation",
                    )

                    add_best_epoch_dot(
                        ax,
                        run,
                        "loss",
                    )

                    ylabel = "Loss"

                else:

                    ax.plot(
                        df["epoch"],
                        df["accuracy"],
                        linestyle="--",
                        label="Train",
                    )

                    ax.plot(
                        df["epoch"],
                        df["val_accuracy"],
                        label="Validation",
                    )

                    add_best_epoch_dot(
                        ax,
                        run,
                        "accuracy",
                    )

                    ylabel = "Accuracy"

            ax.set_title(
                f"{MODEL_NAMES.get(model, model)}\n"
                f"{fraction} data, "
                f"{AUGMENTATION_NAMES[augmentation]}"
            )

            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)

            ax.grid(alpha=0.3)

            ax.legend(fontsize=7)

    fig.suptitle(
        f"Training Evolution — "
        f"{metric.capitalize()}",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        output / f"training_evolution_{metric}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Metrics table
# ============================================================

def create_metrics_table(runs):

    output = OUTPUT_DIR / "tables"

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for run in runs:

        df = run["df"]

        last = df.iloc[-1]

        best_loss_idx = df["val_loss"].idxmin()
        best_accuracy_idx = df["val_accuracy"].idxmax()

        best_loss = df.loc[best_loss_idx]
        best_accuracy = df.loc[best_accuracy_idx]

        rows.append({

            "model":
                run["model_name"],

            "database":
                run["database"],

            "augmentation":
                run["augmentation_name"],

            "data_fraction":
                run["fraction"],

            "seed":
                run["seed"],

            "best_checkpoint_epoch":
                run["best_epoch"],

            "final_epoch":
                last["epoch"],

            "final_train_loss":
                last["loss"],

            "final_val_loss":
                last["val_loss"],

            "final_train_accuracy":
                last["accuracy"],

            "final_val_accuracy":
                last["val_accuracy"],

            "minimum_val_loss":
                best_loss["val_loss"],

            "minimum_val_loss_epoch":
                best_loss["epoch"],

            "maximum_val_accuracy":
                best_accuracy["val_accuracy"],

            "maximum_val_accuracy_epoch":
                best_accuracy["epoch"],

            "max_mem_mb":
                df["max_mem_mb"].max(),

            "total_time_sec":
                df["epoch_time_sec"].sum(),
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "model",
            "data_fraction",
            "augmentation",
        ]
    )

    result.to_csv(
        output / "training_metrics.csv",
        index=False,
    )

    print("\nTraining metrics:")
    print(result.to_string(index=False))


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load best epochs first
    # --------------------------------------------------------

    best_epochs = load_best_epochs(
        BEST_EPOCH_FILE
    )

    print(
        f"Loaded {len(best_epochs)} best epochs "
        f"from {BEST_EPOCH_FILE}"
    )

    # --------------------------------------------------------
    # Load training logs
    # --------------------------------------------------------

    runs = load_runs(
        DATA_DIR,
        best_epochs,
    )

    if not runs:

        print(
            "No matching CSV files found."
        )

        return

    print(
        f"\nLoaded {len(runs)} experiment(s):\n"
    )

    for run in runs:

        print(
            f"{run['model_name']:20s} | "
            f"{run['database']:15s} | "
            f"{run['augmentation_name']:18s} | "
            f"{run['fraction']:>4s} | "
            f"seed {run['seed']} | "
            f"best epoch {run['best_epoch']}"
        )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    plot_individual_runs(runs)

    plot_by_model(runs)

    plot_model_comparison(runs)

    plot_grid(runs, "loss")

    plot_grid(runs, "accuracy")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    create_metrics_table(runs)

    print(
        f"\nResults written to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
