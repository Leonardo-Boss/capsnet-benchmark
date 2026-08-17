import csv
import importlib
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

LOG_LEVELS = {
    0: logging.DEBUG,
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
}


class TensorboardWriter:
    """Handle visualization of training and validation statistics in Tensorboard.

    Attributes:
        writer (Any): The writer object that writes data for Tensorboard.
        selected_module (str): The module name used for Tensorboard, either from
            torch or tensorboardX.
        step (int): The current step in the training/validation process.
        mode (str): The current mode (training or validation).
        tb_writer_fns (Set[str]): A set of functions that the writer can use.
        tag_mode_exceptions (Set[str]): A set of functions that don't require a
            tag mode.
        timer (datetime): A timer for tracking time.
    """

    def __init__(self, log_dir: str, logger: logging.Logger, enabled: bool):
        """Initialize the TensorboardWriter.

        Args:
            log_dir (str): The directory where the logs will be stored.
            logger (logging.logger): The logger object used for logging.
            enabled (bool): A flag indicating whether Tensorboard is enabled or not.
        """
        self.writer: Any = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # retrieve vizualization writer
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    self.selected_module = module
                    break
                except ImportError:
                    succeeded = False
            if not succeeded:
                message = (
                    "Warning: visualization (Tensorboard) is configured to use, "
                    "but currently not installed on this machine. Please install "
                    "either TensorboardX with 'pip install tensorboardx', upgrade "
                    "PyTorch to version >= 1.1 to use 'torch.utils.tensorboard' "
                    "or turn off the option in the 'config.yaml' file."
                )
                logger.warning(message)

        self.step = 0
        self.mode = ""

        self.tb_writer_fns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }
        self.tag_mode_exceptions = {"add_histogram", "add_embedding"}
        self.timer = datetime.now()

    def set_step(self, step: int, mode: str = "train") -> None:
        """Set the step number for the visualization writer (x axis).

        If step is 0, the timer is reset. Otherwise, the time duration between
        the last step and the current step is recorded and added to the
        visualization.

        Args:
            step (int): The step number in the training/validation process.
            mode (str, optional): The mode of operation, either "train" or
                "valid". Defaults to "train".
        """
        self.mode = mode
        self.step = step

        if step == 0:  # reset timer
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name: str) -> Callable:
        """Return a function that adds data to tensorboard or does nothing.

        If visualization is configured to use, this method returns the
        add_data() methods of tensorboard with additional information (step,
        tag) added. Otherwise, it returns a blank function handle that does
        nothing.

        Args:
            name (str): The name of the attribute to get.

        Returns:
            Callable: A function that either adds data to the tensorboard or
                does nothing.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        if name in self.tb_writer_fns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                """A wrapper for add_data function."""
                if add_data is not None:
                    # add mode (train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = f"{self.mode}/{tag}"
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class
            try:
                attr = object.__getattr__(name)
            except AttributeError as exc:
                raise AttributeError(
                    f"Type object '{self.selected_module}' has no attribute '{name}'"
                ) from exc
            return attr


class MetricTracker:
    """Class for tracking metrics during model training and evaluation.

    This class uses a pandas DataFrame to store the total, counts, and average
    for each metric. If a `TensorboardWriter` is provided, it will also logs the
    value to the tensorboard.

    Attributes:
        writer (TensorboardWriter): Tensorboard writer for logging metrics.
        _data (pd.DataFrame): DataFrame for tracking metrics.
    """

    def __init__(self, *keys: str, writer: TensorboardWriter = None):
        """Initialize instance for tracking metrics for a given set of keys.

        Args:
            *keys (str): Metrics keys to be tracked. Used as index in a
                DataFrame with columns "total", "counts", "average".
            writer (TensorboardWriter, optional): Used to log the metrics.
                Defaults to None.
        """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        """Reset all values in the DataFrame to 0."""
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key: str, value: float, n: int = 1):
        """Updates the data on a given key.

        Args:
            key (str): The key for which the metrics are updated.
            value (float): The value to add to the total for the key.
            n (int, optional): The count to add to the counts for the key.
                Defaults to 1.
        """
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = (
            self._data["total"][key] / self._data["counts"][key]
        )

        if self.writer:
            self.writer.add_scalar(key, value)

    def avg(self, key: str) -> float:
        """Returns the average for a given key."""
        return self._data["average"][key]

    def result(self) -> dict:
        """Returns a dictionary of averages for all keys."""
        return dict(self._data["average"])


def global_logger_setup(log_cfg: dict, log_dir: str | Path) -> None:
    """Setup global logging. All loggers will inherit this setup.

    Args:
        log_cfg (dict): The logging configuration dictionary.
        log_dir (str | Path): The directory to save the logs.
    """
    for _, handler in log_cfg["handlers"].items():
        if "filename" in handler:
            handler["filename"] = str(log_dir / handler["filename"])
    logging.config.dictConfig(log_cfg)


def get_logger(name: str, verbosity: int = 2) -> logging.Logger:
    """Instantiate logger with the specified name and verbosity level.

    Args:
        name (str): The name of the logger.
        verbosity (int, optional): The verbosity level of the logger. Defaults to 2.

    Returns:
        logging.Logger: The logger with the specified name and verbosity level.

    Raises:
        AssertionError: If the verbosity level is not within the valid range (0-3).
    """
    assert (
        verbosity in LOG_LEVELS
    ), f"Verbosity option {verbosity} is out of range (0-3)"
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS[verbosity])
    return logger

class CSVLogger:
    """Logs per-epoch training metrics to a CSV file, one row per epoch.

    Column headers are taken from the keys of the first row logged.
    """

    def __init__(self, log_path: str | Path, reset: bool = True):
        """
        Args:
            log_path (str | Path): Path to the CSV file to write.
            reset (bool, optional): If True, start a fresh file (overwriting
                any existing one). If False, append to an existing file,
                reusing its header -- used when resuming training so metrics
                from earlier epochs aren't lost. If no file exists yet,
                behaves the same as reset=True regardless. Defaults to True.
        """
        self.log_path = Path(log_path)
        self._fieldnames: list[str] | None = None

        if not reset and self.log_path.exists():
            with self.log_path.open("r", newline="", encoding="utf8") as f:
                header = next(csv.reader(f), None)
            if header:
                self._fieldnames = header
        else:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists():
                self.log_path.unlink()

    def log(self, row: dict) -> None:
        """Append one row of metrics to the CSV file.

        Args:
            row (dict): Mapping of column name to value for this row. On the
                first call this establishes the header.

        Raises:
            ValueError: If `row`'s keys don't match the established header.
        """
        write_header = False

        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            write_header = True
        elif set(row.keys()) != set(self._fieldnames):
            raise ValueError(
                f"CSV row keys {sorted(row.keys())} do not match this "
                f"log's established header {sorted(self._fieldnames)}. All "
                f"logged epochs must report the same set of metrics."
            )

        with self.log_path.open("a", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class ConfusionTracker:
    """Accumulates per-class TP/FP/FN/TN counts across an epoch.

    Expects one-hot true labels and one-hot predicted labels per batch
    (shape (batch, num_classes) each). Counts are kept separately for each
    class (not pooled/micro-averaged). All ops are vectorized tensor ops with one sync
    per batch, so overhead is negligible next to the forward/backward pass.

    Attributes:
        num_classes (int): Number of classes being tracked.
        tp, fp, fn, tn (torch.Tensor): Running per-class counts, shape
            (num_classes,).
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fn = torch.zeros(self.num_classes, dtype=torch.long)
        self.tn = torch.zeros(self.num_classes, dtype=torch.long)

    def update(self, y_true_onehot: torch.Tensor, y_pred_onehot: torch.Tensor):
        y_true = y_true_onehot.long().cpu()
        y_pred = y_pred_onehot.long().cpu()

        self.tp += (y_true * y_pred).sum(dim=0)
        self.fp += ((1 - y_true) * y_pred).sum(dim=0)
        self.fn += (y_true * (1 - y_pred)).sum(dim=0)
        self.tn += ((1 - y_true) * (1 - y_pred)).sum(dim=0)

    def result(self) -> dict:
        """Returns a flat dict with one key per class per count, e.g.
        {"tp_0": 412, "fp_0": 38, ..., "tp_9": 390, ...} -- flat so it
        drops straight into CSVLogger/epoch_log without any special-casing.
        """
        out = {}
        for c in range(self.num_classes):
            out[f"tp_{c}"] = self.tp[c].item()
            out[f"fp_{c}"] = self.fp[c].item()
            out[f"fn_{c}"] = self.fn[c].item()
            out[f"tn_{c}"] = self.tn[c].item()
        return out
