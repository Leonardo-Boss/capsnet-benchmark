import time
from datetime import datetime
from abc import abstractmethod
from typing import Callable
from pathlib import Path
import hashlib

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F


from .config import Config
from .logger import MetricTracker, TensorboardWriter, get_logger, CSVLogger, ConfusionTracker

class BaseTrainer:
    """Custom base class for all trainers.

    This class provides basic utilities for training a model, including setting
    up the model architecture, initializing the optimizer and loss function, and
    providing logging and visualization utilities. The class also supports
    monitoring model performance and saving the best model.

    Attributes:
        config (Config): The configuration object.
        model (torch.nn.Module): The model architecture.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module | Callable): The loss function.
        metric_fns (list[torch.nn.Module | Callable]): A list of metric functions.
        n_epoch (int): The number of epochs to train the model.
        start_epoch (int): The starting epoch number, used for resuming training.
        early_stop (int): The number of epochs to wait before early stopping.
        logger (Logger): The logger instance.
        writer (TensorboardWriter): The visualization writer instance.
        log_step (int): The frequency of logging training information.
        save_period (int): The frequency of saving model checkpoints.
        checkpoint_dir (Path): The directory to save model checkpoints.
        monitor (str): The model performance monitoring mode.
        mnt_mode (str): The monitoring mode (min or max).
        mnt_best (float): The best monitored metric value.
    """

    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module | Callable,
        metric_fns: list[torch.nn.Module | Callable],
    ):
        self.config = config
        cfg_trainer: dict = self.config["trainer"]

        # setup architecture
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_fns = metric_fns
        self.n_epoch: int = cfg_trainer["epochs"]
        self.start_epoch = 1

        # setup logger and visualization writer instance
        self.logger = get_logger(
            name="trainer", verbosity=config["trainer"]["verbosity"]
        )
        self.writer = TensorboardWriter(
            config.log_dir, self.logger, enabled=cfg_trainer["tensorboard"]
        )
        self.log_step: int = cfg_trainer["log_step"]
        self.save_period: int = cfg_trainer["save_period"]
        self.checkpoint_dir = config.save_dir

        # configuration to monitor model performance and save best
        self.monitor: str = cfg_trainer.get("monitor", "off")
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in [
                "min",
                "max",
            ], "Only support min and max monitor mode"
            self.mnt_best = float("inf") if self.mnt_mode == "min" else float("-inf")
            self.early_stop: int = cfg_trainer.get("early_stop", float("inf"))
            if self.early_stop < 0:
                self.early_stop = float("inf")

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict:
        """Abstract method for training the model for one epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            dict: A dictionary containing logged information for this epoch.

        Raises:
            NotImplementedError: This is an abstract method that should be
                implemented by subclasses.
        """
        raise NotImplementedError

    def train(self) -> None:
        """Full model training logic for a specified number of epochs.

        Raises:
            KeyError: If the specified metric for monitoring is not found in the log.
        """
        csv_logger = CSVLogger(
            self.config.log_dir / "metrics.csv", reset=(self.start_epoch == 1)
        )

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.n_epoch + 1):
            epoch_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            epoch_log = {"epoch": epoch}
            result = self._train_epoch(epoch)
            epoch_log.update(result)

            # timing -- added after _train_epoch so it reflects the actual
            # time spent training + validating this epoch, not including
            # CSV/checkpoint I/O below
            epoch_log["timestamp"] = datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
            epoch_log["epoch_time_sec"] = round(time.time() - epoch_start, 2)
            epoch_log["max_mem_mb"] = (
                round(torch.cuda.max_memory_allocated() / (1024**2), 2)
                if torch.cuda.is_available() else None
            )

            # print logged information to the screen
            for key, value in epoch_log.items():
                self.logger.info("   {:15s}: {}".format(str(key), value))

            csv_logger.log(epoch_log)

            # monitor best performance and perform early stopping
            best = False
            if self.monitor != "off":
                try:  # check improvement on the specified monitor metric
                    improved = (
                        self.mnt_mode == "min"
                        and epoch_log[self.mnt_metric] < self.mnt_best
                    ) or (
                        self.mnt_mode == "max"
                        and epoch_log[self.mnt_metric] > self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '%s' is not found. Model performance monitoring is disabled.",
                        self.mnt_metric,
                    )
                    self.monitor = "off"
                    improved = False

                if improved:
                    self.mnt_best = epoch_log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                # early stopping
                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for %d epochs. Training stops.",
                        self.early_stop,
                    )
                    break

            # save model checkpoint
            save_periodic = epoch % self.save_period == 0
            if best or save_periodic:
                self._save_checkpoint(epoch, save_best=best, save_periodic=save_periodic)

    def _save_checkpoint(self, epoch: int, save_best: bool = False, save_periodic: bool = True) -> None:
        """Save the current model checkpoint.

        Args:
            epoch (int): The current epoch number.
            save_best (bool, optional): Whether to save this checkpoint as the
                best so far. Defaults to False.
            save_periodic (bool, optional): Whether to save the regular,
                epoch-numbered checkpoint file. Defaults to True.
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        if save_periodic:
            fname = str(self.checkpoint_dir / f"ep{epoch}.pth")
            torch.save(state, fname)
            self.logger.info("Checkpoint saved: %s ...", fname)

        if save_best:  # save as the best yet
            best_fname = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_fname)
            self.logger.info("Best checkpoint saved: %s ...", best_fname)

    def _resume_checkpoint(self, resume_path: str | Path) -> None:
        """Resume training from a saved checkpoint.

        Restores the model weights, optimizer state, the epoch to resume from,
        and the best-monitored-metric-so-far, so training continues as if it
        had never stopped.

        Args:
            resume_path (str | Path): Path to the checkpoint file to resume from.
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: %s ...", resume_path)

        device = next(self.model.parameters()).device
        checkpoint = torch.load(resume_path, map_location=device)

        if checkpoint["arch"] != type(self.model).__name__:
            self.logger.warning(
                "Checkpoint architecture '%s' does not match the currently "
                "configured architecture '%s'. Loading may fail or silently "
                "produce incorrect results.",
                checkpoint["arch"],
                type(self.model).__name__,
            )

        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint.get("monitor_best", self.mnt_best)

        self.logger.info(
            "Checkpoint loaded. Resuming training from epoch %d", self.start_epoch
        )


class MnistTrainer(BaseTrainer):
    """Custom trainer for the MNIST dataset, validation included.

    Attributes:
        config (Config): Configuration object.
        device (torch.device): Device to run the model on.
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module | Callable): Loss function.
        metric_fns (list[Any]): List of metric functions.
        train_data_loader (torch.utils.data.DataLoader): Training data loader.
        valid_data_loader (torch.utils.data.DataLoader, optional): Validation data
            loader. Defaults to None.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional):
            Learning rate scheduler. Defaults to None.
        n_batch (int): Number of training steps (batches) in an epoch.
        train_metrics (MetricTracker): Training metric tracker.
        valid_metrics (MetricTracker): Validation metric tracker.
    """

    def __init__(
        self,
        config: Config,
        device: torch.device,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module | Callable,
        metric_fns: list[torch.nn.Module | Callable],
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ):
        super().__init__(config, model, optimizer, criterion, metric_fns)
        self.config = config
        self.device = device
        self.lr_scheduler = lr_scheduler

        # data loader and metric configuration
        self.train_loader = train_data_loader
        self.valid_loader = valid_data_loader
        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in metric_fns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in metric_fns], writer=self.writer
        )
        num_classes = self.train_loader.dataset[0][1].shape[0]  # from one-hot label
        self.train_confusion = ConfusionTracker(num_classes)
        self.valid_confusion = ConfusionTracker(num_classes)
        self.n_batch = len(self.train_loader)
        self.aug_check: bool = self.config["trainer"].get("aug_check", False)

    def _train_epoch(self, epoch: int) -> dict:
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            dict: A dictionary containing logged information for this epoch.
                Valid log is included if a validation data loader is provided.
        """
        self.model.train()  # set the model to training mode
        self.train_metrics.reset()
        self.train_confusion.reset()

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # configure data and optimizer
            images = images.to(self.device)
            labels = labels.to(self.device)

            # check if augmentations are deterministic
            if self.aug_check:
                check_hash = hashlib.sha256(
                    images.detach().cpu().numpy().tobytes()
                ).hexdigest()[:12]
                self.logger.info(
                    "Augmentation check -- epoch %d batch %d hash: %s",
                    epoch, batch_idx, check_hash,
                )

            self.optimizer.zero_grad()  # zero the gradients

            # forward and backward pass
            out_images, out_labels = self.model(images, labels, mode="train")
            loss = self.criterion(images, labels, out_images, out_labels)
            loss.backward()
            self.optimizer.step()

            # get the index of the maximum value
            label = labels.argmax(dim=1)  # from one-hot
            out_label = out_labels.argmax(dim=1)  # from probability

            # update tracker
            self.writer.set_step((epoch - 1) * self.n_batch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            for metric in self.metric_fns:
                self.train_metrics.update(metric.__name__, metric(label, out_label))

            pred_onehot = F.one_hot(out_label, num_classes=labels.shape[1])
            self.train_confusion.update(labels, pred_onehot)

            # log training information
            if batch_idx % self.log_step == 0 or batch_idx == len(self.train_loader):
                self.logger.debug(self._progress(epoch, batch_idx, loss.item()))
                # add input images to the tensorboard
                self.writer.add_image(
                    "input", make_grid(images.cpu(), nrow=8, normalize=True)
                )

        epoch_log = {**self.train_metrics.result(), **self.train_confusion.result()}


        # validate the model, if provided
        if self.valid_loader is not None:
            val_log = self._valid_epoch(epoch)
            # add validation metrics to epoch log
            epoch_log.update(**{"val_" + k: v for k, v in val_log.items()})

        # update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return epoch_log

    def _valid_epoch(self, epoch: int) -> dict:
        """Validate the model for one epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            dict: A dictionary containing logged information for this epoch.
        """
        self.model.eval()  # set the model to evaluation mode
        self.valid_metrics.reset()
        self.valid_confusion.reset()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.valid_loader):
                # configure data and optimizer
                images = images.to(self.device)
                labels = labels.to(self.device)

                # # forward and backward pass
                out_images, out_labels = self.model(images, labels, mode="eval")
                loss = self.criterion(images, labels, out_images, out_labels)

                # get the index of the maximum value
                label = labels.argmax(dim=1)  # from one-hot
                out_label = out_labels.argmax(dim=1)  # from probability

                # update tracker
                self.writer.set_step(
                    (epoch - 1) * len(self.valid_loader) + batch_idx, "valid"
                )
                self.valid_metrics.update("loss", loss.item())
                for metric in self.metric_fns:
                    self.valid_metrics.update(metric.__name__, metric(label, out_label))

                pred_onehot = F.one_hot(out_label, num_classes=labels.shape[1])
                self.valid_confusion.update(labels, pred_onehot)

        # add histogram of model parameters to the tensorboard
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, bins="auto")

        return {**self.valid_metrics.result(), **self.valid_confusion.result()}

    def _progress(self, epoch_idx: int, batch_idx: int, loss_value: float) -> str:
        """Return a string for logging the training progress."""
        # get amount of training samples
        if hasattr(self.train_loader, "n_samples"):
            current = batch_idx * self.train_loader.batch_size
            samples = self.train_loader.n_samples
        else:
            current = batch_idx
            samples = self.n_batch

        base = "Train Epoch: {:>{}}/{} [{:>{}}/{} ({:3.0f}%)], Loss: {:.6f}"
        return base.format(
            epoch_idx,
            len(str(self.n_epoch)),
            self.n_epoch,
            current,
            len(str(samples)),
            samples,
            100 * current / samples,
            loss_value,
        )
