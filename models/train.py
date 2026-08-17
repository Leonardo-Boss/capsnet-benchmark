import argparse

import numpy as np
import torch

import model.losses as module_loss
import model.metrics as module_metric
import model.model as module_arch
import utils.data_loader as module_data
from utils.config import Config
from utils.trainer import MnistTrainer
from utils.logger import get_logger
from utils.tools import find_latest_checkpoint
from utils.profiling import log_model_complexity


def main(cfg: Config) -> None:
    logger = get_logger(
        name="program", verbosity=cfg["main"]["verbosity"]
    )

    # set seed
    seed = cfg["main"]["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    logger.info("Using seed    : %s", seed)

    # set device
    device = torch.device(
        "cuda" if cfg["main"]["cuda"] and torch.cuda.is_available() else "cpu"
    )
    logger.info("Using device  : %s", device)

    # setup data_loader instances
    train_data_loader = cfg.init_obj("data_loader", module_data)
    valid_data_loader = train_data_loader.split_validation()
    logger.info("Data loaded   : %s", type(train_data_loader).__name__)

    # build model architecture
    model = cfg.init_obj("arch", module_arch)
    model = model.to(device)
    logger.info("Model set up  : %s", type(model).__name__)

    # report model size and compute cost
    sample_images, _ = next(iter(train_data_loader))
    input_size = tuple(sample_images.shape[1:])  # e.g. (3, 32, 32)
    log_model_complexity(model, input_size, device, logger)

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = cfg.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = cfg.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    # get function handles of loss and metrics
    criterion = cfg.init_obj("loss", module_loss)
    logger.info("Using loss    : %s", type(criterion).__name__)
    metrics = [getattr(module_metric, met) for met in cfg["metrics"]]
    logger.info("Using metrics : %s", [met.__name__ for met in metrics])
    print()

    trainer = MnistTrainer(
        config=cfg,
        device=device,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metric_fns=metrics,
        lr_scheduler=lr_scheduler,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
    )

    ckpt_path = find_latest_checkpoint(cfg.save_dir)
    if ckpt_path is not None:
        trainer._resume_checkpoint(ckpt_path)
    else:
        logger.info("No existing checkpoints found for this run_id; starting fresh.")

    # reseed so augmentations are the same independent of the model
    torch.manual_seed(seed)
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Efficient CapsNet Training")
    args.add_argument(
        "-c", "--config", type=str, required=True, help="path to config file"
    )
    args.add_argument(
        "-i", "--run_id", type=str, default=None, help="run id for the experiment"
    )
    
    config = Config.from_args(args)
    main(config)
