"""Utilities for reporting model size and compute cost."""
from typing import Any

import torch


def count_params(model: torch.nn.Module) -> int:
    """Total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _routing_caps_flops_hook(module, input, output) -> None:
    """Custom ptflops hook for RoutingCaps.

    RoutingCaps computes its votes via a raw `einsum` rather than a
    standard nn.Linear/Conv2d, so ptflops' default hooks see zero compute
    here unless told otherwise. The dominant cost is the prediction-vector
    einsum "...ji,kjiz->...kjz": for each of N1*N0*D1 output elements, D0
    multiply-accumulates are performed.
    """
    x = input[0]
    batch = x.shape[0]
    macs = batch * module.N1 * module.N0 * module.D0 * module.D1
    module.__flops__ += int(macs)


def count_macs(
    model: torch.nn.Module, input_size: tuple[int, int, int], device: torch.device
) -> float:
    """Estimate multiply-accumulate operations (MACs) for one forward pass.

    Args:
        model: Model to profile. Must accept this repo's shared interface
            `forward(x, y_true=None, mode='train')`. Profiled with
            `mode='eval'` so no `y_true` is required.
        input_size: (channels, height, width) of a single input image, e.g.
            `images.shape[1:]` from a real batch.
        device: Device to run the dummy forward pass on.

    Returns:
        float: Total MACs for a single (batch size 1) forward pass.

    Raises:
        ImportError: If ptflops is not installed (`pip install ptflops`).
    """
    try:
        from ptflops import get_model_complexity_info
    except ImportError as exc:
        raise ImportError(
            "GMAC counting requires ptflops. Install it with 'pip install ptflops'."
        ) from exc

    def input_constructor(res: tuple[int, int, int]) -> dict[str, Any]:
        return {"x": torch.rand(1, *res, device=device), "mode": "eval"}

    custom_hooks = {}
    try:
        from model.layers import RoutingCaps
        custom_hooks[RoutingCaps] = _routing_caps_flops_hook
    except ImportError:
        pass  # not a capsule-based architecture; nothing to add

    was_training = model.training
    model.eval()
    with torch.no_grad():
        macs, _ = get_model_complexity_info(
            model,
            input_size,
            input_constructor=input_constructor,
            custom_modules_hooks=custom_hooks,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
    if was_training:
        model.train()

    return macs


def log_model_complexity(
    model: torch.nn.Module,
    input_size: tuple[int, int, int],
    device: torch.device,
    logger,
) -> None:
    """Logs parameter count and GMACs for `model`.

    GMAC counting failures (ptflops missing, or an architecture ptflops
    can't trace) are caught and logged as a warning rather than crashing
    the run -- parameter counting is unaffected either way.
    """
    params = count_params(model)
    logger.info("Parameters    : %s (%.2fM)", f"{params:,}", params / 1e6)

    try:
        macs = count_macs(model, input_size, device)
        logger.info("GMACs         : %.4f (input size %s)", macs / 1e9, input_size)
    except Exception as exc:  # noqa: BLE001 -- profiling is best-effort
        logger.warning("Could not compute GMACs: %s", exc)
