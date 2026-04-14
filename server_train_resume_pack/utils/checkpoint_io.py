import inspect
import sys

import torch


def _ensure_numpy_core_alias():
    """
    Compatibility shim for checkpoints pickled with newer NumPy layouts.

    Some checkpoints reference `numpy._core.*` during unpickling while older
    NumPy versions only expose `numpy.core.*`. Registering an alias in
    sys.modules avoids ModuleNotFoundError without changing checkpoint files.
    """
    try:
        import numpy.core as numpy_core
    except Exception:
        return

    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = numpy_core


def load_torch_checkpoint(path, map_location="cpu"):
    """
    Load a trusted local checkpoint across torch versions.

    PyTorch 2.6+ changed torch.load default to weights_only=True, which can
    fail on full training checkpoints containing optimizer/scheduler states.
    We explicitly request weights_only=False when supported.
    """
    kwargs = {"map_location": map_location}

    try:
        params = inspect.signature(torch.load).parameters
    except (ValueError, TypeError):
        params = {}

    if "weights_only" in params:
        kwargs["weights_only"] = False

    _ensure_numpy_core_alias()

    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)
