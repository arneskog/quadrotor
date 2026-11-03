"""Learned models: CoSTA correction MLP, black-box DDM MLP, linear-drag baseline."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

Array = NDArray[np.float64]


class Normalizer(nn.Module):
    """Affine normalization with statistics fit on TRAINING data only."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class CostaMLP(nn.Module):
    """Corrective source term: features (10) -> [F_corr(3), T_corr(3)] body frame."""

    def __init__(self, n_in: int = 10, hidden: int = 64, n_out: int = 6) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DdmMLP(nn.Module):
    """Black-box data-driven model: features -> [a_body(3), alpha(3)]."""

    def __init__(self, n_in: int = 10, hidden: int = 64, n_out: int = 6) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def fit_linear_drag(v_body: Array, f_res: Array) -> Array:
    """Least-squares linear drag D (3, 3) with F_res ≈ -D v_body."""
    raise NotImplementedError
