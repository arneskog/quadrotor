"""Training: train-only PBM parameter fit, closed-form linear drag, and ONE network loop
shared by CoSTA and the DDM.

The network loop minimizes standardized MSE with Adam and selects the checkpoint on a
VALIDATION ROLLOUT score (evaluate.validation_score), never on the regression loss:
validate every `val_every` epochs, keep the best state, stop after `patience`
validations without a `min_rel_improvement` gain, restore the best state.
"""

import copy
import dataclasses
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from quadrotor.labels import measured_wrench
from quadrotor.models import DDM, DDMKin, MLP, CoSTA, FrozenMLP, Normalizer, WrenchModel
from quadrotor.physics import QuadParams, motor_wrench

Array = NDArray[np.float64]
N_FEATURES, N_OUTPUTS = 10, 6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_kf(u: Array, acc_z: Array, p: QuadParams) -> float:
    """Least-squares thrust coefficient [N/(rad/s)^2] from TRAINING rows: m acc_z ≈ kf Σ Ω_i².

    acc_z is the measured body-z specific force [m/s^2], u the motor speeds [rad/s] (N, 4).
    """
    s = np.sum(u**2, axis=1)
    return float(np.sum(p.mass*acc_z*s) / np.sum(s*s))


def fit_torque_gain(u: Array, acc: Array, alpha: Array, w: Array, p: QuadParams) -> Array:
    """Per-axis least-squares gain g (3,) with  measured torque ≈ g * ideal motor torque, from TRAINING rows.

    The ideal motor torque uses torque_gain = 1 regardless of p.torque_gain. A gain far below 1
    means the motor-speed differences carry little of the measured body torque.
    """
    tau_motor = motor_wrench(u, dataclasses.replace(p, torque_gain=np.ones(3)))[1]
    tau_meas = measured_wrench(acc, alpha, w, p)[1]
    return np.sum(tau_motor * tau_meas, axis=0) / np.sum(tau_motor**2, axis=0)


def fit_linear_drag(v_body: Array, f_res: Array) -> Array:
    coef, *_ = np.linalg.lstsq(v_body, f_res, rcond=None)
    return coef.T


@dataclass(frozen=True)
class TrainConfig:
    """Network training settings (configs/train.yaml: mlp, optim, early_stopping)."""

    hidden: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "silu"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    max_epochs: int = 500
    val_every: int = 1  # [epochs]
    patience: int = 20  # [validations]
    min_rel_improvement: float = 0.01

    @classmethod
    def from_dict(cls, cfg: dict) -> TrainConfig:
        es = cfg["early_stopping"]
        return cls(
            hidden=list(cfg["mlp"]["hidden"]),
            activation=cfg["mlp"]["activation"],
            lr=float(cfg["optim"]["lr"]),
            weight_decay=float(cfg["optim"]["weight_decay"]),
            batch_size=int(cfg["optim"]["batch_size"]),
            max_epochs=int(cfg["optim"]["max_epochs"]),
            val_every=int(es.get("val_every_epochs", 1)),
            patience=int(es["patience"]),
            min_rel_improvement=float(es.get("min_rel_improvement", 0.01)),
        )

def build_network_model(kind: str, p: QuadParams, net:MLP, norm: Normalizer):
    frozen = FrozenMLP.from_torch(net)
    if kind == "costa":
        return CoSTA(p=p, net=frozen, norm=norm)
    if kind == "costa_force":
        return CoSTA(p=p, net=frozen, norm=norm, name="costa_force", torque_prior=False)
    if kind == "ddm":
        return DDM(net=frozen, norm=norm)
    if kind == "ddm_kin":
        return DDMKin(p=p, net=frozen, norm=norm)
    raise ValueError(f"not a network model: {kind!r}")

def fit_network(kind, p, X, Y, x_fallback_scale, val_score, cfg, seed, min_std=1e-6):
    set_seed(seed)
    norm = Normalizer.fit(X, Y, x_fallback_scale, min_std)
    Xt = torch.as_tensor(norm.scale_in(X), dtype=torch.float64)
    Yt = torch.as_tensor(norm.scale_out(Y), dtype=torch.float64)

    net = MLP([X.shape[1], *cfg.hidden, N_OUTPUTS], cfg.activation).double()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed)

    hist = {"epoch": [], "train_mse": [], "val_score": []}
    best, best_epoch, best_state, waited = float("inf"), 0, copy.deepcopy(net.state_dict()), 0
    for epoch in range(cfg.max_epochs + 1):
        if epoch % cfg.val_every == 0 or epoch == cfg.max_epochs:
            with torch.no_grad():
                train_mse = float(((net(Xt) - Yt) ** 2).mean())
            score = val_score(build_network_model(kind, p, net, norm))
            hist["epoch"].append(epoch)
            hist["train_mse"].append(train_mse)
            hist["val_score"].append(score)
            if score < best * (1.0 - cfg.min_rel_improvement):
                best, best_epoch, best_state, waited = score, epoch, copy.deepcopy(net.state_dict()), 0
            else:
                waited += 1
            if waited >= cfg.patience or epoch == cfg.max_epochs:
                break
        for idx in torch.randperm(len(Xt), generator=gen).split(cfg.batch_size):
            opt.zero_grad()
            loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
            loss.backward()
            opt.step()

    net.load_state_dict(best_state)
    info = {"history": hist, "best_epoch": best_epoch, "best_val_score": best, "stopped_epoch": epoch}
    return build_network_model(kind, p, net, norm), info