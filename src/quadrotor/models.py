"""Models as body-frame wrench suppliers: PBM, PBM + linear drag, CoSTA, DDM.

Every model answers one question, wrench(x, u) -> (force [N] (3,), torque [N m] (3,)),
and make_dynamics() turns it into xdot through the SAME rigid-body equations, so the
integrator, rollouts and evaluation are shared by construction:

    PBM          motor_wrench(u)
    LinearDrag   motor_wrench(u) + D v_body
    CoSTA        motor_wrench(u) + net(features)   (residual)
    DDM          net(features)                     (full wrench)

Networks are trained in torch (MLP) and evaluated in numpy with frozen weights
(FrozenMLP), which is much faster inside RK4 rollouts. Every wrench() must accept
a leading batch dimension: x (..., 13), u (..., 4).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from torch import nn
from scipy.special import expit
from quadrotor.physics import QuadParams, rigid_body, motor_wrench, body_velocity
from quadrotor.labels import features, features_kin, gravity_body

Array = NDArray[np.float64]
Dynamics = Callable[[Array, Array], Array]

class WrenchModel(Protocol):
    name: str

    def wrench(self, x: Array, u: Array) -> tuple[Array, Array]:
        """Body-frame force [N] (..., 3) and torque [N m] (..., 3) at state(s) x, motor speeds u."""
        ...


def make_dynamics(model: WrenchModel, p: QuadParams) -> Dynamics:
    """xdot = rigid_body(x, model.wrench(x, u)): the one bridge from any model to the integrator."""
    def f(x, u):
        force_b, torque_b = model.wrench(x, u)
        return rigid_body(x, force_b, torque_b, p)
    return f

@dataclass(frozen=True)
class PBM:
    p: QuadParams
    name: str = "pbm"

    def wrench(self, x, u):
        return motor_wrench(u, self.p)


@dataclass(frozen=True)
class LinearDrag:
    """PBM plus a linear body-frame drag force D v_body; D (3, 3) [N s/m] fitted on TRAIN residuals."""

    p: QuadParams
    D: Array
    name: str = "linear_drag"

    def wrench(self, x, u):
        force_b, torque_b = motor_wrench(u, self.p)
        return force_b + body_velocity(x) @ self.D.T, torque_b

@dataclass(frozen=True)
class Normalizer:
    """Affine input/output scaling fitted on TRAINING rows only, then frozen."""

    x_mean: Array
    x_scale: Array
    y_mean: Array
    y_scale: Array

    @classmethod
    def fit(cls, X: Array, Y: Array, x_fallback: Array, min_std: float = 1e-6) -> Normalizer:
        """Mean/std per channel; a channel with std < min_std uses its declared physical scale
        (inputs: x_fallback) or 1.0 (outputs) instead of a near-zero std (L04)."""
        x_std, y_std = X.std(axis=0), Y.std(axis=0)
        return cls(
            x_mean = X.mean(axis=0),
            x_scale = np.where(x_std < min_std, x_fallback, x_std),
            y_mean = Y.mean(axis=0),
            y_scale = np.where(y_std < min_std, 1.0, y_std)
        )
        

    def scale_in(self, X: Array) -> Array: return (X - self.x_mean) / self.x_scale
    def unscale_out(self, Z: Array) -> Array: return Z * self.y_scale + self.y_mean
    def scale_out(self, Y: Array) -> Array: return (Y -self.y_mean) / self.y_scale

class MLP(nn.Module):
    """MLP with smooth activations ('silu' or 'tanh'), linear output, Glorot-normal weights, zero biases.

    layers e.g. [10, 64, 64, 6]. Keep the Linear/activation modules in self.net (nn.Sequential)
    and the activation name in self.activation (FrozenMLP.from_torch reads both).
    """

    def __init__(self, layers: list[int], activation: str = "silu"):
        super().__init__()
        act = {"tanh": nn.Tanh, "silu": nn.SiLU}[activation]
        mods = []
        for d_in, d_out in zip(layers[:-1], layers[1:]):
            mods += [nn.Linear(d_in, d_out), act()]
        self.net = nn.Sequential(*mods[:-1]) # Lineær output, fjerner siste aktiveringsfunksjon

        self.activation = activation
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


@dataclass(frozen=True)
class FrozenMLP:
    """Numpy copy of a trained MLP's weights, for fast rollouts. (..., d_in) -> (..., d_out)."""

    weights: tuple[tuple[Array, Array], ...]
    activation: str

    @classmethod
    def from_torch(cls, net):
        layers = tuple((m.weight.detach().double().numpy().copy(), m.bias.detach().double().numpy().copy())
                       for m in net.net if isinstance(m, nn.Linear))
        return cls(weights=layers, activation=net.activation)

    def __call__(self, z):
        act = {"tanh": np.tanh, "silu": lambda a: a * expit(a)}[self.activation]
        for i, (W, b) in enumerate(self.weights):
            z = z @ W.T + b
            if i < len(self.weights) - 1:
                z = act(z)
        return z

@dataclass(frozen=True)
class CoSTA:
    """PBM plus a learned residual wrench from features(x, u) of the MODEL's own state.

    torque_prior=False ('costa_force'): the physics prior is used for the force only and the
    network supplies the whole torque (for when the motor-torque prior is unreliable).
    """

    p: QuadParams
    net: FrozenMLP
    norm: Normalizer
    name: str = "costa"
    torque_prior: bool = True

    def correction(self, x: Array, u: Array) -> Array:
        """Learned residual [F (3) N, T (3) N m], (..., 6)."""
        return self.norm.unscale_out(self.net(self.norm.scale_in(features(x,u))))

    def wrench(self, x: Array, u: Array) -> tuple[Array, Array]:
        force_b, torque_b = motor_wrench(u, self.p)
        c = self.correction(x, u)
        if not self.torque_prior:
            return force_b + c[..., :3], c[..., 3:]
        return force_b + c[..., :3], torque_b + c[..., 3:]


@dataclass(frozen=True)
class DDM:
    """Black-box body-frame wrench from features(x, u); rigid-body equations still apply."""

    net: FrozenMLP
    norm: Normalizer
    name: str = "ddm"

    def wrench(self, x: Array, u: Array) -> tuple[Array, Array]:
        c = self.norm.unscale_out(self.net(self.norm.scale_in(features(x,u))))
        return c[..., :3], c[..., 3:]

@dataclass(frozen=True)
class DDMKin:
    """Kinematics-only DDM: the network predicts body accelerations [vdot_b, wdot] incl. gravity."""
    p: QuadParams
    net: FrozenMLP
    norm: Normalizer
    name: str = "ddm_kin"

    def accelerations(self, x: Array, u: Array) -> Array:
        return self.norm.unscale_out(self.net(self.norm.scale_in(features_kin(x, u))))

    def wrench(self, x: Array, u: Array) -> tuple[Array, Array]:
        a = self.accelerations(x, u)
        w = x[..., 10:13]
        force_b = self.p.mass * (a[..., :3] + self.p.g * gravity_body(x))
        torque_b = a[..., 3:] @ self.p.inertia.T + np.cross(w, w @ self.p.inertia.T)
        return force_b, torque_b


def save_model(model: WrenchModel, path: Path) -> None:
    """Store a model's learned parts in one .npz (PBM parameters are saved separately by the run)."""
    arrays: dict[str, Array] = {"kind": np.array(model.name)}
    if isinstance(model, LinearDrag):
        arrays["D"] = model.D
    if isinstance(model, (CoSTA, DDM, DDMKin)):
        arrays["activation"] = np.array(model.net.activation)
        for i, (W, b) in enumerate(model.net.weights):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        for k in ("x_mean", "x_scale", "y_mean", "y_scale"):
            arrays[k] = getattr(model.norm, k)
    np.savez(path, **arrays)


def load_model(path: Path, p: QuadParams) -> WrenchModel:
    """Inverse of save_model; p are the (fitted) PBM parameters of the same run."""
    z = np.load(path)
    kind = str(z["kind"])
    if kind == "pbm":
        return PBM(p)
    if kind == "linear_drag":
        return LinearDrag(p, z["D"])
    n_layers = sum(1 for k in z.files if k.startswith("W"))
    net = FrozenMLP(tuple((z[f"W{i}"], z[f"b{i}"]) for i in range(n_layers)), str(z["activation"]))
    norm = Normalizer(z["x_mean"], z["x_scale"], z["y_mean"], z["y_scale"])
    if kind == "costa":
        return CoSTA(p=p, net=net, norm=norm)
    if kind == "costa_force":
        return CoSTA(p=p, net=net, norm=norm, name="costa_force", torque_prior=False)
    if kind == "ddm_kin":
        return DDMKin(p=p, net=net, norm=norm)
    return DDM(net=net, norm=norm)