from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class NNUncertaintyConfig:
    hidden_dim: int = 64
    hidden_layers: int = 2
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_log_var: float = -6.0
    max_log_var: float = 2.0
    freeze_target_normalization: bool = True
    seed: int = 0
    device: str = "auto"


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - depends on local env.
        raise ImportError(
            "The neural-network FA-SAL backend requires PyTorch. "
            "Install experiments/ncc/requirements.txt."
        ) from exc
    torch.set_num_threads(1)
    return torch, nn


class _HeteroscedasticMLP:
    def __new__(cls, input_dim: int, cfg: NNUncertaintyConfig):
        torch, nn = _require_torch()
        layers = []
        width = int(cfg.hidden_dim)
        prev = int(input_dim)
        for _ in range(max(1, int(cfg.hidden_layers))):
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, 2))
        return nn.Sequential(*layers)


class NNUncertaintyRegressor:
    """Scalar regression model with a directly predicted uncertainty output.

    The network outputs (mean, log_variance). The log variance is trained with
    a Gaussian NLL loss and then used by FA-SAL wherever the GP posterior
    variance would normally appear.
    """

    def __init__(self, input_dim: int, cfg: NNUncertaintyConfig, noise_std: float = 0.05):
        self.input_dim = int(input_dim)
        self.cfg = cfg
        self.noise_var = float(noise_std) ** 2
        self.torch, _ = _require_torch()
        if cfg.device == "auto":
            self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        else:
            self.device = self.torch.device(cfg.device)
        self.torch.manual_seed(int(self.cfg.seed))
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(int(self.cfg.seed))
        self.model = _HeteroscedasticMLP(self.input_dim, cfg).to(self.device)
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.is_fit = False
        self.fit_calls = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        torch = self.torch
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        fit_seed = int(self.cfg.seed) + int(self.fit_calls)
        torch.manual_seed(fit_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fit_seed)

        if (not self.is_fit) or (not bool(self.cfg.freeze_target_normalization)):
            self.y_mean = float(np.mean(y))
            self.y_scale = float(np.std(y))
            if self.y_scale < 1e-8:
                self.y_scale = 1.0
        y_norm = (y - self.y_mean) / self.y_scale

        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y_norm, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.learning_rate),
            weight_decay=float(self.cfg.weight_decay),
        )
        batch_size = max(1, min(int(self.cfg.batch_size), X.shape[0]))
        for _ in range(max(1, int(self.cfg.epochs))):
            perm = torch.randperm(X.shape[0], device=self.device)
            for start in range(0, X.shape[0], batch_size):
                idx = perm[start : start + batch_size]
                pred = self.model(x_tensor[idx])
                mean = pred[:, :1]
                log_var = torch.clamp(pred[:, 1:2], float(self.cfg.min_log_var), float(self.cfg.max_log_var))
                loss = 0.5 * (torch.exp(-log_var) * (y_tensor[idx] - mean) ** 2 + log_var)
                loss = loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.is_fit = True
        self.fit_calls += 1

    def predict_batch(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fit:
            raise RuntimeError("NNUncertaintyRegressor must be fit before prediction.")
        torch = self.torch
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        with torch.no_grad():
            z_tensor = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
            pred = self.model(z_tensor)
            mean_norm = pred[:, 0].detach().cpu().numpy()
            log_var = torch.clamp(pred[:, 1], float(self.cfg.min_log_var), float(self.cfg.max_log_var))
            var_norm = torch.exp(log_var).detach().cpu().numpy()
        mean = self.y_mean + self.y_scale * mean_norm
        std = self.y_scale * np.sqrt(np.clip(var_norm, 1e-12, None))
        return mean.astype(float), std.astype(float)

    def predict(self, z: np.ndarray) -> Tuple[float, float]:
        mean, std = self.predict_batch(np.asarray(z, dtype=float).reshape(1, -1))
        return float(mean[0]), float(std[0])

    def lcb_batch(self, Z: np.ndarray, beta_g: float) -> np.ndarray:
        mean, std = self.predict_batch(Z)
        return mean - np.sqrt(float(beta_g)) * std

    def lcb(self, z: np.ndarray, beta_g: float) -> float:
        return float(self.lcb_batch(np.asarray(z, dtype=float).reshape(1, -1), beta_g)[0])

    def rollout_information_gain(self, X_plan: np.ndarray) -> float:
        _, std = self.predict_batch(X_plan)
        var = np.clip(std**2, 1e-12, None)
        return 0.5 * float(np.sum(np.log1p(var / max(self.noise_var, 1e-12))))
