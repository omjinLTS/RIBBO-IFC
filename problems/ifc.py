import logging
from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from problems.base import MetaProblemBase
from datasets.datasets import TrajectoryIterableDataset


# search_space_id → dim. Used when dim is not explicitly passed.
IFC_DIM_MAP = {"IFCSE": 3, "IFCSED10": 10, "IFCSED20": 20,
               "IFCEE": 3, "IFCEED10": 10, "IFCEED20": 20}
# search_space_id → reward type. EE variants add the energy-efficiency denominator.
IFC_REWARD_MAP = {"IFCSE": "SE", "IFCSED10": "SE", "IFCSED20": "SE",
                  "IFCEE": "EE", "IFCEED10": "EE", "IFCEED20": "EE"}


def ifc_rates(H_sq: np.ndarray, X: np.ndarray, Ptx: float = 10.0,
              Pfix: float = 1.0, reward_type: str = "SE",
              sigma2: float = 1.0) -> np.ndarray:
    """Vectorized IFC reward (Lee et al. 2025, eq. 32).

    H_sq[d', d] = |h_{d',d}|^2 (tx d' -> rx d). X: [B, D] with x_d in [0, 1].
    SE: r(x) = Σ_d log(1 + SINR_d)
    EE: r(x) = Σ_d log(1 + SINR_d) / (Pfix + Ptx x_d)
    SINR_d = Ptx |h_dd|^2 x_d / (sigma2 + Ptx Σ_{d'≠d} |h_d'd|^2 x_d')
    Returns Y: [B, 1] (maximize).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    diag = np.diag(H_sq)                              # [D]
    signal = Ptx * diag[None, :] * X                  # [B, D]
    total_rx = Ptx * (X @ H_sq)                        # [B, D], Σ_d' Ptx H[d',d] x_d'
    interference = sigma2 + total_rx - signal          # [B, D]
    sinr = signal / np.maximum(interference, 1e-12)
    rate = np.log1p(sinr)                              # [B, D]
    if reward_type == "EE":
        rate = rate / (Pfix + Ptx * X)                 # per-link EE, summed
    return rate.sum(axis=1, keepdims=True)             # [B, 1]


class IFCNumpy:
    def __init__(self, search_space_id, dataset_id, dim=None, lb=0.0, ub=1.0,
                 reward_type=None):
        self.search_space = search_space_id
        self.dataset_id = dataset_id
        self.dim = dim if dim is not None else IFC_DIM_MAP[search_space_id]
        self.lb = np.ones(self.dim) * lb
        self.ub = np.ones(self.dim) * ub
        self.Ptx = 10.0
        self.Pfix = 1.0
        self.reward_type = reward_type if reward_type is not None \
            else IFC_REWARD_MAP.get(search_space_id, "SE")
        self._set_channel(dataset_id)

    def _set_channel(self, dataset_id):
        rng = np.random.default_rng(int(dataset_id))
        real = rng.normal(0, 1/np.sqrt(2), (self.dim, self.dim))
        imag = rng.normal(0, 1/np.sqrt(2), (self.dim, self.dim))
        self.H_sq = real**2 + imag**2  # |h_{d'd}|^2, shape [D, D]

    def reset_task(self, dataset_id):
        self.dataset_id = dataset_id
        self._set_channel(dataset_id)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X: [B, D], x_d in [0, 1]. maximize, no sign flip.
        return ifc_rates(self.H_sq, X, self.Ptx, self.Pfix, self.reward_type)


class IFCTorch(IFCNumpy):
    def forward(self, X: Tensor) -> Tensor:
        X_np = X.cpu().detach().numpy()
        Y_np = super().forward(X_np)
        return torch.from_numpy(Y_np)


def _ifc_sum_rate(H_sq: np.ndarray, p: np.ndarray, sigma2: float = 1.0) -> float:
    # Σ_d log(1 + (H_sq[d,d] p_d) / (1 + Σ_{k≠d} H_sq[k,d] p_k))
    direct = np.diag(H_sq) * p
    total = H_sq.T @ p + sigma2
    interf = total - direct
    sinr = direct / np.maximum(interf, 1e-12)
    return float(np.sum(np.log(1.0 + sinr)))


def wmmse_sum_rate(
    H_sq: np.ndarray,
    Ptx: float = 10.0,
    sigma2: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-7,
    n_restart: int = 10,
    seed: int = 0,
):
    # Shi et al. 2011 WMMSE for IFC sum-rate with per-link power constraint p_d ∈ [0, Ptx].
    # H_sq[d', d] = |h_{d',d}|^2 (channel power tx d' -> rx d).
    # Returns (best_rate_nats, best_x [D] in [0,1] where x = p/Ptx).
    D = H_sq.shape[0]
    rng = np.random.default_rng(seed)
    diag = np.diag(H_sq)
    sqrt_diag = np.sqrt(diag)
    best_rate, best_p = -np.inf, None

    for restart in range(n_restart):
        if restart == 0:
            p = np.full(D, Ptx)              # all-on init
        elif restart == 1:
            p = np.zeros(D); p[np.argmax(diag)] = Ptx  # single strongest link
        else:
            p = rng.uniform(0, Ptx, D)

        prev_rate = -np.inf
        for _ in range(max_iter):
            total = H_sq.T @ p + sigma2          # signal+interf+noise at rx
            sig = sqrt_diag * np.sqrt(p)         # sqrt(g_dd * p_d)
            u = sig / np.maximum(total, 1e-12)   # MMSE receiver
            interf = total - diag * p
            sinr = (diag * p) / np.maximum(interf, 1e-12)
            w = 1.0 + sinr
            numer = w * u * sqrt_diag
            denom = H_sq @ (w * u**2)            # denom[d] = Σ_k w_k u_k² H_sq[d,k]
            v_new = numer / np.maximum(denom, 1e-12)
            p = np.clip(v_new**2, 0.0, Ptx)

            rate = _ifc_sum_rate(H_sq, p, sigma2)
            if abs(rate - prev_rate) < tol:
                break
            prev_rate = rate

        rate = _ifc_sum_rate(H_sq, p, sigma2)
        if rate > best_rate:
            best_rate, best_p = rate, p.copy()

    return best_rate, best_p / Ptx


def ifc_optimum_bruteforce(H_sq, Ptx=10.0, Pfix=1.0, reward_type="SE",
                           grid=41, sigma2=1.0):
    """Exhaustive grid search optimum (use for small D, e.g. D<=3).
    Returns (best_value, best_x [D])."""
    D = H_sq.shape[0]
    axis = np.linspace(0.0, 1.0, grid)
    mesh = np.meshgrid(*([axis] * D), indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=1)        # [grid^D, D]
    Y = ifc_rates(H_sq, X, Ptx, Pfix, reward_type, sigma2).ravel()
    j = int(np.argmax(Y))
    return float(Y[j]), X[j].copy()


def ifc_optimum_multistart(H_sq, Ptx=10.0, Pfix=1.0, reward_type="SE",
                           n_restart=40, sigma2=1.0, seed=0):
    """Multi-start projected-gradient optimum for any D (near-global on the
    smooth IFC reward). Validated against brute force at D=3. Returns
    (best_value, best_x [D])."""
    from scipy.optimize import minimize
    D = H_sq.shape[0]
    rng = np.random.default_rng(seed)
    diag = np.diag(H_sq)

    def neg(x):
        return -float(ifc_rates(H_sq, x[None, :], Ptx, Pfix, reward_type, sigma2)[0, 0])

    starts = [np.full(D, 1.0), np.full(D, 0.5)]
    s = np.zeros(D); s[int(np.argmax(diag))] = 1.0
    starts.append(s)
    for _ in range(max(0, n_restart - len(starts))):
        starts.append(rng.uniform(0.0, 1.0, D))

    best_val, best_x = -np.inf, None
    bounds = [(0.0, 1.0)] * D
    for x0 in starts:
        res = minimize(neg, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-10})
        v = -res.fun
        if v > best_val:
            best_val, best_x = v, res.x.copy()
    return float(best_val), np.clip(best_x, 0.0, 1.0)


class IFCMetaProblem(MetaProblemBase):
    def __init__(
        self,
        search_space_id: Union[str, List[str]],
        root_dir: str,
        data_dir: str,
        cache_dir: str,
        input_seq_len: int = 300,
        max_input_seq_len: int = 300,
        normalize_method: str = 'random',
        scale_clip_range: Optional[List[float]] = None,
        augment: bool = False,
        prioritize: bool = False,
        prioritize_alpha: float = 1.0,
        n_block: int = 1,
        filter_data: bool = False,
    ):
        self.search_space_id = search_space_id if isinstance(
            search_space_id, str) else search_space_id[0]
        self.dim = IFC_DIM_MAP[self.search_space_id]
        self.lb, self.ub = 0.0, 1.0
        self.input_seq_len = input_seq_len
        self.scale_clip_range = scale_clip_range
        self.dataset_id = '0'

        self.func = IFCNumpy(self.search_space_id,
                             self.dataset_id, self.dim, self.lb, self.ub)

        self.dataset = TrajectoryIterableDataset(
            search_space_id=search_space_id,
            data_dir=data_dir,
            cache_dir=cache_dir,
            input_seq_len=input_seq_len,
            max_input_seq_len=max_input_seq_len,
            normalize_method=normalize_method,
            scale_clip_range=scale_clip_range,
            prioritize=prioritize,
            prioritize_alpha=prioritize_alpha,
            n_block=n_block,
            filter_data=filter_data,
        )

        self.dataset.transform_x(
            partial(self.transform_x, reverse=True, lb=self.lb, ub=self.ub))

        self.get_problem_info()
        self.cheat_table = dict()

    def forward(self, X: Tensor):
        assert X.ndim == 2
        assert (X >= -1 - 1e-6).all() and (X <= 1 + 1e-6).all()

        X_np = X.cpu().detach().numpy()
        Y_np = self.func(self.transform_x(X_np, lb=self.lb, ub=self.ub))
        normalized_y, normalized_regret = self.get_normalized_y_and_regret(
            Y_np)
        return torch.from_numpy(normalized_y).reshape(-1, 1), {
            'raw_y': torch.from_numpy(Y_np).reshape(-1, 1),
            'normalized_onestep_regret': torch.from_numpy(normalized_regret).reshape(-1, 1),
        }
