"""Compare FA-SAL (NN), SAL-NX and Tebbe-ABM on the NCC/APS dataset."""
import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from blas_threads import configure_blas_threads

configure_blas_threads()

import numpy as np

from gp_models import DynamicsGP, KernelConfig, SafetyGP
from planning import (FutureAwareTrajectoryPlanner, PlannerConfig,
                      SALNXTrajectoryPlanner, TebbeABMTrajectoryPlanner)
from .surrogate import NNUncertaintyConfig, NNUncertaintyRegressor


PLANNERS = {"fa_sal_nn": FutureAwareTrajectoryPlanner, "salnx": SALNXTrajectoryPlanner,
            "tebbe_abm": TebbeABMTrajectoryPlanner}
METRICS = ("rmse", "svr", "safety_iou")
KERNEL = KernelConfig(kind="se", variance=1.0, length_scale=5.0)
NOISE_STD = 0.05
PLANNER = PlannerConfig(horizon=4, beta_f=2.0, beta_g=2.0, Lx=1.0, Ly=1.0,
                        Lf=1.0, L_ell=0.75, fa_beam_width=1, alpha_safety=0.2,
                        salnx_criterion="logdet", salnx_joint_mc_samples=64,
                        tebbe_alpha=0.2, tebbe_criterion="trace", tebbe_confidence_delta=1e-3,
                        tebbe_sample_start=32, tebbe_sample_stages=6)


def load_dataset(path):
    with np.load(path, allow_pickle=False) as arrays:
        x = arrays["X_scaled"].astype(float)
        y = arrays["Y_scaled"]
        if x.ndim != 2 or x.shape[1] != 29 or y.shape != (len(x), 30):
            raise ValueError("Expected X_scaled (N, 29) and Y_scaled (N, 30).")
        target_idx = int(arrays["noise_node_idx"].item()) if "noise_node_idx" in arrays else 0
        if not 0 <= target_idx < y.shape[1]:
            raise ValueError("noise_node_idx must identify an output column.")
        target = y[:, target_idx].astype(float)
    if not len(target) or not (np.isfinite(x).all() and np.isfinite(target).all()):
        raise ValueError("Expected finite inputs and target values.")
    threshold = float(np.quantile(target, 0.8))
    if np.all(target == target[0]):
        raise ValueError("The selected target is constant. Generate a new graph with --seed.")
    return x, target, threshold, target_idx


def run_trial(method, trial, args, data):
    x, target, threshold, _ = data
    margin = threshold - target
    rng = np.random.default_rng(args.seed + trial)
    order = rng.permutation(len(x))
    evaluation = np.sort(order[:args.eval_size])
    pool = np.sort(order[args.eval_size:args.eval_size + args.pool_size])
    safe_pool = pool[margin[pool] >= 0]
    if len(safe_pool) < args.n_init:
        raise ValueError("Not enough safe initial points in the pool.")
    train = np.sort(rng.choice(safe_pool, args.n_init, replace=False))
    available = np.setdiff1d(pool, train)
    if method == "fa_sal_nn":
        nn = NNUncertaintyConfig(epochs=args.nn_epochs, seed=args.seed + 10_000 * trial)
        dynamics = NNUncertaintyRegressor(x.shape[1], nn, NOISE_STD)
        safety = NNUncertaintyRegressor(x.shape[1], replace(nn, seed=nn.seed + 1), NOISE_STD)
    else:
        dynamics = DynamicsGP(KERNEL, NOISE_STD)
        safety = SafetyGP(KERNEL, NOISE_STD)

    planner = PLANNERS[method](dynamics, safety, PLANNER)
    truth = margin[evaluation] >= 0
    violations = queried = 0
    history = []
    for _ in range(args.rounds):
        dynamics.fit(x[train], target[train])
        safety.fit(x[train], margin[train])
        windows = np.lib.stride_tricks.sliding_window_view(available, PLANNER.horizon)
        candidates = [(window, x[window]) for window in windows]
        selected, _, _, infos = planner.select_trajectory(candidates)
        fallback = selected is None
        if fallback:
            # Original APS unsafe_fallback: pick the strongest pointwise safety bound.
            selected = max(infos, key=lambda info: (info.pointwise_lcb, info.acquisition)).endpoint
        selected = np.asarray(selected, dtype=int)
        violations += np.count_nonzero(margin[selected] < 0)
        queried += len(selected)
        prediction, _ = dynamics.predict_batch(x[evaluation])
        certified = safety.lcb_batch(x[evaluation], PLANNER.beta_g) >= 0
        union = np.count_nonzero(truth | certified)
        history.append({
            "rmse": float(np.sqrt(np.mean((prediction - target[evaluation]) ** 2))),
            "svr": float(violations / queried),
            "safety_iou": float(np.count_nonzero(truth & certified) / union) if union else 1.0,
            "fallback": fallback, "selected_indices": selected.tolist(),
        })
        train = np.union1d(train, selected)
        available = np.setdiff1d(available, selected)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("experiments/ncc/data/aps.npz"))
    parser.add_argument("--output", type=Path, default=Path("experiments/ncc/results/full"))
    parser.add_argument("--methods", nargs="+", choices=PLANNERS, default=list(PLANNERS))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--n-init", type=int, default=80)
    parser.add_argument("--nn-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.trials, args.rounds, args.eval_size, args.n_init, args.nn_epochs) < 1:
        parser.error("Trial, round, evaluation, initialization and epoch counts must be positive.")
    if args.pool_size < args.n_init + args.rounds * PLANNER.horizon:
        parser.error("Pool size must accommodate initialization and all queried rows.")
    if not args.data.is_file():
        parser.error("Missing dataset. Run python -m experiments.ncc.generate or pass --data FILE.")
    data = load_dataset(args.data)
    if len(data[0]) < args.pool_size + args.eval_size:
        parser.error("Dataset must contain at least pool-size + eval-size rows.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty. Pass --output with a new directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_hash = hashlib.sha256()
    with args.data.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            dataset_hash.update(block)
    settings = dict(vars(args), data=str(args.data), output=str(args.output), dataset_sha256=dataset_hash.hexdigest(),
                    output_index=data[3], safety_threshold=data[2], planner=asdict(PLANNER),
                    nn=asdict(NNUncertaintyConfig(epochs=args.nn_epochs)), kernel=asdict(KERNEL),
                    noise_std=NOISE_STD, no_feasible_policy="unsafe_fallback")
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", "mean", "std"])
        for method in args.methods:
            values = []
            with (args.output / f"{method}.jsonl").open("w") as log:
                for trial in range(args.trials):
                    history = run_trial(method, trial, args, data)
                    log.write(json.dumps({"trial": trial, "seed": args.seed + trial,
                                          "rounds": history}, allow_nan=False) + "\n")
                    log.flush()
                    values.append([history[-1][key] for key in METRICS])
                    print(f"{method}: trial {trial + 1}/{args.trials}", flush=True)
            for metric, series in zip(METRICS, np.asarray(values).T):
                writer.writerow([method, metric, series.mean(), series.std()])
            handle.flush()
    print(f"Final round means and standard deviations: {args.output / 'summary.csv'}")


if __name__ == "__main__":
    main()
