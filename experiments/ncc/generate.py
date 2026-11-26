"""Generate the synthetic 29-input, 30-output APS benchmark used for NCC."""
import argparse
from pathlib import Path

import numpy as np


def make_graph(rng):
    """Grow the notebook's random graph until 15 deep nodes cover all inputs."""
    inputs, constants = 29, rng.uniform(-1, 1, 20)
    base = inputs + len(constants)
    depth = [0] * base
    coverage = [{i} if i < inputs else set() for i in range(base)]
    recipes = []
    qualified = 0
    while qualified < 15:
        op = rng.choice(["sum", "sp", "arith"])
        count = rng.integers(4, 11) + 1 if op == "arith" else rng.integers(2, 9)
        src = rng.choice(len(depth), count, replace=False)
        if op == "arith":
            args = (rng.choice(["+", "-", "*"], count - 1), rng.uniform(0, 1, count))
        else:
            args = None
        recipes.append((op, src, args))
        depth.append(max(depth[i] for i in src) + 1)
        coverage.append(set().union(*(coverage[i] for i in src)))
        qualified += len(coverage[-1]) == inputs and depth[-1] >= 10

    deepest = sorted(range(base, len(depth)), key=depth.__getitem__, reverse=True)
    full = [i for i in deepest if len(coverage[i]) == inputs][:15]
    outputs = full + [i for i in deepest if i not in full][:15]
    rng.shuffle(outputs)
    return constants, recipes, outputs


def evaluate(graph, inputs):
    constants, recipes, outputs = graph
    base = inputs.shape[1] + len(constants)
    values = np.empty((len(inputs), base + len(recipes)), dtype=np.float64)
    values[:, :inputs.shape[1]] = inputs
    values[:, inputs.shape[1]:base] = constants
    for idx, (op, src, args) in enumerate(recipes, base):
        if op == "sum":
            result = values[:, src].sum(axis=1)
        elif op == "sp":
            half = len(src) // 2
            left, right = src[:half], src[half:]
            if len(left) != len(right):
                right = np.append(right, right[0])
            # The original batch generator gives zero for non-broadcastable pairs.
            if len(left) == 1 or len(left) == len(right):
                result = (values[:, left] * values[:, right]).sum(axis=1) / (len(left) + len(right))
            else:
                result = 0.0
        else:
            operators, coefficients = args
            result = values[:, src[0]] * coefficients[0]
            for symbol, node, weight in zip(operators, src[1:], coefficients[1:]):
                operand = values[:, node] * weight
                operation = {"+": np.add, "-": np.subtract, "*": np.multiply}[symbol]
                result = np.clip(operation(result, operand), -1e8, 1e8)
        values[:, idx] = np.clip(result, -1e8, 1e8)
    return values[:, outputs].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/ncc/data/aps.npz"))
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=31, help="Seed for the graph and clean input streams.")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    graph_seed, input_seed = np.random.SeedSequence(args.seed).spawn(2)
    graph = make_graph(np.random.default_rng(graph_seed))
    clean = np.random.default_rng(input_seed).random((args.samples, 29)).astype(np.float32)
    noise_rng = np.random.default_rng(42)
    targets = np.empty((args.samples, 30), dtype=np.float32)
    for start in range(0, len(clean), 10_000):
        batch = clean[start:start + 10_000].copy()
        batch[:, 0] += noise_rng.normal(0, 0.05, len(batch)).astype(np.float32)
        targets[start:start + len(batch)] = evaluate(graph, batch)
    x_mean, x_std = clean.mean(axis=0), clean.std(axis=0) + 1e-8
    y_mean, y_std = targets.mean(axis=0), targets.std(axis=0) + 1e-8
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, X_raw=clean, Y_raw=targets,
                            X_scaled=(clean - x_mean) / x_std, Y_scaled=(targets - y_mean) / y_std,
                            x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                            noise_node_idx=0, noise_std=0.05, graph_input_seed=args.seed, noise_seed=42)
    print(f"Saved {args.samples} rows to {args.output}")


if __name__ == "__main__":
    main()
