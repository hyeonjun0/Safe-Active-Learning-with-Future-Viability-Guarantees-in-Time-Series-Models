# Synthetic NCC (APS) experiment

This benchmark compares FA-SAL with a neural surrogate against the GP versions
of SAL-NX and Tebbe-ABM. The runner imports the trajectory planners from the
repository's `planning.py` and both GP models from `gp_models.py`.
The FA-SAL surrogate fits separate PyTorch networks for the target and the
safety margin. Each predicts a mean and log variance and warm-starts on the
growing training set.

From the repository root, in a Python 3.11 or 3.12 environment:

```sh
python -m pip install -r experiments/ncc/requirements.txt
python -m experiments.ncc.generate
python -m experiments.ncc.run
```

The generator writes `experiments/ncc/data/aps.npz`. To use an existing APS
dataset, skip generation and pass its path to the runner:

```sh
python -m experiments.ncc.run --data /path/to/APS_gaussian_xbatch_noise_data.npz
```

For a quick check with all three methods:

```sh
python -m experiments.ncc.generate --samples 256 --output experiments/ncc/data/small.npz
python -m experiments.ncc.run --data experiments/ncc/data/small.npz \
  --output experiments/ncc/results/smoke --rounds 3 --trials 2 \
  --pool-size 40 --eval-size 32 --n-init 12 --nn-epochs 2
```

## Data and protocol

- The default generator samples 100,000 clean 29-dimensional inputs. It
  perturbs input 0 with Gaussian noise (standard deviation 0.05) when
  computing the 30 outputs. Graph and input seeds are fixed for new runs.
  The original notebook did not seed these two draws. The original data and
  trial logs for slide 11 are unavailable, so the regenerated benchmark
  does not establish the reported numerical values.
- The runner reads `X_scaled` and `Y_scaled` from the NPZ. It predicts the
  column indexed by `noise_node_idx` (default 0), and labels values above
  its dataset-wide 80th percentile unsafe.
- Each of 10 trials uses 512 held-out rows and a disjoint pool of 1,000,
  starting with 80 safe observations. Methods share splits and initial points
  for each trial. Each round selects a window of 4 consecutive available pool
  rows in dataset-index order. If no window passes the planner's certificate,
  the original APS protocol queries the window with the strongest pointwise
  safety bound and counts any violations. The default run has 150 rounds.
- FA-SAL uses two hidden layers of width 64, 40 epochs per fit, and
  `L_ell=0.75`, `beta_f=beta_g=2`. SAL-NX and Tebbe-ABM use `alpha=0.2`.
  Run `python -m experiments.ncc.run --help` for CLI options.

## Results

By default each run writes to `experiments/ncc/results/full`. Use a fresh
`--output` directory for each run. `settings.json` records the configuration,
threshold and dataset SHA-256. Each method's JSONL contains one record per
trial with the metrics, queried row indices and fallback flags for every
round. `summary.csv` contains the final-round mean and population standard
deviation across trials for the three report metrics:

| Metric | Definition |
| --- | --- |
| RMSE | Held-out prediction error in standardized target units |
| SVR | Unsafe queried rows divided by all queried rows through the current round |
| Safety IoU | Intersection over union of true and certified safe held-out rows |

As in the local APS experiment, RMSE and IoU use the model fitted at the start
of each round. SVR includes the queries made in that round.
