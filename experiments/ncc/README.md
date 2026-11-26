# Synthetic naphtha cracking (NCC) experiment

This experiment compares FA-SAL with a neural network surrogate, SAL-NX, and Tebbe-ABM on synthetic process data with 29 inputs and 30 outputs.

## Run the experiment

Use Python 3.11 or 3.12. From the repository root, install the dependencies, generate the dataset once, and run all three methods:

```sh
python -m pip install -r experiments/ncc/requirements.txt
python -m experiments.ncc.generate
python -m experiments.ncc.run
```

The default run uses 10 trials of 150 rounds per method. To use an existing APS dataset, skip the generation step and pass its path with `--data`:

```sh
python -m experiments.ncc.run --data /path/to/APS_gaussian_xbatch_noise_data.npz
```

## Read the results

Open `experiments/ncc/results/full/summary.csv` for the mean and standard deviation across trials of three final-round metrics: prediction error (RMSE), safety violation rate (SVR), and safe-set overlap (IoU). For another run, select a fresh directory with `--output`.

The generator uses a fixed seed for new runs.
