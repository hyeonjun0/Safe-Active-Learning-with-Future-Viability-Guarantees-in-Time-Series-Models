# Safe Active Learning with Future Viability Guarantees in Time-Series Models

## Synthetic NCC experiment

Run these commands from the repository root. See the [NCC guide](experiments/ncc/README.md) for dependencies and full run instructions.

| Command | Output |
| --- | --- |
| `python -m experiments.ncc.generate` | Synthetic naphtha cracking data with 29 inputs and 30 outputs. |
| `python -m experiments.ncc.run` | RMSE, safety violation rate, and safe-set IoU for FA-SAL (NN), SAL-NX, and Tebbe-ABM. |

## Paper benchmarks

Run each shell script with `bash scripts/<filename>`.

| Script | Experiment |
| --- | --- |
| [run_base.sh](scripts/run_base.sh) | Synthetic process under the base conditions (Table 1). |
| [run_penalty_stress.sh](scripts/run_penalty_stress.sh) | Synthetic process with a larger penalty for entering a dead end (Table 2). |
| [run_rail_pressure.sh](scripts/run_rail_pressure.sh) | Rail-pressure control benchmark (Table 3). |
| [run_ngsim.sh](scripts/run_ngsim.sh) | Car-following benchmark using NGSIM vehicle trajectories (Table 4). |

## Additional experiments

| Script | Experiment |
| --- | --- |
| [run_penalty_stress_m1.sh](scripts/run_penalty_stress_m1.sh), [m2](scripts/run_penalty_stress_m2.sh), [m4](scripts/run_penalty_stress_m4.sh), [m8](scripts/run_penalty_stress_m8.sh) | FA-SAL continuation policies and baselines at safety horizons of 1, 2, 4, and 8 steps, respectively. |
| [run_penalty_stress_m1_lf_sweep.sh](scripts/run_penalty_stress_m1_lf_sweep.sh), [m2](scripts/run_penalty_stress_m2_lf_sweep.sh) | Sensitivity to the dynamics Lipschitz scale on the penalty-stress benchmark at horizons 1 and 2. |
| [run_penalty_stress_m4_extended_lf_sweep.sh](scripts/run_penalty_stress_m4_extended_lf_sweep.sh), [m8](scripts/run_penalty_stress_m8_extended_lf_sweep.sh) | A wider sweep of the same scale at horizons 4 and 8. |
| [run_fasal_al_mpc_horizon.sh](scripts/run_fasal_al_mpc_horizon.sh) | FA-SAL versus active-learning MPC across planning horizons. |
| [run_rail_pressure_lf_sweep.sh](scripts/run_rail_pressure_lf_sweep.sh) | Sensitivity to the dynamics Lipschitz scale on the rail-pressure benchmark. |
| [run_lag_dimension_sweep.sh](scripts/run_lag_dimension_sweep.sh) | Effect of the number of past inputs and outputs on FA-SAL performance. |
| [run_fixed_adaptive_scale.sh](scripts/run_fixed_adaptive_scale.sh) | Fixed versus adaptive GP parameters and Lipschitz estimates at different initial scales. |
| [run_gp_scale_grid.sh](scripts/run_gp_scale_grid.sh) | Joint sensitivity to the two Lipschitz scale factors, each varied over 0.5, 1, and 2. |
