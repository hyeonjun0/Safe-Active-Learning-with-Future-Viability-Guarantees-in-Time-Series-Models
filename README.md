# Safe Active Learning with Future Viability Guarantees in Time-Series Models

For the project report, follow the [NCC experiment guide](experiments/ncc/README.md) to generate the synthetic data, compare FA-SAL (NN) with SAL-NX and Tebbe-ABM, and read the results.

## Paper experiments

Run the scripts from the repository root.

| Paper table | Command |
| --- | --- |
| Table 1 | `bash scripts/run_base.sh` |
| Table 2 | `bash scripts/run_penalty_stress.sh` |
| Table 3 | `bash scripts/run_rail_pressure.sh` |
| Table 4 | `bash scripts/run_ngsim.sh` |

<details>
<summary>Additional experiments referenced in the paper reviews</summary>

The scripts below are in `scripts/`.

**Reviewer JLLo**

Q1: run_penalty_stress_m4_extended_lf_sweep.sh, run_penalty_stress_m8_extended_lf_sweep.sh

Q3: run_fasal_al_mpc_horizon.sh

---

**Reviewer PGwo**

Q2: run_fasal_al_mpc_horizon.sh, run_penalty_stress_m1.sh, run_penalty_stress_m2.sh, run_penalty_stress_m4.sh, run_penalty_stress_m8.sh

Q3: run_rail_pressure_lf_sweep.sh

---

**Reviewer 5qN4**

Q2: run_penalty_stress_m1.sh, run_penalty_stress_m2.sh, run_penalty_stress_m4.sh, run_penalty_stress_m8.sh, run_penalty_stress_m4_extended_lf_sweep.sh, run_penalty_stress_m8_extended_lf_sweep.sh

Q5: run_penalty_stress_m4.sh, run_penalty_stress_m8.sh

---

**Reviewer 7V5z**

Q2: run_lag_dimension_sweep.sh

Q4: run_penalty_stress_m4_extended_lf_sweep.sh, run_penalty_stress_m8_extended_lf_sweep.sh

Q13: run_fixed_adaptive_scale.sh

Q14: run_gp_scale_grid.sh

</details>
