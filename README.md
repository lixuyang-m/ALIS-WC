# ALIS-WC

> **Asynchronous Logic-Induced Sleep-Wake Coordination**
> for Heterogeneous Multi-Robot Scheduling under Linear Temporal Logic Constraints
>
> Accepted at RSS 2026

![Method Overview](docs/method_overview.png)

## Status

- [x] Paper accepted at RSS 2026
- [x] Method overview and training metrics dashboard
- [x] RL training and evaluation code
- [x] Pretrained checkpoints (released as v0.1.0 assets)
- [ ] GA / AVNR baselines (in preparation)
- [ ] NL-to-LTL translator and benchmark generator suite (in preparation)

## Training Metrics

Live training metrics and per-seed raw curves are available on SwanLab:

🔗 **<https://swanlab.cn/@Monnaloo/ALIS-WC>**

The dashboard contains the two ablation conditions reported in the paper (Fig. 6):

- **ALIS-WC** — RL with progress-based potential shaping (our main method)
- **RL w/o shaping** — sparse-reward baseline

In addition to the makespan and sparse-return curves shown in the paper, the dashboard exposes raw per-seed runs and additional training-time diagnostics.

## Reward Shaping

Two alternative potential-based reward shaping (PBRS) variants are supported under γ=1; both leave the optimal policy unchanged by the Ng-Harada-Russell theorem. Either can be activated by setting its weight to a non-zero value.

| Variant | Potential function | Controlled by | Notes |
|---|---|---|---|
| **Task-completion PBRS** | `ψ(x) = #completed_tasks / #total_tasks ∈ [0,1]` | `REWARD_TASK_COMPLETION_POTENTIAL_WEIGHT` | The conceptual form described in the paper. |
| **Elapsed-time PBRS** | `Φ(x) = −t_current(x)`, per-step penalty `−λ·Δt` | `MDP_DENSE_REWARD_WEIGHT` | Multi-agent safe: only the first robot per simultaneous decision epoch is charged the actual Δt, preserving telescoping. |

> We did **not** further investigate the impact of these variants — or of their coefficients — on the final performance; a controlled comparison is a natural extension for users of this codebase.

## Notes on Retained Code Paths

Beyond the configuration that produced the reported results, several alternative code paths from our development are kept in this release. They are **disabled by default** and do not affect any default training or evaluation run.

| Component | Activated by | Default | Description |
|---|---|---|---|
| **SMDP execution mode** | `TrainParams.EXECUTION_MODE = 'smdp'` | `'mdp'` | Time-dependent discount `γ(τ) = exp(−β·τ)`; alternative to the paper's undiscounted MDP. |
| **LTL encoding variants** | `TrainParams.LTL_ENCODING_TYPE ∈ {'A','B','C'}` | `'C'` | Three different schemes for encoding LTL clauses into the policy state; the paper uses `'C'`. |
| **Soft LTL constraints** | `TrainParams.LTL_CONSTRAINT_TYPE ∈ {'SOFT_POLICY','SOFT_DISCRETE','LTL_POTENTIAL',…}` | `'LTL_POTENTIAL'` | Lagrangian / CVaR / risk-sensitive alternatives to hard shielding (`'HARD'`); activating them requires defining additional hyperparameters not exposed in the default `parameters.py`. |
| **Vehicle failure modeling** | `EnvParams.VEHICLE_FAILURE_ENABLED = True` | `False` | Per-agent Weibull failure modeling and hazard accounting; not used in the paper. |
| **Cost-quantile critic head** | (always present) | inert | `AttentionNet.cost_critic` output retained for legacy checkpoint compatibility; not consumed by any loss in the default configuration. |

> Users wishing to revive any of these branches should expect to re-define the missing hyperparameters and validate end-to-end correctness; we did not maintain these paths after the paper's frozen configuration.

## Citation

```bibtex
@inproceedings{li2026aliswc,
  author    = {Xuyang Li and Leilei Li and Jianwu Fang and Boyuan Chen and Jianru Xue},
  title     = {Event-Driven Sleep-Wake Scheduling for Heterogeneous Robots under {LTL} Constraints},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2026}
}
```

(Final BibTeX will be updated after camera-ready DOI is assigned.)

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
