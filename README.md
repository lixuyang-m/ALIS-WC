# ALIS-WC

> **Asynchronous Logic-Induced Sleep-Wake Coordination**
> for Heterogeneous Multi-Robot Scheduling under Linear Temporal Logic Constraints
>
> Accepted at RSS 2026

![Method Overview](docs/method_overview.png)

## Status

- [x] Paper accepted at RSS 2026
- [x] Method overview and training metrics dashboard
- [ ] Code release (in preparation)
- [ ] Pretrained checkpoints (in preparation)
- [ ] Benchmark suite (in preparation)

## Training Metrics

Live training metrics and per-seed raw curves are available on SwanLab:

🔗 **<https://swanlab.cn/@Monnaloo/ALIS-WC>**

The dashboard contains the two ablation conditions reported in the paper (Fig. 6):

- **ALIS-WC** — RL with progress-based potential shaping (our main method)
- **RL w/o shaping** — sparse-reward baseline

In addition to the makespan and sparse-return curves shown in the paper, the dashboard exposes raw per-seed runs and additional training-time diagnostics.

## Reward Shaping

This release provides **two alternative potential-based reward shaping (PBRS) variants**, both valid under γ=1 and both leaving the optimal policy unchanged by the Ng-Harada-Russell theorem:

1. **Task-completion PBRS** — `ψ(x) = #completed_tasks / #total_tasks ∈ [0,1]`. This is the conceptual form described in the paper, controlled by `REWARD_TASK_COMPLETION_POTENTIAL_WEIGHT` in `parameters.py`.

2. **Elapsed-time PBRS** — `Φ(x) = −t_current(x)`, equivalent to a per-step time penalty `−λ·Δt`. Controlled by `MDP_DENSE_REWARD_WEIGHT`. It additionally provides correct multi-agent handling: when several robots reach decision points simultaneously, only the first agent in the random tie-break order is charged the actual Δt and all subsequent agents within the same epoch receive zero, so that the per-epoch time penalty is counted once (avoiding multi-counting and preserving telescoping).

The two variants are switchable via the corresponding hyperparameters; setting the other to zero disables it. We did **not** further investigate the impact of these two PBRS variants — or of their coefficients — on the final performance; a controlled comparison is a natural extension for users of this codebase.

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

## Coming Soon

The full source code (event-driven simulator, LTL shield, sleep-wake layer, PPO training scripts, baseline implementations, NL-to-LTL translator, and benchmark generation scripts) along with pretrained model checkpoints will be released here shortly.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
