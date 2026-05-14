# ALIS-WC

> **Asynchronous Logic-Induced Sleep-Wake Coordination**
> for Heterogeneous Multi-Robot Scheduling under Linear Temporal Logic Constraints
>
> Accepted at RSS 2026

### Demo: Stress Test (Tier 4, 100 Tasks, 36 Robots, 60 LTL Constraints)

ALIS-WC scheduling 100 tasks with 36 heterogeneous robots under 60 hard LTL constraints (30 safety + 30 sequential). Robots blocked by sequential prerequisites are automatically put to sleep (faded with "Zzz") and woken when the prerequisite completes (gold highlight). Sequential constraint arcs transition from gray (waiting) → orange (predecessor done) → green (satisfied). All 100 tasks completed with 100% LTL satisfaction.

![Stress Test Demo](docs/sleep_wake_stress_tier4.mp4)

![Method Overview](docs/method_overview.png)

## Status

- [x] Paper accepted at RSS 2026
- [x] Method overview and training metrics dashboard
- [x] RL training and evaluation code
- [x] Pretrained checkpoints (released as v0.1.0 assets)
- [x] GA / AVNR / Greedy baselines
- [ ] NL-to-LTL translator and benchmark generator suite (in preparation)

## Installation

Tested with Python 3.8 + CUDA 12.1.

```bash
git clone https://github.com/lixuyang-m/ALIS-WC.git
cd ALIS-WC
conda create -n alis-wc python=3.8 -y && conda activate alis-wc
pip install -r requirements.txt
# Optional, only if LTL_ENABLED=True:
pip install torch-geometric
```

## Evaluation

### ALIS-WC (RL policy)

`evaluate_rl.py` reproduces the paper's RL-side numbers. Each paper row averages **three random seeds** (s42, s142, s242).

```bash
# Single seed, paper Tier 1 (no LTL)
python evaluate_rl.py --model-path <ckpt>.pth --tier tier1 --num-envs 30

# Three-seed aggregation (matches paper Table III / IV row exactly)
python evaluate_rl.py \
    --model-template "mdp_d10_noLTL_mhPPO_vNorm_ent0.005_s{seed}.pth" \
    --seeds 42,142,242 \
    --tier tier1 --num-envs 30

# Evaluate on the bundled benchmark JSON (100 random instances)
python evaluate_rl.py --model-path <ckpt>.pth --benchmark-file evaluation_benchmarks.json

# LTL-aware evaluation
python evaluate_rl.py --model-path <ckpt>.pth --tier tier1 --enable-ltl
```

`--tier` selects one of `tier1` … `tier4` and locks instance dimensions to the paper's configuration. Omit it to use `--benchmark-file` instead. Multi-seed mode (`--model-template` + `--seeds`) prints per-seed summaries and the aggregated `mean ± std` across seeds.

### Baselines (GA / AVNR / Greedy)

`evaluate_baselines.py` runs the three optimisation / search baselines on the same paper tiers, through the same LTL-shielded simulator used by the RL evaluation. Each instance has a per-tier time budget (`tier1=30s`, `tier2=60s`, `tier3=120s`, `tier4=240s`).

```bash
# Single baseline, Tier 1, 30 random instances (no LTL)
python evaluate_baselines.py --method ga    --tier tier1 --num-envs 30
python evaluate_baselines.py --method avnr  --tier tier1 --num-envs 30
python evaluate_baselines.py --method greedy --tier tier1 --num-envs 30

# All three baselines in one pass, with LTL clauses enforced
python evaluate_baselines.py --method ga,avnr,greedy --tier tier1 --num-envs 30 --enable-ltl
```

The shared LTL semantics (`--enable-ltl`) matches `evaluate_rl.py`: SAFETY clauses become per-agent forbidden-node masks and SEQUENTIAL clauses become per-task prerequisites, both enforced inside `baseline/simulator.py`. **GA** evaluates every chromosome through the shielded simulator (its search is LTL-aware); **AVNR** uses an analytical surrogate inside its VND and shields only at final-metric extraction; **Greedy** queries the three-mask pipeline at every decision epoch.

`--decide-quantity` mirrors the same default as RL's hardcoded full-load policy: when omitted (default), all baselines load at the agent's full capacity per cargo type, matching the paper. Passing the flag lets GA/AVNR optimise quantity ratios as part of the search.

## Training

```bash
# Reproduce a single ALIS-WC seed (10M steps, ~24 h on single RTX 4090)
python driver.py --manual_seed 42 --ltl_enabled false --execution_mode mdp

# Three-seed reproduction
for s in 42 142 242; do
  python driver.py --manual_seed $s --ltl_enabled false --execution_mode mdp
done

# Smoke test (1000 steps, ~2 min)
python driver.py --max_training_steps 1000 --ltl_enabled false
```

Defaults in `parameters.py` already match the paper config: `MDP_DENSE_REWARD_WEIGHT=10`, `ENTROPY_BETA=0.005`, `USE_MULTI_HEAD_PPO=True`, `USE_VALUE_LOSS_NORMALIZATION=True`, `MAX_TRAINING_STEPS=10_000_000`.

Logs go to a local SwanLab offline run by default (`./swanlog/`). Set `LoggingParams.MODE = "online"` and run `swanlab login` once for cloud sync.

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
