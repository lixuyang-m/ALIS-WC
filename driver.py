import torch
import torch.optim as optim
import torch.nn.functional as F
import ray
import os
import time
import numpy as np
import random

from parameters import *

# 动态导入日志系统（支持wandb和swanlab）
if LoggingParams.BACKEND == "swanlab":
    import swanlab

    logger_module = swanlab
elif LoggingParams.BACKEND == "wandb":
    import wandb

    logger_module = wandb
else:
    raise ValueError(f"不支持的日志后端: {LoggingParams.BACKEND}")

from async_evaluator import AsyncEvaluator
import json
import argparse

from attention import AttentionNet
from runner import RLRunner
from env.task_env import TaskEnv


# ==================== Running Statistics for Value Loss Normalization ====================
class RunningStats:
    """
    维护returns的running mean和std，用于value loss normalization

    理论依据：
        L_V_normalized = E[(V - R)²] / (σ_R + ε)²

    这将value loss的梯度按returns的尺度自动缩放，确保：
        1. 与reward scale无关（LTL开关改变reward不需要重新调参）
        2. policy loss和value loss的梯度在同一数量级
        3. 最优解不变（仍然是 V → R）

    Args:
        decay: EMA衰减系数（默认0.99，越大越平滑）
        device: torch设备
    """

    def __init__(self, decay=0.99, device="cpu"):
        self.decay = decay
        self.device = device
        self.running_mean = torch.tensor(0.0, device=device)
        self.running_var = torch.tensor(1.0, device=device)  # 初始化为1避免除零
        self.count = 0

    def update(self, batch_data):
        """
        使用新的batch数据更新running statistics

        Args:
            batch_data: Tensor [N], 当前batch的returns
        """
        batch_mean = batch_data.mean()
        batch_var = batch_data.var(unbiased=False)  # 使用N而非N-1（与std()一致）

        if self.count == 0:
            # 第一次更新：直接使用batch统计量
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            # EMA更新
            self.running_mean = (
                self.decay * self.running_mean + (1 - self.decay) * batch_mean
            )
            self.running_var = (
                self.decay * self.running_var + (1 - self.decay) * batch_var
            )

        self.count += 1

    def get_std(self):
        """返回当前的running std"""
        return torch.sqrt(self.running_var + 1e-8)

    def get_mean(self):
        """返回当前的running mean（可选使用）"""
        return self.running_mean

    def reset(self):
        """重置统计量"""
        self.running_mean = torch.tensor(0.0, device=self.device)
        self.running_var = torch.tensor(1.0, device=self.device)
        self.count = 0


# ========================================================================================


def set_random_seeds(seed):
    """
    设置所有随机种子以确保实验可复现
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 只在需要使用GPU时设置CUDA随机种子（避免不必要的CUDA初始化）
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    except:
        # CUDA不可用时跳过
        pass

    print(f"随机种子已设置为: {seed}")


def get_entropy_beta(total_env_steps, training_step):
    if not getattr(TrainParams, "ENTROPY_ANNEALING", False):
        return TrainParams.ENTROPY_BETA

    beta_start = getattr(TrainParams, "ENTROPY_BETA_START", TrainParams.ENTROPY_BETA)
    beta_end = getattr(TrainParams, "ENTROPY_BETA_END", 0.0)

    total_steps = getattr(TrainParams, "ENTROPY_ANNEALING_TOTAL_STEPS", None)
    if total_steps is None:
        total_steps = getattr(TrainParams, "MAX_TRAINING_STEPS", None)

    if total_steps is None or total_steps <= 0:
        return beta_start

    progress = float(total_env_steps) / float(total_steps)
    if progress < 0.0:
        progress = 0.0
    elif progress > 1.0:
        progress = 1.0

    return beta_start + (beta_end - beta_start) * progress


def _actions_to_tensor(actions_list, device):
    """
    把 list[T] 的动作（dict 或 tuple/list）转换成 [T,4] 的 LongTensor，
    列顺序: [type, destination, cargo, quantity]。
    """
    a_types, a_dests, a_cargos, a_quantities = [], [], [], []
    for a in actions_list:
        if isinstance(a, dict):
            a_types.append(int(a.get("type", 0)))
            a_dests.append(int(a.get("destination", 0)))
            a_cargos.append(int(a.get("cargo", 0)))
            a_quantities.append(int(a.get("quantity", 0)))
        else:
            # tuple/list
            a_types.append(int(a[0]) if len(a) > 0 else 0)
            a_dests.append(int(a[1]) if len(a) > 1 else 0)
            a_cargos.append(int(a[2]) if len(a) > 2 else 0)
            a_quantities.append(int(a[3]) if len(a) > 3 else 0)
    arr = list(zip(a_types, a_dests, a_cargos, a_quantities))
    return torch.tensor(arr, dtype=torch.long, device=device)  # [T,4]


def _normalize_eval_actions_output(out):
    """
    【CVAR_SMDP Quantile】evaluate_actions现在返回4个值：
    (log_probs, entropy, reward_value, cost_quantiles)
    """
    if isinstance(out, dict):
        # 尝试常见key
        lp = out.get("logp", out.get("log_prob", out.get("log_probs", None)))
        ent = out.get("entropy", out.get("ent", None))
        if lp is None:
            raise RuntimeError(
                "evaluate_actions returned a dict but no 'logp/log_prob/log_probs' key found."
            )
        lp = lp
        if ent is None:
            ent = torch.zeros_like(lp)
        # Dict模式暂不支持cost_quantiles，返回None
        return lp, ent, None, None

    # 元组/列表形式（标准返回格式）
    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise RuntimeError("evaluate_actions returned empty tuple/list.")

        lp = out[0]  # log_probs
        ent = None
        val = None  # reward_value
        cost_q = None  # cost_quantiles

        if len(out) >= 2 and torch.is_tensor(out[1]):
            ent = out[1]
        if len(out) >= 3 and torch.is_tensor(out[2]):
            val = out[2]
        if len(out) >= 4 and torch.is_tensor(out[3]):  # <-- 新增：cost_quantiles
            cost_q = out[3]

        # 填充默认值
        if ent is None:
            ent = torch.zeros_like(lp)
        if val is None:
            val = torch.zeros_like(lp)
        if cost_q is None:
            # cost_quantiles默认为零矩阵 [B, NUM_QUANTILES]
            cost_q = torch.zeros(
                (lp.shape[0], TrainParams.NUM_QUANTILES), device=lp.device
            )

        return lp, ent, val, cost_q  # <-- 修改返回值：4个

    # 单个张量（仅 logp）
    if torch.is_tensor(out):
        B = out.shape[0]
        return (
            out,
            torch.zeros_like(out),
            torch.zeros_like(out),
            torch.zeros((B, TrainParams.NUM_QUANTILES), device=out.device),
        )

    # 其它未知类型
    raise RuntimeError(f"evaluate_actions returned unsupported type: {type(out)}")


def _oldlogp_to_tensor(vals, device):
    """
    兼容 old_log_prob 的多种缓存形式：
    - list[Tensor]（每步一个张量）
    - list[float] / list[int]
    - 已是 Tensor
    返回 [T] float32 tensor
    """
    if torch.is_tensor(vals):
        return vals.to(device=device, dtype=torch.float32).reshape(-1)
    if isinstance(vals, (list, tuple)):
        if len(vals) == 0:
            return torch.zeros(0, device=device, dtype=torch.float32)
        if torch.is_tensor(vals[0]):
            return (
                torch.stack(
                    [v.reshape(1).detach() if v.ndim > 0 else v.detach() for v in vals]
                )
                .to(device=device, dtype=torch.float32)
                .reshape(-1)
            )
        else:
            return torch.tensor(vals, dtype=torch.float32, device=device).reshape(-1)
    # 其它单值
    return torch.tensor([float(vals)], dtype=torch.float32, device=device).reshape(-1)


def _pop_unroll_batch_from_buffer(experience_buffer, T, max_unrolls):
    N = len(experience_buffer["task_info"])
    if N == 0:
        return [], 0

    dones = experience_buffer["dones"]  # list[bool]
    # 计算 episode 边界（done=True 的下标）：
    boundaries = [
        i for i, d in enumerate(dones) if d
    ]  # 每个边界 index 都代表“该步之后 episode 结束”
    if not boundaries:
        # 没有 episode 完结；若不足 T*max_unrolls 步，就暂不抽取，避免用“半成品”拼 batch
        if N < T * max_unrolls:
            return [], 0
        # 否则，直接切前 T*max_unrolls 步为连续片段，再分 T 段
        cut_end = T * max_unrolls
        segment_ranges = [(0, cut_end - 1)]
    else:
        # 从最早的数据开始，尽量用完整 episode；若合并后超过 T*max_unrolls，也只取到上限
        segment_ranges = []
        start = 0
        used = 0
        for b in boundaries:
            seg_len = b - start + 1
            if used + seg_len > T * max_unrolls:
                # 只取到上限
                remain = T * max_unrolls - used
                if remain > 0:
                    segment_ranges.append((start, start + remain - 1))
                    used += remain
                break
            else:
                segment_ranges.append((start, b))
                used += seg_len
                start = b + 1
            if used >= T * max_unrolls:
                break
        # 如果没有用满上限而且还有剩余未完结的尾部数据，也不强取（等 episode 结束或积累更多）
        if used == 0:
            return [], 0

    # 现在 segment_ranges 是若干个连续区间，把它们再切成 T 段
    slices = []
    for s, e in segment_ranges:
        L = e - s + 1
        k = 0
        while k < L:
            ss = s + k
            ee = min(ss + T - 1, e)
            slices.append((ss, ee))
            k += ee - ss + 1

    # 最多取 max_unrolls 条
    slices = slices[:max_unrolls]

    # 从 experience_buffer 里抽出这些区间的数据
    keys = list(experience_buffer.keys())
    unrolls = []
    # 注意：我们要“弹出”这些区间，所以先拷贝出数据，再整体从 buffer 删除
    # 为了删除方便，记录需要删除的全局下标
    delete_indices = []
    for ss, ee in slices:
        unroll = {k: experience_buffer[k][ss : ee + 1] for k in keys}
        unrolls.append(unroll)
        delete_indices.extend(list(range(ss, ee + 1)))

    # 真正从 buffer 删除（按从后往前删，避免下标位移）
    delete_indices = sorted(set(delete_indices), reverse=True)
    for idx in delete_indices:
        for k in keys:
            del experience_buffer[k][idx]

    total_steps = sum((ee - ss + 1) for (ss, ee) in slices)
    return unrolls, total_steps


class Logger(object):
    def __init__(self):
        # 动态初始化日志系统（wandb或swanlab）
        # 使用自动生成的实验名称（基于配置参数）
        if LoggingParams.RUN_NAME:
            run_name = LoggingParams.RUN_NAME
        else:
            run_name = LoggingParams.generate_run_name()

        # 存储run_name以供模型保存使用
        self.run_name = run_name

        # 获取奖励权重（根据执行模式）
        if TrainParams.EXECUTION_MODE == "mdp":
            sparse_weight = getattr(TrainParams, "MDP_SPARSE_REWARD_WEIGHT", 1.0)
            dense_weight = getattr(TrainParams, "MDP_DENSE_REWARD_WEIGHT", 1.0)
        else:
            sparse_weight = getattr(TrainParams, "SMDP_SPARSE_REWARD_WEIGHT", 1.0)
            dense_weight = getattr(TrainParams, "SMDP_DENSE_REWARD_WEIGHT", 0.0)

        config = {
            # 基础训练参数
            "lr": TrainParams.LR,
            "batch_size": TrainParams.BATCH_SIZE,
            "num_meta_agents": TrainParams.NUM_META_AGENT,
            "decay_step": TrainParams.DECAY_STEP,
            "evaluate": TrainParams.EVALUATE,
            # 执行模式和折扣
            "execution_mode": TrainParams.EXECUTION_MODE,
            "gamma": TrainParams.GAMMA,
            "beta": TrainParams.BETA,
            "delta_t": TrainParams.DELTA_T,  # SMDP时间单位
            # 奖励权重
            "sparse_reward_weight": sparse_weight,
            "dense_reward_weight": dense_weight,
            # LTL约束配置
            "ltl_enabled": TrainParams.LTL_ENABLED,
            "ltl_encoding_type": TrainParams.LTL_ENCODING_TYPE,
            "ltl_constraint_mode": TrainParams.LTL_CONSTRAINT_TYPE,
            # 算法选择
            "algorithm": TrainParams.ALGORITHM,
        }

        logger_module.init(
            project=LoggingParams.PROJECT_NAME,
            name=run_name,
            mode=LoggingParams.MODE,
            config=config,
        )

        print(f"[日志系统] 使用 {LoggingParams.BACKEND} 进行日志记录")
        print(f"[实验名称] {run_name}")
        # Ensure model and gif directories exist
        if SaverParams.SAVE:
            os.makedirs(SaverParams.MODEL_PATH, exist_ok=True)
            os.makedirs(SaverParams.GIFS_PATH, exist_ok=True)
        self.global_net = None
        self.baseline_net = None
        self.optimizer = None
        self.lr_decay = None

    def set(self, global_net, baseline_net, optimizer, lr_decay):
        self.global_net = global_net
        self.baseline_net = baseline_net
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        # Watch model parameters and gradients
        # 注意：log="all"会产生大量数据，可能导致WandB crash
        # 如果遇到WandB频繁crash，可以：
        # 1. 改为 log="gradients" 只记录梯度
        # 2. 增大 log_freq 到 1000 或更大
        # 3. 完全注释掉这一行
        # wandb.watch(self.global_net, log="all", log_freq=100)  # 已禁用，避免数据量过大
        if LoggingParams.BACKEND == "wandb":
            logger_module.watch(self.global_net, log="gradients", log_freq=1000)
        # swanlab不支持watch功能

    # def run_evaluation(self, current_weights, eval_seeds, meta_agents, training_step):
    #     """
    #     在所有 meta_agent 上分发并执行评估任务。
    #     """
    #     print(f"\n{'=' * 25} Running Evaluation at Training Step {training_step} {'=' * 25}")
    #
    #     weights_ref = ray.put(current_weights)
    #     eval_jobs = []
    #     for i, seed in enumerate(eval_seeds):
    #         # 将评估任务分发给不同的 agent
    #         agent = meta_agents[i % len(meta_agents)]
    #         eval_jobs.append(agent.evaluate.remote(weights_ref, seed))
    #
    #     # 等待所有评估任务完成并收集结果
    #     eval_results = ray.get(eval_jobs)
    #
    #     # 聚合所有评估环境的性能指标
    #     agg_metrics = {k: [] for k in eval_results[0].keys()}
    #     for result in eval_results:
    #         for k, v in result.items():
    #             # v 是一个 list of float，我们取其第一个元素
    #             if v:
    #                 agg_metrics[k].append(v[0])
    #
    #     # 计算平均值
    #     mean_eval_metrics = {f"Eval/{k}": np.nanmean(v) for k, v in agg_metrics.items()}
    #
    #     # 打印关键评估指标
    #     print(f"Evaluation Results (avg. over {len(eval_seeds)} episodes):")
    #     print(f"  - Eval/Makespan: {mean_eval_metrics.get('Eval/makespan', 0):.2f}")
    #     print(f"  - Eval/Success Rate: {mean_eval_metrics.get('Eval/success_rate', 0):.2%}")
    #     print(f"  - Eval/Travel Distance: {mean_eval_metrics.get('Eval/travel_dist', 0):.2f}")
    #     print(f"{'=' * 80}\n")
    #
    #     # 将评估结果记录到 wandb
    #     wandb.log(mean_eval_metrics, step=training_step)

    def run_evaluation(
        self,
        current_weights,
        benchmarks,
        meta_agents,
        training_step,
        total_env_steps=None,
    ):
        """
        在所有 meta_agent 上分发并执行评估任务。

        Args:
            current_weights: 当前模型权重
            benchmarks: 评估基准
            meta_agents: worker列表
            training_step: 训练更新步数
            total_env_steps: 总环境交互步数（PPO使用）
        """
        import time

        eval_start_time = time.time()

        print(f"\n{'=' * 80}")
        print(f"[评估诊断] 开始评估")
        print(f"{'=' * 80}")
        print(f"  - 训练步数: {training_step}")
        if total_env_steps is not None:
            print(f"  - 环境交互步数: {total_env_steps}")
        print(f"  - 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - 评估任务数: {len(benchmarks)}")
        print(f"  - Worker数量: {len(meta_agents)}")
        print(f"  - 训练约束模式: {TrainParams.LTL_CONSTRAINT_TYPE}")
        print(f"  - LTL启用: {TrainParams.LTL_ENABLED}")
        print(f"{'=' * 80}")

        weights_ref = ray.put(current_weights)
        eval_jobs = []

        if not benchmarks:
            print("没有可用的评估基准，跳过评估。")
            return

        eval_results = []

        # 提交所有评估任务
        submission_start = time.time()

        # 【评估约束模式】评估时使用硬约束来严格验证性能
        # 说明：
        # - 评估时使用HARD模式：应用动作掩码，确保不违反LTL约束（安全保障）
        # - 训练时可以使用软约束（SOFT_POLICY等）进行学习
        # - 评估时使用HARD模式验证学习效果（是否能在硬约束下完成任务）
        eval_mode = "HARD" if TrainParams.LTL_ENABLED else None

        for i, benchmark in enumerate(benchmarks):
            agent = meta_agents[i % len(meta_agents)]
            seed = benchmark["seed"]
            ltl_clauses = benchmark["ltl_clauses"]
            eval_jobs.append(
                agent.evaluate.remote(weights_ref, seed, ltl_clauses, eval_mode)
            )
        submission_time = time.time() - submission_start
        print(f"[EVAL] 任务提交完成，耗时 {submission_time:.2f}s")

        # 等待所有评估任务完成并收集结果
        print(f"[EVAL] 等待所有评估任务完成...")
        wait_start = time.time()
        eval_results = ray.get(eval_jobs)
        wait_time = time.time() - wait_start

        total_eval_time = time.time() - eval_start_time
        print(f"\n{'=' * 80}")
        print(f"[评估诊断] 所有worker完成评估")
        print(f"{'=' * 80}")
        print(f"  - 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - 任务提交耗时: {submission_time:.1f}秒")
        print(f"  - 评估执行耗时: {wait_time:.1f}秒 ({wait_time / 60:.2f}分钟)")
        print(f"  - 总耗时: {total_eval_time:.1f}秒 ({total_eval_time / 60:.2f}分钟)")
        print(f"  - 平均每个任务耗时: {wait_time / max(len(benchmarks), 1):.1f}秒")
        print(f"  - 并行效率: {wait_time / (total_eval_time - submission_time):.1%}")
        print(f"{'=' * 80}\n")

        # 聚合所有评估环境的性能指标
        agg_metrics = {k: [] for k in eval_results[0].keys()}
        for result in eval_results:
            for k, v in result.items():
                # v 是一个 list of float，我们取其第一个元素
                if v:
                    agg_metrics[k].append(v[0])

        # 计算统计指标：平均值、标准差、最小值、最大值
        mean_eval_metrics = {f"Eval/{k}": np.nanmean(v) for k, v in agg_metrics.items()}
        std_eval_metrics = {
            f"Eval_std/{k}": np.nanstd(v) for k, v in agg_metrics.items()
        }
        min_eval_metrics = {
            f"Eval_min/{k}": np.nanmin(v) for k, v in agg_metrics.items()
        }
        max_eval_metrics = {
            f"Eval_max/{k}": np.nanmax(v) for k, v in agg_metrics.items()
        }

        # 打印详细的评估结果
        print(f"\n{'=' * 80}")
        print(f"[评估诊断] 聚合结果 (统计自 {len(benchmarks)} 个评估任务)")
        print(f"{'=' * 80}")
        print(f"  - 训练约束模式: {TrainParams.LTL_CONSTRAINT_TYPE}")
        print(
            f"  - 评估约束模式: {eval_mode if eval_mode else TrainParams.LTL_CONSTRAINT_TYPE}"
        )
        print(f"  - 评估样本数: {len(benchmarks)}")
        print(f"{'=' * 80}")

        # 辅助函数：打印带有统计信息的指标
        def print_metric(name, key, format_str=".2f", percentage=False):
            mean_val = mean_eval_metrics.get(f"Eval/{key}", 0)
            std_val = std_eval_metrics.get(f"Eval_std/{key}", 0)
            min_val = min_eval_metrics.get(f"Eval_min/{key}", 0)
            max_val = max_eval_metrics.get(f"Eval_max/{key}", 0)

            if percentage:
                print(
                    f"  - {name:20s}: {mean_val:.2%} (±{std_val:.2%}) [min: {min_val:.2%}, max: {max_val:.2%}]"
                )
            else:
                print(
                    f"  - {name:20s}: {mean_val:{format_str}} (±{std_val:{format_str}}) [min: {min_val:{format_str}}, max: {max_val:{format_str}}]"
                )

        # 基础性能指标
        print(f"\n【基础性能指标】")
        print_metric("Makespan", "makespan", ".2f")
        print_metric("Success Rate", "success_rate", percentage=True)
        print_metric("Episode Return", "episode_return", ".2f")

        # 【故障统计指标】
        if "actual_failure_rate" in agg_metrics:
            print(f"\n【故障统计指标】")
            print_metric("Actual Failure Rate", "actual_failure_rate", percentage=True)
            print_metric("Avg Failure Prob", "avg_failure_prob", percentage=True)
            print_metric("Failed Agents Count", "failed_agents_count", ".2f")
            print_metric("Total Agents", "num_agents", ".1f")

        # 打印Episode Return的分项
        if "episode_shaping_reward" in agg_metrics:
            shaping = np.nanmean(agg_metrics["episode_shaping_reward"])
            time_pen = np.nanmean(agg_metrics["episode_time_penalty"])
            terminal_pen = np.nanmean(agg_metrics["episode_terminal_penalty"])
            print(f"    ├─ Shaping Reward:   {shaping:>8.2f}")
            print(f"    ├─ Time Penalty:     {time_pen:>8.2f}")
            print(f"    └─ Terminal Penalty: {terminal_pen:>8.2f}")

        print_metric("Travel Distance", "travel_dist", ".2f")
        print_metric("Efficiency", "efficiency", ".4f")
        print_metric("Waiting Time", "waiting_time", ".2f")

        # LTL约束相关指标
        if TrainParams.LTL_ENABLED:
            print(f"\n【LTL约束指标】")
            print_metric("Num Clauses", "ltl_num_clauses", ".1f")
            print_metric(
                "Overall Satisfaction", "ltl_overall_satisfaction_rate", percentage=True
            )
            print_metric(
                "Safety Violation", "ltl_safety_violation_rate", percentage=True
            )
            print_metric(
                "Sequential Satisfaction",
                "ltl_sequential_satisfaction_rate",
                percentage=True,
            )

            # 软约束方法的额外指标
            if TrainParams.LTL_CONSTRAINT_TYPE != "HARD":
                print(f"\n【软约束Cost指标】")
                print_metric("Cost per Step", "ltl_cost_per_step", ".4f")
                print_metric("Total Cost", "ltl_total_cost", ".2f")
                print_metric("Max Cost", "ltl_max_cost", ".4f")

        # ==================== Comprehensive Metrics（评估阶段）====================
        # 【修改】NEU和RISK模式均打印comprehensive metrics用于对比
        if EnvParams.VEHICLE_FAILURE_ENABLED:
            # 检查是否有comprehensive metrics（从第一个结果中检查）
            has_comprehensive = "C_rel" in agg_metrics or "total_work_W" in agg_metrics

            if has_comprehensive:
                print(f"\n【Comprehensive Metrics - 理论验证指标】")
                # 核心优化目标
                if "C_rel" in agg_metrics:
                    print_metric("C_rel (Relative Excess)", "C_rel", ".4f")
                if "makespan_T" in agg_metrics:
                    print_metric("T (Makespan)", "makespan_T", ".2f")

                # 理论验证指标
                if "total_work_W" in agg_metrics:
                    print_metric("W (Total Work)", "total_work_W", ".2f")
                if "H_actual" in agg_metrics:
                    print_metric("H_actual", "H_actual", ".4f")
                if "H_optimal" in agg_metrics:
                    print_metric("H_optimal", "H_optimal", ".4f")
                if "C_abs" in agg_metrics:
                    print_metric("C_abs (Absolute Excess)", "C_abs", ".4f")

                # 负载均衡指标
                print(f"\n【Comprehensive Metrics - 负载均衡指标】")
                if "work_std" in agg_metrics:
                    print_metric("Work Std", "work_std", ".2f")
                if "work_cv" in agg_metrics:
                    print_metric("Work CV", "work_cv", ".4f")
                if "max_mean_ratio" in agg_metrics:
                    print_metric("Max/Mean Ratio", "max_mean_ratio", ".3f")
                if "gini_coeff" in agg_metrics:
                    print_metric("Gini Coeff", "gini_coeff", ".4f")
        # =========================================================================

        print(f"{'=' * 80}\n")

        # 将评估结果记录到 wandb
        # 根据算法类型选择合适的横坐标（与训练日志保持一致）
        if total_env_steps is not None and TrainParams.ALGORITHM == "PPO":
            x_axis_value = total_env_steps
        else:
            x_axis_value = training_step

        # 添加异常处理
        try:
            # 合并所有统计指标（平均值、标准差、最小值、最大值）
            all_metrics = {}
            all_metrics.update(mean_eval_metrics)
            all_metrics.update(std_eval_metrics)
            all_metrics.update(min_eval_metrics)
            all_metrics.update(max_eval_metrics)

            # 过滤NaN和Inf值
            import math

            filtered_metrics = {}
            for k, v in all_metrics.items():
                if isinstance(v, (int, float)):
                    if not (math.isnan(v) or math.isinf(v)):
                        filtered_metrics[k] = v
                    else:
                        filtered_metrics[k] = 0.0
                else:
                    filtered_metrics[k] = v

            logger_module.log(filtered_metrics, step=x_axis_value)
        except Exception as e:
            print(f"[WARNING] WandB评估日志上传失败 (step={x_axis_value}): {e}")
            print(f"[WARNING] 评估将继续，但该步的日志未上传到WandB")

    # def write_to_board(self, tensorboard_data, curr_episode):
    #     data = np.array(tensorboard_data)
    #     mean_vals = np.nanmean(data, axis=0).tolist()
    #     reward, p_l, entropy, grad_norm, success_rate, makespan, time_cost, waiting, distance, efficiency = mean_vals
    #     logs = {
    #         "Loss/Policy Loss": p_l,
    #         "Loss/Entropy": entropy,
    #         "Loss/Grad Norm": grad_norm,
    #         "Perf/Reward": reward,
    #         "Perf/Makespan": makespan,
    #         "Perf/Success Rate": success_rate,
    #         "Perf/Time Cost": time_cost,
    #         "Perf/Waiting Time": waiting,
    #         "Perf/Travel Distance": distance,
    #         "Perf/Waiting Efficiency": efficiency,
    #         "Loss/Learning Rate": self.optimizer.param_groups[0]['lr'],
    #         "episode": curr_episode,
    #     }
    #     print(logs)
    #     wandb.log(logs, step=curr_episode)

    # ==================== MODIFIED: Changed method signature and log dictionary ====================
    def write_to_board(
        self, tensorboard_data, training_step, total_env_steps, mean_raw_reward
    ):
        # data = np.array(tensorboard_data)
        # mean_vals = np.nanmean(data, axis=0).tolist()
        #
        # reward, p_l, entropy, grad_norm, success_rate, makespan, time_cost, waiting, distance, efficiency, episode_return, ent_type, mag_type, pmax_type, ent_dest, mag_dest, pchosen_dest, ent_cargo, mag_cargo, pchosen_cargo = mean_vals
        #
        # logs = {
        #     "Loss/Policy Loss": p_l,
        #     "Loss/Entropy": entropy,
        #     "Loss/Grad Norm": grad_norm,
        #     "Perf/Value_Target": reward,
        #     "Perf/Makespan": makespan,
        #     "Perf/Success Rate": success_rate,
        #     "Perf/Time Cost": time_cost,
        #     "Perf/Waiting Time": waiting,
        #     "Perf/Travel Distance": distance,
        #     "Perf/Waiting Efficiency": efficiency,
        #     "Loss/Learning Rate": self.optimizer.param_groups[0]['lr'],
        #     "Global/training_step": training_step,  # 将 training_step 也作为一个指标记录下来
        #     "Perf/Episode_Return": episode_return,
        #     "Perf/Raw_Reward_Mean": mean_raw_reward,
        #
        #     "PolicyHead/Entropy_ActionType": ent_type,
        #     "PolicyHead/LogitsMag_ActionType": mag_type,
        #     "PolicyHead/ProbMax_ActionType": pmax_type,
        #
        #     "PolicyHead/Entropy_Destination": ent_dest,
        #     "PolicyHead/LogitsMag_Destination": mag_dest,
        #     "PolicyHead/ProbChosen_Destination": pchosen_dest,
        #
        #     "PolicyHead/Entropy_Cargo": ent_cargo,
        #     "PolicyHead/LogitsMag_Cargo": mag_cargo,
        #     "PolicyHead/ProbChosen_Cargo": pchosen_cargo,
        # }
        # print(logs)
        # wandb.log(logs, step=training_step)  # 使用 training_step作为横坐标

        data = np.array(tensorboard_data, dtype=float)
        mean_vals = np.nanmean(data, axis=0).tolist()

        # 解包训练指标
        # 基础指标（所有模式共有）：[reward, p_l, entropy, grad_norm]
        reward = mean_vals[0]
        p_l = mean_vals[1]
        entropy = mean_vals[2]
        grad_norm = mean_vals[3]

        # 根据LTL_CONSTRAINT_TYPE解包后续指标
        if TrainParams.LTL_CONSTRAINT_TYPE == "RISK_SENSITIVE_SMDP":
            # RISK_SENSITIVE_SMDP模式：extend了19个值（11个原有 + 8个comprehensive metrics）
            # 【原有11个】[C̄_mean, logmgf, makespan, risk_loss, value_loss, entropy_loss,
            #              C̄_std, C̄_min, C̄_max, weight_ratio, ess_ratio]
            # 【新增8个】[W_mean, H_actual_mean, H_optimal_mean, C_abs_mean,
            #            work_std_mean, work_cv_mean, max_mean_ratio_mean, gini_coeff_mean]
            mean_cost = mean_vals[4] if len(mean_vals) > 4 else 0.0
            lambda_val = mean_vals[5] if len(mean_vals) > 5 else 0.0  # logmgf
            makespan_train = mean_vals[6] if len(mean_vals) > 6 else 0.0
            risk_loss_val = mean_vals[7] if len(mean_vals) > 7 else 0.0
            value_loss_val = mean_vals[8] if len(mean_vals) > 8 else 0.0
            entropy_loss_val = mean_vals[9] if len(mean_vals) > 9 else 0.0
            c_bar_std = mean_vals[10] if len(mean_vals) > 10 else 0.0
            c_bar_min = mean_vals[11] if len(mean_vals) > 11 else 0.0
            c_bar_max = mean_vals[12] if len(mean_vals) > 12 else 0.0
            weight_ratio = mean_vals[13] if len(mean_vals) > 13 else 0.0
            ess_ratio = mean_vals[14] if len(mean_vals) > 14 else 0.0

            # 【Comprehensive Metrics】新增8个指标
            W_mean = mean_vals[15] if len(mean_vals) > 15 else 0.0
            H_actual_mean = mean_vals[16] if len(mean_vals) > 16 else 0.0
            H_optimal_mean = mean_vals[17] if len(mean_vals) > 17 else 0.0
            C_abs_mean = mean_vals[18] if len(mean_vals) > 18 else 0.0
            work_std_mean = mean_vals[19] if len(mean_vals) > 19 else 0.0
            work_cv_mean = mean_vals[20] if len(mean_vals) > 20 else 0.0
            max_mean_ratio_mean = mean_vals[21] if len(mean_vals) > 21 else 0.0
            gini_coeff_mean = mean_vals[22] if len(mean_vals) > 22 else 0.0

            # 【PPO诊断指标】新增5个指标
            explained_variance = mean_vals[23] if len(mean_vals) > 23 else 0.0
            advantages_mean = mean_vals[24] if len(mean_vals) > 24 else 0.0
            advantages_std = mean_vals[25] if len(mean_vals) > 25 else 0.0
            kl_divergence = mean_vals[26] if len(mean_vals) > 26 else 0.0
            clip_fraction = mean_vals[27] if len(mean_vals) > 27 else 0.0

            # 【分头熵监控】新增4个指标
            entropy_type = mean_vals[28] if len(mean_vals) > 28 else 0.0
            entropy_dest = mean_vals[29] if len(mean_vals) > 29 else 0.0
            entropy_cargo = mean_vals[30] if len(mean_vals) > 30 else 0.0
            entropy_quantity = mean_vals[31] if len(mean_vals) > 31 else 0.0

            idx = 32  # 下一个索引起点（4个基础 + 19个Risk-Sensitive + 5个PPO诊断 + 4个分头熵 = 32）
        else:
            # 【所有非RISK_SENSITIVE_SMDP模式】包括HARD、LTL_POTENTIAL、CVAR_SMDP等，都extend了24个值（19个原有+5个PPO诊断）
            if EnvParams.VEHICLE_FAILURE_ENABLED:
                # 启用故障模式：extend了24个值（与RISK模式相同，只是前11个Risk指标填0，后5个是PPO诊断）
                mean_cost = mean_vals[4] if len(mean_vals) > 4 else 0.0
                lambda_val = mean_vals[5] if len(mean_vals) > 5 else 0.0
                makespan_train = mean_vals[6] if len(mean_vals) > 6 else 0.0
                risk_loss_val = mean_vals[7] if len(mean_vals) > 7 else 0.0
                value_loss_val = mean_vals[8] if len(mean_vals) > 8 else 0.0
                entropy_loss_val = mean_vals[9] if len(mean_vals) > 9 else 0.0
                c_bar_std = mean_vals[10] if len(mean_vals) > 10 else 0.0
                c_bar_min = mean_vals[11] if len(mean_vals) > 11 else 0.0
                c_bar_max = mean_vals[12] if len(mean_vals) > 12 else 0.0
                weight_ratio = mean_vals[13] if len(mean_vals) > 13 else 0.0
                ess_ratio = mean_vals[14] if len(mean_vals) > 14 else 0.0

                # Comprehensive metrics（真实值）
                W_mean = mean_vals[15] if len(mean_vals) > 15 else 0.0
                H_actual_mean = mean_vals[16] if len(mean_vals) > 16 else 0.0
                H_optimal_mean = mean_vals[17] if len(mean_vals) > 17 else 0.0
                C_abs_mean = mean_vals[18] if len(mean_vals) > 18 else 0.0
                work_std_mean = mean_vals[19] if len(mean_vals) > 19 else 0.0
                work_cv_mean = mean_vals[20] if len(mean_vals) > 20 else 0.0
                max_mean_ratio_mean = mean_vals[21] if len(mean_vals) > 21 else 0.0
                gini_coeff_mean = mean_vals[22] if len(mean_vals) > 22 else 0.0

                # 【PPO诊断指标】新增5个指标
                explained_variance = mean_vals[23] if len(mean_vals) > 23 else 0.0
                advantages_mean = mean_vals[24] if len(mean_vals) > 24 else 0.0
                advantages_std = mean_vals[25] if len(mean_vals) > 25 else 0.0
                kl_divergence = mean_vals[26] if len(mean_vals) > 26 else 0.0
                clip_fraction = mean_vals[27] if len(mean_vals) > 27 else 0.0

                # 【分头熵监控】新增4个指标
                entropy_type = mean_vals[28] if len(mean_vals) > 28 else 0.0
                entropy_dest = mean_vals[29] if len(mean_vals) > 29 else 0.0
                entropy_cargo = mean_vals[30] if len(mean_vals) > 30 else 0.0
                entropy_quantity = mean_vals[31] if len(mean_vals) > 31 else 0.0

                idx = 32  # 与RISK模式一致
            else:
                # VEHICLE_FAILURE_DISABLED (Pure LTL版本): extend了24个值（19个原有 + 5个PPO诊断）
                mean_cost = mean_vals[4] if len(mean_vals) > 4 else 0.0
                lambda_val = mean_vals[5] if len(mean_vals) > 5 else 0.0
                makespan_train = mean_vals[6] if len(mean_vals) > 6 else 0.0
                risk_loss_val = mean_vals[7] if len(mean_vals) > 7 else 0.0
                value_loss_val = mean_vals[8] if len(mean_vals) > 8 else 0.0
                entropy_loss_val = mean_vals[9] if len(mean_vals) > 9 else 0.0
                c_bar_std = mean_vals[10] if len(mean_vals) > 10 else 0.0
                c_bar_min = mean_vals[11] if len(mean_vals) > 11 else 0.0
                c_bar_max = mean_vals[12] if len(mean_vals) > 12 else 0.0
                weight_ratio = mean_vals[13] if len(mean_vals) > 13 else 0.0
                ess_ratio = mean_vals[14] if len(mean_vals) > 14 else 0.0
                # Comprehensive metrics设为0（Pure LTL版本不使用这些指标）
                W_mean = mean_vals[15] if len(mean_vals) > 15 else 0.0
                H_actual_mean = mean_vals[16] if len(mean_vals) > 16 else 0.0
                H_optimal_mean = mean_vals[17] if len(mean_vals) > 17 else 0.0
                C_abs_mean = mean_vals[18] if len(mean_vals) > 18 else 0.0
                work_std_mean = mean_vals[19] if len(mean_vals) > 19 else 0.0
                work_cv_mean = mean_vals[20] if len(mean_vals) > 20 else 0.0
                max_mean_ratio_mean = mean_vals[21] if len(mean_vals) > 21 else 0.0
                gini_coeff_mean = mean_vals[22] if len(mean_vals) > 22 else 0.0
                # 【PPO诊断指标】新增5个指标
                explained_variance = mean_vals[23] if len(mean_vals) > 23 else 0.0
                advantages_mean = mean_vals[24] if len(mean_vals) > 24 else 0.0
                advantages_std = mean_vals[25] if len(mean_vals) > 25 else 0.0
                kl_divergence = mean_vals[26] if len(mean_vals) > 26 else 0.0
                clip_fraction = mean_vals[27] if len(mean_vals) > 27 else 0.0

                # 【分头熵监控】新增4个指标
                entropy_type = mean_vals[28] if len(mean_vals) > 28 else 0.0
                entropy_dest = mean_vals[29] if len(mean_vals) > 29 else 0.0
                entropy_cargo = mean_vals[30] if len(mean_vals) > 30 else 0.0
                entropy_quantity = mean_vals[31] if len(mean_vals) > 31 else 0.0

                idx = 32  # 下一个索引起点（4个基础 + 28个 = 32）

        # 解包性能指标（从idx开始，顺序与perf_keys完全一致）
        # 基础性能指标
        success_rate = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        episode_success = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        makespan = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        actual_time = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        time_cost = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        waiting = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        distance = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        efficiency = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1

        # 奖励指标
        episode_return = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        episode_shaping_reward = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        episode_time_penalty = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        episode_terminal_penalty = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1

        # 决策统计
        decision_steps = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        action_move_count = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        action_load_count = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        action_unload_count = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        action_rejected_count = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        total_actions = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1

        # 动作概率指标
        prob_move = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        prob_load = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        prob_unload = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1

        # PolicyHead指标（4个头的entropy和prob_max）
        ent_action_type = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        pmax_action_type = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ent_destination = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        pmax_destination = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ent_cargo = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        pmax_cargo = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ent_quantity = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        pmax_quantity = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1

        # LTL约束指标（顺序：分项→总计）
        ltl_num_safety = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ltl_num_sequential = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ltl_num_clauses = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ltl_safety_violation_rate = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ltl_sequential_satisfaction_rate = (
            mean_vals[idx] if len(mean_vals) > idx else 0.0
        )
        idx += 1
        ltl_overall_satisfaction_rate = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ltl_safety_violated_count = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ltl_sequential_satisfied_count = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ltl_cost_per_step = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ltl_total_cost = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ltl_max_cost = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        ltl_cost_std = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1

        # 【故障统计指标】
        num_agents = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        failed_agents_count = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        actual_failure_rate = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1
        avg_failure_prob = mean_vals[idx] if len(mean_vals) > idx else 0.0
        idx += 1

        logs = {
            # Loss/训练侧
            "Loss/Policy Loss": p_l,
            "Loss/Value Loss": value_loss_val,
            "Loss/Entropy Loss": entropy_loss_val,
            "Loss/Risk Loss": risk_loss_val,
            "Loss/Entropy": entropy,
            "Loss/Grad Norm": grad_norm,
            "Perf/Value_Target": reward,
            # 【PPO诊断指标】新增
            "PPO/Explained Variance": explained_variance,
            "PPO/Advantages Mean": advantages_mean,
            "PPO/Advantages Std": advantages_std,
            "PPO/KL Divergence": kl_divergence,
            "PPO/Clip Fraction": clip_fraction,
            # 【分头熵监控】新增
            "Entropy/Type": entropy_type,
            "Entropy/Destination": entropy_dest,
            "Entropy/Cargo": entropy_cargo,
            "Entropy/Quantity": entropy_quantity,
            "LTL/Mean Cost": mean_cost,
            "LTL/Lambda": lambda_val,
            # 【Risk-Sensitive诊断指标】
            "Risk/C_bar_Mean": mean_cost,
            "Risk/C_bar_Std": c_bar_std,
            "Risk/C_bar_Min": c_bar_min,
            "Risk/C_bar_Max": c_bar_max,
            "Risk/LogMeanExp": lambda_val,
            "Risk/Makespan_Train": makespan_train,
            "Risk/Episode_Weight_Ratio": weight_ratio,
            "Risk/ESS_Ratio": ess_ratio,
            # ==================== Comprehensive Metrics（新增）====================
            # 【理论验证指标】用于验证H_actual与T的耦合关系
            "Theory/W_Mean": W_mean,  # 总工作量均值
            "Theory/H_Actual_Mean": H_actual_mean,  # 实际总Hazard均值
            "Theory/H_Optimal_Mean": H_optimal_mean,  # 理论最优Hazard均值
            "Theory/C_Abs_Mean": C_abs_mean,  # 绝对超额Hazard均值
            # 【负载均衡指标】多种度量方式的综合对比
            "LoadBalance/Work_Std_Mean": work_std_mean,  # 工作时间标准差
            "LoadBalance/Work_CV_Mean": work_cv_mean,  # 变异系数
            "LoadBalance/Max_Mean_Ratio_Mean": max_mean_ratio_mean,  # 最大/平均比率
            "LoadBalance/Gini_Coeff_Mean": gini_coeff_mean,  # Gini系数
            # ====================================================================
            # 基础性能指标
            "Perf/Success Rate (Tasks)": success_rate,  # 任务完成率
            "Perf/Episode Success": episode_success,  # episode成功率
            "Perf/Makespan": makespan,
            "Perf/Actual Time": actual_time,
            "Perf/Time Cost": time_cost,
            "Perf/Waiting Time": waiting,
            "Perf/Travel Distance": distance,
            "Perf/Waiting Efficiency": efficiency,
            "Perf/TrainingStep": training_step,
            "Perf/TotalEnvSteps": total_env_steps,
            "Perf/MeanRawReward": mean_raw_reward,
            # 奖励指标
            "Perf/Episode Return": episode_return,
            "Perf/Episode Return/Shaping": episode_shaping_reward,
            "Perf/Episode Return/Time Penalty": episode_time_penalty,
            "Perf/Episode Return/Terminal Penalty": episode_terminal_penalty,
            # 决策统计
            "Perf/Decision Steps": decision_steps,
            "Perf/Action/MOVE Count": action_move_count,
            "Perf/Action/LOAD Count": action_load_count,
            "Perf/Action/UNLOAD Count": action_unload_count,
            "Perf/Action/REJECTED Count": action_rejected_count,
            "Perf/Action/Total": total_actions,
            # 动作概率指标
            "PolicyHead/Prob_MOVE": prob_move,
            "PolicyHead/Prob_LOAD": prob_load,
            "PolicyHead/Prob_UNLOAD": prob_unload,
            # PolicyHead指标（4个头：action_type, destination, cargo, quantity）
            "PolicyHead/ActionType/Entropy": ent_action_type,
            "PolicyHead/ActionType/ProbMax": pmax_action_type,
            "PolicyHead/Destination/Entropy": ent_destination,
            "PolicyHead/Destination/ProbMax": pmax_destination,
            "PolicyHead/Cargo/Entropy": ent_cargo,
            "PolicyHead/Cargo/ProbMax": pmax_cargo,
            "PolicyHead/Quantity/Entropy": ent_quantity,
            "PolicyHead/Quantity/ProbMax": pmax_quantity,
            # LTL约束监控指标（训练时记录，顺序：分项→总计）
            "Train_LTL/Num_Safety": ltl_num_safety,
            "Train_LTL/Num_Sequential": ltl_num_sequential,
            "Train_LTL/Num_Clauses": ltl_num_clauses,
            "Train_LTL/Safety_Violation_Rate": ltl_safety_violation_rate,
            "Train_LTL/Sequential_Satisfaction_Rate": ltl_sequential_satisfaction_rate,
            "Train_LTL/Overall_Satisfaction_Rate": ltl_overall_satisfaction_rate,
            "Train_LTL/Safety_Violated_Count": ltl_safety_violated_count,
            "Train_LTL/Sequential_Satisfied_Count": ltl_sequential_satisfied_count,
            "Train_LTL/Cost_Per_Step": ltl_cost_per_step,
            "Train_LTL/Total_Cost": ltl_total_cost,
            "Train_LTL/Max_Cost": ltl_max_cost,
            "Train_LTL/Cost_Std": ltl_cost_std,
            # 【故障统计指标】
            "Failure/Num_Agents": num_agents,
            "Failure/Failed_Count": failed_agents_count,
            "Failure/Actual_Rate": actual_failure_rate,
            "Failure/Avg_Probability": avg_failure_prob,
        }

        print(logs)
        # 根据算法类型选择合适的横坐标
        # IMPALA: 使用training_step（更新次数），因为是异步算法
        # PPO: 使用total_env_steps（环境交互步数），更能反映样本效率
        x_axis_value = (
            training_step
            if TrainParams.ALGORITHM in ["A2C", "IMPALA"]
            else total_env_steps
        )

        # 添加异常处理，避免WandB问题导致训练中断
        try:
            # 过滤掉NaN和Inf值（可能导致WandB拒绝数据）
            import math

            filtered_logs = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    if not (math.isnan(v) or math.isinf(v)):
                        filtered_logs[k] = v
                    else:
                        filtered_logs[k] = 0.0  # 用0替代无效值
                else:
                    filtered_logs[k] = v

            logger_module.log(filtered_logs, step=x_axis_value)
        except Exception as e:
            print(f"[WARNING] WandB训练日志上传失败 (step={x_axis_value}): {e}")
            print(f"[WARNING] 训练将继续，但该步的日志未上传到WandB")

    def load_saved_model(self):
        print("Loading Model...")
        checkpoint = torch.load(os.path.join(SaverParams.MODEL_PATH, "checkpoint.pth"))
        model_key = "best_model" if SaverParams.LOAD_FROM == "best" else "model"
        self.global_net.load_state_dict(checkpoint[model_key])
        self.baseline_net.load_state_dict(checkpoint[model_key])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_decay.load_state_dict(checkpoint["lr_decay"])
        curr_episode = checkpoint["episode"]
        curr_level = checkpoint["level"]
        best_perf = checkpoint["best_perf"]
        print("curr_episode set to", curr_episode)
        print("best_perf so far is", best_perf)
        if TrainParams.RESET_OPT:
            self.optimizer = optim.Adam(self.global_net.parameters(), lr=TrainParams.LR)
            self.lr_decay = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=TrainParams.DECAY_STEP, gamma=0.98
            )
        return curr_episode, curr_level, best_perf

    def save_model(self, curr_episode, curr_level, best_perf):
        print("Saving model")
        checkpoint = {
            "model": self.global_net.state_dict(),
            "best_model": self.baseline_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_decay": self.lr_decay.state_dict(),
            "episode": curr_episode,
            "level": curr_level,
            "best_perf": best_perf,
        }
        # 使用run_name作为模型文件名
        model_filename = f"{self.run_name}.pth"
        path_checkpoint = os.path.join(SaverParams.MODEL_PATH, model_filename)
        torch.save(checkpoint, path_checkpoint)
        print(f"Model saved to: {path_checkpoint}")
        # Save checkpoint as artifact
        if LoggingParams.BACKEND == "wandb":
            logger_module.save(path_checkpoint)
        # swanlab会自动保存run目录下的文件，无需显式调用

    # @staticmethod
    # def generate_env_params(curr_level=None):
    #     per_species_num = np.random.randint(EnvParams.SPECIES_AGENTS_RANGE[0], EnvParams.SPECIES_AGENTS_RANGE[1] + 1)
    #     species_num = np.random.randint(EnvParams.SPECIES_RANGE[0], EnvParams.SPECIES_RANGE[1] + 1)
    #     tasks_num = np.random.randint(EnvParams.TASKS_RANGE[0], EnvParams.TASKS_RANGE[1] + 1)
    #     return [(per_species_num, per_species_num), (species_num, species_num), (tasks_num, tasks_num)]

    @staticmethod
    def generate_env_params(training_step=0):
        # 如果未启用课程学习，则使用 EnvParams 中的默认最大范围
        if not CurriculumParams.ENABLED:
            return [
                EnvParams.SPECIES_AGENTS_RANGE,
                EnvParams.SPECIES_RANGE,
                EnvParams.TASKS_RANGE,
                EnvParams.DEPOT_NUM_RANGE,
            ]

        # 1. 根据 training_step 计算当前所处的难度阶段
        current_stage = training_step // CurriculumParams.DIFFICULTY_INCREASE_STEP

        # 2. 如果超出预设的课程表，则使用最后一个阶段（最难）的参数
        if current_stage >= len(CurriculumParams.SCHEDULE):
            current_stage = len(CurriculumParams.SCHEDULE) - 1

        # 3. 从课程表中获取当前阶段的参数
        schedule = CurriculumParams.SCHEDULE[current_stage]
        species_range, agents_range, depot_range, tasks_range = schedule

        # 4. 返回一个包含所有动态参数的列表，以便传递给 Worker
        # 返回顺序: [agents_range(对应per_species_range), species_range, tasks_range, depot_range]
        # 注意: agents_range 即 per_species_range (每个物种的智能体数量范围)
        return [agents_range, species_range, tasks_range, depot_range]

    @staticmethod
    def generate_test_set_seed():
        return np.random.randint(
            low=0, high=1e8, size=TrainParams.EVALUATION_SAMPLES
        ).tolist()


def fuse_two_dicts(d1, d2):
    if d2 is not None:
        merged = {**d1, **d2}
        return {k: d1[k] + v for k, v in merged.items()}
    return d1


# 该函数位于 driver.py 文件中
def main():
    parser = argparse.ArgumentParser(
        description="Run RL training with custom hyperparameters."
    )

    # ========== 消融实验参数（由run_ablation.py传递）==========
    parser.add_argument(
        "--manual_seed", type=int, default=None, help="Manual seed for reproducibility"
    )
    parser.add_argument(
        "--entropy_beta",
        type=float,
        default=None,
        help="Entropy regularization coefficient",
    )
    parser.add_argument(
        "--execution_mode",
        type=str,
        default=None,
        choices=["mdp", "smdp"],
        help="Execution mode (mdp/smdp)",
    )
    parser.add_argument(
        "--ltl_enabled",
        type=str,
        default=None,
        help="Enable LTL constraints (true/false)",
    )
    parser.add_argument(
        "--ltl_enabled_in_evaluation",
        type=str,
        default=None,
        help="Enable LTL in evaluation (true/false)",
    )
    parser.add_argument(
        "--ltl_constraint_type", type=str, default=None, help="LTL constraint type"
    )
    parser.add_argument(
        "--ltl_encoding_type", type=str, default=None, help="LTL encoding type (A/B/C)"
    )
    parser.add_argument(
        "--use_multi_head_ppo",
        type=str,
        default=None,
        help="Use multi-head PPO (true/false)",
    )
    parser.add_argument(
        "--use_value_loss_normalization",
        type=str,
        default=None,
        help="Use value loss normalization (true/false)",
    )
    parser.add_argument(
        "--max_training_steps", type=int, default=None, help="Maximum training steps"
    )
    parser.add_argument(
        "--use_manual_seed", type=str, default=None, help="Use manual seed (true/false)"
    )

    # ========== 原有参数（向后兼容）==========
    parser.add_argument(
        "--pr",
        type=int,
        default=TrainParams.REWARD_COMPLETED_DEMAND_WEIGHT,
        help=f"progress_reward (default: {TrainParams.REWARD_COMPLETED_DEMAND_WEIGHT})",
    )
    parser.add_argument(
        "--de",
        type=int,
        default=TrainParams.REWARD_TIME_ELAPSED_WEIGHT,
        help=f"dense_time_reward (default: {TrainParams.REWARD_TIME_ELAPSED_WEIGHT})",
    )

    args = parser.parse_args()

    # ========== 应用消融实验参数（如果提供）==========
    if args.manual_seed is not None:
        TrainParams.MANUAL_SEED = args.manual_seed
    if args.entropy_beta is not None:
        TrainParams.ENTROPY_BETA = args.entropy_beta
    if args.execution_mode is not None:
        TrainParams.EXECUTION_MODE = args.execution_mode
    if args.ltl_enabled is not None:
        TrainParams.LTL_ENABLED = args.ltl_enabled.lower() == "true"
    if args.ltl_enabled_in_evaluation is not None:
        TrainParams.LTL_ENABLED_IN_EVALUATION = (
            args.ltl_enabled_in_evaluation.lower() == "true"
        )
    if args.ltl_constraint_type is not None:
        TrainParams.LTL_CONSTRAINT_TYPE = args.ltl_constraint_type
    if args.ltl_encoding_type is not None:
        TrainParams.LTL_ENCODING_TYPE = args.ltl_encoding_type
    if args.use_multi_head_ppo is not None:
        TrainParams.USE_MULTI_HEAD_PPO = args.use_multi_head_ppo.lower() == "true"
    if args.use_value_loss_normalization is not None:
        TrainParams.USE_VALUE_LOSS_NORMALIZATION = (
            args.use_value_loss_normalization.lower() == "true"
        )
    if args.max_training_steps is not None:
        TrainParams.MAX_TRAINING_STEPS = args.max_training_steps
    if args.use_manual_seed is not None:
        TrainParams.USE_MANUAL_SEED = args.use_manual_seed.lower() == "true"

    # ========== 应用原有参数（向后兼容）==========
    TrainParams.REWARD_COMPLETED_DEMAND_WEIGHT = args.pr
    TrainParams.REWARD_TIME_ELAPSED_WEIGHT = args.de

    print("=" * 80)
    print("Running with Configuration:")
    print("=" * 80)
    print(
        f"Random Seed: {TrainParams.MANUAL_SEED} (USE_MANUAL_SEED={TrainParams.USE_MANUAL_SEED})"
    )
    print(f"Execution Mode: {TrainParams.EXECUTION_MODE}")
    print(f"Entropy Beta: {TrainParams.ENTROPY_BETA}")
    print(f"LTL Enabled: {TrainParams.LTL_ENABLED}")
    if TrainParams.LTL_ENABLED:
        print(f"  - Constraint Type: {TrainParams.LTL_CONSTRAINT_TYPE}")
        print(f"  - Encoding Type: {TrainParams.LTL_ENCODING_TYPE}")
        print(f"  - Enabled in Evaluation: {TrainParams.LTL_ENABLED_IN_EVALUATION}")
    print(f"Multi-Head PPO: {TrainParams.USE_MULTI_HEAD_PPO}")
    print(f"Value Loss Normalization: {TrainParams.USE_VALUE_LOSS_NORMALIZATION}")
    print(f"Max Training Steps: {TrainParams.MAX_TRAINING_STEPS:,}")
    print(f"Algorithm: {TrainParams.ALGORITHM}")
    print(
        f"Evaluation Sampling: {TrainParams.EVAL_USE_SAMPLING} ({'采样模式' if TrainParams.EVAL_USE_SAMPLING else '贪婪模式'})"
    )
    print("=" * 80)

    # 设置随机种子以确保可复现性
    if TrainParams.USE_MANUAL_SEED:
        set_random_seeds(TrainParams.MANUAL_SEED)

    logger = Logger()
    # 初始化Ray，禁用日志去重以显示所有workers的输出
    # 确保Ray workers继承CUDA_VISIBLE_DEVICES环境变量
    import os

    env_vars = {
        "RAY_DEDUP_LOGS": "0"  # 禁用日志去重
    }
    # 如果设置了CUDA_VISIBLE_DEVICES，确保Ray workers也使用相同的GPU
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_vars["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        print(f"[GPU设置] 使用GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    ray.init(
        log_to_driver=True,  # 将worker日志重定向到driver
        logging_level="INFO",  # 设置日志级别
        runtime_env={"env_vars": env_vars},
    )
    device = torch.device("cuda") if TrainParams.USE_GPU_GLOBAL else torch.device("cpu")
    local_device = torch.device("cuda") if TrainParams.USE_GPU else torch.device("cpu")

    global_network = AttentionNet(
        TrainParams.AGENT_INPUT_DIM,
        TrainParams.TASK_INPUT_DIM,
        TrainParams.EMBEDDING_DIM,
    ).to(device)
    baseline_network = AttentionNet(
        TrainParams.AGENT_INPUT_DIM,
        TrainParams.TASK_INPUT_DIM,
        TrainParams.EMBEDDING_DIM,
    ).to(device)

    global_optimizer = optim.Adam(
        global_network.parameters(), lr=TrainParams.LR, weight_decay=0.0
    )
    # 使用线性衰减：在100次更新内从初始LR线性衰减到10%
    # 对应SM训练中PPO的约100次更新
    lr_decay = optim.lr_scheduler.LinearLR(
        global_optimizer,
        start_factor=1.0,  # 从100%的初始LR开始
        end_factor=0.1,  # 衰减到10%的初始LR
        total_iters=100,  # 在100次scheduler.step()后完成衰减
    )

    logger.set(global_network, baseline_network, global_optimizer, lr_decay)

    curr_episode, curr_level, best_perf, training_step = 0, 0, -500, 0

    training_step = 0
    total_env_steps = 0

    if SaverParams.LOAD_MODEL:
        curr_episode, curr_level, best_perf = logger.load_saved_model()

    if TrainParams.EVALUATE:
        print(
            f"Generating a fixed set of {TrainParams.EVALUATION_SAMPLES} seeds for evaluation."
        )
        # 使用固定的随机状态生成评估种子，确保每次运行的评估集都一样
        rng = np.random.default_rng(seed=12345)
        evaluation_seeds = rng.integers(
            low=0, high=1e8, size=TrainParams.EVALUATION_SAMPLES
        ).tolist()

    evaluation_benchmarks = []
    if TrainParams.EVALUATE:
        try:
            with open("evaluation_benchmarks.json", "r") as f:
                benchmark_data = json.load(f)

            # 【修复】判断json格式：如果是字典则提取'benchmarks'字段，否则直接使用
            if isinstance(benchmark_data, dict):
                if "benchmarks" in benchmark_data:
                    evaluation_benchmarks = benchmark_data["benchmarks"]
                    print(
                        f"成功加载了 {len(evaluation_benchmarks)} 个固定的评估基准（从字典格式）。"
                    )
                else:
                    print(
                        "错误：evaluation_benchmarks.json是字典但缺少'benchmarks'字段！"
                    )
                    evaluation_benchmarks = []
            elif isinstance(benchmark_data, list):
                evaluation_benchmarks = benchmark_data
                print(
                    f"成功加载了 {len(evaluation_benchmarks)} 个固定的评估基准（从列表格式）。"
                )
            else:
                print(
                    f"错误：evaluation_benchmarks.json格式不正确（类型：{type(benchmark_data)}）！"
                )
                evaluation_benchmarks = []

            if not evaluation_benchmarks:
                print("警告：评估基准文件为空，将不执行LTL评估。")
        except FileNotFoundError:
            print(
                "警告：未找到 'evaluation_benchmarks.json'。评估时将不使用固定的LTL约束。"
            )
            # 作为后备，可以生成随机种子，但不包含LTL
            evaluation_benchmarks = [
                {"seed": s, "ltl_clauses": None}
                for s in np.random.randint(0, 1e8, TrainParams.EVALUATION_SAMPLES)
            ]

    # ==================== 创建Worker池 ====================
    # 创建训练workers（评估时也复用这些workers）
    meta_agents = [RLRunner.remote(i) for i in range(TrainParams.NUM_META_AGENT)]
    print(f"✅ 已创建 {TrainParams.NUM_META_AGENT} 个训练worker（评估时复用）")
    # ====================================================

    # ==================== 初始化异步评估器（串行模式）====================
    async_evaluator = None
    if TrainParams.EVALUATE:
        # 【串行评估模式】复用训练workers进行评估
        # 评估时训练会暂停，评估完成后继续训练
        async_evaluator = AsyncEvaluator(logger, meta_agents)  # 使用训练workers
        print(f"\n{'=' * 80}")
        print(f"[ASYNC EVAL] 🚀 异步评估系统已初始化（串行模式）")
        print(f"[ASYNC EVAL] 👥 复用训练workers: {len(meta_agents)}个")
        print(f"[ASYNC EVAL] 📊 评估样本数: {TrainParams.EVALUATION_SAMPLES}")
        print(f"[ASYNC EVAL] ⏱️  评估间隔: 每{TrainParams.EVALUATE_GAP}次更新")
        print(f"[ASYNC EVAL] ⚠️  串行模式：评估时训练暂停，评估完成后继续")
        print(
            f"[ASYNC EVAL] 💡 优势：无GPU资源冲突，评估使用全部{len(meta_agents)}个workers"
        )
        print(f"{'=' * 80}\n")
    # ====================================================================

    # Synchronous PPO training loop (only supported algorithm).
    if True:
        print(
            f"--- Running in SYNCHRONOUS mode for algorithm: {TrainParams.ALGORITHM} ---"
        )

        # Lagrangian state for soft-constraint LTL modes (only used when
        # LTL_CONSTRAINT_TYPE is not 'HARD'). Kept for compatibility with
        # the broader codebase; the released checkpoint uses hard masking.
        log_lambda = torch.tensor(0.0, dtype=torch.float32, device=device)
        lambda_optimizer = None

        # ==================== Value Loss Normalization：初始化Running Stats ====================
        returns_running_stats = RunningStats(decay=0.99, device=device)
        print(f"[PPO] Value Loss Normalization已启用")
        print(f"[PPO]   - 使用Running Statistics (EMA decay=0.99)")
        print(f"[PPO]   - 理论依据: L_V_norm = E[(V-R)²] / (σ_R + ε)²")
        print(f"[PPO]   - 效果: 自动平衡policy/value梯度，适应reward scale变化")
        # ========================================================================================

        perf_keys = [
            # === 基础性能指标 ===
            "success_rate",  # 任务完成率（任务数）
            "episode_success",  # episode成功率（0或1）
            "makespan",  # 完成时间（失败时=MAX_TIME）
            "actual_time",  # 实际完成时间（无论成功失败）
            "time_cost",
            "waiting_time",
            "travel_dist",
            "efficiency",
            # === 奖励指标 ===
            "episode_return",
            "episode_shaping_reward",
            "episode_time_penalty",
            "episode_terminal_penalty",
            # === 决策统计 ===
            "decision_steps",
            "action_move_count",
            "action_load_count",
            "action_unload_count",
            "action_rejected_count",
            "total_actions",
            # === 动作概率指标 ===
            "prob_move",
            "prob_load",
            "prob_unload",
            # === PolicyHead指标（4个头的entropy和prob_max）===
            "entropy/action_type",
            "prob_max/action_type",
            "entropy/destination",
            "prob_max/destination",
            "entropy/cargo",
            "prob_max/cargo",
            "entropy/quantity",
            "prob_max/quantity",
            # === LTL约束相关指标（顺序：分项→总计）===
            "ltl_num_safety",
            "ltl_num_sequential",
            "ltl_num_clauses",
            "ltl_safety_violation_rate",
            "ltl_sequential_satisfaction_rate",
            "ltl_overall_satisfaction_rate",
            "ltl_safety_violated_count",
            "ltl_sequential_satisfied_count",
            "ltl_cost_per_step",
            "ltl_total_cost",
            "ltl_max_cost",
            "ltl_cost_std",
            # === 故障统计（Weibull）===
            "num_agents",
            "failed_agents_count",
            "actual_failure_rate",
            "avg_failure_prob",
        ]

        perf_metrics_aggregator = {k: [] for k in perf_keys}
        training_metrics_aggregator = []
        raw_rewards_since_last_log = []

        try:
            # Main synchronous training loop
            # 检查是否设置了最大训练步数限制
            while (
                TrainParams.MAX_TRAINING_STEPS is None
                or total_env_steps < TrainParams.MAX_TRAINING_STEPS
            ):
                try:
                    import time

                    training_round_start = time.time()

                    # === Stage 1: Distribute latest weights ===
                    weights_memory = ray.put(global_network.state_dict())
                    baseline_weights_memory = ray.put(baseline_network.state_dict())

                    # === Stage 2: Start all workers for parallel data collection ===
                    env_params = logger.generate_env_params(training_step)
                    print(f"\n{'=' * 80}")
                    print(
                        f"[TRAIN] Training step {training_step} 开始 at {time.strftime('%H:%M:%S')}"
                    )
                    print(f"[TRAIN] 提交 {TrainParams.NUM_META_AGENT} 个训练任务")
                    print(f"{'=' * 80}")

                    # 为每个worker生成确定性的种子（使用大间隔确保环境多样性）
                    # Worker间隔: 10000（参考decision_order_rng的正确做法）
                    # Episode间隔: 100（确保跨episode也有足够差异）
                    if TrainParams.USE_MANUAL_SEED:
                        worker_seeds = [
                            TrainParams.MANUAL_SEED + curr_episode * 100 + i * 10000
                            for i in range(TrainParams.NUM_META_AGENT)
                        ]
                    else:
                        worker_seeds = [None] * TrainParams.NUM_META_AGENT

                    submission_start = time.time()
                    jobs = [
                        agent.training.remote(
                            weights_memory,
                            baseline_weights_memory,
                            curr_episode + i,
                            env_params,
                            worker_seeds[i],
                        )
                        for i, agent in enumerate(meta_agents)
                    ]
                    submission_time = time.time() - submission_start

                    # === Stage 3: Synchronize - wait for ALL workers to finish ===
                    print(
                        f"[TRAIN] 训练任务提交完成 ({submission_time:.2f}s), 等待收集数据..."
                    )
                    wait_start = time.time()
                    all_results = ray.get(jobs, timeout=3600)  # 1小时超时
                    wait_time = time.time() - wait_start
                    curr_episode += TrainParams.NUM_META_AGENT

                    print(
                        f"[TRAIN] 数据收集完成！等待耗时: {wait_time:.1f}s, at {time.strftime('%H:%M:%S')}"
                    )

                except ray.exceptions.RayTaskError as e:
                    print(f"\n{'=' * 60}")
                    print(f"[ERROR] Worker任务失败！")
                    print(f"{'=' * 60}")
                    print(f"错误信息: {e}")
                    print(f"\n可能的原因：")
                    print(f"  1. GPU内存不足（尝试减少NUM_META_AGENT）")
                    print(f"  2. Worker代码中有bug")
                    print(f"  3. 环境配置问题")
                    print(f"\n建议：查看上方的详细错误堆栈")
                    print(f"{'=' * 60}\n")
                    raise  # 重新抛出异常以便完整的traceback

                except ray.exceptions.GetTimeoutError:
                    print(f"\n{'=' * 60}")
                    print(f"[ERROR] Workers超时！（60分钟未完成）")
                    print(f"{'=' * 60}")
                    print(f"当前状态：")
                    print(f"  - Training step: {training_step}")
                    print(f"  - 等待的workers数量: {len(jobs)}")
                    print(f"\n可能的原因：")
                    print(f"  1. 环境太复杂，需要很长时间")
                    print(f"  2. 某个worker卡住了")
                    print(f"  3. PPO_ROLLOUT_LENGTH设置太大")
                    print(f"\n建议：")
                    print(f"  1. 检查worker日志: /tmp/ray/session_latest/logs/")
                    print(f"  2. 减小PPO_ROLLOUT_LENGTH")
                    print(f"  3. 简化环境配置")
                    print(f"{'=' * 60}\n")
                    raise

                # === Stage 4: Aggregate data from all workers ===
                rollout_buffer = {
                    "task_info": [],
                    "agents_info": [],
                    "mask": [],
                    "index": [],
                    "next_task_info": [],
                    "next_agents_info": [],
                    "next_mask": [],
                    "next_index": [],
                    "value": [],
                    "old_log_prob": [],
                    "entropy": [],
                    "actions": [],
                    "rewards": [],
                    "reward": [],
                    "advantage": [],  # 支持两种格式：原始rewards和GAE处理后的reward/advantage
                    "dones": [],
                    "cargo_mask": [],
                    "action_type_mask": [],
                    "costs": [],
                    "taus": [],
                    "ltl_info": [],
                    "next_ltl_info": [],
                    "episode_truly_done": [],
                    "episode_id": [],  # SMDP episode完整性标记
                    # 【方法1：分头独立PPO】添加各头的old_log_prob
                    "old_log_prob_type": [],
                    "old_log_prob_dest": [],
                    "old_log_prob_cargo": [],
                }

                # 【Episode-level CVAR & Risk-Sensitive】收集episode数据映射
                # 使用global episode ID来避免不同worker的episode_idx冲突
                global_episode_risk_map = {}  # {global_episode_id: C_episode} (CVAR模式)
                global_episode_done_map = {}  # {global_episode_id: truly_done}
                global_episode_return_map = {}  # {global_episode_id: episode_return} (已废弃，保留兼容)

                # 【Mean-Based Multi-Objective】核心优化目标指标
                global_episode_C_map = {}  # {global_episode_id: C_rel} (相对超额hazard)
                global_episode_T_map = {}  # {global_episode_id: T} (makespan)

                # 【理论验证指标】用于验证H_actual与T的耦合关系
                global_episode_W_map = {}  # {global_episode_id: W} (总工作量)
                global_episode_H_actual_map = {}  # {global_episode_id: H_actual} (实际总hazard)
                global_episode_H_optimal_map = {}  # {global_episode_id: H_optimal} (理论最优hazard)
                global_episode_C_abs_map = {}  # {global_episode_id: C_abs} (绝对超额hazard)

                # 【负载均衡指标】多种度量方式的综合对比
                global_episode_work_std_map = {}  # {global_episode_id: std(t_i)}
                global_episode_work_cv_map = {}  # {global_episode_id: CV}
                global_episode_max_mean_ratio_map = {}  # {global_episode_id: max/mean}
                global_episode_gini_coeff_map = {}  # {global_episode_id: Gini}

                # 【原始分布数据】用于后处理分析
                global_episode_work_times_map = {}  # {global_episode_id: [t_1, t_2, ..., t_n]}
                global_episode_num_agents_map = {}  # {global_episode_id: n}

                worker_episode_id_offset = 0  # 为每个worker分配不同的ID空间

                for worker_idx, result in enumerate(all_results):
                    buffer, metrics, collected_steps, info = result
                    total_env_steps += collected_steps

                    # 兼容两种reward key名称：'rewards'（原始）和'reward'（GAE处理后）
                    if "rewards" in buffer:
                        raw_rewards_since_last_log.extend(buffer["rewards"])
                    elif "reward" in buffer:
                        # PPO+GAE模式：reward已经是returns，不再是原始rewards
                        raw_rewards_since_last_log.extend(
                            [
                                r.item() if hasattr(r, "item") else r
                                for r in buffer["reward"]
                            ]
                        )

                    # 【Episode-level CVAR】收集该worker的episode风险映射，转换为全局ID
                    if (
                        TrainParams.LTL_CONSTRAINT_TYPE == "CVAR_SMDP"
                        and "episode_risk_index_map" in buffer
                    ):
                        local_map = buffer[
                            "episode_risk_index_map"
                        ]  # {local_episode_id: C_episode}
                        for local_ep_id, C_episode in local_map.items():
                            global_ep_id = worker_episode_id_offset + local_ep_id
                            global_episode_risk_map[global_ep_id] = C_episode
                            if training_step <= 3:
                                print(
                                    f"[Episode-CVAR Driver] Worker {worker_idx}, Local EP {local_ep_id} -> Global EP {global_ep_id}: C = {C_episode:.6f}"
                                )

                    # 【Comprehensive Metrics】收集comprehensive metrics（NEU和RISK模式均需要）
                    if EnvParams.VEHICLE_FAILURE_ENABLED:
                        # 【诊断】检查buffer中的keys（仅前3步）
                        if training_step <= 3:
                            print(
                                f"[DIAGNOSTIC Driver] Worker {worker_idx}, Checking comprehensive metrics:"
                            )
                            metric_keys = [
                                "episode_C_map",
                                "episode_T_map",
                                "episode_total_work_W",
                                "episode_H_actual",
                                "episode_H_optimal",
                                "episode_C_abs",
                                "episode_work_std",
                                "episode_work_cv",
                                "episode_max_mean_ratio",
                                "episode_gini_coeff",
                                "episode_work_times",
                                "episode_num_agents",
                            ]
                            for key in metric_keys:
                                in_buffer = key in buffer
                                print(f"  - '{key}' in buffer: {in_buffer}")

                        # ==================== 核心优化目标 ====================
                        # 收集C_rel（相对超额hazard）
                        if "episode_C_map" in buffer:
                            local_C_map = buffer["episode_C_map"]
                            for local_ep_id, C_value in local_C_map.items():
                                global_ep_id = worker_episode_id_offset + local_ep_id
                                global_episode_C_map[global_ep_id] = C_value
                                if training_step <= 3:
                                    print(
                                        f"  [C_rel] Worker {worker_idx}, EP {local_ep_id} -> Global {global_ep_id}: {C_value:.4f}"
                                    )

                        # 收集makespan T
                        if "episode_T_map" in buffer:
                            local_T_map = buffer["episode_T_map"]
                            for local_ep_id, T_value in local_T_map.items():
                                global_ep_id = worker_episode_id_offset + local_ep_id
                                global_episode_T_map[global_ep_id] = T_value
                                if training_step <= 3:
                                    print(
                                        f"  [T] Worker {worker_idx}, EP {local_ep_id} -> Global {global_ep_id}: {T_value:.2f}"
                                    )

                        # ==================== 理论验证指标 ====================
                        # 总工作量W
                        if "episode_total_work_W" in buffer:
                            local_W_map = buffer["episode_total_work_W"]
                            if isinstance(local_W_map, dict):
                                for local_ep_id, W_value in local_W_map.items():
                                    global_ep_id = (
                                        worker_episode_id_offset + local_ep_id
                                    )
                                    global_episode_W_map[global_ep_id] = W_value

                        # 实际总hazard H_actual
                        if "episode_H_actual" in buffer:
                            local_H_actual_map = buffer["episode_H_actual"]
                            if isinstance(local_H_actual_map, dict):
                                for local_ep_id, H_value in local_H_actual_map.items():
                                    global_ep_id = (
                                        worker_episode_id_offset + local_ep_id
                                    )
                                    global_episode_H_actual_map[global_ep_id] = H_value

                        # 理论最优hazard H_optimal
                        if "episode_H_optimal" in buffer:
                            local_H_optimal_map = buffer["episode_H_optimal"]
                            if isinstance(local_H_optimal_map, dict):
                                for local_ep_id, H_value in local_H_optimal_map.items():
                                    global_ep_id = (
                                        worker_episode_id_offset + local_ep_id
                                    )
                                    global_episode_H_optimal_map[global_ep_id] = H_value

                        # 绝对超额hazard C_abs
                        if "episode_C_abs" in buffer:
                            local_C_abs_map = buffer["episode_C_abs"]
                            if isinstance(local_C_abs_map, dict):
                                for local_ep_id, C_abs_value in local_C_abs_map.items():
                                    global_ep_id = (
                                        worker_episode_id_offset + local_ep_id
                                    )
                                    global_episode_C_abs_map[global_ep_id] = C_abs_value

                        # ==================== 负载均衡指标 ====================
                        # 标准差
                        if "episode_work_std" in buffer:
                            local_std_map = buffer["episode_work_std"]
                            if isinstance(local_std_map, dict):
                                for local_ep_id, std_value in local_std_map.items():
                                    global_ep_id = (
                                        worker_episode_id_offset + local_ep_id
                                    )
                                    global_episode_work_std_map[global_ep_id] = (
                                        std_value
                                    )

                        # 变异系数CV
                        if "episode_work_cv" in buffer:
                            local_cv_map = buffer["episode_work_cv"]
                            if isinstance(local_cv_map, dict):
                                for local_ep_id, cv_value in local_cv_map.items():
                                    global_ep_id = (
                                        worker_episode_id_offset + local_ep_id
                                    )
                                    global_episode_work_cv_map[global_ep_id] = cv_value

                        # 最大/平均比率
                        if "episode_max_mean_ratio" in buffer:
                            local_ratio_map = buffer["episode_max_mean_ratio"]
                            if isinstance(local_ratio_map, dict):
                                for local_ep_id, ratio_value in local_ratio_map.items():
                                    global_ep_id = (
                                        worker_episode_id_offset + local_ep_id
                                    )
                                    global_episode_max_mean_ratio_map[global_ep_id] = (
                                        ratio_value
                                    )

                        # Gini系数
                        if "episode_gini_coeff" in buffer:
                            local_gini_map = buffer["episode_gini_coeff"]
                            if isinstance(local_gini_map, dict):
                                for local_ep_id, gini_value in local_gini_map.items():
                                    global_ep_id = (
                                        worker_episode_id_offset + local_ep_id
                                    )
                                    global_episode_gini_coeff_map[global_ep_id] = (
                                        gini_value
                                    )

                        # ==================== 原始分布数据 ====================
                        # 工作时间数组
                        if "episode_work_times" in buffer:
                            local_work_times_map = buffer["episode_work_times"]
                            if isinstance(local_work_times_map, dict):
                                for (
                                    local_ep_id,
                                    work_times,
                                ) in local_work_times_map.items():
                                    global_ep_id = (
                                        worker_episode_id_offset + local_ep_id
                                    )
                                    global_episode_work_times_map[global_ep_id] = (
                                        work_times
                                    )

                        # Agent数量
                        if "episode_num_agents" in buffer:
                            local_num_agents_map = buffer["episode_num_agents"]
                            if isinstance(local_num_agents_map, dict):
                                for (
                                    local_ep_id,
                                    n_value,
                                ) in local_num_agents_map.items():
                                    global_ep_id = (
                                        worker_episode_id_offset + local_ep_id
                                    )
                                    global_episode_num_agents_map[global_ep_id] = (
                                        n_value
                                    )

                    # 【重要】转换buffer中的episode_id为全局ID
                    if "episode_id" in buffer and buffer["episode_id"]:
                        buffer["episode_id"] = [
                            worker_episode_id_offset + local_id
                            for local_id in buffer["episode_id"]
                        ]
                        # 同时记录哪些episode是完整的
                        if "episode_truly_done" in buffer:
                            for i, (ep_id, truly_done) in enumerate(
                                zip(buffer["episode_id"], buffer["episode_truly_done"])
                            ):
                                global_episode_done_map[ep_id] = truly_done
                        # 更新offset，为下一个worker准备
                        if buffer["episode_id"]:
                            worker_episode_id_offset = max(buffer["episode_id"]) + 1

                    for key in rollout_buffer.keys():
                        if key in buffer:
                            rollout_buffer[key].extend(buffer[key])

                    for key in perf_metrics_aggregator.keys():
                        if key in metrics:
                            perf_metrics_aggregator[key].extend(metrics[key])

                # === Stage 5: Filter complete episodes (SMDP理论要求) ===
                # 智能选择非空的 reward 列表（优先使用 worker 端计算的 'reward'，否则使用原始 'rewards'）
                r_list = (
                    rollout_buffer["reward"]
                    if len(rollout_buffer["reward"]) > 0
                    else rollout_buffer["rewards"]
                )
                total_rollout_size = len(r_list)
                original_total_rollout_size = (
                    total_rollout_size  # Save original size for filtering check
                )
                print(
                    f"[PPO] 收集到 {total_rollout_size} 条transition，总环境步数: {total_env_steps}"
                )

                # 【调试】检查costs数据汇总情况
                if training_step <= 3 and TrainParams.LTL_CONSTRAINT_TYPE != "HARD":
                    costs_total_len = (
                        len(rollout_buffer["costs"]) if rollout_buffer["costs"] else 0
                    )
                    print(
                        f"[COST DEBUG] After aggregation: costs_len={costs_total_len}, rollout_size={total_rollout_size}"
                    )

                if (
                    TrainParams.USE_ONLY_COMPLETE_EPISODES
                    and "episode_truly_done" in rollout_buffer
                ):
                    # 统计完整episode的数据量
                    episode_truly_done_flags = rollout_buffer["episode_truly_done"]
                    complete_indices = [
                        i for i, flag in enumerate(episode_truly_done_flags) if flag
                    ]
                    complete_ratio = len(complete_indices) / max(total_rollout_size, 1)

                    print(
                        f"[SMDP] 完整episode数据: {len(complete_indices)}/{total_rollout_size} ({complete_ratio:.1%})"
                    )

                    # 检查是否满足最小批次大小要求
                    if complete_ratio < TrainParams.MIN_COMPLETE_RATIO:
                        print(
                            f"[SMDP] Warning: 完整episode数据不足 ({complete_ratio:.1%} < {TrainParams.MIN_COMPLETE_RATIO:.1%})"
                        )
                        print(f"[SMDP] 使用所有数据（包括截断episode）以避免批次过小")
                    else:
                        # 过滤出完整episode的数据
                        filtered_buffer = {}
                        for key in rollout_buffer.keys():
                            # 只过滤长度与transition数量一致的列表数据
                            # 对于 dependency_graph 等非transition-aligned数据，保持原样
                            if (
                                key in rollout_buffer
                                and isinstance(rollout_buffer[key], list)
                                and len(rollout_buffer[key])
                                == original_total_rollout_size
                            ):
                                filtered_buffer[key] = [
                                    rollout_buffer[key][i] for i in complete_indices
                                ]
                            else:
                                filtered_buffer[key] = rollout_buffer[key]

                        rollout_buffer = filtered_buffer
                        total_rollout_size = len(complete_indices)
                        print(
                            f"[SMDP] 过滤后数据量: {total_rollout_size} 条transition（仅完整episode）"
                        )

                if total_rollout_size == 0:
                    print("[PPO] Warning: Empty rollout buffer, skipping update")
                    continue

                task_inputs_batch = torch.stack(rollout_buffer["task_info"]).to(device)
                agent_inputs_batch = torch.stack(rollout_buffer["agents_info"]).to(
                    device
                )
                global_mask_batch = torch.stack(rollout_buffer["mask"]).to(device)
                index_batch = torch.stack(rollout_buffer["index"]).to(device)

                # 获取LTL信息（如果有）
                if rollout_buffer["ltl_info"]:
                    # 检查是否为模式C（字典格式）
                    if TrainParams.LTL_ENCODING_TYPE == "C":
                        # 【方案B修复】模式C：feasibility可以stack（固定形状），edge_index和edge_attr用List（变长形状）
                        ltl_info_batch = {
                            "feasibility": torch.stack(
                                [
                                    item["feasibility"]
                                    for item in rollout_buffer["ltl_info"]
                                ]
                            ).to(device),
                            "edge_index": [
                                item["edge_index"].to(device)
                                for item in rollout_buffer["ltl_info"]
                            ],  # List避免torch.stack错误
                            "edge_attr": [
                                item["edge_attr"].to(device)
                                for item in rollout_buffer["ltl_info"]
                            ],  # List避免torch.stack错误
                        }
                    else:
                        # 模式A/B：直接stack tensor
                        ltl_info_batch = torch.stack(rollout_buffer["ltl_info"]).to(
                            device
                        )
                else:
                    ltl_info_batch = None

                # 【方案一】获取dependency_graph（如果有）
                if (
                    "dependency_graph" in rollout_buffer
                    and rollout_buffer["dependency_graph"]
                ):
                    # dependency_graph应该是list of [N_tasks, N_tasks]矩阵
                    # 取第一个作为整个batch的依赖图（假设依赖关系在episode内不变）
                    dependency_graph_batch = rollout_buffer["dependency_graph"][0].to(
                        device
                    )
                else:
                    dependency_graph_batch = None

                # 【必须】准备dones和gammas，无论GAE是否预计算（软约束需要）
                # 【调试】检查dones数据
                if training_step <= 3:
                    print(f"\n[DONES DEBUG] Before creating dones_batch:")
                    print(f"  'dones' in rollout_buffer: {'dones' in rollout_buffer}")
                    print(
                        f"  len(rollout_buffer['dones']): {len(rollout_buffer['dones']) if 'dones' in rollout_buffer else 'N/A'}"
                    )
                    if "dones" in rollout_buffer and len(rollout_buffer["dones"]) > 0:
                        print(f"  First 5 dones: {rollout_buffer['dones'][:5]}")
                    print(f"  total_rollout_size: {total_rollout_size}")

                dones_batch = torch.tensor(
                    rollout_buffer["dones"], dtype=torch.float, device=device
                )

                # 【调试】检查创建后的tensor
                if training_step <= 3:
                    print(f"\n[DONES DEBUG] After creating dones_batch:")
                    print(f"  dones_batch.shape: {dones_batch.shape}")
                    print(f"  dones_batch.numel(): {dones_batch.numel()}")

                if TrainParams.EXECUTION_MODE == "smdp":
                    taus_batch = torch.tensor(
                        rollout_buffer["taus"], dtype=torch.float, device=device
                    )
                    gammas_batch = torch.exp(-TrainParams.BETA * taus_batch)
                else:  # 'mdp'
                    gammas_batch = torch.full(
                        (total_rollout_size,), TrainParams.GAMMA, device=device
                    )
                GAE_LAMBDA = 0.95

                # 【DEBUG】检查当前是否已经有 worker 端计算好的 GAE
                if training_step <= 3:
                    print("\n[GAE DEBUG Driver] Before GAE branch:")
                    print(
                        f"  len(rollout_buffer['reward']): {len(rollout_buffer['reward'])}"
                    )
                    print(
                        f"  len(rollout_buffer['advantage']): {len(rollout_buffer['advantage'])}"
                    )
                    print(
                        f"  len(rollout_buffer['rewards']): {len(rollout_buffer['rewards'])}"
                    )
                    print(f"  total_rollout_size: {total_rollout_size}")

                # 检查worker是否已经计算了GAE
                if rollout_buffer["advantage"] and rollout_buffer["reward"]:
                    # Worker已经计算了GAE，直接使用
                    # 注意：stack后可能是[N, 1]形状，需要squeeze到[N]

                    # 【形状诊断】打印stack之前的形状
                    if training_step <= 2:
                        print(f"\n[SHAPE_DIAG Driver] Before stack:")
                        print(
                            f"  rollout_buffer['reward'][0] type: {type(rollout_buffer['reward'][0])}"
                        )
                        if torch.is_tensor(rollout_buffer["reward"][0]):
                            print(
                                f"  rollout_buffer['reward'][0] shape: {rollout_buffer['reward'][0].shape}"
                            )
                        print(
                            f"  len(rollout_buffer['reward']): {len(rollout_buffer['reward'])}"
                        )

                    advantages = torch.stack(
                        [
                            adv
                            if torch.is_tensor(adv)
                            else torch.tensor(adv, device=device)
                            for adv in rollout_buffer["advantage"]
                        ]
                    ).to(device)
                    returns = torch.stack(
                        [
                            ret
                            if torch.is_tensor(ret)
                            else torch.tensor(ret, device=device)
                            for ret in rollout_buffer["reward"]
                        ]
                    ).to(device)

                    # 【形状诊断】打印stack之后、squeeze之前的形状
                    if training_step <= 2:
                        print(f"\n[SHAPE_DIAG Driver] After stack, before squeeze:")
                        print(
                            f"  advantages shape: {advantages.shape}, dim: {advantages.dim()}"
                        )
                        print(f"  returns shape: {returns.shape}, dim: {returns.dim()}")

                    # 确保形状是[N]而不是[N, 1]
                    if advantages.dim() > 1:
                        advantages = advantages.squeeze(-1)
                    if returns.dim() > 1:
                        returns = returns.squeeze(-1)

                    # 【形状诊断】打印squeeze之后的形状
                    if training_step <= 2:
                        print(f"\n[SHAPE_DIAG Driver] After squeeze:")
                        print(f"  advantages shape: {advantages.shape}")
                        print(f"  returns shape: {returns.shape}")
                else:
                    # Worker没有计算GAE，driver来计算（向后兼容）
                    # 重新计算values用于GAE
                    with torch.no_grad():
                        _, values_batch = global_network(
                            task_inputs_batch,
                            agent_inputs_batch,
                            global_mask_batch,
                            index_batch,
                            ltl_info_batch,
                        )
                        values_batch = values_batch.squeeze(-1)

                        # 获取next state的values用于bootstrap
                        next_task_inputs_batch = torch.stack(
                            rollout_buffer["next_task_info"]
                        ).to(device)
                        next_agent_inputs_batch = torch.stack(
                            rollout_buffer["next_agents_info"]
                        ).to(device)
                        next_global_mask_batch = torch.stack(
                            rollout_buffer["next_mask"]
                        ).to(device)
                        next_index_batch = torch.stack(rollout_buffer["next_index"]).to(
                            device
                        )

                        if rollout_buffer["next_ltl_info"]:
                            # 检查是否为模式C（字典格式）
                            if isinstance(rollout_buffer["next_ltl_info"][0], dict):
                                # 模式C：分别stack字典的每个组件
                                next_ltl_info_batch = {
                                    "feasibility": torch.stack(
                                        [
                                            item["feasibility"]
                                            for item in rollout_buffer["next_ltl_info"]
                                        ]
                                    ).to(device),
                                    "edge_index": torch.stack(
                                        [
                                            item["edge_index"]
                                            for item in rollout_buffer["next_ltl_info"]
                                        ]
                                    ).to(device),
                                    "edge_attr": torch.stack(
                                        [
                                            item["edge_attr"]
                                            for item in rollout_buffer["next_ltl_info"]
                                        ]
                                    ).to(device),
                                }
                            else:
                                # 模式A/B：直接stack tensor
                                next_ltl_info_batch = torch.stack(
                                    rollout_buffer["next_ltl_info"]
                                ).to(device)
                        else:
                            next_ltl_info_batch = None

                        _, next_values_batch = global_network(
                            next_task_inputs_batch,
                            next_agent_inputs_batch,
                            next_global_mask_batch,
                            next_index_batch,
                            next_ltl_info_batch,
                        )
                        next_values_batch = next_values_batch.squeeze(-1)

                    rewards_batch = torch.tensor(
                        rollout_buffer["rewards"], dtype=torch.float, device=device
                    )
                    # dones_batch和gammas_batch已在上面定义

                    # 【Reward GAE】使用GAE计算优势函数
                    advantages = torch.zeros_like(rewards_batch)
                    gae = 0

                    for t in reversed(range(total_rollout_size)):
                        if t == total_rollout_size - 1:
                            next_value = next_values_batch[t]
                        else:
                            next_value = values_batch[t + 1]

                        # 使用时间依赖折扣（SMDP）或固定折扣（MDP）
                        gamma_t = gammas_batch[t]
                        delta = (
                            rewards_batch[t]
                            + gamma_t * next_value * (1 - dones_batch[t])
                            - values_batch[t]
                        )
                        gae = delta + gamma_t * GAE_LAMBDA * (1 - dones_batch[t]) * gae
                        advantages[t] = gae

                    # 计算returns
                    returns = advantages + values_batch

                # === Stage 6: PPO Update with multiple epochs ===
                update_start = time.time()

                # 【形状诊断说明】仅在前2个训练步显示
                if training_step <= 2:
                    print(f"\n{'=' * 80}")
                    print(f"[形状诊断说明] Training Step {training_step}:")
                    print(f"  接下来会打印详细的tensor形状信息，用于验证修复是否正确")
                    print(f"  ✅ 期望: 所有形状是 [N] (1维)")
                    print(f"  ❌ 错误: 如果看到 [N, 1] (2维)，说明需要进一步修复")
                    print(f"{'=' * 80}\n")

                print(f"[TRAIN] 开始PPO更新 ({TrainParams.PPO_EPOCHS} 轮)...")
                actions_batch = torch.stack(rollout_buffer["actions"]).to(device)
                old_log_prob_batch = torch.stack(
                    [
                        lp if torch.is_tensor(lp) else torch.tensor(lp, device=device)
                        for lp in rollout_buffer["old_log_prob"]
                    ]
                ).to(device)

                # 【方法1：分头独立PPO】创建各头的old_log_prob batch
                if (
                    TrainParams.USE_MULTI_HEAD_PPO
                    and "old_log_prob_type" in rollout_buffer
                ):
                    old_log_prob_type_batch = torch.stack(
                        [
                            lp
                            if torch.is_tensor(lp)
                            else torch.tensor(lp, device=device)
                            for lp in rollout_buffer["old_log_prob_type"]
                        ]
                    ).to(device)
                    old_log_prob_dest_batch = torch.stack(
                        [
                            lp
                            if torch.is_tensor(lp)
                            else torch.tensor(lp, device=device)
                            for lp in rollout_buffer["old_log_prob_dest"]
                        ]
                    ).to(device)
                    old_log_prob_cargo_batch = torch.stack(
                        [
                            lp
                            if torch.is_tensor(lp)
                            else torch.tensor(lp, device=device)
                            for lp in rollout_buffer["old_log_prob_cargo"]
                        ]
                    ).to(device)
                else:
                    old_log_prob_type_batch = None
                    old_log_prob_dest_batch = None
                    old_log_prob_cargo_batch = None

                cargo_mask_batch = (
                    torch.stack(rollout_buffer["cargo_mask"]).to(device)
                    if rollout_buffer["cargo_mask"]
                    else None
                )
                action_type_mask_batch = (
                    torch.stack(rollout_buffer["action_type_mask"]).to(device)
                    if rollout_buffer["action_type_mask"]
                    else None
                )
                # quantity_mask is no longer used since we auto-load max capacity

                # 先保留一份基于奖励的优势值（作为主性能信号）
                advantages_reward = advantages.clone().detach()

                # ==================== Episode-level Risk-Sensitive / CVAR ====================
                batch_mean_cost = torch.tensor(0.0, device=device)

                if TrainParams.LTL_CONSTRAINT_TYPE == "CVAR_SMDP":
                    # 【步骤1】筛选完整episode的风险指数
                    complete_episode_ids = [
                        ep_id
                        for ep_id, is_done in global_episode_done_map.items()
                        if is_done
                    ]
                    complete_C_samples = [
                        global_episode_risk_map[ep_id] for ep_id in complete_episode_ids
                    ]

                    if len(complete_C_samples) == 0:
                        print("[Episode-CVAR] Warning: No complete episodes collected!")
                        # Fallback: 标准化reward advantages
                        if advantages_reward.numel() > 1:
                            advantages = (
                                advantages_reward - advantages_reward.mean()
                            ) / (advantages_reward.std() + 1e-8)
                        else:
                            advantages = advantages_reward
                    else:
                        # 【步骤2】计算经验分位数（VaR）和CVaR
                        C_samples = torch.tensor(
                            complete_C_samples, dtype=torch.float32, device=device
                        )
                        nu = torch.quantile(C_samples, q=1 - TrainParams.CVAR_ALPHA)

                        # CVaR_α(C) = E[C | C ≥ ν]
                        tail_mask = C_samples >= nu
                        if tail_mask.sum() > 0:
                            cvar = C_samples[tail_mask].mean()
                        else:
                            cvar = nu  # Fallback

                        batch_mean_cost = C_samples.mean()

                        # 【调试】打印统计信息
                        if training_step <= 2:
                            print(f"\n[Episode-CVAR] Risk Statistics:")
                            print(
                                f"  完整episode数: {len(complete_episode_ids)} / {len(global_episode_risk_map)} 总episodes"
                            )
                            print(
                                f"  C_episode范围: [{C_samples.min().item():.6f}, {C_samples.max().item():.6f}]"
                            )
                            print(
                                f"  C_episode均值: {batch_mean_cost.item():.6f} (std={C_samples.std().item():.6f})"
                            )
                            print(
                                f"  VaR_{TrainParams.CVAR_ALPHA}(C) = ν: {nu.item():.6f}"
                            )
                            print(
                                f"  CVaR_{TrainParams.CVAR_ALPHA}(C): {cvar.item():.6f}"
                            )
                            print(
                                f"  尾部样本数: {tail_mask.sum().item()}/{len(C_samples)}"
                            )

                        # 【步骤3】计算per-episode风险权重 Δ_i
                        # 标准CVaR梯度形式：Δ_i = (C_i - ν)_+ / α
                        #   - (C_i - ν)_+ = max(C_i - ν, 0)：尾部超额
                        #   - α = CVAR_ALPHA：尾部概率（例如0.1表示最坏10%）
                        # 理论语义：风险越高的episode，梯度推力越强
                        episode_risk_weights = {}  # {episode_id: Δ_i}
                        for ep_id in complete_episode_ids:
                            C_i = global_episode_risk_map[ep_id]
                            # 标准形式：按超出VaR的程度加权
                            excess = max(C_i - nu.item(), 0.0)
                            episode_risk_weights[ep_id] = (
                                excess / TrainParams.CVAR_ALPHA
                            )

                        # 【调试】打印权重分布
                        if training_step <= 2:
                            num_risky_episodes = sum(
                                1 for w in episode_risk_weights.values() if w > 0
                            )
                            print(
                                f"  风险episode数: {num_risky_episodes}/{len(episode_risk_weights)}"
                            )

                            # 【新增】打印Δ_i的详细统计
                            if episode_risk_weights:
                                weights_values = list(episode_risk_weights.values())
                                weights_tensor = torch.tensor(weights_values)
                                nonzero_weights = [w for w in weights_values if w > 0]

                                print(f"  Δ_i统计（所有episode）:")
                                print(f"    - Min: {weights_tensor.min().item():.6f}")
                                print(f"    - Max: {weights_tensor.max().item():.6f}")
                                print(f"    - Mean: {weights_tensor.mean().item():.6f}")
                                print(f"    - Std: {weights_tensor.std().item():.6f}")

                                if nonzero_weights:
                                    nonzero_tensor = torch.tensor(nonzero_weights)
                                    print(f"  Δ_i统计（仅风险episode，Δ>0）:")
                                    print(
                                        f"    - Min: {nonzero_tensor.min().item():.6f}"
                                    )
                                    print(
                                        f"    - Max: {nonzero_tensor.max().item():.6f}"
                                    )
                                    print(
                                        f"    - Mean: {nonzero_tensor.mean().item():.6f}"
                                    )
                                    print(
                                        f"    - Std: {nonzero_tensor.std().item():.6f}"
                                    )

                                # 打印前5个episode的权重样例
                                sample_items = list(episode_risk_weights.items())[:5]
                                print(f"  Episode权重样例 (episode_id: Δ_i):")
                                for ep_id, weight in sample_items:
                                    C_i = global_episode_risk_map[ep_id]
                                    print(
                                        f"    EP {ep_id}: C={C_i:.6f}, Δ={weight:.6f}"
                                    )

                        # 【步骤4】映射权重到transitions
                        # 对于每个transition t，查找其所属episode的权重 Δ_{episode(t)}
                        episode_ids_tensor = rollout_buffer["episode_id"]  # List[int]
                        risk_advantages = torch.zeros_like(advantages_reward)

                        for t in range(len(episode_ids_tensor)):
                            ep_id = episode_ids_tensor[t]
                            # 只对完整episode的transitions应用权重
                            if ep_id in episode_risk_weights:
                                risk_advantages[t] = episode_risk_weights[ep_id]
                            else:
                                # 截断episode的transitions：权重为0（不参与CVaR梯度）
                                risk_advantages[t] = 0.0

                        # 【调试】检查权重应用情况
                        if training_step <= 2:
                            print(f"  Transition数: {len(risk_advantages)}")
                            print(
                                f"  非零risk_advantages数: {(risk_advantages > 0).sum().item()}"
                            )
                            print(f"  Risk advantages统计:")
                            print(f"    - Min: {risk_advantages.min().item():.6f}")
                            print(f"    - Max: {risk_advantages.max().item():.6f}")
                            print(f"    - Mean: {risk_advantages.mean().item():.6f}")
                            print(f"    - Std: {risk_advantages.std().item():.6f}")

                            # 打印非零risk_advantages的统计（只看风险transitions）
                            nonzero_risk_adv = risk_advantages[risk_advantages > 0]
                            if nonzero_risk_adv.numel() > 0:
                                print(
                                    f"  Risk advantages统计（仅非零，即风险transitions）:"
                                )
                                print(f"    - Min: {nonzero_risk_adv.min().item():.6f}")
                                print(f"    - Max: {nonzero_risk_adv.max().item():.6f}")
                                print(
                                    f"    - Mean: {nonzero_risk_adv.mean().item():.6f}"
                                )
                                print(f"    - Std: {nonzero_risk_adv.std().item():.6f}")

                        # 【步骤5】标准化reward advantage
                        if advantages_reward.numel() > 1:
                            advantages_reward_normalized = (
                                advantages_reward - advantages_reward.mean()
                            ) / (advantages_reward.std() + 1e-8)
                        else:
                            advantages_reward_normalized = advantages_reward

                        # 【步骤6】构造最终advantages
                        # 理论正确形式：A_total = A_reward - λ·Δ_{episode(t)}
                        # 注意：这里Δ是per-transition的，不是常数！
                        advantages = (
                            advantages_reward_normalized
                            - TrainParams.CVAR_RISK_LAMBDA * risk_advantages
                        )

                        # 【调试】打印最终advantages
                        if training_step <= 2:
                            print(
                                f"  λ (risk aversion): {TrainParams.CVAR_RISK_LAMBDA}"
                            )
                            print(f"  Advantages统计:")
                            print(
                                f"    - Reward (normalized): mean={advantages_reward_normalized.mean().item():.6f}, std={advantages_reward_normalized.std().item():.6f}"
                            )
                            print(
                                f"    - Risk term (λ·Δ): mean={(-TrainParams.CVAR_RISK_LAMBDA * risk_advantages).mean().item():.6f}, std={(-TrainParams.CVAR_RISK_LAMBDA * risk_advantages).std().item():.6f}"
                            )
                            print(
                                f"    - Combined: mean={advantages.mean().item():.6f}, std={advantages.std().item():.6f}"
                            )

                elif TrainParams.LTL_CONSTRAINT_TYPE == "RISK_SENSITIVE_SMDP":
                    # ==================== 路线1: Episode-Level Tail Regularization (IJCAI Eq. 158-169) ====================
                    # 【实现策略（正确的理论框架）】：
                    #
                    # 1. NEU模式（Baseline）：
                    #    - 标准PPO，使用所有transitions（包括截断的rollout）
                    #    - GAE + bootstrap处理截断episodes
                    #    - 目标：min E[T]
                    #
                    # 2. Risk模式（本模式）：
                    #    - 标准PPO部分：使用所有transitions（同NEU）
                    #      → Policy loss基于所有transitions的advantages
                    #      → Value loss基于所有transitions的returns
                    #    - 额外的Risk loss：只在完整episodes上计算
                    #      → 截断episodes的transitions: weight=0（不参与risk_loss）
                    #      → 完整episodes的transitions: weight=exp(α·C̄_i)/mean(exp(α·C̄))
                    #      → 只对weight>0的transitions计算risk_loss
                    #    - 目标：min E[T] + μ·log E[exp(α·C̄)]
                    #
                    # 【关键区别】：
                    #    - Policy optimization: 两种模式都用所有transitions（公平对比）
                    #    - Risk regularization: 只用完整episodes（理论严格性）
                    #    - 截断transitions贡献于policy learning，但不影响risk loss
                    #
                    # 【理论依据】：
                    #    - IJCAI.md第154行："For each **completed episode** τ"
                    #    - IJCAI.md第156行："Given a batch of **N complete episodes**"
                    #    - REINFORCE梯度：E[w(τ)·∇log p_θ(τ)] 要求τ是完整轨迹
                    #
                    # 【目标函数（IJCAI Eq. 48）】：
                    #   J(π) = E_π[T] + μ·log E_π[exp(α·C̄(τ))]
                    #
                    #   其中：
                    #   T: makespan（完工时间）
                    #   C̄(τ): 归一化期望故障率 = (1/n)·Σ_i p_i(τ) ∈ [0,1]，p_i = 1 - exp(-H_i)
                    #          表示车队平均故障率（fleet average failure rate），跨问题规模稳定
                    #   α > 0: 尾部敏感系数（无量纲，作用于C̄ ∈ [0,1]）
                    #   μ > 0: trade-off系数（单位：秒/百分点）
                    #
                    # 【实现细节（IJCAI Section 4.1）】：
                    #   1. Batch log-MGF estimate (Eq. 158-159):
                    #      R̂_α = log(1/N · Σ exp(α·C̄_j))  [只对完整episodes]
                    #
                    #   2. Episode weights (Eq. 161-163):
                    #      w_i = exp(α·C̄_i) / mean(exp(α·C̄))  [只对完整episodes]
                    #
                    #   3. PPO surrogate with risk regularization (Eq. 167-170):
                    #      L(θ) = L_PPO(θ) + μ·R̂_α
                    #      其中 L_PPO 使用所有transitions，R̂_α 只用完整episodes
                    # ================================================================================================

                    # 【步骤1】收集完整episodes的C值和T值
                    complete_episode_ids = [
                        ep_id
                        for ep_id, is_done in global_episode_done_map.items()
                        if is_done
                    ]

                    if len(complete_episode_ids) == 0:
                        print("[Route1] Warning: No complete episodes collected!")
                        # Fallback: 标准化advantages
                        if advantages_reward.numel() > 1:
                            advantages = (
                                advantages_reward - advantages_reward.mean()
                            ) / (advantages_reward.std() + 1e-8)
                        else:
                            advantages = advantages_reward
                        batch_mean_cost = torch.tensor(0.0, device=device)
                        batch_mean_makespan = torch.tensor(0.0, device=device)
                        logmeanexp_alpha_C = torch.tensor(0.0, device=device)
                        # 【新增】统计信息（空batch）
                        C_bar_std = torch.tensor(0.0, device=device)
                        C_bar_min = torch.tensor(0.0, device=device)
                        C_bar_max = torch.tensor(0.0, device=device)
                        episode_weight_min = torch.tensor(0.0, device=device)
                        episode_weight_max = torch.tensor(0.0, device=device)
                        episode_weight_ratio = torch.tensor(0.0, device=device)
                        ess_value = torch.tensor(0.0, device=device)
                        ess_ratio_value = torch.tensor(0.0, device=device)
                    else:
                        # 【步骤2】提取C_rel值（Relative Excess Hazard）和T值（makespan）
                        # C_rel = (H_actual / H_optimal) - 1，衡量负载不均衡的百分比
                        # 【诊断】检查maps的状态
                        if training_step <= 2:
                            print(
                                f"\n[DIAGNOSTIC Extraction] Before extracting C_rel/T values:"
                            )
                            print(f"  complete_episode_ids: {complete_episode_ids}")
                            print(
                                f"  global_episode_C_map keys: {list(global_episode_C_map.keys())}"
                            )
                            print(
                                f"  global_episode_T_map keys: {list(global_episode_T_map.keys())}"
                            )
                            print(
                                f"  global_episode_C_map contents: {global_episode_C_map}"
                            )
                            print(
                                f"  global_episode_T_map contents: {global_episode_T_map}"
                            )

                        C_values = []
                        T_values = []
                        for ep_id in complete_episode_ids:
                            C = global_episode_C_map.get(ep_id, 0.0)
                            T = global_episode_T_map.get(ep_id, 0.0)

                            if training_step <= 2:
                                print(
                                    f"  Extracting ep_id={ep_id}: C_rel={C:.6f}, T={T:.2f}"
                                )

                            C_values.append(C)
                            T_values.append(T)

                        C_tensor = torch.tensor(
                            C_values, dtype=torch.float32, device=device
                        )
                        T_tensor = torch.tensor(
                            T_values, dtype=torch.float32, device=device
                        )

                        # ========================================================================
                        # 【均值型多目标优化】直接优化C_rel的期望
                        #
                        # 目标函数：J(π) = E_π[T] + μ·E_π[C_rel]
                        #
                        # C_rel = (H_actual / H_optimal) - 1：相对超额hazard
                        # - C_rel = 0：完美负载均衡（所有agent工作时间相等）
                        # - C_rel > 0：存在负载不均衡，百分比形式（如0.5表示比最优多50%）
                        # - 完全scale-invariant：与任务难度W无关
                        #
                        # 优化方法：
                        # - 不使用exponential weights（避免高方差和目标偏移）
                        # - 直接最小化batch内C_rel的均值（稳定且高效）
                        # - μ控制makespan优化与负载均衡之间的trade-off
                        # ========================================================================
                        mu = TrainParams.RISK_MU

                        # 用于日志记录
                        batch_mean_cost = C_tensor.mean()
                        batch_mean_makespan = T_tensor.mean()

                        # 计算统计信息用于日志
                        C_bar_std = C_tensor.std()
                        C_bar_min = C_tensor.min()
                        C_bar_max = C_tensor.max()

                        # 【调试】打印统计信息
                        if training_step <= 2:
                            print(f"\n[Mean-Based Multi-Objective] Episode Statistics:")
                            print(f"  完整episode数: {len(complete_episode_ids)}")
                            print(f"  μ (trade-off coefficient): {mu}")
                            print(
                                f"  C_rel范围 (Relative Excess Hazard): [{C_tensor.min().item():.4f}, {C_tensor.max().item():.4f}]"
                            )
                            print(
                                f"  C_rel均值: {batch_mean_cost.item():.4f} (std={C_tensor.std().item():.4f})"
                            )
                            print(
                                f"  T值范围 (Makespan): [{T_tensor.min().item():.2f}, {T_tensor.max().item():.2f}]"
                            )
                            print(
                                f"  T值均值: {batch_mean_makespan.item():.2f} (std={T_tensor.std().item():.2f})"
                            )

                            # 打印前5个episode的详细信息
                            print(f"  Episode样例 (前5个):")
                            for i in range(min(5, len(complete_episode_ids))):
                                ep_id = complete_episode_ids[i]
                                C_val = C_values[i]
                                T_val = T_values[i]
                                print(
                                    f"    EP{ep_id}: C_rel={C_val:.4f} ({C_val * 100:.1f}%), T={T_val:.2f}"
                                )

                        # 【均值型优化】不需要episode权重，直接标准化advantages
                        # Advantages仅反映makespan优化目标（奖励回报）
                        if advantages_reward.numel() > 1:
                            advantages = (
                                advantages_reward - advantages_reward.mean()
                            ) / (advantages_reward.std() + 1e-8)
                        else:
                            advantages = advantages_reward

                        # 【调试】打印advantages统计
                        if training_step <= 2:
                            print(f"\n  Advantages Statistics:")
                            print(
                                f"    - Reward advantages (normalized): mean={advantages.mean().item():.6f}, std={advantages.std().item():.6f}"
                            )
                            print(f"\n  Implementation Strategy:")
                            print(
                                f"    ✓ Policy loss (PPO): Standard clipped surrogate with reward advantages"
                            )
                            print(f"    ✓ Value loss: Standard MSE on returns")
                            print(
                                f"    ✓ Risk regularization: μ·E[C_rel] added to total loss"
                            )
                            print(f"    → Objective: min E[T] + μ·E[C_rel]")
                            print(f"    → No episode weights, direct mean optimization")

                        # 【均值型优化】创建per-transition的C_rel系数用于REINFORCE梯度
                        # 虽然是"均值型"，但仍需REINFORCE来传播梯度
                        # risk gradient = μ·E[C_rel·∇log π] = μ·mean(C_rel[i]·∇log π for episode i)
                        episode_ids_tensor = rollout_buffer["episode_id"]  # List[int]
                        transition_C_rel = torch.zeros(
                            len(episode_ids_tensor), device=device
                        )

                        # 创建episode_id到C_rel的映射
                        ep_id_to_C_rel = {}
                        for i, ep_id in enumerate(complete_episode_ids):
                            ep_id_to_C_rel[ep_id] = C_values[i]

                        # 广播C_rel到每个transition
                        for t in range(len(episode_ids_tensor)):
                            ep_id = episode_ids_tensor[t]
                            if ep_id in ep_id_to_C_rel:
                                transition_C_rel[t] = ep_id_to_C_rel[ep_id]
                            else:
                                # 截断episode：C_rel=0（不参与risk gradient）
                                transition_C_rel[t] = 0.0

                        # 【调试】打印transition_C_rel统计
                        if training_step <= 2:
                            num_complete_trans = (transition_C_rel != 0).sum().item()
                            num_truncated_trans = (transition_C_rel == 0).sum().item()
                            print(f"\n  Transition C_rel Statistics:")
                            print(f"    - Total transitions: {len(transition_C_rel)}")
                            print(
                                f"    - Complete episodes: {num_complete_trans} ({num_complete_trans / len(transition_C_rel) * 100:.1f}%)"
                            )
                            print(
                                f"    - Truncated episodes: {num_truncated_trans} ({num_truncated_trans / len(transition_C_rel) * 100:.1f}%)"
                            )
                            if num_complete_trans > 0:
                                C_rel_nonzero = transition_C_rel[transition_C_rel != 0]
                                print(
                                    f"    - C_rel range (complete): [{C_rel_nonzero.min().item():.4f}, {C_rel_nonzero.max().item():.4f}]"
                                )
                                print(
                                    f"    - C_rel mean (complete): {C_rel_nonzero.mean().item():.4f}"
                                )

                else:
                    # 其他模式：只使用reward advantages
                    # 归一化advantages（在整个batch上）
                    if advantages.numel() > 1:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )

                # ==================== 软约束支持：计算成本advantages ====================
                # 【预定义变量】确保变量在所有代码路径中都存在
                costs_batch = None
                cost_advantages = None
                # batch_mean_cost 对于非CVAR/Risk-Sensitive模式默认初始化
                if TrainParams.LTL_CONSTRAINT_TYPE not in [
                    "CVAR_SMDP",
                    "RISK_SENSITIVE_SMDP",
                ]:
                    batch_mean_cost = torch.tensor(0.0, device=device)

                # 检查条件：1) 不是HARD且不是CVAR且不是Risk-Sensitive模式 2) costs字段存在 3) costs列表非空
                has_costs = (
                    TrainParams.LTL_CONSTRAINT_TYPE
                    not in ["HARD", "CVAR_SMDP", "RISK_SENSITIVE_SMDP"]
                    and "costs" in rollout_buffer
                    and rollout_buffer["costs"] is not None
                    and len(rollout_buffer["costs"]) > 0
                )

                # 【额外安全检查】确保costs数据可用
                if has_costs:
                    try:
                        # 【调试】检查costs内容
                        if training_step <= 3:
                            costs_list = rollout_buffer["costs"]
                            print(f"\n[COST DEBUG] costs_list info:")
                            print(f"  Type: {type(costs_list)}")
                            print(f"  Length: {len(costs_list)}")
                            print(f"  First 5 elements: {costs_list[:5]}")
                            print(
                                f"  Element types: {[type(x) for x in costs_list[:5]]}"
                            )
                            # 检查是否有None
                            none_count = sum(1 for x in costs_list if x is None)
                            print(f"  None count: {none_count} / {len(costs_list)}")

                        # 过滤掉None值并转换为float
                        costs_clean = [
                            float(x) if x is not None else 0.0
                            for x in rollout_buffer["costs"]
                        ]

                        # 【调试】检查转换后的数据
                        if training_step <= 3:
                            print(f"\n[COST DEBUG] After cleaning:")
                            print(f"  costs_clean length: {len(costs_clean)}")
                            print(f"  First 5: {costs_clean[:5]}")
                            print(f"  Types: {[type(x) for x in costs_clean[:5]]}")

                        # 获取成本数据 - 使用numpy作为中间步骤确保正确性
                        costs_np = np.array(costs_clean, dtype=np.float32)
                        costs_batch = torch.from_numpy(costs_np).to(device)

                        # 【调试】检查tensor
                        if training_step <= 3:
                            print(f"\n[COST DEBUG] Tensor created:")
                            print(f"  costs_batch.shape: {costs_batch.shape}")
                            print(f"  costs_batch.numel(): {costs_batch.numel()}")
                            print(f"  costs_batch device: {costs_batch.device}")
                            print(f"  First 5 values: {costs_batch[:5]}")

                        # 立即检查size
                        if costs_batch.numel() == 0:
                            print(
                                f"[ERROR] costs_batch is empty tensor! Skipping cost calculation."
                            )
                            print(
                                f"[ERROR] Original costs_list length: {len(rollout_buffer['costs'])}"
                            )
                            has_costs = False
                            costs_batch = None  # 显式设置为None
                            cost_advantages = None
                            batch_mean_cost = torch.tensor(0.0, device=device)
                    except Exception as e:
                        print(f"[ERROR] Failed to create costs_batch: {e}")
                        print(
                            f"[ERROR] costs_list length: {len(rollout_buffer['costs']) if rollout_buffer['costs'] else 'N/A'}"
                        )
                        if rollout_buffer["costs"]:
                            print(
                                f"[ERROR] First element: {rollout_buffer['costs'][0]}"
                            )
                        import traceback

                        traceback.print_exc()
                        has_costs = False
                        costs_batch = None  # 显式设置为None
                        cost_advantages = None
                        batch_mean_cost = torch.tensor(0.0, device=device)

                # 【最终安全检查】在使用costs_batch之前，triple-check
                if has_costs and costs_batch is not None and costs_batch.numel() > 0:
                    # 验证成本数据长度与rollout长度一致
                    if costs_batch.shape[0] != total_rollout_size:
                        print(
                            f"[WARNING] Cost batch size mismatch: {costs_batch.shape[0]} != {total_rollout_size}"
                        )
                        print(f"[WARNING] Skipping cost calculation for this batch")
                        cost_advantages = None
                        batch_mean_cost = torch.tensor(0.0, device=device)
                    else:
                        # 为成本计算advantages
                        # 注意：由于网络只有一个value_head（用于奖励估计），我们使用蒙特卡洛回报
                        # 这是理论上合理的方法，避免混淆奖励值函数和成本值函数
                        cost_advantages = torch.zeros_like(costs_batch)

                        # 计算成本的折扣回报（Monte-Carlo returns）
                        cost_returns = torch.zeros_like(costs_batch)
                        running_return = 0

                        # 【关键调试】检查costs_batch在使用前的状态
                        if training_step <= 3:
                            print(f"\n[COST DEBUG] ===== RIGHT BEFORE FOR LOOP =====")
                            print(
                                f"  costs_batch is defined: {'costs_batch' in locals()}"
                            )
                            print(f"  costs_batch.shape: {costs_batch.shape}")
                            print(f"  costs_batch.numel(): {costs_batch.numel()}")
                            print(f"  total_rollout_size: {total_rollout_size}")
                            print(f"  has_costs: {has_costs}")
                            print(f"  gammas_batch.shape: {gammas_batch.shape}")
                            print(f"  dones_batch.shape: {dones_batch.shape}")
                            print(f"===========================================\n")

                        for t in reversed(range(total_rollout_size)):
                            # 使用SMDP时间依赖折扣
                            gamma_t = gammas_batch[t]
                            running_return = costs_batch[
                                t
                            ] + gamma_t * running_return * (1 - dones_batch[t])
                            cost_returns[t] = running_return

                        # Cost advantages = 实际成本回报
                        # 不使用baseline（因为没有独立的成本值网络）
                        # 这等价于 A^c(s,a) = Q^c(s,a) - 0
                        cost_advantages = cost_returns

                        # 归一化成本advantages
                        cost_advantages = (cost_advantages - cost_advantages.mean()) / (
                            cost_advantages.std() + 1e-8
                        )

                        # 累加总成本用于lambda更新（保持为tensor）
                        batch_mean_cost = costs_batch.mean()

                        # 【调试】打印成本统计（仅在前几步）
                        if training_step <= 3:
                            print(f"\n[COST DEBUG] Training step {training_step}:")
                            print(f"  Costs batch size: {costs_batch.shape[0]}")
                            print(f"  Mean cost: {batch_mean_cost.item():.6f}")
                            print(
                                f"  Cost range: [{costs_batch.min().item():.6f}, {costs_batch.max().item():.6f}]"
                            )
                            print(
                                f"  Non-zero costs: {(costs_batch > 0).sum().item()} / {costs_batch.shape[0]}"
                            )
                else:
                    cost_advantages = None
                    # 【修复】只对非CVAR/Risk-Sensitive模式重置batch_mean_cost
                    # CVAR/Risk-Sensitive模式已经在前面设置了batch_mean_cost，不应被覆盖
                    if TrainParams.LTL_CONSTRAINT_TYPE not in [
                        "CVAR_SMDP",
                        "RISK_SENSITIVE_SMDP",
                    ]:
                        batch_mean_cost = torch.tensor(0.0, device=device)

                    # 【调试】打印为什么没有成本数据
                    if training_step <= 3 and TrainParams.LTL_CONSTRAINT_TYPE != "HARD":
                        print(
                            f"\n[COST DEBUG] No cost data for training step {training_step}:"
                        )
                        print(
                            f"  'costs' in rollout_buffer: {'costs' in rollout_buffer}"
                        )
                        if "costs" in rollout_buffer:
                            print(
                                f"  len(rollout_buffer['costs']): {len(rollout_buffer['costs']) if rollout_buffer['costs'] else 0}"
                            )
                # =========================================================================

                batch_indices = np.arange(total_rollout_size)
                p_losses, v_losses, entropies, grads = [], [], [], []
                # 【新增】Risk-Sensitive诊断指标
                risk_losses = []
                entropy_losses = []
                policy_grad_norms = []
                risk_grad_norms = []
                # 【新增】PPO诊断指标
                explained_variances = []
                advantages_means = []
                advantages_stds = []
                kl_divergences = []
                clip_fractions = []
                # 【新增】分头熵监控
                entropy_type_list = []
                entropy_dest_list = []
                entropy_cargo_list = []

                for epoch in range(TrainParams.PPO_EPOCHS):
                    print(f"[PPO]   Epoch {epoch + 1}/{TrainParams.PPO_EPOCHS}...")
                    np.random.shuffle(batch_indices)

                    for start in range(
                        0, total_rollout_size, TrainParams.PPO_MINIBATCH_SIZE
                    ):
                        end = min(
                            start + TrainParams.PPO_MINIBATCH_SIZE, total_rollout_size
                        )
                        mb_indices = batch_indices[start:end]

                        # 准备minibatch数据
                        mb_task = task_inputs_batch[mb_indices]
                        mb_agents = agent_inputs_batch[mb_indices]
                        mb_mask = global_mask_batch[mb_indices]
                        mb_index = index_batch[mb_indices]
                        mb_actions = actions_batch[mb_indices]
                        mb_advantages = advantages[mb_indices]
                        mb_returns = returns[mb_indices]
                        mb_old_log_probs = old_log_prob_batch[mb_indices]

                        # 处理ltl_info的minibatch索引（支持模式C的字典格式）
                        if ltl_info_batch is not None:
                            if isinstance(ltl_info_batch, dict):
                                # 【方案B修复】模式C：处理List格式的edge_index/edge_attr
                                mb_ltl_info = {
                                    "feasibility": ltl_info_batch["feasibility"][
                                        mb_indices
                                    ],
                                    "edge_index": [
                                        ltl_info_batch["edge_index"][i]
                                        for i in mb_indices.tolist()
                                    ],  # List索引
                                    "edge_attr": [
                                        ltl_info_batch["edge_attr"][i]
                                        for i in mb_indices.tolist()
                                    ],  # List索引
                                }
                            else:
                                # 模式A/B：直接索引tensor
                                mb_ltl_info = ltl_info_batch[mb_indices]
                        else:
                            mb_ltl_info = None
                        mb_cargo_mask = (
                            cargo_mask_batch[mb_indices]
                            if cargo_mask_batch is not None
                            else None
                        )
                        mb_action_type_mask = (
                            action_type_mask_batch[mb_indices]
                            if action_type_mask_batch is not None
                            else None
                        )
                        mb_quantity_mask = None  # quantity_mask is no longer used since we auto-load max capacity
                        # 【方案一】dependency_graph不需要按minibatch索引，因为整个batch共享同一个依赖图
                        mb_dependency_graph = (
                            dependency_graph_batch
                            if dependency_graph_batch is not None
                            else None
                        )

                        # 使用当前策略重新评估actions
                        # 【方法1：分头独立PPO loss】
                        if TrainParams.USE_MULTI_HEAD_PPO:
                            # 调用evaluate_actions_split获取分头log_prob
                            eval_result = global_network.evaluate_actions_split(
                                tasks=mb_task,
                                agents=mb_agents,
                                global_mask=mb_mask,
                                index=mb_index,
                                actions=mb_actions,
                                cargo_mask=mb_cargo_mask,
                                action_type_mask=mb_action_type_mask,
                                quantity_mask=mb_quantity_mask,
                                ltl_info=mb_ltl_info,
                                dependency_graph=mb_dependency_graph,
                            )

                            # 提取分头log_prob和总log_prob
                            new_log_prob_type = eval_result["logp_type"]
                            new_log_prob_dest = eval_result["logp_dest"]
                            new_log_prob_cargo = eval_result["logp_cargo"]
                            new_log_probs = eval_result["logp_all"]  # 用于KL计算

                            # 提取分头熵（用于监控）
                            ent_type = eval_result["ent_type"]
                            ent_dest = eval_result["ent_dest"]
                            ent_cargo = eval_result["ent_cargo"]

                            # 计算归一化熵（与方案A一致）
                            # 获取动作空间大小（从global_network的last_entropy_diagnostics）
                            if hasattr(global_network, "last_entropy_diagnostics"):
                                diag = global_network.last_entropy_diagnostics
                                max_ent_type = diag["max_ent_type"]
                                max_ent_dest = diag["max_ent_dest"]
                                max_ent_cargo = diag["max_ent_cargo"]

                                new_entropy = (
                                    ent_type / max_ent_type
                                    + ent_dest / max_ent_dest
                                    + ent_cargo / max_ent_cargo
                                )
                            else:
                                # 后备方案：简单求和
                                new_entropy = ent_type + ent_dest + ent_cargo

                            new_values = eval_result["reward_value"].unsqueeze(
                                -1
                            )  # [B] -> [B, 1]
                            new_cost_quantiles = eval_result["cost_quantiles"]

                        else:
                            # 【原始方法：总log_prob】
                            (
                                new_log_probs,
                                new_entropy,
                                new_values,
                                new_cost_quantiles,
                            ) = global_network.evaluate_actions(
                                tasks=mb_task,
                                agents=mb_agents,
                                global_mask=mb_mask,
                                index=mb_index,
                                actions=mb_actions,
                                cargo_mask=mb_cargo_mask,
                                action_type_mask=mb_action_type_mask,
                                quantity_mask=mb_quantity_mask,
                                ltl_info=mb_ltl_info,
                                dependency_graph=mb_dependency_graph,
                            )

                        # 计算PPO的clipped objective
                        # 【方法1：分头独立loss（完整版本）】
                        if TrainParams.USE_MULTI_HEAD_PPO:
                            # 提取minibatch的分头old_log_prob
                            mb_old_log_prob_type = old_log_prob_type_batch[mb_indices]
                            mb_old_log_prob_dest = old_log_prob_dest_batch[mb_indices]
                            mb_old_log_prob_cargo = old_log_prob_cargo_batch[mb_indices]

                            # 计算各头的ratio（真实ratio）
                            ratio_type = torch.exp(
                                new_log_prob_type - mb_old_log_prob_type
                            )
                            ratio_dest = torch.exp(
                                new_log_prob_dest - mb_old_log_prob_dest
                            )
                            ratio_cargo = torch.exp(
                                new_log_prob_cargo - mb_old_log_prob_cargo
                            )

                            # 各头独立进行PPO裁剪
                            surr1_type = ratio_type * mb_advantages
                            surr2_type = (
                                torch.clamp(
                                    ratio_type,
                                    1.0 - TrainParams.PPO_CLIP_EPSILON,
                                    1.0 + TrainParams.PPO_CLIP_EPSILON,
                                )
                                * mb_advantages
                            )
                            loss_type = -torch.min(surr1_type, surr2_type).mean()

                            surr1_dest = ratio_dest * mb_advantages
                            surr2_dest = (
                                torch.clamp(
                                    ratio_dest,
                                    1.0 - TrainParams.PPO_CLIP_EPSILON,
                                    1.0 + TrainParams.PPO_CLIP_EPSILON,
                                )
                                * mb_advantages
                            )
                            loss_dest = -torch.min(surr1_dest, surr2_dest).mean()

                            surr1_cargo = ratio_cargo * mb_advantages
                            surr2_cargo = (
                                torch.clamp(
                                    ratio_cargo,
                                    1.0 - TrainParams.PPO_CLIP_EPSILON,
                                    1.0 + TrainParams.PPO_CLIP_EPSILON,
                                )
                                * mb_advantages
                            )
                            loss_cargo = -torch.min(surr1_cargo, surr2_cargo).mean()

                            # 计算各头的权重（基于动作空间大小归一化）
                            if hasattr(global_network, "last_entropy_diagnostics"):
                                diag = global_network.last_entropy_diagnostics
                                max_ent_type = diag["max_ent_type"]
                                max_ent_dest = diag["max_ent_dest"]
                                max_ent_cargo = diag["max_ent_cargo"]
                                total_max_ent = (
                                    max_ent_type + max_ent_dest + max_ent_cargo
                                )

                                w_type = max_ent_type / total_max_ent
                                w_dest = max_ent_dest / total_max_ent
                                w_cargo = max_ent_cargo / total_max_ent
                            else:
                                # 默认权重（均等，3个头）
                                w_type = w_dest = w_cargo = 1.0 / 3.0

                            # 加权聚合各头loss
                            policy_loss = (
                                w_type * loss_type
                                + w_dest * loss_dest
                                + w_cargo * loss_cargo
                            )

                            # 【诊断】记录分头ratio和clip fraction（前3步）
                            if training_step <= 2 and epoch == 0 and start == 0:
                                print(f"\n[MULTI_HEAD_PPO] Head Weights:")
                                print(
                                    f"  w_type={w_type:.4f}, w_dest={w_dest:.4f}, w_cargo={w_cargo:.4f}"
                                )
                                print(
                                    f"  ratio_type: mean={ratio_type.mean().item():.4f}, std={ratio_type.std().item():.4f}"
                                )
                                print(
                                    f"  ratio_dest: mean={ratio_dest.mean().item():.4f}, std={ratio_dest.std().item():.4f}"
                                )
                                print(
                                    f"  ratio_cargo: mean={ratio_cargo.mean().item():.4f}, std={ratio_cargo.std().item():.4f}"
                                )
                                print(
                                    f"  loss_type={loss_type.item():.6f}, loss_dest={loss_dest.item():.6f}"
                                )
                                print(f"  loss_cargo={loss_cargo.item():.6f}")
                                print(
                                    f"  weighted policy_loss={policy_loss.item():.6f}"
                                )

                            # 计算总ratio（用于KL计算和兼容性）
                            ratio = torch.exp(new_log_probs - mb_old_log_probs)

                        else:
                            # 【原始方法：总ratio】
                            ratio = torch.exp(new_log_probs - mb_old_log_probs)
                            surr1 = ratio * mb_advantages
                            surr2 = (
                                torch.clamp(
                                    ratio,
                                    1.0 - TrainParams.PPO_CLIP_EPSILON,
                                    1.0 + TrainParams.PPO_CLIP_EPSILON,
                                )
                                * mb_advantages
                            )
                            policy_loss = -torch.min(surr1, surr2).mean()

                        # ==================== 软约束支持：添加Lagrangian成本项 ====================
                        if (
                            TrainParams.LTL_CONSTRAINT_TYPE != "HARD"
                            and cost_advantages is not None
                        ):
                            # 获取当前minibatch的成本advantages
                            mb_cost_advantages = cost_advantages[mb_indices]

                            # Lagrangian乘子（从log_lambda恢复）
                            lagrangian_multiplier = torch.exp(log_lambda).detach()

                            # 成本的PPO clipped objective
                            # 理论依据：L(π,λ) = E[奖励] - λ·E[成本]
                            # 奖励项：-min(...) 表示最大化奖励
                            # 成本项：+λ·max(...) 表示最小化成本
                            # 使用max而非min：当cost_adv>0时选择大的ratio来更新策略远离高成本动作
                            cost_surr1 = ratio * mb_cost_advantages
                            cost_surr2 = (
                                torch.clamp(
                                    ratio,
                                    1.0 - TrainParams.PPO_CLIP_EPSILON,
                                    1.0 + TrainParams.PPO_CLIP_EPSILON,
                                )
                                * mb_cost_advantages
                            )
                            cost_loss = torch.max(cost_surr1, cost_surr2).mean()

                            # 添加到policy loss（Lagrangian形式：奖励 - λ·成本）
                            policy_loss = (
                                policy_loss + lagrangian_multiplier * cost_loss
                            )
                        # =========================================================================

                        # Value loss (确保形状匹配：都是[batch_size])
                        # new_values可能是[batch_size, 1]，需要squeeze
                        # mb_returns已经在上面确保是[batch_size]

                        # 【形状诊断】在计算value loss之前打印形状
                        if training_step <= 2 and epoch == 0 and start == 0:
                            print(f"\n[SHAPE_DIAG Driver] Value Loss Computation:")
                            print(
                                f"  new_values shape before squeeze: {new_values.shape}"
                            )
                            print(
                                f"  new_values shape after squeeze: {new_values.squeeze(-1).shape}"
                            )
                            print(f"  mb_returns shape: {mb_returns.shape}")

                        # ==================== Value Loss (可选归一化) ====================
                        # 计算原始 MSE loss
                        raw_value_loss = F.mse_loss(new_values.squeeze(-1), mb_returns)

                        if TrainParams.USE_VALUE_LOSS_NORMALIZATION:
                            # 归一化模式：L_V_norm = E[(V-R)²] / (σ_R + ε)²
                            # 好处：
                            #   1. 自动缩放value loss梯度，使其与policy loss在同一数量级
                            #   2. 适应reward scale变化（LTL开关改变reward不需要重新调参）
                            #   3. 最优解不变（仍然是 V → R）
                            returns_running_stats.update(mb_returns)
                            running_std = returns_running_stats.get_std()
                            value_loss = raw_value_loss / (running_std**2 + 1e-8)
                        else:
                            # 标准模式：直接使用MSE loss（与SMDP4_another2一致）
                            value_loss = raw_value_loss
                            running_std = torch.tensor(
                                1.0, device=device
                            )  # 用于日志打印

                        # 【新增】计算explained variance用于诊断value function质量
                        returns_var = mb_returns.var()
                        residual_var = (mb_returns - new_values.squeeze(-1)).var()
                        explained_var = 1.0 - residual_var / (returns_var + 1e-8)
                        explained_variances.append(explained_var.item())

                        # 【新增】计算advantages统计量用于诊断GAE质量
                        advantages_means.append(mb_advantages.mean().item())
                        advantages_stds.append(mb_advantages.std().item())

                        # 【新增】计算KL散度用于诊断策略更新幅度
                        kl_div = (mb_old_log_probs - new_log_probs).mean()
                        kl_divergences.append(kl_div.item())

                        # 【新增】计算clip_fraction用于诊断PPO裁剪频率
                        clip_fraction = (
                            (
                                (ratio < 1.0 - TrainParams.PPO_CLIP_EPSILON)
                                | (ratio > 1.0 + TrainParams.PPO_CLIP_EPSILON)
                            )
                            .float()
                            .mean()
                        )
                        clip_fractions.append(clip_fraction.item())

                        # 【新增】收集分头熵诊断信息（从global_network.last_entropy_diagnostics）
                        if hasattr(global_network, "last_entropy_diagnostics"):
                            diag = global_network.last_entropy_diagnostics
                            entropy_type_list.append(diag["entropy_type_mean"])
                            entropy_dest_list.append(diag["entropy_dest_mean"])
                            entropy_cargo_list.append(diag["entropy_cargo_mean"])

                        # 【调试】打印value loss信息（前3步）
                        if training_step <= 3 and epoch == 0 and start == 0:
                            print(
                                f"\n[VALUE_LOSS] Mode: {'Normalized' if TrainParams.USE_VALUE_LOSS_NORMALIZATION else 'Standard'}"
                            )
                            print(
                                f"  mb_returns: mean={mb_returns.mean().item():.4f}, std={mb_returns.std().item():.4f}"
                            )
                            print(f"  raw_value_loss: {raw_value_loss.item():.6f}")
                            if TrainParams.USE_VALUE_LOSS_NORMALIZATION:
                                print(f"  running_std: {running_std.item():.4f}")
                                print(
                                    f"  normalized_value_loss: {value_loss.item():.6f}"
                                )
                                print(
                                    f"  scaling_factor (1/σ²): {(1.0 / (running_std**2 + 1e-8)).item():.6f}"
                                )
                            else:
                                print(
                                    f"  value_loss (no normalization): {value_loss.item():.6f}"
                                )
                            print(f"  explained_variance: {explained_var.item():.4f}")
                        # ====================================================================================

                        # Entropy bonus
                        entropy_beta = get_entropy_beta(
                            total_env_steps=total_env_steps, training_step=training_step
                        )
                        entropy_loss = -entropy_beta * new_entropy.mean()

                        # ==================== 均值型多目标优化：Risk Regularization ====================
                        # 理论依据：对于目标 J = E[T] + μ·E[C_rel]
                        # REINFORCE梯度：∇J = E[T·∇log π] + μ·E[C_rel·∇log π]
                        #
                        # 实现：risk_loss = μ·mean(C_rel[i]·log π[t]) for episode i's transitions
                        #
                        # 关键区别（vs 指数权重方法）：
                        # 1. 旧方法：w[i] = exp(α·C[i]) / mean(exp(α·C))  → 指数缩放，高方差
                        # 2. 新方法：w[i] = C[i]                          → 线性缩放，低方差
                        # ============================================================================
                        if TrainParams.LTL_CONSTRAINT_TYPE == "RISK_SENSITIVE_SMDP":
                            # 获取当前minibatch的C_rel系数
                            mb_C_rel = transition_C_rel[mb_indices]

                            # 只对完整episodes计算risk loss（C_rel != 0）
                            complete_mask = mb_C_rel != 0

                            if complete_mask.sum() > 0:
                                # REINFORCE梯度：μ·mean(C_rel·log π) over complete episodes
                                # 注意：C_rel已经detach（来自rollout data），所以梯度只通过log π
                                risk_loss = (
                                    TrainParams.RISK_MU
                                    * (
                                        mb_C_rel[complete_mask]
                                        * new_log_probs[complete_mask]
                                    ).mean()
                                )

                                # 【调试】打印risk_loss统计（仅前2步）
                                if training_step <= 2 and epoch == 0 and start == 0:
                                    print(f"\n[RISK_LOSS DEBUG] Mean-Based REINFORCE:")
                                    print(
                                        f"  Total transitions in minibatch: {len(mb_C_rel)}"
                                    )
                                    print(
                                        f"  Complete episodes (C_rel≠0): {complete_mask.sum().item()}"
                                    )
                                    print(
                                        f"  Truncated episodes (C_rel=0): {(~complete_mask).sum().item()}"
                                    )
                                    print(
                                        f"  mb_C_rel[complete]: min={mb_C_rel[complete_mask].min().item():.4f}, "
                                        f"max={mb_C_rel[complete_mask].max().item():.4f}, "
                                        f"mean={mb_C_rel[complete_mask].mean().item():.4f}"
                                    )
                                    print(
                                        f"  new_log_probs[complete]: min={new_log_probs[complete_mask].min().item():.4f}, "
                                        f"max={new_log_probs[complete_mask].max().item():.4f}, "
                                        f"mean={new_log_probs[complete_mask].mean().item():.4f}"
                                    )
                                    print(
                                        f"  C_rel·log π (before μ): mean={(mb_C_rel[complete_mask] * new_log_probs[complete_mask]).mean().item():.4f}"
                                    )
                                    print(
                                        f"  risk_loss (μ·mean): {risk_loss.item():.4f}"
                                    )
                                    print(f"  policy_loss: {policy_loss.item():.4f}")
                                    print(
                                        f"  Ratio (risk/policy): {(risk_loss.item() / policy_loss.item()):.2%}"
                                    )
                            else:
                                # Minibatch中没有完整episodes
                                risk_loss = 0.0
                                if training_step <= 2 and epoch == 0 and start == 0:
                                    print(
                                        f"\n[RISK_LOSS DEBUG] Warning: No complete episodes in this minibatch, risk_loss=0"
                                    )
                        else:
                            risk_loss = 0.0
                        # ============================================================================

                        # Total loss
                        # 注意：对于RISK_SENSITIVE_SMDP模式，风险正则已通过risk_loss实现
                        # Value loss权重设为0.5，配合RunningStats归一化，平衡policy和value梯度
                        total_loss = (
                            policy_loss + 0.5 * value_loss + entropy_loss + risk_loss
                        )

                        # 【诊断】打印loss组成（前2步，第一个minibatch）
                        if training_step <= 2 and epoch == 0 and start == 0:
                            print(
                                f"\n[LOSS COMPOSITION] Training step {training_step}:"
                            )
                            print(f"  policy_loss: {policy_loss.item():.6f}")
                            print(f"  value_loss: {value_loss.item():.6f}")
                            print(f"  entropy_loss: {entropy_loss.item():.6f}")
                            risk_loss_val = (
                                risk_loss.item()
                                if isinstance(risk_loss, torch.Tensor)
                                else risk_loss
                            )
                            print(f"  risk_loss: {risk_loss_val:.6f}")
                            print(f"  total_loss: {total_loss.item():.6f}")
                            print(
                                f"  risk_loss/policy_loss ratio: {(risk_loss_val / policy_loss.item()):.2%}"
                            )

                        # ==================== 【新增诊断】Loss Magnitude Imbalance ====================
                        # 每个training_step的第一个minibatch都打印，监控整个训练过程
                        if epoch == 0 and start == 0:
                            policy_val = policy_loss.item()
                            value_val = value_loss.item()
                            entropy_val = entropy_loss.item()
                            risk_loss_val = (
                                risk_loss.item()
                                if isinstance(risk_loss, torch.Tensor)
                                else risk_loss
                            )

                            # 计算magnitude ratio
                            value_policy_ratio = value_val / max(policy_val, 1e-8)
                            weighted_value = (
                                0.5 * value_val
                            )  # 考虑权重后的实际贡献（使用0.5配合归一化）
                            weighted_ratio = weighted_value / max(policy_val, 1e-8)

                            print(f"\n{'=' * 80}")
                            print(
                                f"[LOSS DIAGNOSIS] Training Step {training_step}, Epoch {epoch + 1}, First Minibatch"
                            )
                            print(f"{'=' * 80}")
                            print(f"Loss Components:")
                            print(f"  policy_loss:      {policy_val:>12.6f}")
                            print(f"  value_loss:       {value_val:>12.6f}  (raw)")
                            print(
                                f"  0.5*value_loss:   {weighted_value:>12.6f}  (weighted)"
                            )
                            print(f"  entropy_loss:     {entropy_val:>12.6f}")
                            print(f"  risk_loss:        {risk_loss_val:>12.6f}")
                            print(f"  total_loss:       {total_loss.item():>12.6f}")
                            print(f"\nMagnitude Ratios:")
                            print(
                                f"  value_loss / policy_loss:          {value_policy_ratio:>8.1f}x"
                            )
                            print(
                                f"  0.5*value_loss / policy_loss:      {weighted_ratio:>8.1f}x"
                            )
                            if weighted_ratio > 10:
                                print(
                                    f"  ⚠️  WARNING: Weighted value loss is {weighted_ratio:.1f}x larger than policy loss!"
                                )
                                print(
                                    f"      This may cause gradient imbalance in shared layers."
                                )
                            elif weighted_ratio > 5:
                                print(
                                    f"  ⚡ CAUTION: Weighted value loss is {weighted_ratio:.1f}x larger than policy loss."
                                )
                            else:
                                print(f"  ✓ Magnitude balance acceptable (ratio < 5x)")
                            print(f"{'=' * 80}\n")
                        # ============================================================================

                        # Optimize
                        global_optimizer.zero_grad()
                        total_loss.backward()

                        # ==================== 【新增诊断】Gradient Analysis ====================
                        # 每个training_step的第一个minibatch都检查共享层的梯度分布
                        if epoch == 0 and start == 0:
                            print(
                                f"[GRADIENT DIAGNOSIS] Analyzing shared layer gradients..."
                            )

                            # 收集不同类型层的梯度范数
                            encoder_grads = []
                            decoder_grads = []
                            actor_head_grads = []
                            critic_head_grads = []

                            for name, param in global_network.named_parameters():
                                if param.grad is not None:
                                    grad_norm = param.grad.norm().item()

                                    # 按层类型分类
                                    if "Encoder" in name:
                                        encoder_grads.append((name, grad_norm))
                                    elif "Decoder" in name:
                                        decoder_grads.append((name, grad_norm))
                                    elif (
                                        "action_head" in name
                                        or "cargo_head" in name
                                        or "quantity_head" in name
                                    ):
                                        actor_head_grads.append((name, grad_norm))
                                    elif (
                                        "reward_critic" in name or "cost_critic" in name
                                    ):
                                        critic_head_grads.append((name, grad_norm))

                            # 计算平均梯度范数
                            avg_encoder = sum(g for _, g in encoder_grads) / max(
                                len(encoder_grads), 1
                            )
                            avg_decoder = sum(g for _, g in decoder_grads) / max(
                                len(decoder_grads), 1
                            )
                            avg_actor = sum(g for _, g in actor_head_grads) / max(
                                len(actor_head_grads), 1
                            )
                            avg_critic = sum(g for _, g in critic_head_grads) / max(
                                len(critic_head_grads), 1
                            )

                            print(f"\nGradient Norm Statistics:")
                            print(
                                f"  Shared Encoders:  {avg_encoder:>12.6f}  ({len(encoder_grads)} params)"
                            )
                            print(
                                f"  Shared Decoders:  {avg_decoder:>12.6f}  ({len(decoder_grads)} params)"
                            )
                            print(
                                f"  Actor Heads:      {avg_actor:>12.6f}  ({len(actor_head_grads)} params)"
                            )
                            print(
                                f"  Critic Heads:     {avg_critic:>12.6f}  ({len(critic_head_grads)} params)"
                            )

                            # 计算共享层的平均梯度（Encoder+Decoder）
                            avg_shared = (
                                avg_encoder * len(encoder_grads)
                                + avg_decoder * len(decoder_grads)
                            ) / max(len(encoder_grads) + len(decoder_grads), 1)

                            print(f"\nComparison:")
                            print(f"  Avg Shared Layers: {avg_shared:>12.6f}")
                            print(f"  Avg Actor Heads:   {avg_actor:>12.6f}")

                            if avg_shared > 0 and avg_actor > 0:
                                shared_actor_ratio = avg_shared / avg_actor
                                print(
                                    f"  Ratio (Shared/Actor): {shared_actor_ratio:>8.2f}x"
                                )

                                if shared_actor_ratio > 3:
                                    print(
                                        f"  ⚠️  WARNING: Shared layers receive {shared_actor_ratio:.1f}x stronger gradients than actor heads!"
                                    )
                                    print(
                                        f"      This suggests value_loss dominates the shared feature learning."
                                    )
                                elif shared_actor_ratio > 1.5:
                                    print(
                                        f"  ⚡ CAUTION: Shared layers receive {shared_actor_ratio:.1f}x stronger gradients."
                                    )
                                else:
                                    print(f"  ✓ Gradient balance acceptable")

                            # 打印最大的几个梯度（诊断用）
                            all_grads = (
                                encoder_grads
                                + decoder_grads
                                + actor_head_grads
                                + critic_head_grads
                            )
                            all_grads.sort(key=lambda x: x[1], reverse=True)

                            print(f"\nTop 5 Largest Gradients:")
                            for name, grad_norm in all_grads[:5]:
                                layer_type = (
                                    "Shared"
                                    if ("Encoder" in name or "Decoder" in name)
                                    else "Head"
                                )
                                print(
                                    f"  [{layer_type:>6}] {name[:50]:<50} {grad_norm:>12.6f}"
                                )
                            print(f"{'=' * 80}\n")
                        # ============================================================================

                        # 【LTL优化】根据LTL状态调整梯度裁剪阈值
                        grad_clip_norm = TrainParams.GRAD_L2_CLIP
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            global_network.parameters(), max_norm=grad_clip_norm
                        )
                        global_optimizer.step()

                        p_losses.append(policy_loss.item())
                        v_losses.append(value_loss.item())
                        entropies.append(new_entropy.mean().item())
                        grads.append(grad_norm.item())

                        # 【新增】记录loss项用于诊断
                        risk_losses.append(
                            risk_loss.item()
                            if isinstance(risk_loss, torch.Tensor)
                            else risk_loss
                        )
                        entropy_losses.append(entropy_loss.item())

                # ==================== 软约束支持：更新Lagrangian乘子 ====================
                # 【修复污染问题】排除RISK_SENSITIVE_SMDP模式
                # RISK_SENSITIVE_SMDP使用episode-level tail regularizer，不需要Lagrangian
                if TrainParams.LTL_CONSTRAINT_TYPE not in [
                    "HARD",
                    "CVAR_SMDP",
                    "RISK_SENSITIVE_SMDP",
                ]:
                    # lambda的损失函数，目标是让 mean_cost 接近 COST_BUDGET
                    lambda_loss = (
                        -log_lambda
                        * (batch_mean_cost - TrainParams.COST_BUDGET).detach()
                    )

                    lambda_optimizer.zero_grad()
                    lambda_loss.backward()
                    lambda_optimizer.step()
                # =========================================================================

                training_step += 1
                lr_decay.step()

                update_time = time.time() - update_start
                training_round_time = time.time() - training_round_start

                print(f"[TRAIN] PPO更新完成！耗时: {update_time:.1f}s")
                print(
                    f"[TRAIN] Training step {training_step} 完成！总耗时: {training_round_time:.1f}s (收集: {wait_time:.1f}s, 更新: {update_time:.1f}s)"
                )
                print(
                    f"[TRAIN]   Policy Loss: {np.mean(p_losses):.4f}, Value Loss: {np.mean(v_losses):.4f}, Entropy: {np.mean(entropies):.4f}"
                )

                if TrainParams.LTL_CONSTRAINT_TYPE == "CVAR_SMDP":
                    print(f"[TRAIN]   Mean Cost (CVaR): {batch_mean_cost.item():.4f}")
                elif TrainParams.LTL_CONSTRAINT_TYPE == "RISK_SENSITIVE_SMDP":
                    # Mean-Based Multi-Objective实现：显示C_rel和makespan
                    print(
                        f"[TRAIN]   Mean C_rel (Relative Excess Hazard): {batch_mean_cost.item():.4f} ({batch_mean_cost.item() * 100:.1f}%)"
                    )
                    if "batch_mean_makespan" in locals():
                        print(
                            f"[TRAIN]   Mean T (Makespan): {batch_mean_makespan.item():.2f}"
                        )
                    print(f"[TRAIN]   μ={TrainParams.RISK_MU}")
                    print(
                        f"[TRAIN]   (Mean-based multi-objective: no exponential weights)"
                    )
                elif TrainParams.LTL_CONSTRAINT_TYPE != "HARD":
                    print(
                        f"[TRAIN]   Mean Cost: {batch_mean_cost.item():.4f}, Lambda: {torch.exp(log_lambda).item():.4f}"
                    )

                # === Stage 7: Logging ===
                value_target_mean = returns.mean().item()

                # 【新增】聚合PPO诊断指标
                explained_variance_mean = (
                    np.mean(explained_variances)
                    if len(explained_variances) > 0
                    else 0.0
                )
                advantages_mean_val = (
                    np.mean(advantages_means) if len(advantages_means) > 0 else 0.0
                )
                advantages_std_val = (
                    np.mean(advantages_stds) if len(advantages_stds) > 0 else 0.0
                )
                kl_divergence_mean = (
                    np.mean(kl_divergences) if len(kl_divergences) > 0 else 0.0
                )
                clip_fraction_mean = (
                    np.mean(clip_fractions) if len(clip_fractions) > 0 else 0.0
                )

                # 【新增】聚合分头熵监控指标
                entropy_type_mean = (
                    np.mean(entropy_type_list) if len(entropy_type_list) > 0 else 0.0
                )
                entropy_dest_mean = (
                    np.mean(entropy_dest_list) if len(entropy_dest_list) > 0 else 0.0
                )
                entropy_cargo_mean = (
                    np.mean(entropy_cargo_list) if len(entropy_cargo_list) > 0 else 0.0
                )
                entropy_quantity_mean = (
                    0.0  # quantity head removed - auto-load max capacity
                )

                tm = [
                    value_target_mean,
                    np.mean(p_losses),
                    np.mean(entropies),
                    np.mean(grads),
                ]

                # ==================== 计算comprehensive metrics用于WandB ====================
                # 【修改】NEU和RISK模式均需要计算这些指标用于对比实验
                if EnvParams.VEHICLE_FAILURE_ENABLED:
                    # 筛选完整episodes
                    complete_episode_ids_for_stats = [
                        ep_id
                        for ep_id, is_done in global_episode_done_map.items()
                        if is_done
                    ]

                    # 初始化所有comprehensive metrics为0（避免KeyError）
                    batch_W_mean = 0.0
                    batch_H_actual_mean = 0.0
                    batch_H_optimal_mean = 0.0
                    batch_C_abs_mean = 0.0
                    batch_work_std_mean = 0.0
                    batch_work_cv_mean = 0.0
                    batch_max_mean_ratio_mean = 0.0
                    batch_gini_coeff_mean = 0.0

                    if len(complete_episode_ids_for_stats) > 0:
                        # 提取完整episodes的所有metrics
                        W_values = [
                            global_episode_W_map.get(ep_id, 0.0)
                            for ep_id in complete_episode_ids_for_stats
                        ]
                        H_actual_values = [
                            global_episode_H_actual_map.get(ep_id, 0.0)
                            for ep_id in complete_episode_ids_for_stats
                        ]
                        H_optimal_values = [
                            global_episode_H_optimal_map.get(ep_id, 0.0)
                            for ep_id in complete_episode_ids_for_stats
                        ]
                        C_abs_values = [
                            global_episode_C_abs_map.get(ep_id, 0.0)
                            for ep_id in complete_episode_ids_for_stats
                        ]
                        work_std_values = [
                            global_episode_work_std_map.get(ep_id, 0.0)
                            for ep_id in complete_episode_ids_for_stats
                        ]
                        work_cv_values = [
                            global_episode_work_cv_map.get(ep_id, 0.0)
                            for ep_id in complete_episode_ids_for_stats
                        ]
                        max_mean_ratio_values = [
                            global_episode_max_mean_ratio_map.get(ep_id, 1.0)
                            for ep_id in complete_episode_ids_for_stats
                        ]
                        gini_coeff_values = [
                            global_episode_gini_coeff_map.get(ep_id, 0.0)
                            for ep_id in complete_episode_ids_for_stats
                        ]

                        # 计算batch-level均值
                        batch_W_mean = np.mean(W_values) if W_values else 0.0
                        batch_H_actual_mean = (
                            np.mean(H_actual_values) if H_actual_values else 0.0
                        )
                        batch_H_optimal_mean = (
                            np.mean(H_optimal_values) if H_optimal_values else 0.0
                        )
                        batch_C_abs_mean = (
                            np.mean(C_abs_values) if C_abs_values else 0.0
                        )
                        batch_work_std_mean = (
                            np.mean(work_std_values) if work_std_values else 0.0
                        )
                        batch_work_cv_mean = (
                            np.mean(work_cv_values) if work_cv_values else 0.0
                        )
                        batch_max_mean_ratio_mean = (
                            np.mean(max_mean_ratio_values)
                            if max_mean_ratio_values
                            else 1.0
                        )
                        batch_gini_coeff_mean = (
                            np.mean(gini_coeff_values) if gini_coeff_values else 0.0
                        )

                        # 【调试】打印comprehensive metrics（仅前3步）
                        if training_step <= 3:
                            print(f"\n{'=' * 80}")
                            print(
                                f"[Comprehensive Metrics] Batch Statistics from {len(complete_episode_ids_for_stats)} Complete Episodes"
                            )
                            print(f"{'=' * 80}")
                            print(f"  【理论验证指标】")
                            print(f"    总工作量 W (mean):        {batch_W_mean:.2f}")
                            print(
                                f"    实际总Hazard H_actual:    {batch_H_actual_mean:.4f}"
                            )
                            print(
                                f"    理论最优Hazard H_optimal: {batch_H_optimal_mean:.4f}"
                            )
                            print(
                                f"    绝对超额Hazard C_abs:     {batch_C_abs_mean:.4f}"
                            )
                            print(f"  【负载均衡指标】")
                            print(
                                f"    工作时间 std:             {batch_work_std_mean:.2f}"
                            )
                            print(
                                f"    变异系数 CV:              {batch_work_cv_mean:.4f}"
                            )
                            print(
                                f"    最大/平均比率:            {batch_max_mean_ratio_mean:.3f}x"
                            )
                            print(
                                f"    Gini系数:                {batch_gini_coeff_mean:.4f}"
                            )
                            print(f"{'=' * 80}\n")
                # ============================================================================

                # LTL约束相关指标（必须添加，即使是HARD模式，以保持维度一致）
                if TrainParams.LTL_CONSTRAINT_TYPE == "RISK_SENSITIVE_SMDP":
                    # Mean-Based Multi-Objective实现：记录C_rel（相对负载不均衡度）
                    # 注意：这里记录的是监控指标
                    #   - mean_cost字段存储C_rel值（Relative Excess Hazard，百分比形式）
                    #   - lambda_val字段设为0（不再使用tail regularization）
                    makespan_val = (
                        batch_mean_makespan.item()
                        if "batch_mean_makespan" in locals()
                        else 0.0
                    )

                    # 【新增】Loss项诊断指标
                    risk_loss_mean = (
                        np.mean(risk_losses) if len(risk_losses) > 0 else 0.0
                    )
                    value_loss_mean = np.mean(v_losses) if len(v_losses) > 0 else 0.0
                    entropy_loss_mean = (
                        np.mean(entropy_losses) if len(entropy_losses) > 0 else 0.0
                    )

                    # C_rel统计指标
                    c_std_val = C_bar_std.item() if "C_bar_std" in locals() else 0.0
                    c_min_val = C_bar_min.item() if "C_bar_min" in locals() else 0.0
                    c_max_val = C_bar_max.item() if "C_bar_max" in locals() else 0.0

                    tm.extend(
                        [
                            # ==================== 原有指标 ====================
                            batch_mean_cost.item(),  # C_rel (Relative Excess Hazard) mean
                            0.0,  # lambda_val (不再使用，保留兼容性)
                            makespan_val,  # makespan mean
                            risk_loss_mean,  # risk loss
                            value_loss_mean,  # value loss
                            entropy_loss_mean,  # entropy loss
                            c_std_val,  # C_rel std
                            c_min_val,  # C_rel min
                            c_max_val,  # C_rel max
                            0.0,  # weight_ratio (不再使用，保留兼容性)
                            0.0,  # ess_ratio (不再使用，保留兼容性)
                            # ==================== Comprehensive Metrics（新增）====================
                            # 【理论验证指标】用于验证H_actual与T的耦合关系
                            batch_W_mean,  # 总工作量 W (mean)
                            batch_H_actual_mean,  # 实际总Hazard H_actual (mean)
                            batch_H_optimal_mean,  # 理论最优Hazard H_optimal (mean)
                            batch_C_abs_mean,  # 绝对超额Hazard C_abs (mean)
                            # 【负载均衡指标】多种度量方式的综合对比
                            batch_work_std_mean,  # 工作时间标准差 std(t_i)
                            batch_work_cv_mean,  # 变异系数 CV
                            batch_max_mean_ratio_mean,  # 最大/平均比率
                            batch_gini_coeff_mean,  # Gini系数
                            # ==================== PPO诊断指标（新增）====================
                            explained_variance_mean,  # explained variance
                            advantages_mean_val,  # advantages mean
                            advantages_std_val,  # advantages std
                            kl_divergence_mean,  # KL divergence
                            clip_fraction_mean,  # clip fraction
                            # ==================== 分头熵监控（新增）====================
                            entropy_type_mean,  # entropy of action_type head
                            entropy_dest_mean,  # entropy of destination head
                            entropy_cargo_mean,  # entropy of cargo head
                            entropy_quantity_mean,  # entropy of quantity head
                        ]
                    )
                elif TrainParams.LTL_CONSTRAINT_TYPE not in ["HARD", "CVAR_SMDP"]:
                    # 其他软约束模式（LTL_POTENTIAL等）：记录mean_cost和lambda + loss指标 + PPO诊断指标
                    # 【修复】计算loss指标
                    risk_loss_mean = (
                        np.mean(risk_losses) if len(risk_losses) > 0 else 0.0
                    )
                    value_loss_mean = np.mean(v_losses) if len(v_losses) > 0 else 0.0
                    entropy_loss_mean = (
                        np.mean(entropy_losses) if len(entropy_losses) > 0 else 0.0
                    )

                    tm.extend(
                        [
                            batch_mean_cost.item(),  # mean_cost
                            torch.exp(log_lambda).item(),  # lambda_val
                            0.0,  # makespan (软约束模式不记录)
                            risk_loss_mean,  # risk loss
                            value_loss_mean,  # value loss
                            entropy_loss_mean,  # entropy loss
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,  # c_std, c_min, c_max, weight_ratio, ess_ratio (不适用)
                            # Comprehensive metrics (软约束模式不使用)
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            # 【新增】PPO诊断指标
                            explained_variance_mean,
                            advantages_mean_val,
                            advantages_std_val,
                            kl_divergence_mean,
                            clip_fraction_mean,
                            # 【新增】分头熵监控
                            entropy_type_mean,
                            entropy_dest_mean,
                            entropy_cargo_mean,
                            entropy_quantity_mean,
                        ]
                    )
                elif TrainParams.LTL_CONSTRAINT_TYPE == "CVAR_SMDP":
                    # CVaR模式：记录mean cost，lambda设为0 + loss指标 + PPO诊断指标
                    # 【修复】计算loss指标
                    risk_loss_mean = (
                        np.mean(risk_losses) if len(risk_losses) > 0 else 0.0
                    )
                    value_loss_mean = np.mean(v_losses) if len(v_losses) > 0 else 0.0
                    entropy_loss_mean = (
                        np.mean(entropy_losses) if len(entropy_losses) > 0 else 0.0
                    )

                    tm.extend(
                        [
                            batch_mean_cost.item(),  # mean_cost
                            0.0,  # lambda_val (CVaR不使用)
                            0.0,  # makespan (CVaR模式不记录)
                            risk_loss_mean,  # risk loss
                            value_loss_mean,  # value loss
                            entropy_loss_mean,  # entropy loss
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,  # c_std, c_min, c_max, weight_ratio, ess_ratio (不适用)
                            # Comprehensive metrics (CVaR模式不使用)
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            # 【新增】PPO诊断指标
                            explained_variance_mean,
                            advantages_mean_val,
                            advantages_std_val,
                            kl_divergence_mean,
                            clip_fraction_mean,
                            # 【新增】分头熵监控
                            entropy_type_mean,
                            entropy_dest_mean,
                            entropy_cargo_mean,
                            entropy_quantity_mean,
                        ]
                    )
                else:
                    # HARD模式（NEU模式）
                    # 【修复】计算loss指标（即使是HARD模式，这些loss也在计算）
                    risk_loss_mean = (
                        np.mean(risk_losses) if len(risk_losses) > 0 else 0.0
                    )
                    value_loss_mean = np.mean(v_losses) if len(v_losses) > 0 else 0.0
                    entropy_loss_mean = (
                        np.mean(entropy_losses) if len(entropy_losses) > 0 else 0.0
                    )
                    makespan_val = 0.0  # HARD模式没有batch_mean_makespan

                    if EnvParams.VEHICLE_FAILURE_ENABLED:
                        # NEU模式也需要记录comprehensive metrics用于对比实验
                        # 为保持维度一致，extend相同的19个值

                        tm.extend(
                            [
                                # ==================== 原有指标 ====================
                                0.0,  # mean_cost (C_rel) - NEU不优化这个
                                0.0,  # lambda_val
                                makespan_val,
                                risk_loss_mean,  # 【修复】使用实际计算的risk_loss
                                value_loss_mean,  # 【修复】使用实际计算的value_loss
                                entropy_loss_mean,  # 【修复】使用实际计算的entropy_loss
                                0.0,  # c_std
                                0.0,  # c_min
                                0.0,  # c_max
                                0.0,  # weight_ratio
                                0.0,  # ess_ratio
                                # ==================== Comprehensive Metrics ====================
                                # 【NEU和RISK模式共享】用于对比负载均衡性能
                                batch_W_mean if "batch_W_mean" in locals() else 0.0,
                                batch_H_actual_mean
                                if "batch_H_actual_mean" in locals()
                                else 0.0,
                                batch_H_optimal_mean
                                if "batch_H_optimal_mean" in locals()
                                else 0.0,
                                batch_C_abs_mean
                                if "batch_C_abs_mean" in locals()
                                else 0.0,
                                batch_work_std_mean
                                if "batch_work_std_mean" in locals()
                                else 0.0,
                                batch_work_cv_mean
                                if "batch_work_cv_mean" in locals()
                                else 0.0,
                                batch_max_mean_ratio_mean
                                if "batch_max_mean_ratio_mean" in locals()
                                else 0.0,
                                batch_gini_coeff_mean
                                if "batch_gini_coeff_mean" in locals()
                                else 0.0,
                                # ==================== PPO诊断指标（新增）====================
                                explained_variance_mean,  # explained variance
                                advantages_mean_val,  # advantages mean
                                advantages_std_val,  # advantages std
                                kl_divergence_mean,  # KL divergence
                                clip_fraction_mean,  # clip fraction
                                # ==================== 分头熵监控（新增）====================
                                entropy_type_mean,  # entropy of action_type head
                                entropy_dest_mean,  # entropy of destination head
                                entropy_cargo_mean,  # entropy of cargo head
                                entropy_quantity_mean,  # entropy of quantity head
                            ]
                        )
                    else:
                        # HARD模式且未启用故障：extend 24个值以保持维度一致（19个原有+5个PPO诊断）
                        tm.extend(
                            [
                                0.0,  # mean_cost
                                0.0,  # lambda_val
                                makespan_val,  # makespan
                                risk_loss_mean,  # 【修复】使用实际计算的risk_loss
                                value_loss_mean,  # 【修复】使用实际计算的value_loss
                                entropy_loss_mean,  # 【修复】使用实际计算的entropy_loss
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,  # c_std, c_min, c_max, weight_ratio, ess_ratio
                                # Comprehensive metrics (Pure LTL版本不启用故障，这些指标为0)
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                # ==================== PPO诊断指标（新增）====================
                                explained_variance_mean,  # explained variance
                                advantages_mean_val,  # advantages mean
                                advantages_std_val,  # advantages std
                                kl_divergence_mean,  # KL divergence
                                clip_fraction_mean,  # clip fraction
                                # ==================== 分头熵监控（新增）====================
                                entropy_type_mean,  # entropy of action_type head
                                entropy_dest_mean,  # entropy of destination head
                                entropy_cargo_mean,  # entropy of cargo head
                                entropy_quantity_mean,  # entropy of quantity head
                            ]
                        )

                training_metrics_aggregator.append(tm)

                # 定期评估和保存
                if training_step > 0 and training_step % TrainParams.EVALUATE_GAP == 0:
                    if TrainParams.EVALUATE:
                        current_weights = global_network.state_dict()
                        logger.run_evaluation(
                            current_weights=current_weights,
                            benchmarks=evaluation_benchmarks,
                            meta_agents=meta_agents,
                            training_step=training_step,
                            total_env_steps=total_env_steps,
                        )

                        # 【模型更新】评估完成后，将当前模型权重更新到baseline_network
                        print(f"[PPO] 🔄 更新baseline模型权重...")
                        baseline_network.load_state_dict(global_network.state_dict())
                        print(f"[PPO] ✓ Baseline模型已更新")

                # 【PPO模型保存】每PPO_SAVE_INTERVAL步保存最新模型权重（不比较baseline）
                if (
                    training_step > 0
                    and training_step % SaverParams.PPO_SAVE_INTERVAL == 0
                ):
                    if SaverParams.SAVE:
                        print(
                            f"[PPO] 💾 保存最新模型权重 (training_step={training_step})..."
                        )
                        logger.save_model(curr_episode, curr_level, best_perf)
                        print(f"[PPO] ✓ 模型已保存")

                # 日志记录
                if len(training_metrics_aggregator) >= TrainParams.SUMMARY_WINDOW:
                    print(
                        f"[PPO] 记录指标到WandB (已累积 {len(training_metrics_aggregator)} 次更新)..."
                    )
                    training_means = np.nanmean(
                        np.array(training_metrics_aggregator), axis=0
                    ).tolist()

                    perf_data_list = []
                    for k in perf_keys:
                        if perf_metrics_aggregator[k]:
                            perf_data_list.append(
                                np.nanmean(perf_metrics_aggregator[k])
                            )
                        else:
                            perf_data_list.append(0)

                    log_data = training_means + perf_data_list
                    mean_raw_reward = (
                        np.mean(raw_rewards_since_last_log)
                        if raw_rewards_since_last_log
                        else 0.0
                    )

                    logger.write_to_board(
                        [log_data], training_step, total_env_steps, mean_raw_reward
                    )
                    print(
                        f"[PPO] ✓ 指标已上传到WandB (x轴: {total_env_steps} 环境步数)"
                    )

                    # 清空累加器
                    training_metrics_aggregator = []
                    raw_rewards_since_last_log = []
                    for k in perf_metrics_aggregator:
                        perf_metrics_aggregator[k] = []

                # 【训练停止检查】检查是否达到最大训练步数
                if (
                    TrainParams.MAX_TRAINING_STEPS is not None
                    and total_env_steps >= TrainParams.MAX_TRAINING_STEPS
                ):
                    print(f"\n{'=' * 80}")
                    print(f"[训练完成] 达到最大训练步数！")
                    print(f"{'=' * 80}")
                    print(
                        f"  - 总环境步数: {total_env_steps} / {TrainParams.MAX_TRAINING_STEPS}"
                    )
                    print(f"  - 训练轮次: {training_step}")
                    print(f"  - 当前Episode: {curr_episode}")
                    print(f"\n正在保存最终模型...")

                    # 保存最终模型
                    if SaverParams.SAVE:
                        logger.save_model(curr_episode, curr_level, best_perf)
                        print(f"✓ 最终模型已保存")

                    print(f"\n训练结束，准备退出...")
                    break  # 退出训练循环

        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("检测到 Ctrl+C，正在优雅退出...")
            print("=" * 60)
            for a in meta_agents:
                try:
                    ray.kill(a)
                except:
                    pass
            print("所有workers已终止")

        except Exception as e:
            print("\n" + "=" * 60)
            print(f"[CRITICAL ERROR] PPO训练过程中发生未预期的错误！")
            print("=" * 60)
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print("\n完整堆栈跟踪：")
            import traceback

            traceback.print_exc()
            print("\n" + "=" * 60)
            print("正在清理资源...")
            for a in meta_agents:
                try:
                    ray.kill(a)
                except:
                    pass
            print("=" * 60 + "\n")
            raise

        finally:
            print("\n[INFO] PPO训练循环已退出")

            # 等待异步评估完成
            if async_evaluator is not None:
                print("\n[ASYNC EVAL] 检查异步评估状态...")
                async_evaluator.print_statistics()

                if async_evaluator.is_running():
                    print("[ASYNC EVAL] 等待最后一次评估完成...")
                    async_evaluator.wait_completion(timeout=1800)  # 最多等待30分钟

                print("\n[ASYNC EVAL] 最终统计:")
                async_evaluator.print_statistics()

            if SaverParams.SAVE:
                try:
                    # 【模型更新】训练结束前，更新baseline模型为最新权重
                    print("[INFO] 🔄 更新baseline模型为最新权重...")
                    baseline_network.load_state_dict(global_network.state_dict())
                    print("[INFO] ✓ Baseline模型已更新")

                    print("[INFO] 保存最终模型...")
                    logger.save_model(curr_episode, curr_level, best_perf)
                    print("[INFO] ✓ 模型已保存")
                except Exception as e:
                    print(f"[WARNING] 保存模型失败: {e}")


if __name__ == "__main__":
    main()
