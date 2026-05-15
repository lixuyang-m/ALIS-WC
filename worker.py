# This file is derived from the HeteroMRTA codebase by Dai et al.
# (IEEE RA-L 2025), originally released under the Apache-2.0 License.
# Original source: https://github.com/marmotlab/HeteroMRTA
# Modifications: event-driven sleep-wake scheduling, three-mask pipeline
#   (AllocationFilter / FeasibilityMask / SequentialShield), two PBRS
#   variants (elapsed-time / task-completion), LTL monitor integration.

import time
import torch
import numpy as np
import random
from env.task_env import TaskEnv
from attention import AttentionNet
import scipy.signal as signal
from parameters import *
import copy
from torch.nn import functional as F
from torch.distributions import Categorical

from env.ltl_utils import (
    LTLMonitor,
    LTL_SAFETY,
    LTL_SEQUENTIAL,
    FSA_SAFETY_SAFE,
    FSA_SAFETY_VIOLATED,
    FSA_SEQ_INITIAL,
    FSA_SEQ_PREDECESSOR_DONE,
    FSA_SEQ_SATISFIED,
    FSA_SEQ_VIOLATED,
    LTLClause,
)


def discount(x, gamma):  # 定义折扣因子gamma，累加折扣奖励
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def zero_padding(x, padding_size, length):  # 张量0填充
    pad = torch.nn.ZeroPad2d((0, 0, 0, padding_size - length))
    x = pad(x)
    return x


DEBUG = False
print_init_state = False
print_task_detail = False
print_episode_result = False
process_debug = False
Event_Calculation = False


def pre_flight_check(env, agent_id, action_dict):
    """
    在调用 agent_step 之前，模拟其内部的检查逻辑，以进行调试。
    如果动作在预检中就发现问题，则返回False和失败原因。
    """
    agent = env.agent_dic[agent_id]
    action_type = action_dict.get("type")

    if action_type == env.ACTION_MOVE:
        dest_id = action_dict.get("destination")
        if dest_id is None:
            return False, "MOVE action is missing 'destination' key."
        if dest_id >= env.depots_num:
            task_id = dest_id - env.depots_num
            if task_id >= env.tasks_num:
                return (
                    False,
                    f"MOVE to an out-of-bounds task index. Destination ID: {dest_id}, Max Task Num: {env.tasks_num}",
                )
            if env.task_dic.get(task_id, {}).get("finished", True):
                return False, f"MOVE to a finished or invalid task. Task ID: {task_id}"

    elif action_type == env.ACTION_LOAD:
        if not (agent["current_task"] < 0):
            return False, "LOAD rejected: Agent is not at a depot."
        if not (agent["inventory"]["quantity"] == 0):
            return False, "LOAD rejected: Agent is not empty."

    elif action_type == env.ACTION_UNLOAD:
        if agent["inventory"]["quantity"] <= 0:
            return False, "UNLOAD rejected: Agent is empty."

    return True, "OK"


class Worker:
    def __init__(
        self,
        mete_agent_id,
        local_network,
        local_baseline,
        global_step,
        device="cuda",
        save_image=False,
        seed=None,
        env_params=None,
    ):

        self.device = device
        self.metaAgentID = mete_agent_id
        self.global_step = global_step
        self.save_image = save_image

        # 设置Worker实例的随机种子（用于确保random.shuffle等操作可复现）
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        if env_params is None:
            env_params = [
                EnvParams.SPECIES_AGENTS_RANGE,
                EnvParams.SPECIES_RANGE,
                EnvParams.TASKS_RANGE,
                EnvParams.DEPOT_NUM_RANGE,
            ]

        # 环境初始化（与您的代码保持一致）
        self.env = TaskEnv(
            per_species_range=env_params[0],
            species_range=env_params[1],
            tasks_range=env_params[2],
            depot_num_range=env_params[3],
            traits_dim=EnvParams.TRAIT_DIM,
            decision_dim=EnvParams.DECISION_DIM,
            max_task_size=EnvParams.MAX_TASK_SIZE,
            duration_scale=EnvParams.DURATION_SCALE,
            seed=seed,
            plot_figure=save_image,
        )
        self.baseline_env = copy.deepcopy(self.env)

        # 本地策略网络和基线网络
        self.local_baseline = local_baseline
        self.local_net = local_network

        # 经验回放池
        self.experience = {
            "task_info": [],
            "agents_info": [],
            "mask": [],
            "index": [],
            "old_log_prob": [],
            "reward": [],
            "advantage": [],
            "entropy": [],
            "actions": [],
            "risk_z": [],
        }
        self.episode_number = None

        # 存储性能指标
        self.perf_metrics = {}
        self.p_rnn_state = {}
        self.max_time = EnvParams.MAX_TIME

        # === 决策顺序的随机数生成器（用于多智能体同时决策时的顺序打乱）===
        # 使用独立的RandomState以确保可复现性，同时不影响其他随机操作
        if TrainParams.USE_MANUAL_SEED and TrainParams.MANUAL_SEED is not None:
            # 使用 metaAgentID 作为额外的扰动，确保不同Worker有不同的随机序列
            self.decision_order_rng = np.random.RandomState(
                seed=TrainParams.MANUAL_SEED + self.metaAgentID * 10000
            )
        else:
            self.decision_order_rng = np.random.RandomState()

        # worker.py -> class Worker:

    def compute_gae(
        self,
        rewards,
        values,
        dones,
        last_value,
        gamma=None,
        lambda_=None,
        taus=None,
        execution_mode=None,
    ):
        """
        计算Generalized Advantage Estimation (GAE)
        支持MDP和SMDP两种模式

        Args:
            rewards: List or Tensor of rewards [T]
            values: List or Tensor of value estimates [T]
            dones: List or Tensor of done flags [T]
            last_value: Bootstrap value for the last state (0 if done, V(s_T) if truncated)
            gamma: Discount factor for MDP mode (default: TrainParams.GAMMA)
            lambda_: GAE lambda parameter (default: TrainParams.GAE_LAMBDA)
            taus: List or Tensor of time intervals [T] (only for SMDP mode)
            execution_mode: 'mdp' or 'smdp' (default: TrainParams.EXECUTION_MODE)

        Returns:
            returns: Tensor of returns (targets for value function) [T]
            advantages: Tensor of advantages [T]
        """
        if gamma is None:
            gamma = TrainParams.GAMMA
        if lambda_ is None:
            lambda_ = getattr(TrainParams, "GAE_LAMBDA", 0.95)  # 默认0.95
        if execution_mode is None:
            execution_mode = TrainParams.EXECUTION_MODE

        # 转换为tensor
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float, device=self.device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones, dtype=torch.float, device=self.device)

        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        last_gae_lambda = 0

        # ==================== SMDP模式：使用时间依赖折扣 ====================
        if execution_mode == "smdp":
            if taus is None:
                raise ValueError(
                    "SMDP mode requires taus (time intervals) for GAE computation"
                )

            # 转换taus为tensor
            if not isinstance(taus, torch.Tensor):
                taus = torch.tensor(taus, dtype=torch.float, device=self.device)

            beta = TrainParams.BETA

            # 反向计算SMDP-GAE
            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = last_value
                    # 最后一步的tau用于bootstrap（如果被截断）
                    tau_t = taus[t] if t < len(taus) else 0.0
                else:
                    next_value = values[t + 1]
                    tau_t = taus[t]

                # SMDP时间依赖折扣因子：γ(τ) = exp(-β·τ)
                gamma_tau = torch.exp(-beta * tau_t)

                # SMDP-TD error: δ_t = r_t + exp(-β·τ_t) * V(s_{t+1}) * (1 - done_t) - V(s_t)
                delta = rewards[t] + gamma_tau * next_value * (1 - dones[t]) - values[t]

                # SMDP-GAE: A_t = δ_t + exp(-β·τ_t)·λ * (1 - done_t) * A_{t+1}
                # 注意：这里的λ仍然使用固定值，只有γ是时间依赖的
                advantages[t] = (
                    delta + gamma_tau * lambda_ * (1 - dones[t]) * last_gae_lambda
                )
                last_gae_lambda = advantages[t]

        # ==================== MDP模式：使用固定折扣 ====================
        else:  # execution_mode == 'mdp'
            # 反向计算标准MDP-GAE
            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = last_value
                else:
                    next_value = values[t + 1]

                # MDP-TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
                delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

                # MDP-GAE: A_t = δ_t + γλ * (1 - done_t) * A_{t+1}
                advantages[t] = (
                    delta + gamma * lambda_ * (1 - dones[t]) * last_gae_lambda
                )
                last_gae_lambda = advantages[t]

        # Returns = Advantages + Values (用于训练value function)
        returns = advantages + values

        return returns, advantages

    def check_and_wakeup_agents(self, ltl_monitor, constraint_mode=None):
        """
        检查并唤醒临时休眠的agents。
        适用于两种情况：
        1. 因LTL约束而休眠的agents（例如顺序约束）
        2. 因环境限制而休眠的agents（例如depot暂时没有需要的cargo）
        """
        # 遍历所有智能体，筛选出那些正处于"暂时休眠"状态的
        for agent_id, agent in self.env.agent_dic.items():
            if agent.get("is_temp_sleeping"):
                # 使用 agent_observe 作为唯一权威来判断是否应该唤醒。
                # 这避免了在唤醒函数中重复实现复杂的决策逻辑，保证了行为的一致性。
                # 我们只关心它返回的"行动状态原因"。
                # agent_observe返回8个值！不要遗漏cost_info
                # 重要：必须传递constraint_mode，确保与正常决策逻辑一致！
                _, _, _, _, _, _, inaction_reason, _ = self.env.agent_observe(
                    agent_id,
                    ltl_monitor,
                    max_waiting=False,
                    constraint_mode=constraint_mode,
                    ignore_sleeping=True,
                )

                # 唤醒条件：在当前新的世界状态下，智能体有可用的动作
                # 可能的原因：
                # 1. LTL约束解除（前置任务完成）
                # 2. Depot补货了（有可装载的cargo）
                # 3. 其他环境变化使得agent可以行动
                if inaction_reason == "ACTIONS_AVAILABLE":
                    # 执行唤醒操作
                    blocking_info = agent.get("blocking_clauses", [])
                    wakeup_reason = (
                        "LTL constraint lifted"
                        if blocking_info
                        else "Environment changed"
                    )
                    print(
                        f"    (WAKEUP) [T={self.env.current_time:.2f}s] Agent {agent_id} woken up! Reason: {wakeup_reason}"
                    )
                    agent["is_temp_sleeping"] = False
                    agent["next_decision"] = (
                        self.env.current_time
                    )  # 准备在当前时间点立即决策
                    agent["blocking_clauses"] = []  # 清空阻塞记录
                elif DEBUG and agent.get("is_temp_sleeping"):
                    # 如果还是休眠，打印原因（用于调试）
                    print(
                        f"    (STILL_SLEEPING) [T={self.env.current_time:.2f}s] Agent {agent_id} still blocked. Reason: {inaction_reason}"
                    )

    def run_episode(
        self,
        training=True,
        sample=False,
        max_waiting=False,
        ltl_clauses=None,
        constraint_mode_override=None,
    ):
        import time as time_module

        episode_real_start_time = time_module.time()

        self.env.init_state()
        total_initial_requirements = sum(
            np.sum(t["requirements"]) for t in self.env.task_dic.values()
        )

        # 【诊断开关】仅在评估时开启
        ENABLE_EVAL_DIAGNOSTICS = (
            False  # not training  # 评估时自动启用详细诊断（已禁用）
        )
        last_diag_print_time = episode_real_start_time
        DIAG_PRINT_INTERVAL = 30  # 每30秒打印一次诊断信息（更频繁的更新）

        # 【新增】动作执行统计
        action_counters = {"MOVE": 0, "LOAD": 0, "UNLOAD": 0, "REJECTED": 0}
        tasks_completed_count = 0

        # 【诊断】动作类型选择统计（诊断为什么某些动作不被选择）
        action_type_selection_stats = {
            "MOVE": 0,
            "LOAD": 0,
            "UNLOAD": 0,
        }  # 模型选择的次数
        action_type_available_stats = {"MOVE": 0, "LOAD": 0, "UNLOAD": 0}  # 可用的次数

        if ENABLE_EVAL_DIAGNOSTICS:
            print(f"[EPISODE DIAG] 开始运行 (training={training}, sample={sample})")
            print(
                f"[EPISODE DIAG] 环境规模: {len(self.env.agent_dic)}个agent, {len(self.env.task_dic)}个task"
            )
            initial_total_reqs = sum(
                np.sum(t["requirements"]) for t in self.env.task_dic.values()
            )
            print(f"[EPISODE DIAG] 初始总需求: {initial_total_reqs}")

        if print_init_state:
            print("\n" + "=" * 80)
            print(f"--- EPISODE {self.env.current_time} START ---")
            print("=" * 80)

            print("\n[INITIAL STATE] Task Requirements:")
            for tid, task in self.env.task_dic.items():
                if np.sum(task["requirements"]) > 0:
                    print(f"  - Task {tid}: {task['requirements']}")

            print("\n[INITIAL STATE] Depot Stocks:")
            for did, depot in self.env.depot_dic.items():
                stock_str = ", ".join(
                    [f"Type{k}:{v}" for k, v in depot["stock"].items() if v > 0]
                )
                print(f"  - Depot {did}: {stock_str}")

            print("\n[INITIAL STATE] Species Capabilities:")
            for species_id, capabilities in enumerate(
                self.env.species_dict["capacities"]
            ):
                # 为了简洁，只打印该物种能携带的货物类型（能力 > 0）
                cap_str = ", ".join(
                    [f"Type{k}:{int(v)}" for k, v in enumerate(capabilities) if v > 0]
                )
                print(f"  - Species {species_id}: Capacities({cap_str})")

            print("\n[INITIAL STATE] Agent Locations:")
            for aid, agent in self.env.agent_dic.items():
                print(
                    f"  - Agent {aid} (Species {agent['species']}): starts at Node {agent['current_task']}"
                )

            print("\n" + "-" * 80)
            print("--- DECISION LOG ---")
            print("-" * 80 + "\n")

        effective_constraint_mode = (
            constraint_mode_override
            if constraint_mode_override is not None
            else TrainParams.LTL_CONSTRAINT_TYPE
        )

        # 【方法E】评估时强制硬约束（验证学习效果）
        if (
            not training
            and effective_constraint_mode == "LTL_POTENTIAL"
            and TrainParams.LTL_ENFORCE_IN_EVAL
        ):
            effective_constraint_mode = "HARD"
            if print_init_state or decision_step == 0:
                print("[方法E] 评估模式：强制硬约束（验证LTL满足能力）")

        episode_metrics = {
            # === 4个PolicyHead的entropy和prob_max ===
            "entropy/action_type": [],
            "prob_max/action_type": [],
            "entropy/destination": [],
            "prob_max/destination": [],
            "entropy/cargo": [],
            "prob_max/cargo": [],
            "entropy/quantity": [],
            "prob_max/quantity": [],
            # === 动作类型概率分布（用于诊断）===
            "prob_move": [],
            "prob_load": [],
            "prob_unload": [],
        }

        buffer_dict = {
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
            "dones": [],
            "cargo_mask": [],
            "action_type_mask": [],
            "costs": [],
            "taus": [],
            "ltl_info": [],
            "next_ltl_info": [],
            "dependency_graph": [],  # 【方案一】存储任务依赖图
            # 【方法1：分头独立PPO】存储各头的old_log_prob
            "old_log_prob_type": [],
            "old_log_prob_dest": [],
            "old_log_prob_cargo": [],
        }

        # 【Episode-level CVAR】统一风险指数累加器（无折扣）
        episode_unified_risk_index = 0.0  # C_episode = Σ(β + h_k)·τ_k

        # 【CVAR_SMDP】变量初始化（保留用于调试）
        episode_accumulated_hazard = 0.0
        last_time_step = 0.0  # 记录上一步的时间，用于计算delta_t

        perf_metrics = {}
        decision_step = 0

        # 【新增】动作类型选择统计（总是收集，无论是否诊断模式）
        action_type_chosen_counts = {"MOVE": 0, "LOAD": 0, "UNLOAD": 0}

        episode_raw_return = 0.0

        # === 分项记录各个奖励成分（用于诊断和分析）===
        episode_shaping_reward_sum = 0.0  # 进展奖励总和（包含所有势能塑形）
        episode_time_penalty_sum = 0.0  # 时间惩罚总和
        episode_terminal_penalty_sum = 0.0  # 终止惩罚总和

        # === 新增：分项记录各个势能成分（用于详细诊断）===
        episode_progress_potential_sum = 0.0  # 已完成需求势能
        episode_time_potential_sum = 0.0  # 时间势能
        episode_task_completion_potential_sum = 0.0  # 任务完成百分比势能

        last_time_step = -1.0
        time_stuck_counter = 0
        STUCK_THRESHOLD = 50

        # 初始化上一个事件的时间
        time_of_last_event = self.env.current_time

        if TrainParams.LTL_ENABLED:
            ltl_monitor = LTLMonitor(self.env, fixed_clauses=ltl_clauses)

            # 【性能优化】提取当前episode中LTL顺序约束的前置任务ID集合
            # 只有这些前置任务完成时才需要检查唤醒，其他任务完成时不需要
            sequential_prerequisite_tasks = set()
            if ltl_monitor and ltl_monitor.clauses:
                for clause in ltl_monitor.clauses:
                    if clause.type == LTL_SEQUENTIAL:
                        # param1是前置任务，param2是后续任务
                        # 只有前置任务完成才可能唤醒被阻塞的agents
                        sequential_prerequisite_tasks.add(clause.param1)
        else:
            ltl_monitor = None
            sequential_prerequisite_tasks = set()

        # 设置决策步数上限（训练时使用较小值避免卡死，评估时使用更大值）
        max_decision_steps = 2000  # if training else 2000  # 统一使用2000步

        # 【威布尔故障】记录故障智能体数量（用于统计，不终止episode）
        num_failed_agents = 0

        while (
            not self.env.finished
            and self.env.current_time < EnvParams.MAX_TIME
            and decision_step < max_decision_steps
        ):
            self.decision_step = decision_step
            mode = effective_constraint_mode  # 使用effective_constraint_mode而不是TrainParams.LTL_CONSTRAINT_TYPE
            # self._log_system_state("Start of Loop Iteration")

            # 【威布尔故障】检测故障智能体并设置为失活状态，但不终止episode
            if (
                EnvParams.VEHICLE_FAILURE_ENABLED
                and EnvParams.FAILURE_MODE == "simulate"
            ):
                for agent in self.env.agent_dic.values():
                    if agent.get("failed", False) and not agent.get(
                        "is_inactive", False
                    ):
                        # 设置智能体为失活状态
                        agent["is_inactive"] = True
                        agent["next_decision"] = np.inf
                        num_failed_agents += 1

                        if DEBUG or not training:  # 评估时总是打印
                            print(f"\n{'=' * 80}")
                            print(f"AGENT DEACTIVATED: Vehicle {agent['ID']} Failed!")
                            print(f"{'=' * 80}")
                            print(f"  - Failure time: {self.env.current_time:.2f}")
                            print(
                                f"  - Cumulative working time: {agent.get('cumulative_time', 0):.2f}"
                            )
                            print(f"  - Decision step: {decision_step}")
                            print(f"  - Remaining agents continue working...")
                            print(f"{'=' * 80}\n")

            # 【诊断】定期打印评估进度
            if ENABLE_EVAL_DIAGNOSTICS:
                current_real_time = time_module.time()
                if current_real_time - last_diag_print_time >= DIAG_PRINT_INTERVAL:
                    elapsed = current_real_time - episode_real_start_time
                    remaining_tasks = sum(
                        1
                        for t in self.env.task_dic.values()
                        if np.sum(t["requirements"]) > 0
                    )
                    completed_ratio = 1.0 - (remaining_tasks / len(self.env.task_dic))
                    current_total_reqs = sum(
                        np.sum(t["requirements"]) for t in self.env.task_dic.values()
                    )
                    delivered_reqs = initial_total_reqs - current_total_reqs

                    print(f"\n[EPISODE DIAG] 进度更新 (已运行 {elapsed:.0f}秒)")
                    print(f"  - Decision step: {decision_step}")
                    print(
                        f"  - Sim time: {self.env.current_time:.1f}/{EnvParams.MAX_TIME}"
                    )
                    print(
                        f"  - 剩余任务: {remaining_tasks}/{len(self.env.task_dic)} ({completed_ratio:.1%} 已完成)"
                    )
                    print(
                        f"  - 已交付需求: {delivered_reqs}/{initial_total_reqs} ({delivered_reqs / max(initial_total_reqs, 1):.1%})"
                    )
                    print(
                        f"  - 执行动作: MOVE={action_counters['MOVE']}, LOAD={action_counters['LOAD']}, UNLOAD={action_counters['UNLOAD']}, REJECT={action_counters['REJECTED']}"
                    )
                    total_sel = sum(action_type_selection_stats.values())
                    if total_sel > 0:
                        print(
                            f"  - 模型选择: MOVE={action_type_selection_stats['MOVE']} ({action_type_selection_stats['MOVE'] / total_sel:.1%}), "
                            f"LOAD={action_type_selection_stats['LOAD']} ({action_type_selection_stats['LOAD'] / total_sel:.1%}), "
                            f"UNLOAD={action_type_selection_stats['UNLOAD']} ({action_type_selection_stats['UNLOAD'] / total_sel:.1%})"
                        )
                    print(f"  - 完成任务数: {tasks_completed_count}")
                    print(f"  - Finished flag: {self.env.finished}")
                    last_diag_print_time = current_real_time

            # PPO模式下的rollout长度限制（仅训练时生效）
            if (
                training
                and TrainParams.ALGORITHM == "PPO"
                and decision_step >= TrainParams.PPO_ROLLOUT_LENGTH
            ):
                if ENABLE_EVAL_DIAGNOSTICS:
                    print(
                        f"[Worker {self.metaAgentID}] PPO rollout完成: {decision_step} 步"
                    )
                break

            # PPO进度日志（已注释）
            # if TrainParams.ALGORITHM == 'PPO' and training and decision_step % 100 == 0:
            #     print(f"[Worker {self.metaAgentID}] PPO进度: {decision_step}/{TrainParams.PPO_ROLLOUT_LENGTH} 步")

            if self.env.current_time == last_time_step:
                time_stuck_counter += 1
            else:
                last_time_step = self.env.current_time
                time_stuck_counter = 0

            if time_stuck_counter > STUCK_THRESHOLD:
                print(
                    "\n"
                    + "!" * 25
                    + f" EPISODE SEEMS STUCK! (WORKER {self.metaAgentID}) "
                    + "!" * 25
                )
                print(
                    f"  - Current Time: {self.env.current_time:.2f} (has not advanced for {time_stuck_counter} loops)"
                )
                print(f"  - Decision Step: {decision_step}")
                print(f"  - Algorithm: {TrainParams.ALGORITHM}")
                print(
                    f"  - PPO Rollout Progress: {decision_step}/{TrainParams.PPO_ROLLOUT_LENGTH if TrainParams.ALGORITHM == 'PPO' else 'N/A'}"
                )

                # PPO模式下：如果卡住且已收集足够数据，提前退出（仅训练时）
                if (
                    training
                    and TrainParams.ALGORITHM == "PPO"
                    and decision_step >= TrainParams.PPO_ROLLOUT_LENGTH
                ):
                    print(
                        f"[Worker {self.metaAgentID}] PPO已收集足够数据，提前退出卡住的episode"
                    )
                    break

                # ==================== NEW ENHANCED DIAGNOSTIC BLOCK START ====================
                print("\n--- AGENT STATES AT STALL (ENHANCED DIAGNOSIS) ---")
                for agent_id, agent_data in self.env.agent_dic.items():
                    # 提取基础信息
                    next_decision_str = (
                        f"{agent_data.get('next_decision', 'N/A'):.2f}"
                        if isinstance(agent_data.get("next_decision"), (int, float))
                        else str(agent_data.get("next_decision", "N/A"))
                    )
                    is_inactive = agent_data.get("is_inactive", False)
                    is_sleeping = agent_data.get("is_temp_sleeping", False)
                    location = agent_data.get("current_task", "N/A")
                    inventory = agent_data.get("inventory", {})
                    route = agent_data.get("route", [])
                    arrival_times = agent_data.get("arrival_time", [])

                    # 推断目的地 (下一个事件节点) 和到达时间
                    destination_node = "Idle / Awaiting Decision"
                    arrival_time_str = "N/A"

                    # 如果智能体的下一个决策时间是无穷大，且其路径中不止一个节点，那么它就在途中
                    if agent_data.get("next_decision") == np.inf and len(route) > 1:
                        # 目的地是路径的最后一个节点
                        destination_node = route[-1]
                        # 对应的到达时间是其列表中的最后一个
                        if arrival_times:
                            arrival_time_str = f"{arrival_times[-1]:.2f}"

                    # 结构化地打印每个智能体的详细信息
                    print(f"  - Agent {agent_id:<2}:")
                    print(
                        f"      Status       : Inactive={is_inactive!s:<5}, TempSleeping={is_sleeping!s:<5}"
                    )
                    print(
                        f"      State        : At Node={location:<4}, Inventory={inventory}"
                    )
                    print(
                        f"      Timing       : Next Decision @ T={next_decision_str}, Arrival @ T={arrival_time_str}"
                    )
                    print(f"      Route Plan   : Destination Node={destination_node}")
                    print(f"      Full Route   : {route}")
                # ===================== NEW ENHANCED DIAGNOSTIC BLOCK END =====================

                print("\n--- UNFINISHED TASK STATUSES AT STALL ---")
                for task_id, task_data in self.env.task_dic.items():
                    if not task_data.get("finished", False):
                        print(f"  - Task {task_id:<2}: {task_data.get('status')}")
                print("!" * 85 + "\n")

                # 主动跳出循环，防止日志刷屏并结束这个卡死的episode
                if TrainParams.ALGORITHM == "PPO":
                    print(
                        f"[Worker {self.metaAgentID}] PPO检测到死锁，退出当前episode（已收集{decision_step}步，目标{TrainParams.PPO_ROLLOUT_LENGTH}步）"
                    )
                break

            if decision_step > 4990:
                print(decision_step)
            # decision_times = [a.get('next_decision', np.inf) for a in self.env.agent_dic.values() if
            #                   not a.get('is_inactive', False)]
            # decision_times = [
            #     a.get('next_decision', np.inf)
            #     for a in self.env.agent_dic.values()
            #     if (not a.get('is_inactive', False)) and (not a.get('is_temp_sleeping', False))
            # ]
            #
            # arrival_times = [a['arrival_time'][-1] for a in self.env.agent_dic.values() if
            #                  not a.get('is_inactive', False) and a.get('next_decision') == np.inf and len(
            #                      a.get('arrival_time', [])) > 1]
            #
            # all_event_times = decision_times + arrival_times
            #
            # future_event_times = sorted([t for t in set(all_event_times) if t >= self.env.current_time])
            #
            # if not future_event_times:
            #     if not any(t == self.env.current_time for t in decision_times):
            #         # # ==================== 新增：终止前快照打印 ====================
            #         # print("\n" + "!" * 25 + " EPISODE TERMINATING: No future events found! " + "!" * 25)
            #         # print(f"Current Time: {self.env.current_time:.2f}")
            #         # print("Final state of all agents:")
            #         # for agent_id, agent_data in self.env.agent_dic.items():
            #         #     is_inactive = agent_data.get('is_inactive', False)
            #         #     next_decision = agent_data.get('next_decision', 'N/A')
            #         #     if isinstance(next_decision, float): next_decision = f"{next_decision:.2f}"
            #         #     inventory = agent_data.get('inventory', {})
            #         #     print(
            #         #         f"  - Agent {agent_id:<2}: Inactive={is_inactive!s:<5} | Next Decision at T={str(next_decision):<7} | Location={agent_data.get('current_task', 'N/A'):<4} | Inventory={inventory}")
            #         # print("!" * 95 + "\n")
            #         # # ===========================
            #         break
            #     else:
            #         next_event_time = self.env.current_time
            # else:
            #     next_event_time = future_event_times[0]
            #
            # if Event_Calculation:
            #     print(f"DEBUG [T={self.env.current_time:.4f}]: --- Event Calculation ---")
            #     print(f"  - Next event time calculated as: {next_event_time:.4f}")

            # 使用env.next_decision()来获取下一个事件时间（已修复arrival_time逻辑）
            (ready_agents_temp, blocked_agents_temp), next_event_time_from_env = (
                self.env.next_decision()
            )

            # 检查是否有未来事件
            if np.isinf(next_event_time_from_env) or np.isnan(next_event_time_from_env):
                if DEBUG:
                    print(
                        f"    (INFO) No future events found at T={self.env.current_time:.2f}. Terminating episode."
                    )

                # 详细诊断：打印所有agents的状态
                print("\n" + "=" * 80)
                print(
                    f"[STUCK DIAGNOSIS] Episode ending with no future events at T={self.env.current_time:.2f}"
                )
                print("=" * 80)
                print("\nAgent States:")
                for aid, agent in self.env.agent_dic.items():
                    next_dec = agent.get("next_decision", "N/A")
                    arrival_times = agent.get("arrival_time", [])
                    route = agent.get("route", [])
                    is_inactive = agent.get("is_inactive", False)
                    is_sleeping = agent.get("is_temp_sleeping", False)
                    inventory = agent.get("inventory", {})
                    current_loc = agent.get("current_task", "N/A")

                    # 判断状态
                    if is_inactive:
                        state = "INACTIVE"
                    elif is_sleeping:
                        state = "SLEEPING"
                    elif next_dec == np.inf and len(route) > len(arrival_times):
                        state = "IN-TRANSIT (WRONG!)"
                    elif next_dec == np.inf and arrival_times:
                        last_arrival = arrival_times[-1]
                        if last_arrival > self.env.current_time:
                            state = f"IN-TRANSIT (arrives@{last_arrival:.2f})"
                        else:
                            state = f"STUCK! (should have arrived@{last_arrival:.2f})"
                    elif next_dec == self.env.current_time:
                        state = "READY (should be in queue)"
                    elif next_dec < np.inf:
                        state = f"WAITING (decides@{next_dec:.2f})"
                    else:
                        state = "UNKNOWN"

                    print(f"  Agent {aid}: {state}")
                    print(f"    - Location: {current_loc}, Inventory: {inventory}")
                    print(f"    - next_decision: {next_dec}")
                    print(f"    - Route: {route}")
                    print(f"    - Arrival times: {arrival_times}")

                print("\nUnfinished Tasks:")
                for tid, task in self.env.task_dic.items():
                    if not task.get("finished", False):
                        print(
                            f"  Task {tid}: Status={task['status'].tolist()}, Req={task['requirements'].tolist()}"
                        )
                print("=" * 80 + "\n")
                break

            next_event_time = next_event_time_from_env

            tau = next_event_time - self.env.current_time

            self.env.current_time = next_event_time
            delta_t = self.env.current_time - time_of_last_event
            # 注意：不在这里更新time_of_last_event，而是在第一个智能体处理后更新
            # time_of_last_event = self.env.current_time  # <-- 这行导致tau=0的bug！
            # 记录agent_update之前的状态
            agents_before_update = {
                aid: {
                    "next_decision": a.get("next_decision"),
                    "arrival_time": a.get("arrival_time", [])[-1]
                    if a.get("arrival_time")
                    else None,
                    "is_temp_sleeping": a.get("is_temp_sleeping", False),
                }
                for aid, a in self.env.agent_dic.items()
            }

            self.env.agent_update()
            self.env.task_update()

            # 【CVAR_SMDP】累积全车队 hazard
            # delta_t 是上一个事件到当前事件的时间间隔
            if TrainParams.LTL_CONSTRAINT_TYPE == "CVAR_SMDP":
                # get_total_hazard_normalized 返回的是当前时刻全车队的 hazard rate 总和
                # 我们假设在这个小时间步 delta_t 内 rate 是常数 (或者用梯形法则，这里简化为矩形)
                current_hazard_rate = self.env.get_total_hazard_normalized()
                episode_accumulated_hazard += current_hazard_rate * delta_t

            # 【SOFT_POLICY】计算当前事件时间步的全车队总hazard
            event_hazard_sum = 0.0
            if TrainParams.LTL_CONSTRAINT_TYPE == "SOFT_POLICY":
                event_hazard_sum = self.env.get_total_hazard_normalized()

            # 检查哪些agents的状态被更新了
            agents_updated = []
            for aid, a in self.env.agent_dic.items():
                before = agents_before_update[aid]
                after_next_dec = a.get("next_decision")
                if (
                    before["next_decision"] != after_next_dec
                    and after_next_dec == self.env.current_time
                ):
                    agents_updated.append(aid)
                    if DEBUG:
                        print(
                            f"    (AGENT_ARRIVAL) [T={self.env.current_time:.2f}s] Agent {aid} has arrived and is ready for decision."
                        )

            # 【性能优化】精准的唤醒检查逻辑
            # 时间推进时不需要检查唤醒，因为agents到达不会改变LTL约束状态
            # 只有任务完成（在agent_step后处理）才会解除顺序约束
            # 因此这里完全不需要调用check_and_wakeup_agents
            initial_ready_agents, _ = self.env.next_decision()
            # agents_to_process_queue = [aid for aid in self.env.agent_dic.keys() if
            #                            self.env.agent_dic[aid]['next_decision'] == self.env.current_time and not
            #                            self.env.agent_dic[aid].get('is_inactive', False)]

            for aid in self.env.agent_dic.keys():
                agent = self.env.agent_dic[aid]
                if agent.get("is_temp_sleeping", False):
                    pass
                    # print(
                    #     f"[DEBUG] Agent {aid} is TEMP_SLEEPING and filtered out: next_decision={agent['next_decision']}")

            if process_debug:
                sleepers_now = [
                    aid
                    for aid, a in self.env.agent_dic.items()
                    if a.get("is_temp_sleeping", False)
                ]
                print(
                    f"[WORKER][DEBUG] t={self.env.current_time:.2f}, sleeping={sleepers_now}"
                )

            agents_to_process_queue = [
                aid
                for aid, a in self.env.agent_dic.items()
                if (a.get("next_decision", np.inf) == self.env.current_time)
                and (not a.get("is_inactive", False))
                and (not a.get("is_temp_sleeping", False))
            ]
            if process_debug:
                print(
                    f"[WORKER][DEBUG] t={self.env.current_time:.2f}, to_process={agents_to_process_queue}"
                )

            # 诊断：检查是否有agents的next_decision等于current_time但没有被加入队列
            agents_ready_but_not_queued = []
            for aid, a in self.env.agent_dic.items():
                if a.get("next_decision", np.inf) == self.env.current_time:
                    if aid not in agents_to_process_queue:
                        agents_ready_but_not_queued.append(
                            {
                                "id": aid,
                                "is_inactive": a.get("is_inactive", False),
                                "is_temp_sleeping": a.get("is_temp_sleeping", False),
                                "route": a.get("route", []),
                                "arrival_time": a.get("arrival_time", []),
                            }
                        )

            if agents_ready_but_not_queued:
                print(
                    f"\n[WARNING] [T={self.env.current_time:.2f}] Agents with next_decision = current_time but NOT in queue:"
                )
                for info in agents_ready_but_not_queued:
                    print(
                        f"  Agent {info['id']}: inactive={info['is_inactive']}, sleeping={info['is_temp_sleeping']}"
                    )
                    print(
                        f"    Route: {info['route']}, Arrival: {info['arrival_time']}"
                    )

            # === 使用独立的RNG来打乱决策顺序（确保可复现性）===
            # 这样多个智能体同时决策时，第一个智能体承担时间惩罚的选择是随机但可复现的
            self.decision_order_rng.shuffle(agents_to_process_queue)

            num_acting_agents = len(agents_to_process_queue)

            # 【Per-Agent Hazard】保存原始队列副本，用于调试和验证
            agents_in_this_batch = list(agents_to_process_queue)  # 浅拷贝

            # 注意：不在这里提前更新time_of_last_event
            # 应该在第一个智能体处理并计算奖励后才更新（第1053行）
            # if num_acting_agents > 0:
            #     time_of_last_event = self.env.current_time

            if decision_step > 4990:
                print("agents_to_process_queue:", agents_to_process_queue)

            inner_loop_step = 0
            # 设置一个非常宽松的阈值，正常情况下绝不应该达到
            # 在一个时间步内，总决策次数不应超过智能体总数的数倍
            INNER_LOOP_THRESHOLD = self.env.agents_num * 10 + 5

            processed_this_turn = set()

            # === 用于跟踪是否是第一个处理的智能体（用于时间惩罚分配）===
            agent_index_in_batch = 0  # 当前处理的是第几个智能体（从0开始）
            # 记录这一批智能体开始处理时的上一次事件时间
            batch_start_last_event_time = time_of_last_event

            while agents_to_process_queue:
                # ==================== NEW: Inner Loop Stall Detection Logic ====================
                if inner_loop_step > INNER_LOOP_THRESHOLD:
                    print(
                        "\n"
                        + "#" * 25
                        + f" EPISODE STUCK (INNER LOOP)! (WORKER {self.metaAgentID}) "
                        + "#" * 25
                    )
                    print(
                        f"  - Current Time: {self.env.current_time:.2f} (Inner loop spinning without advancing time)"
                    )
                    print(f"  - Decision Step: {decision_step}")
                    print(f"  - Inner Loop Iterations: {inner_loop_step}")
                    print("\n--- AGENT STATES AT STALL ---")
                    for agent_id, agent_data in self.env.agent_dic.items():
                        next_decision_str = (
                            f"{agent_data.get('next_decision', 'N/A'):.2f}"
                            if isinstance(agent_data.get("next_decision"), (int, float))
                            else str(agent_data.get("next_decision", "N/A"))
                        )
                        print(
                            f"  - Agent {agent_id:<2}: Inactive={agent_data.get('is_inactive', False)!s:<5} | Next Decision at T={next_decision_str:<7} | Loc={agent_data.get('current_task', 'N/A'):<4} | Inv={agent_data.get('inventory', {})}"
                        )
                    print("\n--- AGENTS IN QUEUE AT STALL ---")
                    print(f"  - {agents_to_process_queue}")
                    print("\n--- UNFINISHED TASK STATUSES AT STALL ---")
                    for task_id, task_data in self.env.task_dic.items():
                        if not task_data.get("finished", False):
                            print(f"  - Task {task_id:<2}: {task_data.get('status')}")
                    print("#" * 85 + "\n")
                    # 主动跳出内层循环
                    # break
                inner_loop_step += 1
                # ==============================================================================================

                agent_id = agents_to_process_queue.pop(0)

                if agent_id in processed_this_turn:
                    continue
                processed_this_turn.add(agent_id)

                agent = self.env.agent_dic[agent_id]
                if agent.get("is_temp_sleeping", False):
                    print(
                        f"DEBUG [T={self.env.current_time:.4f}]: Skipping decision for sleeping Agent {agent_id}."
                    )
                    continue
                final_action_dict = None

                # 初始化用于存储的动作字典 (type, destination, cargo, quantity)
                action_to_store = torch.full((4,), -1, dtype=torch.long)

                with torch.no_grad():
                    # ===== 效率优化：避免双重网络调用 =====
                    # 修正后的方法B不再需要预先调用网络，因为：
                    # 1. 成本计算只需要策略概率，可以在一次前向传播后计算
                    # 2. 这样可以将计算开销减半

                    # 第一步：获取观测和掩码（不需要预先传入policy_logits）
                    (
                        task_info,
                        total_agents,
                        global_mask,
                        ltl_info,
                        masks_dict,
                        _,
                        inaction_reason,
                        blocking_clauses,
                    ) = self.env.agent_observe(
                        agent_id,
                        ltl_monitor,
                        max_waiting,
                        policy_logits=None,  # 不需要预先计算
                        constraint_mode=effective_constraint_mode,
                    )

                    action_type_mask = masks_dict["action_type"]

                    # 【诊断】统计哪些动作类型可用
                    if ENABLE_EVAL_DIAGNOSTICS:
                        # action_type_mask: True表示被mask（不可用），False表示可用
                        if not action_type_mask[self.env.ACTION_MOVE]:
                            action_type_available_stats["MOVE"] += 1
                        if not action_type_mask[self.env.ACTION_LOAD]:
                            action_type_available_stats["LOAD"] += 1
                        if not action_type_mask[self.env.ACTION_UNLOAD]:
                            action_type_available_stats["UNLOAD"] += 1

                    # 【关键修复】不要在这里直接标记为inactive！
                    # 应该让后续的inaction_reason处理逻辑来决定是永久失活还是临时休眠
                    # 如果all actions masked，我们跳过网络推理，但仍然需要处理inaction_reason
                    if np.all(action_type_mask):
                        # 跳过网络推理，直接进入inaction处理逻辑（第1039行）
                        pass
                    else:
                        # --- 只有在有可选动作时，才进入核心决策逻辑 ---
                        # 第二步：转换为tensor并padding
                        task_info_t, total_agents_t, global_mask_t = self.convert_torch(
                            [task_info, total_agents, global_mask]
                        )
                        task_info_t, total_agents_t, global_mask_t = self.obs_padding(
                            task_info_t, total_agents_t, global_mask_t
                        )
                        index_t = (
                            torch.LongTensor([agent_id])
                            .reshape(1, 1, 1)
                            .to(self.device)
                        )

                        # 转换ltl_info到tensor（根据编码类型处理）
                        if TrainParams.LTL_ENCODING_TYPE == "C":
                            # 模式C：ltl_info是字典，需要分别转换每个组件
                            ltl_info_t = {
                                "feasibility": torch.from_numpy(
                                    ltl_info["feasibility"]
                                ).to(self.device),
                                "edge_index": torch.from_numpy(
                                    ltl_info["edge_index"]
                                ).to(self.device),
                                "edge_attr": torch.from_numpy(ltl_info["edge_attr"]).to(
                                    self.device
                                ),
                            }
                        else:
                            # 模式A/B：ltl_info是numpy数组
                            ltl_info_t = torch.from_numpy(ltl_info).to(self.device)

                        # ===== 【方案一关键】：获取依赖图 =====
                        if TrainParams.LTL_ENABLED and ltl_monitor is not None:
                            dependency_adj = (
                                ltl_monitor.get_dependency_adjacency_matrix()
                            )
                            dependency_graph_t = torch.from_numpy(dependency_adj).to(
                                self.device
                            )
                        else:
                            dependency_graph_t = None

                        # 第三步：一次网络前向传播（同时用于成本计算和决策）
                        policy_logits, reward_value, cost_quantiles = self.local_net(
                            task_info_t,
                            total_agents_t,
                            global_mask_t,
                            index_t,
                            ltl_info_t,
                            dependency_graph_t,
                        )

                        # 【CVAR_SMDP Quantile】存储cost_quantiles用于后续的quantile regression
                        # cost_quantiles: [1, NUM_QUANTILES] - 当前状态的成本分布估计

                        # 第四步：使用同一个policy_logits计算成本（方法B/D）
                        # 【改进】评估时也计算成本用于监控，即使使用HARD模式掩码
                        if (
                            not training
                            and effective_constraint_mode == "HARD"
                            and TrainParams.LTL_CONSTRAINT_TYPE != "HARD"
                        ):
                            # 评估模式下，训练时使用软约束，评估时使用硬约束
                            # 但仍计算软约束的成本用于监控训练效果
                            if (
                                TrainParams.LTL_CONSTRAINT_TYPE == "SOFT_POLICY"
                                and ltl_monitor
                            ):
                                cost_info = self.env._calculate_policy_based_cost(
                                    agent_id, policy_logits, ltl_monitor
                                )
                            elif (
                                TrainParams.LTL_CONSTRAINT_TYPE == "SOFT_HYBRID_STATE"
                                and ltl_monitor
                            ):
                                cost_info = self.env._calculate_hybrid_state_cost(
                                    agent_id, policy_logits, ltl_monitor
                                )
                            else:
                                cost_info = {"total_cost": 0.0}
                        elif (
                            effective_constraint_mode
                            in ["SOFT_POLICY", "SOFT_HYBRID_STATE"]
                            and ltl_monitor
                        ):
                            # 训练模式下，计算策略基础成本或混合成本
                            cost_info = (
                                self.env._calculate_policy_based_cost(
                                    agent_id, policy_logits, ltl_monitor
                                )
                                if effective_constraint_mode == "SOFT_POLICY"
                                else self.env._calculate_hybrid_state_cost(
                                    agent_id, policy_logits, ltl_monitor
                                )
                            )
                        elif effective_constraint_mode == "SOFT_DISCRETE":
                            # SOFT_DISCRETE模式：在observe阶段使用占位成本0.0
                            # 真实成本将在动作执行后计算（见第1073行）
                            cost_info = {"total_cost": 0.0}
                        else:
                            # HARD模式或无LTL约束
                            cost_info = {"total_cost": 0.0}

                        if inaction_reason == "ACTIONS_AVAILABLE":
                            action_type_mask_t = (
                                torch.tensor(action_type_mask, dtype=torch.bool)
                                .to(self.device)
                                .unsqueeze(0)
                            )
                            action_type_logits = policy_logits[
                                "action_type"
                            ].masked_fill(action_type_mask_t, -1e9)
                            action_type_dist = Categorical(logits=action_type_logits)

                            if sample:
                                chosen_action_type = action_type_dist.sample()
                            else:
                                chosen_action_type = torch.argmax(
                                    action_type_dist.logits, dim=-1
                                )

                            # 【诊断】打印前10次决策的action_type选择详情
                            if ENABLE_EVAL_DIAGNOSTICS and decision_step < 10:
                                print(
                                    f"\n[决策诊断 Step {decision_step}] Agent {agent_id}"
                                )
                                print(
                                    f"  原始logits: {policy_logits['action_type'].cpu().numpy()}"
                                )
                                print(f"  mask: {action_type_mask}")
                                print(
                                    f"  masked logits: {action_type_logits.cpu().numpy()}"
                                )
                                print(
                                    f"  probs: {action_type_dist.probs.cpu().numpy()}"
                                )
                                print(
                                    f"  chosen: {chosen_action_type.item()} (0=MOVE, 1=LOAD, 2=UNLOAD)"
                                )

                            episode_metrics["entropy/action_type"].append(
                                action_type_dist.entropy().item()
                            )
                            episode_metrics["prob_max/action_type"].append(
                                action_type_dist.probs.max().item()
                            )

                            # 【新增】记录每种动作类型的概率（用于诊断）
                            action_probs = action_type_dist.probs.cpu().numpy()[
                                0
                            ]  # [3]
                            if "prob_move" not in episode_metrics:
                                episode_metrics["prob_move"] = []
                                episode_metrics["prob_load"] = []
                                episode_metrics["prob_unload"] = []
                            episode_metrics["prob_move"].append(float(action_probs[0]))
                            episode_metrics["prob_load"].append(float(action_probs[1]))
                            episode_metrics["prob_unload"].append(
                                float(action_probs[2])
                            )

                            # 记录动作类型
                            action_to_store[0] = chosen_action_type.item()

                            # 【新增】总是统计模型选择的动作类型（无论是否诊断模式）
                            if chosen_action_type.item() == self.env.ACTION_MOVE:
                                action_type_chosen_counts["MOVE"] += 1
                            elif chosen_action_type.item() == self.env.ACTION_LOAD:
                                action_type_chosen_counts["LOAD"] += 1
                            elif chosen_action_type.item() == self.env.ACTION_UNLOAD:
                                action_type_chosen_counts["UNLOAD"] += 1

                            # 【诊断】统计模型选择的动作类型
                            if ENABLE_EVAL_DIAGNOSTICS:
                                if chosen_action_type.item() == self.env.ACTION_MOVE:
                                    action_type_selection_stats["MOVE"] += 1
                                elif chosen_action_type.item() == self.env.ACTION_LOAD:
                                    action_type_selection_stats["LOAD"] += 1
                                elif (
                                    chosen_action_type.item() == self.env.ACTION_UNLOAD
                                ):
                                    action_type_selection_stats["UNLOAD"] += 1

                            if chosen_action_type.item() == self.env.ACTION_MOVE:
                                destination_raw_logits = policy_logits["destination"]
                                destination_logits = policy_logits[
                                    "destination"
                                ].masked_fill(global_mask_t.bool(), -1e9)
                                destination_dist = Categorical(
                                    logits=destination_logits
                                )

                                if sample:
                                    chosen_destination = destination_dist.sample()
                                else:
                                    chosen_destination = torch.argmax(
                                        destination_dist.logits, dim=-1
                                    )

                                episode_metrics["entropy/destination"].append(
                                    destination_dist.entropy().item()
                                )
                                episode_metrics["prob_max/destination"].append(
                                    destination_dist.probs.max().item()
                                )

                                # NEW: 记录目的地
                                action_to_store[1] = chosen_destination.item()

                                final_action_dict = {
                                    "type": self.env.ACTION_MOVE,
                                    "destination": chosen_destination.item(),
                                }

                            else:  # LOAD or UNLOAD
                                if chosen_action_type.item() == self.env.ACTION_LOAD:
                                    cargo_mask_t = (
                                        torch.tensor(
                                            masks_dict["cargo_to_load"],
                                            dtype=torch.bool,
                                        )
                                        .to(self.device)
                                        .unsqueeze(0)
                                    )

                                    cargo_raw_logits = policy_logits["cargo"]
                                    cargo_logits = policy_logits["cargo"].masked_fill(
                                        cargo_mask_t, -1e9
                                    )

                                    if not torch.all(cargo_mask_t):
                                        cargo_dist = Categorical(logits=cargo_logits)
                                        if sample:
                                            chosen_cargo_type = cargo_dist.sample()
                                        else:
                                            chosen_cargo_type = torch.argmax(
                                                cargo_dist.logits, dim=-1
                                            )

                                        episode_metrics["entropy/cargo"].append(
                                            cargo_dist.entropy().item()
                                        )
                                        episode_metrics["prob_max/cargo"].append(
                                            cargo_dist.probs.max().item()
                                        )

                                        # NEW: 记录货物类型
                                        action_to_store[2] = chosen_cargo_type.item()

                                        # 自动使用最大容量装载（与Greedy策略一致）
                                        max_capacity = int(
                                            agent["capacity"][chosen_cargo_type.item()]
                                        )
                                        actual_quantity = max_capacity
                                        action_to_store[3] = (
                                            max_capacity - 1 if max_capacity > 0 else 0
                                        )

                                        quantity_vec = np.zeros(self.env.traits_dim)
                                        quantity_vec[chosen_cargo_type.item()] = (
                                            actual_quantity
                                        )
                                        final_action_dict = {
                                            "type": self.env.ACTION_LOAD,
                                            "quantity_vec": quantity_vec,
                                        }

                                elif (
                                    chosen_action_type.item() == self.env.ACTION_UNLOAD
                                ):
                                    carried_type = agent["inventory"]["type"]
                                    if carried_type is not None:
                                        action_to_store[2] = carried_type

                                    quantity_vec = np.zeros(self.env.traits_dim)
                                    if carried_type is not None:
                                        quantity_vec[carried_type] = agent["inventory"][
                                            "quantity"
                                        ]
                                    final_action_dict = {
                                        "type": self.env.ACTION_UNLOAD,
                                        "quantity_vec": quantity_vec,
                                    }

                        elif inaction_reason == "NO_ACTION_BY_SAFETY_LTL":
                            # 被静态安全约束永久阻塞 -> 永久失活
                            if DEBUG:
                                print(
                                    f"    (LTL PERMANENT BLOCK) Agent {agent_id} is permanently inactivated by a static SAFETY constraint."
                                )
                            agent["is_inactive"] = True
                            agent["next_decision"] = np.inf
                            continue  # 跳过该智能体
                        elif inaction_reason == "NO_ACTION_BY_LTL":
                            # 被动态LTL约束阻塞（如顺序约束）-> 临时休眠
                            if DEBUG:
                                print(
                                    f"    (LTL BLOCK) Agent {agent_id} entering temporary sleep due to LTL constraints."
                                )
                            agent["is_temp_sleeping"] = True
                            agent["next_decision"] = np.inf
                            agent["blocking_clauses"] = blocking_clauses
                            continue  # Skip to the next agent in the queue
                        elif inaction_reason == "NO_ACTION_TEMPORARILY":
                            # 当前无可执行动作，但有能力贡献 -> 临时休眠，等待环境变化
                            if DEBUG:
                                print(
                                    f"    (TEMPORARY BLOCK) Agent {agent_id} entering temporary sleep. Will retry when environment changes."
                                )
                            agent["is_temp_sleeping"] = True
                            agent["next_decision"] = np.inf
                            agent[
                                "blocking_clauses"
                            ] = []  # 不是LTL阻塞，所以没有blocking clauses
                            continue  # Skip to the next agent in the queue
                        elif inaction_reason == "NO_ACTION_BY_DEFAULT":
                            # 永久无法对任何未完成任务贡献 -> 永久失活
                            if DEBUG:
                                print(
                                    f"    (PERMANENT BLOCK) Agent {agent_id} permanently inactive - cannot contribute to any unfinished task."
                                )
                            agent["is_inactive"] = True
                            agent["next_decision"] = np.inf
                            continue  # Skip to the next agent in the queue

                        elif inaction_reason == "TEMP_SLEEPING":
                            continue

                if final_action_dict:
                    # 【改进】评估时也计算离散成本用于监控，即使使用HARD模式掩码
                    if mode == "SOFT_DISCRETE" and ltl_monitor:
                        discrete_cost = self.env.get_discrete_action_cost(
                            agent_id, final_action_dict, ltl_monitor
                        )
                        # 用精确计算的成本覆盖掉从observe阶段获取的占位成本
                        cost_info = {"total_cost": discrete_cost}
                    elif (
                        not training
                        and mode == "HARD"
                        and TrainParams.LTL_CONSTRAINT_TYPE == "SOFT_DISCRETE"
                        and ltl_monitor
                    ):
                        # 评估模式下，训练时使用SOFT_DISCRETE，评估时使用HARD模式
                        # 但仍计算离散成本用于监控训练效果
                        discrete_cost = self.env.get_discrete_action_cost(
                            agent_id, final_action_dict, ltl_monitor
                        )
                        cost_info = {"total_cost": discrete_cost}
                    remaining_before = sum(
                        np.sum(t.get("status", 0)) for t in self.env.task_dic.values()
                    )
                    potential_before = (
                        total_initial_requirements - remaining_before
                    )  # 已完成的工作量

                    # 【方法E】保存动作前的LTL势能（在update_monitor之前）
                    if ltl_monitor:
                        if TrainParams.LTL_POTENTIAL_TYPE == "TOPOLOGY":
                            phi_ltl_before = ltl_monitor.compute_topology_potential()
                        else:
                            phi_ltl_before = ltl_monitor.compute_ltl_potential()
                    else:
                        phi_ltl_before = 0.0

                    # 【修复时间方向】保存动作执行前的agent age（用于正确计算hazard）
                    # 因为agent_step会更新cumulative_time，所以必须在此之前保存
                    agent_age_before = self.env.agent_dic[agent_id].get(
                        "cumulative_time", 0.0
                    )

                    _, doable, f_t, event_info = self.env.agent_step(
                        agent_id, final_action_dict, decision_step
                    )

                    # 【统计动作执行】（无论是否诊断模式都要统计）
                    if doable:
                        action_type = final_action_dict["type"]
                        if action_type == self.env.ACTION_MOVE:
                            action_counters["MOVE"] += 1
                        elif action_type == self.env.ACTION_LOAD:
                            action_counters["LOAD"] += 1
                        elif action_type == self.env.ACTION_UNLOAD:
                            action_counters["UNLOAD"] += 1
                    else:
                        action_counters["REJECTED"] += 1

                    # 【诊断】详细打印（仅诊断模式）
                    if ENABLE_EVAL_DIAGNOSTICS:
                        if doable:
                            action_type = final_action_dict["type"]
                            # 前20次LOAD打印详细信息
                            if (
                                action_type == self.env.ACTION_LOAD
                                and action_counters["LOAD"] <= 20
                            ):
                                agent_info = self.env.agent_dic[agent_id]
                                qvec = final_action_dict.get(
                                    "quantity_vec", [0] * self.env.traits_dim
                                )
                                cargo_type = np.argmax(qvec)
                                quantity = int(qvec[cargo_type])
                                loc = agent_info["current_task"]
                                print(
                                    f"[ACTION DIAG] LOAD #{action_counters['LOAD']}: Agent {agent_id} at node {loc} loads {quantity}x Type{cargo_type}"
                                )
                            # 前20次UNLOAD打印详细信息
                            elif (
                                action_type == self.env.ACTION_UNLOAD
                                and action_counters["UNLOAD"] <= 20
                            ):
                                agent_info = self.env.agent_dic[agent_id]
                                qvec = final_action_dict.get(
                                    "quantity_vec", [0] * self.env.traits_dim
                                )
                                cargo_type = np.argmax(qvec)
                                quantity = int(qvec[cargo_type])
                                loc = agent_info["current_task"]
                                if loc >= 0:  # 任务节点
                                    task = self.env.task_dic[loc]
                                    reqs_before = task["requirements"][cargo_type]
                                    print(
                                        f"[ACTION DIAG] UNLOAD #{action_counters['UNLOAD']}: Agent {agent_id} at Task {loc} unloads {quantity}x Type{cargo_type} (task需求: {reqs_before})"
                                    )
                                else:  # 仓库
                                    depot_id = -loc - 1
                                    print(
                                        f"[ACTION DIAG] UNLOAD #{action_counters['UNLOAD']}: Agent {agent_id} at Depot {depot_id} unloads {quantity}x Type{cargo_type}"
                                    )

                            # 任务完成统计
                            if f_t:
                                tasks_completed_count += len(f_t)
                                print(
                                    f"\n[ACTION DIAG] ✅✅✅ 任务完成！Agent {agent_id} 完成了 {len(f_t)} 个任务: {f_t} ✅✅✅\n"
                                )

                    if ltl_monitor and event_info:
                        ltl_monitor.update_monitor(
                            event_info["type"], event_info["params"]
                        )

                        # 【性能优化】最精准的唤醒检查逻辑
                        # 只在LTL顺序约束的前置任务完成时才检查唤醒
                        # 例如：如果约束是"Task 5 must precede Task 10"，只有Task 5完成时才检查
                        if event_info["type"] == "task_finish":
                            finished_task_id = event_info["params"].get("task_id")
                            if finished_task_id in sequential_prerequisite_tasks:
                                self.check_and_wakeup_agents(
                                    ltl_monitor,
                                    constraint_mode=effective_constraint_mode,
                                )

                    if doable:
                        remaining_after = sum(
                            np.sum(t.get("status", 0))
                            for t in self.env.task_dic.values()
                        )
                        potential_after = total_initial_requirements - remaining_after

                        # === 【方案D一致性】势能折扣应该与buffer存储的tau一致 ===
                        # 关键理论：同一事件内的agents在同一时刻决策（Event-Boundary Discounting）
                        # - 第一个agent: 时间从t推进到t'，状态转移 s(t)→s'(t')，τ=Δt
                        # - 后续agents: 都在t'时刻做瞬时决策，状态转移 s'(t')→s''(t')，τ=0
                        #
                        # 势能塑形正确性验证：
                        #   Agent 0: F_0 = exp(-β·τ)·φ(s_1) - φ(s_0)
                        #   Agent 1: F_1 = exp(-β·0)·φ(s_2) - φ(s_1) = φ(s_2) - φ(s_1)
                        #   Agent 2: F_2 = exp(-β·0)·φ(s_3) - φ(s_2) = φ(s_3) - φ(s_2)
                        #
                        # 累积回报（考虑SMDP折扣）：
                        #   G = F_0 + exp(-β·τ)·F_1 + exp(-β·τ)·F_2 + ...
                        #     = [exp(-β·τ)·φ(s_1) - φ(s_0)]
                        #     + exp(-β·τ)·[φ(s_2) - φ(s_1)]
                        #     + exp(-β·τ)·[φ(s_3) - φ(s_2)]
                        #     = exp(-β·τ)·φ(s_final) - φ(s_0)  ✅ 完美抵消！
                        #
                        # 这确保了：
                        # 1. reward折扣和GAE折扣完全一致
                        # 2. 势能中间项可以完美抵消（满足势能塑形定理）
                        if agent_index_in_batch == 0:
                            tau_for_this_agent = tau  # 第一个agent：承担真实时间推进
                        else:
                            tau_for_this_agent = 0.0  # 后续agents：瞬时决策（同一时刻）

                        # === 势能塑形奖励（标准形式，满足势能塑形定理）===
                        # 【进展势能】基于已完成的需求量
                        if TrainParams.EXECUTION_MODE == "mdp":
                            # MDP标准势能塑形：r' = γ·φ(s') - φ(s)
                            # 确保势能塑形的gamma与PPO算法的gamma一致，满足策略不变性定理
                            # 累积回报：G' = Σ γ^t·[γ·φ(s_{t+1}) - φ(s_t)]
                            #          = γ^T·φ(s_T) - φ(s_0)  （中间项完全抵消）
                            # 其中 φ = 已完成需求量
                            progress_shaping_reward = (
                                TrainParams.REWARD_COMPLETED_DEMAND_WEIGHT
                                * (
                                    TrainParams.GAMMA * potential_after
                                    - potential_before
                                )
                            )

                            # DEBUG: 打印前20步的势能变化
                            if decision_step < 20 and not training:
                                print(
                                    f"\n[DEBUG SHAPING] Step {decision_step}, Agent {agent_id} (batch #{agent_index_in_batch})"
                                )
                                print(
                                    f"  potential_before: {potential_before:.2f}, potential_after: {potential_after:.2f}"
                                )
                                print(f"  gamma: {TrainParams.GAMMA}")
                                print(
                                    f"  delta_potential: {potential_after - potential_before:.2f}"
                                )
                                print(
                                    f"  progress_shaping_reward: {progress_shaping_reward:.4f}"
                                )
                                if TrainParams.REWARD_TIME_POTENTIAL_WEIGHT != 0:
                                    print(
                                        f"  [Time Potential Debug - MDP mode, time potential = 0]"
                                    )
                        else:
                            # SMDP标准势能塑形：r' = exp(-β·Δt)·φ(s') - φ(s)
                            # 确保中间项完全抵消，满足势能塑形定理
                            # 累积回报：G' = Σ exp(-β·T_i)·[exp(-β·Δt_i)·φ(s_{i+1}) - φ(s_i)]
                            #          = exp(-β·T_final)·φ(s_final) - φ(s_0)  （中间项完全抵消）
                            # 其中 φ = 已完成需求量, β = 时间折扣率
                            discount_factor_for_potential = np.exp(
                                -TrainParams.BETA * tau_for_this_agent
                            )
                            progress_shaping_reward = (
                                TrainParams.REWARD_COMPLETED_DEMAND_WEIGHT
                                * (
                                    discount_factor_for_potential * potential_after
                                    - potential_before
                                )
                            )

                        # === 【时间势能】基于累积时间 ===
                        # 势能函数：φ_time(s) = -T_current
                        # SMDP标准势能塑形：r_time = c * [exp(-β·τ) * φ_time(s') - φ_time(s)]
                        #                  = c * [exp(-β·τ) * (-T_after) - (-T_before)]
                        #                  = c * [T_before - exp(-β·τ) * T_after]
                        # 累积贡献：G_time = -c * exp(-β·T_final) * T_final（伸缩级数性质）
                        time_shaping_reward = 0.0
                        if TrainParams.REWARD_TIME_POTENTIAL_WEIGHT != 0:
                            if TrainParams.EXECUTION_MODE == "mdp":
                                # MDP模式：时间势能不太适用（因为没有显式的时间变量）
                                # 这里保持为0或者可以基于步数
                                time_shaping_reward = 0.0
                            else:
                                # SMDP模式：使用累积时间作为势能
                                T_before = batch_start_last_event_time
                                T_after = self.env.current_time
                                discount_factor = np.exp(
                                    -TrainParams.BETA * tau_for_this_agent
                                )
                                # r_time = c * [T_before - exp(-β·τ) * T_after]
                                time_shaping_reward = (
                                    TrainParams.REWARD_TIME_POTENTIAL_WEIGHT
                                    * (T_before - discount_factor * T_after)
                                )

                                # DEBUG: 打印前20步的时间势能计算
                                if decision_step < 20 and not training:
                                    print(
                                        f"\n[DEBUG TIME POTENTIAL] Step {decision_step}, Agent {agent_id} (batch #{agent_index_in_batch})"
                                    )
                                    print(
                                        f"  T_before: {T_before:.4f}, T_after: {T_after:.4f}, τ: {tau_for_this_agent:.4f}"
                                    )
                                    print(
                                        f"  β: {TrainParams.BETA:.6f}, exp(-β·τ): {discount_factor:.6f}"
                                    )
                                    print(
                                        f"  weight: {TrainParams.REWARD_TIME_POTENTIAL_WEIGHT}"
                                    )
                                    print(
                                        f"  time_shaping_reward: {time_shaping_reward:.6f}"
                                    )
                                    print(
                                        f"  progress_shaping_reward: {progress_shaping_reward:.4f}"
                                    )

                        # === 【任务完成百分比势能】基于已完成任务数 ===
                        # 势能函数：φ_task(s) = num_finished_tasks / total_tasks
                        # 理论：这是一个标准势能塑形，满足策略不变性定理
                        # 累积贡献在episode结束时：
                        #   MDP:  G_task = γ^T·φ(s_T) - φ(s_0) = γ^T·1 - 0
                        #   SMDP: G_task = exp(-β·T)·φ(s_T) - φ(s_0) = exp(-β·T)·1 - 0
                        task_completion_shaping_reward = 0.0

                        # 获取权重系数
                        dense_weight = 0.0
                        if TrainParams.EXECUTION_MODE == "smdp" and hasattr(
                            TrainParams, "SMDP_DENSE_REWARD_WEIGHT"
                        ):
                            dense_weight = TrainParams.SMDP_DENSE_REWARD_WEIGHT
                        elif hasattr(
                            TrainParams, "REWARD_TASK_COMPLETION_POTENTIAL_WEIGHT"
                        ):
                            dense_weight = (
                                TrainParams.REWARD_TASK_COMPLETION_POTENTIAL_WEIGHT
                            )

                        if dense_weight != 0:
                            # 计算动作前后的任务完成百分比
                            # 势函数语义: φ(s) = num_finished_tasks / total_tasks ∈ [0,1]
                            total_tasks = len(self.env.task_dic)
                            if total_tasks > 0:
                                # 【修复】agent_step() 执行后环境已更新，task['finished'] 已包含本步完成的任务
                                # 因此先统计动作后状态，再用 f_t 反推动作前状态

                                # 动作后：统计当前已完成任务数（包含本步新完成的）
                                finished_after = sum(
                                    1
                                    for t in self.env.task_dic.values()
                                    if t.get("finished", False)
                                )

                                # 动作前：减去本步新完成的任务数
                                finished_before = (
                                    finished_after - len(f_t) if f_t else finished_after
                                )

                                potential_task_before = finished_before / total_tasks
                                potential_task_after = finished_after / total_tasks

                                # 应用势能塑形公式
                                if TrainParams.EXECUTION_MODE == "mdp":
                                    # MDP: r' = weight × [γ·φ(s') - φ(s)]
                                    task_completion_shaping_reward = dense_weight * (
                                        TrainParams.GAMMA * potential_task_after
                                        - potential_task_before
                                    )
                                else:
                                    # SMDP: r' = weight × [exp(-β·τ)·φ(s') - φ(s)]
                                    discount_factor_for_task = np.exp(
                                        -TrainParams.BETA * tau_for_this_agent
                                    )
                                    task_completion_shaping_reward = dense_weight * (
                                        discount_factor_for_task * potential_task_after
                                        - potential_task_before
                                    )

                                # DEBUG: 打印任务完成势能（当有任务完成时）
                                if f_t and decision_step < 100 and not training:
                                    print(
                                        f"\n[DEBUG TASK COMPLETION POTENTIAL] Step {decision_step}, Agent {agent_id}"
                                    )
                                    print(f"  Tasks finished this step: {f_t}")
                                    print(
                                        f"  potential_task_before: {potential_task_before:.4f} ({finished_before}/{total_tasks})"
                                    )
                                    print(
                                        f"  potential_task_after: {potential_task_after:.4f} ({finished_after}/{total_tasks})"
                                    )
                                    print(f"  dense_weight: {dense_weight}")
                                    print(
                                        f"  task_completion_shaping_reward: {task_completion_shaping_reward:.4f}"
                                    )

                        # === 【方法E：LTL势能塑形】===
                        ltl_shaping_reward = 0.0
                        if effective_constraint_mode == "LTL_POTENTIAL" and ltl_monitor:
                            # phi_ltl_before已在agent_step之前保存（第1042行）
                            # 计算动作后的LTL势能（ltl_monitor已通过update_monitor更新）
                            if TrainParams.LTL_POTENTIAL_TYPE == "TOPOLOGY":
                                phi_ltl_after = ltl_monitor.compute_topology_potential()
                            else:
                                phi_ltl_after = ltl_monitor.compute_ltl_potential()

                            # SMDP势能塑形：F = exp(-β·τ)·φ(q') - φ(q)
                            if TrainParams.EXECUTION_MODE == "mdp":
                                ltl_shaping_reward = (
                                    TrainParams.LTL_POTENTIAL_WEIGHT
                                    * (
                                        TrainParams.GAMMA * phi_ltl_after
                                        - phi_ltl_before
                                    )
                                )
                            else:  # SMDP
                                discount_factor_ltl = np.exp(
                                    -TrainParams.BETA * tau_for_this_agent
                                )
                                ltl_shaping_reward = (
                                    TrainParams.LTL_POTENTIAL_WEIGHT
                                    * (
                                        discount_factor_ltl * phi_ltl_after
                                        - phi_ltl_before
                                    )
                                )

                            # DEBUG: 打印LTL势能变化（当势能显著变化时）
                            if (
                                abs(phi_ltl_after - phi_ltl_before) > 0.01
                                and decision_step < 100
                                and not training
                            ):
                                print(
                                    f"\n[DEBUG LTL POTENTIAL] Step {decision_step}, Agent {agent_id}"
                                )
                                print(f"  phi_ltl_before: {phi_ltl_before:.4f}")
                                print(f"  phi_ltl_after: {phi_ltl_after:.4f}")
                                print(
                                    f"  delta_phi: {phi_ltl_after - phi_ltl_before:.4f}"
                                )
                                print(f"  discount_factor: {discount_factor_ltl:.6f}")
                                print(f"  ltl_shaping_reward: {ltl_shaping_reward:.6f}")
                                # 打印LTL状态变化
                                stats = ltl_monitor.get_statistics()
                                print(
                                    f"  LTL stats: satisfied={stats['num_clauses'] - stats['safety_violated']}/{stats['num_clauses']}"
                                )

                        # 【总势能塑形奖励】= 进展势能 + 时间势能 + 任务完成势能 + LTL势能
                        shaping_reward = (
                            progress_shaping_reward
                            + time_shaping_reward
                            + task_completion_shaping_reward
                            + ltl_shaping_reward
                        )

                        # === 计算稠密奖励（只有第一个智能体承担）===
                        time_penalty = 0.0

                        # 【MDP模式：稠密时间惩罚】
                        if TrainParams.EXECUTION_MODE == "mdp" and hasattr(
                            TrainParams, "MDP_DENSE_REWARD_WEIGHT"
                        ):
                            # MDP方案：每步给予时间间隔的负值作为稠密奖励
                            # r_t = weight × (-Δt)
                            # 理论：累积后得到 weight × Σ(-Δt) = weight × (-T_total)
                            dense_weight = TrainParams.MDP_DENSE_REWARD_WEIGHT
                            if dense_weight != 0:
                                time_penalty = dense_weight * (-tau_for_this_agent)

                                # DEBUG: 打印MDP稠密奖励
                                if decision_step < 20 and not training:
                                    print(
                                        f"\n[DEBUG MDP DENSE REWARD] Step {decision_step}, Agent {agent_id} (batch #{agent_index_in_batch})"
                                    )
                                    print(
                                        f"  tau_for_this_agent: {tau_for_this_agent:.4f}"
                                    )
                                    print(f"  dense_weight: {dense_weight}")
                                    print(
                                        f"  time_penalty (MDP dense): {time_penalty:.4f}"
                                    )
                                    print(
                                        f"  accumulated_time: {self.env.current_time:.4f}"
                                    )

                        # 【旧版兼容：MDP时间奖励模式】
                        elif (
                            TrainParams.EXECUTION_MODE == "mdp"
                            and TrainParams.USE_MDP_TIME_REWARD
                        ):
                            pass
                        #         potential_based_time_penalty = time_penalty_weight * (
                        #                     t_current - discount_factor_for_time * t_next)
                        #
                        #         # 将惩罚分摊给当前行动的智能体
                        #         time_penalty = potential_based_time_penalty / num_acting_agents

                        # === 计算基础奖励（进展奖励 + 时间惩罚）===
                        r_base = shaping_reward + time_penalty

                        # 【方案A：Per-Agent Hazard Cost】
                        # Risk-Sensitive SMDP with Individual Accountability
                        # 理论框架：
                        #   - 目标函数：J_ρ(π) = ρ⁻¹ log E_π[exp(ρ·G)]
                        #   - 个体累积回报：G_i = Σ_k r̃_i,k
                        #   - 修正后的即时奖励：r̃_i,k = r_task_i,k - w·h_i(t_i)·τ_k
                        # 理论解释：
                        #   - Individual Accountability：每个agent只承担自己的hazard cost
                        #   - 所有agents使用相同的物理时间间隔τ（SMDP一致性）
                        #   - 这确保了每个transition的reward-discount匹配
                        # 实现优势：
                        #   1. 理论严格：符合SMDP定义（τ既影响discount也影响reward）
                        #   2. 个体责任：每个agent学习自己的风险成本
                        #   3. 易于防御：审稿人容易理解的标准做法
                        # ==================== 【路线1：已废弃per-step hazard cost】====================
                        # 旧版：在每个transition扣除hazard cost
                        #   r̃_i,k = r_task_i,k - w·h_i(t_i)·τ_k
                        # 问题：R >> ω*H 导致hazard项数值上消失（exp(-ρ*ω*H)≈1）
                        #
                        # 新版（路线1）：hazard不在reward中，改为episode级别的风险正则项
                        #   J(π) = E[T] + μ·log E[exp(α·C)]
                        #   其中 C = Σ_i (1 - exp(-H_i)) 在episode结束时计算
                        # ============================================================================
                        hazard_cost_this_step = 0.0
                        # [已注释] 路线1不再使用per-step hazard cost
                        # if (TrainParams.LTL_CONSTRAINT_TYPE == 'RISK_SENSITIVE_SMDP' and
                        #     EnvParams.VEHICLE_FAILURE_ENABLED):
                        #     ...（per-step hazard计算和调试打印已移除）

                        # 最终reward = r_base（路线1：hazard不在reward中）
                        r = r_base

                        # === 分项累加奖励成分 ===
                        episode_shaping_reward_sum += shaping_reward
                        episode_time_penalty_sum += time_penalty

                        # === 分项累加势能成分（用于详细诊断）===
                        episode_progress_potential_sum += progress_shaping_reward
                        episode_time_potential_sum += time_shaping_reward
                        episode_task_completion_potential_sum += (
                            task_completion_shaping_reward
                        )

                        # === 检查是否当前步之后任务完成 ===
                        done_after_step = self.env.check_finished()

                        # 注意：不在这里检测episode终止，因为while循环会提前退出
                        # terminal_penalty将在while循环结束后统一处理

                        episode_raw_return += r

                        # if episode_raw_return > 0:
                        #     print(f"DEBUG STEP: shaping_reward={shaping_reward:.4f}, time_penalty={time_penalty:.4f}, r_this_step={r:.4f}, cumulative_episode_return={episode_raw_return:.4f}")

                    # 2) 该步之后、同一个 agent 在下一状态下的观测
                    #    注意：和上面取当前状态时同一套 API + 同一套 padding
                    # next_task_info, next_total_agents, next_global_mask, _ = self.env.agent_observe(agent_id,
                    #                                                                                 max_waiting)

                    (
                        next_task_info,
                        next_total_agents,
                        next_global_mask,
                        next_ltl_info,
                        _,
                        _,
                        _,
                        _,
                    ) = self.env.agent_observe(agent_id, ltl_monitor, max_waiting)

                    next_task_info_t, next_total_agents_t, next_global_mask_t = (
                        self.convert_torch(
                            [next_task_info, next_total_agents, next_global_mask]
                        )
                    )

                    # 转换next_ltl_info到tensor（根据编码类型处理）
                    if TrainParams.LTL_ENCODING_TYPE == "C":
                        # 模式C：next_ltl_info是字典
                        next_ltl_info_t = {
                            "feasibility": torch.from_numpy(
                                next_ltl_info["feasibility"]
                            ).to(self.device),
                            "edge_index": torch.from_numpy(
                                next_ltl_info["edge_index"]
                            ).to(self.device),
                            "edge_attr": torch.from_numpy(
                                next_ltl_info["edge_attr"]
                            ).to(self.device),
                        }
                    else:
                        # 模式A/B：next_ltl_info是numpy数组
                        next_ltl_info_t = torch.from_numpy(next_ltl_info).to(
                            self.device
                        )

                    # if training:
                    next_task_info_t, next_total_agents_t, next_global_mask_t = (
                        self.obs_padding(
                            next_task_info_t, next_total_agents_t, next_global_mask_t
                        )
                    )

                    # if ltl_monitor:
                    #     ltl_tensor = torch.from_numpy(ltl_monitor.get_state_tensor()).to(self.device)
                    #     buffer_dict['ltl_info'].append(ltl_tensor)
                    # else:
                    #     dummy_tensor = torch.zeros((TrainParams.LTL_MAX_CLAUSES, 7), dtype=torch.float32).to(
                    #         self.device)
                    #     buffer_dict['ltl_info'].append(dummy_tensor)

                    next_index_t = (
                        torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    )

                    # ==================== 新增：决策点日志打印 ====================
                    if doable:
                        action_type = final_action_dict["type"]
                        action_str = "UNKNOWN"
                        if action_type == self.env.ACTION_MOVE:
                            dest = final_action_dict["destination"]
                            # 判断是仓库还是任务
                            if dest < self.env.depots_num:
                                action_str = f"MOVE to Depot {dest}"
                            else:
                                action_str = (
                                    f"MOVE to Task {dest - self.env.depots_num}"
                                )
                        elif action_type == self.env.ACTION_LOAD:
                            q_vec = final_action_dict["quantity_vec"]
                            c_type = np.argmax(q_vec)
                            quant = int(q_vec[c_type])
                            action_str = f"LOAD {quant} of Type {c_type}"
                        elif action_type == self.env.ACTION_UNLOAD:
                            q_vec = final_action_dict["quantity_vec"]
                            c_type = np.argmax(q_vec)
                            quant = int(q_vec[c_type])
                            if agent["current_task"] < 0:
                                action_str = f"DROPOFF {quant} of Type {c_type} at Depot {-agent['current_task'] - 1}"
                            else:
                                action_str = f"UNLOAD {quant} of Type {c_type} at Task {agent['current_task']}"
                        if print_task_detail:
                            print(
                                f"[T={self.env.current_time:.2f}s] Agent {agent_id:<2}: {action_str}"
                            )

                        # 检查是否有任务完成
                        if print_task_detail:
                            if f_t:
                                for finished_task_id in f_t:
                                    print(
                                        f"    └── ★★★ TASK {finished_task_id} COMPLETED! ★★★"
                                    )
                    # =============================================================

                    if not doable:
                        # ==================== ADVANCED DIAGNOSTIC BLOCK V3 ====================
                        print(
                            "\n"
                            + "#" * 25
                            + f" ACTION REJECTED (ADVANCED DIAGNOSIS) "
                            + "#" * 25
                        )
                        print(
                            f"  - Time: {self.env.current_time:.2f}, Agent ID: {agent_id}"
                        )
                        print(f"  - Rejected Action Dict: {final_action_dict}")
                        print("-" * 85)

                        # 1. Agent's full state at time of decision
                        print("  --- 1. Agent's State (at time of decision) ---")
                        agent_state_snapshot = self.env.agent_dic[agent_id]
                        print(
                            f"  - Location Node: {agent_state_snapshot.get('current_task')}, Inventory: {agent_state_snapshot.get('inventory')}"
                        )
                        print("-" * 85)

                        # 2. Ground truth of the target destination
                        print("  --- 2. Ground Truth of Target Destination ---")
                        if final_action_dict.get("type") == self.env.ACTION_MOVE:
                            dest_id = final_action_dict.get("destination")
                            if dest_id is not None:
                                if dest_id < self.env.depots_num:
                                    target_node_id = -dest_id - 1
                                    target_node_state = self.env.depot_dic.get(
                                        dest_id, "Invalid Depot ID"
                                    )
                                    print(
                                        f"  - Target Type: Depot, Action ID: {dest_id}, Node ID: {target_node_id}"
                                    )
                                    print(f"  - Target State: {target_node_state}")
                                else:
                                    target_node_id = dest_id - self.env.depots_num
                                    target_node_state = self.env.task_dic.get(
                                        target_node_id, "Invalid Task ID"
                                    )
                                    print(
                                        f"  - Target Type: Task, Action ID: {dest_id}, Node ID: {target_node_id}"
                                    )
                                    print(f"  - Target State: {target_node_state}")
                        else:
                            print("  - Action was not a MOVE action.")
                        print("-" * 85)

                        # 3. Agent's worldview used for generating masks
                        print("  --- 3. Agent's Worldview (Effective Statuses) ---")
                        try:
                            effective_statuses_snapshot = {
                                tid: s.astype(int).tolist()
                                for tid, s in self.env.last_effective_statuses.items()
                            }
                            print(f"  - {effective_statuses_snapshot}")
                        except Exception as e:
                            print(
                                f"  - Could not retrieve last effective statuses: {e}"
                            )
                        print("-" * 85)

                        # 4. The complete set of masks GIVEN to the agent
                        print(
                            "  --- 4. Masks Given to Agent (Original 'Instructions') ---"
                        )
                        print(f"  - Action Type Mask: {masks_dict.get('action_type')}")
                        if final_action_dict.get("type") == self.env.ACTION_MOVE:
                            dest_id = final_action_dict.get("destination")
                            if dest_id is not None:
                                original_dest_mask = masks_dict.get("destination")
                                if original_dest_mask is not None and dest_id < len(
                                    original_dest_mask
                                ):
                                    was_masked_orig = original_dest_mask[dest_id]
                                    print(
                                        f"  - Destination Mask (Original) at index [{dest_id}]: {was_masked_orig} (True=Forbidden)"
                                    )
                                else:
                                    print(
                                        f"  - Chosen index {dest_id} is out of bounds for Original Mask (len={len(original_dest_mask) if original_dest_mask is not None else 'N/A'})."
                                    )
                        print("-" * 85)

                        # 5. The data USED for sampling and the final decision
                        print("  --- 5. Data Used for Sampling & Final Choice ---")
                        print(f"  - Sampled Action Type: {chosen_action_type.item()}")
                        if final_action_dict.get("type") == self.env.ACTION_MOVE:
                            dest_id = final_action_dict.get("destination")
                            if dest_id is not None:
                                # This is the most critical comparison:
                                if dest_id < global_mask_t.shape[1]:
                                    was_masked_padded = global_mask_t[0, dest_id].item()
                                    print(
                                        f"  - Destination Mask (PADDED, USED FOR SAMPLING) at index [{dest_id}]: {was_masked_padded} (True=Forbidden)"
                                    )
                                else:
                                    print(
                                        f"  - Chosen index {dest_id} is out of bounds for Padded Mask (shape={global_mask_t.shape})."
                                    )

                            print(
                                f"  - Sampled Destination: {chosen_destination.item()}"
                            )
                            # Printing a slice of logits around the target can be very insightful
                            if dest_id is not None:
                                start = max(0, dest_id - 2)
                                end = min(
                                    policy_logits["destination"].shape[1], dest_id + 3
                                )
                                print(
                                    f"  - Logits slice [{start}:{end}]: {policy_logits['destination'][:, start:end].tolist()}"
                                )

                        print("#" * 85 + "\n")
                        # ======================================================================

                        # agent['next_decision'] = self.env.current_time
                        # agent['is_inactive'] = True
                        continue

                    if training and doable:
                        # 初始化 log_prob 和 entropy
                        log_prob = torch.tensor(0.0, device=self.device)
                        entropy = torch.tensor(0.0, device=self.device)

                        # 【方法1：分头独立PPO】初始化各头的log_prob
                        log_prob_type = torch.tensor(0.0, device=self.device)
                        log_prob_dest = torch.tensor(0.0, device=self.device)
                        log_prob_cargo = torch.tensor(0.0, device=self.device)

                        num_available_actions = np.sum(
                            masks_dict["action_type"] == False
                        )

                        # 累加动作类型的 log_prob 和 entropy
                        log_prob_type = action_type_dist.log_prob(
                            chosen_action_type
                        ).squeeze()
                        log_prob += log_prob_type
                        entropy += action_type_dist.entropy().squeeze()

                        # 如果是 MOVE 动作，累加目标地的 log_prob 和 entropy
                        if chosen_action_type.item() == self.env.ACTION_MOVE:
                            log_prob_dest = destination_dist.log_prob(
                                chosen_destination
                            ).squeeze()
                            log_prob += log_prob_dest
                            entropy += destination_dist.entropy().squeeze()

                        # 如果是 LOAD 动作，累加货物的 log_prob 和 entropy
                        elif chosen_action_type.item() == self.env.ACTION_LOAD:
                            if not torch.all(cargo_mask_t):
                                log_prob_cargo = cargo_dist.log_prob(
                                    chosen_cargo_type
                                ).squeeze()
                                log_prob += log_prob_cargo
                                entropy += cargo_dist.entropy().squeeze()

                        # 转换ltl_info到tensor并添加到buffer（根据编码类型处理）
                        if TrainParams.LTL_ENCODING_TYPE == "C":
                            # 模式C：ltl_info是字典
                            ltl_info_t_for_buffer = {
                                "feasibility": torch.from_numpy(
                                    ltl_info["feasibility"]
                                ).to(self.device),
                                "edge_index": torch.from_numpy(
                                    ltl_info["edge_index"]
                                ).to(self.device),
                                "edge_attr": torch.from_numpy(ltl_info["edge_attr"]).to(
                                    self.device
                                ),
                            }
                        else:
                            # 模式A/B：ltl_info是numpy数组
                            ltl_info_t_for_buffer = torch.from_numpy(ltl_info).to(
                                self.device
                            )

                        buffer_dict["ltl_info"].append(ltl_info_t_for_buffer)

                        # 【方案一】存储dependency_graph
                        buffer_dict["dependency_graph"].append(
                            dependency_graph_t
                            if dependency_graph_t is not None
                            else torch.zeros(1)
                        )

                        buffer_dict["task_info"].append(task_info_t.squeeze(0))
                        buffer_dict["agents_info"].append(total_agents_t.squeeze(0))
                        buffer_dict["mask"].append(global_mask_t.squeeze(0))
                        buffer_dict["value"].append(
                            reward_value
                        )  # 奖励期望值（用于reward GAE）
                        buffer_dict["index"].append(index_t.squeeze(0))
                        buffer_dict["old_log_prob"].append(log_prob)
                        # 【方法1：分头独立PPO】存储各头的old_log_prob
                        buffer_dict["old_log_prob_type"].append(log_prob_type)
                        buffer_dict["old_log_prob_dest"].append(log_prob_dest)
                        buffer_dict["old_log_prob_cargo"].append(log_prob_cargo)
                        buffer_dict["entropy"].append(entropy)
                        buffer_dict["actions"].append(action_to_store)
                        buffer_dict["rewards"].append(r)

                        # 【Episode-level CVAR】累加统一风险指数（无折扣）
                        # C_episode = Σ(β + h_k)·τ_k
                        if TrainParams.LTL_CONSTRAINT_TYPE == "CVAR_SMDP":
                            delta_t = tau_for_this_agent  # 时间间隔
                            current_hazard_rate = (
                                self.env.get_total_hazard_normalized()
                            )  # 全车队hazard rate
                            step_time_cost = TrainParams.BETA * delta_t
                            step_hazard_cost = current_hazard_rate * delta_t
                            step_unified_cost = step_time_cost + step_hazard_cost

                            # 直接累加到episode级风险指数（无折扣）
                            episode_unified_risk_index += step_unified_cost

                            # [DEBUG] 前几步打印
                            if decision_step < 3:
                                print(
                                    f"[Episode-CVAR] Step {decision_step}: δt={delta_t:.4f}, "
                                    f"h(t)={current_hazard_rate:.4f}, "
                                    f"step_cost={step_unified_cost:.6f} (β·δt={step_time_cost:.6f}, h·δt={step_hazard_cost:.6f}), "
                                    f"累计C={episode_unified_risk_index:.6f}"
                                )

                        # 【SOFT_POLICY】hazard cost 计算逻辑
                        # 【修复】方案A：将hazard cost平摊给同一事件中的所有active agents
                        # 理论依据：N个agents各记录H/N，累加后总量=H（数值正确）
                        # 符合Individual Reward Accounting + Centralized Training范式
                        fail_cost = 0.0
                        if TrainParams.LTL_CONSTRAINT_TYPE == "SOFT_POLICY":
                            # event_hazard_sum: 全车队总hazard rate
                            # tau: 物理时间间隔
                            # agents_in_this_batch: 同一事件中决策的所有agents
                            num_active_agents = len(agents_in_this_batch)
                            if num_active_agents > 0:
                                # 平摊：每个agent承担 (总hazard × 时间间隔) / agents数量
                                fail_cost = (event_hazard_sum * tau) / num_active_agents
                            else:
                                fail_cost = 0.0

                        # 根据模式选择写入哪种成本
                        # HARD模式下 costs 全为0，SOFT_POLICY下写入hazard cost
                        if TrainParams.LTL_CONSTRAINT_TYPE == "SOFT_POLICY":
                            buffer_dict["costs"].append(fail_cost)
                        else:
                            buffer_dict["costs"].append(cost_info["total_cost"])

                        buffer_dict["cargo_mask"].append(
                            torch.tensor(
                                masks_dict.get(
                                    "cargo_to_load",
                                    np.zeros(self.env.traits_dim, dtype=bool),
                                ),
                                dtype=torch.bool,
                            )
                        )
                        buffer_dict["action_type_mask"].append(
                            torch.tensor(
                                masks_dict.get("action_type", np.zeros(3, dtype=bool)),
                                dtype=torch.bool,
                            )
                        )
                        buffer_dict["next_task_info"].append(
                            next_task_info_t.squeeze(0)
                        )
                        buffer_dict["next_agents_info"].append(
                            next_total_agents_t.squeeze(0)
                        )
                        buffer_dict["next_mask"].append(next_global_mask_t.squeeze(0))
                        buffer_dict["next_index"].append(next_index_t.squeeze(0))

                        # 根据编码类型处理next_ltl_info的存储
                        if TrainParams.LTL_ENCODING_TYPE == "C":
                            # 模式C：字典格式，直接添加（无需squeeze）
                            buffer_dict["next_ltl_info"].append(next_ltl_info_t)
                        else:
                            # 模式A/B：tensor格式，需要squeeze掉batch维度
                            buffer_dict["next_ltl_info"].append(
                                next_ltl_info_t.squeeze(0)
                            )

                        buffer_dict["dones"].append(done_after_step)

                        # 【方案D：事件边界折扣】
                        # 只有第一个agent承担环境状态转移的时间间隔τ
                        # 后续agents的τ=0，表示相对于第一个agent的瞬时决策
                        # 这避免了同一事件内的重复SMDP折扣
                        if agent_index_in_batch == 0:
                            buffer_dict["taus"].append(tau)  # 第一个：真实时间推进
                        else:
                            buffer_dict["taus"].append(0.0)  # 后续：瞬时决策

                        # === 【修复】增加智能体处理计数器 ===
                        # 必须在tau存储之后递增，否则第一个agent也会被判定为"后续agents"
                        # 导致所有transitions都存储tau=0.0，使SMDP折扣失效
                        agent_index_in_batch += 1

                    if (
                        self.env.agent_dic[agent_id]["next_decision"]
                        == self.env.current_time
                    ):
                        agents_to_process_queue.append(agent_id)

            # 诊断：检查是否有agents的next_decision是过去的时间（说明存在逻辑错误）
            agents_stuck_in_past = []
            critical_issues = []

            for aid, a in self.env.agent_dic.items():
                next_dec = a.get("next_decision", np.inf)
                if next_dec < self.env.current_time and next_dec != np.inf:
                    agents_stuck_in_past.append(
                        {
                            "id": aid,
                            "next_decision": next_dec,
                            "current_task": a.get("current_task"),
                            "route": a.get("route", []),
                            "arrival_time": a.get("arrival_time", []),
                            "is_inactive": a.get("is_inactive", False),
                            "is_temp_sleeping": a.get("is_temp_sleeping", False),
                        }
                    )

                    # 根据agent状态决定如何处理
                    is_inactive = a.get("is_inactive", False)
                    is_sleeping = a.get("is_temp_sleeping", False)

                    if is_inactive:
                        # 永久失活的agent卡在历史时间 - 这应该被修复后不再出现
                        # 只记录到日志，不打印（太啰嗦）
                        pass
                    elif is_sleeping:
                        # 临时休眠的agent应该next_decision=inf，如果在历史时间则是bug
                        critical_issues.append(
                            f"Sleeping agent {aid}: next_decision={next_dec:.2f} (should be inf)"
                        )
                        print(
                            f"\n[WARNING] Sleeping agent {aid} has next_decision in the past! Fixing to inf."
                        )
                        a["next_decision"] = np.inf
                    else:
                        # 既不是inactive也不是sleeping，但卡在历史时间 -> 这是严重bug！
                        critical_issues.append(
                            f"Active agent {aid}: stuck at T={next_dec:.2f}"
                        )
                        print(
                            f"\n[CRITICAL BUG] Active agent {aid} stuck at T={next_dec:.2f}, forcing to current_time {self.env.current_time:.2f}"
                        )
                        print(
                            f"  Location: {a.get('current_task')}, will re-check its status"
                        )

                        # 强制设置决策时间并加入队列，让agent_observe重新评估其状态
                        a["next_decision"] = self.env.current_time
                        if aid not in agents_to_process_queue:
                            agents_to_process_queue.append(aid)
                            print(f"  -> Added to queue for re-evaluation")

            # 只在有关键问题时打印摘要
            if critical_issues:
                print(
                    f"\n[DIAGNOSIS] [T={self.env.current_time:.2f}] Critical issues detected:"
                )
                for issue in critical_issues:
                    print(f"  - {issue}")

            # 【修复】更新 time_of_last_event，用于下一个事件的 delta_t 计算
            # 必须在处理完当前事件的所有agents后才更新
            time_of_last_event = self.env.current_time

            self.env.finished = self.env.check_finished()
            decision_step += 1

        # ==================== 判断Episode终止类型 ====================
        # 检查episode是否真正结束（而非被PPO截断）
        # 注意：故障不再终止episode，只是智能体失活
        episode_truly_done = (
            self.env.finished  # 所有任务完成
            or self.env.current_time >= EnvParams.MAX_TIME  # 时间上限
            or decision_step >= max_decision_steps  # 决策步数上限（非PPO截断）
            or not (
                training
                and TrainParams.ALGORITHM == "PPO"
                and decision_step >= TrainParams.PPO_ROLLOUT_LENGTH
            )
        )

        # 获取finished_tasks信息用于计算success_rate（需要在sparse奖励之前）
        _, finished_tasks = self.env.get_episode_reward(self.max_time)
        if finished_tasks:
            success_rate = np.mean(finished_tasks)
        else:
            # 如果环境中没有任务，也算作100%成功
            success_rate = 1.0

        # ==================== 稀疏终止奖励/惩罚 ====================
        if episode_truly_done:
            total_terminal_reward = 0.0

            # 【MDP模式：稀疏终止奖励】
            if TrainParams.EXECUTION_MODE == "mdp":
                # 获取权重系数
                sparse_weight = getattr(TrainParams, "MDP_SPARSE_REWARD_WEIGHT", 1.0)

                if sparse_weight != 0:
                    # MDP方案：终止奖励 = weight × (-makespan)（成功）或 weight × (-T_max)（失败）
                    # 理论：G = weight × Σ(-Δt) + R_terminal
                    #      对于成功：G = weight × (-T_makespan) + weight × (-T_makespan) = weight × (-2×T_makespan)
                    #      对于失败：G = weight × (-T_fail) + weight × (-T_max)

                    if self.env.finished and success_rate >= 1.0:
                        # 成功：给予当前时间的负值（加权）
                        total_terminal_reward = sparse_weight * (-self.env.current_time)
                        reward_reason = f"Success: terminal reward = {sparse_weight} × (-T_makespan) = {total_terminal_reward:.2f}"
                    else:
                        # 失败：给予最大时间的负值（加权）
                        # 注意：MAX_TIME默认为500，所以sparse_weight=1时，失败惩罚为-500
                        total_terminal_reward = sparse_weight * (-EnvParams.MAX_TIME)
                        if self.env.current_time >= EnvParams.MAX_TIME:
                            reward_reason = f"Timeout: terminal reward = {sparse_weight} × (-T_max) = {total_terminal_reward:.2f}"
                        elif decision_step >= max_decision_steps:
                            reward_reason = f"Max steps exceeded: terminal reward = {sparse_weight} × (-T_max) = {total_terminal_reward:.2f}"
                        else:
                            reward_reason = f"Partial completion: terminal reward = {sparse_weight} × (-T_max) = {total_terminal_reward:.2f}"

            # 【SMDP模式：稀疏终止奖励】
            elif TrainParams.EXECUTION_MODE == "smdp":
                # 获取权重系数
                sparse_weight = getattr(TrainParams, "SMDP_SPARSE_REWARD_WEIGHT", 1.0)

                if sparse_weight != 0:
                    # 【修改】SMDP与MDP统一：使用完工时间的负数
                    # 理论优势：SMDP的时间依赖折扣 exp(-β·T) 自动施加时间偏好
                    #          无需固定的成功奖励，直接优化makespan
                    #
                    # 终止奖励分配：
                    # - 成功（所有任务完成）：weight × (-current_time)
                    # - 失败（超时/超步数/部分完成）：weight × (-MAX_TIME)

                    # 检查是否成功完成所有任务
                    if self.env.finished and success_rate >= 1.0:
                        # 完全成功：给予完工时间的负数（加权）
                        # SMDP折扣：最终return = exp(-β·T) × (-T)，T越小，return越高
                        total_terminal_reward = sparse_weight * (-self.env.current_time)
                        reward_reason = f"Success: terminal reward = {sparse_weight} × (-T_makespan) = {total_terminal_reward:.2f}"

                        # 【方法E】检查LTL违规，添加惩罚（防止奖励劫持）
                        if effective_constraint_mode == "LTL_POTENTIAL" and ltl_monitor:
                            num_violations = 0
                            for clause in ltl_monitor.clauses:
                                if (
                                    clause.type == LTL_SAFETY
                                    and clause.state == FSA_SAFETY_VIOLATED
                                ):
                                    num_violations += 1
                                elif (
                                    clause.type == LTL_SEQUENTIAL
                                    and clause.state == FSA_SEQ_VIOLATED
                                ):
                                    num_violations += 1

                            if num_violations > 0:
                                # 违规惩罚（防止"快速违规"比"慢速满足"获得更高奖励）
                                violation_penalty = (
                                    num_violations * TrainParams.LTL_VIOLATION_PENALTY
                                )
                                total_terminal_reward -= violation_penalty
                                reward_reason += f" | LTL violations: {num_violations}, penalty: -{violation_penalty:.1f}"
                            else:
                                reward_reason += f" | LTL: all satisfied ✓"
                    else:
                        # 失败（超时/超步数/部分完成）：给予最大时间的负值（加权）
                        # 注意：MAX_TIME默认为500，所以sparse_weight=1时，失败惩罚为-500
                        total_terminal_reward = sparse_weight * (-EnvParams.MAX_TIME)
                        if self.env.current_time >= EnvParams.MAX_TIME:
                            reward_reason = f"Timeout: terminal reward = {sparse_weight} × (-T_max) = {total_terminal_reward:.2f}"
                        elif decision_step >= max_decision_steps:
                            reward_reason = f"Max steps exceeded: terminal reward = {sparse_weight} × (-T_max) = {total_terminal_reward:.2f}"
                        else:
                            reward_reason = f"Partial completion: terminal reward = {sparse_weight} × (-T_max) = {total_terminal_reward:.2f}"

            # 应用终止奖励
            if total_terminal_reward != 0:
                # 将终止奖励/惩罚分配给buffer中的最后一个transition
                if training and buffer_dict["rewards"]:
                    # 只给最后一个transition添加
                    buffer_dict["rewards"][-1] += total_terminal_reward

                    # 累加到统计变量
                    episode_terminal_penalty_sum += total_terminal_reward
                    episode_raw_return += total_terminal_reward

                    if DEBUG:
                        print(f"\n[TERMINAL REWARD] {reward_reason}")
                        print(
                            f"  - Terminal reward/penalty: {total_terminal_reward:.2f}"
                        )
                        print(f"  - Applied to last transition only")
                else:
                    # 评估模式：只累加到统计变量
                    episode_terminal_penalty_sum += total_terminal_reward
                    episode_raw_return += total_terminal_reward
        # ================================================================

        # 检查是否因PPO rollout截断
        ppo_truncated = (
            TrainParams.ALGORITHM == "PPO"
            and decision_step >= TrainParams.PPO_ROLLOUT_LENGTH
            and not episode_truly_done
        )

        # 在sparse模式下，terminal_reward设为0（已通过total_terminal_reward处理）
        # 在dense模式下，可以保留旧的逻辑（如果需要）
        if TrainParams.REWARD_TIME_PENALTY_MODE == "sparse":
            terminal_reward = 0.0  # sparse模式下已通过固定的成功奖励/失败惩罚处理
        else:
            # Dense模式或其他模式的逻辑（保留向后兼容）
            terminal_reward = 0.0  # 暂不使用

            # 如果成功率低于100%，则打印详细的失败诊断信息（但不包括PPO截断的情况）
        if (
            success_rate < 1.0
            and self.env.current_time < EnvParams.MAX_TIME
            and decision_step < max_decision_steps
            and not ppo_truncated
        ):
            print("\n" + "#" * 30 + " EPISODE FAILURE ANALYSIS " + "#" * 30)
            print(
                f"# Worker: {self.metaAgentID}, Final Time: {self.env.current_time:.2f}, Success Rate: {success_rate:.0%}"
            )
            print(f"# Decision Steps: {decision_step}, Max Time: {EnvParams.MAX_TIME}")

            # 打印终止原因
            print(f"# Termination Reason: ", end="")
            if self.env.finished:
                print("All tasks completed (but some agents failed)")
            elif self.env.current_time >= EnvParams.MAX_TIME:
                print("Time limit exceeded")
            elif decision_step >= max_decision_steps:
                print(f"Decision step limit ({max_decision_steps}) exceeded")
            elif decision_step >= TrainParams.PPO_ROLLOUT_LENGTH:
                print(
                    f"PPO rollout truncated (reached {TrainParams.PPO_ROLLOUT_LENGTH} steps) - Episode still has active agents"
                )
            else:
                print("No future events (all agents inactive/sleeping)")

            # 1. 打印本次Episode的LTL约束
            print("\n--- [1] LTL CONSTRAINTS ACTIVE THIS EPISODE ---")
            if ltl_monitor and ltl_monitor.clauses:
                for i, clause in enumerate(ltl_monitor.clauses):
                    print(f"  - Clause {i}: {clause}")
            else:
                print("  - No LTL constraints were active.")

            # 2. 打印最终仍处于“暂时休眠”状态的智能体
            print("\n--- [2] AGENTS ENDING IN 'TEMP_SLEEPING' STATE ---")
            sleeping_agents_found = False
            for agent_id, agent_data in self.env.agent_dic.items():
                if agent_data.get("is_temp_sleeping", False):
                    sleeping_agents_found = True
                    blocking_clauses = agent_data.get("blocking_clauses", [])
                    print(
                        f"  - Agent {agent_id}: Blocked by clauses {blocking_clauses}"
                    )
            if not sleeping_agents_found:
                print("  - None.")

            # 3. 打印所有未完成的任务及其剩余需求
            print("\n--- [3] UNFINISHED TASKS ---")
            unfinished_tasks_found = False
            for task_id, task_data in self.env.task_dic.items():
                if not task_data.get("finished", False):
                    unfinished_tasks_found = True
                    status = task_data.get("status")
                    requirements = task_data.get("requirements")
                    print(
                        f"  - Task {task_id}: Remaining={status.tolist()}, Original={requirements.tolist()}"
                    )
            if not unfinished_tasks_found:
                # 这种情况理论上不应发生，但作为代码健壮性检查
                print(
                    "  - None (Inconsistency detected: Success rate is < 1.0 but no unfinished tasks found)."
                )

            # 4. 【新增】打印所有智能体的能力和最终状态
            print("\n--- [4] AGENT CAPABILITIES & FINAL STATES ---")
            for agent_id, agent_data in self.env.agent_dic.items():
                species = agent_data.get("species")
                # 安全地获取能力，如果不存在则默认为0向量
                capacity = agent_data.get("capacity", np.zeros(self.env.traits_dim))
                location = agent_data.get("current_task", "N/A")
                inventory = agent_data.get("inventory", {})
                is_sleeping = agent_data.get("is_temp_sleeping", False)
                is_inactive = agent_data.get("is_inactive", False)
                next_decision = agent_data.get("next_decision", "N/A")
                arrival_times = agent_data.get("arrival_time", [])
                route = agent_data.get("route", [])

                # 构造一个更具描述性的状态字符串
                state_str = "Active/In-Transit"
                if is_inactive:
                    state_str = "Inactive"
                elif is_sleeping:
                    state_str = "TempSleeping"

                # 判断是否真的在途中
                if not is_inactive and not is_sleeping:
                    if next_decision == np.inf and arrival_times:
                        last_arrival = arrival_times[-1]
                        if last_arrival > self.env.current_time:
                            state_str = f"In-Transit (arrives@{last_arrival:.2f})"
                        else:
                            state_str = (
                                f"STUCK! (should have arrived@{last_arrival:.2f})"
                            )
                    elif next_decision == self.env.current_time:
                        state_str = "Ready (should have been processed)"
                    elif next_decision < np.inf:
                        state_str = f"Waiting (decides@{next_decision:.2f})"

                print(f"  - Agent {agent_id} (Species {species}):")
                print(f"      State        : {state_str}")
                print(f"      Location     : Node {location}")
                print(f"      Inventory    : {inventory}")
                print(f"      Capacity     : {capacity.astype(int).tolist()}")

                # Format next_decision
                next_dec_str = (
                    next_decision
                    if isinstance(next_decision, str)
                    else f"{next_decision:.2f}"
                )
                print(f"      next_decision: {next_dec_str}")

                # Format arrival time
                last_arrival_str = (
                    f"{arrival_times[-1]:.2f}" if arrival_times else "N/A"
                )
                print(
                    f"      Route length : {len(route)}, Last arrival: {last_arrival_str}"
                )

            print("#" * 88 + "\n")

        # ==================== 处理Terminal Reward ====================
        # 只有episode真正结束时才添加terminal reward
        # 如果被PPO截断，则用value bootstrap，不添加terminal reward
        if training and buffer_dict["rewards"]:
            if episode_truly_done:
                # 真正结束：将terminal reward加到最后一步
                buffer_dict["rewards"][-1] += terminal_reward
                # if TrainParams.ALGORITHM == 'PPO':
                #     print(f"[Worker {self.metaAgentID}] Episode truly ended, adding terminal_reward={terminal_reward:.2f}")
            elif ppo_truncated:
                # PPO截断：terminal reward会被忽略，用value bootstrap代替
                pass
                # if TrainParams.ALGORITHM == 'PPO':
                #     print(f"[Worker {self.metaAgentID}] PPO truncated, will use value bootstrap (terminal_reward={terminal_reward:.2f} ignored)")

        if print_episode_result:
            # ==================== 新增：结束状态日志打印 ====================
            success_rate = (
                np.sum(finished_tasks) / len(finished_tasks)
                if len(finished_tasks) > 0
                else 1.0
            )
            print("\n" + "-" * 80)
            print(f"--- EPISODE END ---")
            print(f"  - Final Time: {self.env.current_time:.2f}s")
            print(f"  - Task Completion Rate: {success_rate:.0%}")
            print("=" * 80 + "\n")
            # =============================================================

        # ==================== 使用GAE计算Advantages和Returns ====================
        final_buffer = {
            "task_info": [],
            "agents_info": [],
            "mask": [],
            "index": [],
            "old_log_prob": [],
            "reward": [],
            "advantage": [],
            "entropy": [],
            "actions": [],
        }

        # 【紧急诊断】打印 buffer_dict['value'] 状态
        print(
            f"[Worker {self.metaAgentID}] Episode end: len(buffer_dict['value'])={len(buffer_dict['value'])}, len(buffer_dict['rewards'])={len(buffer_dict['rewards'])}"
        )

        if training and buffer_dict["value"]:
            # 1. 提取values并转换为tensor
            values = torch.cat(buffer_dict["value"]).squeeze()

            # 2. 获取每一步的即时奖励和done标志
            step_rewards = buffer_dict["rewards"]  # list of floats
            dones = buffer_dict["dones"]  # list of bools

            # 3. 确定bootstrap value（用于最后一步的next value）
            if episode_truly_done:
                # Episode真正结束：next_value = 0（所有future rewards都已经在rewards中）
                last_value = 0.0
            else:
                # Episode被截断：需要估计剩余return
                # 使用最后一步的value作为bootstrap
                last_value = values[-1].item()
                # print(f"[Worker {self.metaAgentID}] Using last_value={last_value:.2f} for bootstrap")

            # 4. 使用GAE计算returns和advantages
            # 根据EXECUTION_MODE选择MDP-GAE或SMDP-GAE
            if TrainParams.EXECUTION_MODE == "smdp":
                # SMDP模式：使用时间依赖折扣
                step_taus = buffer_dict["taus"]  # list of floats
                returns, advantages = self.compute_gae(
                    rewards=step_rewards,
                    values=values,
                    dones=dones,
                    last_value=last_value,
                    gamma=TrainParams.GAMMA,  # 仅用于fallback，SMDP模式会被忽略
                    lambda_=TrainParams.GAE_LAMBDA,
                    taus=step_taus,
                    execution_mode="smdp",
                )
            else:
                # MDP模式：使用固定折扣
                returns, advantages = self.compute_gae(
                    rewards=step_rewards,
                    values=values,
                    dones=dones,
                    last_value=last_value,
                    gamma=TrainParams.GAMMA,
                    lambda_=TrainParams.GAE_LAMBDA,
                    execution_mode="mdp",
                )

            # 【Episode-level CVAR】传递episode级统一风险指数（无折扣，直接累加）
            # C_episode = Σ(β + h_k)·τ_k
            # 这是整个episode的总风险成本
            if TrainParams.LTL_CONSTRAINT_TYPE == "CVAR_SMDP":
                print(
                    f"[Episode-CVAR] Episode统一风险指数 C_episode = {episode_unified_risk_index:.6f}"
                )

            # 5. 填充final_buffer
            final_buffer["task_info"] = buffer_dict["task_info"]
            final_buffer["agents_info"] = buffer_dict["agents_info"]
            final_buffer["mask"] = buffer_dict["mask"]
            final_buffer["index"] = buffer_dict["index"]
            final_buffer["old_log_prob"] = buffer_dict["old_log_prob"]
            # 【方法1：分头独立PPO】传递各头的old_log_prob
            final_buffer["old_log_prob_type"] = buffer_dict["old_log_prob_type"]
            final_buffer["old_log_prob_dest"] = buffer_dict["old_log_prob_dest"]
            final_buffer["old_log_prob_cargo"] = buffer_dict["old_log_prob_cargo"]
            final_buffer["entropy"] = buffer_dict["entropy"]
            final_buffer["actions"] = buffer_dict["actions"]

            # 【Episode-level CVAR】传递episode级统一风险指数
            if TrainParams.LTL_CONSTRAINT_TYPE == "CVAR_SMDP":
                final_buffer["episode_unified_risk_index"] = episode_unified_risk_index

            # 【Relative Excess Hazard方案】计算并传递相对超额hazard（负载不均衡度）
            # 【修改】NEU模式也需要记录comprehensive metrics用于对比实验
            if EnvParams.VEHICLE_FAILURE_ENABLED:
                # ========================================================================
                # 理论基础：Weibull累积hazard的凸性（Jensen不等式）
                #
                # 给定总工作量W，最优负载分配（均衡）的总hazard为：
                #   H_optimal = n·(W/n/λ)^k
                #
                # 任何不均衡的分配都会导致总hazard增加：
                #   H_actual = Σ(t_i/λ)^k ≥ H_optimal
                #
                # Relative Excess Hazard定义：
                #   C_rel = (H_actual / H_optimal) - 1 ≥ 0
                #
                # 性质：
                #   - C_rel=0 当且仅当负载完全均衡（所有t_i相等）
                #   - C_rel完全scale-invariant：与任务难度W无关
                #   - 百分比形式：C_rel=0.5 表示"实际hazard比最优多50%"
                #   - 直接衡量负载不均衡的严重程度
                # ========================================================================

                lambda_param = EnvParams.WEIBULL_LAMBDA
                k_param = EnvParams.WEIBULL_K

                # 收集所有agent的工作时间
                work_times = []
                for agent_id, agent in self.env.agent_dic.items():
                    cumulative_time = agent.get("cumulative_time", 0.0)
                    work_times.append(cumulative_time)

                n = len(work_times)
                W = sum(work_times)  # 总工作量

                # 计算实际总hazard
                H_actual = sum([(t / lambda_param) ** k_param for t in work_times])

                # 计算理论最优总hazard（完美负载均衡）
                if n > 0 and W > 0:
                    H_optimal = n * (W / n / lambda_param) ** k_param
                else:
                    H_optimal = 1e-8  # 避免除零

                # ==================== 核心风险指标 ====================
                # Relative Excess Hazard（相对超额百分比）
                C_rel = (H_actual / H_optimal) - 1.0
                # Absolute Excess Hazard（绝对超额hazard）
                C_abs = H_actual - H_optimal

                # ==================== 负载均衡度量指标 ====================
                if n > 0 and W > 0:
                    mean_work = W / n
                    # 标准差（经典负载不均衡度量）
                    work_std = np.std(work_times) if n > 1 else 0.0
                    # 变异系数（归一化标准差，scale-invariant）
                    work_cv = work_std / mean_work if mean_work > 1e-8 else 0.0
                    # 最大/平均比率（最忙agent的相对负载）
                    max_mean_ratio = (
                        max(work_times) / mean_work if mean_work > 1e-8 else 1.0
                    )
                    # Gini系数（经济学不平等度量，范围[0,1]）
                    # Gini = (Σ|t_i - t_j|) / (2n^2·mean)
                    sorted_times = sorted(work_times)
                    gini_sum = sum(
                        abs(t_i - t_j) for t_i in work_times for t_j in work_times
                    )
                    gini_coeff = (
                        gini_sum / (2.0 * n * n * mean_work)
                        if mean_work > 1e-8
                        else 0.0
                    )
                else:
                    work_std = 0.0
                    work_cv = 0.0
                    max_mean_ratio = 1.0
                    gini_coeff = 0.0

                # ==================== 存储到buffer（传递给driver）====================
                # 【核心优化目标】
                final_buffer["episode_expected_failures_C"] = (
                    C_rel  # 相对超额hazard（优化目标）
                )
                final_buffer["episode_makespan_T"] = (
                    self.env.current_time
                )  # makespan（优化目标）

                # 【理论验证指标】用于验证H_actual与T的耦合关系
                final_buffer["episode_total_work_W"] = W  # 总工作量（验证scale相关性）
                final_buffer["episode_H_actual"] = H_actual  # 实际总hazard（验证T耦合）
                final_buffer["episode_H_optimal"] = (
                    H_optimal  # 理论最优hazard（Jensen基线）
                )
                final_buffer["episode_C_abs"] = C_abs  # 绝对超额hazard（对比C_rel）

                # 【负载均衡指标】多种度量方式的综合对比
                final_buffer["episode_work_std"] = work_std  # 标准差
                final_buffer["episode_work_cv"] = work_cv  # 变异系数
                final_buffer["episode_max_mean_ratio"] = max_mean_ratio  # 最大/平均比率
                final_buffer["episode_gini_coeff"] = gini_coeff  # Gini系数

                # 【原始分布数据】用于后处理分析（直方图、散点图等）
                final_buffer["episode_work_times"] = (
                    work_times  # 所有agent的工作时间数组
                )
                final_buffer["episode_num_agents"] = n  # agent数量

                # [DEBUG] 打印统计信息
                if not training:
                    print(f"\n{'=' * 80}")
                    print(f"[Episode Metrics] Comprehensive Statistics at Termination")
                    print(f"{'=' * 80}")

                    # 核心优化目标
                    print(f"\n【核心优化目标】")
                    print(f"  Makespan (T):           {self.env.current_time:.2f}s")
                    print(
                        f"  Relative Excess Hazard (C_rel): {C_rel:.4f} ({C_rel * 100:.1f}%)"
                    )
                    print(f"  → Objective: J = E[T] + μ·E[C_rel]")

                    # 理论验证指标
                    print(f"\n【理论验证指标】")
                    print(f"  总工作量 (W):           {W:.2f}s")
                    print(f"  实际总Hazard (H_actual):  {H_actual:.4f}")
                    print(f"  理论最优Hazard (H_optimal): {H_optimal:.4f}")
                    print(f"  绝对超额Hazard (C_abs):   {C_abs:.4f}")
                    print(
                        f"  → Jensen不等式: H_actual ≥ H_optimal ✓ 验证: {H_actual >= H_optimal - 1e-6}"
                    )

                    # 负载均衡指标
                    print(f"\n【负载均衡指标】")
                    print(f"  车辆数 (n):             {n}")
                    print(f"  平均工作时间:           {W / n:.2f}s")
                    print(f"  工作时间std:           {work_std:.2f}s")
                    print(f"  变异系数 (CV):          {work_cv:.4f}")
                    print(f"  最大/平均比率:          {max_mean_ratio:.3f}x")
                    print(f"  Gini系数:              {gini_coeff:.4f}")

                    # 原始分布数据
                    print(f"\n【工作时间分布】")
                    print(f"  Raw times: {[f'{t:.1f}' for t in work_times]}")
                    print(f"  Min/Max: {min(work_times):.1f}s / {max(work_times):.1f}s")
                    print(f"  Range: {max(work_times) - min(work_times):.1f}s")

                    print(f"{'=' * 80}\n")

            # 【修复】传递成本数据（CMDP软约束需要，非CVAR模式）
            if (
                "costs" in buffer_dict
                and TrainParams.LTL_CONSTRAINT_TYPE != "CVAR_SMDP"
            ):
                final_buffer["costs"] = buffer_dict["costs"]

            # 【关键修复】传递dones数据（成本计算的蒙特卡洛回报需要）
            if "dones" in buffer_dict:
                final_buffer["dones"] = buffer_dict["dones"]

            # 【修复】传递其他可能需要的字段
            if "taus" in buffer_dict:
                final_buffer["taus"] = buffer_dict["taus"]
            if "ltl_info" in buffer_dict:
                final_buffer["ltl_info"] = buffer_dict["ltl_info"]
            if "next_ltl_info" in buffer_dict:
                final_buffer["next_ltl_info"] = buffer_dict["next_ltl_info"]
            # 【方案一】传递dependency_graph
            if "dependency_graph" in buffer_dict:
                final_buffer["dependency_graph"] = buffer_dict["dependency_graph"]
            if "cargo_mask" in buffer_dict:
                final_buffer["cargo_mask"] = buffer_dict["cargo_mask"]
            if "action_type_mask" in buffer_dict:
                final_buffer["action_type_mask"] = buffer_dict["action_type_mask"]

            # 将 reward 和 advantage 拆分成 list of tensors
            # 保持原始split的结果，让driver端处理形状
            final_buffer["reward"] = list(torch.split(returns, 1))
            final_buffer["advantage"] = list(torch.split(advantages, 1))

            # 【GAE DEBUG】确认final_buffer中reward/advantage的长度
            if (
                TrainParams.LTL_CONSTRAINT_TYPE == "CVAR_SMDP"
                and training
                and decision_step < 2
            ):
                print(
                    f"[GAE DEBUG Worker {self.metaAgentID}] final_buffer reward_len={len(final_buffer['reward'])}, adv_len={len(final_buffer['advantage'])}"
                )

            # 【形状诊断】打印worker端返回的形状
            if decision_step < 5:  # 只在前几步打印
                print(f"\n[SHAPE_DIAG Worker {self.metaAgentID}]")
                print(f"  returns shape before split: {returns.shape}")
                print(f"  advantages shape before split: {advantages.shape}")
                print(
                    f"  final_buffer['reward'][0] shape: {final_buffer['reward'][0].shape}"
                )
                print(
                    f"  final_buffer['advantage'][0] shape: {final_buffer['advantage'][0].shape}"
                )
                print(f"  len(final_buffer['reward']): {len(final_buffer['reward'])}")

            # 打印统计信息（用于调试）（已注释）
            # if TrainParams.ALGORITHM == 'PPO':
            #     print(f"[Worker {self.metaAgentID}] GAE Stats: avg_advantage={advantages.mean().item():.4f}, "
            #           f"avg_return={returns.mean().item():.4f}, last_return={returns[-1].item():.4f}")
        else:
            # 如果不是training或没有value数据，直接使用buffer_dict
            final_buffer = buffer_dict

        for key, value_list in episode_metrics.items():
            if value_list:  # 列表非空
                perf_metrics[key] = [np.nanmean(value_list)]
            else:  # 如果列表为空（例如从未选择MOVE），则记录为0
                perf_metrics[key] = [0.0]

        # 计算任务完成率（原有的success_rate，保留）
        if len(finished_tasks) > 0:
            num_finished = np.sum(finished_tasks)
            num_total = len(finished_tasks)
            success_rate_value = num_finished / num_total
            perf_metrics["success_rate"] = [success_rate_value]  # 任务完成率（任务数）
        else:
            perf_metrics["success_rate"] = [1.0]

        # 【新增】episode级别的成功率（是否完全成功）
        episode_success = (
            1.0 if (self.env.finished and success_rate_value >= 1.0) else 0.0
        )
        perf_metrics["episode_success"] = [episode_success]  # episode成功率（0或1）

        # 添加episode return指标（总和及分项）
        perf_metrics["episode_return"] = [episode_raw_return]
        perf_metrics["episode_shaping_reward"] = [episode_shaping_reward_sum]
        perf_metrics["episode_time_penalty"] = [episode_time_penalty_sum]
        perf_metrics["episode_terminal_penalty"] = [episode_terminal_penalty_sum]

        # 【故障统计】计算并记录故障相关指标
        # 【Pure LTL版本修改】仅在故障建模开启时记录故障统计指标
        if EnvParams.VEHICLE_FAILURE_ENABLED:
            failure_stats = self._compute_failure_statistics()
            perf_metrics["num_agents"] = [failure_stats["num_agents"]]
            perf_metrics["failed_agents_count"] = [failure_stats["failed_count"]]
            perf_metrics["actual_failure_rate"] = [
                failure_stats["actual_failure_rate"]
            ]  # 实际故障比例
            perf_metrics["avg_failure_prob"] = [
                failure_stats["avg_failure_prob"]
            ]  # 平均理论故障概率
            perf_metrics["expected_failures_C"] = [
                failure_stats["expected_failures_C"]
            ]  # 期望故障车辆数 C(τ) (IJCAI Eq. 41)
        else:
            # Pure LTL模式：只记录agent数量，其他故障指标不记录
            perf_metrics["num_agents"] = [len(self.env.agent_dic)]

        # ==================== Comprehensive Metrics（评估阶段）====================
        # 【修改】NEU模式也需要记录comprehensive metrics用于对比实验
        # 这些指标用于评估时的详细分析和理论验证
        if EnvParams.VEHICLE_FAILURE_ENABLED and not training:
            # 从final_buffer中提取comprehensive metrics（如果存在）
            # 训练阶段这些metrics已经在final_buffer中，评估阶段需要显式添加到perf_metrics
            if "episode_expected_failures_C" in final_buffer:
                perf_metrics["C_rel"] = [final_buffer["episode_expected_failures_C"]]
            if "episode_makespan_T" in final_buffer:
                perf_metrics["makespan_T"] = [final_buffer["episode_makespan_T"]]
            if "episode_total_work_W" in final_buffer:
                perf_metrics["total_work_W"] = [final_buffer["episode_total_work_W"]]
            if "episode_H_actual" in final_buffer:
                perf_metrics["H_actual"] = [final_buffer["episode_H_actual"]]
            if "episode_H_optimal" in final_buffer:
                perf_metrics["H_optimal"] = [final_buffer["episode_H_optimal"]]
            if "episode_C_abs" in final_buffer:
                perf_metrics["C_abs"] = [final_buffer["episode_C_abs"]]
            if "episode_work_std" in final_buffer:
                perf_metrics["work_std"] = [final_buffer["episode_work_std"]]
            if "episode_work_cv" in final_buffer:
                perf_metrics["work_cv"] = [final_buffer["episode_work_cv"]]
            if "episode_max_mean_ratio" in final_buffer:
                perf_metrics["max_mean_ratio"] = [
                    final_buffer["episode_max_mean_ratio"]
                ]
            if "episode_gini_coeff" in final_buffer:
                perf_metrics["gini_coeff"] = [final_buffer["episode_gini_coeff"]]
            # 原始分布数据不添加到perf_metrics（太大，仅用于训练时的后处理分析）
        # =========================================================================

        # 【新增】添加决策步数统计
        perf_metrics["decision_steps"] = [decision_step]

        # 【新增】添加动作执行统计
        perf_metrics["action_move_count"] = [action_counters["MOVE"]]
        perf_metrics["action_load_count"] = [action_counters["LOAD"]]
        perf_metrics["action_unload_count"] = [action_counters["UNLOAD"]]
        perf_metrics["action_rejected_count"] = [action_counters["REJECTED"]]
        total_actions = sum(action_counters.values())
        perf_metrics["total_actions"] = [total_actions]

        # 验证：总和应该等于各分项之和
        expected_return = (
            episode_shaping_reward_sum
            + episode_time_penalty_sum
            + episode_terminal_penalty_sum
        )
        if abs(episode_raw_return - expected_return) > 1e-6:
            print(
                f"[WARNING] Episode return mismatch: {episode_raw_return:.4f} != {expected_return:.4f}"
            )
            print(
                f"  Shaping: {episode_shaping_reward_sum:.4f}, Time: {episode_time_penalty_sum:.4f}, Terminal: {episode_terminal_penalty_sum:.4f}"
            )

        # 打印终止原因和奖励/惩罚
        if not training and episode_terminal_penalty_sum != 0:
            terminal_type = "Reward" if episode_terminal_penalty_sum > 0 else "Penalty"
            print(
                f"\n[Episode End] Terminal {terminal_type} applied: {episode_terminal_penalty_sum:.2f}"
            )
            print(
                f"  Completion status: {'All tasks completed' if self.env.finished else 'Tasks incomplete'}"
            )
            print(f"  Makespan: {self.env.current_time:.2f} / {EnvParams.MAX_TIME}")

        # === 无条件打印奖励分项（用于诊断）===
        if not training:  # 只在评估时打印
            # 【MDP模式】的打印
            if TrainParams.EXECUTION_MODE == "mdp":
                sparse_weight = getattr(TrainParams, "MDP_SPARSE_REWARD_WEIGHT", 1.0)
                dense_weight = getattr(TrainParams, "MDP_DENSE_REWARD_WEIGHT", 1.0)

                print(f"\n[Episode Reward Breakdown - MDP模式 (γ=1)]")
                print(f"  Episode Return: {episode_raw_return:.2f}")
                print(
                    f"    ├─ Dense Time Penalty (稠密):  {episode_time_penalty_sum:.2f} (weight={dense_weight})"
                )
                print(
                    f"    └─ Terminal Reward (稀疏):     {episode_terminal_penalty_sum:.2f} (weight={sparse_weight})"
                )
                print(
                    f"  Makespan: {self.env.current_time:.2f}, Success: {self.env.finished}"
                )
                print(f"  Mode: MDP (γ=1)")

                # 理论分析
                print(f"\n[MDP Theoretical Analysis]")
                print(f"  Dense weight: {dense_weight}, Sparse weight: {sparse_weight}")
                print(f"  Dense time reward sum: {episode_time_penalty_sum:.2f}")
                print(
                    f"    (should ≈ {dense_weight} × (-{self.env.current_time:.2f}) = {dense_weight * (-self.env.current_time):.2f})"
                )
                print(f"  Terminal reward: {episode_terminal_penalty_sum:.2f}")
                if self.env.finished:
                    theoretical_terminal = sparse_weight * (-self.env.current_time)
                    theoretical_return = episode_time_penalty_sum + theoretical_terminal
                    print(
                        f"    (should = {sparse_weight} × (-{self.env.current_time:.2f}) = {theoretical_terminal:.2f})"
                    )
                    print(
                        f"  Theoretical total (success): {dense_weight}×(-T) + {sparse_weight}×(-T)"
                    )
                    print(
                        f"    = ({dense_weight} + {sparse_weight}) × (-{self.env.current_time:.2f}) = {theoretical_return:.2f}"
                    )
                else:
                    theoretical_terminal = sparse_weight * (-EnvParams.MAX_TIME)
                    theoretical_return = episode_time_penalty_sum + theoretical_terminal
                    print(
                        f"    (should = {sparse_weight} × (-{EnvParams.MAX_TIME}) = {theoretical_terminal:.2f})"
                    )
                    print(
                        f"  Theoretical total (failure): {dense_weight}×(-T_fail) + {sparse_weight}×(-T_max)"
                    )
                    print(f"    ≈ {theoretical_return:.2f}")
                print(f"  Actual return: {episode_raw_return:.2f}")
                print(
                    f"  Difference: {abs(episode_raw_return - theoretical_return):.4f}"
                )

            # 【SMDP模式】的打印
            else:
                sparse_weight = getattr(TrainParams, "SMDP_SPARSE_REWARD_WEIGHT", 1.0)
                dense_weight = getattr(TrainParams, "SMDP_DENSE_REWARD_WEIGHT", 0.0)

                print(f"\n[Episode Reward Breakdown - SMDP模式]")
                print(f"  Episode Return: {episode_raw_return:.2f}")
                print(
                    f"    ├─ Shaping Reward (势能):  {episode_shaping_reward_sum:.2f}"
                )
                print(
                    f"    │   ├─ Progress Potential (需求完成): {episode_progress_potential_sum:.2f}"
                )
                print(
                    f"    │   ├─ Time Potential (时间势):      {episode_time_potential_sum:.2f}"
                )
                print(
                    f"    │   └─ Task Completion Potential (任务完成%): {episode_task_completion_potential_sum:.2f} (weight={dense_weight})"
                )
                print(f"    ├─ Dense Time Penalty:     {episode_time_penalty_sum:.2f}")
                print(
                    f"    └─ Sparse Terminal Reward: {episode_terminal_penalty_sum:.2f} (weight={sparse_weight})"
                )
                print(f"  Reward Weights: sparse={sparse_weight}, dense={dense_weight}")
                print(
                    f"  Makespan: {self.env.current_time:.2f}, Success: {self.env.finished}"
                )
                print(
                    f"  Mode: EXECUTION_MODE={TrainParams.EXECUTION_MODE}, PENALTY_MODE={TrainParams.REWARD_TIME_PENALTY_MODE}"
                )

                # 打印时间势能的理论累积值（用于验证）
                if (
                    TrainParams.REWARD_TIME_POTENTIAL_WEIGHT != 0
                    and TrainParams.EXECUTION_MODE == "smdp"
                ):
                    T_final = self.env.current_time
                    theoretical_time_potential = (
                        -TrainParams.REWARD_TIME_POTENTIAL_WEIGHT
                        * np.exp(-TrainParams.BETA * T_final)
                        * T_final
                    )
                    print(f"\n[Time Potential Analysis]")
                    print(
                        f"  Time Potential Weight: {TrainParams.REWARD_TIME_POTENTIAL_WEIGHT}"
                    )
                    print(
                        f"  Theoretical Cumulative (伸缩级数): {theoretical_time_potential:.4f}"
                    )
                    print(f"  Formula: -c * exp(-β*T_final) * T_final")
                    print(
                        f"    = -{TrainParams.REWARD_TIME_POTENTIAL_WEIGHT} * exp(-{TrainParams.BETA:.6f}*{T_final:.2f}) * {T_final:.2f}"
                    )
                    print(
                        f"    = -{TrainParams.REWARD_TIME_POTENTIAL_WEIGHT} * {np.exp(-TrainParams.BETA * T_final):.6f} * {T_final:.2f}"
                    )
                    print(f"  Note: 实际累积可能略有差异（由于数值精度和离散化）")

        # makespan: 对于成功的episode记录实际时间，对于失败的episode记录MAX_TIME
        if episode_success >= 1.0:
            perf_metrics["makespan"] = [self.env.current_time]  # 成功：记录实际makespan
        else:
            perf_metrics["makespan"] = [EnvParams.MAX_TIME]  # 失败：记录MAX_TIME (500)

        # 【新增】实际完成时间（无论成功失败都记录实际时间）
        perf_metrics["actual_time"] = [self.env.current_time]
        if self.env.task_dic and "time_start" in next(
            iter(self.env.task_dic.values()), {}
        ):
            time_starts = [
                t.get("time_start", np.nan) for t in self.env.task_dic.values()
            ]
            perf_metrics["time_cost"] = [
                np.nanmean(time_starts) if not np.all(np.isnan(time_starts)) else 0
            ]
        else:
            perf_metrics["time_cost"] = [0]
        if self.env.agent_dic and "sum_waiting_time" in next(
            iter(self.env.agent_dic.values()), {}
        ):
            wait_times = [
                a.get("sum_waiting_time", 0) for a in self.env.agent_dic.values()
            ]
            perf_metrics["waiting_time"] = [np.mean(wait_times)]
        else:
            perf_metrics["waiting_time"] = [0]
        perf_metrics["travel_dist"] = [
            np.sum(self.env.get_matrix(self.env.agent_dic, "travel_dist"))
        ]
        perf_metrics["efficiency"] = [self.env.get_efficiency()]

        # ==================== 新增：收集LTL约束相关指标 ====================
        if TrainParams.LTL_ENABLED and ltl_monitor is not None:
            ltl_stats = ltl_monitor.get_statistics()

            # 添加LTL统计信息到perf_metrics（顺序：分项→总计）
            perf_metrics["ltl_num_safety"] = [ltl_stats["num_safety"]]
            perf_metrics["ltl_num_sequential"] = [ltl_stats["num_sequential"]]
            perf_metrics["ltl_num_clauses"] = [
                ltl_stats["num_clauses"]
            ]  # = num_safety + num_sequential
            perf_metrics["ltl_safety_violation_rate"] = [
                ltl_stats["safety_violation_rate"]
            ]
            perf_metrics["ltl_sequential_satisfaction_rate"] = [
                ltl_stats["sequential_satisfaction_rate"]
            ]
            perf_metrics["ltl_overall_satisfaction_rate"] = [
                ltl_stats["overall_satisfaction_rate"]
            ]
            perf_metrics["ltl_safety_violated_count"] = [ltl_stats["safety_violated"]]
            perf_metrics["ltl_sequential_satisfied_count"] = [
                ltl_stats["sequential_satisfied"]
            ]

            # 对于软约束方法（B/C/D），添加cost相关指标
            if (
                effective_constraint_mode != "HARD"
                and "costs" in buffer_dict
                and buffer_dict["costs"]
            ):
                costs_array = np.array(buffer_dict["costs"])
                perf_metrics["ltl_cost_per_step"] = [np.mean(costs_array)]
                perf_metrics["ltl_total_cost"] = [np.sum(costs_array)]
                perf_metrics["ltl_max_cost"] = [np.max(costs_array)]
                perf_metrics["ltl_cost_std"] = [np.std(costs_array)]
            else:
                # HARD模式或无cost数据时，这些指标为0
                perf_metrics["ltl_cost_per_step"] = [0.0]
                perf_metrics["ltl_total_cost"] = [0.0]
                perf_metrics["ltl_max_cost"] = [0.0]
                perf_metrics["ltl_cost_std"] = [0.0]
        else:
            # 如果LTL未启用或无ltl_monitor，所有LTL指标设为默认值
            perf_metrics["ltl_num_safety"] = [0]
            perf_metrics["ltl_num_sequential"] = [0]
            perf_metrics["ltl_num_clauses"] = [0]
            perf_metrics["ltl_safety_violation_rate"] = [0.0]
            perf_metrics["ltl_sequential_satisfaction_rate"] = [1.0]
            perf_metrics["ltl_overall_satisfaction_rate"] = [1.0]
            perf_metrics["ltl_safety_violated_count"] = [0]
            perf_metrics["ltl_sequential_satisfied_count"] = [0]
            perf_metrics["ltl_cost_per_step"] = [0.0]
            perf_metrics["ltl_total_cost"] = [0.0]
            perf_metrics["ltl_max_cost"] = [0.0]
            perf_metrics["ltl_cost_std"] = [0.0]
        # ====================================================================

        # print("terminal_reward", terminal_reward, "episode_raw_return", episode_raw_return)

        # 【诊断】评估结束时的总结
        if ENABLE_EVAL_DIAGNOSTICS:
            total_episode_time = time_module.time() - episode_real_start_time
            final_total_reqs = sum(
                np.sum(t["requirements"]) for t in self.env.task_dic.values()
            )
            delivered_reqs = initial_total_reqs - final_total_reqs

            print(f"\n{'=' * 80}")
            print(f"[评估诊断] Episode完成总结")
            print(f"{'=' * 80}")
            print(f"\n【基础信息】")
            print(f"  - Worker ID: {self.metaAgentID}")
            print(f"  - 训练模式: {training}")
            print(f"  - 采样模式: {sample}")
            print(f"  - 约束模式: {effective_constraint_mode}")

            print(f"\n【时间与效率】")
            print(f"  - 总决策步数: {decision_step}")
            print(f"  - 仿真时间: {self.env.current_time:.1f}/{EnvParams.MAX_TIME}")
            print(
                f"  - 真实耗时: {total_episode_time:.1f}秒 ({total_episode_time / 60:.2f}分钟)"
            )
            print(
                f"  - 平均每步耗时: {total_episode_time / max(decision_step, 1):.3f}秒"
            )
            print(
                f"  - 决策速率: {decision_step / max(total_episode_time, 0.001):.1f} 步/秒"
            )

            print(f"\n【环境规模】")
            print(f"  - Agent数量: {len(self.env.agent_dic)}")
            print(f"  - Task数量: {len(self.env.task_dic)}")
            print(f"  - Depot数量: {len(self.env.depot_dic)}")

            print(f"\n【任务完成情况】")
            print(f"  - 初始总需求: {initial_total_reqs}")
            print(f"  - 最终剩余需求: {final_total_reqs}")
            print(
                f"  - 已交付需求: {delivered_reqs} ({delivered_reqs / max(initial_total_reqs, 1):.1%})"
            )
            print(f"  - 完成任务数: {tasks_completed_count}/{len(self.env.task_dic)}")
            print(f"  - 环境Finished标志: {self.env.finished}")

            print(f"\n【执行动作统计】（实际执行的动作）")
            total_actions = sum(action_counters.values())
            print(
                f"  - MOVE动作: {action_counters['MOVE']} ({action_counters['MOVE'] / max(total_actions, 1):.1%})"
            )
            print(
                f"  - LOAD动作: {action_counters['LOAD']} ({action_counters['LOAD'] / max(total_actions, 1):.1%})"
            )
            print(
                f"  - UNLOAD动作: {action_counters['UNLOAD']} ({action_counters['UNLOAD'] / max(total_actions, 1):.1%})"
            )
            print(
                f"  - REJECTED动作: {action_counters['REJECTED']} ({action_counters['REJECTED'] / max(total_actions, 1):.1%})"
            )
            print(f"  - 总动作数: {total_actions}")

            print(f"\n【模型选择统计】（模型决策时选择的动作类型）")
            total_selections = sum(action_type_selection_stats.values())
            if total_selections > 0:
                print(
                    f"  - MOVE选择: {action_type_selection_stats['MOVE']} ({action_type_selection_stats['MOVE'] / total_selections:.1%})"
                )
                print(
                    f"  - LOAD选择: {action_type_selection_stats['LOAD']} ({action_type_selection_stats['LOAD'] / total_selections:.1%})"
                )
                print(
                    f"  - UNLOAD选择: {action_type_selection_stats['UNLOAD']} ({action_type_selection_stats['UNLOAD'] / total_selections:.1%})"
                )

            print(f"\n【动作可用性统计】（环境允许的动作类型次数）")
            total_available = sum(action_type_available_stats.values())
            if total_available > 0:
                print(f"  - MOVE可用: {action_type_available_stats['MOVE']} 次")
                print(
                    f"  - LOAD可用: {action_type_available_stats['LOAD']} 次 ({action_type_available_stats['LOAD'] / max(decision_step, 1):.1%} 的决策时刻)"
                )
                print(
                    f"  - UNLOAD可用: {action_type_available_stats['UNLOAD']} 次 ({action_type_available_stats['UNLOAD'] / max(decision_step, 1):.1%} 的决策时刻)"
                )

            # 分析：模型选择 vs 实际执行的差异
            if action_type_selection_stats["LOAD"] > 0 and action_counters["LOAD"] == 0:
                print(
                    f"\n⚠️  警告：模型选择了 {action_type_selection_stats['LOAD']} 次LOAD，但没有成功执行！"
                )
            if (
                action_type_selection_stats["UNLOAD"] > 0
                and action_counters["UNLOAD"] == 0
            ):
                print(
                    f"⚠️  警告：模型选择了 {action_type_selection_stats['UNLOAD']} 次UNLOAD，但没有成功执行！"
                )

            print(f"\n【性能指标】")
            print(f"  - Success Rate: {perf_metrics.get('success_rate', [0])[0]:.3f}")
            print(f"  - Makespan: {perf_metrics.get('makespan', [0])[0]:.2f}")
            print(
                f"  - Episode Return: {perf_metrics.get('episode_return', [0])[0]:.2f}"
            )
            print(
                f"    └─ Shaping Reward: {perf_metrics.get('episode_shaping_reward', [0])[0]:.2f}"
            )
            print(
                f"    └─ Time Penalty: {perf_metrics.get('episode_time_penalty', [0])[0]:.2f}"
            )
            print(
                f"    └─ Terminal Penalty: {perf_metrics.get('episode_terminal_penalty', [0])[0]:.2f}"
            )
            print(f"  - Travel Distance: {perf_metrics.get('travel_dist', [0])[0]:.2f}")
            print(f"  - Efficiency: {perf_metrics.get('efficiency', [0])[0]:.3f}")
            print(
                f"  - Waiting Time (平均): {perf_metrics.get('waiting_time', [0])[0]:.2f}"
            )

            # LTL约束相关信息
            if TrainParams.LTL_ENABLED and ltl_monitor is not None:
                print(f"\n【LTL约束】")
                print(f"  - 安全约束数: {perf_metrics.get('ltl_num_safety', [0])[0]}")
                print(
                    f"  - 顺序约束数: {perf_metrics.get('ltl_num_sequential', [0])[0]}"
                )
                print(f"  - 总约束数量: {perf_metrics.get('ltl_num_clauses', [0])[0]}")
                print(
                    f"  - 安全约束违反率: {perf_metrics.get('ltl_safety_violation_rate', [0])[0]:.3f}"
                )
                print(
                    f"  - 顺序约束满足率: {perf_metrics.get('ltl_sequential_satisfaction_rate', [0])[0]:.3f}"
                )
                print(
                    f"  - 整体满足率: {perf_metrics.get('ltl_overall_satisfaction_rate', [0])[0]:.3f}"
                )
                if effective_constraint_mode != "HARD":
                    print(
                        f"  - 平均每步Cost: {perf_metrics.get('ltl_cost_per_step', [0])[0]:.4f}"
                    )
                    print(
                        f"  - 总Cost: {perf_metrics.get('ltl_total_cost', [0])[0]:.2f}"
                    )
                    print(
                        f"  - 最大单步Cost: {perf_metrics.get('ltl_max_cost', [0])[0]:.4f}"
                    )

            print(f"\n【决策质量指标】")
            print(
                f"  - Action Type Entropy: {perf_metrics.get('entropy/action_type', [0])[0]:.3f}"
            )
            print(
                f"  - Destination Entropy: {perf_metrics.get('entropy/destination', [0])[0]:.3f}"
            )
            print(f"  - Cargo Entropy: {perf_metrics.get('entropy/cargo', [0])[0]:.3f}")
            print(
                f"  - Quantity Entropy: {perf_metrics.get('entropy/quantity', [0])[0]:.3f}"
            )

            print(f"\n{'=' * 80}")

        # 【CVAR_SMDP Quantile】将 risk_step_rewards 同步到 final_buffer
        # 这些per-step costs会被GAE自动累积成Future Cost-to-Go
        if "risk_step_rewards" in buffer_dict and buffer_dict["risk_step_rewards"]:
            final_buffer["risk_step_rewards"] = buffer_dict["risk_step_rewards"]
            if decision_step < 3:
                print(
                    f"[CVAR_SMDP Quantile] Collected {len(buffer_dict['risk_step_rewards'])} per-step risk costs"
                )
                print(
                    f"[CVAR_SMDP Quantile] Mean per-step cost: {-np.mean(buffer_dict['risk_step_rewards']):.6f}"
                )

        # 返回episode完整性标志（用于SMDP理论要求）
        return (
            terminal_reward,
            episode_raw_return,
            final_buffer,
            perf_metrics,
            episode_truly_done,
        )

    def baseline_test(self):
        self.baseline_env.plot_figure = False
        perf_metrics = {}
        current_action_index = 0
        start = time.time()
        while (
            not self.baseline_env.finished
            and self.baseline_env.current_time < self.max_time
            and current_action_index < 2000
        ):
            # while not self.baseline_env.finished:
            with torch.no_grad():
                release_agents, current_time = self.baseline_env.next_decision()
                random.shuffle(release_agents[0])
                self.baseline_env.current_time = current_time
                if time.time() - start > 30:
                    break
                while release_agents[0] or release_agents[1]:
                    agent_id = (
                        release_agents[0].pop(0)
                        if release_agents[0]
                        else release_agents[1].pop(0)
                    )
                    agent = self.baseline_env.agent_dic[agent_id]
                    task_info, total_agents, mask = self.convert_torch(
                        self.baseline_env.agent_observe(agent_id, False)
                    )

                    return_flag = mask[0, 1:].all().item()

                    # 【已修改】删除了对 'feasible_assignment' 的无效判断逻辑
                    if return_flag and agent["current_task"] < 0:
                        agent["no_choice"] = True
                        continue

                    task_info, total_agents, mask = self.obs_padding(
                        task_info, total_agents, mask
                    )
                    index = (
                        torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    )
                    probs, _ = self.local_baseline(task_info, total_agents, mask, index)
                    action = torch.argmax(probs, 1)
                    self.baseline_env.agent_step(agent_id, action.item(), None)
                    current_action_index += 1
                self.baseline_env.finished = self.baseline_env.check_finished()

        reward, finished_tasks = self.baseline_env.get_episode_reward(self.max_time)
        return reward

    def _compute_failure_statistics(self):
        """
        计算故障相关统计指标

        Returns:
            dict: 包含以下字段：
                - num_agents: 智能体总数
                - failed_count: 实际故障的智能体数量
                - actual_failure_rate: 实际故障比例 (failed_count / num_agents)
                - avg_failure_prob: 平均故障概率（基于Weibull模型的理论值）
                - expected_failures_C: 归一化期望故障率 C̄(τ) = (1/n)·Σ p_i ∈ [0,1]（IJCAI Eq. 41, normalized）
        """
        num_agents = len(self.env.agent_dic)
        failed_count = 0
        total_failure_prob = 0.0

        for agent in self.env.agent_dic.values():
            # 统计实际故障数量
            if agent.get("failed", False):
                failed_count += 1

            # 计算累积故障概率（基于Weibull分布）
            # 注意：此计算与episode-level的C̄值计算（line 2143-2153）数学上等价
            # 确保perf_metrics中的C̄值与传递给driver的C̄值一致
            cumulative_time = agent.get("cumulative_time", 0.0)
            if EnvParams.VEHICLE_FAILURE_ENABLED:
                # Weibull累积分布函数：F(t) = 1 - exp(-(t/λ)^k)
                # 等价于：p_i = 1 - exp(-H_i)，其中 H_i = (t/λ)^k
                failure_prob = 1.0 - np.exp(
                    -np.power(
                        cumulative_time / EnvParams.WEIBULL_LAMBDA, EnvParams.WEIBULL_K
                    )
                )
            else:
                failure_prob = 0.0

            total_failure_prob += failure_prob

        return {
            "num_agents": num_agents,
            "failed_count": failed_count,
            "actual_failure_rate": failed_count / num_agents if num_agents > 0 else 0.0,
            "avg_failure_prob": total_failure_prob / num_agents
            if num_agents > 0
            else 0.0,
            "expected_failures_C": total_failure_prob / num_agents
            if num_agents > 0
            else 0.0,  # C̄(τ) = (1/n)·Σ p_i ∈ [0,1] (IJCAI Eq. 41, normalized)
        }

    # work, convert_torch, obs_padding 方法无需修改，保持原样
    def work(self, episode_number):
        if episode_number == 0 or episode_number % 50 == 0:
            print(
                f"[Worker {self.metaAgentID}] Work start Ep={episode_number}. Mode={TrainParams.LTL_CONSTRAINT_TYPE}"
            )

        episode_returns = []
        baseline_rewards = []
        buffers = []
        metrics = []
        max_waiting = TrainParams.FORCE_MAX_OPEN_TASK

        # PPO模式下持续收集直到达到目标步数
        # IMPALA模式下运行POMO_SIZE个完整episode

        # 【Episode-level CVAR & Risk-Sensitive】追踪每个episode的指标
        episode_risk_index_map = {}  # {episode_id: C_episode} (CVAR故障成本)
        episode_return_map = {}  # {episode_id: episode_raw_return} (旧版Risk-Sensitive，已弃用)

        # 【路线1 Risk-Sensitive】新的episode级别指标
        episode_C_map = {}  # {episode_id: C_rel} (相对超额hazard)
        episode_T_map = {}  # {episode_id: T} (makespan)

        # 【Comprehensive Metrics】额外的理论验证和负载均衡指标
        episode_W_map = {}  # {episode_id: W} (总工作量)
        episode_H_actual_map = {}  # {episode_id: H_actual} (实际总hazard)
        episode_H_optimal_map = {}  # {episode_id: H_optimal} (理论最优hazard)
        episode_C_abs_map = {}  # {episode_id: C_abs} (绝对超额hazard)
        episode_work_std_map = {}  # {episode_id: work_std} (工作时间标准差)
        episode_work_cv_map = {}  # {episode_id: work_cv} (变异系数)
        episode_max_mean_ratio_map = {}  # {episode_id: max_mean_ratio} (最大/平均比率)
        episode_gini_coeff_map = {}  # {episode_id: gini_coeff} (Gini系数)

        if TrainParams.ALGORITHM == "PPO":
            # PPO模式：持续运行episode直到收集够PPO_ROLLOUT_LENGTH步
            total_collected_steps = 0
            max_ppo_episodes = 10  # 防止无限循环

            for episode_idx in range(max_ppo_episodes):
                (
                    terminal_reward,
                    episode_raw_return,
                    buffer,
                    perf_metrics,
                    episode_truly_done,
                ) = self.run_episode(
                    training=True,
                    sample=True,
                    max_waiting=max_waiting,
                    ltl_clauses=None,
                )

                # 注意：PPO使用GAE后，buffer中的key是'reward'（单数），不是'rewards'（复数）
                buffer_size = len(buffer.get("reward", buffer.get("rewards", [])))

                if terminal_reward is np.nan:
                    print(
                        f"[Worker {self.metaAgentID}] Episode {episode_idx} returned NaN reward, skipping"
                    )
                    max_waiting = True
                    continue

                if buffer_size == 0:
                    continue

                episode_returns.append(episode_raw_return)
                baseline_rewards.append(terminal_reward)

                # 为buffer添加episode完整性标志
                # 为每个transition标记所属episode ID和该episode是否完整
                buffer["episode_truly_done"] = [episode_truly_done] * buffer_size
                buffer["episode_id"] = [episode_idx] * buffer_size

                # 【Episode-level CVAR】存储该episode的统一风险指数
                if (
                    TrainParams.LTL_CONSTRAINT_TYPE == "CVAR_SMDP"
                    and "episode_unified_risk_index" in buffer
                ):
                    episode_risk_index_map[episode_idx] = buffer[
                        "episode_unified_risk_index"
                    ]

                # 【Comprehensive Metrics】存储所有comprehensive metrics
                # 【修改】NEU和RISK模式均需要记录
                if EnvParams.VEHICLE_FAILURE_ENABLED:
                    # 核心优化目标
                    if "episode_expected_failures_C" in buffer:
                        episode_C_map[episode_idx] = buffer[
                            "episode_expected_failures_C"
                        ]
                    if "episode_makespan_T" in buffer:
                        episode_T_map[episode_idx] = buffer["episode_makespan_T"]

                    # 理论验证指标
                    if "episode_total_work_W" in buffer:
                        episode_W_map[episode_idx] = buffer["episode_total_work_W"]
                    if "episode_H_actual" in buffer:
                        episode_H_actual_map[episode_idx] = buffer["episode_H_actual"]
                    if "episode_H_optimal" in buffer:
                        episode_H_optimal_map[episode_idx] = buffer["episode_H_optimal"]
                    if "episode_C_abs" in buffer:
                        episode_C_abs_map[episode_idx] = buffer["episode_C_abs"]

                    # 负载均衡指标
                    if "episode_work_std" in buffer:
                        episode_work_std_map[episode_idx] = buffer["episode_work_std"]
                    if "episode_work_cv" in buffer:
                        episode_work_cv_map[episode_idx] = buffer["episode_work_cv"]
                    if "episode_max_mean_ratio" in buffer:
                        episode_max_mean_ratio_map[episode_idx] = buffer[
                            "episode_max_mean_ratio"
                        ]
                    if "episode_gini_coeff" in buffer:
                        episode_gini_coeff_map[episode_idx] = buffer[
                            "episode_gini_coeff"
                        ]

                    # 前几个episode打印诊断信息
                    if (
                        episode_idx < 3
                        and episode_idx in episode_C_map
                        and episode_idx in episode_T_map
                    ):
                        print(
                            f"[Comprehensive Metrics Worker] EP {episode_idx}: "
                            f"C_rel={episode_C_map[episode_idx]:.4f}, T={episode_T_map[episode_idx]:.2f}"
                        )

                buffers.append(buffer)
                metrics.append(perf_metrics)

                # 统计收集的步数
                collected_this_episode = len(
                    buffer.get("reward", buffer.get("rewards", []))
                )
                total_collected_steps += collected_this_episode

                # 如果收集够了，截断最后一个buffer到目标长度
                if total_collected_steps >= TrainParams.PPO_ROLLOUT_LENGTH:
                    # 计算需要从最后一个buffer中保留多少步
                    overflow = total_collected_steps - TrainParams.PPO_ROLLOUT_LENGTH
                    if overflow > 0 and collected_this_episode > overflow:
                        # 截断最后一个buffer
                        keep_steps = collected_this_episode - overflow
                        for key in buffer.keys():
                            if isinstance(buffer[key], list) and len(buffer[key]) > 0:
                                buffer[key] = buffer[key][:keep_steps]
                    break
        else:
            # IMPALA/A2C模式：运行固定数量的完整episode
            for _ in range(TrainParams.POMO_SIZE):
                # self.env.init_state()
                (
                    terminal_reward,
                    episode_raw_return,
                    buffer,
                    perf_metrics,
                    episode_truly_done,
                ) = self.run_episode(
                    training=True,
                    sample=True,
                    max_waiting=max_waiting,
                    ltl_clauses=None,
                )
                if terminal_reward is np.nan:
                    max_waiting = True
                    continue
                episode_returns.append(episode_raw_return)

                baseline_rewards.append(terminal_reward)
                buffers.append(buffer)
                metrics.append(perf_metrics)
        baseline_reward = np.nanmean(baseline_rewards)

        # 清空 self.experience 以免累积旧数据
        self.experience = {k: [] for k in self.experience.keys()}
        self.perf_metrics = {k: [] for k in self.perf_metrics.keys()}

        for idx, buffer in enumerate(buffers):
            for key in buffer.keys():
                if key == 6:
                    for i in range(len(buffer[key])):
                        buffer[key][i] += baseline_rewards[idx] - baseline_reward
                if key not in self.experience.keys():
                    self.experience[key] = buffer[key]
                else:
                    self.experience[key] += buffer[key]

        for metric in metrics:
            for key in metric.keys():
                if key not in self.perf_metrics.keys():
                    self.perf_metrics[key] = metric[key]
                else:
                    self.perf_metrics[key] += metric[key]

        # print("episode_returns", episode_returns)
        if episode_returns:
            self.perf_metrics["episode_return"] = [np.nanmean(episode_returns)]

        # === 计算各个奖励成分的平均值（从metrics中提取）===
        if metrics:
            shaping_rewards = [
                m.get("episode_shaping_reward", [0])[0]
                for m in metrics
                if "episode_shaping_reward" in m
            ]
            time_penalties = [
                m.get("episode_time_penalty", [0])[0]
                for m in metrics
                if "episode_time_penalty" in m
            ]
            terminal_penalties = [
                m.get("episode_terminal_penalty", [0])[0]
                for m in metrics
                if "episode_terminal_penalty" in m
            ]

            if shaping_rewards:
                self.perf_metrics["episode_shaping_reward"] = [
                    np.nanmean(shaping_rewards)
                ]
            if time_penalties:
                self.perf_metrics["episode_time_penalty"] = [np.nanmean(time_penalties)]
            if terminal_penalties:
                self.perf_metrics["episode_terminal_penalty"] = [
                    np.nanmean(terminal_penalties)
                ]

        # 注意：PPO使用'reward'（单数），IMPALA可能使用'rewards'（复数）
        total_steps = len(
            self.experience.get("reward", self.experience.get("rewards", []))
        )

        # 【Episode-level CVAR & Risk-Sensitive】将episode映射添加到experience中
        if TrainParams.LTL_CONSTRAINT_TYPE == "CVAR_SMDP" and episode_risk_index_map:
            self.experience["episode_risk_index_map"] = episode_risk_index_map

        # 【Comprehensive Metrics】将comprehensive metrics映射添加到experience中
        # 【修改】NEU和RISK模式均需要
        if EnvParams.VEHICLE_FAILURE_ENABLED:
            # 核心优化目标
            if episode_C_map:
                self.experience["episode_C_map"] = episode_C_map
            if episode_T_map:
                self.experience["episode_T_map"] = episode_T_map

            # 理论验证指标
            if episode_W_map:
                self.experience["episode_total_work_W"] = episode_W_map
            if episode_H_actual_map:
                self.experience["episode_H_actual"] = episode_H_actual_map
            if episode_H_optimal_map:
                self.experience["episode_H_optimal"] = episode_H_optimal_map
            if episode_C_abs_map:
                self.experience["episode_C_abs"] = episode_C_abs_map

            # 负载均衡指标
            if episode_work_std_map:
                self.experience["episode_work_std"] = episode_work_std_map
            if episode_work_cv_map:
                self.experience["episode_work_cv"] = episode_work_cv_map
            if episode_max_mean_ratio_map:
                self.experience["episode_max_mean_ratio"] = episode_max_mean_ratio_map
            if episode_gini_coeff_map:
                self.experience["episode_gini_coeff"] = episode_gini_coeff_map

            # 打印统计信息
            if episode_C_map:
                C_values = list(episode_C_map.values())
                T_values = list(episode_T_map.values())
                print(f"[Comprehensive Metrics Worker] 收集{len(C_values)}个episode:")
                print(
                    f"  核心指标: C_rel平均={np.mean(C_values):.4f}, T平均={np.mean(T_values):.2f}"
                )

                # 打印其他comprehensive metrics摘要（如果存在）
                if episode_W_map:
                    W_values = list(episode_W_map.values())
                    H_actual_str = (
                        f"{np.mean(list(episode_H_actual_map.values())):.4f}"
                        if episode_H_actual_map
                        else "N/A"
                    )
                    H_optimal_str = (
                        f"{np.mean(list(episode_H_optimal_map.values())):.4f}"
                        if episode_H_optimal_map
                        else "N/A"
                    )
                    print(
                        f"  理论验证: W平均={np.mean(W_values):.2f}, "
                        f"H_actual平均={H_actual_str}, "
                        f"H_optimal平均={H_optimal_str}"
                    )
                if episode_work_std_map:
                    CV_str = (
                        f"{np.mean(list(episode_work_cv_map.values())):.4f}"
                        if episode_work_cv_map
                        else "N/A"
                    )
                    Gini_str = (
                        f"{np.mean(list(episode_gini_coeff_map.values())):.4f}"
                        if episode_gini_coeff_map
                        else "N/A"
                    )
                    print(
                        f"  负载均衡: Std平均={np.mean(list(episode_work_std_map.values())):.2f}, "
                        f"CV平均={CV_str}, "
                        f"Gini平均={Gini_str}"
                    )

        if self.save_image:
            try:
                self.env.plot_animation(SaverParams.GIFS_PATH, episode_number)
            except:
                pass
        self.episode_number = episode_number

        return self.experience, self.perf_metrics, total_steps

    def convert_torch(self, args):
        data = []
        for arg in args:
            data.append(torch.tensor(arg, dtype=torch.float).to(self.device))
        return data

    # @staticmethod
    # def obs_padding(task_info, agents, mask):
    #     task_info = F.pad(task_info, (0, 0, 0, EnvParams.TASKS_RANGE[1] + 1 - task_info.shape[1]), 'constant', 0)
    #     agents = F.pad(agents,
    #                    (0, 0, 0, EnvParams.SPECIES_AGENTS_RANGE[1] * EnvParams.SPECIES_RANGE[1] - agents.shape[1]),
    #                    'constant', 0)
    #     mask = F.pad(mask, (0, EnvParams.TASKS_RANGE[1] + 1 - mask.shape[1]), 'constant', 1)
    #     return task_info, agents, mask

    @staticmethod
    def obs_padding(task_info, agents, mask):

        task_info = F.pad(
            task_info,
            (0, 0, 0, EnvParams.MAX_DESTINATIONS - task_info.shape[1]),
            "constant",
            0,
        )
        agents = F.pad(
            agents,
            (
                0,
                0,
                0,
                EnvParams.SPECIES_AGENTS_RANGE[1] * EnvParams.SPECIES_RANGE[1]
                - agents.shape[1],
            ),
            "constant",
            0,
        )
        mask = F.pad(
            mask, (0, EnvParams.MAX_DESTINATIONS - mask.shape[1]), "constant", 1
        )
        return task_info, agents, mask
