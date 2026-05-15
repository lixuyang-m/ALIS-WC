# This file is derived from the HeteroMRTA codebase by Dai et al.
# (IEEE RA-L 2025), originally released under the Apache-2.0 License.
# Original source: https://github.com/marmotlab/HeteroMRTA

import torch
import numpy as np
import ray
import os
from attention import AttentionNet
from worker import Worker
from parameters import *
from env.task_env import TaskEnv


class Runner(object):
    """Actor object to start running simulation on workers.
    Gradient computation is also executed on this object."""

    def __init__(self, metaAgentID):
        self.metaAgentID = metaAgentID
        self.device = (
            torch.device("cuda") if TrainParams.USE_GPU else torch.device("cpu")
        )
        self.localNetwork = AttentionNet(
            TrainParams.AGENT_INPUT_DIM,
            TrainParams.TASK_INPUT_DIM,
            TrainParams.EMBEDDING_DIM,
        )
        self.localNetwork.to(self.device)
        self.localBaseline = AttentionNet(
            TrainParams.AGENT_INPUT_DIM,
            TrainParams.TASK_INPUT_DIM,
            TrainParams.EMBEDDING_DIM,
        )
        self.localBaseline.to(self.device)

    def get_weights(self):
        return self.localNetwork.state_dict()

    def set_weights(self, weights):
        self.localNetwork.load_state_dict(weights)

    def set_baseline_weights(self, weights):
        self.localBaseline.load_state_dict(weights)

    def training(
        self, global_weights, baseline_weights, curr_episode, env_params, seed=None
    ):
        print(
            "starting episode {} on metaAgent {}".format(curr_episode, self.metaAgentID)
        )
        # set the local weights to the global weight values from the master network
        self.set_weights(global_weights)
        self.set_baseline_weights(baseline_weights)
        save_img = False
        if SaverParams.SAVE_IMG:
            if curr_episode % SaverParams.SAVE_IMG_GAP == 0:
                save_img = True
        worker = Worker(
            self.metaAgentID,
            self.localNetwork,
            self.localBaseline,
            curr_episode,
            self.device,
            save_img,
            seed,
            env_params,
        )

        buffer, perf_metrics, collected_steps = worker.work(curr_episode)

        info = {
            "id": self.metaAgentID,
            "episode_number": curr_episode,
        }

        return buffer, perf_metrics, collected_steps, info

    def evaluate(self, global_weights, seed, ltl_benchmark=None, eval_mode=None):
        """
        评估模型性能

        Args:
            global_weights: 模型权重
            seed: 随机种子
            ltl_benchmark: LTL约束基准（如果有）
            eval_mode: 评估模式，可选值：
                - None: 使用训练时的约束模式（推荐，评估真实性能）
                - 'HARD': 强制使用硬约束（评估严格满足约束的能力）
                - 'SOFT_*': 强制使用特定的软约束模式

        Returns:
            perf_metrics: 性能指标字典
        """
        import time

        eval_start = time.time()

        # 根据 LTL_ENABLED_IN_EVALUATION 参数决定是否使用LTL约束
        effective_ltl_benchmark = None
        if TrainParams.LTL_ENABLED_IN_EVALUATION and ltl_benchmark is not None:
            effective_ltl_benchmark = ltl_benchmark

        print(f"\n{'=' * 80}")
        print(f"[评估开始] Worker {self.metaAgentID}")
        print(f"{'=' * 80}")
        print(f"  - 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - 随机种子: {seed}")
        print(
            f"  - 动作选择模式: {'采样(Sampling)' if TrainParams.EVAL_USE_SAMPLING else '贪婪(Greedy/Argmax)'}"
        )
        print(
            f"  - LTL评估开关: {'启用' if TrainParams.LTL_ENABLED_IN_EVALUATION else '禁用'}"
        )
        print(
            f"  - 评估模式: {eval_mode if eval_mode else TrainParams.LTL_CONSTRAINT_TYPE}"
        )
        print(f"  - LTL基准: {'是' if effective_ltl_benchmark else '否'}")
        if effective_ltl_benchmark:
            print(f"    - 约束数量: {len(effective_ltl_benchmark)}")
        elif ltl_benchmark and not TrainParams.LTL_ENABLED_IN_EVALUATION:
            print(
                f"    - ⚠️  提供了 {len(ltl_benchmark)} 个约束，但已被 LTL_ENABLED_IN_EVALUATION=False 禁用"
            )
        print(f"{'=' * 80}\n")

        self.set_weights(global_weights)

        # 评估使用与训练一致的环境参数范围
        eval_env_params = [
            EnvParams.SPECIES_AGENTS_RANGE,
            EnvParams.SPECIES_RANGE,
            EnvParams.TASKS_RANGE,
            EnvParams.DEPOT_NUM_RANGE,
        ]

        worker = Worker(
            self.metaAgentID,
            self.localNetwork,
            self.localBaseline,
            0,
            self.device,
            save_image=False,
            seed=seed,
            env_params=eval_env_params,
        )

        # 使用与训练一致的约束模式（除非明确指定eval_mode）
        episode_start = time.time()

        # 根据参数决定评估时使用采样还是贪婪选择
        eval_sample_mode = TrainParams.EVAL_USE_SAMPLING

        _terminal_reward, _raw_return, _buffer, perf_metrics, _episode_truly_done = (
            worker.run_episode(
                training=False,
                sample=eval_sample_mode,  # 使用配置的采样模式
                ltl_clauses=effective_ltl_benchmark,  # 使用过滤后的LTL约束
                constraint_mode_override=eval_mode,  # None表示使用训练时的模式
            )
        )
        episode_duration = time.time() - episode_start
        total_duration = time.time() - eval_start

        print(f"\n{'=' * 80}")
        print(f"[评估完成] Worker {self.metaAgentID}")
        print(f"{'=' * 80}")
        print(f"  - 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - 总耗时: {total_duration:.1f}秒 ({total_duration / 60:.2f}分钟)")
        print(f"  - Episode耗时: {episode_duration:.1f}秒")
        print(f"  - 种子: {seed}")
        print(f"\n  【核心指标总结】")
        print(f"  - Success Rate: {perf_metrics.get('success_rate', [0])[0]:.3f}")
        print(f"  - Makespan: {perf_metrics.get('makespan', [0])[0]:.2f}")
        print(f"  - Episode Return: {perf_metrics.get('episode_return', [0])[0]:.2f}")
        print(f"  - Efficiency: {perf_metrics.get('efficiency', [0])[0]:.3f}")
        if TrainParams.LTL_ENABLED:
            print(
                f"  - LTL Overall Satisfaction: {perf_metrics.get('ltl_overall_satisfaction_rate', [0])[0]:.3f}"
            )

        # 【新增】打印决策步数和动作统计
        decision_steps = int(perf_metrics.get("decision_steps", [0])[0])
        total_actions = int(perf_metrics.get("total_actions", [0])[0])
        move_count = int(perf_metrics.get("action_move_count", [0])[0])
        load_count = int(perf_metrics.get("action_load_count", [0])[0])
        unload_count = int(perf_metrics.get("action_unload_count", [0])[0])
        rejected_count = int(perf_metrics.get("action_rejected_count", [0])[0])

        print(f"\n  【决策与动作统计】")
        print(f"  - 决策步数: {decision_steps}")
        print(f"  - 总动作数: {total_actions}")
        if total_actions > 0:
            print(f"  - MOVE:   {move_count:4d} ({move_count / total_actions:6.1%})")
            print(f"  - LOAD:   {load_count:4d} ({load_count / total_actions:6.1%})")
            print(
                f"  - UNLOAD: {unload_count:4d} ({unload_count / total_actions:6.1%})"
            )
            print(
                f"  - REJECTED: {rejected_count:4d} ({rejected_count / total_actions:6.1%})"
            )
        else:
            print(f"  - ⚠️  未执行任何动作！")

        print(f"{'=' * 80}\n")

        return perf_metrics

    def testing(self, seed=None):
        worker = Worker(
            self.metaAgentID,
            self.localNetwork,
            self.localBaseline,
            0,
            self.device,
            False,
            seed,
        )
        reward = worker.baseline_test()
        return reward, seed, self.metaAgentID


@ray.remote(num_cpus=1, num_gpus=TrainParams.NUM_GPU / TrainParams.NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID):
        super().__init__(metaAgentID)

        # 为每个worker设置确定的随机种子（用于PPO可复现性）
        if TrainParams.USE_MANUAL_SEED:
            import random
            import numpy as np

            worker_seed = TrainParams.MANUAL_SEED + metaAgentID
            torch.manual_seed(worker_seed)
            torch.cuda.manual_seed(worker_seed)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            print(f"[Worker {metaAgentID}] 随机种子已设置为: {worker_seed}")


if __name__ == "__main__":
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.singleThreadedJob.remote(1)
    out = ray.get(job_id)
    print(out[1])
