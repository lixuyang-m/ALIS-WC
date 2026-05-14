"""
Greedy Baseline Solver for Heterogeneous Multi-Robot Task Allocation

Strategy:
1. Empty + At Depot → LOAD cargo with max demand, quantity = min(capacity, demand) → MOVE to nearest task needing that cargo
2. Carrying + At Task → UNLOAD all cargo → MOVE to nearest depot
3. Empty + At Task → MOVE to nearest depot

Key Features:
- Reuses existing Worker and TaskEnv infrastructure
- Supports LTL constraints (temporary sleep/wakeup)
- Uses agent_observe for action masking
- Uses species_distance_matrix for distance calculations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from env.task_env import TaskEnv
from env.ltl_utils import LTLMonitor, LTL_SAFETY, LTL_SEQUENTIAL


class GreedySolver:
    """
    Greedy baseline solver that makes locally optimal decisions

    The solver implements a simple greedy strategy:
    - Select cargo type with maximum total demand across all tasks
    - Load as much as possible: min(agent_capacity, task_demand, depot_stock)
    - Move to nearest task that needs the loaded cargo
    - Unload all cargo at task
    - Return to nearest depot
    """

    def __init__(self, env: TaskEnv, ltl_monitor: Optional[LTLMonitor] = None, seed: int = 42):
        """
        Initialize greedy solver

        Args:
            env: Task environment
            ltl_monitor: Optional LTL constraint monitor
            seed: Random seed for tie-breaking
        """
        self.env = env
        self.ltl_monitor = ltl_monitor
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        # 决策顺序的随机数生成器（与RL Worker保持一致）
        # 用于多智能体同时决策时的顺序打乱，确保可复现性
        self.decision_order_rng = np.random.RandomState(seed=seed)

        # Convert LTL clauses to tuple format if provided
        self.ltl_clauses = []
        if ltl_monitor:
            for clause in ltl_monitor.clauses:
                self.ltl_clauses.append((clause.type, clause.param1, clause.param2))

    def solve(self) -> Dict:
        """
        Run greedy algorithm to solve the task allocation problem

        Returns:
            metrics: Dictionary containing performance metrics
        """
        print(f"\n{'='*60}")
        print("Running Greedy Baseline")
        print(f"{'='*60}")
        print(f"Environment:")
        print(f"  - Agents: {self.env.agents_num}")
        print(f"  - Tasks: {self.env.tasks_num}")
        print(f"  - Depots: {self.env.depots_num}")
        print(f"  - Cargo types: {self.env.traits_dim}")
        if self.ltl_clauses:
            print(f"  - LTL constraints: {len(self.ltl_clauses)}")
        print(f"{'='*60}")

        # Reset environment
        self.env.init_state()

        # Reset LTL monitor state
        if self.ltl_monitor:
            self.ltl_monitor.reset_states()

        # Run episode with greedy policy
        metrics = self._run_greedy_episode()

        print(f"\n{'='*60}")
        print("Greedy Baseline Results")
        print(f"{'='*60}")
        print(f"  Success: {metrics.get('success', False)}")
        print(f"  Makespan: {metrics.get('makespan', float('inf')):.2f}")
        print(f"  Total distance: {metrics.get('total_distance', 0):.2f}")
        print(f"  Completed tasks: {metrics.get('completed_tasks', 0)}/{self.env.tasks_num}")
        print(f"{'='*60}")

        return metrics

    def _run_greedy_episode(self) -> Dict:
        """
        Run a single episode using greedy policy

        Returns:
            metrics: Performance metrics
        """
        max_steps = 10000
        step = 0
        done = False

        # Track metrics
        completed_tasks = set()

        while not done and step < max_steps:
            step += 1

            # Get next event time and ready agents
            (ready_agents, blocked_agents), next_event_time = self.env.next_decision()

            # Check if episode is done
            if next_event_time == np.inf:
                done = True
                break

            # Advance time to next decision point
            if next_event_time > self.env.current_time:
                self.env.current_time = next_event_time
                self.env.agent_update()
                finished_task_ids = self.env.task_update()

                # Update LTL monitor for finished tasks (SEQUENTIAL constraints)
                if self.ltl_monitor and finished_task_ids:
                    for tid in finished_task_ids:
                        self.ltl_monitor.update_monitor('task_finish', {'task_id': tid})

            # 【关键修复】在agent_update()之后重新获取需要决策的agents
            # 复用RL的逻辑：构建agents_to_process队列
            agents_to_process = [
                aid for aid, a in self.env.agent_dic.items()
                if (a.get('next_decision', np.inf) == self.env.current_time)
                   and (not a.get('is_inactive', False))
                   and (not a.get('is_temp_sleeping', False))
            ]

            # 打乱决策顺序（与RL Worker保持一致，确保可复现性）
            self.decision_order_rng.shuffle(agents_to_process)

            # Make greedy decisions for ready agents
            for agent_id in agents_to_process:
                # Make greedy decision
                action_dict = self._make_greedy_decision(agent_id)
                if action_dict is not None:
                    reward, doable, finished_tasks_from_step, event_info = self.env.agent_step(agent_id, action_dict, step)

                    # 【复用RL逻辑】更新LTL监视器并检查唤醒
                    if self.ltl_monitor and event_info:
                        self.ltl_monitor.update_monitor(event_info['type'], event_info['params'])

                        # 当任务完成时，检查是否需要唤醒休眠的agents
                        if event_info['type'] == 'task_finish':
                            self.check_and_wakeup_agents()

            # Track completed tasks
            for task_id, task in self.env.task_dic.items():
                if self._is_task_completed(task) and task_id not in completed_tasks:
                    completed_tasks.add(task_id)

            # Check if episode is done
            done = self._check_episode_done()

        # Calculate final metrics
        success = len(completed_tasks) == self.env.tasks_num
        makespan = self.env.current_time
        total_distance = sum(agent.get('travel_dist', 0) for agent in self.env.agent_dic.values())

        metrics = {
            'success': success,
            'makespan': makespan,
            'total_distance': total_distance,
            'completed_tasks': len(completed_tasks),
            'steps': step
        }

        return metrics

    def _make_greedy_decision(self, agent_id: int) -> Optional[Dict]:
        """
        Make greedy decision for a single agent

        使用环境的agent_observe获取掩码，确保遵守所有约束（包括LTL）

        Args:
            agent_id: Agent ID

        Returns:
            action_dict: Action dictionary with 'type' and parameters, or None
        """
        agent = self.env.agent_dic[agent_id]

        # 【关键】使用环境的agent_observe获取掩码
        # 这会自动处理：LTL约束、agent状态、库存、任务状态等
        task_info, total_agents, global_mask, ltl_info, masks_dict, cost_info, inaction_reason, blocking_clauses = \
            self.env.agent_observe(agent_id, self.ltl_monitor, max_waiting=False)

        # 【复用RL逻辑】根据inaction_reason处理agent状态
        if inaction_reason == 'NO_ACTION_BY_SAFETY_LTL':
            # 被静态安全约束永久阻塞 -> 永久失活
            agent['is_inactive'] = True
            agent['next_decision'] = np.inf
            return None
        elif inaction_reason == 'NO_ACTION_BY_LTL':
            # 被动态LTL约束阻塞（如顺序约束）-> 临时休眠
            agent['is_temp_sleeping'] = True
            agent['next_decision'] = np.inf
            agent['blocking_clauses'] = blocking_clauses
            return None
        elif inaction_reason == 'NO_ACTION_TEMPORARILY':
            # 当前无可执行动作，但有能力贡献 -> 临时休眠
            agent['is_temp_sleeping'] = True
            agent['next_decision'] = np.inf
            agent['blocking_clauses'] = []
            return None
        elif inaction_reason == 'NO_ACTION_BY_DEFAULT':
            # 永久无法对任何未完成任务贡献 -> 永久失活
            agent['is_inactive'] = True
            agent['next_decision'] = np.inf
            return None
        elif inaction_reason == 'TEMP_SLEEPING':
            return None

        # inaction_reason == 'ACTIONS_AVAILABLE'，可以行动
        action_type_mask = masks_dict['action_type']
        destination_mask = masks_dict['destination']
        cargo_to_load_mask = masks_dict['cargo_to_load']

        # 根据可用的动作类型做greedy选择
        # 优先级：UNLOAD > LOAD > MOVE（确保先卸货，再装货，最后移动）

        # 策略1：如果可以UNLOAD，优先卸货
        if not action_type_mask[self.env.ACTION_UNLOAD]:
            return self._select_unload_action(agent_id)

        # 策略2：如果可以LOAD，装载需求最大的货物
        if not action_type_mask[self.env.ACTION_LOAD]:
            return self._select_load_action_with_mask(agent_id, cargo_to_load_mask)

        # 策略3：如果可以MOVE，移动到最近的目标
        if not action_type_mask[self.env.ACTION_MOVE]:
            return self._select_move_action_with_mask(agent_id, destination_mask)

        # 没有可用动作
        return None

    def _select_load_action_with_mask(self, agent_id: int, cargo_to_load_mask: np.ndarray) -> Optional[Dict]:
        """
        使用掩码选择装载动作
        只使用掩码判断，选择第一个未被掩码的货物类型，装满

        Args:
            agent_id: Agent ID
            cargo_to_load_mask: 货物类型掩码（True表示不可用）

        Returns:
            action_dict: LOAD action dictionary or None
        """
        agent = self.env.agent_dic[agent_id]

        # 找到第一个未被掩码的货物类型
        best_cargo = None
        for cargo_type in range(self.env.traits_dim):
            if not cargo_to_load_mask[cargo_type]:
                best_cargo = cargo_type
                break

        if best_cargo is not None:
            # 装载该货物类型的最大容量
            quantity_to_load = int(agent['capacity'][best_cargo])

            # 创建数量向量
            quantity_vec = np.zeros(self.env.traits_dim, dtype=int)
            quantity_vec[best_cargo] = quantity_to_load

            return {
                'type': self.env.ACTION_LOAD,
                'quantity_vec': quantity_vec
            }

        return None

    def _select_unload_action(self, agent_id: int) -> Optional[Dict]:
        """
        选择卸载动作（卸载所有货物）

        Args:
            agent_id: Agent ID

        Returns:
            action_dict: UNLOAD action dictionary or None
        """
        agent = self.env.agent_dic[agent_id]
        cargo_type = agent['inventory']['type']
        quantity = agent['inventory']['quantity']

        if quantity > 0 and cargo_type is not None:
            # 创建数量向量
            quantity_vec = np.zeros(self.env.traits_dim, dtype=int)
            quantity_vec[cargo_type] = quantity

            return {
                'type': self.env.ACTION_UNLOAD,
                'quantity_vec': quantity_vec
            }

        return None

    def _select_move_action_with_mask(self, agent_id: int, destination_mask: np.ndarray) -> Optional[Dict]:
        """
        使用掩码选择移动动作（移动到最近的未被掩码的目标）
        只使用掩码判断，不做额外的状态检查

        Args:
            agent_id: Agent ID
            destination_mask: 目标掩码（True表示不可用）

        Returns:
            action_dict: MOVE action dictionary or None
        """
        agent = self.env.agent_dic[agent_id]
        current_location = agent['location']

        best_destination = None
        best_distance = float('inf')

        # 遍历所有目标（depot + task），只检查掩码
        num_destinations = self.env.depots_num + self.env.tasks_num
        for dest_id in range(num_destinations):
            # 只检查掩码
            if destination_mask[dest_id]:
                continue

            # 获取目标位置
            if dest_id < self.env.depots_num:
                target_location = self.env.depot_dic[dest_id]['location']
            else:
                task_id = dest_id - self.env.depots_num
                target_location = self.env.task_dic[task_id]['location']

            # 计算距离
            distance = np.linalg.norm(current_location - target_location)

            if distance < best_distance:
                best_distance = distance
                best_destination = dest_id

        if best_destination is not None:
            return {
                'type': self.env.ACTION_MOVE,
                'destination': best_destination
            }

        return None

    def _is_task_completed(self, task: Dict) -> bool:
        """
        Check if a task is completed (all requirements satisfied)

        Args:
            task: Task dictionary

        Returns:
            completed: True if task is completed
        """
        # Task is completed when status (remaining requirements) is all zeros
        return all(req == 0 for req in task['status'])

    def _check_episode_done(self) -> bool:
        """
        Check if episode is done (all tasks completed)

        Returns:
            done: True if episode should end
        """
        # Check if all tasks are completed
        all_completed = all(self._is_task_completed(task) for task in self.env.task_dic.values())

        return all_completed

    def check_and_wakeup_agents(self):
        """
        检查并唤醒临时休眠的agents（复用RL的逻辑）
        使用agent_observe作为唯一权威来判断是否应该唤醒
        """
        for agent_id, agent in self.env.agent_dic.items():
            if agent.get('is_temp_sleeping'):
                # 使用agent_observe判断是否应该唤醒
                _, _, _, _, _, _, inaction_reason, _ = self.env.agent_observe(
                    agent_id,
                    self.ltl_monitor,
                    max_waiting=False,
                    ignore_sleeping=True,
                )

                # 唤醒条件：在当前新的世界状态下，智能体有可用的动作
                if inaction_reason == 'ACTIONS_AVAILABLE':
                    agent['is_temp_sleeping'] = False
                    agent['next_decision'] = self.env.current_time
                    agent['blocking_clauses'] = []

