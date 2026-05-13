import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
import copy
import torch
import sys

# 添加父目录到路径以便导入 parameters
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameters import TrainParams

# from ltl_utils import *

DEBUG = False
DEBUG_TASK = False

# Define constants for clarity
LTL_SAFETY = 0
LTL_SEQUENTIAL = 1

# Safety FSA States
FSA_SAFETY_SAFE = 0
FSA_SAFETY_VIOLATED = 1

# Sequential FSA States
FSA_SEQ_INITIAL = 0
FSA_SEQ_PREDECESSOR_DONE = 1
FSA_SEQ_SATISFIED = 2
FSA_SEQ_VIOLATED = 3  # 新增：违反顺序约束的状态（吸收态）


class TaskEnv:
    def __init__(
        self,
        per_species_range=(10, 10),
        species_range=(5, 5),
        tasks_range=(30, 30),
        depot_num_range=(3, 5),
        traits_dim=5,
        decision_dim=10,
        max_task_size=2,
        max_cargo_per_type=5,  # MODIFIED: Changed name from original for clarity
        duration_scale=5,
        seed=None,
        plot_figure=False,
    ):
        """
        :param traits_dim: number of "cargo" types in this problem.
        :param seed: seed to generate pseudo random problem instance
        """
        self.rng = None
        self.per_species_range = per_species_range
        self.species_range = species_range
        self.tasks_range = tasks_range
        self.depot_num_range = depot_num_range
        self.max_task_size = max_task_size
        self.max_cargo_per_type = max_cargo_per_type  # MODIFIED: was max_agent_ability
        self.duration_scale = duration_scale
        self.plot_figure = plot_figure
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.traits_dim = traits_dim
        self.decision_dim = decision_dim

        self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = (
            self.generate_env()
        )

        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.species_num = len(self.species_dict["number"])
        self.depots_num = len(self.depot_dic)
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))

        self.species_distance_matrix, self.species_neighbor_matrix = (
            self.generate_distance_matrix()
        )

        self.current_time = 0
        self.dt = 0.1
        self.max_waiting_time = 200
        self.depot_waiting_time = 0
        self.finished = False
        self.reactive_planning = False

        ## NEW: Define action types as constants for clarity.
        self.ACTION_MOVE = 0
        self.ACTION_LOAD = 1
        self.ACTION_UNLOAD = 2

        self.fully_assigned_printed = set()

        self.last_policy_logits = None

    def recalculate_soft_cost(self, agent_id, policy_logits, ltl_monitor):
        """
        在获取策略 logits 后，根据不同的软约束模式重新计算成本。
        """
        mode = TrainParams.LTL_CONSTRAINT_TYPE
        if not ltl_monitor or not TrainParams.LTL_ENABLED or policy_logits is None:
            return {"total_cost": 0.0}

        if mode == "SOFT_POLICY":
            return self._calculate_policy_based_cost(
                agent_id, policy_logits, ltl_monitor
            )
        elif mode == "SOFT_HYBRID_STATE":
            return self._calculate_hybrid_state_cost(
                agent_id, policy_logits, ltl_monitor
            )
        # 模式'SOFT_DISCRETE'的成本在动作采样后计算，不在这里处理
        # 模式'HARD'没有成本
        return {"total_cost": 0.0}

    def get_discrete_action_cost(self, agent_id, action_dict, ltl_monitor):
        """
        (方法C的实现) 检查一个具体的、已采样的动作是否违反LTL约束。

        【修复】累加多个违规约束的成本，而不是只返回0或1
        - 如果违反N个约束，成本为N（原始计数）
        - 可以通过COST_BUDGET参数来调节惩罚强度

        返回值：
            float: 违规约束的数量（0表示无违规）
        """
        if not ltl_monitor or not action_dict:
            return 0.0

        action_type = action_dict.get("type")

        # 只有MOVE动作可能违反我们目前定义的LTL约束
        if action_type != self.ACTION_MOVE:
            return 0.0

        destination_id = action_dict.get("destination")
        violation_count = 0  # 改为累加违规数

        for clause in ltl_monitor.clauses:
            # 检查安全性约束 (G !p)
            if (
                clause.type == LTL_SAFETY
                and clause.state == FSA_SAFETY_SAFE
                and clause.param1 == agent_id
            ):
                forbidden_dest_id = clause.param2
                if destination_id == forbidden_dest_id:
                    violation_count += 1  # 累加违规

            # 检查顺序性约束 (A -> B)
            elif clause.type == LTL_SEQUENTIAL and clause.state == FSA_SEQ_INITIAL:
                # B是后继任务，A是前置任务
                task_b_id = clause.param2
                successor_dest_id = self.depots_num + task_b_id
                if destination_id == successor_dest_id:
                    violation_count += 1  # 累加违规

        # 返回违规数量（可以是0, 1, 2, ...）
        return float(violation_count)

    def _calculate_policy_based_cost(self, agent_id, policy_logits, ltl_monitor):
        """
        (方法B) 计算基于纯策略意图的成本 - 理论保证版本

        理论基础：
        - 成本函数: c(s,π) = E_{a~π}[I(a违规)] = P_π(违规动作|s)
        - 这是一个策略依赖但马尔可夫的成本函数
        - FSA状态已经编码了历史信息，无需显式混入状态进度

        为什么不混入状态信息？
        1. FSA状态q已经跟踪了"任务A是否完成"
        2. 策略网络的输入包含(s,q)，所以P_π已经条件在了正确的状态上
        3. 混入状态会破坏成本函数的马尔可夫性和理论保证

        收敛性保证：在标准CMDP假设下，算法收敛到约束最优策略
        """
        if policy_logits is None:
            return {"total_cost": 0.0, "type": "policy_based"}

        total_cost = 0.0
        clause_costs = []

        # 预计算目的地概率分布（所有子句共享，提高效率）
        dest_probs = torch.softmax(policy_logits["destination"], dim=-1)

        for clause in ltl_monitor.clauses:
            clause_cost = 0.0

            # 1. 安全性子句 (G ¬p): 智能体i不能访问节点j
            if clause.type == LTL_SAFETY and clause.param1 == agent_id:
                forbidden_dest_id = clause.param2

                if clause.state == FSA_SAFETY_SAFE:
                    # 安全状态：成本 = P_π(MOVE到禁区 | s, q=SAFE)
                    # 如果智能体选择去禁区，将会违反约束
                    clause_cost = dest_probs[0, forbidden_dest_id].item()

                elif clause.state == FSA_SAFETY_VIOLATED:
                    # 已违反状态：约束已经被违反（吸收态）
                    # 给予固定的高成本以反映违规状态
                    # 注意：这个成本不依赖于当前动作，因为违规已经发生
                    clause_cost = 1.0  # 最大成本

            # 2. 顺序性子句 (A → B): 任务B必须在任务A之后
            elif clause.type == LTL_SEQUENTIAL:
                task_b_id = clause.param2
                task_b_dest_id = self.depots_num + task_b_id

                if clause.state == FSA_SEQ_INITIAL:
                    # 初始状态：任务A未完成，不应该去任务B
                    # 成本 = P_π(MOVE到任务B | s, q=INITIAL)
                    clause_cost = dest_probs[0, task_b_dest_id].item()

                elif clause.state == FSA_SEQ_PREDECESSOR_DONE:
                    # 前驱完成状态：任务A已完成，可以去任务B
                    # 此时去任务B是允许的，成本为0
                    clause_cost = 0.0

                elif clause.state == FSA_SEQ_SATISFIED:
                    # 满足状态：任务B也已完成，约束已满足
                    # 成本为0
                    clause_cost = 0.0

                elif clause.state == FSA_SEQ_VIOLATED:
                    # 违反状态：在任务A完成前就开始了任务B
                    # 给予固定的高成本
                    clause_cost = 1.0  # 最大成本

            clause_costs.append(clause_cost)

        if clause_costs:
            total_cost = np.mean(clause_costs)

        return {
            "total_cost": total_cost,
            "type": "policy_based",
            "clause_costs": clause_costs,
        }

    def _calculate_discrete_cost(self, agent_id, masks_dict, ltl_monitor):
        """
        (方法C的占位符) 在SOFT_DISCRETE模式下, 真实的0/1成本计算已转移到worker中、
        在动作采样后进行。此函数在agent_observe阶段仅返回一个0成本的占位符。
        """
        return {"total_cost": 0.0, "type": "discrete_post_action"}

    def _calculate_hybrid_state_cost(self, agent_id, policy_logits, ltl_monitor):
        """(方法D) 计算混合成本。安全性使用策略意图，顺序性仅使用状态进度。"""
        if policy_logits is None:
            return {"total_cost": 0.0, "type": "hybrid"}

        total_cost = 0.0
        clause_costs = []

        for clause in ltl_monitor.clauses:
            clause_cost = 0.0
            # 安全性子句: 成本为智能体选择前往被禁止区域的概率 (策略意图)
            if (
                clause.type == LTL_SAFETY
                and clause.state == FSA_SAFETY_SAFE
                and clause.param1 == agent_id
            ):
                forbidden_dest_id = clause.param2
                dest_probs = torch.softmax(policy_logits["destination"], dim=-1)
                clause_cost = dest_probs[0, forbidden_dest_id].item()

            # 顺序性子句: 成本仅为前置任务的未完成度 (状态进度)
            elif clause.type == LTL_SEQUENTIAL and clause.state == FSA_SEQ_INITIAL:
                task_a_id = clause.param1
                task_a = self.task_dic.get(task_a_id)
                state_based_penalty = 1.0

                if task_a and not task_a.get("finished", False):
                    original_req = np.sum(task_a["requirements"])
                    current_status = np.sum(task_a["status"])
                    if original_req > 0:
                        # 任务A的完成度
                        progress_A = (original_req - current_status) / original_req
                        # 成本是未完成度
                        state_based_penalty = 1.0 - progress_A
                    else:
                        state_based_penalty = 0.0  # 任务A无需求, 视为已完成
                else:
                    state_based_penalty = 0.0  # 任务A不存在或已完成, 无惩罚

                clause_cost = state_based_penalty

            clause_costs.append(clause_cost)

        if clause_costs:
            total_cost = np.mean(clause_costs)

        return {
            "total_cost": total_cost,
            "type": "hybrid",
            "clause_costs": clause_costs,
        }

    def _is_instance_solvable(self, tasks_ini, species_capacities_ini):
        """
        验证生成的任务实例是否可解。
        检查每个被需求的货物类型，是否至少有一个智能体物种能够运输它。
        """
        # 1. 找出所有任务总共需要哪些货物类型
        total_requirements = np.sum(tasks_ini, axis=0)
        required_cargo_types = np.where(total_requirements > 0)[0]

        # 2. 找出所有智能体总共能运输哪些货物类型
        total_capacities = np.sum(species_capacities_ini, axis=0)

        # 3. 检查每一种被需求的货物，是否有对应的运输能力
        for cargo_type in required_cargo_types:
            if total_capacities[cargo_type] == 0:
                # 发现一种被需求的货物，但没有任何物种能运输它
                if DEBUG:
                    print(
                        f"[ENV_VALIDATION] Failed: Required cargo type {cargo_type} cannot be transported by any agent. Regenerating..."
                    )
                return False

        # 所有被需求的货物类型都能被运输
        return True

    # ==============================================================================

    def random_int(self, low, high, size=None):
        if self.rng is not None:
            integer = self.rng.integers(low, high, size)
        else:
            integer = np.random.randint(low, high, size)
        return integer

    def random_value(self, row, col):
        if self.rng is not None:
            value = self.rng.random((row, col))
        else:
            value = np.random.rand(row, col)
        return value

    def random_choice(self, a, size=None, replace=True):
        if self.rng is not None:
            choice = self.rng.choice(a, size, replace)
        else:
            choice = np.random.choice(a, size, replace)
        return choice

    def generate_task(self, tasks_num):
        tasks_ini = self.random_int(
            0, self.max_task_size + 1, (tasks_num, self.traits_dim)
        )
        while not np.all(np.sum(tasks_ini, axis=1) != 0):
            tasks_ini = self.random_int(
                0, self.max_task_size, (tasks_num, self.traits_dim)
            )
        return tasks_ini

    # def generate_agent_capacities(self, species_num):
    #     # MODIFIED: Logic to generate capacity vectors
    #     capacities_ini = self.random_int(0, self.max_cargo_per_type + 1, (species_num, self.traits_dim))
    #     while not np.all(np.sum(capacities_ini, axis=1) > 0):
    #         capacities_ini = self.random_int(0, self.max_cargo_per_type + 1, (species_num, self.traits_dim))
    #     return capacities_ini

    def generate_agent_capacities(self, species_num):
        """
        MODIFIED: Generates a capabilities matrix that is guaranteed to be solvable.
        It sequentially checks each capability dimension (cargo type) and ensures
        at least one species can service it, patching if necessary.
        """
        # Step 1: Generate a completely random capability matrix, including zeros.
        capacities_ini = self.random_int(
            0, self.max_cargo_per_type + 1, (species_num, self.traits_dim)
        )

        # Step 2: Ensure every species has at least one capability (prevents all-zero rows).
        # This check remains important.
        while not np.all(np.sum(capacities_ini, axis=1) > 0):
            capacities_ini = self.random_int(
                0, self.max_cargo_per_type + 1, (species_num, self.traits_dim)
            )

        # ==================== NEW: Sequentially check and patch each capability ====================
        # Step 3: Iterate through each capability dimension (each cargo type).
        for k in range(self.traits_dim):
            # Check if the sum of capabilities for the current cargo type is zero across all species.
            if np.sum(capacities_ini[:, k]) == 0:
                # If no species can handle cargo type 'k', we must patch it.
                # Randomly pick one species to gain this capability.
                random_species_idx = self.random_int(0, species_num)
                # Assign a random capability between 1 and max.
                capacities_ini[random_species_idx, k] = self.random_int(
                    1, self.max_cargo_per_type + 1
                )
        # ========================================================================================

        return capacities_ini

    # task_env.py 中 generate_env 函数的完整代码
    def generate_env(self):
        while True:
            # --- Part 1: Generate the core components of the problem ---
            tasks_num = self.random_int(self.tasks_range[0], self.tasks_range[1] + 1)
            species_num = self.random_int(
                self.species_range[0], self.species_range[1] + 1
            )
            depots_num = self.random_int(
                self.depot_num_range[0], self.depot_num_range[1] + 1
            )
            agents_species_num = [
                self.random_int(
                    self.per_species_range[0], self.per_species_range[1] + 1
                )
                for _ in range(species_num)
            ]

            species_capacities_ini = self.generate_agent_capacities(species_num)
            tasks_ini = self.generate_task(tasks_num)

            # --- Part 2: Validate the generated instance ---
            if self._is_instance_solvable(tasks_ini, species_capacities_ini):
                # If the instance is solvable, break the loop and proceed
                break

        # --- Part 3: Construct the final environment dictionaries using the validated components ---
        total_requirements = np.sum(tasks_ini, axis=0)
        depot_loc = self.random_value(depots_num, 2)

        depot_stocks = []
        for s in range(depots_num):
            stock = {
                i: int(req * 100 / depots_num) + self.random_int(5, 10)
                for i, req in enumerate(total_requirements)
            }
            depot_stocks.append(stock)

        generated_total_stock = np.sum([list(s.values()) for s in depot_stocks], axis=0)
        assert np.all(generated_total_stock >= total_requirements), (
            "Environment generation failed: Insufficient stock for tasks."
        )

        tasks_loc = self.random_value(tasks_num, 2)
        tasks_time = self.random_value(tasks_num, 1) * self.duration_scale

        task_dic = dict()
        agent_dic = dict()
        depot_dic = dict()
        species_dict = dict()

        species_dict["capacities"] = species_capacities_ini
        species_dict["number"] = agents_species_num

        for d in range(depots_num):
            depot_dic[d] = {
                "location": depot_loc[d, :],
                "members": [],
                "stock": depot_stocks[d],
                "ID": -d - 1,
            }
            depot_dic[d]["initial_stock"] = depot_stocks[d].copy()

        i = 0
        for s, n in enumerate(agents_species_num):
            species_dict[s] = []
            for j in range(n):
                random_depot_idx = self.random_int(0, depots_num)
                random_depot_node_id = -random_depot_idx - 1
                random_depot_location = depot_dic[random_depot_idx]["location"]

                agent_dic[i] = {
                    "ID": i,
                    "species": s,
                    "capacity": species_capacities_ini[s, :],
                    "inventory": {"type": None, "quantity": 0},
                    "location": random_depot_location,
                    "route": [random_depot_node_id],
                    "current_task": random_depot_node_id,
                    "arrival_time": [0.0],
                    "velocity": 0.2,
                    "next_decision": 0,
                    "depot": depot_loc[s % depots_num, :],
                    "travel_dist": 0,
                    "trajectory": [],
                    "angle": 0,
                    "returned": False,
                    "pre_set_route": None,
                    "is_inactive": False,  # MODIFIED: Add inactive state
                }
                species_dict[s].append(i)
                depot_dic[random_depot_idx]["members"].append(i)
                i += 1

        for i in range(tasks_num):
            task_dic[i] = {
                "ID": i,
                "requirements": tasks_ini[i, :],
                "location": tasks_loc[i, :],
                "finished": False,
                "status": tasks_ini[i, :].copy(),
                "time": float(tasks_time[i, :]),
            }

        if DEBUG:
            print("=== init requirement ===")
            for tid, t in task_dic.items():
                print(f"  Task{tid}: {t['requirements']}")
            print("====================")

        return task_dic, agent_dic, depot_dic, species_dict

    def generate_distance_matrix(self):
        species_distance_matrix = {}
        species_neighbor_matrix = {}
        for species in range(len(self.species_dict["number"])):
            tmp_dic = {-1: self.depot_dic[species % self.depots_num], **self.task_dic}
            distances = {}
            for from_counter, from_node in tmp_dic.items():
                distances[from_counter] = {}
                for to_counter, to_node in tmp_dic.items():
                    if from_counter == to_counter:
                        distances[from_counter][to_counter] = 0
                    else:
                        distances[from_counter][to_counter] = (
                            self.calculate_eulidean_distance(from_node, to_node)
                        )

            sorted_distance_matrix = {
                k: sorted(dist, key=lambda x: dist[x]) for k, dist in distances.items()
            }
            species_distance_matrix[species] = distances
            species_neighbor_matrix[species] = sorted_distance_matrix
        return species_distance_matrix, species_neighbor_matrix

    def reset(self, test_env=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = None
        if test_env is not None:
            self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = (
                copy.deepcopy(test_env)
            )
        else:
            self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = (
                self.generate_env()
            )
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.species_num = len(self.species_dict["number"])
        self.depots_num = len(self.depot_dic)
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))
        self.current_time = 0
        self.finished = False

    # task_env.py 中 init_state 函数的完整代码
    def init_state(self):
        for task in self.task_dic.values():
            task.update(finished=False, status=task["requirements"].copy())

        for agent in self.agent_dic.values():
            random_depot_idx = self.random_int(0, self.depots_num)
            random_depot_node_id = -random_depot_idx - 1
            random_depot_location = self.depot_dic[random_depot_idx]["location"]

            agent.update(
                route=[random_depot_node_id],
                location=random_depot_location,
                current_task=random_depot_node_id,
                inventory={"type": None, "quantity": 0},
                next_decision=0,
                travel_dist=0,
                arrival_time=[0.0],
                trajectory=[],
                angle=0,
                returned=False,
                pre_set_route=None,
                is_inactive=False,
                is_temp_sleeping=False,
                blocking_clauses=[],
            )  # Pure LTL: 移除故障相关字段
        for depot in self.depot_dic.values():
            depot["stock"] = depot["initial_stock"].copy()

        self.current_time = 0
        self.finished = False

        self.fully_assigned_printed.clear()

    @staticmethod
    def get_matrix(dictionary, key):
        key_matrix = []
        for value in dictionary.values():
            key_matrix.append(value[key])
        return key_matrix

    @staticmethod
    def calculate_eulidean_distance(agent_or_node, task_or_node):
        return np.linalg.norm(agent_or_node["location"] - task_or_node["location"])

    @staticmethod
    def calculate_weibull_features(
        cumulative_time,
        lambda_param,
        k_param,
        min_reliability,
        hazard_threshold,
        max_time,
    ):
        """
        计算威布尔分布相关的故障特征

        Args:
            cumulative_time: 累积工作时间
            lambda_param: 威布尔尺度参数
            k_param: 威布尔形状参数
            min_reliability: 最低可靠性阈值
            hazard_threshold: 故障率归一化阈值
            max_time: 最大时间（用于归一化）

        Returns:
            tuple: (cumulative_time_normalized, reliability, hazard_rate_normalized, remaining_safe_time_normalized)
        """
        # 1. 归一化累积时间（使用威布尔尺度参数归一化更合理）
        cumulative_time_normalized = cumulative_time / lambda_param

        # 2. 可靠性 R(t) = exp(-(t/λ)^k)
        reliability = np.exp(-np.power(cumulative_time / lambda_param, k_param))

        # 3. 瞬时故障率（hazard rate）h(t) = (k/λ) * (t/λ)^(k-1)
        if cumulative_time > 0:
            hazard_rate = (k_param / lambda_param) * np.power(
                cumulative_time / lambda_param, k_param - 1
            )
        else:
            hazard_rate = 0.0
        hazard_rate_normalized = min(hazard_rate / hazard_threshold, 1.0)

        # 4. 剩余安全时间（达到最低可靠性阈值的时间）
        # R(t_safe) = min_reliability => t_safe = λ * (-ln(R_min))^(1/k)
        if min_reliability > 0 and min_reliability < 1:
            t_safe_max = lambda_param * np.power(
                -np.log(min_reliability), 1.0 / k_param
            )
            remaining_safe_time = max(0.0, t_safe_max - cumulative_time)
            remaining_safe_time_normalized = remaining_safe_time / max_time
        else:
            remaining_safe_time_normalized = 1.0  # 无限安全

        return (
            cumulative_time_normalized,
            reliability,
            hazard_rate_normalized,
            remaining_safe_time_normalized,
        )

    def get_current_agent_status(self, agent):
        # Pure LTL版本：仅包含任务分配相关的基础特征
        from parameters import EnvParams

        status = []

        for a in self.agent_dic.values():
            travel_time = 0
            current_waiting_time = 0
            is_assigned = 0

            # Check if the agent is currently on a route to a destination
            if len(a["route"]) > len(
                a["arrival_time"]
            ):  # a move was decided but not yet reflected in arrival_time
                pass  # this case might need careful handling depending on when status is called
            elif a.get("next_decision") == np.inf:
                travel_time = np.clip(
                    a["arrival_time"][-1] - self.current_time, a_min=0, a_max=None
                )
                is_assigned = 1

            inventory_vec = np.zeros(1 + self.traits_dim)
            inv = a["inventory"]
            if inv["type"] is not None and inv["quantity"] > 0:
                cap_for_type = a["capacity"][inv["type"]]
                if cap_for_type > 0:
                    inventory_vec[0] = inv["quantity"] / cap_for_type
                else:
                    inventory_vec[0] = 1.0
                inventory_vec[1 + inv["type"]] = 1.0

            # Agent能力向量（归一化到[0,1]）
            # 用于推断失活原因：agent能力与剩余任务需求的匹配情况
            capacity_normalized = a["capacity"] / EnvParams.MAX_AGENT_CAPACITY

            # Pure LTL特征组合（18维）：
            # inventory_vec(6) + travel_time(1) + current_waiting_time(1) +
            # relative_location(2) + is_assigned(1) + capacity_normalized(5)
            temp_status = np.hstack(
                [
                    inventory_vec,
                    travel_time,
                    current_waiting_time,
                    agent["location"] - a["location"],
                    is_assigned,
                    capacity_normalized,
                ]
            )

            status.append(temp_status)

        current_agents = np.vstack(status)

        # 添加全局时间特征（每个agent都附加相同的时间信息）
        # 这对于预测sparse makespan penalty至关重要
        max_time = EnvParams.MAX_TIME
        normalized_current_time = self.current_time / max_time  # 归一化到[0,1]
        normalized_remaining_time = (max_time - self.current_time) / max_time

        # 将时间特征广播到每个agent的状态上 (+2维)
        num_agents = current_agents.shape[0]
        time_features = np.tile(
            [normalized_current_time, normalized_remaining_time], (num_agents, 1)
        )
        current_agents = np.hstack([current_agents, time_features])

        return current_agents  # 最终维度：16 + 2 = 18维

    def get_current_task_status(self, agent, ltl_monitor=None):
        """
        【迁移学习优化版：基础任务特征，LTL信息独立编码】

        为当前agent获取所有目的地（depot + tasks）的特征向量。
        LTL约束信息不嵌入到任务特征中，而是通过独立的ltl_info参数传递给网络。

        这样设计的优势：
        1. task_embedding专注于编码基础任务特征（状态、需求、位置等）
        2. 预训练的task_embedding可以完全冻结，无需担心LTL维度权重未训练
        3. LTL信息通过独立模块（ltl_embedding, GAT等）处理，更灵活

        Args:
            agent: 当前agent
            ltl_monitor: LTL监视器（可选），用于生成独立的ltl_info，不影响任务特征

        Returns:
            np.ndarray: [num_destinations, 15] 特征矩阵
            - 对于depot: feature_dim = (traits_dim * 2 + 1) + 1 + 2 + 1 = 15
            - 对于task: feature_dim = traits_dim + traits_dim + 1 + 1 + 2 + 1 = 15
        """
        status = []

        # ===== Depot特征（填充实际库存信息） =====
        for depot_id, depot in self.depot_dic.items():
            travel_time = (
                self.calculate_eulidean_distance(agent, depot) / agent["velocity"]
            )

            # 构建depot_feature_vec: [current_stock(TRAIT_DIM), remaining_ratio(TRAIT_DIM), depot_id_norm(1)]
            # 前TRAIT_DIM维：当前库存（归一化到agent的capacity，表示"能装几次"）
            current_stock_vec = np.zeros(self.traits_dim, dtype=np.float32)
            for c_type in range(self.traits_dim):
                stock_qty = depot["stock"].get(c_type, 0)
                agent_cap = agent["capacity"][c_type]
                # 归一化：库存量 / (agent能力 + 1)，避免除零，+1使得1单位库存映射到~1.0
                current_stock_vec[c_type] = (
                    stock_qty / (agent_cap + 1.0) if agent_cap > 0 else stock_qty
                )

            # 中TRAIT_DIM维：库存剩余比例（当前库存 / 初始库存）
            remaining_ratio_vec = np.zeros(self.traits_dim, dtype=np.float32)
            for c_type in range(self.traits_dim):
                initial_qty = depot["initial_stock"].get(c_type, 0)
                current_qty = depot["stock"].get(c_type, 0)
                remaining_ratio_vec[c_type] = (
                    current_qty / initial_qty if initial_qty > 0 else 0.0
                )

            # 最后1维：depot_id归一化（范围[0,1]）
            depot_id_norm = depot_id / max(self.depots_num, 1)

            depot_feature_vec = np.hstack(
                [
                    current_stock_vec,  # TRAIT_DIM维
                    remaining_ratio_vec,  # TRAIT_DIM维
                    [depot_id_norm],  # 1维
                ]
            )  # 总计：TRAIT_DIM*2 + 1 = 11维

            # 基础特征
            base_features = np.hstack(
                [
                    depot_feature_vec,  # 11维（TRAIT_DIM=5）
                    travel_time,  # 1维
                    agent["location"] - depot["location"],  # 2维
                    1,  # is_available（depot总是可用）  # 1维
                ]
            )  # 总计：15维

            # 【迁移学习优化】不在基础特征中包含LTL信息
            # LTL约束通过独立的编码模块处理（ltl_info参数）
            temp_status = base_features  # 15维
            status.append(temp_status)

        # ===== Task特征 =====
        for task_id, t in self.task_dic.items():
            travel_time = self.calculate_eulidean_distance(agent, t) / agent["velocity"]
            is_available = not t.get("finished", True)

            # 基础任务特征
            base_features = np.hstack(
                [
                    t["status"],  # 完成进度向量 (TRAIT_DIM=5)
                    t["requirements"],  # 需求向量 (TRAIT_DIM=5)
                    t.get("time", 0),  # 持续时间 (1)
                    travel_time,  # 旅行时间 (1)
                    agent["location"] - t["location"],  # 相对位置 (2)
                    is_available,  # 可用性 (1)
                ]
            )  # 总计：15维

            # 【迁移学习优化】不在基础特征中包含LTL信息
            # LTL约束信息完全通过独立的ltl_info参数传递给网络
            # 这样task_embedding可以专注于编码基础任务特征，便于冻结和迁移
            temp_status = base_features  # 15维
            status.append(temp_status)

        current_tasks = np.vstack(status)
        return current_tasks

    def get_arrival_time(self, agent_id, task_id):
        arrival_time = self.agent_dic[agent_id]["arrival_time"]
        route_indices = np.where(
            np.array(self.agent_dic[agent_id]["route"]) == task_id
        )[0]
        if len(route_indices) == 0:
            return np.inf
        arrival_for_task = route_indices[-1]
        return float(arrival_time[arrival_for_task])

    def get_waiting_tasks(self):
        # This original function's logic is ambiguous in the new model and is kept as a placeholder.
        waiting_tasks = np.ones(self.tasks_num, dtype=bool)
        waiting_agents = []
        for task in self.task_dic.values():
            if (
                not task["finished"]
                and len(task.get("members", [])) > 0
                and len(task.get("members", []))
                > len(task.get("contributing_agents", []))
            ):
                waiting_tasks[task["ID"]] = False
                waiting_agents += task["members"]
        return waiting_tasks, waiting_agents

    ## MODIFIED: agent_update is simplified to only handle consequences of arrival.
    def agent_update(self):
        """
        This function now also handles the automatic drop-off of cargo
        when a laden agent arrives at a depot, closing the logic loop for stuck agents.
        """
        for agent in self.agent_dic.values():
            # Check if agent is scheduled for an arrival at the current time
            # Skip agents that are permanently inactive or temporarily sleeping
            # if agent.get('next_decision') == np.inf and self.current_time >= agent.get('arrival_time', [np.inf])[-1]:
            if (
                agent.get("next_decision") == np.inf
                and not agent.get("is_temp_sleeping", False)
                and not agent.get("is_inactive", False)
                and self.current_time >= agent.get("arrival_time", [np.inf])[-1]
            ):
                # Update agent's logical and physical location upon arrival
                destination_node_id = agent["route"][-1]
                agent["current_task"] = destination_node_id
                if destination_node_id >= 0:
                    agent["location"] = self.task_dic[destination_node_id]["location"]
                else:
                    agent["location"] = self.depot_dic[-destination_node_id - 1][
                        "location"
                    ]

                if DEBUG:
                    print(
                        f"    (ENV UPDATE) [T={self.current_time:.2f}s] Agent {agent['ID']} has arrived at Node {destination_node_id}."
                    )

                # Check if the agent arrived at a depot while carrying cargo
                is_at_depot = destination_node_id < 0
                is_carrying_cargo = agent["inventory"]["quantity"] > 0

                if is_at_depot and is_carrying_cargo:
                    # This is the "return to base with useless cargo" scenario.
                    # Automatically drop off the cargo back into the depot's stock.

                    depot = self.depot_dic[-destination_node_id - 1]
                    inv = agent["inventory"]
                    carried_type = inv["type"]
                    carried_qty = inv["quantity"]

                    # Return cargo to the depot's stock
                    if carried_type is not None:
                        depot["stock"][carried_type] += carried_qty

                    # Empty the agent's inventory
                    agent["inventory"]["type"] = None
                    agent["inventory"]["quantity"] = 0

                    if DEBUG:
                        print(
                            f"    (DROP-OFF) [T={self.current_time:.2f}s] Agent {agent['ID']} dropped off {carried_qty} of cargo type {carried_type} at Depot {-destination_node_id - 1}. Agent is now empty."
                        )

                # Agent is now ready for a new decision
                agent["next_decision"] = self.current_time

    def task_update(self):
        f_task = []
        for task in self.task_dic.values():
            if task.get("finished"):
                continue
            if np.all(task.get("status", [1]) <= 0):
                task["finished"] = True
                if DEBUG:
                    print(
                        f"    -> [STATE CHANGE] Task {task['ID']} 'finished' flag set to True at T={self.current_time:.2f}"
                    )
                task["time_finish"] = self.current_time
                f_task.append(task["ID"])

        all_finished = np.all(self.get_matrix(self.task_dic, "finished"))
        if all_finished:
            for agent in self.agent_dic.values():
                if agent["current_task"] == -agent["species"] - 1:
                    agent["returned"] = True
        return f_task

    def next_decision(self):
        decision_time = np.array(
            [a.get("next_decision", np.inf) for a in self.agent_dic.values()],
            dtype=float,
        )

        # 同时检查所有智能体的arrival_time（只考虑未来的到达，即arrival_time > current_time）
        # 这样当智能体在移动中（next_decision=inf）时，也能推进到arrival_time
        arrival_times = []
        for a in self.agent_dic.values():
            arrival_time_list = a.get("arrival_time", [])
            # 只考虑最后一次到达时间，且必须大于当前时间
            if len(arrival_time_list) > 0:
                last_arrival = arrival_time_list[-1]
                if last_arrival > self.current_time:
                    arrival_times.append(last_arrival)
                else:
                    arrival_times.append(np.inf)
            else:
                arrival_times.append(np.inf)
        arrival_time_array = np.array(arrival_times, dtype=float)

        # 找出下一个事件时间：取decision_time和arrival_time的最小值
        all_event_times = np.minimum(decision_time, arrival_time_array)

        if np.all(np.isnan(all_event_times)):
            return ([], []), np.nan

        finite_times = all_event_times[np.isfinite(all_event_times)]
        if len(finite_times) == 0:
            return ([], []), np.inf

        next_time = np.min(finite_times)

        # ready_agents 是那些 next_decision == next_time 的智能体
        # 注意：即使某个智能体的arrival_time == next_time，如果它的next_decision != next_time，
        # 它也不算ready，因为agent_update()会在current_time推进后处理到达事件
        ready_agents = np.where(decision_time == next_time)[0].tolist()
        blocked_agents = np.where(np.isinf(decision_time))[0].tolist()
        return (ready_agents, blocked_agents), next_time

    def agent_step(self, agent_id: int, action: dict, decision_step: int = None):
        agent = self.agent_dic[agent_id]
        action_type = action.get("type")
        event_info = None

        # --- ACTION TYPE: MOVE ---
        if action_type == self.ACTION_MOVE:
            destination_action_id = action.get("destination")
            if destination_action_id is None:
                return 0, False, []

            # Logic for moving to a depot
            if destination_action_id < self.depots_num:
                depot_id = destination_action_id
                target_id = -depot_id - 1
                target_node = self.depot_dic[depot_id]
            # Logic for moving to a task
            else:
                task_id = destination_action_id - self.depots_num
                if task_id >= self.tasks_num or self.task_dic.get(task_id, {}).get(
                    "finished", True
                ):
                    if DEBUG:
                        print(
                            f"    (WARN) [T={self.current_time:.2f}s] Agent {agent_id} attempted to move to an invalid/finished task {task_id}."
                        )
                    return 0, False, []
                target_id = task_id
                target_node = self.task_dic[task_id]

                if "time_start" not in target_node:
                    target_node["time_start"] = self.current_time

            event_info = {
                "type": "agent_move",
                "params": {
                    "agent_id": agent_id,
                    "destination_node_id": destination_action_id,
                },
            }

            current_location = agent["location"]
            dist = np.linalg.norm(target_node["location"] - current_location)
            travel_time = dist / agent["velocity"]

            agent["travel_dist"] += dist
            agent["route"].append(target_id)
            agent["arrival_time"].append(self.current_time + travel_time)

            agent["next_decision"] = np.inf

            if DEBUG:
                print(
                    f"    (ACTION) [T={self.current_time:.2f}s] Agent {agent_id} is MOVING from Node {agent['current_task']} to Node {target_id}. Arrival at {agent['arrival_time'][-1]:.2f}s."
                )

            return 0, True, [], event_info

        # --- ACTION TYPE: LOAD ---
        elif action_type == self.ACTION_LOAD:
            # Check for legality: must be at a depot and empty
            if not (agent["current_task"] < 0 and agent["inventory"]["quantity"] == 0):
                if DEBUG:
                    print(
                        f"    (ACTION) [T={self.current_time:.2f}s] Agent {agent_id} illegal LOAD: Not at depot or not empty."
                    )
                return 0, False, [], None

            quantity_vec = action.get("quantity_vec", [])
            non_zero_indices = np.nonzero(quantity_vec)[0]
            if len(non_zero_indices) != 1:
                if DEBUG:
                    print(
                        f"    (ACTION) [T={self.current_time:.2f}s] Agent {agent_id} illegal LOAD: Must load exactly one cargo type."
                    )
                return 0, False, [], None

            cargo_type = non_zero_indices[0]
            requested_quantity = quantity_vec[cargo_type]

            depot = self.depot_dic[-agent["current_task"] - 1]
            available_stock = depot["stock"].get(cargo_type, 0)
            agent_capacity_for_type = agent["capacity"][cargo_type]

            # 核心计算
            quantity_to_load = int(
                min(requested_quantity, available_stock, agent_capacity_for_type)
            )

            if DEBUG:
                # ==================== 新增的超详细“LOAD动作”追踪器 ====================
                print(
                    f"\n--- [LOAD ATTEMPT] by Agent {agent_id} at T={self.current_time:.2f} ---"
                )
                print(f"  - At Depot: {-agent['current_task'] - 1}")
                print(f"  - Requested: {int(requested_quantity)} of Type {cargo_type}")
                print(f"  - Depot Stock Available: {int(available_stock)}")
                print(f"  - Agent Capacity for Type: {int(agent_capacity_for_type)}")
                print(f"  - Calculated Quantity to Load: {quantity_to_load}")
                # ===================================================================

            if quantity_to_load > 0:
                agent["inventory"]["type"] = cargo_type
                agent["inventory"]["quantity"] = quantity_to_load
                depot["stock"][cargo_type] -= quantity_to_load
                if DEBUG:
                    print(f"    -> SUCCESS: Loaded {quantity_to_load} units.")
            else:
                if DEBUG:
                    print(
                        f"    -> FAILURE: Load quantity is 0. Agent state will not change."
                    )

            agent["next_decision"] = self.current_time + 0.1

            if DEBUG:
                print(
                    f"    -> Final next_decision set to: {agent['next_decision']:.2f}"
                )
                print(f"--------------------------------------------------")

            return 0, True, [], None

        # --- ACTION TYPE: UNLOAD (MODIFIED LOGIC) ---
        elif action_type == self.ACTION_UNLOAD:
            # Universal check: agent must not be empty to unload
            if agent["inventory"]["quantity"] <= 0:
                if DEBUG:
                    print(
                        f"    (ACTION) [T={self.current_time:.2f}s] Agent {agent_id} illegal UNLOAD: Agent is empty."
                    )
                return 0, False, [], None

            # Branch 1: Unloading at a Task
            if agent["current_task"] >= 0:
                quantity_vec = action.get("quantity_vec", [])
                non_zero_indices = np.nonzero(quantity_vec)[0]
                if len(non_zero_indices) != 1:
                    if DEBUG:
                        print(
                            f"    (ACTION) [T={self.current_time:.2f}s] Agent {agent_id} illegal UNLOAD: Must unload exactly one cargo type."
                        )
                    return 0, False, [], None

                cargo_type = non_zero_indices[0]
                if cargo_type != agent["inventory"]["type"]:
                    if DEBUG:
                        print(
                            f"    (ACTION) [T={self.current_time:.2f}s] Agent {agent_id} illegal UNLOAD: Mismatch between inventory and unload action."
                        )
                    return 0, False, [], None

                requested_quantity = quantity_vec[cargo_type]
                task = self.task_dic[agent["current_task"]]
                needed_by_task = task["status"][cargo_type]
                quantity_to_unload = int(
                    min(
                        requested_quantity,
                        needed_by_task,
                        agent["inventory"]["quantity"],
                    )
                )

                reward = 0

                if DEBUG:
                    # ==================== 新增的调试打印 ====================
                    if quantity_to_unload == 0 and agent["inventory"]["quantity"] > 0:
                        print(
                            f"[DEBUG ZERO_UNLOAD at T={self.current_time:.2f}] Agent {agent_id} attempted to UNLOAD cargo type {carried_type}, but task needed 0. Agent still has {agent['inventory']['quantity']} units."
                        )
                    # =======================================================

                if quantity_to_unload > 0:
                    # reward = TrainParams.REWARD_SCALE_ALPHA * quantity_to_unload
                    if DEBUG_TASK:
                        pre_unload_status = task["status"].copy()

                    agent["inventory"]["quantity"] -= quantity_to_unload
                    task["status"][cargo_type] -= quantity_to_unload

                    if agent["inventory"]["quantity"] <= 0:
                        agent["inventory"]["type"] = None
                        agent["inventory"]["quantity"] = 0

                    if DEBUG:
                        print(
                            f"    (ACTION) [T={self.current_time:.2f}s] Agent {agent_id} UNLOADED {quantity_to_unload} of type {cargo_type} at Task {agent['current_task']}."
                        )

                    if DEBUG_TASK:
                        post_unload_status = task["status"]
                        print(
                            f"    (TASK_UPDATE) [T={self.current_time:.2f}s] Task {agent['current_task']} status changed from {pre_unload_status} to {post_unload_status}."
                        )
                else:
                    return 0, False, [], None

                agent["next_decision"] = self.current_time + 0.1

                finished_tasks = self.task_update()

                if finished_tasks:
                    finished_task_id = finished_tasks[0]
                    event_info = {
                        "type": "task_finish",
                        "params": {"task_id": finished_task_id},
                    }

                return reward, True, finished_tasks, event_info

            # Branch 2: Unloading (Dropping off) at a Depot
            elif agent["current_task"] < 0:
                depot_id = -agent["current_task"] - 1
                depot = self.depot_dic[depot_id]
                inv = agent["inventory"]
                returned_type = inv["type"]
                returned_qty = inv["quantity"]

                # Add the cargo back to the depot's stock
                depot["stock"][returned_type] += returned_qty

                if DEBUG:
                    print(
                        f"    (ACTION) [T={self.current_time:.2f}s] Agent {agent_id} DROPPED OFF {returned_qty} of type {returned_type} at Depot {depot_id}."
                    )

                # Clear the agent's inventory
                agent["inventory"]["type"] = None
                agent["inventory"]["quantity"] = 0

                # Agent is now ready for an immediate new decision
                agent["next_decision"] = self.current_time + 0.1

                return 0, True, [], None

            # Fallback for any other case
            return 0, False, [], None

        # Fallback for unknown or invalid action type
        if DEBUG:
            print(
                f"    (WARN) [T={self.current_time:.2f}s] Agent {agent_id} passed an unknown action type: {action_type}"
            )
        return 0, False, [], None

    # def agent_observe(self, agent_id, ltl_monitor=None, max_waiting=False):
    #     agent = self.agent_dic[agent_id]
    #
    #     if ltl_monitor and TrainParams.LTL_ENABLED:
    #         ltl_info_tensor = ltl_monitor.get_state_tensor()
    #     else:
    #         ltl_info_tensor = np.zeros((TrainParams.LTL_MAX_CLAUSES, 7), dtype=np.float32)
    #
    #     # 规则 0 (最高优先级): 休眠检查
    #     if agent.get('is_inactive', False):
    #         tasks_info = self.get_current_task_status(agent)
    #         agents_info = self.get_current_agent_status(agent)
    #         num_dest = self.depots_num + self.tasks_num
    #         destination_mask = np.ones(num_dest, dtype=bool)
    #         masks_dict = {'action_type': np.ones(3, dtype=bool),
    #                       'cargo_to_load': np.ones(self.traits_dim, dtype=bool), 'destination': destination_mask}
    #         tasks_info_exp = np.expand_dims(tasks_info, axis=0)
    #         agents_info_exp = np.expand_dims(agents_info, axis=0)
    #         mask_for_model = np.expand_dims(destination_mask, axis=0)
    #         return tasks_info_exp, agents_info_exp, mask_for_model, ltl_info_tensor, masks_dict, 'NO_ACTION_BY_DEFAULT', []
    #
    #     if agent.get('is_temp_sleeping', False):
    #         # This block creates a fully masked observation, preserving your original code's variable names and format.
    #         tasks_info = self.get_current_task_status(agent)
    #         agents_info = self.get_current_agent_status(agent)
    #         num_dest = self.depots_num + self.tasks_num
    #         destination_mask = np.ones(num_dest, dtype=bool)
    #         masks_dict = {'action_type': np.ones(3, dtype=bool),
    #                       'cargo_to_load': np.ones(self.traits_dim, dtype=bool),
    #                       'destination': destination_mask}
    #         tasks_info_exp = np.expand_dims(tasks_info, axis=0)
    #         agents_info_exp = np.expand_dims(agents_info, axis=0)
    #         mask_for_model = np.expand_dims(destination_mask, axis=0)
    #
    #         # Returns with a consistent signature. The agent is blocked by default while sleeping.
    #         return tasks_info_exp, agents_info_exp, mask_for_model, ltl_info_tensor, masks_dict, 'TEMP_SLEEPING', []
    #
    #     effective_statuses = {tid: t['status'].copy() for tid, t in self.task_dic.items() if not t.get('finished')}
    #     self.last_effective_statuses = effective_statuses
    #     for other_agent in self.agent_dic.values():
    #         is_moving_to_task = (
    #                     other_agent.get('next_decision') == np.inf and len(other_agent.get('route', [])) > 1 and
    #                     other_agent['route'][-1] >= 0)
    #         if is_moving_to_task and other_agent['inventory']['quantity'] > 0:
    #             dest_task_id = other_agent['route'][-1]
    #             if dest_task_id in effective_statuses:
    #                 carried_type = other_agent['inventory']['type']
    #                 if carried_type is not None:
    #                     carried_qty = other_agent['inventory']['quantity']
    #                     needed_by_task = effective_statuses[dest_task_id][carried_type]
    #                     committed_qty = min(carried_qty, needed_by_task)
    #                     effective_statuses[dest_task_id][carried_type] -= committed_qty
    #
    #     if agent['inventory']['quantity'] == 0:
    #         raw_demand = np.sum(list(effective_statuses.values()), axis=0)
    #         if np.isscalar(raw_demand): raw_demand = np.zeros(self.traits_dim)
    #         can_be_useful = False
    #         for c_type, capacity in enumerate(agent['capacity']):
    #             if capacity > 0 and raw_demand[c_type] > 0:
    #                 can_be_useful = True
    #                 break
    #         if not can_be_useful:
    #             agent['is_inactive'] = True
    #             agent['next_decision'] = np.inf
    #             if DEBUG:
    #                 print(f"[INACTIVE] Agent {agent_id} set to inactive. No remaining tasks match its capabilities.")
    #             return self.agent_observe(agent_id, ltl_monitor, max_waiting)
    #
    #     # 规则 1: 游戏终局规则
    #     all_tasks_done = np.all(self.get_matrix(self.task_dic, 'finished'))
    #     if all_tasks_done and not agent.get('returned', False):
    #         action_type_mask = np.array([False, True, True], dtype=bool)
    #         cargo_to_load_mask = np.ones(self.traits_dim, dtype=bool)
    #         destination_task_mask = np.ones(self.tasks_num, dtype=bool)
    #         destination_depot_mask = np.ones(self.depots_num, dtype=bool)
    #         home_depot_id = agent['species'] % self.depots_num
    #         is_at_home_depot = (agent['current_task'] == -home_depot_id - 1)
    #         if is_at_home_depot:
    #             agent['returned'] = True
    #             action_type_mask[self.ACTION_MOVE] = True
    #         else:
    #             destination_depot_mask[home_depot_id] = False
    #         destination_mask = np.concatenate((destination_depot_mask, destination_task_mask))
    #         masks_dict = {'action_type': action_type_mask, 'cargo_to_load': cargo_to_load_mask,
    #                       'destination': destination_mask}
    #         tasks_info = self.get_current_task_status(agent)
    #         agents_info = self.get_current_agent_status(agent)
    #         tasks_info_exp = np.expand_dims(tasks_info, axis=0)
    #         agents_info_exp = np.expand_dims(agents_info, axis=0)
    #         mask_for_model = np.expand_dims(destination_mask, axis=0)
    #         return tasks_info_exp, agents_info_exp, mask_for_model, ltl_info_tensor, masks_dict, 'ACTIONS_AVAILABLE', []
    #
    #     # 规则 2: “必须卸货”规则
    #     is_at_task = agent['current_task'] >= 0
    #     is_carrying_cargo = agent['inventory']['quantity'] > 0
    #     if is_at_task and is_carrying_cargo:
    #         task = self.task_dic[agent['current_task']]
    #         carried_type = agent['inventory']['type']
    #         if carried_type is not None and task['status'][carried_type] > 0:
    #             action_type_mask = np.array([True, True, False], dtype=bool)
    #             destination_mask = np.ones(self.depots_num + self.tasks_num, dtype=bool)
    #             masks_dict = {'action_type': action_type_mask,
    #                           'cargo_to_load': np.ones(self.traits_dim, dtype=bool),
    #                           'destination': destination_mask}
    #             tasks_info = self.get_current_task_status(agent)
    #             agents_info = self.get_current_agent_status(agent)
    #             tasks_info_exp = np.expand_dims(tasks_info, axis=0)
    #             agents_info_exp = np.expand_dims(agents_info, axis=0)
    #             mask_for_model = np.expand_dims(destination_mask, axis=0)
    #             return tasks_info_exp, agents_info_exp, mask_for_model, ltl_info_tensor, masks_dict, 'ACTIONS_AVAILABLE', []
    #
    #     # ==================== 规则 3: 默认决策逻辑 (最终整合版) ====================
    #     action_type_mask = np.zeros(3, dtype=bool)
    #     cargo_to_load_mask = np.ones(self.traits_dim, dtype=bool)
    #     destination_depot_mask = np.ones(self.depots_num, dtype=bool)
    #     destination_task_mask = np.ones(self.tasks_num, dtype=bool)
    #     is_at_depot = not is_at_task
    #
    #     if is_carrying_cargo:
    #         action_type_mask[self.ACTION_LOAD] = True
    #         carried_type = agent['inventory']['type']
    #
    #         valid_task_destinations = []
    #         if carried_type is not None:
    #             for task_id, effective_status_vec in effective_statuses.items():
    #                 if effective_status_vec[carried_type] > 0:
    #                     valid_task_destinations.append(task_id)
    #         is_cargo_useful = len(valid_task_destinations) > 0
    #
    #         if is_cargo_useful:
    #             destination_depot_mask[:] = True
    #             destination_task_mask[:] = True
    #             for task_id in valid_task_destinations:
    #                 destination_task_mask[task_id] = False
    #             action_type_mask[self.ACTION_UNLOAD] = True
    #         else:
    #             destination_task_mask[:] = True
    #             if is_at_depot:
    #                 action_type_mask[self.ACTION_MOVE] = True
    #                 action_type_mask[self.ACTION_UNLOAD] = False
    #             else:  # is_at_task
    #                 action_type_mask[self.ACTION_UNLOAD] = True
    #                 destination_depot_mask[:] = False
    #
    #     else:  # Agent is empty
    #         action_type_mask[self.ACTION_UNLOAD] = True
    #         if is_at_task:
    #             destination_depot_mask[:] = False
    #             destination_task_mask[:] = True
    #             action_type_mask[self.ACTION_LOAD] = True
    #         else:  # is_at_depot
    #             global_demand = np.sum(list(effective_statuses.values()), axis=0)
    #             if np.isscalar(global_demand): global_demand = np.zeros(self.traits_dim)
    #
    #             can_load_anything_useful = False
    #             depot = self.depot_dic[-agent['current_task'] - 1]
    #             for c_type in range(self.traits_dim):
    #                 if depot['stock'].get(c_type, 0) > 0 and agent['capacity'][c_type] > 0 and global_demand[
    #                     c_type] > 0:
    #                     cargo_to_load_mask[c_type] = False
    #                     can_load_anything_useful = True
    #
    #             if can_load_anything_useful:
    #                 action_type_mask[self.ACTION_LOAD] = False
    #                 action_type_mask[self.ACTION_MOVE] = True
    #             else:
    #                 action_type_mask[self.ACTION_LOAD] = True
    #                 destination_depot_mask[:] = False
    #                 destination_task_mask[:] = True
    #
    #     if is_at_depot:
    #         current_depot_action_id = -agent['current_task'] - 1
    #         if current_depot_action_id < self.depots_num:
    #             destination_depot_mask[current_depot_action_id] = True
    #     elif is_at_task:
    #         current_task_id = agent['current_task']
    #         if current_task_id < self.tasks_num:
    #             destination_task_mask[current_task_id] = True
    #
    #     destination_mask = np.concatenate((destination_depot_mask, destination_task_mask))
    #     if np.all(destination_mask):
    #         action_type_mask[self.ACTION_MOVE] = True
    #
    #     masks_dict = {'action_type': action_type_mask, 'cargo_to_load': cargo_to_load_mask,
    #                   'destination': destination_mask}
    #     tasks_info = self.get_current_task_status(agent)
    #     agents_info = self.get_current_agent_status(agent)
    #     tasks_info_exp = np.expand_dims(tasks_info, axis=0)
    #     agents_info_exp = np.expand_dims(agents_info, axis=0)
    #     mask_for_model = np.expand_dims(destination_mask, axis=0)
    #
    #     base_masks_dict = masks_dict
    #
    #     # Check if blocked by default environment rules
    #     is_move_blocked_base = np.all(base_masks_dict['destination'])
    #     # An agent is blocked if all action types are masked, OR if MOVE is the only available action type and all destinations are masked.
    #     is_blocked_by_default = False
    #     if not np.any(~base_masks_dict['action_type']):  # if all action types are masked
    #         is_blocked_by_default = True
    #     elif not base_masks_dict['action_type'][self.ACTION_MOVE] and is_move_blocked_base:
    #         if base_masks_dict['action_type'][self.ACTION_LOAD] and base_masks_dict['action_type'][self.ACTION_UNLOAD]:
    #             is_blocked_by_default = True
    #
    #     if is_blocked_by_default:
    #         return tasks_info_exp, agents_info_exp, mask_for_model, ltl_info_tensor, base_masks_dict, 'NO_ACTION_BY_DEFAULT', []
    #
    #     # If not blocked, proceed to apply LTL constraints (Pass 2)
    #     final_masks_dict = copy.deepcopy(base_masks_dict)
    #     blocking_clause_indices = []
    #
    #     if ltl_monitor and TrainParams.LTL_ENABLED:
    #         for i, clause in enumerate(ltl_monitor.clauses):
    #             is_blocking_this_step = False
    #             # Apply SAFETY constraint
    #             if clause.type == LTL_SAFETY and clause.state == FSA_SAFETY_SAFE and clause.param1 == agent_id:
    #                 if not final_masks_dict['destination'][clause.param2]:  # If it was previously allowed
    #                     final_masks_dict['destination'][clause.param2] = True
    #                     is_blocking_this_step = True
    #
    #             # Apply SEQUENTIAL constraint
    #             elif clause.type == LTL_SEQUENTIAL and clause.state == FSA_SEQ_INITIAL:
    #                 successor_dest_id = self.depots_num + clause.param2
    #                 if not final_masks_dict['destination'][successor_dest_id]:  # If it was previously allowed
    #                     final_masks_dict['destination'][successor_dest_id] = True
    #                     is_blocking_this_step = True
    #
    #             if is_blocking_this_step:
    #                 blocking_clause_indices.append(i)
    #
    #     if np.all(final_masks_dict['destination']):
    #         final_masks_dict['action_type'][self.ACTION_MOVE] = True
    #
    #     # Recalculate the mask_for_model based on the final destination mask
    #     final_mask_for_model = np.expand_dims(final_masks_dict['destination'], axis=0)
    #
    #     # --- CORRECTED FINAL BLOCKAGE CHECK ---
    #     # Check if agent is blocked AFTER LTL constraints are applied.
    #     move_allowed = not final_masks_dict['action_type'][self.ACTION_MOVE]
    #     load_allowed = not final_masks_dict['action_type'][self.ACTION_LOAD]
    #     unload_allowed = not final_masks_dict['action_type'][self.ACTION_UNLOAD]
    #     all_dests_masked = np.all(final_masks_dict['destination'])
    #
    #     is_ultimately_blocked = True
    #     if load_allowed:
    #         is_ultimately_blocked = False
    #     if unload_allowed:
    #         is_ultimately_blocked = False
    #     if move_allowed and not all_dests_masked:
    #         is_ultimately_blocked = False
    #
    #     # if is_ultimately_blocked:
    #     #     return tasks_info_exp, agents_info_exp, final_mask_for_model, ltl_info_tensor, final_masks_dict, 'NO_ACTION_BY_LTL', blocking_clause_indices
    #     #
    #     # # If we reach here, actions are available
    #     # return tasks_info_exp, agents_info_exp, final_mask_for_model, ltl_info_tensor, final_masks_dict, 'ACTIONS_AVAILABLE', []
    #
    #     if is_ultimately_blocked:
    #         # 如果被LTL阻塞，分析阻塞的原因
    #         is_blocked_by_safety_only = True
    #         if not blocking_clause_indices:  # 如果列表为空，则不是安全约束独占
    #             is_blocked_by_safety_only = False
    #         else:
    #             for clause_idx in blocking_clause_indices:
    #                 # 检查是否存在任何非安全性约束
    #                 if ltl_monitor.clauses[clause_idx].type != LTL_SAFETY:
    #                     is_blocked_by_safety_only = False
    #                     break
    #
    #         # 如果只是因为静态的安全性约束导致阻塞，返回一个特定的新原因
    #         if is_blocked_by_safety_only:
    #             return tasks_info_exp, agents_info_exp, final_mask_for_model, ltl_info_tensor, final_masks_dict, 'NO_ACTION_BY_SAFETY_LTL', blocking_clause_indices
    #         else:
    #             # 否则，使用原来的原因（现在特指由顺序性等动态约束导致）
    #             return tasks_info_exp, agents_info_exp, final_mask_for_model, ltl_info_tensor, final_masks_dict, 'NO_ACTION_BY_LTL', blocking_clause_indices
    #
    #         # 如果未被阻塞，则照常返回可用动作
    #     return tasks_info_exp, agents_info_exp, final_mask_for_model, ltl_info_tensor, final_masks_dict, 'ACTIONS_AVAILABLE', []

    # task_env.py

    def can_agent_contribute_to_unfinished_tasks(self, agent_id):
        """
        判断agent是否有能力对任何未完成任务做出贡献。
        返回True表示agent有能力贡献（即使当前无法执行动作），False表示永久无法贡献。
        """
        agent = self.agent_dic[agent_id]
        agent_capacity = agent["capacity"]

        # 收集所有未完成任务的需求
        all_unfinished_demands = np.zeros(self.traits_dim)
        for task in self.task_dic.values():
            if not task.get("finished", False):
                all_unfinished_demands += task.get("status", np.zeros(self.traits_dim))

        # 检查agent是否有能力运输任何一种未完成任务需要的cargo
        for cargo_type in range(self.traits_dim):
            if (
                all_unfinished_demands[cargo_type] > 0
                and agent_capacity[cargo_type] > 0
            ):
                # agent可以运输这种cargo，且有未完成任务需要它
                return True

        # agent的所有能力都无法匹配任何未完成任务的需求
        return False

    def agent_observe(
        self,
        agent_id,
        ltl_monitor=None,
        max_waiting=False,
        policy_logits=None,
        constraint_mode=None,
        ignore_sleeping=False,
    ):
        agent = self.agent_dic[agent_id]
        cost_info = {"total_cost": 0.0}

        mode = (
            constraint_mode
            if constraint_mode is not None
            else TrainParams.LTL_CONSTRAINT_TYPE
        )

        if ltl_monitor and TrainParams.LTL_ENABLED:
            # 根据编码类型选择不同的表征
            if TrainParams.LTL_ENCODING_TYPE == "A":
                # 方案A：ID-specific编码 [max_clauses, 4 + max_agents*3]
                ltl_info_tensor = ltl_monitor.get_sparse_state_tensor()
            elif TrainParams.LTL_ENCODING_TYPE == "B":
                # 方案B：Task feasibility编码 [max_agents, max_tasks]
                ltl_info_tensor = ltl_monitor.get_task_feasibility_matrix()
            elif TrainParams.LTL_ENCODING_TYPE == "C":
                # 方案C：Task feasibility + Dependency graph
                # 返回字典包含三个组件
                feasibility_matrix = ltl_monitor.get_task_feasibility_matrix()
                edge_index, edge_attr = ltl_monitor.get_dependency_graph_dynamic()
                ltl_info_tensor = {
                    "feasibility": feasibility_matrix,  # [max_agents, max_tasks]
                    "edge_index": edge_index,  # [2, E]
                    "edge_attr": edge_attr,  # [E, 1]
                }
            else:
                raise ValueError(
                    f"Unknown LTL_ENCODING_TYPE: {TrainParams.LTL_ENCODING_TYPE}"
                )
        else:
            # 无LTL约束时，返回零张量
            if TrainParams.LTL_ENCODING_TYPE == "A":
                from parameters import EnvParams

                # 计算张量维度：前4列是LTL约束，后max_agents*3列是智能体状态
                max_agents = (
                    EnvParams.SPECIES_RANGE[1] * EnvParams.SPECIES_AGENTS_RANGE[1]
                )
                tensor_width = 4 + max_agents * 3
                ltl_info_tensor = np.zeros(
                    (TrainParams.LTL_MAX_CLAUSES, tensor_width), dtype=np.float32
                )
            elif TrainParams.LTL_ENCODING_TYPE == "B":
                from parameters import EnvParams

                # 总agent数 = species数 × 每个species的agents数
                max_agents = (
                    EnvParams.SPECIES_RANGE[1] * EnvParams.SPECIES_AGENTS_RANGE[1]
                )
                max_tasks = EnvParams.TASKS_RANGE[1]
                ltl_info_tensor = np.zeros((max_agents, max_tasks), dtype=np.float32)
            elif TrainParams.LTL_ENCODING_TYPE == "C":
                from parameters import EnvParams

                max_agents = (
                    EnvParams.SPECIES_RANGE[1] * EnvParams.SPECIES_AGENTS_RANGE[1]
                )
                max_tasks = EnvParams.TASKS_RANGE[1]
                # 返回空的字典结构
                ltl_info_tensor = {
                    "feasibility": np.zeros((max_agents, max_tasks), dtype=np.float32),
                    "edge_index": np.zeros((2, 0), dtype=np.int64),
                    "edge_attr": np.zeros((0, 1), dtype=np.float32),
                }
            else:
                raise ValueError(
                    f"Unknown LTL_ENCODING_TYPE: {TrainParams.LTL_ENCODING_TYPE}"
                )

        self.last_policy_logits = policy_logits

        # 规则 0 (最高优先级): 休眠检查
        if agent.get("is_inactive", False):
            tasks_info = self.get_current_task_status(agent, ltl_monitor)
            agents_info = self.get_current_agent_status(agent)
            num_dest = self.depots_num + self.tasks_num
            destination_mask = np.ones(num_dest, dtype=bool)
            masks_dict = {
                "action_type": np.ones(3, dtype=bool),
                "cargo_to_load": np.ones(self.traits_dim, dtype=bool),
                "destination": destination_mask,
            }
            tasks_info_exp = np.expand_dims(tasks_info, axis=0)
            agents_info_exp = np.expand_dims(agents_info, axis=0)
            mask_for_model = np.expand_dims(destination_mask, axis=0)
            return (
                tasks_info_exp,
                agents_info_exp,
                mask_for_model,
                ltl_info_tensor,
                masks_dict,
                cost_info,
                "NO_ACTION_BY_DEFAULT",
                [],
            )

        if agent.get("is_temp_sleeping", False) and not ignore_sleeping:
            tasks_info = self.get_current_task_status(agent, ltl_monitor)
            agents_info = self.get_current_agent_status(agent)
            num_dest = self.depots_num + self.tasks_num
            destination_mask = np.ones(num_dest, dtype=bool)
            masks_dict = {
                "action_type": np.ones(3, dtype=bool),
                "cargo_to_load": np.ones(self.traits_dim, dtype=bool),
                "destination": destination_mask,
            }
            tasks_info_exp = np.expand_dims(tasks_info, axis=0)
            agents_info_exp = np.expand_dims(agents_info, axis=0)
            mask_for_model = np.expand_dims(destination_mask, axis=0)
            return (
                tasks_info_exp,
                agents_info_exp,
                mask_for_model,
                ltl_info_tensor,
                masks_dict,
                cost_info,
                "TEMP_SLEEPING",
                [],
            )

        effective_statuses = {
            tid: t["status"].copy()
            for tid, t in self.task_dic.items()
            if not t.get("finished")
        }
        self.last_effective_statuses = effective_statuses
        for other_agent in self.agent_dic.values():
            is_moving_to_task = (
                other_agent.get("next_decision") == np.inf
                and len(other_agent.get("route", [])) > 1
                and other_agent["route"][-1] >= 0
            )
            if is_moving_to_task and other_agent["inventory"]["quantity"] > 0:
                dest_task_id = other_agent["route"][-1]
                if dest_task_id in effective_statuses:
                    carried_type = other_agent["inventory"]["type"]
                    if carried_type is not None:
                        carried_qty = other_agent["inventory"]["quantity"]
                        needed_by_task = effective_statuses[dest_task_id][carried_type]
                        committed_qty = min(carried_qty, needed_by_task)
                        effective_statuses[dest_task_id][carried_type] -= committed_qty

        if agent["inventory"]["quantity"] == 0:
            raw_demand = np.sum(list(effective_statuses.values()), axis=0)
            if np.isscalar(raw_demand):
                raw_demand = np.zeros(self.traits_dim)
            can_be_useful = False
            for c_type, capacity in enumerate(agent["capacity"]):
                if capacity > 0 and raw_demand[c_type] > 0:
                    can_be_useful = True
                    break
            if not can_be_useful:
                agent["is_inactive"] = True
                agent["next_decision"] = np.inf
                if DEBUG:
                    print(
                        f"[INACTIVE] Agent {agent_id} set to inactive. No remaining tasks match its capabilities."
                    )
                return self.agent_observe(agent_id, ltl_monitor, max_waiting)

        # 规则 1: 游戏终局规则
        all_tasks_done = np.all(self.get_matrix(self.task_dic, "finished"))
        if all_tasks_done and not agent.get("returned", False):
            action_type_mask = np.array([False, True, True], dtype=bool)
            cargo_to_load_mask = np.ones(self.traits_dim, dtype=bool)
            destination_task_mask = np.ones(self.tasks_num, dtype=bool)
            destination_depot_mask = np.ones(self.depots_num, dtype=bool)
            home_depot_id = agent["species"] % self.depots_num
            is_at_home_depot = agent["current_task"] == -home_depot_id - 1
            if is_at_home_depot:
                agent["returned"] = True
                action_type_mask[self.ACTION_MOVE] = True
            else:
                destination_depot_mask[home_depot_id] = False
            destination_mask = np.concatenate(
                (destination_depot_mask, destination_task_mask)
            )
            masks_dict = {
                "action_type": action_type_mask,
                "cargo_to_load": cargo_to_load_mask,
                "destination": destination_mask,
            }
            tasks_info = self.get_current_task_status(agent, ltl_monitor)
            agents_info = self.get_current_agent_status(agent)
            tasks_info_exp = np.expand_dims(tasks_info, axis=0)
            agents_info_exp = np.expand_dims(agents_info, axis=0)
            mask_for_model = np.expand_dims(destination_mask, axis=0)
            return (
                tasks_info_exp,
                agents_info_exp,
                mask_for_model,
                ltl_info_tensor,
                masks_dict,
                cost_info,
                "ACTIONS_AVAILABLE",
                [],
            )

        # 规则 2: “必须卸货”规则
        is_at_task = agent["current_task"] >= 0
        is_carrying_cargo = agent["inventory"]["quantity"] > 0
        if is_at_task and is_carrying_cargo:
            task = self.task_dic[agent["current_task"]]
            carried_type = agent["inventory"]["type"]
            if carried_type is not None and task["status"][carried_type] > 0:
                action_type_mask = np.array([True, True, False], dtype=bool)
                destination_mask = np.ones(self.depots_num + self.tasks_num, dtype=bool)
                masks_dict = {
                    "action_type": action_type_mask,
                    "cargo_to_load": np.ones(self.traits_dim, dtype=bool),
                    "destination": destination_mask,
                }
                tasks_info = self.get_current_task_status(agent, ltl_monitor)
                agents_info = self.get_current_agent_status(agent)
                tasks_info_exp = np.expand_dims(tasks_info, axis=0)
                agents_info_exp = np.expand_dims(agents_info, axis=0)
                mask_for_model = np.expand_dims(destination_mask, axis=0)
                return (
                    tasks_info_exp,
                    agents_info_exp,
                    mask_for_model,
                    ltl_info_tensor,
                    masks_dict,
                    cost_info,
                    "ACTIONS_AVAILABLE",
                    [],
                )

        #################################################################
        ## MODIFICATION START: 核心逻辑修改区域
        #################################################################

        # === Pass 1: 计算基础掩码 (生成 global_useful_actions) ===
        action_type_mask = np.zeros(3, dtype=bool)
        cargo_to_load_mask = np.ones(self.traits_dim, dtype=bool)
        destination_depot_mask = np.ones(self.depots_num, dtype=bool)
        destination_task_mask = np.ones(self.tasks_num, dtype=bool)
        is_at_depot = not is_at_task

        if is_carrying_cargo:
            action_type_mask[self.ACTION_LOAD] = True  # 不能再装载
            carried_type = agent["inventory"]["type"]

            # 【修复】在判断货物有用性时考虑LTL约束
            valid_task_destinations = []
            if carried_type is not None:
                for task_id, effective_status_vec in effective_statuses.items():
                    if effective_status_vec[carried_type] > 0:
                        # 检查LTL约束是否会屏蔽这个任务
                        is_ltl_blocked = False
                        # 【修复】所有使用硬约束的模式都需要检查LTL（与决策掩码应用阶段保持一致）
                        if (
                            mode
                            in ["HARD", "SOFT_POLICY", "CVAR_SMDP", "LTL_POTENTIAL"]
                            and ltl_monitor
                            and TrainParams.LTL_ENABLED
                        ):
                            if TrainParams.LTL_ENCODING_TYPE == "A":
                                # 方案A：检查SAFETY和SEQUENTIAL约束
                                for clause in ltl_monitor.clauses:
                                    # SAFETY 的 param2 是节点索引（包含depots与tasks），任务节点应使用 depots_num + task_id
                                    if (
                                        clause.type == LTL_SAFETY
                                        and clause.param1 == agent_id
                                        and clause.param2 == (self.depots_num + task_id)
                                    ):
                                        is_ltl_blocked = True
                                        break
                                    elif (
                                        clause.type == LTL_SEQUENTIAL
                                        and clause.state == FSA_SEQ_INITIAL
                                        and clause.param2 == task_id
                                    ):
                                        is_ltl_blocked = True
                                        break
                            elif TrainParams.LTL_ENCODING_TYPE == "B":
                                # 方案B：检查feasibility矩阵
                                max_agents_in_matrix = ltl_info_tensor.shape[0]
                                if agent_id < max_agents_in_matrix:
                                    agent_feasibility = ltl_info_tensor[agent_id]
                                    if (
                                        task_id < len(agent_feasibility)
                                        and agent_feasibility[task_id] > 0.5
                                    ):
                                        is_ltl_blocked = True
                            elif TrainParams.LTL_ENCODING_TYPE == "C":
                                # 方案C：检查feasibility矩阵（从字典中提取）
                                feasibility_matrix = ltl_info_tensor["feasibility"]
                                max_agents_in_matrix = feasibility_matrix.shape[0]
                                if agent_id < max_agents_in_matrix:
                                    agent_feasibility = feasibility_matrix[agent_id]
                                    if (
                                        task_id < len(agent_feasibility)
                                        and agent_feasibility[task_id] > 0.5
                                    ):
                                        is_ltl_blocked = True

                        # 只有任务需求存在且LTL约束不屏蔽时，才是有效目的地
                        if not is_ltl_blocked:
                            valid_task_destinations.append(task_id)

            is_cargo_useful = len(valid_task_destinations) > 0

            if is_cargo_useful:
                # 允许前往所有需要该货物的任务点
                destination_task_mask[:] = True
                for task_id in valid_task_destinations:
                    destination_task_mask[task_id] = False

                # destination_depot_mask[:] = False
                ## 【关键修复】: 携带有用货物时，应该去任务点，而不是仓库！
                ## 只有在任务点时，才允许返回仓库（例如货物送达后返回）
                if is_at_depot:
                    # 在仓库时，是否允许带货跨仓由开关控制
                    from parameters import EnvParams

                    if EnvParams.RELAXED_CARRYING_POLICY:
                        destination_depot_mask[:] = False  # 放宽：允许跨仓
                    else:
                        destination_depot_mask[:] = True  # 严格：禁止跨仓
                else:
                    # 在任务点时，允许返回仓库（送货后）或去其他任务点
                    destination_depot_mask[:] = False

                # 【修复】在仓库时，如果货物有用，禁止UNLOAD（避免装卸循环）
                # 只有在任务点且任务需要该货物时，才允许UNLOAD
                from parameters import EnvParams

                if is_at_depot:
                    if not EnvParams.RELAXED_CARRYING_POLICY:
                        action_type_mask[self.ACTION_UNLOAD] = (
                            True  # 严格：禁止在仓库UNLOAD有用货物
                        )
                else:  # at task
                    # 检查当前任务是否需要携带的货物
                    current_task_id = agent["current_task"]
                    if current_task_id >= 0 and current_task_id in effective_statuses:
                        current_task_needs_cargo = (
                            effective_statuses[current_task_id][carried_type] > 0
                        )
                        if current_task_needs_cargo:
                            action_type_mask[self.ACTION_UNLOAD] = (
                                False  # 允许UNLOAD送货
                            )
                        else:
                            action_type_mask[self.ACTION_UNLOAD] = (
                                True  # 任务不需要，禁止UNLOAD
                            )
                    else:
                        action_type_mask[self.ACTION_UNLOAD] = (
                            True  # 任务不存在或已完成，禁止UNLOAD
                        )

            else:  # 货物对所有现存任务都无用，必须返回仓库
                destination_task_mask[:] = True  # 禁止去任何任务点
                destination_depot_mask[:] = False  # 必须去仓库

                if is_at_depot:
                    # 已经在仓库了，必须卸货，不能移动
                    action_type_mask[self.ACTION_MOVE] = True
                    action_type_mask[self.ACTION_UNLOAD] = False
                else:  # 在任务点，携带无用货物，必须移动回仓库，不能在任务点卸货
                    action_type_mask[self.ACTION_UNLOAD] = (
                        True  # 禁止在任务点UNLOAD无用货物
                    )
                    # MOVE被允许(默认值action_type_mask[MOVE]=False)

        else:  # Agent is empty
            action_type_mask[self.ACTION_UNLOAD] = True
            if is_at_task:
                # 在任务点，只能回仓库
                destination_depot_mask[:] = False
                destination_task_mask[:] = True
                action_type_mask[self.ACTION_LOAD] = True
            else:  # is_at_depot
                # 在仓库，可以装载有用货物，也可以去其他仓库
                global_demand = np.sum(list(effective_statuses.values()), axis=0)
                if np.isscalar(global_demand):
                    global_demand = np.zeros(self.traits_dim)

                can_load_anything_useful = False
                depot = self.depot_dic[-agent["current_task"] - 1]
                for c_type in range(self.traits_dim):
                    # 【修复】检查仓库库存、agent能力和全局需求
                    if (
                        depot["stock"].get(c_type, 0) > 0
                        and agent["capacity"][c_type] > 0
                        and global_demand[c_type] > 0
                    ):
                        # 【修复】进一步检查：是否存在未被LTL约束屏蔽的任务需要这种货物
                        has_reachable_demand = False
                        for task_id, effective_status_vec in effective_statuses.items():
                            if effective_status_vec[c_type] > 0:
                                # 检查LTL约束是否会屏蔽这个任务
                                is_ltl_blocked = False
                                # 【修复】所有使用硬约束的模式都需要检查LTL（与决策掩码应用阶段保持一致）
                                if (
                                    mode
                                    in [
                                        "HARD",
                                        "SOFT_POLICY",
                                        "CVAR_SMDP",
                                        "LTL_POTENTIAL",
                                    ]
                                    and ltl_monitor
                                    and TrainParams.LTL_ENABLED
                                ):
                                    if TrainParams.LTL_ENCODING_TYPE == "A":
                                        # 方案A：检查SAFETY和SEQUENTIAL约束
                                        for clause in ltl_monitor.clauses:
                                            # SAFETY 的 param2 是节点索引，任务节点应使用 depots_num + task_id
                                            if (
                                                clause.type == LTL_SAFETY
                                                and clause.param1 == agent_id
                                                and clause.param2
                                                == (self.depots_num + task_id)
                                            ):
                                                is_ltl_blocked = True
                                                break
                                            elif (
                                                clause.type == LTL_SEQUENTIAL
                                                and clause.state == FSA_SEQ_INITIAL
                                                and clause.param2 == task_id
                                            ):
                                                is_ltl_blocked = True
                                                break
                                    elif TrainParams.LTL_ENCODING_TYPE == "B":
                                        # 方案B：检查feasibility矩阵
                                        max_agents_in_matrix = ltl_info_tensor.shape[0]
                                        if agent_id < max_agents_in_matrix:
                                            agent_feasibility = ltl_info_tensor[
                                                agent_id
                                            ]
                                            if (
                                                task_id < len(agent_feasibility)
                                                and agent_feasibility[task_id] > 0.5
                                            ):
                                                is_ltl_blocked = True
                                    elif TrainParams.LTL_ENCODING_TYPE == "C":
                                        # 方案C：检查feasibility矩阵（从字典中提取）
                                        feasibility_matrix = ltl_info_tensor[
                                            "feasibility"
                                        ]
                                        max_agents_in_matrix = feasibility_matrix.shape[
                                            0
                                        ]
                                        if agent_id < max_agents_in_matrix:
                                            agent_feasibility = feasibility_matrix[
                                                agent_id
                                            ]
                                            if (
                                                task_id < len(agent_feasibility)
                                                and agent_feasibility[task_id] > 0.5
                                            ):
                                                is_ltl_blocked = True

                                # 如果找到至少一个未被LTL屏蔽的任务需要这种货物，则允许装载
                                if not is_ltl_blocked:
                                    has_reachable_demand = True
                                    break

                        # 只有存在可达的需求时，才允许装载这种货物类型
                        if has_reachable_demand:
                            cargo_to_load_mask[c_type] = False
                            can_load_anything_useful = True

                if can_load_anything_useful:
                    action_type_mask[self.ACTION_LOAD] = False

                    # 【优化】检查当前仓库是否有足够货物满足智能体能力
                    # 【修复】只考虑未被LTL约束屏蔽的需求
                    # 如果所有可达需求的货物类型在当前仓库的库存都>=智能体能力，
                    # 则限制移动到其他仓库（避免无意义的空载移动）
                    depot = self.depot_dic[-agent["current_task"] - 1]
                    all_cargo_sufficient = True
                    for c_type in range(self.traits_dim):
                        # 【修复】只检查未被LTL屏蔽的需求（即cargo_to_load_mask中未被屏蔽的类型）
                        if (
                            not cargo_to_load_mask[c_type]
                            and agent["capacity"][c_type] > 0
                        ):
                            # 如果这种货物类型可以装载（有可达需求），检查库存是否充足
                            if (
                                depot["stock"].get(c_type, 0)
                                < agent["capacity"][c_type]
                            ):
                                all_cargo_sufficient = False
                                break

                    if all_cargo_sufficient:
                        # 当前仓库货物充足，禁止空载移动到其他仓库
                        destination_depot_mask[:] = True  # 禁止所有仓库
                        # destination_task_mask 保持初始值True (空载时不能去任务点)
                    else:
                        # 当前仓库某些货物不足，允许移动到其他仓库
                        destination_depot_mask[:] = False
                        # destination_task_mask 保持初始值True (空载时不能直接去任务点)
                else:
                    # 仓库里没有任何有用的东西可装
                    action_type_mask[self.ACTION_LOAD] = True
                    # 只能去别的仓库看看
                    destination_depot_mask[:] = False
                    # destination_task_mask 保持初始值True (空载时不能去任务点)

        # 禁止移动到当前位置
        if is_at_depot:
            current_depot_action_id = -agent["current_task"] - 1
            if current_depot_action_id < self.depots_num:
                destination_depot_mask[current_depot_action_id] = True
        elif is_at_task:
            current_task_id = agent["current_task"]
            if current_task_id < self.tasks_num:
                destination_task_mask[current_task_id] = True

        destination_mask = np.concatenate(
            (destination_depot_mask, destination_task_mask)
        )
        if np.all(destination_mask):
            action_type_mask[self.ACTION_MOVE] = True

        # 保存基础掩码
        base_masks_dict = {
            "action_type": action_type_mask.copy(),
            "cargo_to_load": cargo_to_load_mask.copy(),
            "destination": destination_mask.copy(),
        }

        # === 检查全局是否有可用动作 ===
        base_move_possible = not base_masks_dict["action_type"][
            self.ACTION_MOVE
        ] and not np.all(base_masks_dict["destination"])
        base_load_possible = not base_masks_dict["action_type"][
            self.ACTION_LOAD
        ] and not np.all(base_masks_dict["cargo_to_load"])
        base_unload_possible = not base_masks_dict["action_type"][self.ACTION_UNLOAD]
        is_globally_stuck = not (
            base_move_possible or base_load_possible or base_unload_possible
        )

        if is_globally_stuck:
            # 当前没有可执行动作，但需要区分：永久失活 vs 临时休眠
            tasks_info = self.get_current_task_status(agent, ltl_monitor)
            agents_info = self.get_current_agent_status(agent)
            tasks_info_exp = np.expand_dims(tasks_info, axis=0)
            agents_info_exp = np.expand_dims(agents_info, axis=0)
            mask_for_model = np.expand_dims(base_masks_dict["destination"], axis=0)

            # 关键判断：agent是否有能力对未完成任务做出贡献
            can_contribute = self.can_agent_contribute_to_unfinished_tasks(agent_id)

            if can_contribute:
                # agent有能力贡献，但当前无法执行动作 -> 临时休眠，等待环境变化
                # 例如：等待depot补货，等待其他agent完成任务后产生新需求等
                return (
                    tasks_info_exp,
                    agents_info_exp,
                    mask_for_model,
                    ltl_info_tensor,
                    base_masks_dict,
                    cost_info,
                    "NO_ACTION_TEMPORARILY",
                    [],
                )
            else:
                # agent永久无法对任何未完成任务做出贡献 -> 永久失活
                return (
                    tasks_info_exp,
                    agents_info_exp,
                    mask_for_model,
                    ltl_info_tensor,
                    base_masks_dict,
                    cost_info,
                    "NO_ACTION_BY_DEFAULT",
                    [],
                )

        # === Pass 2: 应用LTL约束 (生成 local_executable_actions) ===
        final_masks_dict = copy.deepcopy(base_masks_dict)
        blocking_clause_indices = []

        is_ultimately_blocked = False
        # mode = TrainParams.LTL_CONSTRAINT_TYPE

        # 【关键修复】LTL_POTENTIAL模式也应该使用硬约束（动作屏蔽）
        # 区别只在于额外添加势能塑形奖励，掩码逻辑应该与HARD模式相同
        if mode in ["HARD", "SOFT_POLICY", "CVAR_SMDP", "LTL_POTENTIAL"]:
            if ltl_monitor and TrainParams.LTL_ENABLED:
                # 【修复】根据LTL编码类型选择不同的掩码应用方式
                if TrainParams.LTL_ENCODING_TYPE == "A":
                    # 方案A：使用clause的具体ID进行掩码
                    for i, clause in enumerate(ltl_monitor.clauses):
                        is_blocking_this_step = False
                        # 应用 SAFETY 约束
                        # 🔧 【修复】移除状态检查：safety约束应该永远禁止，不管当前状态
                        if clause.type == LTL_SAFETY and clause.param1 == agent_id:
                            if not final_masks_dict["destination"][clause.param2]:
                                final_masks_dict["destination"][clause.param2] = True
                                is_blocking_this_step = True

                        # 应用 SEQUENTIAL 约束
                        # 注意：sequential约束的状态检查是正确的（只在INITIAL状态禁止）
                        elif (
                            clause.type == LTL_SEQUENTIAL
                            and clause.state == FSA_SEQ_INITIAL
                        ):
                            successor_dest_id = self.depots_num + clause.param2
                            if not final_masks_dict["destination"][successor_dest_id]:
                                final_masks_dict["destination"][successor_dest_id] = (
                                    True
                                )
                                is_blocking_this_step = True

                        if is_blocking_this_step:
                            blocking_clause_indices.append(i)
                elif TrainParams.LTL_ENCODING_TYPE == "B":
                    # 方案B：使用Task feasibility矩阵进行掩码
                    # ltl_info_tensor的shape: [max_agents, max_tasks]
                    # 值: 0=可行，1=不可行

                    # 【调试】打印矩阵形状和内容
                    if DEBUG:
                        print(f"\n[LTL MASK DEBUG] Agent {agent_id}:")
                        print(f"  ltl_info_tensor shape: {ltl_info_tensor.shape}")
                        print(f"  agent_id: {agent_id}, tasks_num: {self.tasks_num}")

                    # 边界检查：确保agent_id在矩阵范围内
                    max_agents_in_matrix = ltl_info_tensor.shape[0]
                    if agent_id >= max_agents_in_matrix:
                        if DEBUG:
                            print(
                                f"[LTL MASK WARNING] agent_id {agent_id} >= max_agents {max_agents_in_matrix}, using last row"
                            )
                        agent_feasibility = ltl_info_tensor[max_agents_in_matrix - 1]
                    else:
                        agent_feasibility = ltl_info_tensor[agent_id]  # [max_tasks]

                    if DEBUG:
                        print(
                            f"  agent_feasibility: {agent_feasibility[: self.tasks_num]}"
                        )

                    # 应用到destination_mask的任务部分（depot部分不受影响）
                    # destination_mask: [depots_num + tasks_num]
                    # 任务部分从depots_num开始
                    for task_id in range(self.tasks_num):
                        if task_id < len(agent_feasibility):
                            # 如果feasibility=1（不可行），则屏蔽该目的地
                            if agent_feasibility[task_id] > 0.5:  # 使用阈值避免浮点误差
                                dest_idx = self.depots_num + task_id
                                if not final_masks_dict["destination"][dest_idx]:
                                    final_masks_dict["destination"][dest_idx] = True
                                    if DEBUG:
                                        print(
                                            f"    Masking destination {dest_idx} (Task {task_id})"
                                        )
                                    # 记录所有阻塞的clause（对于方案B，我们记录所有约束）
                                    for i, clause in enumerate(ltl_monitor.clauses):
                                        if i not in blocking_clause_indices:
                                            blocking_clause_indices.append(i)
                elif TrainParams.LTL_ENCODING_TYPE == "C":
                    # 方案C：使用Task feasibility矩阵进行掩码（与B相同）
                    # edge_index和edge_attr仅用于GNN特征传播，不影响掩码
                    # ltl_info_tensor是字典: {'feasibility': [max_agents, max_tasks], ...}
                    feasibility_matrix = ltl_info_tensor["feasibility"]

                    # 【调试】打印矩阵形状和内容
                    if DEBUG:
                        print(f"\n[LTL MASK DEBUG Mode C] Agent {agent_id}:")
                        print(f"  feasibility_matrix shape: {feasibility_matrix.shape}")
                        print(
                            f"  edge_index shape: {ltl_info_tensor['edge_index'].shape}"
                        )
                        print(
                            f"  edge_attr shape: {ltl_info_tensor['edge_attr'].shape}"
                        )

                    # 边界检查：确保agent_id在矩阵范围内
                    max_agents_in_matrix = feasibility_matrix.shape[0]
                    if agent_id >= max_agents_in_matrix:
                        if DEBUG:
                            print(
                                f"[LTL MASK WARNING] agent_id {agent_id} >= max_agents {max_agents_in_matrix}, using last row"
                            )
                        agent_feasibility = feasibility_matrix[max_agents_in_matrix - 1]
                    else:
                        agent_feasibility = feasibility_matrix[agent_id]  # [max_tasks]

                    if DEBUG:
                        print(
                            f"  agent_feasibility: {agent_feasibility[: self.tasks_num]}"
                        )

                    # 应用到destination_mask的任务部分（与模式B相同）
                    for task_id in range(self.tasks_num):
                        if task_id < len(agent_feasibility):
                            # 如果feasibility=1（不可行），则屏蔽该目的地
                            if agent_feasibility[task_id] > 0.5:
                                dest_idx = self.depots_num + task_id
                                if not final_masks_dict["destination"][dest_idx]:
                                    final_masks_dict["destination"][dest_idx] = True
                                    if DEBUG:
                                        print(
                                            f"    Masking destination {dest_idx} (Task {task_id})"
                                        )
                                    # 记录所有阻塞的clause
                                    for i, clause in enumerate(ltl_monitor.clauses):
                                        if i not in blocking_clause_indices:
                                            blocking_clause_indices.append(i)
                else:
                    raise ValueError(
                        f"Unknown LTL_ENCODING_TYPE: {TrainParams.LTL_ENCODING_TYPE}"
                    )

            # 如果所有目的地都被屏蔽，则MOVE动作也应被屏蔽
            if np.all(final_masks_dict["destination"]):
                final_masks_dict["action_type"][self.ACTION_MOVE] = True
        else:
            # 对于软约束, 我们不修改 final_masks_dict (不进行硬屏蔽)
            # 我们只计算成本
            if ltl_monitor and TrainParams.LTL_ENABLED:
                if mode == "SOFT_DISCRETE":
                    cost_info = self._calculate_discrete_cost(
                        agent_id, final_masks_dict, ltl_monitor
                    )
                elif mode == "SOFT_HYBRID_STATE":
                    cost_info = self._calculate_hybrid_state_cost(
                        agent_id, policy_logits, ltl_monitor
                    )
                # SOFT_POLICY 模式不再在此处计算 LTL 成本，因为它是基于 Hazard 的软约束，且 LTL 采用硬约束。

        # === 【关键修复】处理LTL约束导致的僵局 ===
        # 如果Agent在仓库携带有用货物，但所有有用的任务点都被LTL禁止，允许UNLOAD
        is_at_depot = agent["current_task"] < 0
        is_carrying = agent["inventory"]["quantity"] > 0

        if is_at_depot and is_carrying and np.all(final_masks_dict["destination"]):
            # 所有destination都被禁止，检查是否是因为LTL约束
            # 如果base_masks允许去某些任务点，但final_masks全部禁止，说明是LTL导致的
            base_has_valid_tasks = not np.all(
                base_masks_dict["destination"][self.depots_num :]
            )
            if base_has_valid_tasks:
                # 货物对某些任务有用，但这些任务都被LTL禁止
                # 允许在仓库UNLOAD，避免僵局
                final_masks_dict["action_type"][self.ACTION_UNLOAD] = False
                if DEBUG:
                    print(
                        f"[LTL DEADLOCK FIX] Agent {agent_id} at depot with useful cargo, but all tasks blocked by LTL. Allowing UNLOAD."
                    )

        # === 比较LTL过滤前后的动作集 ===
        final_move_possible = not final_masks_dict["action_type"][
            self.ACTION_MOVE
        ] and not np.all(final_masks_dict["destination"])
        final_load_possible = not final_masks_dict["action_type"][
            self.ACTION_LOAD
        ] and not np.all(final_masks_dict["cargo_to_load"])
        final_unload_possible = not final_masks_dict["action_type"][self.ACTION_UNLOAD]
        is_locally_stuck = not (
            final_move_possible or final_load_possible or final_unload_possible
        )

        tasks_info = self.get_current_task_status(agent, ltl_monitor)
        agents_info = self.get_current_agent_status(agent)
        tasks_info_exp = np.expand_dims(tasks_info, axis=0)
        agents_info_exp = np.expand_dims(agents_info, axis=0)
        final_mask_for_model = np.expand_dims(final_masks_dict["destination"], axis=0)

        if not is_locally_stuck:
            # LTL过滤后仍有事可做
            return (
                tasks_info_exp,
                agents_info_exp,
                final_mask_for_model,
                ltl_info_tensor,
                final_masks_dict,
                cost_info,
                "ACTIONS_AVAILABLE",
                [],
            )
        else:
            # LTL过滤后无事可做（但全局有事可做），这是需要休眠的信号
            # 同时分析阻塞原因
            is_blocked_by_safety_only = True
            if not blocking_clause_indices:
                is_blocked_by_safety_only = False
            else:
                for clause_idx in blocking_clause_indices:
                    if ltl_monitor.clauses[clause_idx].type != LTL_SAFETY:
                        is_blocked_by_safety_only = False
                        break

            # if is_blocked_by_safety_only:
            #     return tasks_info_exp, agents_info_exp, final_mask_for_model, ltl_info_tensor, final_masks_dict, 'NO_ACTION_BY_SAFETY_LTL', blocking_clause_indices
            # else:
            #     return tasks_info_exp, agents_info_exp, final_mask_for_model, ltl_info_tensor, final_masks_dict, 'NO_ACTION_BY_LTL', blocking_clause_indices

            if is_blocked_by_safety_only:
                inaction_reason = "NO_ACTION_BY_SAFETY_LTL"
            else:
                inaction_reason = "NO_ACTION_BY_LTL"

        return (
            tasks_info_exp,
            agents_info_exp,
            final_mask_for_model,
            ltl_info_tensor,
            final_masks_dict,
            cost_info,
            inaction_reason,
            blocking_clause_indices,
        )

    def get_total_hazard_normalized(self):
        """
        Pure LTL版本：故障建模已移除，返回0.0
        保留此方法以兼容worker.py中的调用
        """
        return 0.0

    def calculate_waiting_time(self):
        for agent in self.agent_dic.values():
            agent["sum_waiting_time"] = 0
        for task in self.task_dic.values():
            task["sum_waiting_time"] = 0
            if task.get("finished"):
                for member_id in task.get("members", []):
                    arrival = self.get_arrival_time(member_id, task["ID"])
                    time_spent = task["time_finish"] - arrival
                    if "sum_waiting_time" in self.agent_dic[member_id]:
                        self.agent_dic[member_id]["sum_waiting_time"] += time_spent
                    task["sum_waiting_time"] += time_spent

    def check_finished(self):
        self.task_update()
        all_tasks_done = np.all(self.get_matrix(self.task_dic, "finished"))
        return all_tasks_done
        # if not all_tasks_done:
        #     return False
        #
        # all_agents_returned = np.all(self.get_matrix(self.agent_dic, 'returned'))
        #
        # return all_agents_returned

    def generate_traj(self):
        for agent in self.agent_dic.values():
            agent["trajectory"] = []
            angle = 0
            start_loc = agent["depot"]
            agent["trajectory"].append(np.hstack([start_loc, angle]))
            last_loc = start_loc
            last_time = 0.0

            for i in range(1, len(agent["route"])):
                route_id = agent["route"][i]
                arrival_time = agent["arrival_time"][i]

                if route_id >= 0:
                    dest_node = self.task_dic[route_id]
                else:
                    dest_node = self.depot_dic[-route_id - 1]

                dest_loc = dest_node["location"]
                travel_duration = arrival_time - last_time

                if travel_duration <= 0:
                    last_time = arrival_time
                    continue

                angle = np.arctan2(dest_loc[1] - last_loc[1], dest_loc[0] - last_loc[0])

                num_steps = int(travel_duration / self.dt)
                if num_steps == 0:
                    num_steps = 1

                for step in range(1, num_steps + 1):
                    fraction = step / num_steps
                    x = last_loc[0] + fraction * (dest_loc[0] - last_loc[0])
                    y = last_loc[1] + fraction * (dest_loc[1] - last_loc[1])
                    agent["trajectory"].append(np.hstack([x, y, angle]))

                last_loc = dest_loc
                last_time = arrival_time

            if agent["trajectory"]:
                final_pos = agent["trajectory"][-1]
                final_time = len(agent["trajectory"]) * self.dt
                while final_time < self.current_time + self.dt:
                    agent["trajectory"].append(final_pos)
                    final_time += self.dt

    def get_episode_reward(self, max_time=200):
        self.calculate_waiting_time()
        finished_tasks = self.get_matrix(self.task_dic, "finished")
        dist = np.sum(self.get_matrix(self.agent_dic, "travel_dist"))
        reward = -self.current_time if self.finished else -max_time
        return reward, finished_tasks

    def get_efficiency(self):
        total_required = np.sum(
            [np.sum(t["requirements"]) for t in self.task_dic.values()]
        )
        if total_required == 0:
            return 1.0
        work_done = np.sum(
            [
                np.sum(np.maximum(0, t["requirements"] - t["status"]))
                for t in self.task_dic.values()
            ]
        )
        return work_done / total_required

    def stack_trajectory(self):
        max_len = 0
        for agent in self.agent_dic.values():
            if agent.get("trajectory") and len(agent["trajectory"]) > max_len:
                max_len = len(agent["trajectory"])

        if max_len == 0:
            return

        for agent in self.agent_dic.values():
            if not agent.get("trajectory"):
                agent["trajectory"] = [np.hstack([agent["depot"], 0.0])] * max_len

            if len(agent["trajectory"]) < max_len:
                last_pos = agent["trajectory"][-1]
                padding = [last_pos] * (max_len - len(agent["trajectory"]))
                agent["trajectory"].extend(padding)

            agent["trajectory"] = np.vstack(agent["trajectory"])

    def plot_animation(self, path, n):
        self.generate_traj()
        self.stack_trajectory()

        plot_robot_icon = False
        if plot_robot_icon:
            # drone = plt.imread('env/drone.png')
            # drone_oi = OffsetImage(drone, zoom=0.05)
            pass

        def get_cmap(n, name="Dark2"):
            return plt.cm.get_cmap(name, n)

        cmap = get_cmap(self.species_num)
        finished_tasks = self.get_matrix(self.task_dic, "finished")
        finished_rate = (
            np.sum(finished_tasks) / len(finished_tasks)
            if len(finished_tasks) > 0
            else 1.0
        )

        if self.agents_num == 0 or not any(
            a.get("trajectory") and a["trajectory"].size > 0
            for a in self.agent_dic.values()
        ):
            print("Cannot generate animation: No agents or no trajectory data.")
            return

        gif_len = self.agent_dic[0]["trajectory"].shape[0]
        if gif_len == 0:
            print("Cannot generate animation: Trajectory length is zero.")
            return

        fig, ax = plt.subplots(dpi=100)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        plt.subplots_adjust(left=0, right=0.85, top=0.87, bottom=0.02)
        lines = [
            ax.plot([], [], color=cmap(a["species"]), zorder=0)[0]
            for a in self.agent_dic.values()
        ]
        ax.set_title(
            f"Agents finish {finished_rate * 100:.0f}% tasks within {self.current_time:.2f}s."
            f"\nCurrent time is {0:.2f}s"
        )
        color_map = [
            patches.Patch(color=cmap(i), label="Agent species " + str(i))
            for i in range(self.species_num)
        ]
        color_map.append(patches.Patch(color="g", label="Finished task"))
        color_map.append(patches.Patch(color="b", label="Unfinished task"))
        color_map.append(
            patches.Patch(facecolor="b", edgecolor="y", hatch="/", label="Locked Task")
        )

        ax.legend(handles=color_map, bbox_to_anchor=(0.99, 0.7))

        task_squares = [
            ax.add_patch(
                patches.RegularPolygon(
                    xy=t["location"],
                    numVertices=int(np.sum(t["requirements"])) + 3,
                    radius=0.03,
                    color="b",
                )
            )
            for t in self.task_dic.values()
        ]
        depot_circles = [
            ax.add_patch(patches.Circle(d["location"], 0.02, color="r"))
            for d in self.depot_dic.values()
        ]
        agent_triangles = [
            ax.add_patch(
                patches.RegularPolygon(
                    a["depot"], numVertices=3, radius=0.02, color=cmap(a["species"])
                )
            )
            for a in self.agent_dic.values()
        ]

        def update(frame):
            time_now = frame * self.dt
            ax.set_title(
                f"Agents finish {finished_rate * 100:.0f}% tasks within {self.current_time:.2f}s.\nCurrent time: {time_now:.2f}s"
            )
            for agent in self.agent_dic.values():
                if frame < agent["trajectory"].shape[0]:
                    pos = agent["trajectory"][frame]
                    agent_triangles[agent["ID"]].xy = tuple(pos[0:2])
                    agent_triangles[agent["ID"]].orientation = pos[2] - np.pi / 2

                    start_frame = max(0, frame - 40)
                    lines[agent["ID"]].set_data(
                        agent["trajectory"][start_frame : frame + 1, 0],
                        agent["trajectory"][start_frame : frame + 1, 1],
                    )

            for task in self.task_dic.values():
                if task.get("time_finish", 0) > 0 and time_now >= task["time_finish"]:
                    task_squares[task["ID"]].set_color("g")
                    task_squares[task["ID"]].set_hatch(None)
                elif task.get("feasible_assignment"):
                    task_squares[task["ID"]].set_hatch("/")
                    task_squares[task["ID"]].set_edgecolor("y")
            return lines + agent_triangles + task_squares

        ani = FuncAnimation(fig, update, frames=gif_len, interval=50, blit=False)
        ani.save(f"{path}/episode_{n}_{self.current_time:.1f}.gif", writer="pillow")
        plt.close(fig)
