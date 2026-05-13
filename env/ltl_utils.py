# ltl_utils.py

import math
import numpy as np
from collections import deque
from parameters import TrainParams

# --- LTL Constants ---
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


class LTLClause:
    """Represents a single LTL clause with its type, parameters, and current state."""

    def __init__(self, clause_type, param1, param2):
        self.type = clause_type
        self.param1 = int(param1)
        self.param2 = int(param2)
        self.state = FSA_SAFETY_SAFE if self.type == LTL_SAFETY else FSA_SEQ_INITIAL

    def __repr__(self):
        if self.type == LTL_SAFETY:
            # Note: param2 is a node_id (0..depots+tasks-1)
            return f"LTL_SAFETY(Agent {self.param1} cannot visit Node {self.param2})"
        elif self.type == LTL_SEQUENTIAL:
            # Note: param1 and param2 are task_ids (0..tasks-1)
            return f"LTL_SEQUENTIAL(Task {self.param1} must precede Task {self.param2})"
        return "Unknown LTL Clause"


class LTLMonitor:
    """
    Generates, validates, and monitors LTL constraints for an environment episode.
    """

    # def __init__(self, env):
    #     self.env = env
    #     self.clauses = []
    #     # Ensure the environment's seeded random number generator is used for reproducibility
    #     self.rng = env.rng if env.rng is not None else np.random.default_rng()
    #
    #     if not TrainParams.LTL_ENABLED:
    #         return
    #
    #     self._generate_clauses()

    def __init__(self, env, fixed_clauses=None):
        """
        LTL 监视器.
        :param env: TaskEnv 实例.
        :param fixed_clauses: (可选) 一个包含LTL条款元组的列表,
                               格式为 [(type, param1, param2), ...]。
                               如果提供, 则使用这些固定的条款; 否则随机生成。
        """
        self.env = env
        self.clauses = []
        self.rng = env.rng if env.rng is not None else np.random.default_rng()
        self._topo_E0 = None

        if fixed_clauses is not None:
            # 【修复】如果提供了固定的条款，需要验证任务ID是否在当前环境中存在
            valid_task_ids = set(self.env.task_dic.keys())
            num_agents = self.env.agents_num
            num_nodes = self.env.depots_num + self.env.tasks_num
            
            filtered_count = 0
            for c_type, p1, p2 in fixed_clauses:
                # 验证约束的有效性
                if c_type == LTL_SEQUENTIAL:
                    # 顺序约束：两个任务ID都必须存在
                    if p1 in valid_task_ids and p2 in valid_task_ids:
                        self.clauses.append(LTLClause(c_type, p1, p2))
                    else:
                        filtered_count += 1
                        print(f"[LTL FILTER] 跳过无效顺序约束: Task {p1} → Task {p2} (环境只有任务 {min(valid_task_ids)}-{max(valid_task_ids)})")
                elif c_type == LTL_SAFETY:
                    # 安全约束：agent ID和node ID都必须有效
                    if p1 < num_agents and p2 < num_nodes:
                        self.clauses.append(LTLClause(c_type, p1, p2))
                    else:
                        filtered_count += 1
                        print(f"[LTL FILTER] 跳过无效安全约束: Agent {p1} cannot visit Node {p2} (环境有 {num_agents} agents, {num_nodes} nodes)")
            
            if filtered_count > 0:
                print(f"[LTL FILTER] ⚠️  过滤掉 {filtered_count} 个无效约束，保留 {len(self.clauses)} 个有效约束")
            return

        self._generate_clauses()

    # ==================== MODIFICATION 3: ENHANCED STATE MACHINE UPDATES ====================
    def update_monitor(self, event_type, params):
        """
        Updates the state of all clauses based on an environment event.
        This version handles a more complete set of state transitions.
        """
        if not self.clauses:
            return

        # Handle events related to task completion for SEQUENTIAL clauses
        if event_type == 'task_finish':
            finished_task_id = params.get('task_id')
            if finished_task_id is None:
                return

            for clause in self.clauses:
                if clause.type == LTL_SEQUENTIAL:
                    # Transition from INITIAL to PREDECESSOR_DONE (正常路径)
                    if clause.state == FSA_SEQ_INITIAL and clause.param1 == finished_task_id:
                        clause.state = FSA_SEQ_PREDECESSOR_DONE
                    # Transition from INITIAL to VIOLATED (违规路径：后继任务在前置任务之前完成)
                    elif clause.state == FSA_SEQ_INITIAL and clause.param2 == finished_task_id:
                        clause.state = FSA_SEQ_VIOLATED
                    # Transition from PREDECESSOR_DONE to SATISFIED (正常路径)
                    elif clause.state == FSA_SEQ_PREDECESSOR_DONE and clause.param2 == finished_task_id:
                        clause.state = FSA_SEQ_SATISFIED
                    # VIOLATED和SATISFIED是吸收态，不再转移

        # Handle events related to agent movement for SAFETY clauses
        elif event_type == 'agent_move':
            agent_id = params.get('agent_id')
            destination_node_id = params.get('destination_node_id')
            if agent_id is None or destination_node_id is None:
                return

            for clause in self.clauses:
                if clause.type == LTL_SAFETY and clause.state == FSA_SAFETY_SAFE:
                    if clause.param1 == agent_id and clause.param2 == destination_node_id:
                        clause.state = FSA_SAFETY_VIOLATED

    # ====================================================================================

    # --- Feasibility Checking Logic ---

    def _is_cyclic_util(self, u, dependency_graph, visiting, visited):
        """Recursive helper for cycle detection using DFS."""
        visiting.add(u)
        visited.add(u)
        for v in dependency_graph.get(u, []):
            if v in visiting:
                return True  # Cycle detected
            if v not in visited:
                if self._is_cyclic_util(v, dependency_graph, visiting, visited):
                    return True
        visiting.remove(u)
        return False

    def _is_cyclic(self, dependency_graph):
        """Checks if a directed graph contains a cycle."""
        visited = set()
        for node in dependency_graph:
            if node not in visited:
                if self._is_cyclic_util(node, dependency_graph, set(), visited):
                    return True
        return False

    # ==================== MODIFICATION 1: NEW GLOBAL FEASIBILITY CHECKER ====================
    def _is_globally_feasible(self, all_clauses):
        """
        Performs a holistic check on a given set of clauses to ensure no task becomes impossible.
        Focuses on ensuring at least one valid supply chain exists for every task requirement.
        """
        env = self.env

        # Step 1: Build a quick-lookup map of all safety restrictions from the clauses
        forbidden_map = {i: set() for i in range(env.agents_num)}
        for clause in all_clauses:
            if clause.type == LTL_SAFETY:
                agent_id = clause.param1
                node_id = clause.param2  # This is the destination_id (0..depots+tasks-1)
                forbidden_map[agent_id].add(node_id)

        # Step 2: Iterate through every requirement of every unfinished task
        for task_id, task in env.task_dic.items():
            if task.get('finished', False):
                continue

            task_destination_id = env.depots_num + task_id  # Convert task_id to its corresponding destination_id

            required_cargo_types = np.where(task['requirements'] > 0)[0]
            for cargo_type in required_cargo_types:
                # For this specific requirement (task_id, cargo_type), we need to find if
                # AT LEAST ONE supply chain is still viable.
                is_requirement_serviceable = False

                # Step 3: Find all potential suppliers (agents with capability, depots with stock)
                capable_agents = [a_id for a_id, agent in env.agent_dic.items() if agent['capacity'][cargo_type] > 0]
                stocked_depots = [d_id for d_id, depot in env.depot_dic.items() if
                                  depot['stock'].get(cargo_type, 0) > 0]

                # Step 4: Check for at least one valid (Agent, Depot) pair that can service the task
                for agent_id in capable_agents:
                    # Condition A: Agent must be allowed to visit the destination task
                    if task_destination_id in forbidden_map[agent_id]:
                        continue  # This agent can't deliver, so try the next one

                    for depot_id in stocked_depots:
                        # Depot ID is the same as its destination_id for nodes 0..depots_num-1
                        depot_destination_id = depot_id

                        # Condition B: Agent must be allowed to visit the source depot
                        if depot_destination_id not in forbidden_map[agent_id]:
                            # We found a valid supply chain!
                            # Agent `agent_id` can pick up `cargo_type` from Depot `depot_id`
                            # and deliver it to Task `task_id`.
                            is_requirement_serviceable = True
                            break  # Found a valid depot, no need to check other depots for this agent

                    if is_requirement_serviceable:
                        break  # Found a valid agent, no need to check other agents for this requirement

                # Step 5: If after checking all possibilities, no supply chain was found...
                if not is_requirement_serviceable:
                    # This set of clauses makes this specific requirement impossible to fulfill.
                    # The entire set is therefore infeasible.
                    return False

        # If we get through all tasks and all requirements without returning False, the set of clauses is feasible.
        return True

    # ====================================================================================

    # --- Main Generation and Tensorization Logic ---

    def _generate_clauses(self):
        """Generates a set of random, but feasible, LTL clauses."""
        # 根据配置选择生成模式
        if TrainParams.LTL_NUM_CLAUSES_MODE == 'fixed':
            # 固定数量模式：总数固定为 LTL_MAX_CLAUSES
            total_clauses = TrainParams.LTL_MAX_CLAUSES
            if getattr(TrainParams, 'LTL_FIXED_SPLIT_RANDOM', False):
                if total_clauses > 0:
                    # 随机拆分 safety / sequential，但总数保持不变
                    num_safety = int(self.rng.integers(0, total_clauses + 1))
                    num_sequential = int(total_clauses - num_safety)
                else:
                    num_safety = 0
                    num_sequential = 0

                # 任务数不足时无法生成顺序性约束：退化为全 safety
                if len(self.env.task_dic) < 2:
                    num_safety = total_clauses
                    num_sequential = 0

                # 无智能体时无法生成 safety：退化为全 sequential（若可生成）
                if self.env.agents_num == 0:
                    num_safety = 0
                    num_sequential = total_clauses if len(self.env.task_dic) >= 2 else 0
            else:
                # 固定数量模式：分别指定安全性和顺序性约束的数量
                num_safety = TrainParams.LTL_FIXED_NUM_SAFETY
                num_sequential = TrainParams.LTL_FIXED_NUM_SEQUENTIAL
        else:
            # 随机数量模式（默认）：在0到LTL_MAX_CLAUSES之间随机
            total_clauses = self.rng.integers(0, TrainParams.LTL_MAX_CLAUSES + 1)
            # 随机模式下，随机分配安全性和顺序性约束的比例
            if len(self.env.task_dic) >= 2 and total_clauses > 0:
                # 优先生成顺序性约束（60%概率）
                num_sequential = self.rng.binomial(total_clauses, 0.6)
                num_safety = total_clauses - num_sequential
            else:
                # 任务数少于2时，全部生成安全性约束
                num_safety = total_clauses
                num_sequential = 0

        if num_safety + num_sequential == 0:
            return

        dependency_graph = {task_id: [] for task_id in self.env.task_dic}

        # ==================== 生成顺序性约束 (SEQUENTIAL) ====================
        sequential_generated = 0
        for _ in range(num_sequential):
            if len(self.env.task_dic) < 2:
                break  # 任务数不足，无法生成顺序性约束

            for _ in range(TrainParams.LTL_GEN_MAX_RETRIES):
                task_ids = self.rng.choice(list(self.env.task_dic.keys()), 2, replace=False)
                p1, p2 = task_ids[0], task_ids[1]
                dependency_graph[p1].append(p2)
                if not self._is_cyclic(dependency_graph):
                    self.clauses.append(LTLClause(LTL_SEQUENTIAL, p1, p2))
                    sequential_generated += 1
                    break
                else:
                    dependency_graph[p1].pop()  # Backtrack

        # ==================== 生成安全性约束 (SAFETY) ====================
        safety_generated = 0
        for _ in range(num_safety):
            if self.env.agents_num == 0:
                break  # 无智能体，无法生成安全性约束

            for _ in range(TrainParams.LTL_GEN_MAX_RETRIES):
                # Step 1: Propose a candidate clause
                p1_agent = self.rng.choice(self.env.agents_num)
                # p2_node = self.rng.choice(self.env.depots_num + self.env.tasks_num)
                p2_node = self.rng.integers(low=self.env.depots_num, high=self.env.depots_num + self.env.tasks_num)
                candidate_clause = LTLClause(LTL_SAFETY, p1_agent, p2_node)

                # Step 2: Create a temporary list including the new candidate
                temp_clauses = self.clauses + [candidate_clause]

                # Step 3: Run the new, comprehensive feasibility check on the entire set
                if self._is_globally_feasible(temp_clauses):
                    # Step 4: If it passes, permanently add the clause and break the retry loop
                    self.clauses.append(candidate_clause)
                    safety_generated += 1
                    break
        # ================================================================================

    def get_sparse_state_tensor(self):
        """
        【改进的稀疏编码 + 智能体状态】生成紧凑且完备的LTL状态表示

        理论基础：
        - 添加类型one-hot编码（消除歧义，确保完备性）
        - 归一化所有参数到[0,1]（使用训练范围上界max值，确保跨episode一致性）
        - 支持扩展的FSA状态（包括VIOLATED）
        - 添加智能体三状态信息（活跃/临时休眠/失活）
        - 新维度：[max_clauses, 4 + max_agents*3]

        Returns:
            np.ndarray: [max_clauses, 4 + max_agents*3] 稀疏张量
                前4列: [type_onehot, param1_normalized, param2_normalized, state_normalized]
                后max_agents*3列: 每个智能体的三状态one-hot编码

        编码格式（LTL约束部分，前4列）：
            [0]: type (0=SAFETY, 1=SEQUENTIAL)
            [1]: param1_normalized ∈ [0,1]
            [2]: param2_normalized ∈ [0,1]
            [3]: state_normalized ∈ [0,1]

        编码格式（智能体状态部分，后max_agents*3列）：
            对于第i个智能体（i=0..max_agents-1）：
            [4 + i*3 + 0]: active（活跃）= 1 if agent is active else 0
            [4 + i*3 + 1]: temporary_sleeping（临时休眠）= 1 if agent is temp_sleeping else 0
            [4 + i*3 + 2]: inactive（失活）= 1 if agent is inactive else 0

        状态归一化（LTL约束）：
            SAFETY: SAFE=0 → 0.0, VIOLATED=1 → 1.0
            SEQUENTIAL: INITIAL=0 → 0.0, PREDECESSOR_DONE=1 → 0.33,
                       SATISFIED=2 → 0.67, VIOLATED=3 → 1.0

        归一化基准（修复）：
            使用训练范围的上界（max_agents, max_tasks）而非当前值（current_agents, current_tasks）
            确保不同episode之间的特征尺度一致，提升泛化能力

        复杂度：O(C + A)，线性于约束数量和智能体数量
        """
        from parameters import EnvParams

        # 使用最大智能体数和最大任务数（训练时范围的上界）
        max_agents = EnvParams.SPECIES_RANGE[1] * EnvParams.SPECIES_AGENTS_RANGE[1]
        max_tasks = EnvParams.TASKS_RANGE[1]
        max_nodes = EnvParams.MAX_DESTINATIONS  # depots + tasks

        # 计算张量维度：前4列是LTL约束，后max_agents*3列是智能体状态
        tensor_width = 4 + max_agents * 3
        tensor = np.zeros((TrainParams.LTL_MAX_CLAUSES, tensor_width), dtype=np.float32)

        # ==================== 第一部分：LTL约束编码（前4列）====================
        for i, clause in enumerate(self.clauses):
            if i >= TrainParams.LTL_MAX_CLAUSES:
                break

            # 类型编码（one-hot）
            type_encoding = float(clause.type)  # 0=SAFETY, 1=SEQUENTIAL

            # 参数和状态编码（使用max值归一化，确保跨episode一致性）
            if clause.type == LTL_SAFETY:
                # 安全约束：param1=agent_id, param2=node_id
                param1_norm = clause.param1 / max(max_agents, 1)  # 使用max_agents
                # node_id范围是[0, depots+tasks-1]
                param2_norm = clause.param2 / max(max_nodes, 1)   # 使用max_nodes
                # 状态：SAFE=0 → 0.0, VIOLATED=1 → 1.0
                state_norm = clause.state / 1.0
            else:  # LTL_SEQUENTIAL
                # 顺序约束：param1=task_id, param2=task_id
                param1_norm = clause.param1 / max(max_tasks, 1)   # 使用max_tasks
                param2_norm = clause.param2 / max(max_tasks, 1)   # 使用max_tasks
                # 状态：INITIAL=0 → 0.0, PREDECESSOR_DONE=1 → 0.33,
                #       SATISFIED=2 → 0.67, VIOLATED=3 → 1.0
                state_norm = clause.state / 3.0

            # 组合：[type, param1, param2, state]
            tensor[i, :4] = [type_encoding, param1_norm, param2_norm, state_norm]

        # ==================== 第二部分：智能体状态编码（后max_agents*3列）====================
        # 遍历当前环境的所有智能体，获取其状态
        for agent_id, agent in self.env.agent_dic.items():
            if agent_id >= max_agents:
                # 超出最大智能体数，跳过（理论上不会发生）
                continue

            # 获取智能体状态
            is_inactive = agent.get('is_inactive', False)
            is_temp_sleeping = agent.get('is_temp_sleeping', False)

            # 计算该智能体在张量中的起始位置
            agent_status_start = 4 + agent_id * 3

            # 三状态one-hot编码
            if is_inactive:
                # 失活状态：[0, 0, 1]
                tensor[:, agent_status_start + 2] = 1.0
            elif is_temp_sleeping:
                # 临时休眠状态：[0, 1, 0]
                tensor[:, agent_status_start + 1] = 1.0
            else:
                # 活跃状态：[1, 0, 0]
                tensor[:, agent_status_start + 0] = 1.0

        return tensor
    
    def get_task_feasibility_matrix(self):
        """
        【方案B】生成任务可行性矩阵 - ID无关的泛化表征

        理论基础：
        - 为每个agent-task对编码可行性（0=可行，1=不可行）
        - 完全ID无关，只编码关系
        - 大幅提升泛化能力

        Returns:
            np.ndarray: [max_total_agents, max_tasks]
            - 0.0 = 该agent可以执行该task（不被LTL禁止）
            - 1.0 = 该agent不能执行该task（被LTL禁止）

        编码规则：
        1. SAFETY约束："Agent X cannot visit Task Y"
           → matrix[X, Y] = 1.0
           注意：如果SAFETY约束禁止访问depot（node_id < depots_num），
                该约束不会编码到此矩阵中，因为本矩阵仅表示agent-task关系。
                Depot相关的约束由动作掩码（action mask）在决策时处理。
        2. SEQUENTIAL约束："Task A must precede Task B"
           → 如果A未完成，所有agent都不能做B：matrix[:, B] = 1.0

        设计说明（编码B的限制）：
        - 本矩阵只编码agent-task的可行性关系，不包含depot信息
        - 如果LTL约束包含"Agent X不能访问Depot Y"，该信息会丢失
        - 这种设计是合理的，因为：
          1. Depot访问约束在动作掩码阶段会被正确处理（agent_observe函数）
          2. 本矩阵的目标是提供ID无关的任务分配指导，而非完整的约束建模
          3. Depot通常不是任务目标，而是资源获取点，其约束影响较小
        - 如果需要完整的depot约束建模，应使用编码A或编码C

        维度：[SPECIES_RANGE[1] * SPECIES_AGENTS_RANGE[1], TASKS_RANGE[1]]
        """
        from parameters import EnvParams
        
        # 使用环境的最大值（训练时范围的上界）
        # 总agent数 = species数 × 每个species的agents数
        max_agents = EnvParams.SPECIES_RANGE[1] * EnvParams.SPECIES_AGENTS_RANGE[1]
        max_tasks = EnvParams.TASKS_RANGE[1]
        
        # 初始化为全0（所有任务对所有agent都可行）
        matrix = np.zeros((max_agents, max_tasks), dtype=np.float32)
        
        # 应用LTL约束
        for clause in self.clauses:
            if clause.type == LTL_SAFETY:
                # 安全约束："Agent X cannot visit Node Y"
                agent_id = clause.param1
                node_id = clause.param2
                
                # 检查node_id是否是任务节点（而非depot）
                if node_id >= self.env.depots_num:
                    task_id = node_id - self.env.depots_num
                    
                    # 确保在矩阵范围内
                    if agent_id < max_agents and task_id < max_tasks:
                        # 如果约束有效（未违反），标记为不可行
                        if clause.state == FSA_SAFETY_SAFE:
                            matrix[agent_id, task_id] = 1.0
            
            elif clause.type == LTL_SEQUENTIAL:
                # 顺序约束："Task A must precede Task B"
                task_a_id = clause.param1
                task_b_id = clause.param2
                
                # 如果Task A未完成（状态为INITIAL），则所有agent都不能做Task B
                if clause.state == FSA_SEQ_INITIAL and task_b_id < max_tasks:
                    # 所有agent都不能执行task B
                    matrix[:, task_b_id] = 1.0
        
        return matrix
    
    def get_state_tensor(self):
        """Generates the (max_clauses, 8) tensor representation of the current LTL state.

        【修复】扩展张量维度以支持顺序约束的4个状态（包括VIOLATED）
        原问题：顺序约束有4个状态但只有3个one-hot位置，导致VIOLATED状态编码错误
        
        【改进】添加参数归一化以提升数值稳定性和学习效率。
        理论基础：统一特征尺度，防止梯度爆炸/消失，提升网络收敛性。
        
        张量格式：[max_clauses, 8]
        - [0:2]: type one-hot (SAFETY, SEQUENTIAL)
        - [2:6]: state one-hot (4个位置，支持顺序约束的4个状态)
        - [6]: param1 (归一化)
        - [7]: param2 (归一化)
        """
        tensor = np.zeros((TrainParams.LTL_MAX_CLAUSES, 8), dtype=np.float32)
        for i, clause in enumerate(self.clauses):
            if i >= TrainParams.LTL_MAX_CLAUSES: break

            # One-hot encode clause type
            tensor[i, clause.type] = 1.0
            # One-hot encode FSA state (offset by 2, now with 4 positions)
            # 安全约束：state ∈ {0,1} → 使用位置[2,3]
            # 顺序约束：state ∈ {0,1,2,3} → 使用位置[2,3,4,5]
            if 0 <= clause.state < 4:  # 安全检查
                tensor[i, 2 + clause.state] = 1.0

            # ✅ 归一化参数到 [0,1] 范围，提升数值稳定性和学习效率
            # 理论证明：参数归一化保证了输入分布的一致性，避免了大数值导致的梯度问题
            max_agents = self.env.agents_num
            max_nodes = self.env.depots_num + self.env.tasks_num

            # 安全除法：避免除零错误
            tensor[i, 6] = clause.param1 / max_agents if max_agents > 0 else 0.0
            tensor[i, 7] = clause.param2 / max_nodes if max_nodes > 0 else 0.0

        return tensor
    
    def get_statistics(self):
        """
        获取LTL约束的统计信息（用于评估和日志记录）
        
        Returns:
            dict: 包含以下键的字典
                - num_clauses: 约束总数
                - num_safety: 安全约束数量
                - num_sequential: 顺序约束数量
                - safety_violated: 违规的安全约束数量
                - safety_safe: 未违规的安全约束数量
                - sequential_initial: 未开始的顺序约束数量
                - sequential_predecessor_done: 前置完成的顺序约束数量
                - sequential_satisfied: 完全满足的顺序约束数量
                - safety_violation_rate: 安全约束违规率 (0-1)
                - sequential_satisfaction_rate: 顺序约束满足率 (0-1)
                - overall_satisfaction_rate: 总体满足率 (0-1)
        """
        if not self.clauses:
            return {
                'num_clauses': 0,
                'num_safety': 0,
                'num_sequential': 0,
                'safety_violated': 0,
                'safety_safe': 0,
                'sequential_initial': 0,
                'sequential_predecessor_done': 0,
                'sequential_satisfied': 0,
                'safety_violation_rate': 0.0,
                'sequential_satisfaction_rate': 1.0,  # 没有约束时认为100%满足
                'overall_satisfaction_rate': 1.0
            }
        
        # 初始化计数器
        num_safety = 0
        num_sequential = 0
        safety_violated = 0
        safety_safe = 0
        sequential_initial = 0
        sequential_predecessor_done = 0
        sequential_satisfied = 0
        
        # 统计各种状态
        for clause in self.clauses:
            if clause.type == LTL_SAFETY:
                num_safety += 1
                if clause.state == FSA_SAFETY_VIOLATED:
                    safety_violated += 1
                else:  # FSA_SAFETY_SAFE
                    safety_safe += 1
            
            elif clause.type == LTL_SEQUENTIAL:
                num_sequential += 1
                if clause.state == FSA_SEQ_INITIAL:
                    sequential_initial += 1
                elif clause.state == FSA_SEQ_PREDECESSOR_DONE:
                    sequential_predecessor_done += 1
                elif clause.state == FSA_SEQ_SATISFIED:
                    sequential_satisfied += 1
        
        # 计算违规率和满足率
        safety_violation_rate = safety_violated / num_safety if num_safety > 0 else 0.0
        
        # 顺序约束的满足率：只有完全到达SATISFIED状态才算满足
        sequential_satisfaction_rate = sequential_satisfied / num_sequential if num_sequential > 0 else 1.0
        
        # 总体满足率：(未违规的约束数) / (总约束数)
        # 对于安全约束：SAFE状态算满足
        # 对于顺序约束：SATISFIED状态算满足
        satisfied_count = safety_safe + sequential_satisfied
        overall_satisfaction_rate = satisfied_count / len(self.clauses) if self.clauses else 1.0
        
        return {
            'num_clauses': len(self.clauses),
            'num_safety': num_safety,
            'num_sequential': num_sequential,
            'safety_violated': safety_violated,
            'safety_safe': safety_safe,
            'sequential_initial': sequential_initial,
            'sequential_predecessor_done': sequential_predecessor_done,
            'sequential_satisfied': sequential_satisfied,
            'safety_violation_rate': safety_violation_rate,
            'sequential_satisfaction_rate': sequential_satisfaction_rate,
            'overall_satisfaction_rate': overall_satisfaction_rate
        }
    
    def get_violation_count(self):
        """
        获取违规的约束总数
        
        Returns:
            int: 违规的约束数量（安全约束被违规 + 顺序约束未满足）
        """
        violation_count = 0
        for clause in self.clauses:
            if clause.type == LTL_SAFETY and clause.state == FSA_SAFETY_VIOLATED:
                violation_count += 1
            elif clause.type == LTL_SEQUENTIAL and clause.state != FSA_SEQ_SATISFIED:
                # 顺序约束如果没有完全满足，也算作某种程度的"违规"
                # 这里我们可以选择只统计VIOLATED，或者统计所有非SATISFIED状态
                # 为了与safety对称，我们这里不统计sequential的未满足
                pass
        return violation_count
    
    def get_satisfied_count(self):
        """
        获取满足的约束总数
        
        Returns:
            int: 满足的约束数量
        """
        satisfied_count = 0
        for clause in self.clauses:
            if clause.type == LTL_SAFETY and clause.state == FSA_SAFETY_SAFE:
                satisfied_count += 1
            elif clause.type == LTL_SEQUENTIAL and clause.state == FSA_SEQ_SATISFIED:
                satisfied_count += 1
        return satisfied_count
    
    def compute_ltl_potential(self):
        """
        【方法E：全局LTL势能函数】
        
        理论基础：
        - 基于约束满足状态的势能函数
        - 满足SMDP势能塑形定理（策略不变性）
        - 终止态势能为0（所有约束满足时）
        
        Returns:
            float: LTL势能值，归一化后 ∈ [-1, 0]
            
        势能定义：
            φ_ltl(q) = Σ score(clause) / C - 1.0
            
        Score定义：
            SAFETY:
                SAFE=0 → score=1.0（未违反）
                VIOLATED=1 → score=0.0（已违反）
            SEQUENTIAL:
                INITIAL=0 → score=0.0（未开始）
                PREDECESSOR_DONE=1 → score=0.5（进行中）
                SATISFIED=2 → score=1.0（已满足）
                VIOLATED=3 → score=-1.0（违规，负势能）
        
        首项：φ(q_0) ≈ -0.4 到 -0.6（取决于约束类型分布）
        末项：φ(q_terminal) = 0.0（所有约束满足）
        """
        if not self.clauses:
            return 0.0  # 无约束时势能为0
        
        total_score = 0.0
        
        for clause in self.clauses:
            if clause.type == LTL_SAFETY:
                # 安全约束：2个状态
                if clause.state == FSA_SAFETY_SAFE:
                    score = 1.0      # 未违反（好）
                elif clause.state == FSA_SAFETY_VIOLATED:
                    score = 0.0      # 已违反（坏）
                else:
                    score = 0.0
            
            elif clause.type == LTL_SEQUENTIAL:
                # 顺序约束：4个状态
                if clause.state == FSA_SEQ_INITIAL:
                    score = 0.0      # 未开始
                elif clause.state == FSA_SEQ_PREDECESSOR_DONE:
                    score = 0.5      # 进行中
                elif clause.state == FSA_SEQ_SATISFIED:
                    score = 1.0      # 已满足
                elif clause.state == FSA_SEQ_VIOLATED:
                    score = -1.0     # 违规（负势能，强烈惩罚）
                else:
                    score = 0.0
            else:
                score = 0.0
            
            total_score += score
        
        # 归一化：平均score - 1.0
        # 使得终止态（所有约束满足）势能 = 1.0 - 1.0 = 0
        phi_raw = total_score / len(self.clauses)
        phi_normalized = phi_raw - 1.0
        
        return phi_normalized

    # ====================================================================================
    # 拓扑势能（Blocked-Depth Energy）：仅基于“仍在阻塞的”顺序依赖图
    # ====================================================================================
    def compute_topology_potential(self):
        """
        计算"被阻塞的深度能量"作为势能（带对数权重版本）：
            E_bd = Σ_{v ∈ unfinished, indeg(v)>0} log(1 + indeg(v)) * depth(v)
            depth(v) = 1 + max 后继深度（无后继则为1，按边数+1）
            Φ = - E_bd

        理论依据：
            - log(1 + indeg(v)) 作为阻塞严重度权重
            - indeg越大 → 被阻塞越严重 → 权重越高
            - 对数函数确保边际递减效应（避免高indeg主导）

        设计要点：
            - 只使用当前仍阻塞后继的顺序边（state == FSA_SEQ_INITIAL）
            - 只考虑未完成任务
            - 若出现环（外部fixed_clauses未过滤），返回0以避免数值异常
            - 无顺序约束或无阻塞时返回0

        权重函数性质：
            - 单调性：indeg ↑ → weight ↑ → E_bd ↑
            - 当完成前置任务，indeg ↓ → weight ↓ → E_bd ↓ → Φ ↑ → 正奖励 ✓
            - 边际递减：第10个前置的边际贡献 < 第1个前置的边际贡献
        """
        if not self.clauses or not TrainParams.LTL_ENABLED:
            return 0.0

        # 收集未完成任务
        unfinished_tasks = {tid for tid, t in self.env.task_dic.items() if not t.get('finished', False)}
        if not unfinished_tasks:
            return 0.0

        # 构建剩余依赖图（仅阻塞边）
        adj = {tid: [] for tid in unfinished_tasks}
        indeg = {tid: 0 for tid in unfinished_tasks}
        soft_indeg = {tid: 0.0 for tid in unfinished_tasks}
        has_seq_edge = False

        for clause in self.clauses:
            if clause.type != LTL_SEQUENTIAL:
                continue
            if clause.state != FSA_SEQ_INITIAL:
                continue  # 仅未解锁的边才会阻塞
            a, b = clause.param1, clause.param2
            if a in unfinished_tasks and b in unfinished_tasks:
                # 连续阻塞度：与编码C的 edge_attr 语义一致
                task_A = self.env.task_dic.get(a)
                if task_A is None:
                    blocking_degree = 1.0
                else:
                    total_required = np.sum(task_A['requirements'])
                    if total_required > 0:
                        remaining = np.sum(task_A['status'])
                        completion_ratio = (total_required - remaining) / total_required
                        completion_ratio = np.clip(completion_ratio, 0.0, 1.0)
                    else:
                        completion_ratio = 1.0 if task_A.get('finished', False) else 0.0
                    blocking_degree = 1.0 - completion_ratio

                # 选项A：边存在当且仅当阻塞度>0（前置任务未完全完成）
                if blocking_degree <= 1e-8:
                    continue

                adj.setdefault(a, []).append(b)
                indeg[b] = indeg.get(b, 0) + 1
                soft_indeg[b] = soft_indeg.get(b, 0.0) + float(blocking_degree)
                has_seq_edge = True

        if not has_seq_edge:
            return 0.0  # 没有阻塞边，势能为0

        # 拓扑排序检测环
        q = deque([v for v, d in indeg.items() if d == 0])
        topo = []
        while q:
            v = q.popleft()
            topo.append(v)
            for u in adj.get(v, []):
                indeg[u] -= 1
                if indeg[u] == 0:
                    q.append(u)

        if len(topo) != len(adj):
            # 存在环，防御性返回0，避免异常 shaping
            return 0.0

        # 反向DP计算深度
        # 注意：叶子节点也应贡献非零深度（否则简单的A->B约束中B会贡献0，解锁信号变弱）
        depth = {v: 1 for v in adj}
        for v in reversed(topo):
            children = adj.get(v, [])
            if children:
                depth[v] = 1 + max(depth[c] for c in children)
            else:
                depth[v] = 1

        # 【改进二：对数权重】被阻塞节点的加权深度和
        # weight(v) = log(1 + soft_indeg(v))
        # 语义：soft_indeg越大，任务被阻塞越严重，在E_bd中贡献越大
        # 当推进前置任务时，soft_indeg减小 → weight减小 → E_bd减小 → Φ增大 → 正奖励
        E_bd = 0.0
        for v, d in depth.items():
            soft_indeg_v = soft_indeg.get(v, 0.0)
            if soft_indeg_v > 0.0:
                # 对数权重：压缩量级，避免高soft_indeg主导
                weight = math.log(1.0 + soft_indeg_v)
                E_bd += weight * d

        if E_bd <= 0.0:
            return 0.0

        # 【修改：移除归一化以恢复跨episode马尔可夫性】
        # 归一化会导致相同状态在不同episode有不同势能值，破坏价值函数泛化
        # 直接返回 -E_bd，满足PBRS定理且保持马尔可夫性
        return -E_bd

    def get_dependency_adjacency_matrix(self):
        """
        【方案一：构建任务依赖图的邻接矩阵】用于GAT

        构建一个表示任务依赖关系的邻接矩阵，专门用于Graph Attention Network。
        这个矩阵是ID无关的，只表达关系结构。

        Returns:
            np.ndarray: [MAX_DESTINATIONS, MAX_DESTINATIONS] 邻接矩阵
            - adjacency[i, j] = 1.0 表示 Task i → Task j (i是j的前置任务)
            - adjacency[i, j] = 0.0 表示 没有依赖关系
            - 维度匹配padding后的task_status（任务+仓库）

        注意：这个矩阵用于GAT的信息传播，与任务ID无关，只关注关系结构
        """
        from parameters import EnvParams

        # 【修复】：使用MAX_DESTINATIONS而不是TASKS_RANGE[1]
        # 这样维度才能匹配padding后的task_embedding（包含任务+仓库）
        max_destinations = EnvParams.MAX_DESTINATIONS
        adjacency = np.zeros((max_destinations, max_destinations), dtype=np.float32)

        # 遍历所有顺序约束，构建依赖边
        for clause in self.clauses:
            if clause.type == LTL_SEQUENTIAL:
                pred_task_id = clause.param1  # 前置任务
                succ_task_id = clause.param2  # 后继任务

                # 确保在矩阵范围内（只有任务节点0-99才有LTL依赖，仓库节点100-105没有）
                if pred_task_id < max_destinations and succ_task_id < max_destinations:
                    # 添加有向边：pred → succ
                    adjacency[pred_task_id, succ_task_id] = 1.0

        return adjacency

    def get_dependency_graph_dynamic(self):
        """
        【方案C：稀疏依赖图 + 连续边权重】用于GNN

        构建稀疏图表示（edge_index）+ 连续边权重（edge_attr），
        专门用于Graph Neural Network。

        Returns:
            tuple: (edge_index, edge_attr)
            - edge_index: np.ndarray [2, E] 边索引列表（稀疏表示）
                格式: [[source_0, source_1, ...],
                       [target_0, target_1, ...]]
            - edge_attr: np.ndarray [E, 1] 连续边权重
                语义: 阻塞度 ∈ [0, 1]
                - 0.0 = 前置任务完全完成（边已解锁）
                - 1.0 = 前置任务未开始（边完全阻塞）
                - 0.5 = 前置任务完成50%（半阻塞）

        理论依据:
        - 连续权重编码"解锁进度"，指导"优先完成接近完成的前置任务"策略
        - 与LTL硬约束兼容：掩码仍基于finished二值标志，连续权重仅用于表征
        - 与TOPOLOGY势能互补：TOPOLOGY关注宏观图结构，边权重关注微观边状态

        复杂度: O(C) 线性于顺序约束数量
        """
        edge_list = []
        edge_blocking_degree = []  # 阻塞度 ∈ [0, 1]

        for clause in self.clauses:
            if clause.type == LTL_SEQUENTIAL:
                pred_task_id = clause.param1  # 前置任务A
                succ_task_id = clause.param2  # 后继任务B

                edge_list.append([pred_task_id, succ_task_id])

                # 计算前置任务A的完成度
                task_A = self.env.task_dic.get(pred_task_id)

                if task_A is None:
                    # 任务不存在（异常情况），默认完全阻塞
                    blocking_degree = 1.0
                else:
                    # 计算完成度: (total_required - remaining) / total_required
                    total_required = np.sum(task_A['requirements'])

                    if total_required > 0:
                        remaining = np.sum(task_A['status'])
                        completion_ratio = (total_required - remaining) / total_required
                        completion_ratio = np.clip(completion_ratio, 0.0, 1.0)
                    else:
                        # 零需求任务，根据finished标志判断
                        completion_ratio = 1.0 if task_A.get('finished', False) else 0.0

                    # 阻塞度 = 1 - 完成度
                    # 完成度1.0 → 阻塞度0.0（完全解锁）
                    # 完成度0.0 → 阻塞度1.0（完全阻塞）
                    blocking_degree = 1.0 - completion_ratio

                edge_blocking_degree.append(blocking_degree)

        # 转换为numpy数组（稀疏表示）
        if edge_list:
            # edge_index: [2, E] - PyTorch Geometric格式
            edge_index = np.array(edge_list, dtype=np.int64).T  # 转置为 [2, E]
            # edge_attr: [E, 1] - 每条边的连续权重
            edge_attr = np.array(edge_blocking_degree, dtype=np.float32).reshape(-1, 1)
        else:
            # 无顺序约束时，返回空数组
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 1), dtype=np.float32)

        return edge_index, edge_attr

    def get_task_ltl_features(self, task_id):
        """
        【方案二：为单个任务计算ID无关的LTL状态特征】

        计算一个任务的LTL相关特征，这些特征完全不依赖具体的任务ID，
        只表达语义信息（"有前置任务吗"、"前置任务完成了吗"等）。

        Args:
            task_id: 任务ID

        Returns:
            np.ndarray: [5] 特征向量，包含：
                [0] has_prerequisites: 是否有前置任务 (0或1)
                [1] prerequisites_all_done: 前置任务是否都完成 (0或1)
                [2] pending_prerequisites_ratio: 待完成前置任务的比例 [0,1]
                [3] has_successors: 是否有后继任务 (0或1)
                [4] blocking_successors_ratio: 被阻塞的后继任务比例 [0,1]
        """
        # 查找所有与这个任务相关的约束
        predecessors = []  # (task_id, state)
        successors = []    # (task_id, state)

        for clause in self.clauses:
            if clause.type == LTL_SEQUENTIAL:
                if clause.param2 == task_id:  # 这个任务是后继
                    predecessors.append((clause.param1, clause.state))
                elif clause.param1 == task_id:  # 这个任务是前置
                    successors.append((clause.param2, clause.state))

        # ===== 特征1: 是否有前置任务 =====
        has_prerequisites = float(len(predecessors) > 0)

        # ===== 特征2-3: 前置任务的完成状态 =====
        if len(predecessors) > 0:
            # 检查所有前置约束的状态
            prereq_states = [state for _, state in predecessors]

            # 前置任务"完成"的定义：约束处于PREDECESSOR_DONE或SATISFIED状态
            done_count = sum(1 for s in prereq_states
                           if s in [FSA_SEQ_PREDECESSOR_DONE, FSA_SEQ_SATISFIED])

            prerequisites_all_done = float(done_count == len(predecessors))

            # 待完成的前置任务比例
            pending_count = sum(1 for s in prereq_states if s == FSA_SEQ_INITIAL)
            pending_prerequisites_ratio = pending_count / len(predecessors)
        else:
            # 没有前置任务，认为"都完成了"
            prerequisites_all_done = 1.0
            pending_prerequisites_ratio = 0.0

        # ===== 特征4-5: 后继任务的阻塞状态 =====
        has_successors = float(len(successors) > 0)

        if len(successors) > 0:
            # 计算有多少后继任务被这个任务阻塞（状态为INITIAL）
            blocking_count = sum(1 for _, state in successors if state == FSA_SEQ_INITIAL)
            blocking_successors_ratio = blocking_count / len(successors)
        else:
            blocking_successors_ratio = 0.0

        return np.array([
            has_prerequisites,
            prerequisites_all_done,
            pending_prerequisites_ratio,
            has_successors,
            blocking_successors_ratio
        ], dtype=np.float32)


# The following `if __name__ == '__main__':` block is for testing and does not need to be changed.
if __name__ == '__main__':
    print("")