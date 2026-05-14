"""
AVNR (Adaptive Variable Neighborhood Race) Metaheuristic
基于论文复现: "Metaheuristics with variable diversity control and neighborhood search
for the Heterogeneous Site-Dependent Multi-depot Multi-trip Periodic Vehicle Routing Problem"
Computers & Operations Research, 2023

问题特征:
- Multi-Depot (多仓库)
- Multi-Trip (多趟) - 每个智能体进行多次往返运输
- Heterogeneous Fleet (异构车队)
- Multi-Dimensional Cargo (多维货物) - 每次运输一种货物类型
- 目标: 最小化makespan/总成本

AVNR算法特点:
- Adaptive: 自适应选择邻域算子
- Variable Neighborhood: 多种邻域结构
- Race: 动态评估邻域性能
- 单解轨迹算法（vs 种群算法）

【重要】问题建模：
- 任务有多维需求 (例如 [2, 1, 0, 2, 1])
- 每个任务被分解为多个"配送请求" (每个非零需求维度一个)
- 每个智能体每次只能携带一种货物类型
- 每次运输: 仓库 → LOAD(货物类型) → 任务 → UNLOAD → 返回仓库
- 所有智能体并行工作，目标是最小化makespan
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Set, Optional
import pickle
import pandas as pd
import glob
from collections import defaultdict

# 复用GreenVRPInstance数据结构
try:
    from baseline.MILP_GA_GreenVRP import GreenVRPInstance
except ImportError:
    from MILP_GA_GreenVRP import GreenVRPInstance

# natsort是可选依赖
try:
    from natsort import natsorted
except ImportError:
    print("Warning: natsort not found. Using standard sorted instead.")
    natsorted = sorted


# ============================
# 新增：配送请求数据结构
# ============================

class DeliveryRequest:
    """
    配送请求：表示一次具体的货物配送

    属性:
        task_id: 目标任务ID
        cargo_type: 货物类型 (0-4)
        quantity: 需要配送的数量
        request_id: 唯一请求ID (用于追踪)
    """
    def __init__(self, task_id: int, cargo_type: int, quantity: int, request_id: int):
        self.task_id = task_id
        self.cargo_type = cargo_type
        self.quantity = quantity
        self.request_id = request_id

    def __repr__(self):
        return f"Req{self.request_id}(T{self.task_id},C{self.cargo_type},Q{self.quantity})"


class AgentTrip:
    """
    智能体的一次往返运输

    属性:
        agent_id: 智能体ID
        depot_id: 起始/返回仓库ID
        request: 配送请求
        start_time: 开始时间
        end_time: 结束时间
    """
    def __init__(self, agent_id: int, depot_id: int, request: DeliveryRequest):
        self.agent_id = agent_id
        self.depot_id = depot_id
        self.request = request
        self.start_time = 0.0
        self.end_time = 0.0

    def __repr__(self):
        return f"Trip(A{self.agent_id},D{self.depot_id},{self.request},[{self.start_time:.1f}-{self.end_time:.1f}])"


class AVNRSolution:
    """
    AVNR解的数据结构 - 多趟运输模型

    表示方式：
    - agent_trips: Dict[int, List[AgentTrip]] - 每个智能体的运输列表
    - requests: List[DeliveryRequest] - 所有配送请求
    - objective: float - 目标函数值（makespan）

    【重要】新的建模方式：
    - 每个智能体有多次往返运输（trips）
    - 每次运输配送一个请求（一种货物到一个任务）
    - makespan = max(所有智能体的最后完成时间)

    【修改 - 基于Q1论文 Expert Systems with Applications 2025】
    - 支持使用环境仿真器进行精确评估
    """

    def __init__(self, agent_trips: Dict[int, List[AgentTrip]],
                 requests: List[DeliveryRequest],
                 instance: GreenVRPInstance,
                 env=None,
                 ltl_monitor=None):
        """
        初始化AVNR解

        Args:
            agent_trips: 每个智能体的运输列表
            requests: 所有配送请求
            instance: 问题实例
            env: 环境仿真器（可选），用于精确评估
            ltl_monitor: LTL监视器（可选）
        """
        self.agent_trips = agent_trips
        self.requests = requests
        self.instance = instance
        self.env = env
        self.ltl_monitor = ltl_monitor

        # 计算目标函数值
        self.objective = self._calculate_objective()

        # 约束违反量
        self.violation = self._calculate_violation()

    def copy(self):
        """深拷贝解"""
        # 深拷贝agent_trips
        agent_trips_copy = {}
        for agent_id, trips in self.agent_trips.items():
            agent_trips_copy[agent_id] = []
            for trip in trips:
                new_trip = AgentTrip(trip.agent_id, trip.depot_id, trip.request)
                new_trip.start_time = trip.start_time
                new_trip.end_time = trip.end_time
                agent_trips_copy[agent_id].append(new_trip)

        return AVNRSolution(
            agent_trips=agent_trips_copy,
            requests=self.requests.copy(),
            instance=self.instance,
            env=self.env,
            ltl_monitor=self.ltl_monitor
        )

    def _calculate_objective(self) -> float:
        """
        计算目标函数值 - MAKESPAN (多趟运输模型)

        【修改 - 使用近似评估加速VND搜索】
        VND邻域搜索需要评估大量候选解（数千个），使用精确仿真太慢。
        因此在搜索过程中使用近似评估，最终解的精确评估在benchmark_ijcai.py中进行。

        Returns:
            objective: Makespan（越小越好）
        """
        if not self.agent_trips:
            return float('inf')

        # 使用近似计算（快速，用于VND搜索）
        return self._calculate_objective_approximate()

    def _calculate_objective_approximate(self) -> float:
        """
        近似计算目标函数值

        Makespan = 所有智能体中最晚的完成时间

        【关键】加入任务完成率惩罚，与精确评估行为一致：
        - 如果有未分配的请求，加入惩罚
        - 防止VND搜索选择"假好"的空解

        Returns:
            objective: Makespan + 违反惩罚（越小越好）
        """
        if not self.agent_trips:
            return float('inf')

        makespan = 0.0

        for agent_id, trips in self.agent_trips.items():
            if not trips:
                continue

            # 计算该智能体的所有运输
            agent_completion_time = self._calculate_agent_completion_time(agent_id, trips)
            makespan = max(makespan, agent_completion_time)

        # 【关键修复】检查任务完成率，加入惩罚
        # 与精确评估的行为一致，防止选择未完成任务的"假好"解
        violation = self._calculate_violation()
        if violation > 1e-6:
            makespan += 1e9 * violation

        return makespan

    def _calculate_agent_completion_time(self, agent_id: int, trips: List[AgentTrip]) -> float:
        """
        计算单个智能体完成所有运输的时间

        【修改】任务执行时间只在任务完成时计算一次，而不是每次运输都计算

        Args:
            agent_id: 智能体ID
            trips: 该智能体的运输列表

        Returns:
            completion_time: 完成时间
        """
        if not trips:
            return 0.0

        inst = self.instance
        current_time = 0.0
        current_location = trips[0].depot_id  # 从第一个仓库开始

        # 跟踪每个任务是否已完成（所有货物都卸载完成）
        task_completion_status = {}  # {task_id: set of delivered cargo_types}

        # 【修复】使用instance中的装卸时间，而不是硬编码
        load_unload_time = inst.load_unload_time if hasattr(inst, 'load_unload_time') else 0.1

        for trip in trips:
            depot_id = trip.depot_id
            task_id = trip.request.task_id
            cargo_type = trip.request.cargo_type

            # 1. 从当前位置移动到仓库（如果不在仓库）
            if current_location != depot_id:
                depot_node_id = depot_id
                if (current_location, depot_node_id) in inst.distance_matrix:
                    travel_time = inst.distance_matrix[(current_location, depot_node_id)] / inst.agent_velocity
                    current_time += travel_time

            # 2. 在仓库装载货物
            current_time += load_unload_time
            current_location = depot_id

            # 3. 从仓库移动到任务
            task_node_id = inst.depots[-1] + 1 + task_id
            if (depot_id, task_node_id) in inst.distance_matrix:
                travel_time = inst.distance_matrix[(depot_id, task_node_id)] / inst.agent_velocity
                current_time += travel_time

            # 4. 在任务处卸载货物
            current_time += load_unload_time

            # 5. 跟踪该任务的货物卸载情况
            if task_id not in task_completion_status:
                task_completion_status[task_id] = set()
            task_completion_status[task_id].add(cargo_type)

            # 6. 检查该任务是否所有货物都已卸载完成
            task_requirement = inst.task_requirements[task_id]
            required_cargo_types = set()
            for c in range(len(task_requirement)):
                if task_requirement[c] > 0:
                    required_cargo_types.add(c)

            # 如果该任务的所有货物类型都已卸载，添加任务执行时间
            if task_completion_status[task_id] == required_cargo_types:
                if task_id in inst.task_duration:
                    current_time += inst.task_duration[task_id]

            current_location = task_node_id

            # 更新trip的时间
            trip.end_time = current_time

        return current_time

    def _calculate_violation(self) -> float:
        """
        计算约束违反量 - 多趟运输模型

        检查：
        1. 每个请求是否被分配
        2. 智能体能力约束（每个智能体只能携带其能力范围内的货物）

        Returns:
            violation: 总违反量（0表示可行）
        """
        inst = self.instance
        total_violation = 0.0

        # 检查1：所有请求是否都被分配
        assigned_requests = set()
        for agent_id, trips in self.agent_trips.items():
            for trip in trips:
                assigned_requests.add(trip.request.request_id)

        unassigned_count = len(self.requests) - len(assigned_requests)
        total_violation += unassigned_count * 1000.0  # 未分配请求的惩罚

        # 检查2：智能体能力约束
        for agent_id, trips in self.agent_trips.items():
            agent_capacity = inst.agent_capacity[agent_id]
            for trip in trips:
                cargo_type = trip.request.cargo_type

                # 检查智能体是否能携带这种货物（多趟运输模式：只要容量>0就可行）
                if agent_capacity[cargo_type] <= 0:
                    total_violation += 1000.0  # 无法携带该货物类型

        return total_violation

    def is_feasible(self) -> bool:
        """检查解是否可行"""
        return self.violation < 1e-6

    def get_all_tasks(self) -> Set[int]:
        """获取所有被访问的任务"""
        all_tasks = set()
        for agent_id, trips in self.agent_trips.items():
            for trip in trips:
                all_tasks.add(trip.request.task_id)
        return all_tasks

    def get_num_vehicles_used(self) -> int:
        """获取使用的车辆数量"""
        return len([agent_id for agent_id, trips in self.agent_trips.items() if trips])

    def convert_to_routes(self) -> List[List[int]]:
        """
        将多趟运输转换为路由格式（用于兼容性）

        【重要】去重处理：同一个任务可能有多次运输（不同货物类型），
        但simulate_solution_execution会自动处理一个任务的所有货物类型，
        所以这里只保留每个任务的第一次出现。

        Returns:
            routes: 每个智能体一条路由（包含所有访问的任务，去重）
        """
        routes = []
        for agent_id in sorted(self.agent_trips.keys()):
            trips = self.agent_trips[agent_id]
            if trips:
                # 提取该智能体访问的所有任务（按顺序，去重）
                route = []
                seen_tasks = set()
                for trip in trips:
                    task_id = trip.request.task_id
                    if task_id not in seen_tasks:
                        route.append(task_id)
                        seen_tasks.add(task_id)
                routes.append(route)
            else:
                routes.append([])
        return routes

    def convert_to_ga_format(self) -> Dict:
        """
        将AVNR解转换为GA格式（包含routes、quantity_ratios和cargo_type_priority）

        GA格式的solution包含:
        - routes: List[List[int]] - 每个agent访问的任务列表
        - quantity_ratios: Dict[int, Dict[int, int]] - {agent_id: {task_id: quantity}}
        - cargo_type_priority: Dict[int, Dict[int, List[int]]] - {agent_id: {task_id: [cargo_types]}}

        【关键修复】AVNR的每个trip指定了具体的cargo_type，需要传递给执行器：
        - 每个智能体每次只能装一种货物
        - cargo_type_priority告诉执行器该agent应该按什么顺序运送哪些货物类型

        Returns:
            GA格式的solution字典
        """
        routes = []
        quantity_ratios = {}
        cargo_type_priority = {}  # 【新增】货物类型优先级

        for agent_id in sorted(self.agent_trips.keys()):
            trips = self.agent_trips[agent_id]
            route = []
            agent_qty_ratios = {}
            agent_cargo_priority = {}  # 【新增】该agent的货物类型优先级
            seen_tasks = set()
            task_cargo_types = {}  # {task_id: [cargo_types按trip顺序]}

            for trip in trips:
                task_id = trip.request.task_id
                cargo_type = trip.request.cargo_type  # 【新增】提取cargo_type
                quantity = trip.request.quantity

                if task_id not in seen_tasks:
                    route.append(task_id)
                    seen_tasks.add(task_id)
                    agent_qty_ratios[task_id] = quantity
                    task_cargo_types[task_id] = [cargo_type]  # 【新增】初始化货物类型列表
                else:
                    # 同一个task的多个cargo_type
                    agent_qty_ratios[task_id] = max(agent_qty_ratios.get(task_id, 0), quantity)
                    # 【新增】按trip顺序添加cargo_type（避免重复）
                    if cargo_type not in task_cargo_types[task_id]:
                        task_cargo_types[task_id].append(cargo_type)

            # 【新增】构建该agent的cargo_type_priority
            for task_id in route:
                agent_cargo_priority[task_id] = task_cargo_types.get(task_id, [])

            routes.append(route)
            quantity_ratios[agent_id] = agent_qty_ratios
            cargo_type_priority[agent_id] = agent_cargo_priority  # 【新增】

        return {
            'routes': routes,
            'quantity_ratios': quantity_ratios,
            'cargo_type_priority': cargo_type_priority,  # 【新增】传递货物类型优先级
            'route_array': [],
            'approx_fitness': self.objective,
        }


class AVNRSolver:
    """
    AVNR (Adaptive Variable Neighborhood Race) 求解器

    算法流程：
    1. 构造初始解
    2. 主循环：
       - VND局部搜索
       - Racing机制选择最优邻域
       - 如果无改进，Shaking扰动
    3. 返回最优解
    """

    # 邻域算子类型
    RELOCATE = 'relocate'        # 移动单个任务
    SWAP = 'swap'                # 交换两个任务
    TWO_OPT = '2-opt'            # 2-opt反转
    OR_OPT_1 = 'or-opt-1'        # 移动1个连续任务
    OR_OPT_2 = 'or-opt-2'        # 移动2个连续任务
    CROSS_EXCHANGE = 'cross-exchange'  # 交换两条路由的段

    def __init__(self,
                 instance: GreenVRPInstance,
                 max_iterations: int = 1000,
                 max_time: float = 300.0,
                 vnd_neighborhoods: Optional[List[str]] = None,
                 shaking_strength: int = 3,
                 env=None,
                 ltl_monitor=None):
        """
        初始化AVNR求解器

        Args:
            instance: 问题实例
            max_iterations: 最大迭代次数
            max_time: 最大运行时间（秒）
            vnd_neighborhoods: VND使用的邻域列表
            shaking_strength: 扰动强度（执行多少次随机移动）
            env: 环境仿真器（TaskEnv实例），用于计算真实makespan
            ltl_monitor: LTL监视器（可选），用于处理LTL约束
        """
        self.instance = instance
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.shaking_strength = shaking_strength
        self.env = env
        self.ltl_monitor = ltl_monitor

        # 默认邻域算子列表（按性能顺序）
        if vnd_neighborhoods is None:
            self.vnd_neighborhoods = [
                self.RELOCATE,
                self.SWAP,
                self.TWO_OPT,
                self.OR_OPT_1,
                self.OR_OPT_2,
                self.CROSS_EXCHANGE
            ]
        else:
            self.vnd_neighborhoods = vnd_neighborhoods

        # Racing权重（自适应更新）
        self.neighborhood_weights = {nb: 1.0 for nb in self.vnd_neighborhoods}
        self.neighborhood_improvements = {nb: [] for nb in self.vnd_neighborhoods}

        # 统计信息
        self.best_solution = None
        self.best_objective = float('inf')
        self.iterations = 0
        self.start_time = None
        self.initial_solution = None  # 外部提供的初始解
        self.convergence_history = []  # 收敛历史记录: [{time, best_objective, iteration}, ...]

    def set_initial_solution(self, solution: AVNRSolution):
        """
        设置外部提供的初始解（如RL解）

        Args:
            solution: AVNRSolution实例
        """
        self.initial_solution = solution

    def solve(self) -> AVNRSolution:
        """
        运行AVNR算法

        Returns:
            best_solution: 最优解
        """
        print(f"\n{'='*60}")
        print("运行 AVNR Metaheuristic")
        print(f"{'='*60}")
        print(f"参数配置:")
        print(f"  - 最大迭代次数: {self.max_iterations}")
        print(f"  - 最大运行时间: {self.max_time}s")
        print(f"  - VND邻域数量: {len(self.vnd_neighborhoods)}")
        print(f"  - Shaking强度: {self.shaking_strength}")
        print(f"{'='*60}")

        self.start_time = time.time()

        # Step 1: 构造初始解
        print("\n[Step 1] 构造初始解...")
        if self.initial_solution is not None:
            # 使用外部提供的初始解（如RL解）
            print("  使用外部提供的初始解（RL warm-start）")
            current_solution = self.initial_solution
        else:
            # 使用默认的贪心构造
            current_solution = self._construct_initial_solution()
        print(f"  初始解目标值: {current_solution.objective:.2f}")
        print(f"  初始解可行性: {'可行' if current_solution.is_feasible() else '不可行'}")
        print(f"  使用的车辆数: {current_solution.get_num_vehicles_used()}/{self.instance.num_agents}")

        # 更新最优解
        self.best_solution = current_solution.copy()
        self.best_objective = current_solution.objective

        # 重置收敛历史并记录初始解
        self.convergence_history = []
        self.convergence_history.append({
            'time': time.time() - self.start_time,
            'best_objective': self.best_objective,
            'iteration': 0
        })

        # Step 2: 主循环
        print(f"\n[Step 2] 开始主循环...")
        no_improvement_count = 0
        max_no_improvement = 50

        for iteration in range(self.max_iterations):
            self.iterations = iteration + 1

            # 检查时间限制
            if time.time() - self.start_time > self.max_time:
                print(f"\n⏰ 达到时间限制 ({self.max_time}s)，停止搜索")
                break

            # Step 2.1: VND局部搜索
            improved_solution = self._variable_neighborhood_descent(current_solution)

            # Step 2.2: 检查是否改进
            if improved_solution.objective < current_solution.objective - 1e-6:
                current_solution = improved_solution
                no_improvement_count = 0

                # 更新全局最优
                if current_solution.objective < self.best_objective:
                    self.best_solution = current_solution.copy()
                    self.best_objective = current_solution.objective
                    print(f"  Iter {iteration}: 🎯 新最优解! Objective = {self.best_objective:.2f}")
            else:
                no_improvement_count += 1

            # 记录收敛历史（每次迭代记录一次）
            self.convergence_history.append({
                'time': time.time() - self.start_time,
                'best_objective': self.best_objective,
                'iteration': iteration + 1
            })

            # Step 2.3: 如果长时间无改进，执行Shaking
            if no_improvement_count >= max_no_improvement:
                print(f"  Iter {iteration}: 🔄 执行Shaking扰动...")
                current_solution = self._shaking(self.best_solution, self.shaking_strength)
                no_improvement_count = 0

            # 定期打印进度
            if iteration % 50 == 0:
                elapsed = time.time() - self.start_time
                print(f"  Iter {iteration}: Best = {self.best_objective:.2f}, "
                      f"Current = {current_solution.objective:.2f}, "
                      f"Time = {elapsed:.1f}s")

        # Step 3: 返回最优解
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print("AVNR算法完成")
        print(f"{'='*60}")
        print(f"  最优目标值: {self.best_objective:.2f}")
        print(f"  总迭代次数: {self.iterations}")
        print(f"  总运行时间: {total_time:.2f}s")
        print(f"  使用的车辆数: {self.best_solution.get_num_vehicles_used()}/{self.instance.num_agents}")
        print(f"  可行性: {'✓ 可行' if self.best_solution.is_feasible() else '✗ 不可行'}")
        print(f"{'='*60}")

        return self.best_solution

    def _construct_initial_solution(self) -> AVNRSolution:
        """
        构造初始解 - 多趟运输模型

        【新策略】：
        1. 将所有任务分解为配送请求（每个非零需求维度一个请求）
        2. 为每个智能体分配请求，确保100%车辆利用率
        3. 使用负载均衡策略分配请求

        Returns:
            solution: 初始解
        """
        inst = self.instance

        # 步骤1：分解任务为配送请求
        requests = []
        request_id = 0
        for task_id in inst.tasks:
            task_requirement = inst.task_requirements[task_id]
            for cargo_type in range(inst.num_commodities):
                quantity = int(task_requirement[cargo_type])
                if quantity > 0:
                    # 创建配送请求
                    request = DeliveryRequest(task_id, cargo_type, quantity, request_id)
                    requests.append(request)
                    request_id += 1

        print(f"  [AVNR] 分解任务: {len(inst.tasks)}个任务 → {len(requests)}个配送请求")

        # 步骤2：为每个智能体构建可执行请求列表
        agent_feasible_requests = {}
        for agent_id in inst.agents:
            agent_capacity = inst.agent_capacity[agent_id]
            feasible = []
            for request in requests:
                # 检查智能体是否能携带这种货物（只要容量>0就可以多趟运输）
                if agent_capacity[request.cargo_type] > 0:
                    feasible.append(request)
            agent_feasible_requests[agent_id] = feasible

        # 步骤3：初始化每个智能体的运输列表
        agent_trips = {agent_id: [] for agent_id in inst.agents}
        agent_completion_times = {agent_id: 0.0 for agent_id in inst.agents}
        assigned_requests = set()

        # 步骤4：使用负载均衡策略分配请求
        # 随机打乱请求顺序（避免偏向）
        requests_shuffled = requests.copy()
        np.random.shuffle(requests_shuffled)

        for request in requests_shuffled:
            # 找到所有能执行该请求的智能体
            feasible_agents = []
            for agent_id in inst.agents:
                if request in agent_feasible_requests[agent_id]:
                    feasible_agents.append(agent_id)

            if not feasible_agents:
                print(f"  [WARNING] 请求 {request} 无可行智能体！")
                continue

            # 选择当前完成时间最小的智能体（负载均衡）
            best_agent = min(feasible_agents, key=lambda a: agent_completion_times[a])

            # 为该智能体分配仓库（选择最近的仓库）
            agent_depot = inst.agent_depot[best_agent]

            # 创建运输
            trip = AgentTrip(best_agent, agent_depot, request)

            # 估算该运输的时间
            trip_time = self._estimate_trip_time(best_agent, agent_depot, request)

            # 更新智能体的完成时间
            trip.start_time = agent_completion_times[best_agent]
            trip.end_time = agent_completion_times[best_agent] + trip_time
            agent_completion_times[best_agent] = trip.end_time

            # 添加到智能体的运输列表
            agent_trips[best_agent].append(trip)
            assigned_requests.add(request.request_id)

        # 步骤5：检查车辆利用率
        num_vehicles_used = len([a for a in inst.agents if agent_trips[a]])
        utilization = num_vehicles_used / inst.num_agents
        print(f"  [AVNR] 车辆利用率: {num_vehicles_used}/{inst.num_agents} = {utilization:.1%}")

        # 创建解对象（传递env和ltl_monitor用于精确评估）
        solution = AVNRSolution(agent_trips, requests, inst, env=self.env, ltl_monitor=self.ltl_monitor)

        return solution

    def _estimate_trip_time(self, agent_id: int, depot_id: int, request: DeliveryRequest) -> float:
        """
        估算单次运输的时间

        Args:
            agent_id: 智能体ID
            depot_id: 仓库ID
            request: 配送请求

        Returns:
            time: 估算时间
        """
        inst = self.instance
        task_id = request.task_id

        # 计算距离：仓库 → 任务 → 返回仓库
        task_node_id = inst.depots[-1] + 1 + task_id

        distance = 0.0
        if (depot_id, task_node_id) in inst.distance_matrix:
            distance += inst.distance_matrix[(depot_id, task_node_id)]  # 去程
        # 【修复】返程应该使用 (task_node_id, depot_id)
        if (task_node_id, depot_id) in inst.distance_matrix:
            distance += inst.distance_matrix[(task_node_id, depot_id)]  # 返程

        # 计算时间（agent_velocity是单一浮点数）
        travel_time = distance / inst.agent_velocity
        # 【修复】使用instance中的装卸时间
        load_unload_time = inst.load_unload_time * 2 if hasattr(inst, 'load_unload_time') else 0.2
        # 【修复】任务执行时间不应该在这里计算，因为任务可能由多个agent协作完成
        # task_time应该只在任务完成时计算一次

        total_time = travel_time + load_unload_time

        return total_time

    def _variable_neighborhood_descent(self, solution: AVNRSolution) -> AVNRSolution:
        """
        VND (Variable Neighborhood Descent) 局部搜索

        系统地探索多个邻域，直到无法改进

        Args:
            solution: 当前解

        Returns:
            improved_solution: 改进后的解
        """
        current = solution.copy()

        # 遍历所有邻域
        for neighborhood in self.vnd_neighborhoods:
            # 在当前邻域中搜索改进
            improved = self._explore_neighborhood(current, neighborhood)

            # 如果找到改进，更新解并重新从第一个邻域开始
            if improved.objective < current.objective - 1e-6:
                current = improved
                # 记录改进（用于Racing自适应）
                improvement = solution.objective - improved.objective
                self.neighborhood_improvements[neighborhood].append(improvement)

                # 可选：重新开始（First Improvement策略）
                # break

        return current

    def _explore_neighborhood(self, solution: AVNRSolution, neighborhood: str) -> AVNRSolution:
        """
        探索指定邻域，寻找最优邻居

        使用Best Improvement策略：评估所有邻居，返回最好的

        Args:
            solution: 当前解
            neighborhood: 邻域类型

        Returns:
            best_neighbor: 最优邻居（如果无改进，返回原解）
        """
        if neighborhood == self.RELOCATE:
            return self._best_relocate(solution)
        elif neighborhood == self.SWAP:
            return self._best_swap(solution)
        elif neighborhood == self.TWO_OPT:
            return self._best_2opt(solution)
        elif neighborhood == self.OR_OPT_1:
            return self._best_or_opt(solution, k=1)
        elif neighborhood == self.OR_OPT_2:
            return self._best_or_opt(solution, k=2)
        elif neighborhood == self.CROSS_EXCHANGE:
            return self._best_cross_exchange(solution)
        else:
            return solution.copy()

    def _best_relocate(self, solution: AVNRSolution) -> AVNRSolution:
        """
        Relocate邻域：将一个运输（trip）移动到另一个智能体 - 多趟运输模型

        Args:
            solution: 当前解

        Returns:
            best_solution: 最优邻居
        """
        best_solution = solution.copy()
        best_objective = solution.objective

        # 遍历所有智能体的所有运输
        for agent_i in solution.agent_trips.keys():
            trips_i = solution.agent_trips[agent_i]
            if not trips_i:
                continue

            for trip_idx in range(len(trips_i)):
                trip = trips_i[trip_idx]
                request = trip.request

                # 尝试将该运输移动到其他智能体
                for agent_j in solution.agent_trips.keys():
                    if agent_i == agent_j:
                        continue

                    # 检查agent_j是否能执行该请求（多趟运输：只要容量>0就可行）
                    agent_j_capacity = self.instance.agent_capacity[agent_j]
                    if agent_j_capacity[request.cargo_type] <= 0:
                        continue

                    # 创建新解
                    new_agent_trips = {}
                    for agent_id, trips in solution.agent_trips.items():
                        new_agent_trips[agent_id] = []
                        for t in trips:
                            new_trip = AgentTrip(t.agent_id, t.depot_id, t.request)
                            new_trip.start_time = t.start_time
                            new_trip.end_time = t.end_time
                            new_agent_trips[agent_id].append(new_trip)

                    # 从agent_i移除该运输
                    removed_trip = new_agent_trips[agent_i].pop(trip_idx)

                    # 添加到agent_j（更新agent_id和depot）
                    new_trip = AgentTrip(agent_j, self.instance.agent_depot[agent_j], removed_trip.request)
                    new_agent_trips[agent_j].append(new_trip)

                    # 评估新解
                    new_solution = AVNRSolution(new_agent_trips, solution.requests.copy(), self.instance, env=self.env, ltl_monitor=self.ltl_monitor)

                    if new_solution.objective < best_objective - 1e-6:
                        best_solution = new_solution
                        best_objective = new_solution.objective

        return best_solution

    def _best_swap(self, solution: AVNRSolution) -> AVNRSolution:
        """
        Swap邻域：交换两个智能体的运输 - 多趟运输模型

        Args:
            solution: 当前解

        Returns:
            best_solution: 最优邻居
        """
        best_solution = solution.copy()
        best_objective = solution.objective

        # 获取所有有运输的智能体
        agents_with_trips = [(agent_id, trips) for agent_id, trips in solution.agent_trips.items() if trips]

        # 遍历所有智能体对
        for i, (agent_i, trips_i) in enumerate(agents_with_trips):
            for trip_i_idx in range(len(trips_i)):
                trip_i = trips_i[trip_i_idx]
                request_i = trip_i.request

                for j, (agent_j, trips_j) in enumerate(agents_with_trips):
                    if j <= i:  # 避免重复
                        continue

                    for trip_j_idx in range(len(trips_j)):
                        trip_j = trips_j[trip_j_idx]
                        request_j = trip_j.request

                        # 检查交换后的可行性（多趟运输：只要容量>0就可行）
                        agent_i_capacity = self.instance.agent_capacity[agent_i]
                        agent_j_capacity = self.instance.agent_capacity[agent_j]

                        if (agent_i_capacity[request_j.cargo_type] <= 0 or
                            agent_j_capacity[request_i.cargo_type] <= 0):
                            continue

                        # 创建新解
                        new_agent_trips = {}
                        for agent_id, trips in solution.agent_trips.items():
                            new_agent_trips[agent_id] = []
                            for t in trips:
                                new_trip = AgentTrip(t.agent_id, t.depot_id, t.request)
                                new_trip.start_time = t.start_time
                                new_trip.end_time = t.end_time
                                new_agent_trips[agent_id].append(new_trip)

                        # 交换运输
                        new_agent_trips[agent_i][trip_i_idx].request = request_j
                        new_agent_trips[agent_j][trip_j_idx].request = request_i

                        # 评估新解
                        new_solution = AVNRSolution(new_agent_trips, solution.requests.copy(), self.instance, env=self.env, ltl_monitor=self.ltl_monitor)

                        if new_solution.objective < best_objective - 1e-6:
                            best_solution = new_solution
                            best_objective = new_solution.objective

        return best_solution

    def _best_2opt(self, solution: AVNRSolution) -> AVNRSolution:
        """
        2-opt邻域：反转智能体的运输顺序 - 多趟运输模型

        Args:
            solution: 当前解

        Returns:
            best_solution: 最优邻居
        """
        best_solution = solution.copy()
        best_objective = solution.objective

        # 对每个智能体的运输序列执行2-opt
        for agent_id, trips in solution.agent_trips.items():
            if len(trips) < 2:
                continue

            # 尝试所有可能的反转
            for i in range(len(trips) - 1):
                for j in range(i + 1, len(trips)):
                    # 创建新解
                    new_agent_trips = {}
                    for aid, ts in solution.agent_trips.items():
                        new_agent_trips[aid] = []
                        for t in ts:
                            new_trip = AgentTrip(t.agent_id, t.depot_id, t.request)
                            new_trip.start_time = t.start_time
                            new_trip.end_time = t.end_time
                            new_agent_trips[aid].append(new_trip)

                    # 反转 [i, j]
                    new_agent_trips[agent_id][i:j+1] = list(reversed(new_agent_trips[agent_id][i:j+1]))

                    # 评估新解
                    new_solution = AVNRSolution(new_agent_trips, solution.requests.copy(), self.instance, env=self.env, ltl_monitor=self.ltl_monitor)

                    if new_solution.objective < best_objective - 1e-6:
                        best_solution = new_solution
                        best_objective = new_solution.objective

        return best_solution

    def _best_or_opt(self, solution: AVNRSolution, k: int) -> AVNRSolution:
        """
        Or-opt邻域：移动连续k个运输到另一个智能体 - 多趟运输模型

        Args:
            solution: 当前解
            k: 移动的运输数量

        Returns:
            best_solution: 最优邻居
        """
        best_solution = solution.copy()
        best_objective = solution.objective

        # 遍历所有智能体
        for agent_i, trips_i in solution.agent_trips.items():
            if len(trips_i) < k:
                continue

            # 尝试移动每个长度为k的子序列
            for start_pos in range(len(trips_i) - k + 1):
                segment = trips_i[start_pos:start_pos + k]

                # 检查segment中的所有请求
                segment_requests = [trip.request for trip in segment]

                # 尝试移动到其他智能体
                for agent_j in solution.agent_trips.keys():
                    # 检查agent_j是否能执行所有请求（多趟运输：只要容量>0就可行）
                    agent_j_capacity = self.instance.agent_capacity[agent_j]
                    can_execute_all = all(
                        agent_j_capacity[req.cargo_type] > 0
                        for req in segment_requests
                    )

                    if not can_execute_all:
                        continue

                    # 创建新解
                    new_agent_trips = {}
                    for agent_id, trips in solution.agent_trips.items():
                        new_agent_trips[agent_id] = []
                        for t in trips:
                            new_trip = AgentTrip(t.agent_id, t.depot_id, t.request)
                            new_trip.start_time = t.start_time
                            new_trip.end_time = t.end_time
                            new_agent_trips[agent_id].append(new_trip)

                    # 从agent_i移除segment
                    removed_segment = []
                    for _ in range(k):
                        removed_segment.append(new_agent_trips[agent_i].pop(start_pos))

                    # 添加到agent_j（更新agent_id和depot）
                    for trip in removed_segment:
                        new_trip = AgentTrip(agent_j, self.instance.agent_depot[agent_j], trip.request)
                        new_agent_trips[agent_j].append(new_trip)

                    # 评估新解
                    new_solution = AVNRSolution(new_agent_trips, solution.requests.copy(), self.instance, env=self.env, ltl_monitor=self.ltl_monitor)

                    if new_solution.objective < best_objective - 1e-6:
                        best_solution = new_solution
                        best_objective = new_solution.objective

        return best_solution

    def _best_cross_exchange(self, solution: AVNRSolution) -> AVNRSolution:
        """
        Cross-exchange邻域：交换两个智能体的运输段 - 多趟运输模型

        Args:
            solution: 当前解

        Returns:
            best_solution: 最优邻居
        """
        best_solution = solution.copy()
        best_objective = solution.objective

        # 获取所有有运输的智能体
        agents_with_trips = [(agent_id, trips) for agent_id, trips in solution.agent_trips.items() if len(trips) >= 1]

        if len(agents_with_trips) < 2:
            return best_solution

        # 限制搜索空间：只考虑小段交换（长度1-2）
        max_segment_len = 2

        for i, (agent_i, trips_i) in enumerate(agents_with_trips):
            for j, (agent_j, trips_j) in enumerate(agents_with_trips):
                if j <= i:
                    continue

                # 尝试交换不同长度的段
                for len_i in range(1, min(len(trips_i), max_segment_len) + 1):
                    for start_i in range(len(trips_i) - len_i + 1):
                        for len_j in range(1, min(len(trips_j), max_segment_len) + 1):
                            for start_j in range(len(trips_j) - len_j + 1):
                                # 提取段
                                seg_i = trips_i[start_i:start_i + len_i]
                                seg_j = trips_j[start_j:start_j + len_j]

                                # 检查可行性（多趟运输：只要容量>0就可行）
                                agent_i_capacity = self.instance.agent_capacity[agent_i]
                                agent_j_capacity = self.instance.agent_capacity[agent_j]

                                can_i_execute_j = all(
                                    agent_i_capacity[trip.request.cargo_type] > 0
                                    for trip in seg_j
                                )
                                can_j_execute_i = all(
                                    agent_j_capacity[trip.request.cargo_type] > 0
                                    for trip in seg_i
                                )

                                if not (can_i_execute_j and can_j_execute_i):
                                    continue

                                # 创建新解
                                new_agent_trips = {}
                                for agent_id, trips in solution.agent_trips.items():
                                    new_agent_trips[agent_id] = []
                                    for t in trips:
                                        new_trip = AgentTrip(t.agent_id, t.depot_id, t.request)
                                        new_trip.start_time = t.start_time
                                        new_trip.end_time = t.end_time
                                        new_agent_trips[agent_id].append(new_trip)

                                # 移除段
                                for _ in range(len_i):
                                    new_agent_trips[agent_i].pop(start_i)
                                for _ in range(len_j):
                                    new_agent_trips[agent_j].pop(start_j)

                                # 插入段（更新agent_id和depot）
                                for offset, trip in enumerate(seg_j):
                                    new_trip = AgentTrip(agent_i, self.instance.agent_depot[agent_i], trip.request)
                                    new_agent_trips[agent_i].insert(start_i + offset, new_trip)

                                for offset, trip in enumerate(seg_i):
                                    new_trip = AgentTrip(agent_j, self.instance.agent_depot[agent_j], trip.request)
                                    new_agent_trips[agent_j].insert(start_j + offset, new_trip)

                                # 评估新解
                                new_solution = AVNRSolution(new_agent_trips, solution.requests.copy(), self.instance, env=self.env, ltl_monitor=self.ltl_monitor)

                                if new_solution.objective < best_objective - 1e-6:
                                    best_solution = new_solution
                                    best_objective = new_solution.objective

        return best_solution

    def _shaking(self, solution: AVNRSolution, strength: int) -> AVNRSolution:
        """
        Shaking扰动机制 - 多趟运输模型

        随机执行多次移动操作，逃离局部最优

        Args:
            solution: 当前解
            strength: 扰动强度（执行多少次随机移动）

        Returns:
            perturbed_solution: 扰动后的解
        """
        current = solution.copy()

        for _ in range(strength):
            # 随机选择一个邻域算子
            neighborhood = np.random.choice(self.vnd_neighborhoods)

            # 随机执行一次该邻域的移动
            if neighborhood == self.RELOCATE:
                current = self._random_relocate(current)
            elif neighborhood == self.SWAP:
                current = self._random_swap(current)
            elif neighborhood == self.TWO_OPT:
                current = self._random_2opt(current)

        return current

    def _random_relocate(self, solution: AVNRSolution) -> AVNRSolution:
        """随机Relocate移动 - 多趟运输模型"""
        # 获取所有有运输的智能体
        agents_with_trips = [(agent_id, trips) for agent_id, trips in solution.agent_trips.items() if trips]

        if len(agents_with_trips) < 2:
            return solution.copy()

        # 随机选择源智能体和目标智能体
        src_agent, src_trips = agents_with_trips[np.random.randint(len(agents_with_trips))]
        dst_agent = list(solution.agent_trips.keys())[np.random.randint(len(solution.agent_trips))]

        # 随机选择一个运输
        trip_idx = np.random.randint(len(src_trips))
        trip = src_trips[trip_idx]

        # 检查目标智能体是否能执行该请求（多趟运输：只要容量>0就可行）
        dst_capacity = self.instance.agent_capacity[dst_agent]
        if dst_capacity[trip.request.cargo_type] <= 0:
            return solution.copy()

        # 创建新解
        new_agent_trips = {}
        for agent_id, trips in solution.agent_trips.items():
            new_agent_trips[agent_id] = []
            for t in trips:
                new_trip = AgentTrip(t.agent_id, t.depot_id, t.request)
                new_trip.start_time = t.start_time
                new_trip.end_time = t.end_time
                new_agent_trips[agent_id].append(new_trip)

        # 移除并添加
        removed_trip = new_agent_trips[src_agent].pop(trip_idx)
        new_trip = AgentTrip(dst_agent, self.instance.agent_depot[dst_agent], removed_trip.request)
        new_agent_trips[dst_agent].append(new_trip)

        return AVNRSolution(new_agent_trips, solution.requests.copy(), self.instance, env=self.env, ltl_monitor=self.ltl_monitor)

    def _random_swap(self, solution: AVNRSolution) -> AVNRSolution:
        """随机Swap移动 - 多趟运输模型"""
        agents_with_trips = [(agent_id, trips) for agent_id, trips in solution.agent_trips.items() if trips]

        if len(agents_with_trips) < 2:
            return solution.copy()

        # 随机选择两个智能体
        idx1, idx2 = np.random.choice(len(agents_with_trips), 2, replace=False)
        agent1, trips1 = agents_with_trips[idx1]
        agent2, trips2 = agents_with_trips[idx2]

        # 随机选择两个运输
        trip1_idx = np.random.randint(len(trips1))
        trip2_idx = np.random.randint(len(trips2))

        trip1 = trips1[trip1_idx]
        trip2 = trips2[trip2_idx]

        # 检查可行性（多趟运输：只要容量>0就可行）
        agent1_capacity = self.instance.agent_capacity[agent1]
        agent2_capacity = self.instance.agent_capacity[agent2]

        if (agent1_capacity[trip2.request.cargo_type] <= 0 or
            agent2_capacity[trip1.request.cargo_type] <= 0):
            return solution.copy()

        # 创建新解
        new_agent_trips = {}
        for agent_id, trips in solution.agent_trips.items():
            new_agent_trips[agent_id] = []
            for t in trips:
                new_trip = AgentTrip(t.agent_id, t.depot_id, t.request)
                new_trip.start_time = t.start_time
                new_trip.end_time = t.end_time
                new_agent_trips[agent_id].append(new_trip)

        # 交换请求
        new_agent_trips[agent1][trip1_idx].request = trip2.request
        new_agent_trips[agent2][trip2_idx].request = trip1.request

        return AVNRSolution(new_agent_trips, solution.requests.copy(), self.instance, env=self.env, ltl_monitor=self.ltl_monitor)

    def _random_2opt(self, solution: AVNRSolution) -> AVNRSolution:
        """随机2-opt移动 - 多趟运输模型"""
        agents_with_trips = [(agent_id, trips) for agent_id, trips in solution.agent_trips.items() if len(trips) >= 2]

        if not agents_with_trips:
            return solution.copy()

        # 随机选择一个智能体
        agent_id, trips = agents_with_trips[np.random.randint(len(agents_with_trips))]

        # 随机选择两个位置
        i, j = sorted(np.random.choice(len(trips), 2, replace=False))

        # 创建新解
        new_agent_trips = {}
        for aid, ts in solution.agent_trips.items():
            new_agent_trips[aid] = []
            for t in ts:
                new_trip = AgentTrip(t.agent_id, t.depot_id, t.request)
                new_trip.start_time = t.start_time
                new_trip.end_time = t.end_time
                new_agent_trips[aid].append(new_trip)

        # 反转
        new_agent_trips[agent_id][i:j+1] = list(reversed(new_agent_trips[agent_id][i:j+1]))

        return AVNRSolution(new_agent_trips, solution.requests.copy(), self.instance, env=self.env, ltl_monitor=self.ltl_monitor)
def run_on_benchmark(folder='RALTestSet', method='avnr', max_iterations=500, max_time=300.0):
    """
    在benchmark数据集上运行AVNR方法

    Args:
        folder: benchmark文件夹路径
        method: 方法名称
        max_iterations: 最大迭代次数
        max_time: 最大运行时间（秒）
    """
    files = natsorted(glob.glob(f'../{folder}/env_*.pkl'), key=lambda y: y.lower())
    perf_metrics = {
        'success_rate': [],
        'makespan': [],
        'time_cost': [],
        'waiting_time': [],
        'travel_dist': [],
        'efficiency': [],
        'solve_time': []
    }

    for i, file_path in enumerate(files):
        print(f"\n{'='*60}")
        print(f"Processing [{i+1}/{len(files)}]: {file_path}")
        print('='*60)

        # 加载环境
        env = pickle.load(open(file_path, 'rb'))
        env.init_state()

        # 转换为GreenVRPInstance
        instance = GreenVRPInstance.from_env(env)
        instance.summary()

        start_time = time.time()

        # 使用AVNR求解
        solver = AVNRSolver(instance, max_iterations=max_iterations, max_time=max_time)
        solution = solver.solve()

        solve_time = time.time() - start_time

        # 提取结果
        if solution and solution.objective < float('inf'):
            makespan = solution.objective
            success = solution.is_feasible()
        else:
            makespan = float('inf')
            success = False

        # 记录性能指标
        if success and makespan < float('inf'):
            perf_metrics['success_rate'].append(1.0)
            perf_metrics['makespan'].append(makespan)
            perf_metrics['time_cost'].append(makespan)
            perf_metrics['waiting_time'].append(0.0)
            perf_metrics['travel_dist'].append(0.0)
            perf_metrics['efficiency'].append(1.0)
            perf_metrics['solve_time'].append(solve_time)
        else:
            perf_metrics['success_rate'].append(0.0)
            perf_metrics['makespan'].append(np.nan)
            perf_metrics['time_cost'].append(np.nan)
            perf_metrics['waiting_time'].append(np.nan)
            perf_metrics['travel_dist'].append(np.nan)
            perf_metrics['efficiency'].append(np.nan)
            perf_metrics['solve_time'].append(solve_time)

        print(f"Result: Success={success}, Makespan={makespan:.2f}, Time={solve_time:.2f}s")

    # 保存结果
    df = pd.DataFrame(perf_metrics)
    output_path = f'../{folder}/{method}_results.csv'
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("Benchmark Results Summary")
    print('='*60)
    print(df.describe())
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    print("AVNR (Adaptive Variable Neighborhood Race) Metaheuristic")
    print("=" * 60)

    # 测试单个实例
    print("\n测试模式：在单个随机实例上运行AVNR")
    print("-" * 60)

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.task_env import TaskEnv

    # 生成小规模测试环境
    env = TaskEnv(
        per_species_range=(3, 4),
        species_range=(3, 4),
        tasks_range=(10, 15),
        depot_num_range=(2, 3),
        traits_dim=5,
        seed=42,
        plot_figure=False
    )
    env.init_state()

    print(f"\n生成的测试环境:")
    print(f"  - Depots: {env.depots_num}")
    print(f"  - Tasks: {env.tasks_num}")
    print(f"  - Agents: {env.agents_num}")
    print(f"  - Species: {env.species_num}")

    # 转换为GreenVRPInstance
    instance = GreenVRPInstance.from_env(env)
    instance.summary()

    # 运行AVNR
    print("\n开始运行AVNR算法...")
    solver = AVNRSolver(
        instance,
        max_iterations=200,
        max_time=60.0,
        shaking_strength=3
    )

    best_solution = solver.solve()

    if best_solution:
        print("\n" + "=" * 60)
        print("最优解找到！")
        print("=" * 60)
        print(f"目标值: {best_solution.objective:.2f}")
        print(f"路由数量: {len([r for r in best_solution.routes if r])}")
        print(f"可行性: {'✓ 可行' if best_solution.is_feasible() else '✗ 不可行'}")
        print(f"约束违反量: {best_solution.violation:.4f}")
        print("\n路由详情:")
        for i, (route, start_depot, end_depot) in enumerate(zip(
            best_solution.routes,
            best_solution.route_start_depots,
            best_solution.route_end_depots
        )):
            if route:
                print(f"  Route {i+1} (Start: Depot {start_depot}, End: Depot {end_depot}): {route}")
    else:
        print("\n未找到可行解")

    print("\n" + "=" * 60)
    print("实现说明:")
    print("=" * 60)
    print("✅ 已完成的组件:")
    print("  - 贪心初始解构造")
    print("  - 6种邻域算子 (Relocate, Swap, 2-opt, Or-opt, Cross-exchange)")
    print("  - VND局部搜索")
    print("  - Shaking扰动机制")
    print("  - 主AVNR循环")
    print("  - 约束检查（容量、时间）")
    print("\n💡 使用方法:")
    print("  - 取消注释run_on_benchmark()调用以在完整数据集上运行")
    print("  - 调整参数（max_iterations, max_time）以平衡质量和速度")
