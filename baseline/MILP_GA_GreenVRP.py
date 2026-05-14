"""
MILP + Genetic Algorithm for Green VRP
基于论文复现: Kabadurmus & Erdogan (2023)
"A green vehicle routing problem with multi-depot, multi-tour, heterogeneous fleet and split deliveries"
Journal of Combinatorial Optimization

问题特征:
- Multi-Depot (多仓库)
- Multi-Tour (多趟)
- Heterogeneous Fleet (异构车队)
- Split Deliveries (允许拆分配送)
- 目标: 最小化碳排放/makespan
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Set, Optional
import pickle
import pandas as pd
import glob

# natsort是可选依赖，如果没有则使用标准sorted
try:
    from natsort import natsorted
except ImportError:
    print("Warning: natsort not found. Using standard sorted instead.")
    natsorted = sorted

# Gurobi是可选依赖，只有在使用MILP求解器时才需要
try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print(
        "Warning: gurobipy not found. MILP solver will not be available. GA will work fine."
    )


class GreenVRPInstance:
    """
    Green VRP问题实例数据结构
    对应多仓库、多趟、异构车队的任务分配问题
    """

    def __init__(
        self,
        # 基本集合
        depots: List[int],  # 仓库集合 (depot IDs)
        tasks: List[int],  # 任务集合 (task IDs)
        agents: List[int],  # 智能体集合 (agent IDs)
        species: List[int],  # 物种/车型集合 (species IDs)
        commodities: List[int],  # 货物类型集合 (commodity types)
        # 位置和距离
        locations: Dict[int, np.ndarray],  # 节点位置 {node_id: [x, y]}
        distance_matrix: Dict[Tuple[int, int], float],  # 距离矩阵
        # 智能体属性
        agent_species: Dict[int, int],  # 智能体所属物种 {agent_id: species_id}
        agent_capacity: Dict[
            int, np.ndarray
        ],  # 智能体容量向量 {agent_id: [cap_0, cap_1, ...]}
        agent_depot: Dict[int, int],  # 智能体归属仓库 {agent_id: depot_id}
        # 任务需求
        task_requirements: Dict[
            int, np.ndarray
        ],  # 任务需求向量 {task_id: [req_0, req_1, ...]}
        task_duration: Dict[int, float],  # 任务持续时间 {task_id: duration}
        # 仓库库存
        depot_stock: Dict[
            int, Dict[int, float]
        ],  # 仓库库存 {depot_id: {commodity: quantity}}
        # 时间限制
        max_time: float = 500.0,
        # 环境参数
        load_unload_time: float = 0.2,  # 装卸时间
        agent_velocity: float = 0.2,  # 智能体速度
    ):

        # 存储所有参数
        self.depots = depots
        self.tasks = tasks
        self.agents = agents
        self.species = species
        self.commodities = commodities

        self.locations = locations
        self.distance_matrix = distance_matrix

        self.agent_species = agent_species
        self.agent_capacity = agent_capacity
        self.agent_depot = agent_depot

        self.task_requirements = task_requirements
        self.task_duration = task_duration

        self.depot_stock = depot_stock

        self.max_time = max_time
        self.load_unload_time = load_unload_time
        self.agent_velocity = agent_velocity

        # 衍生属性
        self.num_depots = len(depots)
        self.num_tasks = len(tasks)
        self.num_agents = len(agents)
        self.num_species = len(species)
        self.num_commodities = len(commodities)
        self.num_nodes = self.num_depots + self.num_tasks

        # 所有节点 (depots + tasks)
        self.all_nodes = depots + tasks

    @staticmethod
    def from_env(env):
        """
        从TaskEnv环境实例构建GreenVRPInstance

        Args:
            env: TaskEnv实例

        Returns:
            GreenVRPInstance对象
        """
        # 提取基本集合
        depots = list(range(env.depots_num))
        tasks = list(range(env.tasks_num))
        agents = list(range(env.agents_num))
        species = list(range(env.species_num))
        commodities = list(range(env.traits_dim))

        # 提取位置信息
        locations = {}
        # 仓库位置
        for d_id, depot in env.depot_dic.items():
            locations[d_id] = depot["location"]
        # 任务位置 (offset by depot_num)
        for t_id, task in env.task_dic.items():
            locations[env.depots_num + t_id] = task["location"]

        # 构建距离矩阵
        distance_matrix = {}
        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    dist = np.linalg.norm(locations[i] - locations[j])
                    distance_matrix[(i, j)] = dist

        # 提取智能体属性
        agent_species = {}
        agent_capacity = {}
        agent_depot = {}

        for a_id, agent in env.agent_dic.items():
            agent_species[a_id] = agent["species"]
            agent_capacity[a_id] = agent["capacity"]
            # 找到agent的初始depot
            initial_depot_node_id = agent["route"][0]  # 负数
            agent_depot[a_id] = -initial_depot_node_id - 1

        # 提取任务需求
        task_requirements = {}
        task_duration = {}
        for t_id, task in env.task_dic.items():
            task_requirements[t_id] = task[
                "requirements"
            ].copy()  # 复制数组，避免引用同一对象
            task_duration[t_id] = task["time"]

        # 提取仓库库存
        depot_stock = {}
        for d_id, depot in env.depot_dic.items():
            depot_stock[d_id] = depot["stock"].copy()

        # 提取species信息
        species_capacities = env.species_dict["capacities"]
        agents_species_num = env.species_dict["number"]

        # 创建实例
        instance = GreenVRPInstance(
            depots=depots,
            tasks=tasks,
            agents=agents,
            species=species,
            commodities=commodities,
            locations=locations,
            distance_matrix=distance_matrix,
            agent_species=agent_species,
            agent_capacity=agent_capacity,
            agent_depot=agent_depot,
            task_requirements=task_requirements,
            task_duration=task_duration,
            depot_stock=depot_stock,
            max_time=env.max_time if hasattr(env, "max_time") else 500.0,
            load_unload_time=env.LOAD_UNLOAD_DURATION
            if hasattr(env, "LOAD_UNLOAD_DURATION")
            else 0.2,
            agent_velocity=0.2,  # 默认速度
        )

        # 添加species_dict（用于GA中的车辆分配）
        instance.species_dict = {
            "capacities": species_capacities,
            "number": agents_species_num,
        }

        return instance

    def summary(self):
        """打印问题实例摘要"""
        print("=" * 50)
        print("   Green VRP Instance Summary")
        print("=" * 50)
        print(f"Depots:      {self.num_depots}")
        print(f"Tasks:       {self.num_tasks}")
        print(f"Agents:      {self.num_agents}")
        print(f"Species:     {self.num_species}")
        print(f"Commodities: {self.num_commodities}")
        print(f"Max Time:    {self.max_time}")
        print("-" * 50)
        print(
            f"Agent capacity range: {[np.sum(cap) for cap in self.agent_capacity.values()]}"
        )
        print(
            f"Task demand range:    {[np.sum(req) for req in self.task_requirements.values()]}"
        )
        print("=" * 50)


class MILPSolver:
    """
    MILP精确求解器
    用于求解小规模Green VRP实例或为GA提供评估
    """

    def __init__(self, instance: GreenVRPInstance, time_limit: float = 3600.0):
        """
        初始化MILP求解器

        Args:
            instance: GreenVRPInstance问题实例
            time_limit: 求解时间限制(秒)
        """
        if not GUROBI_AVAILABLE:
            raise ImportError(
                "MILP solver requires gurobipy. Please install: pip install gurobipy"
            )

        self.instance = instance
        self.time_limit = time_limit
        self.model = None
        self.solution = None

    def build_model(self):
        """
        构建MILP模型

        决策变量:
        - x[i,j,a,t]: 智能体a在第t趟从节点i到节点j (binary)
        - y[i,a,t,c]: 智能体a在第t趟访问节点i时携带的商品c数量
        - z[i,t]: 任务i的完成时间
        - makespan: 最大完成时间

        目标函数:
        - 最小化makespan (或总旅行时间/碳排放)
        """
        print("Building MILP model...")

        inst = self.instance

        # 创建模型
        self.model = gp.Model("GreenVRP_MILP")

        # 估计最大趟数 (简化: 每个agent最多跑的趟数)
        max_trips = min(10, inst.num_tasks)  # 限制最大趟数以控制变量数量

        # ========== 决策变量 ==========

        # x[i,j,a,t]: 智能体a在第t趟从节点i到节点j
        x = {}
        for a in inst.agents:
            for t in range(max_trips):
                for i in inst.all_nodes:
                    for j in inst.all_nodes:
                        if i != j:
                            x[i, j, a, t] = self.model.addVar(
                                vtype=GRB.BINARY, name=f"x_{i}_{j}_{a}_{t}"
                            )

        # y[i,a,t,c]: 智能体a在第t趟访问节点i时携带的商品c数量
        y = {}
        for a in inst.agents:
            for t in range(max_trips):
                for i in inst.all_nodes:
                    for c in inst.commodities:
                        y[i, a, t, c] = self.model.addVar(
                            vtype=GRB.CONTINUOUS, lb=0, name=f"y_{i}_{a}_{t}_{c}"
                        )

        # z[i]: 任务i的完成时间
        z = {}
        for i in inst.tasks:
            z[i] = self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=inst.max_time, name=f"z_{i}"
            )

        # makespan: 最大完成时间
        makespan = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="makespan")

        self.model.update()

        # ========== 目标函数 ==========
        # 最小化makespan
        self.model.setObjective(makespan, GRB.MINIMIZE)

        # ========== 约束条件 ==========

        # (1) Makespan约束
        for i in inst.tasks:
            task_idx = inst.depots[-1] + 1 + i
            self.model.addConstr(makespan >= z[i], name=f"makespan_task_{i}")

        # (2) 每个任务必须被完成 (所有需求被满足)
        for i in inst.tasks:
            task_idx = inst.depots[-1] + 1 + i
            for c in inst.commodities:
                if inst.task_requirements[i][c] > 0:
                    self.model.addConstr(
                        gp.quicksum(
                            y[task_idx, a, t, c]
                            for a in inst.agents
                            for t in range(max_trips)
                        )
                        >= inst.task_requirements[i][c],
                        name=f"task_demand_{i}_{c}",
                    )

        # (3) 容量约束
        for a in inst.agents:
            for t in range(max_trips):
                for c in inst.commodities:
                    self.model.addConstr(
                        gp.quicksum(y[i, a, t, c] for i in inst.all_nodes)
                        <= inst.agent_capacity[a][c],
                        name=f"capacity_{a}_{t}_{c}",
                    )

        # (4) 流守恒约束 (每趟的路径必须形成有效tour)
        for a in inst.agents:
            for t in range(max_trips):
                depot = inst.agent_depot[a]
                # 从depot出发
                self.model.addConstr(
                    gp.quicksum(x[depot, j, a, t] for j in inst.all_nodes if j != depot)
                    == gp.quicksum(
                        x[i, depot, a, t] for i in inst.all_nodes if i != depot
                    ),
                    name=f"flow_depot_{a}_{t}",
                )

                # 中间节点流守恒
                for k in inst.all_nodes:
                    if k != depot:
                        self.model.addConstr(
                            gp.quicksum(x[i, k, a, t] for i in inst.all_nodes if i != k)
                            == gp.quicksum(
                                x[k, j, a, t] for j in inst.all_nodes if j != k
                            ),
                            name=f"flow_{k}_{a}_{t}",
                        )

        # (5) 时间约束 (简化版本，未包含详细的时间累积)
        # 这里仅添加总时间不超过max_time的约束
        for a in inst.agents:
            for t in range(max_trips):
                total_travel = gp.quicksum(
                    x[i, j, a, t]
                    * inst.distance_matrix.get((i, j), 0)
                    / inst.agent_velocity
                    for i in inst.all_nodes
                    for j in inst.all_nodes
                    if i != j
                )
                self.model.addConstr(
                    total_travel <= inst.max_time, name=f"time_limit_{a}_{t}"
                )

        print(
            f"Model built: {self.model.NumVars} variables, {self.model.NumConstrs} constraints"
        )

        # 存储变量以便后续使用
        self.vars = {"x": x, "y": y, "z": z, "makespan": makespan}

        return self.model

    def solve(self):
        """
        求解MILP模型

        Returns:
            solution: dict with keys 'makespan', 'routes', 'success'
        """
        if self.model is None:
            self.build_model()

        print("Solving MILP model...")
        self.model.setParam("TimeLimit", self.time_limit)
        self.model.setParam("MIPGap", 0.01)  # 1% optimality gap

        start_time = time.time()
        self.model.optimize()
        solve_time = time.time() - start_time

        # 提取解
        solution = {
            "success": False,
            "makespan": float("inf"),
            "solve_time": solve_time,
            "routes": {},
            "gap": None,
        }

        if self.model.Status == GRB.OPTIMAL or self.model.Status == GRB.TIME_LIMIT:
            if self.model.SolCount > 0:
                solution["success"] = True
                solution["makespan"] = self.vars["makespan"].X
                solution["gap"] = self.model.MIPGap

                # 提取路径 (简化版本)
                print(f"Solution found: Makespan = {solution['makespan']:.2f}")
                print(f"Solve time: {solve_time:.2f}s, Gap: {solution['gap']:.2%}")
        else:
            print(f"No solution found. Status: {self.model.Status}")

        self.solution = solution
        return solution


class GeneticAlgorithm:
    """
    遗传算法求解器

    基于Kabadurmus & Erdogan (2023)的GA设计:
    - Niching技术维持种群多样性
    - 约束处理技术确保可行性
    """

    def __init__(
        self,
        instance: GreenVRPInstance,
        pop_size: int = 100,
        max_generations: int = 500,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        niching_radius: float = 0.1,
        time_limit: Optional[float] = None,
        env=None,
        ltl_monitor=None,
        decide_quantity: bool = False,
    ):
        """
        初始化遗传算法

        Args:
            instance: 问题实例
            pop_size: 种群大小
            max_generations: 最大迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            niching_radius: Niching半径
            time_limit: 时间限制（秒），如果为None则不限制时间
            env: 环境仿真器（TaskEnv实例），用于计算真实makespan
            ltl_monitor: LTL监视器（可选），用于处理LTL约束
            decide_quantity: 是否让GA决定装货数量（True=编码数量，False=总是装满）
        """
        self.instance = instance
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.niching_radius = niching_radius
        self.time_limit = time_limit
        self.env = env
        self.ltl_monitor = ltl_monitor
        self.decide_quantity = decide_quantity

        self.population = []
        self.best_solution = None
        self.best_fitness = float("inf")
        self.start_time = None
        self.convergence_history = []  # 收敛历史记录: [(time, best_fitness, generation), ...]

    def initialize_population(self):
        """
        初始化种群 (Section 4.2) - 修改为多趟运输模型

        【重要修改】：
        - 为每个智能体创建一条路由，确保100%车辆利用率
        - 允许多个agents分配到同一个task（多agent协作）
        - 每个agent只负责运输它能运输的cargo types（应用agent能力掩码）
        - 每条路由可以包含多个任务（多趟运输）
        - 【新增】当decide_quantity=True时，为每个(agent, task)分配装货数量比例

        编码方式:
        - 路由编码为大小为(num_agents + num_tasks - 1)的数组
        - num_agents条路由由(num_agents-1)个分隔符分隔
        - 每条路由对应一个智能体
        - 【新增】quantity_ratios: Dict[agent_id, Dict[task_id, float]]，存储装货比例(0.25/0.5/0.75/1.0)
        """
        if self.decide_quantity:
            print(
                "Initializing population (Multi-trip model with quantity decision)..."
            )
        else:
            print(
                "Initializing population (Multi-trip model with agent capability mask)..."
            )

        inst = self.instance
        n_tasks = inst.num_tasks
        n_agents = inst.num_agents

        # 预计算每个agent能运输的任务列表（应用agent能力掩码）
        agent_capable_tasks = self._compute_agent_capable_tasks()

        # 可选的装货数量值（与RL动作空间一致）
        quantity_options = [1, 2, 3, 4, 5]

        for _ in range(self.pop_size):
            # 创建num_agents条空路由
            routes = [[] for _ in range(n_agents)]

            # 【关键修改】：使用agent能力掩码分配任务
            all_tasks = list(inst.tasks)
            np.random.shuffle(all_tasks)

            # 为每个任务找到能运输它的agents，并分配
            for task_id in all_tasks:
                # 找到能运输这个任务的agents
                capable_agents = [
                    agent_id
                    for agent_id in range(n_agents)
                    if task_id in agent_capable_tasks[agent_id]
                ]

                if not capable_agents:
                    # 如果没有agent能运输这个任务，随机分配（会得到很差的fitness）
                    capable_agents = list(range(n_agents))

                # 随机选择1-3个有能力的agents来协作完成这个task
                num_agents_for_task = np.random.randint(
                    1, min(4, len(capable_agents) + 1)
                )
                selected_agents = np.random.choice(
                    capable_agents, size=num_agents_for_task, replace=False
                )

                for agent_id in selected_agents:
                    if task_id not in routes[agent_id]:
                        routes[agent_id].append(task_id)

            # 随机打乱每条路由内的任务顺序
            for route in routes:
                if route:
                    np.random.shuffle(route)

            # 【关键修复】确保所有agents都有任务（100%利用率）
            routes = self._ensure_all_agents_used(routes, agent_capable_tasks)

            # 【关键修复】确保所有任务的所有货物类型都被覆盖（多agent协作）
            routes = self._ensure_complete_coverage(routes, agent_capable_tasks)

            # 构建route_array（使用-1作为分隔符）
            route_array = []
            for i, route in enumerate(routes):
                route_array.extend(route)
                if i < n_agents - 1:  # 不在最后一条路由后添加分隔符
                    route_array.append(-1)

            # 为每条路由分配起点和终点仓库
            route_start_depots = []
            route_end_depots = []
            for agent_id in range(n_agents):
                # 使用智能体的归属仓库
                depot_id = inst.agent_depot[agent_id]
                route_start_depots.append(depot_id)
                route_end_depots.append(depot_id)  # 返回同一仓库

            # 【新增】生成装货数量比例（当decide_quantity=True时）
            quantity_ratios = {}
            if self.decide_quantity:
                for agent_id, route in enumerate(routes):
                    quantity_ratios[agent_id] = {}
                    for task_id in route:
                        # 为每个(agent, task)随机选择一个装货数量
                        quantity_ratios[agent_id][task_id] = np.random.choice(
                            quantity_options
                        )

            # 【新增】生成货物类型优先级（cargo_type_priority）
            # 格式：{agent_id: {task_id: [cargo_type_0, cargo_type_1, ...]}}
            # 表示该agent在执行该task时，按此顺序运输货物类型
            cargo_type_priority = {}
            for agent_id, route in enumerate(routes):
                cargo_type_priority[agent_id] = {}
                agent_capacity = inst.agent_capacity[agent_id]
                for task_id in route:
                    task_requirement = inst.task_requirements[task_id]
                    # 获取该任务需要的、且该agent能运输的货物类型
                    relevant_cargo_types = [
                        ct
                        for ct in range(inst.num_commodities)
                        if task_requirement[ct] > 0 and agent_capacity[ct] > 0
                    ]
                    # 随机打乱顺序作为优先级
                    np.random.shuffle(relevant_cargo_types)
                    cargo_type_priority[agent_id][task_id] = relevant_cargo_types

            # 创建个体
            individual = {
                "route_array": route_array,
                "routes": routes,
                "route_start_depots": route_start_depots,
                "route_end_depots": route_end_depots,
                "quantity_ratios": quantity_ratios,  # 装货数量
                "cargo_type_priority": cargo_type_priority,  # 【新增】货物类型优先级
                "approx_fitness": float("inf"),
                "exact_fitness": float("inf"),
                "violation": 0,
                "cleared": False,
            }

            # 计算近似适应度
            individual["approx_fitness"] = self._approximate_emission(individual)

            self.population.append(individual)

        print(f"Population initialized with {len(self.population)} individuals")
        print(f"  每个个体有 {n_agents} 条路由（每个智能体一条）")
        print(f"  每个智能体都有任务（100%车辆利用率）")
        print(f"  应用了agent能力掩码（只分配agent能运输的任务）")
        print(f"  【新增】货物类型优先级由GA优化（公平对比RL）")
        if self.decide_quantity:
            print(f"  装货数量由GA决定（数量选项: {quantity_options}）")

    def _compute_agent_capable_tasks(self):
        """
        计算每个agent能运输的任务列表（应用agent能力掩码）

        一个agent能运输一个任务，当且仅当：
        - agent至少能运输该任务所需的一种货物类型

        Returns:
            agent_capable_tasks: dict {agent_id: [task_id1, task_id2, ...]}
        """
        inst = self.instance
        agent_capable_tasks = {}

        for agent_id in range(inst.num_agents):
            agent_capacity = inst.agent_capacity[agent_id]
            capable_tasks = []

            for task_id in inst.tasks:
                task_requirement = inst.task_requirements[task_id]

                # 检查agent是否能运输至少一种所需货物
                can_transport = False
                for cargo_type in range(inst.num_commodities):
                    if (
                        task_requirement[cargo_type] > 0
                        and agent_capacity[cargo_type] > 0
                    ):
                        can_transport = True
                        break

                if can_transport:
                    capable_tasks.append(task_id)

            agent_capable_tasks[agent_id] = capable_tasks

        return agent_capable_tasks

    def _extract_routes(self, route_array):
        """从路由数组提取路由列表（使用-1作为分隔符）"""
        routes = []
        current_route = []

        for element in route_array:
            if element == -1:
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:
                current_route.append(element)

        # 添加最后一条路由
        if current_route:
            routes.append(current_route)

        return routes

    def _repair_agent_capability_violations(self, routes, agent_capable_tasks):
        """
        修复违反agent能力约束的任务分配

        如果一个agent被分配了它不能运输的任务，将该任务移动到能运输它的agent

        Args:
            routes: 路由列表
            agent_capable_tasks: dict {agent_id: [task_id1, task_id2, ...]}

        Returns:
            repaired_routes: 修复后的路由列表
        """
        inst = self.instance
        n_agents = inst.num_agents

        # 确保routes有足够的长度
        while len(routes) < n_agents:
            routes.append([])

        # 收集所有违反约束的任务
        violations = []  # [(agent_id, task_id), ...]

        for agent_id, route in enumerate(routes):
            if agent_id >= n_agents:
                break

            capable_tasks = agent_capable_tasks.get(agent_id, [])

            for task_id in route:
                if task_id not in capable_tasks:
                    # 该agent不能运输这个任务
                    violations.append((agent_id, task_id))

        # 修复违反
        for agent_id, task_id in violations:
            # 从该agent的路由中移除任务
            if task_id in routes[agent_id]:
                routes[agent_id].remove(task_id)

            # 找到能运输这个任务的agents
            capable_agents = [
                aid
                for aid in range(n_agents)
                if task_id in agent_capable_tasks.get(aid, [])
            ]

            if capable_agents:
                # 随机选择一个有能力的agent
                target_agent = np.random.choice(capable_agents)

                # 确保target_agent在routes范围内
                if target_agent < len(routes):
                    # 如果该任务还没有在目标agent的路由中，添加它
                    if task_id not in routes[target_agent]:
                        routes[target_agent].append(task_id)
            # 如果没有agent能运输这个任务，任务就丢失了（会导致低fitness）

        return routes

    def _ensure_complete_coverage(self, routes, agent_capable_tasks):
        """
        确保所有任务的所有货物类型都被覆盖（多agent协作）

        对于每个任务，检查是否所有需要的货物类型都被至少一个分配到该任务的agent覆盖。
        如果有未覆盖的货物类型，添加能运输这些货物的agents到该任务。

        Args:
            routes: 路由列表
            agent_capable_tasks: dict {agent_id: [task_id1, task_id2, ...]}

        Returns:
            repaired_routes: 修复后的路由列表，确保完全覆盖
        """
        inst = self.instance
        n_agents = inst.num_agents

        # 确保routes有足够的长度
        while len(routes) < n_agents:
            routes.append([])

        # 对每个任务检查覆盖情况
        for task_id in inst.tasks:
            task_requirement = inst.task_requirements[task_id]

            # 找出需要的货物类型
            required_cargo_types = set()
            for cargo_type in range(inst.num_commodities):
                if task_requirement[cargo_type] > 0:
                    required_cargo_types.add(cargo_type)

            if not required_cargo_types:
                continue  # 任务不需要任何货物

            # 找出哪些agents被分配了这个任务
            assigned_agents = []
            for agent_id, route in enumerate(routes):
                if agent_id >= n_agents:
                    break
                if task_id in route:
                    assigned_agents.append(agent_id)

            # 检查这些agents能覆盖哪些货物类型
            covered_cargo_types = set()
            for agent_id in assigned_agents:
                agent_capacity = inst.agent_capacity[agent_id]
                for cargo_type in required_cargo_types:
                    if agent_capacity[cargo_type] > 0:
                        covered_cargo_types.add(cargo_type)

            # 找出未覆盖的货物类型
            uncovered_cargo_types = required_cargo_types - covered_cargo_types

            if uncovered_cargo_types:
                # 需要添加更多agents来覆盖这些货物类型
                # 找出能运输未覆盖货物类型的agents
                for cargo_type in uncovered_cargo_types:
                    # 找到能运输这种货物的agents
                    capable_agents = []
                    for agent_id in range(n_agents):
                        agent_capacity = inst.agent_capacity[agent_id]
                        # 只需要检查agent能否运输这种货物，不需要检查task是否在capable_tasks中
                        # 因为我们需要强制添加这个任务来确保覆盖
                        if agent_capacity[cargo_type] > 0:
                            capable_agents.append(agent_id)

                    if capable_agents:
                        # 选择一个还没有被分配这个任务的agent（如果可能）
                        unassigned_capable = [
                            a for a in capable_agents if task_id not in routes[a]
                        ]
                        if unassigned_capable:
                            selected_agent = np.random.choice(unassigned_capable)
                        else:
                            # 所有能运输的agents都已经被分配了，选择第一个
                            selected_agent = capable_agents[0]

                        # 添加任务到该agent的路由
                        if task_id not in routes[selected_agent]:
                            routes[selected_agent].append(task_id)

        return routes

    def _ensure_all_agents_used(self, routes, agent_capable_tasks=None):
        """
        确保所有智能体都被使用（100%车辆利用率）

        如果某些路由为空，从其他路由中复制任务到空路由
        在多agent协作模型中，多个agents可以共同完成同一个task

        【修改】：应用agent能力掩码，只分配agent能运输的任务

        Args:
            routes: 路由列表
            agent_capable_tasks: dict {agent_id: [task_id1, task_id2, ...]}，可选

        Returns:
            repaired_routes: 修复后的路由列表，确保所有路由都非空
        """
        inst = self.instance
        n_agents = inst.num_agents

        # 确保路由数量正确
        while len(routes) < n_agents:
            routes.append([])
        routes = routes[:n_agents]

        # 找出所有空路由
        empty_routes = [i for i, route in enumerate(routes) if not route]

        if not empty_routes:
            return routes  # 所有路由都非空，无需修复

        # 找出所有非空路由
        non_empty_routes = [(i, route) for i, route in enumerate(routes) if route]

        if not non_empty_routes:
            # 所有路由都是空的，这不应该发生
            # 为每个agent随机分配一个任务
            if agent_capable_tasks:
                # 使用agent能力掩码
                for agent_id in range(n_agents):
                    if agent_capable_tasks[agent_id]:
                        task_id = agent_capable_tasks[agent_id][0]
                    else:
                        task_id = agent_id % inst.num_tasks
                    routes[agent_id].append(task_id)
            else:
                # 不使用掩码
                for agent_id in range(n_agents):
                    task_id = agent_id % inst.num_tasks
                    routes[agent_id].append(task_id)
            return routes

        # 为每个空路由分配任务
        # 【关键】：在多agent协作模型中，可以直接复制任务
        # 因为多个agents可以共同完成同一个task
        for empty_idx in empty_routes:
            if agent_capable_tasks and agent_capable_tasks[empty_idx]:
                # 使用agent能力掩码：从该agent能运输的任务中选择
                # 优先选择已经在其他路由中的任务（协作）
                capable_tasks = agent_capable_tasks[empty_idx]

                # 找到已经在其他路由中且该agent能运输的任务
                existing_tasks = []
                for _, route in non_empty_routes:
                    for task_id in route:
                        if task_id in capable_tasks and task_id not in existing_tasks:
                            existing_tasks.append(task_id)

                if existing_tasks:
                    # 从已存在的任务中随机选择
                    task_to_copy = existing_tasks[
                        np.random.randint(len(existing_tasks))
                    ]
                else:
                    # 从该agent能运输的任务中随机选择
                    task_to_copy = capable_tasks[np.random.randint(len(capable_tasks))]

                routes[empty_idx].append(task_to_copy)
            else:
                # 不使用掩码：从非空路由中随机选择一个任务复制过来
                donor_idx, donor_route = non_empty_routes[
                    np.random.randint(len(non_empty_routes))
                ]
                task_to_copy = donor_route[np.random.randint(len(donor_route))]
                routes[empty_idx].append(task_to_copy)

        return routes

    def _approximate_emission(self, individual):
        """
        近似makespan计算 - 多趟运输模型

        【重要修改 - 基于Q1论文 Expert Systems with Applications 2025】：
        - 每条路由对应一个智能体（agent）
        - 使用正确的agent属性（agent_species, agent_capacity, agent_depot）
        - 智能体需要多次往返运输来完成路由中的所有任务
        - 每次运输只能携带一种货物类型
        - 目标：计算makespan（所有智能体完成任务的最大时间）

        核心思想：
        1. 对于每个智能体（每条路由），计算需要多少次运输
        2. 每个任务的每种货物类型需求可能需要多次运输
        3. 估算完成时间 = 运输次数 × 平均往返时间
        4. Makespan = max(所有智能体的完成时间)

        Args:
            individual: 包含'routes'字段的个体

        Returns:
            approximate_makespan: 近似makespan值
        """
        inst = self.instance
        routes = individual["routes"]

        if not routes:
            return 0.0

        agent_completion_times = []

        for route_idx, route in enumerate(routes):
            if not route:
                # 空路由的智能体完成时间为0
                agent_completion_times.append(0.0)
                continue

            # 【关键修复】使用正确的agent属性
            # route_idx 对应 agent_id
            agent_id = route_idx

            # 获取该智能体的容量（直接使用agent_capacity，而不是通过species间接获取）
            if agent_id < inst.num_agents and agent_id in inst.agent_capacity:
                agent_capacity = inst.agent_capacity[agent_id]
            else:
                # 如果agent_id超出范围，使用species容量作为fallback
                species_id = inst.agent_species.get(
                    agent_id, agent_id % inst.num_species
                )
                agent_capacity = inst.species_dict["capacities"][species_id]

            # 【关键修复】使用agent的归属depot，而不是找最近的depot
            if agent_id < inst.num_agents and agent_id in inst.agent_depot:
                depot_id = inst.agent_depot[agent_id]
            else:
                # 如果没有归属depot信息，使用最近的depot作为fallback
                first_task_loc = inst.locations[inst.depots[-1] + 1 + route[0]]
                closest_depot = None
                min_depot_dist = float("inf")
                for d_id in inst.depots:
                    depot_loc = inst.locations[d_id]
                    dist = np.linalg.norm(first_task_loc - depot_loc)
                    if dist < min_depot_dist:
                        min_depot_dist = dist
                        closest_depot = d_id
                depot_id = closest_depot

            depot_loc = inst.locations[depot_id]

            # 计算该智能体需要的总运输次数
            # 【关键修改】：agent只负责运输它能运输的cargo types
            # 不能运输的cargo types由其他agents负责
            total_trips = 0

            for task_id in route:
                task_requirement = inst.task_requirements[task_id]
                task_loc = inst.locations[inst.depots[-1] + 1 + task_id]

                # 对于每种货物类型
                for cargo_type in range(inst.num_commodities):
                    quantity_needed = int(task_requirement[cargo_type])
                    if quantity_needed == 0:
                        continue

                    # 检查智能体是否能运输这种货物
                    agent_capacity_for_type = agent_capacity[cargo_type]
                    if agent_capacity_for_type == 0:
                        # 智能体无法运输这种货物，跳过
                        # 这种cargo type由其他agents负责
                        continue

                    # 计算需要多少次运输
                    trips_needed = int(
                        np.ceil(quantity_needed / agent_capacity_for_type)
                    )
                    total_trips += trips_needed

            # 估算平均往返距离
            # 简化假设：每次运输的平均距离 = depot到路由中心的距离 × 2
            route_center = np.zeros(2)
            for task_id in route:
                task_loc = inst.locations[inst.depots[-1] + 1 + task_id]
                route_center += task_loc
            route_center /= len(route)

            avg_trip_distance = np.linalg.norm(route_center - depot_loc) * 2

            # 计算完成时间
            # 时间 = 距离 / 速度 + 任务处理时间 + 装卸时间
            travel_time = (avg_trip_distance * total_trips) / inst.agent_velocity
            task_time = sum(inst.task_duration[task_id] for task_id in route)
            # 【新增】考虑装卸时间：每次trip需要一次装货和一次卸货
            load_unload_time = total_trips * 2 * inst.load_unload_time

            completion_time = travel_time + task_time + load_unload_time
            agent_completion_times.append(completion_time)

        # Makespan = 最大完成时间
        makespan = max(agent_completion_times) if agent_completion_times else 0.0

        return makespan

    def evaluate_fitness(self, individual, use_exact=False):
        """
        评估个体适应度（使用真实环境仿真器）

        【重要修改】：
        - 使用环境仿真器计算真实makespan，而不是近似值
        - 确保与RL方法使用相同的评估标准
        - 应用基础掩码，确保动作符合物理规则
        - 【新增】支持decide_quantity参数

        Args:
            individual: 个体字典
            use_exact: 是否使用精确评估（已废弃，保留参数以兼容）

        Returns:
            fitness: 适应度值（越小越好）
        """
        # 检查是否有环境仿真器
        if not hasattr(self, "env") or self.env is None:
            # 如果没有环境仿真器，回退到近似方法
            print(
                "Warning: No environment simulator available, using approximate method"
            )
            violation = self._calculate_violation(individual)
            individual["violation"] = violation
            fitness = self._approximate_emission(individual)
            individual["approx_fitness"] = fitness
            if violation > 1e-6:
                fitness = fitness + 1e6 * violation
            return fitness

        # 使用环境仿真器计算真实makespan
        try:
            # Shared LTL-shielded simulator (same one used for final-metric
            # extraction); GA invokes it once per chromosome per generation,
            # making the search itself LTL-aware.
            from baseline.simulator import simulate_solution_execution

            makespan, travel_dist, time_cost, completion_rate = (
                simulate_solution_execution(
                    self.env,
                    individual,
                    self.ltl_monitor if hasattr(self, "ltl_monitor") else None,
                    debug=False,  # 关闭调试输出
                    decide_quantity=self.decide_quantity,  # 传递decide_quantity参数
                )
            )

            # 计算fitness
            fitness = makespan

            # 如果任务未完成，加大惩罚
            if completion_rate < 1.0:
                fitness += 1e9 * (1.0 - completion_rate)

            # 保存结果到individual
            individual["approx_fitness"] = fitness
            individual["exact_fitness"] = fitness
            individual["makespan"] = makespan
            individual["travel_distance"] = travel_dist
            individual["completion_rate"] = completion_rate

            return fitness

        except Exception as e:
            # 如果环境仿真器执行失败，使用近似方法作为fallback
            print(
                f"Warning: Environment simulation failed ({str(e)}), using approximate method"
            )
            violation = self._calculate_violation(individual)
            individual["violation"] = violation
            fitness = self._approximate_emission(individual)
            individual["approx_fitness"] = fitness
            if violation > 1e-6:
                fitness = fitness + 1e6 * violation
            return fitness

    def selection(self, tournament_size=3):
        """
        锦标赛选择 (Tournament Selection with Niching)

        从未被清除的个体中随机选择tournament_size个，返回适应度最好的

        Args:
            tournament_size: 锦标赛大小

        Returns:
            selected: 选中的个体
        """
        # 获取未被清除的个体
        valid_individuals = [
            ind for ind in self.population if not ind.get("cleared", False)
        ]

        if len(valid_individuals) < tournament_size:
            tournament_size = max(1, len(valid_individuals))

        # 随机选择tournament_size个个体
        candidates = np.random.choice(
            len(valid_individuals), size=tournament_size, replace=False
        )
        tournament = [valid_individuals[i] for i in candidates]

        # 返回适应度最好的（最小的）
        selected = min(tournament, key=lambda x: x["approx_fitness"])

        return selected

    def crossover(self, parent1, parent2):
        """
        两点交叉操作 + 修复机制 (Two-Point Crossover with Repair, Figure 2, Page 11) - 扩展为Open VRP

        【修改】：在多agent协作模型中，route_array长度可能不同（任务可以重复）
        使用路由级别的交叉而不是数组级别的交叉
        【新增】：应用agent能力掩码，确保交叉后的路由满足agent能力约束
        【新增】：处理quantity_ratios的交叉

        核心步骤：
        1. 直接在routes级别进行交叉
        2. 随机选择交叉点，交换部分routes
        3. 修复违反agent能力约束的任务分配
        4. 确保所有agents都有任务
        5. 为子代重新分配仓库（继承父代或随机）
        6. 【新增】处理quantity_ratios的继承和生成

        Args:
            parent1, parent2: 父代个体字典（包含'routes'字段）

        Returns:
            offspring1, offspring2: 修复后的子代个体字典
        """
        inst = self.instance
        p1_routes = [list(r) for r in parent1["routes"]]  # 深拷贝
        p2_routes = [list(r) for r in parent2["routes"]]  # 深拷贝

        n_agents = len(p1_routes)

        # 计算agent能力掩码
        agent_capable_tasks = self._compute_agent_capable_tasks()

        # Step 1: 随机选择交叉点
        crossover_point = np.random.randint(1, n_agents)

        # Step 2: 交换routes
        offspring1_routes = p1_routes[:crossover_point] + p2_routes[crossover_point:]
        offspring2_routes = p2_routes[:crossover_point] + p1_routes[crossover_point:]

        # Step 3: 修复违反agent能力约束的任务分配
        offspring1_routes = self._repair_agent_capability_violations(
            offspring1_routes, agent_capable_tasks
        )
        offspring2_routes = self._repair_agent_capability_violations(
            offspring2_routes, agent_capable_tasks
        )

        # Step 4: 确保所有智能体都被使用（100%车辆利用率）
        offspring1_routes = self._ensure_all_agents_used(
            offspring1_routes, agent_capable_tasks
        )
        offspring2_routes = self._ensure_all_agents_used(
            offspring2_routes, agent_capable_tasks
        )

        # Step 5: 确保所有任务的所有货物类型都被覆盖（多agent协作）
        offspring1_routes = self._ensure_complete_coverage(
            offspring1_routes, agent_capable_tasks
        )
        offspring2_routes = self._ensure_complete_coverage(
            offspring2_routes, agent_capable_tasks
        )

        # Step 6: 重建route_array
        offspring1_array = []
        for i, route in enumerate(offspring1_routes):
            offspring1_array.extend(route)
            if i < len(offspring1_routes) - 1:
                offspring1_array.append(-1)

        offspring2_array = []
        for i, route in enumerate(offspring2_routes):
            offspring2_array.extend(route)
            if i < len(offspring2_routes) - 1:
                offspring2_array.append(-1)

        # Step 7: 为子代分配仓库：80%概率继承父代，20%概率随机
        offspring1_start_depots = []
        offspring1_end_depots = []
        for route_idx, route in enumerate(offspring1_routes):
            if route:
                if np.random.random() < 0.8 and route_idx < len(
                    parent1.get("route_start_depots", [])
                ):
                    # 继承父代1的仓库
                    offspring1_start_depots.append(
                        parent1["route_start_depots"][route_idx]
                    )
                    offspring1_end_depots.append(parent1["route_end_depots"][route_idx])
                else:
                    # 随机分配
                    offspring1_start_depots.append(np.random.choice(inst.depots))
                    offspring1_end_depots.append(np.random.choice(inst.depots))
            else:
                offspring1_start_depots.append(None)
                offspring1_end_depots.append(None)

        offspring2_start_depots = []
        offspring2_end_depots = []
        for route_idx, route in enumerate(offspring2_routes):
            if route:
                if np.random.random() < 0.8 and route_idx < len(
                    parent2.get("route_start_depots", [])
                ):
                    # 继承父代2的仓库
                    offspring2_start_depots.append(
                        parent2["route_start_depots"][route_idx]
                    )
                    offspring2_end_depots.append(parent2["route_end_depots"][route_idx])
                else:
                    # 随机分配
                    offspring2_start_depots.append(np.random.choice(inst.depots))
                    offspring2_end_depots.append(np.random.choice(inst.depots))
            else:
                offspring2_start_depots.append(None)
                offspring2_end_depots.append(None)

        # 【新增】Step 8: 处理quantity_ratios的继承
        quantity_options = [1, 2, 3, 4, 5]
        offspring1_quantity_ratios = {}
        offspring2_quantity_ratios = {}

        if self.decide_quantity:
            p1_ratios = parent1.get("quantity_ratios", {})
            p2_ratios = parent2.get("quantity_ratios", {})

            # 为offspring1生成quantity_ratios
            for agent_id, route in enumerate(offspring1_routes):
                offspring1_quantity_ratios[agent_id] = {}
                for task_id in route:
                    # 尝试从父代继承，否则随机生成
                    if (
                        agent_id < crossover_point
                        and agent_id in p1_ratios
                        and task_id in p1_ratios.get(agent_id, {})
                    ):
                        offspring1_quantity_ratios[agent_id][task_id] = p1_ratios[
                            agent_id
                        ][task_id]
                    elif (
                        agent_id >= crossover_point
                        and agent_id in p2_ratios
                        and task_id in p2_ratios.get(agent_id, {})
                    ):
                        offspring1_quantity_ratios[agent_id][task_id] = p2_ratios[
                            agent_id
                        ][task_id]
                    else:
                        offspring1_quantity_ratios[agent_id][task_id] = (
                            np.random.choice(quantity_options)
                        )

            # 为offspring2生成quantity_ratios
            for agent_id, route in enumerate(offspring2_routes):
                offspring2_quantity_ratios[agent_id] = {}
                for task_id in route:
                    # 尝试从父代继承，否则随机生成
                    if (
                        agent_id < crossover_point
                        and agent_id in p2_ratios
                        and task_id in p2_ratios.get(agent_id, {})
                    ):
                        offspring2_quantity_ratios[agent_id][task_id] = p2_ratios[
                            agent_id
                        ][task_id]
                    elif (
                        agent_id >= crossover_point
                        and agent_id in p1_ratios
                        and task_id in p1_ratios.get(agent_id, {})
                    ):
                        offspring2_quantity_ratios[agent_id][task_id] = p1_ratios[
                            agent_id
                        ][task_id]
                    else:
                        offspring2_quantity_ratios[agent_id][task_id] = (
                            np.random.choice(quantity_options)
                        )

        # 【新增】Step 9: 处理cargo_type_priority的继承
        p1_cargo_priority = parent1.get("cargo_type_priority", {})
        p2_cargo_priority = parent2.get("cargo_type_priority", {})
        offspring1_cargo_priority = {}
        offspring2_cargo_priority = {}

        inst = self.instance

        # 为offspring1生成cargo_type_priority
        for agent_id, route in enumerate(offspring1_routes):
            offspring1_cargo_priority[agent_id] = {}
            agent_capacity = inst.agent_capacity[agent_id]
            for task_id in route:
                task_requirement = inst.task_requirements[task_id]
                # 获取该任务需要的、且该agent能运输的货物类型
                relevant_cargo_types = [
                    ct
                    for ct in range(inst.num_commodities)
                    if task_requirement[ct] > 0 and agent_capacity[ct] > 0
                ]
                # 尝试从父代继承优先级
                inherited = False
                if agent_id < crossover_point:
                    # 从parent1继承
                    if (
                        agent_id in p1_cargo_priority
                        and task_id in p1_cargo_priority.get(agent_id, {})
                    ):
                        parent_priority = p1_cargo_priority[agent_id][task_id]
                        # 过滤掉不再相关的货物类型
                        filtered = [
                            ct for ct in parent_priority if ct in relevant_cargo_types
                        ]
                        # 添加新的货物类型
                        for ct in relevant_cargo_types:
                            if ct not in filtered:
                                filtered.append(ct)
                        offspring1_cargo_priority[agent_id][task_id] = filtered
                        inherited = True
                else:
                    # 从parent2继承
                    if (
                        agent_id in p2_cargo_priority
                        and task_id in p2_cargo_priority.get(agent_id, {})
                    ):
                        parent_priority = p2_cargo_priority[agent_id][task_id]
                        filtered = [
                            ct for ct in parent_priority if ct in relevant_cargo_types
                        ]
                        for ct in relevant_cargo_types:
                            if ct not in filtered:
                                filtered.append(ct)
                        offspring1_cargo_priority[agent_id][task_id] = filtered
                        inherited = True
                if not inherited:
                    # 随机生成
                    np.random.shuffle(relevant_cargo_types)
                    offspring1_cargo_priority[agent_id][task_id] = relevant_cargo_types

        # 为offspring2生成cargo_type_priority
        for agent_id, route in enumerate(offspring2_routes):
            offspring2_cargo_priority[agent_id] = {}
            agent_capacity = inst.agent_capacity[agent_id]
            for task_id in route:
                task_requirement = inst.task_requirements[task_id]
                relevant_cargo_types = [
                    ct
                    for ct in range(inst.num_commodities)
                    if task_requirement[ct] > 0 and agent_capacity[ct] > 0
                ]
                inherited = False
                if agent_id < crossover_point:
                    if (
                        agent_id in p2_cargo_priority
                        and task_id in p2_cargo_priority.get(agent_id, {})
                    ):
                        parent_priority = p2_cargo_priority[agent_id][task_id]
                        filtered = [
                            ct for ct in parent_priority if ct in relevant_cargo_types
                        ]
                        for ct in relevant_cargo_types:
                            if ct not in filtered:
                                filtered.append(ct)
                        offspring2_cargo_priority[agent_id][task_id] = filtered
                        inherited = True
                else:
                    if (
                        agent_id in p1_cargo_priority
                        and task_id in p1_cargo_priority.get(agent_id, {})
                    ):
                        parent_priority = p1_cargo_priority[agent_id][task_id]
                        filtered = [
                            ct for ct in parent_priority if ct in relevant_cargo_types
                        ]
                        for ct in relevant_cargo_types:
                            if ct not in filtered:
                                filtered.append(ct)
                        offspring2_cargo_priority[agent_id][task_id] = filtered
                        inherited = True
                if not inherited:
                    np.random.shuffle(relevant_cargo_types)
                    offspring2_cargo_priority[agent_id][task_id] = relevant_cargo_types

        # 创建子代个体
        offspring1 = {
            "route_array": offspring1_array,
            "routes": offspring1_routes,
            "route_start_depots": offspring1_start_depots,
            "route_end_depots": offspring1_end_depots,
            "quantity_ratios": offspring1_quantity_ratios,
            "cargo_type_priority": offspring1_cargo_priority,  # 【新增】
            "approx_fitness": float("inf"),
            "exact_fitness": float("inf"),
            "violation": 0,
            "cleared": False,
        }

        offspring2 = {
            "route_array": offspring2_array,
            "routes": offspring2_routes,
            "route_start_depots": offspring2_start_depots,
            "route_end_depots": offspring2_end_depots,
            "quantity_ratios": offspring2_quantity_ratios,
            "cargo_type_priority": offspring2_cargo_priority,  # 【新增】
            "approx_fitness": float("inf"),
            "exact_fitness": float("inf"),
            "violation": 0,
            "cleared": False,
        }

        return offspring1, offspring2

    def _repair_chromosome(self, chromosome):
        """
        修复染色体：移除重复任务，补充缺失任务 (Figure 2 Repair Mechanism)

        Args:
            chromosome: 染色体数组 (包含任务ID和-1分隔符)

        Returns:
            repaired: 修复后的染色体
        """
        inst = self.instance
        all_tasks = set(inst.tasks)

        # 统计出现的任务（排除分隔符-1）
        appeared_tasks = []
        for gene in chromosome:
            if gene != -1:  # 非分隔符
                appeared_tasks.append(gene)

        # 找出重复和缺失的任务
        task_counts = {}
        for task in appeared_tasks:
            task_counts[task] = task_counts.get(task, 0) + 1

        duplicates = [task for task, count in task_counts.items() if count > 1]
        appeared_set = set(appeared_tasks)
        missing = list(all_tasks - appeared_set)

        if not duplicates and not missing:
            return chromosome  # 已经合法

        # 修复策略：从左到右扫描，遇到重复任务则替换为缺失任务
        repaired = chromosome.copy()
        seen_tasks = set()
        missing_iter = iter(missing)

        for i, gene in enumerate(repaired):
            if gene == -1:
                continue

            if gene in seen_tasks:
                # 重复任务，替换为缺失任务
                try:
                    replacement = next(missing_iter)
                    repaired[i] = replacement
                    seen_tasks.add(replacement)
                except StopIteration:
                    # 理论上不应该发生（重复数 = 缺失数）
                    pass
            else:
                seen_tasks.add(gene)

        # 如果还有缺失的任务（可能是因为分隔符太多），替换一些分隔符
        if missing_iter:
            remaining_missing = list(missing_iter)
            if remaining_missing:
                for i, gene in enumerate(repaired):
                    if gene == -1 and remaining_missing:
                        repaired[i] = remaining_missing.pop(0)

        return repaired

    def mutation(self, individual):
        """
        变异操作 (Figure 3, Pages 11-12) - 扩展为Open VRP

        三种变异算子（概率分配）:
        - Block Insertion (50%): 移动一个子序列到另一位置
        - Block Swap (30%): 交换两个子序列
        - 2-opt (20%): 反转一个子序列
        - 额外：10%概率变异仓库分配
        【新增】：应用agent能力掩码，确保变异后的路由满足agent能力约束
        【新增】：处理quantity_ratios的变异

        Args:
            individual: 个体字典

        Returns:
            mutated: 变异后的个体字典
        """
        inst = self.instance

        # 计算agent能力掩码
        agent_capable_tasks = self._compute_agent_capable_tasks()

        # 随机选择变异算子
        rand = np.random.random()

        if rand < 0.5:
            # Block Insertion (50%)
            mutated_array = self._block_insertion(individual["route_array"].copy())
        elif rand < 0.8:
            # Block Swap (30%)
            mutated_array = self._block_swap(individual["route_array"].copy())
        else:
            # 2-opt (20%)
            mutated_array = self._two_opt(individual["route_array"].copy())

        # 提取路由
        mutated_routes = self._extract_routes(mutated_array)

        # 修复违反agent能力约束的任务分配
        mutated_routes = self._repair_agent_capability_violations(
            mutated_routes, agent_capable_tasks
        )

        # 确保所有智能体都被使用（100%车辆利用率）
        mutated_routes = self._ensure_all_agents_used(
            mutated_routes, agent_capable_tasks
        )

        # 确保所有任务的所有货物类型都被覆盖（多agent协作）
        mutated_routes = self._ensure_complete_coverage(
            mutated_routes, agent_capable_tasks
        )

        # 仓库分配：继承原个体的仓库，但有10%概率变异
        mutated_start_depots = []
        mutated_end_depots = []
        for route_idx, route in enumerate(mutated_routes):
            if route:
                if route_idx < len(individual.get("route_start_depots", [])):
                    # 继承原仓库
                    start_depot = individual["route_start_depots"][route_idx]
                    end_depot = individual["route_end_depots"][route_idx]

                    # 10%概率变异起点仓库
                    if np.random.random() < 0.1:
                        start_depot = np.random.choice(inst.depots)

                    # 10%概率变异终点仓库
                    if np.random.random() < 0.1:
                        end_depot = np.random.choice(inst.depots)

                    mutated_start_depots.append(start_depot)
                    mutated_end_depots.append(end_depot)
                else:
                    # 新路由，随机分配
                    mutated_start_depots.append(np.random.choice(inst.depots))
                    mutated_end_depots.append(np.random.choice(inst.depots))
            else:
                mutated_start_depots.append(None)
                mutated_end_depots.append(None)

        # 【新增】处理quantity_ratios的继承和变异
        quantity_options = [1, 2, 3, 4, 5]
        mutated_quantity_ratios = {}

        if self.decide_quantity:
            original_ratios = individual.get("quantity_ratios", {})

            for agent_id, route in enumerate(mutated_routes):
                mutated_quantity_ratios[agent_id] = {}
                for task_id in route:
                    # 尝试从原个体继承
                    if agent_id in original_ratios and task_id in original_ratios.get(
                        agent_id, {}
                    ):
                        quantity = original_ratios[agent_id][task_id]
                        # 20%概率变异数量
                        if np.random.random() < 0.2:
                            quantity = np.random.choice(quantity_options)
                        mutated_quantity_ratios[agent_id][task_id] = quantity
                    else:
                        # 新的(agent, task)组合，随机生成
                        mutated_quantity_ratios[agent_id][task_id] = np.random.choice(
                            quantity_options
                        )

        # 【新增】处理cargo_type_priority的继承和变异
        original_cargo_priority = individual.get("cargo_type_priority", {})
        mutated_cargo_priority = {}

        for agent_id, route in enumerate(mutated_routes):
            mutated_cargo_priority[agent_id] = {}
            agent_capacity = inst.agent_capacity[agent_id]
            for task_id in route:
                task_requirement = inst.task_requirements[task_id]
                # 获取该任务需要的、且该agent能运输的货物类型
                relevant_cargo_types = [
                    ct
                    for ct in range(inst.num_commodities)
                    if task_requirement[ct] > 0 and agent_capacity[ct] > 0
                ]

                # 尝试从原个体继承
                if (
                    agent_id in original_cargo_priority
                    and task_id in original_cargo_priority.get(agent_id, {})
                ):
                    parent_priority = original_cargo_priority[agent_id][task_id]
                    # 过滤掉不再相关的货物类型，保持原顺序
                    filtered = [
                        ct for ct in parent_priority if ct in relevant_cargo_types
                    ]
                    # 添加新的货物类型
                    for ct in relevant_cargo_types:
                        if ct not in filtered:
                            filtered.append(ct)

                    # 30%概率变异优先级（随机交换两个位置）
                    if np.random.random() < 0.3 and len(filtered) >= 2:
                        i, j = np.random.choice(len(filtered), size=2, replace=False)
                        filtered[i], filtered[j] = filtered[j], filtered[i]

                    mutated_cargo_priority[agent_id][task_id] = filtered
                else:
                    # 新的(agent, task)组合，随机生成
                    np.random.shuffle(relevant_cargo_types)
                    mutated_cargo_priority[agent_id][task_id] = relevant_cargo_types

        # 创建变异后的个体
        mutated = {
            "route_array": mutated_array,
            "routes": mutated_routes,
            "route_start_depots": mutated_start_depots,
            "route_end_depots": mutated_end_depots,
            "quantity_ratios": mutated_quantity_ratios,
            "cargo_type_priority": mutated_cargo_priority,  # 【新增】
            "approx_fitness": float("inf"),
            "exact_fitness": float("inf"),
            "violation": 0,
            "cleared": False,
        }

        return mutated

    def _block_insertion(self, chromosome):
        """
        块插入变异 (Block Insertion Mutation)

        随机选择一个子序列，移动到另一个随机位置
        """
        n = len(chromosome)
        if n < 4:
            return chromosome

        # 随机选择块的起始和结束位置
        start = np.random.randint(0, n - 1)
        end = np.random.randint(start + 1, min(start + 5, n))  # 块大小限制为1-4

        # 随机选择插入位置（不在当前块内）
        insert_pos = np.random.randint(0, n - (end - start))
        if insert_pos >= start:
            insert_pos += end - start

        # 提取块
        block = chromosome[start:end]

        # 移除原位置的块
        new_chromosome = np.concatenate([chromosome[:start], chromosome[end:]])

        # 插入到新位置
        if insert_pos <= start:
            result = np.concatenate(
                [new_chromosome[:insert_pos], block, new_chromosome[insert_pos:]]
            )
        else:
            adj_insert = insert_pos - (end - start)
            result = np.concatenate(
                [new_chromosome[:adj_insert], block, new_chromosome[adj_insert:]]
            )

        return result

    def _block_swap(self, chromosome):
        """
        块交换变异 (Block Swap Mutation)

        随机选择两个不重叠的子序列并交换它们的位置
        """
        n = len(chromosome)
        if n < 6:
            return chromosome

        # 随机选择第一个块
        start1 = np.random.randint(0, n - 3)
        end1 = np.random.randint(start1 + 1, min(start1 + 4, n - 2))

        # 随机选择第二个块（不重叠）
        start2 = np.random.randint(end1, n - 1)
        end2 = np.random.randint(start2 + 1, min(start2 + 4, n))

        # 交换两个块
        result = chromosome.copy()
        block1 = chromosome[start1:end1].copy()
        block2 = chromosome[start2:end2].copy()

        # 如果块大小不同，需要调整
        if len(block1) == len(block2):
            result[start1:end1] = block2
            result[start2:end2] = block1
        else:
            # 先移除两个块，然后插入到对方位置
            temp = np.concatenate(
                [chromosome[:start1], chromosome[end1:start2], chromosome[end2:]]
            )
            insert1 = start1
            temp = np.concatenate([temp[:insert1], block2, temp[insert1:]])
            insert2 = start2 - (end1 - start1) + len(block2)
            result = np.concatenate([temp[:insert2], block1, temp[insert2:]])

        return result

    def _two_opt(self, chromosome):
        """
        2-opt变异 (2-opt Mutation)

        随机选择一个子序列并反转（用于改善路由顺序）
        """
        n = len(chromosome)
        if n < 3:
            return chromosome

        # 随机选择两个点
        i = np.random.randint(0, n - 1)
        j = np.random.randint(i + 1, min(i + 6, n))  # 限制反转长度

        # 反转子序列
        result = chromosome.copy()
        result[i:j] = result[i:j][::-1]

        return result

    def _calculate_niching_distance(self, ind1, ind2):
        """
        计算两个个体之间的Niching距离 (Algorithm 5, Page 17)

        使用Hamming距离的变体：比较两个route_array中不同位置的数量

        Args:
            ind1, ind2: 两个个体字典

        Returns:
            distance: 归一化距离 [0, 1]
        """
        array1 = ind1["route_array"]
        array2 = ind2["route_array"]

        if len(array1) != len(array2):
            return 1.0  # 完全不同

        # 计算不同位置的数量
        diff_count = np.sum(array1 != array2)
        max_diff = len(array1)

        # 归一化到[0, 1]
        distance = diff_count / max_diff if max_diff > 0 else 0.0

        return distance

    def _clearing_procedure(self):
        """
        清除程序 (Clearing Procedure, Algorithm 6, Page 17)

        基于Petrowski (1996)的Niching技术：
        1. 按适应度排序种群
        2. 对每个个体，清除其niching半径内的劣质个体
        3. 被清除的个体标记为'cleared'，不参与选择

        这确保了种群多样性，避免过早收敛
        """
        # Step 1: 按适应度排序（使用approximate_fitness）
        sorted_pop = sorted(self.population, key=lambda x: x["approx_fitness"])

        # Step 2: 清除程序
        for i, ind_i in enumerate(sorted_pop):
            if ind_i.get("cleared", False):
                continue  # 已被清除，跳过

            # 检查后续个体是否在niching半径内
            for j in range(i + 1, len(sorted_pop)):
                ind_j = sorted_pop[j]

                if ind_j.get("cleared", False):
                    continue  # 已被清除

                # 计算距离
                distance = self._calculate_niching_distance(ind_i, ind_j)

                # 如果在niching半径内，清除劣质个体
                if distance < self.niching_radius:
                    ind_j["cleared"] = True

    def _calculate_violation(self, individual):
        """
        计算约束违反量 (Algorithm 7, Page 18) - 扩展为Open VRP

        【修复】检查以下约束：
        1. 容量约束：路由需求不超过可用车辆容量
        2. 【新增】车辆数量约束：每个物种使用的车辆数不超过species_dict['number']
        3. 时间约束：路由时间不超过最大时间限制（考虑起点和终点仓库）
        4. 库存约束：仓库库存足够满足所有路由需求

        Args:
            individual: 个体字典

        Returns:
            violation: 总违反量（越小越好，0表示可行）
        """
        inst = self.instance
        routes = individual["routes"]
        route_start_depots = individual.get("route_start_depots", [])
        route_end_depots = individual.get("route_end_depots", [])
        total_violation = 0.0

        # 【新增】跟踪所有路由中每个物种已使用的车辆总数
        global_species_used = {s: 0 for s in range(inst.num_species)}

        # 对每条路由检查约束
        for route_idx, route in enumerate(routes):
            if not route:
                continue

            # 计算路由需求
            route_demand = np.zeros(inst.num_commodities)
            for task_id in route:
                route_demand += inst.task_requirements[task_id]

            # 检查容量约束（使用贪婪车辆分配）
            # 【修改】每辆车每次只能装载一种货物类型，与RL环境保持一致
            remaining_demand = route_demand.copy()

            # 按容量从大到小尝试分配车辆
            species_capacities = [
                (s, inst.species_dict["capacities"][s]) for s in range(inst.num_species)
            ]
            species_capacities.sort(key=lambda x: np.sum(x[1]), reverse=True)

            for species_id, capacity_vec in species_capacities:
                # 【新增】获取该物种的可用车辆数量限制
                max_vehicles_for_species = inst.species_dict["number"][species_id]
                vehicles_used_for_species = 0

                while np.any(remaining_demand > 0):
                    # 【新增】检查该物种的车辆是否已用完
                    if (
                        global_species_used[species_id] + vehicles_used_for_species
                        >= max_vehicles_for_species
                    ):
                        break  # 该物种车辆已用完，尝试下一个物种

                    # 【修改】每辆车只能装载一种货物类型
                    # 选择剩余需求最大的货物类型
                    can_carry = False
                    best_cargo_type = -1
                    best_cargo_demand = 0

                    for c in range(inst.num_commodities):
                        if remaining_demand[c] > 0 and capacity_vec[c] > 0:
                            can_carry = True
                            if remaining_demand[c] > best_cargo_demand:
                                best_cargo_demand = remaining_demand[c]
                                best_cargo_type = c

                    if not can_carry:
                        break

                    # 【修改】只装载选定的一种货物类型
                    carry_amount = np.zeros(inst.num_commodities)
                    carry_amount[best_cargo_type] = min(
                        remaining_demand[best_cargo_type], capacity_vec[best_cargo_type]
                    )
                    remaining_demand -= carry_amount
                    vehicles_used_for_species += 1

                # 【新增】更新全局物种使用计数
                global_species_used[species_id] += vehicles_used_for_species

                if np.all(remaining_demand <= 0):
                    break

            # 如果仍有未满足需求，记录违反量
            capacity_violation = np.sum(np.maximum(0, remaining_demand))
            total_violation += capacity_violation

            # 检查时间约束（考虑起点和终点仓库）
            if len(route) >= 1:
                # 获取起点和终点仓库
                if (
                    route_idx < len(route_start_depots)
                    and route_start_depots[route_idx] is not None
                ):
                    start_depot = route_start_depots[route_idx]
                    end_depot = route_end_depots[route_idx]

                    # 确保end_depot不是None
                    if end_depot is None:
                        end_depot = start_depot
                else:
                    # 如果没有指定，使用最近的仓库
                    first_task_loc = inst.locations[inst.depots[-1] + 1 + route[0]]
                    closest_depot = None
                    min_depot_dist = float("inf")
                    for depot_id in inst.depots:
                        depot_loc = inst.locations[depot_id]
                        dist = np.linalg.norm(first_task_loc - depot_loc)
                        if dist < min_depot_dist:
                            min_depot_dist = dist
                            closest_depot = depot_id
                    start_depot = closest_depot
                    end_depot = closest_depot

                # 计算路由距离（起点仓库 → 任务序列 → 终点仓库）
                route_distance = 0.0

                # 从起点仓库到第一个任务
                start_depot_loc = inst.locations[start_depot]
                first_task_node = inst.depots[-1] + 1 + route[0]
                first_task_loc = inst.locations[first_task_node]
                route_distance += np.linalg.norm(first_task_loc - start_depot_loc)

                # 任务之间的距离
                prev_task_id = route[0]
                for next_task_id in route[1:]:
                    prev_node = inst.depots[-1] + 1 + prev_task_id
                    next_node = inst.depots[-1] + 1 + next_task_id
                    route_distance += inst.distance_matrix.get(
                        (prev_node, next_node), 0
                    )
                    prev_task_id = next_task_id

                # 从最后一个任务到终点仓库
                last_task_node = inst.depots[-1] + 1 + route[-1]
                last_task_loc = inst.locations[last_task_node]
                end_depot_loc = inst.locations[end_depot]
                route_distance += np.linalg.norm(end_depot_loc - last_task_loc)

                # 时间估算：distance / velocity + 装卸时间（与task_env.py保持一致）
                route_time = route_distance / inst.agent_velocity
                for task_id in route:
                    # 计算装卸时间：每个任务需要的装卸次数 = 2 × 货物类型数量
                    num_cargo_types = np.count_nonzero(inst.task_requirements[task_id])
                    num_operations = num_cargo_types * 2  # LOAD + UNLOAD
                    # 使用0.1与task_env.py保持一致（而不是inst.load_unload_time=0.2）
                    load_unload_time = 0.1
                    route_time += num_operations * load_unload_time

                # 如果超过最大时间，记录违反量
                if route_time > inst.max_time:
                    time_violation = route_time - inst.max_time
                    total_violation += time_violation

        # 检查全局库存约束
        total_demand = np.zeros(inst.num_commodities)
        for task_id in inst.tasks:
            total_demand += inst.task_requirements[task_id]

        total_stock = np.zeros(inst.num_commodities)
        for depot_stock in inst.depot_stock.values():
            for c, qty in depot_stock.items():
                total_stock[c] += qty

        stock_violation = np.sum(np.maximum(0, total_demand - total_stock))
        total_violation += stock_violation

        return total_violation

    def run(self):
        """
        主遗传算法循环 (Algorithm 1, Page 10)

        核心流程：
        1. 初始化种群
        2. 对每一代：
           - 评估适应度（前SwitchGen代用近似，之后用精确）
           - 应用clearing procedure（niching维持多样性）
           - 选择、交叉、变异生成子代
           - 精英保留策略
        3. 返回最优解

        Returns:
            best_solution: 最优解字典
        """
        import time

        print(
            f"Running GA: pop_size={self.pop_size}, generations={self.max_generations}"
        )
        print(
            f"Crossover rate={self.crossover_rate}, Mutation rate={self.mutation_rate}"
        )
        print(f"Niching radius={self.niching_radius}")
        if self.time_limit:
            print(f"Time limit={self.time_limit}s")

        # 记录开始时间
        self.start_time = time.time()
        self.convergence_history = []  # 重置收敛历史

        # Step 1: 初始化种群
        self.initialize_population()

        # 【关键修改 - 基于Q1论文 Expert Systems with Applications 2025】
        # 论文明确指出："The fitness value is calculated by simulating the execution of the solution"
        # 因此我们从一开始就使用精确仿真评估，确保fitness值准确反映解的质量
        # 这避免了近似评估的误差导致搜索方向偏离
        use_exact = True  # 始终使用精确评估

        # 主进化循环
        for gen in range(self.max_generations):
            # 检查时间限制
            if self.time_limit and (time.time() - self.start_time) > self.time_limit:
                print(f"\n⏰ 达到时间限制 ({self.time_limit}s)，停止搜索")
                print(f"   完成代数: {gen}/{self.max_generations}")
                break

            # Step 2: 评估种群适应度（始终使用精确评估）
            for individual in self.population:
                if individual["approx_fitness"] == float("inf"):
                    self.evaluate_fitness(individual, use_exact=use_exact)

            # Step 3: 应用Clearing Procedure（维持多样性）
            self._clearing_procedure()

            # Step 4: 更新全局最优解
            valid_individuals = [
                ind for ind in self.population if not ind.get("cleared", False)
            ]
            if valid_individuals:
                current_best = min(valid_individuals, key=lambda x: x["approx_fitness"])
                if current_best["approx_fitness"] < self.best_fitness:
                    self.best_fitness = current_best["approx_fitness"]
                    self.best_solution = current_best.copy()

            # 记录收敛历史（每代记录一次）
            elapsed = time.time() - self.start_time
            self.convergence_history.append(
                {"time": elapsed, "best_fitness": self.best_fitness, "generation": gen}
            )

            # Step 5: 生成新种群
            new_population = []

            # 精英保留（保留最好的10%）
            elite_size = max(1, int(0.1 * self.pop_size))
            sorted_pop = sorted(self.population, key=lambda x: x["approx_fitness"])
            elites = sorted_pop[:elite_size]
            new_population.extend([ind.copy() for ind in elites])

            # 生成剩余个体
            while len(new_population) < self.pop_size:
                # 选择父代
                parent1 = self.selection()
                parent2 = self.selection()

                # 交叉
                if np.random.random() < self.crossover_rate:
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                else:
                    offspring1 = parent1.copy()
                    offspring2 = parent2.copy()

                # 变异
                if np.random.random() < self.mutation_rate:
                    offspring1 = self.mutation(offspring1)
                if np.random.random() < self.mutation_rate:
                    offspring2 = self.mutation(offspring2)

                # 添加到新种群
                new_population.append(offspring1)
                if len(new_population) < self.pop_size:
                    new_population.append(offspring2)

            # 更新种群
            self.population = new_population[: self.pop_size]

            # 重置cleared标记
            for ind in self.population:
                ind["cleared"] = False

            # 打印进度
            if gen % 50 == 0:
                avg_fitness = np.mean(
                    [
                        ind["approx_fitness"]
                        for ind in self.population
                        if ind["approx_fitness"] < float("inf")
                    ]
                )
                elapsed = time.time() - self.start_time
                print(
                    f"Generation {gen}/{self.max_generations}: "
                    f"Best={self.best_fitness:.2f}, Avg={avg_fitness:.2f}, "
                    f"Time={elapsed:.1f}s"
                )

        print(f"\nGA finished. Best fitness: {self.best_fitness:.2f}")

        # 确保最优解的所有智能体都被使用
        if self.best_solution:
            self.best_solution["routes"] = self._ensure_all_agents_used(
                self.best_solution["routes"]
            )
            print(f"Best solution routes: {self.best_solution['routes']}")
        else:
            print(f"Best solution routes: None")

        return self.best_solution


def run_on_benchmark(
    folder="RALTestSet", method="milp_ga", use_milp=False, use_ga=True
):
    """
    在benchmark数据集上运行MILP+GA方法

    Args:
        folder: benchmark文件夹路径
        method: 方法名称
        use_milp: 是否使用MILP求解器
        use_ga: 是否使用GA求解器
    """
    files = natsorted(glob.glob(f"../{folder}/env_*.pkl"), key=lambda y: y.lower())
    perf_metrics = {
        "success_rate": [],
        "makespan": [],
        "time_cost": [],
        "waiting_time": [],
        "travel_dist": [],
        "efficiency": [],
        "solve_time": [],
    }

    for i, file_path in enumerate(files):
        print(f"\n{'=' * 60}")
        print(f"Processing [{i + 1}/{len(files)}]: {file_path}")
        print("=" * 60)

        # 加载环境
        env = pickle.load(open(file_path, "rb"))
        env.init_state()

        # 转换为GreenVRPInstance
        instance = GreenVRPInstance.from_env(env)
        instance.summary()

        start_time = time.time()

        # 选择求解方法
        if use_milp:
            # 使用MILP求解
            solver = MILPSolver(instance, time_limit=300.0)  # 5分钟时间限制
            solution = solver.solve()
            makespan = solution["makespan"]
            success = solution["success"]

        elif use_ga:
            # 使用GA求解
            ga = GeneticAlgorithm(instance, pop_size=50, max_generations=200)
            best_solution = ga.run()

            # 从最优解提取makespan（需要计算完成所有任务的时间）
            if best_solution and best_solution["approx_fitness"] < float("inf"):
                makespan = best_solution["approx_fitness"]  # 使用碳排放作为目标
                success = True
            else:
                makespan = float("inf")
                success = False

        else:
            print("Error: Must specify either use_milp=True or use_ga=True")
            continue

        solve_time = time.time() - start_time

        # 记录性能指标
        if success and makespan < float("inf"):
            perf_metrics["success_rate"].append(1.0)
            perf_metrics["makespan"].append(makespan)
            perf_metrics["time_cost"].append(makespan)  # 简化
            perf_metrics["waiting_time"].append(0.0)  # TODO: 计算
            perf_metrics["travel_dist"].append(0.0)  # TODO: 计算
            perf_metrics["efficiency"].append(1.0)  # TODO: 计算
            perf_metrics["solve_time"].append(solve_time)
        else:
            perf_metrics["success_rate"].append(0.0)
            perf_metrics["makespan"].append(np.nan)
            perf_metrics["time_cost"].append(np.nan)
            perf_metrics["waiting_time"].append(np.nan)
            perf_metrics["travel_dist"].append(np.nan)
            perf_metrics["efficiency"].append(np.nan)
            perf_metrics["solve_time"].append(solve_time)

        print(
            f"Result: Success={success}, Makespan={makespan:.2f}, Time={solve_time:.2f}s"
        )

    # 保存结果
    df = pd.DataFrame(perf_metrics)
    output_path = f"../{folder}/{method}_results.csv"
    df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print("Benchmark Results Summary")
    print("=" * 60)
    print(df.describe())
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    # 测试代码
    print("MILP + GA for Green VRP (Kabadurmus & Erdogan 2023)")
    print("=" * 60)

    # 选项1: 在benchmark上运行MILP (小规模问题)
    # run_on_benchmark(folder='RALTestSet', method='milp', use_milp=True, use_ga=False)

    # 选项2: 在benchmark上运行GA (大规模问题)
    # run_on_benchmark(folder='RALTestSet', method='ga', use_milp=False, use_ga=True)

    # 选项3: 测试单个实例（快速验证）
    print("\n测试模式：在单个随机实例上运行GA")
    print("-" * 60)

    # 创建一个简单的测试环境
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
        plot_figure=False,
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

    # 运行GA
    print("\n开始运行遗传算法...")
    ga = GeneticAlgorithm(
        instance,
        pop_size=30,  # 小规模测试用较小种群
        max_generations=100,  # 测试用较少代数
        crossover_rate=0.8,
        mutation_rate=0.2,
        niching_radius=0.1,
    )

    best_solution = ga.run()

    if best_solution:
        print("\n" + "=" * 60)
        print("最优解找到！")
        print("=" * 60)
        print(f"适应度值（近似碳排放）: {best_solution['approx_fitness']:.2f}")
        print(f"路由数量: {len(best_solution['routes'])}")
        print(f"约束违反量: {best_solution.get('violation', 0):.4f}")
        print("\n路由详情:")
        for i, route in enumerate(best_solution["routes"]):
            print(f"  Route {i + 1}: {route}")
    else:
        print("\n未找到可行解")

    print("\n" + "=" * 60)
    print("实现说明:")
    print("=" * 60)
    print("✅ 已完成的组件:")
    print("  - 种群初始化（(2n-1)编码）")
    print("  - 近似碳排放计算（Algorithm 3）")
    print("  - 两点交叉+修复机制（Figure 2）")
    print("  - 三种变异算子（Block Insertion/Swap/2-opt）")
    print("  - Niching距离计算（Algorithm 5）")
    print("  - Clearing Procedure（Algorithm 6）")
    print("  - 约束违反计算（Algorithm 7）")
    print("  - ε-constraint方法（适应度评估）")
    print("  - 锦标赛选择")
    print("  - 主GA循环（Algorithm 1）")
    print("\n⚠️  简化/待完善:")
    print("  - MILP精确评估（当前使用近似值代替）")
    print("  - VNS局部搜索（Algorithm 2，可选）")
    print("  - 更详细的时间窗约束建模")
    print("\n💡 使用方法:")
    print("  - 取消注释run_on_benchmark()调用以在完整数据集上运行")
    print("  - 调整GA参数（pop_size, max_generations）以平衡质量和速度")
