"""Shared LTL-shielded simulator and metric helpers used by every baseline
(GA, AVNR, Greedy) at evaluation time.

This module exposes three things:

- ``BENCHMARK_CONFIGS`` / ``TIME_LIMITS`` / ``GA_PARAMS`` / ``AVNR_PARAMS``:
  the per-tier configuration that produced the numbers reported in the
  paper (Tables III / IV).
- ``generate_fixed_ltl_constraints``: deterministic generator of the
  paper's LTL clause set for a given environment + tier.
- ``simulate_solution_execution`` and ``calculate_solution_metrics``:
  the same shielded executor and metric extractor used by every baseline
  in the paper. Identical to the version in
  ``RSS_2026/ALIS-WC_code/benchmark_ijcai.py`` at the time of submission.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from env.task_env import TaskEnv
from env.ltl_utils import LTLMonitor, LTL_SAFETY, LTL_SEQUENTIAL, LTLClause


# ---------------------------------------------------------------------------
# Per-tier benchmark configuration matching the paper (Tables III / IV).
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS = {
    "tier1": {"depots": 3, "species": 3, "agents_per_species": 3, "tasks": 15,
              "ltl_safety": 2,  "ltl_sequential": 2},
    "tier2": {"depots": 4, "species": 4, "agents_per_species": 4, "tasks": 40,
              "ltl_safety": 3,  "ltl_sequential": 3},
    "tier3": {"depots": 5, "species": 5, "agents_per_species": 5, "tasks": 65,
              "ltl_safety": 4,  "ltl_sequential": 4},
    "tier4": {"depots": 6, "species": 6, "agents_per_species": 6, "tasks": 100,
              "ltl_safety": 30, "ltl_sequential": 30},
}

TIME_LIMITS = {"tier1": 30.0, "tier2": 60.0, "tier3": 120.0, "tier4": 240.0}

GA_PARAMS = {
    "tier1": {"pop_size": 50,  "max_generations": 10000},
    "tier2": {"pop_size": 100, "max_generations": 10000},
    "tier3": {"pop_size": 150, "max_generations": 10000},
    "tier4": {"pop_size": 200, "max_generations": 10000},
}

AVNR_PARAMS = {
    "tier1": {"max_iterations": 10000},
    "tier2": {"max_iterations": 10000},
    "tier3": {"max_iterations": 10000},
    "tier4": {"max_iterations": 10000},
}


def generate_fixed_ltl_constraints(env: TaskEnv, num_safety: int, num_sequential: int, seed: int) -> List[Tuple]:
    """
    生成固定的LTL约束（用于可复现的benchmark）

    Args:
        env: 环境实例
        num_safety: SAFETY约束数量
        num_sequential: SEQUENTIAL约束数量
        seed: 随机种子

    Returns:
        constraints: 约束列表 [(type, param1, param2), ...]
    """
    rng = np.random.default_rng(seed + 1000)  # 使用不同的种子避免与环境生成冲突

    constraints = []

    # 生成SAFETY约束: "Agent X cannot visit Task Y"
    # 只禁止访问任务节点（不禁止仓库）
    if num_safety > 0 and env.agents_num > 0 and env.tasks_num > 0:
        for _ in range(num_safety):
            agent_id = rng.integers(0, env.agents_num)
            # node_id是目标节点ID（depot在0..depots-1，task在depots..depots+tasks-1）
            # 我们只禁止访问任务节点
            task_id = rng.integers(0, env.tasks_num)
            node_id = env.depots_num + task_id  # 转换为节点ID
            constraints.append((LTL_SAFETY, agent_id, node_id))

    # 生成SEQUENTIAL约束: "Task A must precede Task B"
    if num_sequential > 0 and env.tasks_num >= 2:
        used_pairs = set()
        attempts = 0
        max_attempts = num_sequential * 10

        while len([c for c in constraints if c[0] == LTL_SEQUENTIAL]) < num_sequential and attempts < max_attempts:
            attempts += 1
            # 随机选择两个不同的任务
            task_ids = rng.choice(list(env.task_dic.keys()), 2, replace=False)
            pair = tuple(sorted(task_ids))

            if pair not in used_pairs:
                used_pairs.add(pair)
                # 随机决定顺序（避免固定模式）
                if rng.random() < 0.5:
                    constraints.append((LTL_SEQUENTIAL, task_ids[0], task_ids[1]))
                else:
                    constraints.append((LTL_SEQUENTIAL, task_ids[1], task_ids[0]))

    print(f"  成功生成 {len([c for c in constraints if c[0] == LTL_SAFETY])} 个SAFETY约束")
    print(f"  成功生成 {len([c for c in constraints if c[0] == LTL_SEQUENTIAL])} 个SEQUENTIAL约束")

    return constraints



def _check_and_wakeup_agents(env: TaskEnv, ltl_monitor: Optional[LTLMonitor], agent_states: Dict):
    """
    检查并唤醒临时休眠的agents（复用RL/Greedy的逻辑）
    使用agent_observe作为唯一权威来判断是否应该唤醒

    Args:
        env: 环境实例
        ltl_monitor: LTL监视器
        agent_states: agent状态字典（用于更新finished标记）
    """
    for agent_id, agent in env.agent_dic.items():
        if agent.get('is_temp_sleeping'):
            # 使用agent_observe判断是否应该唤醒
            _, _, _, _, _, _, inaction_reason, _ = env.agent_observe(
                agent_id,
                ltl_monitor,
                max_waiting=False,
                ignore_sleeping=True,
            )

            # 唤醒条件：在当前新的世界状态下，智能体有可用的动作
            if inaction_reason == 'ACTIONS_AVAILABLE':
                agent['is_temp_sleeping'] = False
                agent['next_decision'] = env.current_time
                agent['blocking_clauses'] = []


def simulate_solution_execution(
    env: TaskEnv,
    solution: Dict,
    ltl_monitor: Optional[LTLMonitor],
    debug: bool = False,
    decide_quantity: bool = False,
    use_in_flight_reservation: bool = True
) -> Tuple[float, float, float, float]:
    """
    使用真实环境逻辑执行solution，获取准确的makespan和其他指标

    完全使用TaskEnv的agent_step逻辑，包括：
    - 多agent决策队列
    - 资源竞争（depot库存）
    - 真实的时间推进
    - 装卸货时间
    - 任务执行时间
    - 【新增】LTL约束检查（SAFETY和SEQUENTIAL）

    LTL处理策略（统一评估，不帮GA做LTL-aware优化）：
    - SAFETY约束：如果agent被禁止访问某任务，跳过该任务
    - SEQUENTIAL约束：如果前置任务未完成，跳过当前任务
    - GA是离线规划，无法动态调整，跳过的任务可能导致完成率<100%

    Args:
        env: 环境实例
        solution: GA/ALNS solution，包含routes和quantity_ratios
        ltl_monitor: LTL监视器
        debug: 是否打印调试信息
        decide_quantity: 是否使用solution中的quantity_ratios决定装货数量
        use_in_flight_reservation: 是否启用在途货物预留（GA/AVNR应设为False）

    Returns:
        makespan: 最大完成时间
        travel_distance: 总旅行距离
        time_cost: 总时间成本
        task_completion_rate: 任务完成率
    """
    routes = solution.get('routes', [])
    quantity_ratios = solution.get('quantity_ratios', {})
    cargo_type_priority = solution.get('cargo_type_priority', {})  # 【新增】货物类型优先级

    if not routes:
        return float('inf'), float('inf'), float('inf'), 0.0

    # 重置环境到初始状态（使用init_state而不是reset，避免重新生成环境）
    env.init_state()

    # 【新增】设置在途货物预留开关（GA/AVNR不需要预留，避免掩码冲突）
    env.use_in_flight_reservation = use_in_flight_reservation

    # 【LTL】重置LTL监视器状态（用于多次评估之间）
    if ltl_monitor:
        ltl_monitor.reset_states()

    # 【LTL】解析LTL约束，构建快速查找结构
    safety_forbidden = {}  # {agent_id: set(forbidden_node_ids)}
    sequential_prereqs = {}  # {task_id: set(prerequisite_task_ids)}

    if ltl_monitor and ltl_monitor.clauses:
        for clause in ltl_monitor.clauses:
            if clause.type == LTL_SAFETY:
                # SAFETY: Agent param1 cannot visit Node param2
                agent_id = clause.param1
                node_id = clause.param2
                if agent_id not in safety_forbidden:
                    safety_forbidden[agent_id] = set()
                safety_forbidden[agent_id].add(node_id)
            elif clause.type == LTL_SEQUENTIAL:
                # SEQUENTIAL: Task param1 must precede Task param2
                prereq_task = clause.param1
                dependent_task = clause.param2
                if dependent_task not in sequential_prereqs:
                    sequential_prereqs[dependent_task] = set()
                sequential_prereqs[dependent_task].add(prereq_task)

    if debug and ltl_monitor:
        print(f"[LTL] SAFETY forbidden: {safety_forbidden}")
        print(f"[LTL] SEQUENTIAL prereqs: {sequential_prereqs}")

    # 为每个agent分配route
    # 假设routes[i]对应agent i
    agent_routes = {}
    for agent_id in range(min(len(routes), env.agents_num)):
        if routes[agent_id]:
            agent_routes[agent_id] = list(routes[agent_id])

    # 为每个agent创建任务队列和状态跟踪
    agent_states = {}
    for agent_id in range(env.agents_num):
        agent = env.agent_dic[agent_id]
        agent_states[agent_id] = {
            'route': agent_routes.get(agent_id, []),
            'route_index': 0,  # 当前正在执行的任务索引
            'current_task_cargo_types': [],  # 当前任务需要运输的货物类型列表
            'current_cargo_index': 0,  # 当前正在运输的货物类型索引
            'finished': False,
            'quantity_ratios': quantity_ratios.get(agent_id, {}),  # 该agent的装货比例
            'cargo_type_priority': cargo_type_priority.get(agent_id, {}),  # 【新增】货物类型优先级
            'skipped_tasks': []  # 【LTL】被跳过的任务（因LTL约束）
        }

    # 主循环：模拟环境执行（复用worker.py的事件循环机制）
    max_steps = 10000
    step_count = 0

    while step_count < max_steps:
        step_count += 1

        # 检查是否所有任务完成
        if env.finished:
            break

        # 【复用环境API】使用env.next_decision()获取下一个事件时间
        (ready_agents_temp, blocked_agents_temp), next_event_time = env.next_decision()

        # 检查是否有未来事件
        if np.isinf(next_event_time) or np.isnan(next_event_time):
            break

        # 推进时间到下一个事件点
        env.current_time = next_event_time

        # 【复用环境API】调用agent_update和task_update处理到达事件
        env.agent_update()
        finished_task_ids = env.task_update()

        # 【LTL】在任务完成时立即更新SEQUENTIAL约束状态
        if ltl_monitor and finished_task_ids:
            for tid in finished_task_ids:
                ltl_monitor.update_monitor('task_finish', {'task_id': tid})

        # 收集当前时刻需要决策的agents（排除已完成路由的）
        # 【关键修复】对于已完成路由但被agent_update唤醒的agent，重置其next_decision为inf
        agents_to_process = []
        for aid, a in env.agent_dic.items():
            if a.get('next_decision', np.inf) == env.current_time:
                if agent_states[aid]['finished']:
                    # 已完成路由的agent，设置next_decision=inf以避免阻塞时间推进
                    a['next_decision'] = np.inf
                else:
                    agents_to_process.append(aid)

        if not agents_to_process:
            continue

        # 依次处理队列中的agents
        for agent_id in agents_to_process:
            agent = env.agent_dic[agent_id]
            state = agent_states[agent_id]

            # 【关键修复】使用agent_observe获取掩码，确保动作合法性
            # 这与RL和Greedy的逻辑一致，保证公平对比
            task_info, total_agents, global_mask, ltl_info, masks_dict, cost_info, inaction_reason, blocking_clauses = \
                env.agent_observe(agent_id, ltl_monitor, max_waiting=False)

            # 处理inaction_reason（复用RL/Greedy的逻辑）
            if inaction_reason == 'NO_ACTION_BY_SAFETY_LTL':
                # 被静态安全约束永久阻塞 -> 标记完成
                state['finished'] = True
                agent['is_inactive'] = True
                agent['next_decision'] = np.inf
                continue
            elif inaction_reason == 'NO_ACTION_BY_LTL':
                # 被动态LTL约束阻塞 -> 临时休眠
                agent['is_temp_sleeping'] = True
                agent['next_decision'] = np.inf
                agent['blocking_clauses'] = blocking_clauses
                continue
            elif inaction_reason == 'NO_ACTION_TEMPORARILY':
                # 当前无可执行动作 -> 临时休眠
                agent['is_temp_sleeping'] = True
                agent['next_decision'] = np.inf
                continue
            elif inaction_reason == 'NO_ACTION_BY_DEFAULT':
                # 永久无法贡献 -> 标记完成
                state['finished'] = True
                agent['is_inactive'] = True
                agent['next_decision'] = np.inf
                continue
            elif inaction_reason == 'TEMP_SLEEPING':
                continue

            # inaction_reason == 'ACTIONS_AVAILABLE'，可以行动
            # 决定下一步动作（传入masks_dict用于验证动作合法性）
            action = _decide_next_action(
                env, agent_id, state, decide_quantity,
                safety_forbidden, sequential_prereqs, debug,
                masks_dict  # 【新增】传入掩码
            )

            if action is None:
                # 该agent完成所有任务（或所有剩余任务都被LTL阻塞）
                state['finished'] = True
                agent['next_decision'] = np.inf
                continue

            # 执行动作
            reward, done, events, event_info = env.agent_step(agent_id, action)

            # 【LTL】更新LTL监视器状态（SAFETY约束：agent_move事件）
            if ltl_monitor and event_info:
                ltl_monitor.update_monitor(event_info['type'], event_info['params'])

                # 【复用RL/Greedy逻辑】当任务完成时，检查是否需要唤醒休眠的agents
                if event_info['type'] == 'task_finish':
                    _check_and_wakeup_agents(env, ltl_monitor, agent_states)

    # 统计完成的任务
    finished_tasks = [tid for tid, task in env.task_dic.items() if task.get('finished', False)]

    # 计算makespan
    makespan = env.current_time

    # 计算总旅行距离
    total_distance = sum(
        env.agent_dic[aid].get('travel_dist', 0.0)
        for aid in range(env.agents_num)
    )

    # 计算任务完成率
    task_completion_rate = len(finished_tasks) / env.tasks_num if env.tasks_num > 0 else 0.0

    if debug:
        # 打印被跳过的任务统计
        total_skipped = sum(len(s['skipped_tasks']) for s in agent_states.values())
        if total_skipped > 0:
            print(f"[LTL] 总共跳过 {total_skipped} 个任务分配（因LTL约束）")
            for aid, s in agent_states.items():
                if s['skipped_tasks']:
                    print(f"  Agent {aid}: 跳过 {s['skipped_tasks']}")

    return makespan, total_distance, makespan, task_completion_rate


def _decide_next_action(env: TaskEnv, agent_id: int, state: Dict, decide_quantity: bool = False,
                        safety_forbidden: Dict = None, sequential_prereqs: Dict = None,
                        debug: bool = False, masks_dict: Dict = None,
                        _recursion_depth: int = 0) -> Optional[Dict]:
    """
    根据agent的route和当前状态，决定下一步动作

    状态机：
    1. idle -> 选择下一个任务 -> moving_to_depot
    2. moving_to_depot -> 到达depot -> loading
    3. loading -> 装货完成 -> moving_to_task
    4. moving_to_task -> 到达任务点 -> unloading
    5. unloading -> 卸货完成 -> 检查是否需要更多货物
       - 如果当前任务还有其他货物类型 -> moving_to_depot
       - 如果当前任务完成 -> 选择下一个任务

    【关键修复】使用masks_dict验证动作合法性，确保与RL/Greedy公平对比

    Args:
        env: 环境实例
        agent_id: agent ID
        state: agent状态字典
        decide_quantity: 是否使用quantity_ratios决定装货数量
        safety_forbidden: {agent_id: set(forbidden_node_ids)} SAFETY约束
        sequential_prereqs: {task_id: set(prerequisite_task_ids)} SEQUENTIAL约束
        debug: 是否打印调试信息
        masks_dict: 【新增】从agent_observe获取的掩码字典，用于验证动作合法性
        _recursion_depth: 【内部】递归深度计数器，防止无限递归
    """
    # 【防止无限递归】检查递归深度
    MAX_RECURSION_DEPTH = 500
    if _recursion_depth > MAX_RECURSION_DEPTH:
        if debug:
            print(f"  [MAX-RECURSION] Agent {agent_id} 递归深度超过 {MAX_RECURSION_DEPTH}，强制结束")
        return None
    if safety_forbidden is None:
        safety_forbidden = {}
    if sequential_prereqs is None:
        sequential_prereqs = {}
    if masks_dict is None:
        masks_dict = {}
    agent = env.agent_dic[agent_id]
    route = state['route']

    # 如果route为空，agent完成
    if not route or state['route_index'] >= len(route):
        return None

    current_location = agent['current_task']
    inventory = agent['inventory']

    # 获取当前任务
    if state['route_index'] < len(route):
        current_task_id = int(route[state['route_index']])  # GA route中的task ID
    else:
        return None

    # 【关键修复】检查任务是否已完成（被其他agent完成）
    task = env.task_dic[current_task_id]
    if task.get('finished', False):
        # 任务已完成，跳到下一个任务
        state['route_index'] += 1
        state['current_task_cargo_types'] = []
        state['current_cargo_index'] = 0
        return _decide_next_action(env, agent_id, state, decide_quantity,
                                   safety_forbidden, sequential_prereqs, debug, masks_dict)

    # 【修复】将GA的task_id转换为环境的节点ID
    # GA route中存储的是task_id (0, 1, 2, ...)
    # 环境中task的节点ID是 depots_num + task_id
    current_task_node_id = env.depots_num + current_task_id

    # 【LTL约束检查】
    # 1. SAFETY约束：检查agent是否被禁止访问该任务节点
    if current_task_node_id in safety_forbidden.get(agent_id, set()):
        if debug:
            print(f"  [LTL-SAFETY] Agent {agent_id} 被禁止访问任务 {current_task_id}，跳过")
        # 记录跳过的任务
        if 'skipped_tasks' not in state:
            state['skipped_tasks'] = []
        state['skipped_tasks'].append(('SAFETY', current_task_id))
        # 跳到下一个任务
        state['route_index'] += 1
        state['current_task_cargo_types'] = []
        state['current_cargo_index'] = 0
        return _decide_next_action(env, agent_id, state, decide_quantity,
                                   safety_forbidden, sequential_prereqs, debug, masks_dict)

    # 2. SEQUENTIAL约束：检查前置任务是否已完成
    if current_task_id in sequential_prereqs:
        prereqs = sequential_prereqs[current_task_id]
        for prereq_task_id in prereqs:
            prereq_task = env.task_dic[prereq_task_id]
            if not prereq_task.get('finished', False):
                if debug:
                    print(f"  [LTL-SEQUENTIAL] 任务 {current_task_id} 的前置任务 {prereq_task_id} 未完成，跳过")
                # 记录跳过的任务
                if 'skipped_tasks' not in state:
                    state['skipped_tasks'] = []
                state['skipped_tasks'].append(('SEQUENTIAL', current_task_id, prereq_task_id))
                # 跳到下一个任务
                state['route_index'] += 1
                state['current_task_cargo_types'] = []
                state['current_cargo_index'] = 0
                return _decide_next_action(env, agent_id, state, decide_quantity,
                                           safety_forbidden, sequential_prereqs, debug, masks_dict,
                                           _recursion_depth + 1)

    # 如果是新任务，初始化货物类型列表
    # 【关键修改】使用cargo_type_priority决定货物类型顺序（公平对比RL）
    if not state['current_task_cargo_types']:
        # 先获取所有任务需要的货物类型
        all_needed_cargo_types = [
            ct for ct in range(env.traits_dim) if task['status'][ct] > 0
        ]

        # 获取该agent对该task的货物类型优先级
        cargo_priority = state.get('cargo_type_priority', {})
        if current_task_id in cargo_priority and cargo_priority[current_task_id]:
            # 【关键修复】AVNR/GA的cargo_type_priority是排他性的：
            # 每个agent只负责运送其被分配的货物类型，不要尝试运送其他类型
            # （其他类型由其他agent负责）
            priority_list = cargo_priority[current_task_id]
            # 只使用被分配的货物类型（且任务仍需要）
            cargo_types = [ct for ct in priority_list if ct in all_needed_cargo_types]
        else:
            # 兼容旧版本：按固定顺序遍历所有需要的货物类型
            cargo_types = all_needed_cargo_types

        state['current_task_cargo_types'] = cargo_types
        state['current_cargo_index'] = 0

    # 如果当前任务的所有货物都运输完成（或不需要运输）
    if state['current_cargo_index'] >= len(state['current_task_cargo_types']):
        # 【关键修复】检查任务是否仍需要agent被分配运输的货物（多趟运输支持）
        # 重新获取任务当前状态
        cargo_priority = state.get('cargo_type_priority', {})
        if current_task_id in cargo_priority and cargo_priority[current_task_id]:
            # 如果有cargo_type_priority，只考虑被分配的货物类型
            assigned_cargo_types = cargo_priority[current_task_id]
            still_needed_cargo_types = [
                ct for ct in assigned_cargo_types
                if task['status'][ct] > 0 and agent['capacity'][ct] > 0
            ]
        else:
            # 兼容旧版本：考虑所有agent能运输的货物类型
            still_needed_cargo_types = [
                ct for ct in range(env.traits_dim)
                if task['status'][ct] > 0 and agent['capacity'][ct] > 0
            ]

        if still_needed_cargo_types:
            # 任务仍有agent能运输的货物需求

            # 【防止无限循环】检查多趟尝试次数
            multi_trip_key = f"multi_trip_{current_task_id}"
            state.setdefault(multi_trip_key, 0)
            state[multi_trip_key] += 1

            # 如果已经尝试了太多轮（每轮尝试所有cargo type一次），放弃
            max_multi_trip_rounds = 20  # 最多20轮多趟运输
            if state[multi_trip_key] > max_multi_trip_rounds:
                if debug:
                    print(f"  [MAX-ROUNDS] Agent {agent_id} 任务 {current_task_id} "
                          f"已尝试 {max_multi_trip_rounds} 轮，放弃")
                state['route_index'] += 1
                state['current_task_cargo_types'] = []
                state['current_cargo_index'] = 0
                return _decide_next_action(env, agent_id, state, decide_quantity,
                                           safety_forbidden, sequential_prereqs, debug, masks_dict,
                                           _recursion_depth + 1)

            # 【防止无限循环】检查是否有depot有库存
            has_stock = False
            for ct in still_needed_cargo_types:
                for depot_id in range(env.depots_num):
                    depot = env.depot_dic[depot_id]
                    if depot['stock'].get(ct, 0) > 0:
                        has_stock = True
                        break
                if has_stock:
                    break

            if not has_stock:
                # 没有depot有库存，无法继续，移动到下一个任务
                if debug:
                    print(f"  [NO-STOCK] Agent {agent_id} 无法继续任务 {current_task_id}，"
                          f"所有depot都没有货物类型 {still_needed_cargo_types} 的库存")
                state['route_index'] += 1
                state['current_task_cargo_types'] = []
                state['current_cargo_index'] = 0
                # 递归处理下一个任务（有界递归：route_index递增）
                return _decide_next_action(env, agent_id, state, decide_quantity,
                                           safety_forbidden, sequential_prereqs, debug, masks_dict,
                                           _recursion_depth + 1)
            else:
                # 重新初始化货物类型列表继续运输
                if debug:
                    print(f"  [MULTI-TRIP] Agent {agent_id} 重新开始任务 {current_task_id}，"
                          f"仍需货物类型: {still_needed_cargo_types}，第{state[multi_trip_key]}轮")

                # 【已修复】still_needed_cargo_types已经只包含被分配的货物类型
                # 直接使用即可（已按cargo_type_priority过滤）
                state['current_task_cargo_types'] = still_needed_cargo_types
                state['current_cargo_index'] = 0
                # 不递归，继续执行下面的代码处理第一个cargo type
        else:
            # 任务完成（至少对于这个agent能做的部分），移动到下一个任务
            state['route_index'] += 1
            state['current_task_cargo_types'] = []
            state['current_cargo_index'] = 0
            # 递归处理下一个任务（有界递归：route_index递增）
            return _decide_next_action(env, agent_id, state, decide_quantity,
                                       safety_forbidden, sequential_prereqs, debug, masks_dict)

    # 获取当前需要运输的货物类型
    target_cargo_type = state['current_task_cargo_types'][state['current_cargo_index']]

    # 检查agent是否有能力运输这种货物
    if agent['capacity'][target_cargo_type] == 0:
        # 跳过这种货物类型
        state['current_cargo_index'] += 1
        return _decide_next_action(env, agent_id, state, decide_quantity,
                                   safety_forbidden, sequential_prereqs, debug, masks_dict)

    # 状态机逻辑
    if inventory['quantity'] == 0:
        # Agent为空，需要去depot装货
        # 【关键修复】先检查任务是否还需要这种货物
        if task['status'][target_cargo_type] == 0:
            # 任务不再需要这种货物（被其他agent完成了），跳过
            state['current_cargo_index'] += 1
            return _decide_next_action(env, agent_id, state, decide_quantity,
                                       safety_forbidden, sequential_prereqs, debug, masks_dict)

        if current_location < 0:
            # 已经在depot，执行LOAD
            # 【关键修复】使用掩码验证LOAD动作是否合法
            action_type_mask = masks_dict.get('action_type', np.zeros(3, dtype=bool))
            cargo_to_load_mask = masks_dict.get('cargo_to_load', np.zeros(env.traits_dim, dtype=bool))

            # 检查LOAD动作类型是否被掩码
            if action_type_mask[env.ACTION_LOAD]:
                # LOAD动作被掩码，跳过这种货物类型
                if debug:
                    print(f"  [MASK] Agent {agent_id}: LOAD动作被掩码，跳过货物类型 {target_cargo_type}")
                state['current_cargo_index'] += 1
                return _decide_next_action(env, agent_id, state, decide_quantity,
                                           safety_forbidden, sequential_prereqs, debug, masks_dict,
                                           _recursion_depth + 1)

            # 检查目标货物类型是否被掩码
            if cargo_to_load_mask[target_cargo_type]:
                # 该货物类型被掩码（可能depot没有库存），跳过
                if debug:
                    print(f"  [MASK] Agent {agent_id}: 货物类型 {target_cargo_type} 被掩码，跳过")
                state['current_cargo_index'] += 1
                return _decide_next_action(env, agent_id, state, decide_quantity,
                                           safety_forbidden, sequential_prereqs, debug, masks_dict,
                                           _recursion_depth + 1)

            quantity_vec = np.zeros(env.traits_dim)

            # 【新增】根据decide_quantity决定装货数量
            if decide_quantity:
                # 使用quantity_ratios中的直接数量值
                quantity_ratios = state.get('quantity_ratios', {})
                chosen_quantity = quantity_ratios.get(current_task_id, 5)  # 默认装满(5)
                max_capacity = agent['capacity'][target_cargo_type]
                # 实际装货量 = min(选择的数量, agent容量)，确保至少装1个
                load_quantity = max(1, min(chosen_quantity, max_capacity))
                quantity_vec[target_cargo_type] = load_quantity
            else:
                # 默认行为：装满
                quantity_vec[target_cargo_type] = agent['capacity'][target_cargo_type]

            return {'type': env.ACTION_LOAD, 'quantity_vec': quantity_vec}
        else:
            # 需要移动到depot
            # 【关键修复】使用掩码验证MOVE动作是否合法
            action_type_mask = masks_dict.get('action_type', np.zeros(3, dtype=bool))
            destination_mask = masks_dict.get('destination', np.zeros(env.depots_num + env.tasks_num, dtype=bool))

            # 检查MOVE动作类型是否被掩码
            if action_type_mask[env.ACTION_MOVE]:
                # MOVE动作被掩码，跳过这种货物类型
                if debug:
                    print(f"  [MASK] Agent {agent_id}: MOVE动作被掩码，跳过货物类型 {target_cargo_type}")
                state['current_cargo_index'] += 1
                return _decide_next_action(env, agent_id, state, decide_quantity,
                                           safety_forbidden, sequential_prereqs, debug, masks_dict,
                                           _recursion_depth + 1)

            # 找到最近的有库存且未被掩码的depot
            best_depot = None
            min_dist = float('inf')
            for depot_id in range(env.depots_num):
                # 【关键修复】检查depot是否被destination掩码
                if destination_mask[depot_id]:
                    continue  # 该depot被掩码，跳过

                depot = env.depot_dic[depot_id]
                if depot['stock'].get(target_cargo_type, 0) > 0:
                    # destination应该是depot_id（0, 1, 2...），不是depot_node_id
                    dist = np.linalg.norm(
                        agent['location'] - depot['location']
                    )
                    if dist < min_dist:
                        min_dist = dist
                        best_depot = depot_id  # 使用depot_id而不是depot_node_id

            if best_depot is None:
                # 没有depot有库存或都被掩码，跳过这种货物
                state['current_cargo_index'] += 1
                return _decide_next_action(env, agent_id, state, decide_quantity,
                                           safety_forbidden, sequential_prereqs, debug, masks_dict,
                                           _recursion_depth + 1)

            return {'type': env.ACTION_MOVE, 'destination': best_depot}
    else:
        # Agent有货物，需要去任务点卸货
        carried_cargo_type = inventory['type']

        # 【关键修复】检查任务是否还需要agent携带的货物
        if task['status'][carried_cargo_type] == 0:
            # 任务不再需要这种货物，需要把货物带回depot
            if current_location < 0:
                # 已经在depot，可以"卸载"到depot（实际上是丢弃或存回）
                # 环境不支持在depot卸载，所以我们清空inventory并继续
                # 【简化处理】直接跳过这个货物类型，让agent重新装载正确的货物
                state['current_cargo_index'] += 1
                # 注意：这里agent仍然携带着货物，但我们跳过了
                # 实际上应该返回depot卸载，但环境不支持
                # 所以我们需要先去depot
                return _decide_next_action(env, agent_id, state, decide_quantity,
                                           safety_forbidden, sequential_prereqs, debug, masks_dict,
                                           _recursion_depth + 1)
            else:
                # 需要先回depot
                # 【关键修复】使用掩码验证MOVE动作是否合法
                action_type_mask = masks_dict.get('action_type', np.zeros(3, dtype=bool))
                destination_mask = masks_dict.get('destination', np.zeros(env.depots_num + env.tasks_num, dtype=bool))

                if action_type_mask[env.ACTION_MOVE]:
                    # MOVE动作被掩码，跳过
                    state['current_cargo_index'] += 1
                    return _decide_next_action(env, agent_id, state, decide_quantity,
                                               safety_forbidden, sequential_prereqs, debug, masks_dict)

                # 找到最近的未被掩码的depot
                best_depot = None
                min_dist = float('inf')
                for depot_id in range(env.depots_num):
                    if destination_mask[depot_id]:
                        continue  # 该depot被掩码，跳过
                    depot = env.depot_dic[depot_id]
                    dist = np.linalg.norm(agent['location'] - depot['location'])
                    if dist < min_dist:
                        min_dist = dist
                        best_depot = depot_id

                if best_depot is None:
                    # 所有depot都被掩码，跳过
                    state['current_cargo_index'] += 1
                    return _decide_next_action(env, agent_id, state, decide_quantity,
                                               safety_forbidden, sequential_prereqs, debug, masks_dict)

                return {'type': env.ACTION_MOVE, 'destination': best_depot}

        # 注意：current_location是task_id（正数）或depot_node_id（负数）
        # current_task_node_id是节点ID（depots_num + task_id）
        # 需要统一比较格式
        if current_location >= 0:
            # current_location是task_id，需要转换为节点ID
            current_location_node_id = env.depots_num + current_location
        else:
            # current_location是depot_node_id（负数），保持不变
            current_location_node_id = current_location

        if current_location_node_id == current_task_node_id:
            # 已经在任务点，执行UNLOAD
            # 【关键修复】使用掩码验证UNLOAD动作是否合法
            action_type_mask = masks_dict.get('action_type', np.zeros(3, dtype=bool))

            if action_type_mask[env.ACTION_UNLOAD]:
                # UNLOAD动作被掩码，跳过
                if debug:
                    print(f"  [MASK] Agent {agent_id}: UNLOAD动作被掩码，跳过")
                state['current_cargo_index'] += 1
                return _decide_next_action(env, agent_id, state, decide_quantity,
                                           safety_forbidden, sequential_prereqs, debug, masks_dict,
                                           _recursion_depth + 1)

            quantity_vec = np.zeros(env.traits_dim)
            quantity_vec[inventory['type']] = inventory['quantity']

            # 卸货后，移动到下一个货物类型
            state['current_cargo_index'] += 1

            return {'type': env.ACTION_UNLOAD, 'quantity_vec': quantity_vec}
        else:
            # 需要移动到任务点
            # 【关键修复】使用掩码验证MOVE动作是否合法
            action_type_mask = masks_dict.get('action_type', np.zeros(3, dtype=bool))
            destination_mask = masks_dict.get('destination', np.zeros(env.depots_num + env.tasks_num, dtype=bool))

            # 【新增修复】检查是否所有任务目的地都被掩码（货物对所有任务都"无用"）
            # 这可能发生在其他agent的在途货物已经覆盖了任务需求时
            all_task_destinations_masked = np.all(destination_mask[env.depots_num:])

            if all_task_destinations_masked:
                # 货物对所有任务都无用（可能被其他agent的在途货物覆盖）
                # 需要返回depot卸货或在depot卸货
                if current_location < 0:
                    # 已经在depot，检查是否可以UNLOAD
                    if not action_type_mask[env.ACTION_UNLOAD]:
                        # 可以UNLOAD
                        if debug:
                            print(f"  [USELESS-CARGO] Agent {agent_id}: 货物无用，在depot卸货")
                        quantity_vec = np.zeros(env.traits_dim)
                        quantity_vec[inventory['type']] = inventory['quantity']
                        state['current_cargo_index'] += 1
                        return {'type': env.ACTION_UNLOAD, 'quantity_vec': quantity_vec}
                    else:
                        # UNLOAD被掩码，agent完全卡住
                        if debug:
                            print(f"  [FULLY-STUCK] Agent {agent_id}: 在depot但无法卸货")
                        return None
                else:
                    # 不在depot，需要移动到depot
                    # 首先检查MOVE是否被掩码
                    if action_type_mask[env.ACTION_MOVE]:
                        if debug:
                            print(f"  [FULLY-STUCK] Agent {agent_id}: 货物无用但MOVE被掩码，无法返回depot")
                        return None

                    for depot_id in range(env.depots_num):
                        if not destination_mask[depot_id]:
                            if debug:
                                print(f"  [USELESS-CARGO] Agent {agent_id}: 货物无用，返回depot {depot_id}")
                            return {'type': env.ACTION_MOVE, 'destination': depot_id}

                    # 所有depot都被掩码，agent卡住
                    if debug:
                        print(f"  [FULLY-STUCK] Agent {agent_id}: 货物无用但所有depot被掩码")
                    return None

            if action_type_mask[env.ACTION_MOVE]:
                # MOVE动作被掩码，跳过
                if debug:
                    print(f"  [MASK] Agent {agent_id}: MOVE动作被掩码，跳过")
                state['current_cargo_index'] += 1
                return _decide_next_action(env, agent_id, state, decide_quantity,
                                           safety_forbidden, sequential_prereqs, debug, masks_dict,
                                           _recursion_depth + 1)

            # 检查目标任务节点是否被掩码
            if destination_mask[current_task_node_id]:
                # 目标被掩码，但agent还携带着货物！
                # 【关键修复】如果有任何未被掩码的目的地，就去那里

                # 首先找任何未被掩码的任务目的地
                carried_cargo = inventory['type']

                for alt_task_node in range(env.depots_num, env.depots_num + env.tasks_num):
                    if not destination_mask[alt_task_node]:
                        # 找到一个未被掩码的任务目的地
                        # 【修复】不再检查alt_task['status']，因为mask已经反映了哪些任务可以接收货物
                        alt_task_id = alt_task_node - env.depots_num
                        if debug:
                            print(f"  [REDIRECT] Agent {agent_id}: 目标任务 {current_task_id} 被掩码，"
                                  f"改为送往任务 {alt_task_id}")
                        return {'type': env.ACTION_MOVE, 'destination': alt_task_node}

                # 没有任务可达，尝试返回depot
                for depot_id in range(env.depots_num):
                    if not destination_mask[depot_id]:
                        if debug:
                            print(f"  [RETURN-DEPOT] Agent {agent_id}: 目标被掩码且无任务可达，"
                                  f"返回depot {depot_id}")
                        return {'type': env.ACTION_MOVE, 'destination': depot_id}

                # 所有目的地都被掩码，检查是否可以在当前位置UNLOAD
                if current_location < 0 and not action_type_mask[env.ACTION_UNLOAD]:
                    # 在depot且可以UNLOAD
                    if debug:
                        print(f"  [UNLOAD-STUCK] Agent {agent_id}: 所有目的地被掩码，在depot卸货")
                    quantity_vec = np.zeros(env.traits_dim)
                    quantity_vec[inventory['type']] = inventory['quantity']
                    state['current_cargo_index'] += 1
                    return {'type': env.ACTION_UNLOAD, 'quantity_vec': quantity_vec}

                # 完全卡住 - 返回None结束此agent
                if debug:
                    print(f"  [FULLY-STUCK] Agent {agent_id}: 所有目的地被掩码，无法行动")
                return None

            # 【修复】使用正确的节点ID
            return {'type': env.ACTION_MOVE, 'destination': current_task_node_id}

# ============================
# Baseline方法接口
# ============================


def calculate_solution_metrics(
    env: TaskEnv,
    solution: Optional[Dict],
    ltl_monitor: Optional[LTLMonitor],
    method_name: str,
    elapsed_time: float,
    decide_quantity: bool = False,
    use_in_flight_reservation: bool = True
) -> Dict:
    """
    从解中提取并计算所有评估指标

    Args:
        env: 环境实例
        solution: 求解器返回的解
        ltl_monitor: LTL监视器（可选）
        method_name: 方法名称
        elapsed_time: 求解时间
        decide_quantity: 是否使用solution中的quantity_ratios决定装货数量
        use_in_flight_reservation: 是否启用在途货物预留（GA/AVNR应设为False）

    Returns:
        metrics: 包含所有评估指标的字典
    """
    metrics = {
        'method': method_name,
        'success': False,
        'solving_time': elapsed_time,

        # 核心性能指标
        'makespan': float('inf'),
        'travel_distance': float('inf'),
        'time_cost': float('inf'),
        'task_completion_rate': 0.0,

        # 资源利用指标
        'num_vehicles_used': 0,
        'vehicle_utilization': 0.0,
        'load_balance_std': float('inf'),

        # LTL约束指标
        'ltl_satisfaction_rate': 1.0,  # 默认100%（无约束时）
        'ltl_violations_count': 0,
        'ltl_total_constraints': 0,
    }

    # 检查解是否存在
    if solution is None:
        return metrics

    # 初步检查：解是否存在（fitness不是inf）
    solution_exists = solution.get('approx_fitness', float('inf')) < float('inf')

    if not solution_exists:
        return metrics

    # ========== 1. 从GA solution中提取基础信息 ==========
    routes = solution.get('routes', [])
    route_array = solution.get('route_array', np.array([]))

    # 计算使用的车辆数（非空路由数量）
    num_vehicles_used = len([r for r in routes if len(r) > 0])
    metrics['num_vehicles_used'] = num_vehicles_used

    # 车辆利用率
    total_vehicles = env.agents_num
    metrics['vehicle_utilization'] = num_vehicles_used / total_vehicles if total_vehicles > 0 else 0.0

    # ========== 2. 通过模拟执行计算makespan和距离 ==========
    task_completion = 0.0
    try:
        makespan, travel_dist, time_cost, task_completion = simulate_solution_execution(
            env, solution, ltl_monitor, decide_quantity=decide_quantity,
            use_in_flight_reservation=use_in_flight_reservation
        )
        metrics['task_completion_rate'] = task_completion

        # 【严格成功标准】只有任务完成率100%才算成功
        if task_completion >= 0.9999:  # 使用0.9999避免浮点数精度问题
            metrics['success'] = True
            metrics['makespan'] = makespan
            metrics['travel_distance'] = travel_dist
            metrics['time_cost'] = time_cost
        else:
            # 任务未100%完成，视为失败，makespan设为inf
            metrics['success'] = False
            metrics['makespan'] = float('inf')
            metrics['travel_distance'] = float('inf')
            metrics['time_cost'] = float('inf')

    except Exception as e:
        print(f"  ⚠️  模拟执行失败: {str(e)}")
        # 模拟执行失败，视为失败
        metrics['success'] = False
        metrics['makespan'] = float('inf')
        metrics['travel_distance'] = float('inf')
        metrics['time_cost'] = float('inf')

    # ========== 3. 计算负载均衡 ==========
    # 简化计算：基于路由长度的标准差
    if routes:
        route_lengths = [len(r) for r in routes if len(r) > 0]
        if len(route_lengths) > 1:
            metrics['load_balance_std'] = float(np.std(route_lengths))
        else:
            metrics['load_balance_std'] = 0.0

    # ========== 4. 计算LTL约束满足情况 ==========
    if ltl_monitor and ltl_monitor.clauses:
        stats = ltl_monitor.get_statistics()
        metrics['ltl_total_constraints'] = stats['num_clauses']
        metrics['ltl_satisfaction_rate'] = stats['overall_satisfaction_rate'] * 100  # 转为百分比

        # 违反数量（安全约束被违反 + 顺序约束未满足）
        violations = stats['safety_violated']
        # 顺序约束：INITIAL和VIOLATED都算未满足
        violations += stats['sequential_initial']
        violations += stats.get('sequential_violated', 0)

        metrics['ltl_violations_count'] = violations

        # 【调试】打印详细的LTL统计信息
        print(f"  [LTL DEBUG] 约束统计:")
        print(f"    - SAFETY: {stats['num_safety']} 个 (safe={stats['safety_safe']}, violated={stats['safety_violated']})")
        print(f"    - SEQUENTIAL: {stats['num_sequential']} 个 (initial={stats['sequential_initial']}, predecessor_done={stats['sequential_predecessor_done']}, satisfied={stats['sequential_satisfied']})")
        print(f"    - 总违反数: {violations} (safety_violated={stats['safety_violated']}, seq_initial={stats['sequential_initial']}, seq_violated={stats.get('sequential_violated', 0)})")

    return metrics
