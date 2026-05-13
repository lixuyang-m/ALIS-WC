import math


class CurriculumParams:
    """Curriculum learning hyperparameters.

    Disabled by default: the released best checkpoint was trained without a
    curriculum, drawing instances uniformly from the static ranges declared
    in EnvParams (SPECIES_AGENTS_RANGE, SPECIES_RANGE, DEPOT_NUM_RANGE,
    TASKS_RANGE). Set ENABLED=True and provide a multi-stage SCHEDULE if
    you want to revive the curriculum mechanism.
    """

    ENABLED = False

    # 每次增加难度的 training_step 间隔
    DIFFICULTY_INCREASE_STEP = 300000

    # 课程学习阶段性参数表
    # 格式: (SPECIES_RANGE, SPECIES_AGENTS_RANGE, DEPOT_NUM_RANGE, TASKS_RANGE)
    SCHEDULE = [
        ((3, 6), (3, 6), (3, 6), (15, 100)),
    ]


class EnvParams:
    """环境参数"""

    # 基础环境配置
    SPECIES_AGENTS_RANGE = (3, 6)
    SPECIES_RANGE = (3, 6)
    TASKS_RANGE = (15, 100)
    MAX_TIME = 500
    TRAIT_DIM = 5
    DECISION_DIM = 30
    LOAD_UNLOAD_DURATION = 0.2

    # 任务和智能体参数
    MAX_TASK_SIZE = 2  # 任务对单一货物类型的最大需求量
    DURATION_SCALE = 5  # 任务持续时间缩放因子
    DEPOT_NUM_RANGE = (3, 6)  # 仓库数量范围
    MAX_AGENT_CAPACITY = 5  # 智能体对单一货物类型的最大携带量

    # 计算最大目的地数量
    MAX_DESTINATIONS = TASKS_RANGE[1] + DEPOT_NUM_RANGE[1]

    # Pure LTL版本：故障建模已禁用（保留此参数以兼容代码检查）
    VEHICLE_FAILURE_ENABLED = False

    # 货物携带策略
    # False（默认，严格）：禁止带货跨仓；在仓库禁止卸"有用货物"
    # True（放宽）：允许带货跨仓；在仓库允许卸"有用货物"
    RELAXED_CARRYING_POLICY = False


class TrainParams:
    # ==================== 基础设置 ====================
    USE_GPU = True
    USE_GPU_GLOBAL = True
    NUM_GPU = 1
    NUM_META_AGENT = 8
    DECAY_STEP = 2e3
    RESET_OPT = False
    EVALUATE = True
    EVALUATION_SAMPLES = 1
    RESET_RAY = False
    INCREASE_DIFFICULTY = 20000
    SUMMARY_WINDOW = 1
    DEMON_RATE = 0.5
    IL_DECAY = -1e-5
    BATCH_SIZE = 2048
    # =================================================

    # ==================== 网络结构参数 ====================
    # 智能体特征维度（Pure LTL版本）：
    # - inventory_vec: 1 + TRAIT_DIM (库存量 + 货物类型one-hot)
    # - travel_time: 1
    # - current_waiting_time: 1
    # - relative_location: 2 (x, y)
    # - is_assigned: 1
    # - capacity_normalized: TRAIT_DIM (能力向量，归一化到[0,1])
    # - time_features: 2 (normalized_current_time, normalized_remaining_time)
    # Pure LTL总计: (1 + TRAIT_DIM) + 1 + 1 + 2 + 1 + TRAIT_DIM + 2 = 8 + 2*TRAIT_DIM

    AGENT_INPUT_DIM = 8 + 2 * EnvParams.TRAIT_DIM  # Pure LTL: 18维（TRAIT_DIM=5）

    # 任务特征维度（迁移学习优化版）:
    # LTL特征通过独立的编码模块处理，不嵌入到任务基础特征中
    # - Depot: depot_feature_vec(TRAIT_DIM*2+1=11) + travel_time(1) + relative_location(2) + is_available(1) = 15维
    # - Task: status(TRAIT_DIM) + requirements(TRAIT_DIM) + time(1) + travel_time(1) + relative_location(2) + is_available(1) = 15维
    # 总计: (2*TRAIT_DIM + 1) + 1 + 2 + 1 = 15维（TRAIT_DIM=5）
    TASK_INPUT_DIM = (
        2 * EnvParams.TRAIT_DIM + 1 + 1 + 2 + 1
    )  # 15维（移除了+5的LTL特征）

    EMBEDDING_DIM = 128
    SAMPLE_SIZE = 200
    PADDING_SIZE = 50
    POMO_SIZE = 10
    FORCE_MAX_OPEN_TASK = False
    # ====================================================

    # ==================== 通用训练参数 ====================
    ENTROPY_BETA = 0.005
    ENTROPY_ANNEALING = False
    ENTROPY_BETA_START = ENTROPY_BETA
    ENTROPY_BETA_END = 0.0
    ENTROPY_ANNEALING_TOTAL_STEPS = None
    GRAD_L2_CLIP = 40  # 梯度裁剪阈值
    ALGORITHM = "PPO"  # 算法选择：'A2C', 'PPO', 'IMPALA'
    # ====================================================

    # ==================== PPO算法超参数 ====================
    PPO_ROLLOUT_LENGTH = 2048  # 每个worker每轮收集的步数
    PPO_EPOCHS = 4  # PPO更新轮数
    PPO_MINIBATCH_SIZE = 512  # Mini-batch大小
    PPO_CLIP_EPSILON = 0.2  # PPO裁剪参数
    GAE_LAMBDA = 0.95  # GAE lambda参数

    # ==================== 方法1：分头独立loss ====================
    # 是否启用分头独立PPO loss（Multi-head PPO）
    USE_MULTI_HEAD_PPO = True

    # 分头loss权重（根据动作空间大小归一化）
    # 自动计算：w_i = log(|A_i|) / Σ log(|A_j|)
    # 不需要手动配置，driver.py会根据实际动作空间大小自动计算
    # ============================================================

    # ==================== Value Loss归一化 ====================
    # 是否启用Value Loss的RunningStats归一化
    # True:  value_loss = mse / (running_std² + ε)，自适应reward scale
    # False: value_loss = mse，标准做法（与SMDP4_another2一致）
    USE_VALUE_LOSS_NORMALIZATION = True
    # ======================================================

    # ==================== IMPALA算法超参数 ====================
    IMPALA_RHO_CLIP = 1.0  # V-trace rho裁剪
    IMPALA_C_CLIP = 1.0  # V-trace c裁剪
    UNROLL_LEN = 20  # 序列长度
    UNROLLS_PER_UPDATE = 32  # 每次更新的序列数
    MAX_BUFFER_SIZE = 10000  # 经验缓冲区容量
    # ==========================================================

    # ==================== 评估设置 ====================
    EVALUATE = True
    EVALUATE_GAP = 20  # 每隔多少次更新进行一次评估
    EVAL_USE_SAMPLING = False  # 评估时是否使用采样
    # True: 使用随机采样
    # False: 使用贪婪argmax
    # ==================================================

    # ==================== 训练停止条件 ====================
    MAX_TRAINING_STEPS = 10000000
    # None: 无限制，手动停止
    # 整数: 达到指定步数后自动停止
    # ====================================================

    # ==================== 执行模式选择 ====================
    # 'mdp'  -> MDP模式（γ=1，显式时间奖励）
    # 'smdp' -> SMDP模式（时间依赖折扣，γ(τ)=exp(-β·τ)）
    EXECUTION_MODE = "mdp"

    if EXECUTION_MODE == "mdp":
        # ==================== MDP模式配置 ====================
        # 折扣因子: γ=1（无折扣）
        # 【稀疏奖励】终止时: weight × (-current_time)
        # 【稠密奖励】每步: weight × (-Δt)

        LR = 5e-4  # 学习率（将配合线性衰减）
        GAMMA = 1.0
        USE_ONLY_COMPLETE_EPISODES = False
        MIN_COMPLETE_RATIO = 0.3

        DELTA_T = 30.0
        BETA = 0.0

        # 奖励权重配置
        # Two PBRS variants are supported under gamma=1:
        #   (a) Elapsed-time PBRS  ->  per-step time penalty -lambda * dt,
        #       controlled by MDP_DENSE_REWARD_WEIGHT below.
        #   (b) Task-completion PBRS  ->  lambda * (psi(s') - psi(s))
        #       where psi = #finished_tasks / #total_tasks, controlled by
        #       REWARD_TASK_COMPLETION_POTENTIAL_WEIGHT a few lines below.
        # Both are valid PBRS under the Ng-Harada-Russell theorem; setting
        # either weight to zero disables that variant. The released best
        # checkpoint was trained with elapsed-time PBRS (MDP_DENSE_REWARD_WEIGHT=10
        # and REWARD_TASK_COMPLETION_POTENTIAL_WEIGHT=0).
        MDP_SPARSE_REWARD_WEIGHT = 1.0  # 稀疏终止奖励权重
        MDP_DENSE_REWARD_WEIGHT = 10.0  # Elapsed-time PBRS coefficient (lambda)

        # 其他奖励组件（MDP模式下默认禁用）
        REWARD_COMPLETED_DEMAND_WEIGHT = 0.0
        # Task-completion PBRS coefficient. Set non-zero to enable the
        # alternative task-fraction shaping (see README, "Reward Shaping").
        REWARD_TASK_COMPLETION_POTENTIAL_WEIGHT = 0.0
        REWARD_TIME_POTENTIAL_WEIGHT = 0.0

        # 向后兼容参数
        USE_MDP_TIME_REWARD = (
            MDP_SPARSE_REWARD_WEIGHT != 0 or MDP_DENSE_REWARD_WEIGHT != 0
        )
        REWARD_TIME_PENALTY_MODE = "sparse" if MDP_DENSE_REWARD_WEIGHT == 0 else "dense"
        TERMINAL_REWARD_SUCCESS = 100.0
        TERMINAL_PENALTY_FAILURE = -100.0
        REWARD_TIME_ELAPSED_WEIGHT = 0.0

    elif EXECUTION_MODE == "smdp":
        # ==================== SMDP模式配置 ====================
        # 折扣因子: γ(τ) = exp(-β·τ)（时间依赖折扣）
        # 【稀疏奖励】终止时: weight × (-current_time)
        # 【稠密奖励】势能塑形: weight × [exp(-β·τ)·φ(s') - φ(s)]
        #            其中 φ(s) = 已完成任务数 / 总任务数

        LR = 5e-5  # 学习率（将配合线性衰减）
        GAMMA = 0.99
        USE_ONLY_COMPLETE_EPISODES = False
        MIN_COMPLETE_RATIO = 0.3

        DELTA_T = 30.0

        # BETA自动计算：β = -ln(γ) / DELTA_T
        try:
            BETA = -math.log(GAMMA) / DELTA_T
        except (ValueError, ZeroDivisionError):
            BETA = 0.0

        # 奖励权重配置
        SMDP_SPARSE_REWARD_WEIGHT = 1.0  # 稀疏终止奖励权重
        SMDP_DENSE_REWARD_WEIGHT = 0.0

        # 向后兼容参数
        TERMINAL_REWARD_SUCCESS = 100.0
        TERMINAL_PENALTY_FAILURE = -100.0
        REWARD_COMPLETED_DEMAND_WEIGHT = 0.0
        REWARD_TASK_COMPLETION_POTENTIAL_WEIGHT = SMDP_DENSE_REWARD_WEIGHT
        REWARD_TIME_POTENTIAL_WEIGHT = 0.0
        USE_MDP_TIME_REWARD = False
        REWARD_TIME_PENALTY_MODE = "sparse"
        REWARD_TIME_ELAPSED_WEIGHT = 0.0

    else:
        raise ValueError(
            f"未知的 EXECUTION_MODE: {EXECUTION_MODE}，请选择 'mdp' 或 'smdp'"
        )

    # ==================== 损失计算方法 ====================
    LOSS_CALCULATION_METHOD = 1  # 0:标准模式, 1:梯度投影模式
    # ====================================================

    # ==================== 随机种子 ====================
    USE_MANUAL_SEED = True
    MANUAL_SEED = 42
    # ==================================================

    # ==================== LTL约束参数 ====================
    # Default False to match the released best checkpoint (trained without LTL).
    # Set to True at runtime to enable hard LTL shield + sleep-wake at deployment.
    # Note: LTL_ENABLED=True additionally requires torch-geometric (optional dep).
    LTL_ENABLED = False
    LTL_TASK_FEATURES_ENABLED = True
    LTL_MAX_CLAUSES = 10  # 最大LTL子句数
    LTL_GEN_MAX_RETRIES = 200  # LTL生成最大重试次数

    # LTL约束数量生成模式
    # 'random' - 随机数量模式：生成0到LTL_MAX_CLAUSES之间的随机数量
    # 'fixed'  - 固定数量模式：生成固定数量的约束
    LTL_NUM_CLAUSES_MODE = "random"

    # Fixed模式下的约束数量配置
    LTL_FIXED_NUM_SAFETY = 5  # 安全性约束数量 (G !p)
    LTL_FIXED_NUM_SEQUENTIAL = 5  # 顺序性约束数量 (A -> B)
    LTL_FIXED_SPLIT_RANDOM = False

    # 向后兼容参数
    LTL_FIXED_NUM_CLAUSES = LTL_MAX_CLAUSES

    # LTL编码方式
    # 'A' - ID-specific编码：[max_clauses, 4]
    # 'B' - Task feasibility编码：[max_agents, max_tasks]
    # 'C' - Task feasibility + Dependency graph：
    #       feasibility矩阵 [max_agents, max_tasks] +
    #       依赖图 (edge_index [2, E], edge_attr [E, 1])
    #       其中edge_attr为连续阻塞度 ∈ [0,1]
    LTL_ENCODING_TYPE = "C"

    # LTL约束处理类型
    # 'HARD' - 硬约束（动作屏蔽）
    # 'SOFT' - 软约束（惩罚处理）
    # 'LTL_POTENTIAL' - 使用LTL势能塑形
    LTL_CONSTRAINT_TYPE = "LTL_POTENTIAL"

    # LTL势能类型（奖励塑形开关）
    # 'AUTOMATON' - 使用现有 compute_ltl_potential()
    # 'TOPOLOGY'  - 使用拓扑势能 compute_topology_potential()
    LTL_POTENTIAL_TYPE = "TOPOLOGY"
    LTL_POTENTIAL_WEIGHT = 10  # 势能权重（两者复用同一权重）

    # 评估时LTL控制
    # True:  评估时使用benchmark中的LTL约束
    # False: 评估时忽略LTL约束
    LTL_ENABLED_IN_EVALUATION = True

    # ==================== LTL势能塑形评估参数 ====================
    # 评估时是否强制使用硬约束（用于验证LTL满足能力）
    # True:  评估时将LTL_POTENTIAL模式切换为HARD模式
    # False: 评估时保持训练时的约束模式
    LTL_ENFORCE_IN_EVAL = False

    # LTL违规惩罚系数（用于LTL_POTENTIAL模式的终止奖励）
    # violation_penalty = num_violations × LTL_VIOLATION_PENALTY
    # 建议值：50-100（相对于TERMINAL_REWARD_SUCCESS=100）
    LTL_VIOLATION_PENALTY = 50.0
    # ==============================================================

    # Output dimension of the (retained-for-compatibility) cost_critic head
    # in attention.py. The released checkpoint contains weights for this
    # head; the head's output is not consumed by any loss in the current
    # codebase but is kept so that legacy checkpoints load without
    # "unexpected key" warnings.
    NUM_QUANTILES = 32


class SaverParams:
    """模型保存参数"""

    FOLDER_NAME = "save_1"
    MODEL_PATH = f"model/{FOLDER_NAME}"
    TRAIN_PATH = f"train/{FOLDER_NAME}"
    GIFS_PATH = f"gifs/{FOLDER_NAME}"
    LOAD_MODEL = False
    LOAD_FROM = "current"
    SAVE = True
    SAVE_IMG = True
    SAVE_IMG_GAP = 1000
    SAVE_GAP = 100

    # PPO专用保存间隔
    PPO_SAVE_INTERVAL = 20


class LoggingParams:
    """日志系统配置"""

    # 日志后端选择：'wandb' 或 'swanlab'
    BACKEND = "swanlab"

    # 日志配置
    PROJECT_NAME = "hetero_mrta"
    RUN_NAME = None  # None则自动生成
    # Default 'offline' so the script runs without requiring SwanLab/W&B credentials.
    # Switch to 'online' (and run `swanlab login` / `wandb login` first) to enable
    # cloud logging and live dashboards.
    MODE = "offline"  # 'online' 或 'offline'

    @staticmethod
    def generate_run_name():
        """
        自动生成实验运行名称
        格式: {mode}_d{dense}_{ltl_constraint}_{encoding}_{potential}_{pretrain}_{ppo}_{vloss}_ent{entropy}_s{seed}

        示例:
        - smdp_d10.0_hard_encA_pt_mhPPO_vNorm_ent0.01_s42          (SMDP, dense=10.0, HARD约束, 编码A, 预训练, 多头PPO, Value归一化, entropy=0.01, seed=42)
        - smdp_d10.0_ltl_potential_encB_auto_nopt_mhPPO_vNorm_ent0.02_s42  (SMDP, LTL势能, 编码B, AUTOMATON, 无预训练, 多头PPO, Value归一化, entropy=0.02, seed=42)
        - mdp_d0.0_noLTL_pt_mhPPO_vNorm_ent0.01_s43                (MDP, dense=0.0, 无LTL, 预训练, 多头PPO, Value归一化, entropy=0.01, seed=43)
        """
        # 执行模式
        mode = TrainParams.EXECUTION_MODE.lower()

        # 稠密奖励权重
        if mode == "mdp":
            dense_weight = getattr(TrainParams, "MDP_DENSE_REWARD_WEIGHT", 0.0)
        else:  # smdp
            dense_weight = getattr(TrainParams, "SMDP_DENSE_REWARD_WEIGHT", 0.0)

        def format_weight(w):
            if w == int(w):
                return f"{int(w)}"
            elif w < 0.1:
                return f"{w:.3f}"
            else:
                return f"{w:.1f}"

        dense_str = format_weight(dense_weight)

        # LTL约束类型
        if TrainParams.LTL_ENABLED:
            ltl_type = TrainParams.LTL_CONSTRAINT_TYPE.lower()

            # LTL编码类型
            encoding_type = f"enc{TrainParams.LTL_ENCODING_TYPE}"

            # LTL势能类型（仅在LTL_POTENTIAL模式下添加）
            if TrainParams.LTL_CONSTRAINT_TYPE == "LTL_POTENTIAL":
                potential_type = (
                    "auto" if TrainParams.LTL_POTENTIAL_TYPE == "AUTOMATON" else "topo"
                )
                ltl_suffix = f"{ltl_type}_{encoding_type}_{potential_type}"
            else:
                ltl_suffix = f"{ltl_type}_{encoding_type}"
        else:
            ltl_suffix = "noLTL"

        # 随机种子
        seed = TrainParams.MANUAL_SEED if TrainParams.USE_MANUAL_SEED else "rand"

        # PPO类型
        ppo_str = (
            "mhPPO" if getattr(TrainParams, "USE_MULTI_HEAD_PPO", False) else "stdPPO"
        )

        # Value Loss归一化
        vloss_str = (
            "vNorm"
            if getattr(TrainParams, "USE_VALUE_LOSS_NORMALIZATION", False)
            else "vStd"
        )

        # 熵系数
        entropy_beta = getattr(TrainParams, "ENTROPY_BETA", 0.01)
        entropy_str = format_weight(entropy_beta)

        # 组装名称
        run_name = f"{mode}_d{dense_str}_{ltl_suffix}_{ppo_str}_{vloss_str}_ent{entropy_str}_s{seed}"

        return run_name
