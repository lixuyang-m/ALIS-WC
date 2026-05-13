# generate_benchmarks_strict.py
"""
严格版评估基准生成脚本
特点:
1. 完整的环境参数
2. 严格的多层验证
3. 详细的统计和诊断信息
4. 自动过滤无效benchmark
"""

import json
import numpy as np
from env.task_env import TaskEnv
from env.ltl_utils import LTLMonitor, LTL_SAFETY, LTL_SEQUENTIAL
from parameters import EnvParams, TrainParams

# ==================== 配置 ====================
NUM_BENCHMARKS = 100          # 目标生成数量
OUTPUT_FILENAME = 'evaluation_benchmarks.json'
MAX_GENERATION_ATTEMPTS = 200  # 最大尝试次数（考虑到会过滤无效的）

# 验证严格程度
STRICT_VALIDATION = True       # 是否启用严格验证
MIN_TASKS_FOR_SEQUENTIAL = 2   # 生成顺序约束所需的最小任务数
# =================================================


class BenchmarkValidator:
    """Benchmark验证器 - 执行多层验证"""
    
    def __init__(self, env):
        self.env = env
        self.errors = []
        self.warnings = []
    
    def validate_ltl_clauses(self, ltl_clauses):
        """验证LTL约束的有效性"""
        self.errors = []
        self.warnings = []
        
        valid_task_ids = set(self.env.task_dic.keys())
        num_agents = self.env.agents_num
        num_tasks = self.env.tasks_num
        num_depots = self.env.depots_num
        num_nodes = num_depots + num_tasks
        
        for idx, (clause_type, p1, p2) in enumerate(ltl_clauses):
            clause_str = f"Clause {idx}: ({clause_type}, {p1}, {p2})"
            
            # 验证安全约束
            if clause_type == LTL_SAFETY:
                # Agent ID必须有效
                if p1 < 0 or p1 >= num_agents:
                    self.errors.append(f"{clause_str} - Invalid Agent ID {p1} (valid: 0-{num_agents-1})")
                
                # Node ID必须有效
                if p2 < 0 or p2 >= num_nodes:
                    self.errors.append(f"{clause_str} - Invalid Node ID {p2} (valid: 0-{num_nodes-1})")
                
                # 检查是否禁止去仓库（可能导致无解）
                if p2 < num_depots:
                    self.warnings.append(f"{clause_str} - Forbids depot {p2} (may cause unsolvable instances)")
            
            # 验证顺序约束
            elif clause_type == LTL_SEQUENTIAL:
                # 两个任务ID都必须有效
                if p1 not in valid_task_ids:
                    self.errors.append(f"{clause_str} - Task {p1} not in valid range {sorted(valid_task_ids)}")
                
                if p2 not in valid_task_ids:
                    self.errors.append(f"{clause_str} - Task {p2} not in valid range {sorted(valid_task_ids)}")
                
                # 不能是同一个任务
                if p1 == p2:
                    self.errors.append(f"{clause_str} - Sequential constraint with same task")
            
            else:
                self.errors.append(f"{clause_str} - Unknown clause type {clause_type}")
        
        return len(self.errors) == 0
    
    def validate_environment(self):
        """验证环境的基本合理性"""
        # 检查环境是否可解
        if self.env.agents_num == 0:
            self.errors.append("Environment has 0 agents")
            return False
        
        if self.env.tasks_num == 0:
            self.errors.append("Environment has 0 tasks")
            return False
        
        if self.env.depots_num == 0:
            self.errors.append("Environment has 0 depots")
            return False
        
        return True
    
    def get_report(self):
        """获取验证报告"""
        return {
            'is_valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings
        }


def generate_single_benchmark(seed, attempt_num):
    """
    生成单个benchmark并进行验证
    
    Returns:
        (benchmark_dict, is_valid, report)
    """
    # 创建环境
    temp_env = TaskEnv(
        per_species_range=EnvParams.SPECIES_AGENTS_RANGE,
        species_range=EnvParams.SPECIES_RANGE,
        tasks_range=EnvParams.TASKS_RANGE,
        depot_num_range=EnvParams.DEPOT_NUM_RANGE,
        traits_dim=EnvParams.TRAIT_DIM,
        decision_dim=EnvParams.DECISION_DIM,
        max_task_size=EnvParams.MAX_TASK_SIZE,
        duration_scale=EnvParams.DURATION_SCALE,
        max_cargo_per_type=EnvParams.MAX_AGENT_CAPACITY,
        seed=seed,
        plot_figure=False
    )
    
    # 验证环境基本合理性
    validator = BenchmarkValidator(temp_env)
    if not validator.validate_environment():
        return None, False, validator.get_report()
    
    # 生成LTL约束
    ltl_monitor = LTLMonitor(temp_env)
    
    # 转换为可序列化格式
    ltl_clauses_serializable = [
        (clause.type, clause.param1, clause.param2)
        for clause in ltl_monitor.clauses
    ]
    
    # 验证LTL约束
    is_valid = validator.validate_ltl_clauses(ltl_clauses_serializable)
    
    # 构建benchmark字典
    env_info = {
        "num_agents": temp_env.agents_num,
        "num_species": temp_env.species_num,
        "num_tasks": temp_env.tasks_num,
        "num_depots": temp_env.depots_num,
        "species_capacities": temp_env.species_dict['capacities'].tolist(),
    }
    
    benchmark = {
        "seed": seed,
        "ltl_clauses": ltl_clauses_serializable,
        "env_info": env_info,
        "validation": validator.get_report()
    }
    
    return benchmark, is_valid, validator.get_report()


def generate():
    """主生成函数"""
    print("="*80)
    print("严格模式 - 生成评估Benchmark")
    print("="*80)
    
    # 记录生成配置
    generation_config = {
        "version": "v3.0_strict",
        "target_num_benchmarks": NUM_BENCHMARKS,
        "strict_validation": STRICT_VALIDATION,
        "env_params": {
            "species_agents_range": list(EnvParams.SPECIES_AGENTS_RANGE),
            "species_range": list(EnvParams.SPECIES_RANGE),
            "tasks_range": list(EnvParams.TASKS_RANGE),
            "depot_num_range": list(EnvParams.DEPOT_NUM_RANGE),
            "traits_dim": EnvParams.TRAIT_DIM,
            "decision_dim": EnvParams.DECISION_DIM,
            "max_task_size": EnvParams.MAX_TASK_SIZE,
            "duration_scale": EnvParams.DURATION_SCALE,
            "max_cargo_per_type": EnvParams.MAX_AGENT_CAPACITY
        },
        "ltl_params": {
            "enabled": TrainParams.LTL_ENABLED,
            "max_clauses": TrainParams.LTL_MAX_CLAUSES,
            "constraint_mode": TrainParams.LTL_CONSTRAINT_TYPE,
            "enabled_in_evaluation": TrainParams.LTL_ENABLED_IN_EVALUATION
        }
    }
    
    print(f"\n配置信息:")
    print(f"  目标数量: {NUM_BENCHMARKS}")
    print(f"  最大尝试: {MAX_GENERATION_ATTEMPTS}")
    print(f"  严格验证: {STRICT_VALIDATION}")
    print(f"  环境参数: {generation_config['env_params']}")
    print(f"  LTL参数: {generation_config['ltl_params']}")
    print("")
    
    # 生成过程
    rng = np.random.default_rng()
    valid_benchmarks = []
    invalid_benchmarks = []
    
    # 统计信息
    stats = {
        'total_attempts': 0,
        'valid_count': 0,
        'invalid_count': 0,
        'total_clauses': 0,
        'total_safety': 0,
        'total_sequential': 0,
        'safety_to_depot': 0,
        'safety_to_task': 0,
        'errors_by_type': {},
        'warnings_by_type': {}
    }
    
    print("开始生成...\n")
    
    attempt = 0
    while len(valid_benchmarks) < NUM_BENCHMARKS and attempt < MAX_GENERATION_ATTEMPTS:
        attempt += 1
        seed = int(rng.integers(0, 1e8))
        
        benchmark, is_valid, report = generate_single_benchmark(seed, attempt)
        stats['total_attempts'] = attempt
        
        if benchmark is None:
            stats['invalid_count'] += 1
            print(f"  ✗ Attempt {attempt}: 环境生成失败 (seed={seed})")
            continue
        
        if STRICT_VALIDATION and not is_valid:
            stats['invalid_count'] += 1
            invalid_benchmarks.append(benchmark)
            print(f"  ✗ Attempt {attempt}: 验证失败 (seed={seed}, errors={len(report['errors'])})")
            for error in report['errors'][:2]:  # 只显示前2个错误
                print(f"      - {error}")
        else:
            stats['valid_count'] += 1
            valid_benchmarks.append(benchmark)
            
            # 更新统计
            env_info = benchmark['env_info']
            ltl_clauses = benchmark['ltl_clauses']
            num_clauses = len(ltl_clauses)
            stats['total_clauses'] += num_clauses
            
            for clause_type, p1, p2 in ltl_clauses:
                if clause_type == LTL_SAFETY:
                    stats['total_safety'] += 1
                    if p2 < env_info['num_depots']:
                        stats['safety_to_depot'] += 1
                    else:
                        stats['safety_to_task'] += 1
                elif clause_type == LTL_SEQUENTIAL:
                    stats['total_sequential'] += 1
            
            # 记录警告
            for warning in report['warnings']:
                warning_type = warning.split('-')[0].strip()
                stats['warnings_by_type'][warning_type] = stats['warnings_by_type'].get(warning_type, 0) + 1
            
            status = "✓"
            if report['warnings']:
                status = "⚠️"
            
            print(f"  {status} Benchmark {len(valid_benchmarks)}/{NUM_BENCHMARKS} "
                  f"(seed={seed}, agents={env_info['num_agents']}, "
                  f"tasks={env_info['num_tasks']}, ltl={num_clauses})")
    
    # 生成完成
    print("\n" + "="*80)
    print("生成统计")
    print("="*80)
    print(f"总尝试次数: {stats['total_attempts']}")
    print(f"有效benchmark: {stats['valid_count']}")
    print(f"无效benchmark: {stats['invalid_count']}")
    print(f"成功率: {stats['valid_count']/max(stats['total_attempts'],1)*100:.1f}%")
    
    if stats['total_clauses'] > 0:
        print(f"\nLTL约束统计:")
        print(f"  总约束数: {stats['total_clauses']}")
        print(f"  平均约束数: {stats['total_clauses']/len(valid_benchmarks):.2f}")
        print(f"  安全约束: {stats['total_safety']} ({stats['total_safety']/stats['total_clauses']*100:.1f}%)")
        print(f"    - 针对Depot: {stats['safety_to_depot']} ({stats['safety_to_depot']/max(stats['total_safety'],1)*100:.1f}%)")
        print(f"    - 针对Task: {stats['safety_to_task']} ({stats['safety_to_task']/max(stats['total_safety'],1)*100:.1f}%)")
        print(f"  顺序约束: {stats['total_sequential']} ({stats['total_sequential']/stats['total_clauses']*100:.1f}%)")
    
    if stats['warnings_by_type']:
        print(f"\n警告统计:")
        for warning_type, count in stats['warnings_by_type'].items():
            print(f"  - {warning_type}: {count}")
    
    # 保存结果
    output_data = {
        "config": generation_config,
        "statistics": stats,
        "benchmarks": valid_benchmarks
    }
    
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✓ 成功生成 {len(valid_benchmarks)} 个有效benchmark")
    print(f"✓ 已保存到: {OUTPUT_FILENAME}")
    print(f"{'='*80}")
    
    # 使用建议
    print(f"\n💡 使用说明:")
    print(f"1. 文件包含完整的配置和统计信息")
    print(f"2. 每个benchmark都经过严格验证")
    print(f"3. 在parameters.py中设置 LTL_ENABLED_IN_EVALUATION 控制评估时是否使用LTL:")
    print(f"   - True:  评估时加载LTL约束（测试约束满足能力）")
    print(f"   - False: 评估时忽略LTL约束（只测试任务完成）")
    
    if stats['invalid_count'] > 0:
        print(f"\n⚠️  注意: 有 {stats['invalid_count']} 个benchmark因验证失败被过滤")
        print(f"   如果需要，可以设置 STRICT_VALIDATION = False 放宽验证")
    
    return len(valid_benchmarks) == NUM_BENCHMARKS


if __name__ == '__main__':
    # 确保LTL功能在生成时是启用的
    original_ltl_enabled = TrainParams.LTL_ENABLED
    TrainParams.LTL_ENABLED = True
    
    try:
        success = generate()
        if not success:
            print(f"\n⚠️  警告: 未能生成足够的有效benchmark")
            print(f"   尝试增加 MAX_GENERATION_ATTEMPTS 或放宽验证条件")
            exit(1)
    finally:
        # 恢复原始设置
        TrainParams.LTL_ENABLED = original_ltl_enabled

