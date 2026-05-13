#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验启动脚本

用法:
    python run_ablation.py --config v1   # 运行v1配置组的所有实验
    python run_ablation.py --config v2   # 运行v2配置组的所有实验
    python run_ablation.py -c v3         # 简写形式

特点:
1. 支持4组预定义配置（v1/v2/v3/v4），每组可包含多个实验
2. 配置内支持列表值的笛卡尔积（自动生成多个实验组合）
3. 自动生成带时间戳的唯一实验名称，避免覆盖
4. 批量运行、进度保存、统计总结
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# 添加项目路径到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ========================================================================
# 消融实验配置组
# ========================================================================
# 说明：
# - 4组配置（v1/v2/v3/v4），通过命令行参数选择
# - 每组配置支持列表值，会自动生成笛卡尔积
# - 例如: {'MANUAL_SEED': [42, 123], 'ENTROPY_BETA': [0.01, 0.02]}
#   会生成4个实验: (42, 0.01), (42, 0.02), (123, 0.01), (123, 0.02)
# ========================================================================

ABLATION_CONFIGS = {
    "v1": {
        "MANUAL_SEED": [42],
        "ENTROPY_BETA": [0.005],
        "EXECUTION_MODE": ["mdp"],
        "LTL_ENABLED": [True],  # 消融：是否开启LTL
        "LTL_ENABLED_IN_EVALUATION": [True],
        "LTL_CONSTRAINT_TYPE": ["LTL_POTENTIAL"],
        "LTL_ENCODING_TYPE": ["C"],
        "USE_MULTI_HEAD_PPO": [True],  # 消融：是否使用多头PPO
        "USE_VALUE_LOSS_NORMALIZATION": [True],  # 消融：是否使用Value Loss归一化
    },
    # 不使用预训练模型
    "v2": {
        "MANUAL_SEED": [42],
        "ENTROPY_BETA": [0.005],
        "EXECUTION_MODE": ["mdp"],
        "LTL_ENABLED": [True],  # 消融：是否开启LTL
        "LTL_ENABLED_IN_EVALUATION": [True],
        "LTL_CONSTRAINT_TYPE": ["LTL_POTENTIAL"],
        "LTL_ENCODING_TYPE": ["C"],
        "USE_MULTI_HEAD_PPO": [True],  # 消融：是否使用多头PPO
        "USE_VALUE_LOSS_NORMALIZATION": [True],  # 消融：是否使用Value Loss归一化
    },
    # 不使用编码模式C
    "v3": {
        "MANUAL_SEED": [42],
        "ENTROPY_BETA": [0.005],
        "EXECUTION_MODE": ["mdp"],
        "LTL_ENABLED": [True],  # 消融：是否开启LTL
        "LTL_ENABLED_IN_EVALUATION": [True],
        "LTL_CONSTRAINT_TYPE": ["LTL_POTENTIAL"],
        "LTL_ENCODING_TYPE": ["A"],
        "USE_MULTI_HEAD_PPO": [True],  # 消融：是否使用多头PPO
        "USE_VALUE_LOSS_NORMALIZATION": [True],  # 消融：是否使用Value Loss归一化
    },
    # 不使用拓扑势能函数
    "v4": {
        "MANUAL_SEED": [42],
        "ENTROPY_BETA": [0.005],
        "EXECUTION_MODE": ["mdp"],
        "LTL_ENABLED": [True],  # 消融：是否开启LTL
        "LTL_ENABLED_IN_EVALUATION": [True],
        "LTL_CONSTRAINT_TYPE": ["HARD"],
        "LTL_ENCODING_TYPE": ["C"],
        "USE_MULTI_HEAD_PPO": [True],  # 消融：是否使用多头PPO
        "USE_VALUE_LOSS_NORMALIZATION": [True],  # 消融：是否使用Value Loss归一化
    },
    "v5": {
        "MANUAL_SEED": [42, 142, 242],
        "ENTROPY_BETA": [0.01],
        "EXECUTION_MODE": ["mdp"],
        "LTL_ENABLED": [True],  # 消融：是否开启LTL
        "LTL_ENABLED_IN_EVALUATION": [True],
        "LTL_CONSTRAINT_TYPE": ["HARD"],
        "LTL_ENCODING_TYPE": ["C"],
        "USE_MULTI_HEAD_PPO": [True],  # 消融：是否使用多头PPO
        "USE_VALUE_LOSS_NORMALIZATION": [True],  # 消融：是否使用Value Loss归一化
    },
    "v6": {
        "MANUAL_SEED": [42, 142, 242],
        "ENTROPY_BETA": [0.02],
        "EXECUTION_MODE": ["smdp"],
        "LTL_ENABLED": [True],  # 消融：是否开启LTL
        "LTL_ENABLED_IN_EVALUATION": [True],
        "LTL_CONSTRAINT_TYPE": ["LTL_POTENTIAL"],
        "LTL_ENCODING_TYPE": ["A"],
        "USE_MULTI_HEAD_PPO": [True],  # 消融：是否使用多头PPO
        "USE_VALUE_LOSS_NORMALIZATION": [True],  # 消融：是否使用Value Loss归一化
    },
}

# 固定参数（所有配置共享）
MAX_TRAINING_STEPS = 10_000_000
USE_MANUAL_SEED = True

# ========================================================================
# 辅助函数
# ========================================================================


def generate_experiment_configs(config_group):
    """
    生成指定配置组的所有实验配置（笛卡尔积）

    Args:
        config_group: 配置组字典，例如 ABLATION_CONFIGS['v1']

    Returns:
        实验配置列表
    """
    keys = list(config_group.keys())
    values = list(config_group.values())

    configs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)

    return configs


def get_experiment_name(config):
    """
    根据配置生成实验名称

    格式: {mode}_d{dense}_{ltl}_{pretrain}_{ppo}_{vloss}_ent{entropy}_s{seed}_{timestamp}
    示例: smdp_d0.0_noLTL_pt_stdPPO_vNorm_ent0.01_s42_20250115_143022
    """
    mode = str(config.get("EXECUTION_MODE", "smdp")).lower()

    # 获取dense权重（根据mode选择）
    if mode == "smdp":
        dense_weight = config.get("SMDP_DENSE_REWARD_WEIGHT", 0.0)
    else:
        dense_weight = config.get("MDP_DENSE_REWARD_WEIGHT", 0.0)

    # 格式化权重的通用函数
    def format_weight(w):
        if w == int(w):
            return f"{int(w)}"
        elif w < 0.1:
            return f"{w:.3f}"
        else:
            return f"{w:.1f}"

    dense_str = format_weight(dense_weight)

    # LTL配置
    ltl_enabled = bool(config.get("LTL_ENABLED", False))
    if ltl_enabled:
        ltl_type = str(config.get("LTL_CONSTRAINT_TYPE", "HARD")).lower()
        enc = str(config.get("LTL_ENCODING_TYPE", "C"))

        # LTL势能类型（如果适用）
        if ltl_type == "ltl_potential":
            potential_type = config.get("LTL_POTENTIAL_TYPE", "TOPOLOGY")
            potential_suffix = "topo" if potential_type == "TOPOLOGY" else "auto"
            ltl_str = f"{ltl_type}_enc{enc}_{potential_suffix}"
        else:
            ltl_str = f"{ltl_type}_enc{enc}"
    else:
        ltl_str = "noLTL"

    # Seed
    seed = int(config.get("MANUAL_SEED", 42))

    # 多头PPO配置
    use_multi_head = bool(config.get("USE_MULTI_HEAD_PPO", True))
    ppo_str = "mhPPO" if use_multi_head else "stdPPO"

    # Value Loss归一化配置
    use_vloss_norm = bool(config.get("USE_VALUE_LOSS_NORMALIZATION", False))
    vloss_str = "vNorm" if use_vloss_norm else "vStd"

    # 熵系数配置
    entropy_beta = config.get("ENTROPY_BETA", 0.01)
    entropy_str = format_weight(entropy_beta)

    # 时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 组装名称
    base_name = (
        f"{mode}_d{dense_str}_{ltl_str}_{ppo_str}_{vloss_str}_ent{entropy_str}_s{seed}"
    )
    full_name = f"{base_name}_{timestamp}"

    return full_name


def update_parameters_file(config):
    """
    动态更新 parameters.py 文件中的参数

    注意：这会直接修改 parameters.py 文件
    """
    params_file = os.path.join(project_root, "parameters.py")

    # 读取当前文件内容
    with open(params_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 准备所有要更新的参数
    all_updates = {
        "MAX_TRAINING_STEPS": MAX_TRAINING_STEPS,
        "USE_MANUAL_SEED": USE_MANUAL_SEED,
    }
    all_updates.update(config)  # 合并配置中的参数

    # 逐行更新
    new_lines = []
    updated_params = set()

    for line in lines:
        updated = False
        for param_name, param_value in all_updates.items():
            # 使用正则表达式匹配赋值语句
            pattern = rf"^\s*{param_name}\s*=\s*.*$"
            if re.match(pattern, line) and not line.strip().startswith("#"):
                # 保留缩进
                indent = line[: len(line) - len(line.lstrip())]

                # 格式化新值
                if isinstance(param_value, bool):
                    new_value = "True" if param_value else "False"
                elif isinstance(param_value, str):
                    new_value = f"'{param_value}'"
                elif param_value is None:
                    new_value = "None"
                else:
                    new_value = str(param_value)

                new_line = f"{indent}{param_name} = {new_value}\n"
                new_lines.append(new_line)
                updated_params.add(param_name)
                updated = True
                break

        if not updated:
            new_lines.append(line)

    # 写回文件
    with open(params_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    missing = [k for k in all_updates.keys() if k not in updated_params]
    if missing:
        print(f"  ⚠ parameters.py 未找到以下参数（未更新）：{missing}")
    else:
        print(f"  ✓ parameters.py 已更新")


def run_single_experiment(config, exp_idx, total_exps):
    """运行单个实验"""
    exp_name = get_experiment_name(config)

    print(f"\n{'=' * 100}")
    print(f"实验 [{exp_idx}/{total_exps}]: {exp_name}")
    print(f"{'=' * 100}")
    print(f"配置:")
    for key, value in sorted(config.items()):
        print(f"  - {key}: {value}")
    print(f"  - MAX_TRAINING_STEPS: {MAX_TRAINING_STEPS:,}")
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 100}\n")

    # 更新参数文件
    update_parameters_file(config)

    # 运行训练
    start_time = time.time()
    try:
        driver_path = os.path.join(project_root, "driver.py")
        result = subprocess.run(
            ["python", driver_path],
            cwd=project_root,
            capture_output=False,  # 直接输出到终端
            text=True,
        )

        if result.returncode == 0:
            status = "success"
            print(f"\n✓ 实验 {exp_name} 完成！")
        else:
            status = "failed"
            print(f"\n✗ 实验 {exp_name} 失败！退出码: {result.returncode}")

    except KeyboardInterrupt:
        status = "interrupted"
        print(f"\n⚠ 实验 {exp_name} 被用户中断")
        raise

    except Exception as e:
        status = "error"
        print(f"\n✗ 实验 {exp_name} 发生错误: {e}")

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n{'=' * 100}")
    print(f"实验结束:")
    print(f"  - 状态: {status}")
    print(f"  - 耗时: {duration / 3600:.2f} 小时 ({duration / 60:.1f} 分钟)")
    print(f"  - 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 100}\n")

    return {
        "name": exp_name,
        "config": config,
        "status": status,
        "duration": duration,
        "start_time": start_time,
        "end_time": end_time,
    }


def save_progress(results, config_name, progress_file=None):
    """保存实验进度"""
    if progress_file is None:
        progress_file = f"ablation_progress_{config_name}.json"

    progress_path = os.path.join(project_root, progress_file)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 进度已保存到: {progress_path}")


def print_experiment_summary(configs, config_name):
    """打印实验总结信息"""
    print(f"\n{'=' * 100}")
    print(f"实验配置总结 - {config_name}")
    print(f"{'=' * 100}")
    print(f"总实验数: {len(configs)}")

    # 统计各参数的取值范围
    if configs:
        all_keys = set()
        for cfg in configs:
            all_keys.update(cfg.keys())

        print(f"消融维度:")
        for key in sorted(all_keys):
            values = sorted(list(set(cfg.get(key) for cfg in configs)))
            print(f"  - {key}: {values}")

    print(f"固定参数:")
    print(f"  - MAX_TRAINING_STEPS: {MAX_TRAINING_STEPS:,}")
    print(f"  - USE_MANUAL_SEED: {USE_MANUAL_SEED}")
    print(f"{'=' * 100}")


def main():
    """主函数"""
    # ========== 1. 解析命令行参数 ==========
    parser = argparse.ArgumentParser(
        description="运行消融实验（Multi-head PPO）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置说明:
  v1: 单seed测试
  v2: 多seed测试 (3个seed)
  v3: Beta消融测试 (3个beta值)
  v4: Seed×Beta交叉测试 (4个组合)

示例:
  python run_ablation.py --config v1   # 运行v1配置组
  python run_ablation.py -c v2         # 运行v2配置组（3个实验）

可在 ABLATION_CONFIGS 中自定义各配置组的参数
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        choices=["v1", "v2", "v3", "v4", "v5", "v6"],
        help="选择消融实验配置组 (v1/v2/v3/v4)",
    )

    args = parser.parse_args()

    # ========== 2. 获取选定的配置组 ==========
    print(f"\n{'=' * 100}")
    print(f"消融实验批量运行脚本 - 配置组: {args.config}")
    print(f"{'=' * 100}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    selected_config_group = ABLATION_CONFIGS[args.config]

    # ========== 3. 生成所有实验配置（笛卡尔积）==========
    all_configs = generate_experiment_configs(selected_config_group)
    total_exps = len(all_configs)

    # ========== 4. 打印总结 ==========
    print_experiment_summary(all_configs, args.config)

    # ========== 5. 询问用户确认 ==========
    response = input(f"\n确认开始运行所有 {total_exps} 个实验? (y/n): ")
    if response.lower() != "y":
        print("已取消")
        return

    # ========== 6. 批量运行所有实验 ==========
    results = []
    experiment_start_time = time.time()

    try:
        for i, config in enumerate(all_configs, 1):
            result = run_single_experiment(config, i, total_exps)
            results.append(result)

            # 每完成一个实验就保存进度
            save_progress(results, args.config)

            # 显示总体进度
            completed = len(results)
            success_count = sum(1 for r in results if r["status"] == "success")
            failed_count = sum(1 for r in results if r["status"] in ["failed", "error"])
            avg_duration = sum(r["duration"] for r in results) / completed
            remaining = total_exps - completed
            estimated_remaining_time = remaining * avg_duration

            print(f"\n【总体进度 - {args.config}】")
            print(
                f"  - 已完成: {completed}/{total_exps} ({completed / total_exps * 100:.1f}%)"
            )
            print(f"  - 成功: {success_count}, 失败: {failed_count}")
            print(f"  - 平均耗时: {avg_duration / 3600:.2f} 小时")
            print(f"  - 预计剩余时间: {estimated_remaining_time / 3600:.2f} 小时")
            print(f"\n")

            # 短暂休息，让系统清理资源
            if i < total_exps:
                print("等待 10 秒，让系统清理资源...")
                time.sleep(10)

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 100}")
        print("检测到 Ctrl+C，停止批量运行")
        print(f"{'=' * 100}")
        save_progress(results, args.config)

    # ========== 7. 最终总结 ==========
    total_duration = time.time() - experiment_start_time
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] in ["failed", "error"])
    interrupted_count = sum(1 for r in results if r["status"] == "interrupted")

    print(f"\n\n{'=' * 100}")
    print(f"所有实验运行完成！- 配置组: {args.config}")
    print(f"{'=' * 100}")
    print(f"总耗时: {total_duration / 3600:.2f} 小时 ({total_duration / 60:.1f} 分钟)")
    print(f"结果统计:")
    print(f"  - 总实验数: {len(results)}/{total_exps}")
    print(f"  - 成功: {success_count}")
    print(f"  - 失败: {failed_count}")
    print(f"  - 中断: {interrupted_count}")
    print(f"\n详细结果已保存到: ablation_progress_{args.config}.json")
    print(f"{'=' * 100}\n")

    # 列出所有失败的实验
    if failed_count > 0:
        print(f"\n失败的实验:")
        for r in results:
            if r["status"] in ["failed", "error"]:
                print(f"  - {r['name']}")


if __name__ == "__main__":
    main()
