"""
异步评估器 - 零阻塞训练的评估系统

特性：
1. 评估在后台线程运行，训练不被阻塞
2. 参数快照保证评估时使用固定的模型
3. 智能跳过机制：如果评估未完成，自动跳过后续评估请求
4. 完整的统计信息和日志
"""

import threading
import time
import copy
from typing import Optional, Dict, Any, List
from collections import deque


class AsyncEvaluator:
    """异步评估管理器"""
    
    def __init__(self, logger, meta_agents, max_history=10):
        """
        Args:
            logger: Logger实例
            meta_agents: Ray worker列表
            max_history: 保留的评估历史记录数量
        """
        self.logger = logger
        self.meta_agents = meta_agents
        
        # 评估线程状态
        self.eval_thread: Optional[threading.Thread] = None
        self.is_evaluating = threading.Event()  # 线程安全的标志
        
        # 当前评估信息
        self.current_eval_step = None
        self.current_eval_env_steps = None
        self.eval_start_time = None
        
        # 跳过的评估统计
        self.skipped_evals = []  # 记录被跳过的评估点
        self.completed_evals = deque(maxlen=max_history)  # 最近完成的评估
        self.lock = threading.Lock()  # 保护共享数据
        
    def is_running(self) -> bool:
        """检查评估是否正在进行"""
        return self.is_evaluating.is_set()
    
    def start_evaluation(self, weights: Dict[str, Any], benchmarks: List[Dict], 
                        training_step: int, total_env_steps: int) -> bool:
        """
        启动异步评估（非阻塞）
        
        Args:
            weights: 模型参数字典
            benchmarks: 评估基准列表
            training_step: 当前训练步数
            total_env_steps: 当前总环境交互步数
            
        Returns:
            bool: True=成功启动, False=上次评估未完成,已跳过
        """
        if self.is_running():
            # 上次评估还在进行，记录跳过信息
            with self.lock:
                skip_info = {
                    'training_step': training_step,
                    'total_env_steps': total_env_steps,
                    'time': time.time(),
                    'reason': f'评估{self.current_eval_step}仍在进行'
                }
                self.skipped_evals.append(skip_info)
            
            elapsed = time.time() - self.eval_start_time if self.eval_start_time else 0
            print(f"\n{'='*80}")
            print(f"[ASYNC EVAL] ⚠️  跳过评估请求")
            print(f"  - 请求评估: step={training_step}, env_steps={total_env_steps}")
            print(f"  - 当前评估: step={self.current_eval_step}, 已运行{elapsed:.1f}秒")
            print(f"  - 累计跳过: {len(self.skipped_evals)}次")
            print(f"{'='*80}\n")
            return False
        
        # 创建参数快照（深拷贝确保完全独立）
        print(f"\n[ASYNC EVAL] 📸 创建参数快照 (step={training_step})...")
        weights_snapshot = copy.deepcopy(weights)
        
        # 标记评估开始
        self.is_evaluating.set()
        with self.lock:
            self.current_eval_step = training_step
            self.current_eval_env_steps = total_env_steps
            self.eval_start_time = time.time()
        
        # 在后台线程启动评估
        self.eval_thread = threading.Thread(
            target=self._run_eval_thread,
            args=(weights_snapshot, benchmarks, training_step, total_env_steps),
            daemon=True,  # 主进程退出时自动终止
            name=f"AsyncEval-{training_step}"
        )
        self.eval_thread.start()
        
        print(f"[ASYNC EVAL] ✅ 异步评估已启动 (step={training_step})")
        print(f"[ASYNC EVAL] 🚀 训练继续运行，无需等待...")
        print(f"{'='*80}\n")
        return True
    
    def _run_eval_thread(self, weights: Dict[str, Any], benchmarks: List[Dict],
                         training_step: int, total_env_steps: int):
        """
        在后台线程执行评估
        
        注意：这个函数在独立线程中运行
        """
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            print(f"\n{'='*80}")
            print(f"[ASYNC EVAL] 📊 开始评估 (step={training_step}, env_steps={total_env_steps})")
            print(f"[ASYNC EVAL] 📋 使用快照时刻的模型参数")
            print(f"{'='*80}\n")
            
            # 调用原有的评估函数
            self.logger.run_evaluation(
                current_weights=weights,
                benchmarks=benchmarks,
                meta_agents=self.meta_agents,
                training_step=training_step,
                total_env_steps=total_env_steps
            )
            
            success = True
            
        except Exception as e:
            success = False
            error_msg = str(e)
            print(f"\n{'='*80}")
            print(f"[ASYNC EVAL] ❌ 评估失败 (step={training_step})")
            print(f"[ASYNC EVAL] 错误: {e}")
            print(f"{'='*80}\n")
            import traceback
            traceback.print_exc()
        
        finally:
            elapsed = time.time() - start_time
            
            # 记录评估完成
            with self.lock:
                eval_record = {
                    'training_step': training_step,
                    'total_env_steps': total_env_steps,
                    'elapsed_time': elapsed,
                    'success': success,
                    'error': error_msg,
                    'timestamp': time.time()
                }
                self.completed_evals.append(eval_record)
                
                # 清空跳过记录（新的评估周期开始）
                num_skipped = len(self.skipped_evals)
                self.skipped_evals.clear()
            
            # 打印总结
            print(f"\n{'='*80}")
            if success:
                print(f"[ASYNC EVAL] ✅ 评估完成")
            else:
                print(f"[ASYNC EVAL] ❌ 评估失败")
            print(f"  - 评估时刻: step={training_step}, env_steps={total_env_steps}")
            print(f"  - 耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
            if num_skipped > 0:
                print(f"  - 期间跳过: {num_skipped}次评估请求")
            print(f"  - 历史记录: 已完成{len(self.completed_evals)}次评估")
            print(f"{'='*80}\n")
            
            # 清除状态标志
            self.is_evaluating.clear()
            self.current_eval_step = None
            self.current_eval_env_steps = None
            self.eval_start_time = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取评估统计信息
        
        Returns:
            dict: 包含评估历史、跳过次数等信息
        """
        with self.lock:
            return {
                'is_running': self.is_running(),
                'current_eval_step': self.current_eval_step,
                'current_eval_env_steps': self.current_eval_env_steps,
                'eval_running_time': time.time() - self.eval_start_time if self.eval_start_time else 0,
                'num_completed': len(self.completed_evals),
                'num_skipped': len(self.skipped_evals),
                'completed_evals': list(self.completed_evals),
                'skipped_evals': list(self.skipped_evals),
            }
    
    def print_statistics(self):
        """打印评估统计信息"""
        stats = self.get_statistics()
        
        print(f"\n{'='*80}")
        print(f"[ASYNC EVAL] 📊 评估统计")
        print(f"{'='*80}")
        print(f"当前状态: {'🔄 评估中' if stats['is_running'] else '⏸️  空闲'}")
        
        if stats['is_running']:
            print(f"  - 评估时刻: step={stats['current_eval_step']}")
            print(f"  - 已运行: {stats['eval_running_time']:.1f}秒")
        
        print(f"\n已完成评估: {stats['num_completed']}次")
        if stats['completed_evals']:
            for i, record in enumerate(reversed(list(stats['completed_evals'])[-5:]), 1):
                status = "✅" if record['success'] else "❌"
                print(f"  {i}. {status} step={record['training_step']}, "
                      f"耗时={record['elapsed_time']:.1f}s")
        
        print(f"\n跳过评估: {stats['num_skipped']}次")
        if stats['skipped_evals']:
            print(f"  (因上次评估未完成)")
            for skip in stats['skipped_evals'][-3:]:
                print(f"  - step={skip['training_step']}: {skip['reason']}")
        
        print(f"{'='*80}\n")
    
    def wait_completion(self, timeout: Optional[float] = None) -> bool:
        """
        等待当前评估完成（训练结束时调用）
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            bool: True=成功完成, False=超时或无评估
        """
        if not self.is_running():
            print("[ASYNC EVAL] 无正在进行的评估")
            return True
        
        print(f"\n{'='*80}")
        print(f"[ASYNC EVAL] ⏳ 等待评估完成 (step={self.current_eval_step})...")
        if timeout:
            print(f"[ASYNC EVAL] 超时设置: {timeout}秒")
        print(f"{'='*80}\n")
        
        self.eval_thread.join(timeout=timeout)
        
        if self.is_running():
            print(f"\n[ASYNC EVAL] ⚠️  评估超时未完成")
            return False
        else:
            print(f"\n[ASYNC EVAL] ✅ 评估已完成")
            return True
    
    def force_stop(self):
        """
        强制停止评估（紧急情况使用）
        
        注意：由于Python的GIL，无法真正"杀死"线程
        只能标记为已停止，线程会自然结束
        """
        if self.is_running():
            print(f"\n[ASYNC EVAL] ⚠️  强制停止评估 (step={self.current_eval_step})")
            print(f"[ASYNC EVAL] 注意：线程将自然结束，无法立即终止")
            self.is_evaluating.clear()


# 使用示例
if __name__ == "__main__":
    # 这只是示例代码，不会实际运行
    print("AsyncEvaluator 类已定义")
    print("使用方法见 driver.py 的集成代码")

