"""
默认奖励计算函数

提供奖励计算的默认实现。
"""

from typing import Optional

from traj_for_rl.schema import Trajectory


def default_reward_compute_cls(
    trajectory: Trajectory,
    answer: Optional[str] = None,
) -> Trajectory:
    """默认奖励计算器

    返回填充 traj_reward=0.0 的 Trajectory。
    实际使用时应提供自定义的奖励计算函数。

    Args:
        trajectory: Trajectory 对象（包含 prompt_ids、response_ids、metadata）
        answer: 标准答案（此默认实现不使用）

    Returns:
        Trajectory: 填充 traj_reward 后的 Trajectory 对象
    """
    trajectory.traj_reward = 0.0
    return trajectory
