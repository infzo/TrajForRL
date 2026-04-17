"""
TrajForRL - 轨迹处理系统

处理 Rollout 阶段的轨迹数据，转换为 verl 框架训练所需的 DataProto 格式。

核心模块：
- VAEEHandler: 单轨迹处理器
- VerlConverter: 批量格式转换器
- Trajectory: 标准化轨迹数据类
"""

from .schema import Trajectory
from .vaee_handler import VAEEHandler
from .verl_converter import VerlConverter

__all__ = [
    'VAEEHandler',
    'VerlConverter',
    'Trajectory',
]
