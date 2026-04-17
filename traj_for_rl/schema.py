"""
Trajectory 数据类和 Protocol 定义

提供轨迹处理相关的基础数据结构和接口定义。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class Trajectory:
    """处理后的单条轨迹

    V-AEE Handler 处理单个 session 后输出的标准化轨迹对象。

    Attributes:
        trajectory_id: 轨迹唯一标识（通常等于 session_id）
        prompt_ids: Prompt 部分的 token IDs
        response_ids: Response 部分的 token IDs
        traj_reward: 轨迹级别奖励（先置空，由 reward_compute_cls 填充）
        step_rewards: 预留字段，Step 级别奖励
        metadata: 元数据（可包含 model、token 统计等）
    """
    trajectory_id: str
    prompt_ids: List[int]
    response_ids: List[int]
    traj_reward: Optional[float] = None  # 先置空，后填充
    step_rewards: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryConstructCls(Protocol):
    """轨迹处理函数签名

    职责：
    1. 聚合/过滤/重组多条 records（多轮对话历史）
    2. 选择 Token 来源（已有的 token_ids 或使用 tokenizer 重新 tokenize）
    3. 返回 Trajectory 对象（traj_reward 置空）

    Args:
        session_id: 会话 ID
        records: 从 TrajProxy 获取的原始记录列表（多轮对话）
        tokenizer: Tokenizer 对象（用于 tokenize）
        answer: 可选的标准答案

    Returns:
        Trajectory: 轨迹对象（traj_reward 置空，由 reward_compute_cls 填充）
    """
    def __call__(
        self,
        session_id: str,
        records: List[Dict[str, Any]],
        tokenizer: "PreTrainedTokenizer",
        answer: Optional[str] = None,
    ) -> Trajectory: ...


class RewardComputeCls(Protocol):
    """奖励计算函数签名

    Args:
        trajectory: Trajectory 对象（包含 prompt_ids、response_ids、metadata）
        answer: 标准答案

    Returns:
        Trajectory: 填充 traj_reward 后的 Trajectory 对象
    """
    def __call__(
        self,
        trajectory: Trajectory,
        answer: Optional[str] = None,
    ) -> Trajectory: ...
