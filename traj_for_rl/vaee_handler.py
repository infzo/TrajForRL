"""
V-AEE Handler - 单轨迹处理器

对 Agent 单个 session 的请求结果进行后处理，输出标准化的 Trajectory 对象。
"""

import logging
from typing import TYPE_CHECKING, Optional

from .schema import Trajectory, TrajectoryConstructCls, RewardComputeCls

if TYPE_CHECKING:
    from traj_proxy.store.request_repository import RequestRepository

logger = logging.getLogger(__name__)


class VAEEHandler:
    """V-AEE Handler - 单轨迹处理器

    处理单个 session 的多轮对话记录，输出标准化的 Trajectory 对象。
    内部调用 TrajProxy 获取 records。

    一个 session 包含多条 records（多轮对话历史），Handler 负责聚合这些记录并计算奖励。

    设计原则：
    - Handler 内部调用 TrajProxy 获取 records，上层只需传入 session_id
    - 采用异步调用方式（async/await）
    - 每次处理一个 session，上层循环调用处理多个 session
    """

    def __init__(
        self,
        request_repository: "RequestRepository",
        trajectory_construct_cls: TrajectoryConstructCls,
        reward_compute_cls: RewardComputeCls,
    ):
        """
        Args:
            request_repository: TrajProxy 的 RequestRepository（用于获取 records）
            trajectory_construct_cls: 轨迹处理函数（负责聚合+tokenize）
            reward_compute_cls: 奖励计算函数
        """
        self.request_repository = request_repository
        self.trajectory_construct_cls = trajectory_construct_cls
        self.reward_compute_cls = reward_compute_cls

    async def process(
        self,
        session_id: str,
        tokenizer: "PreTrainedTokenizer",
        answer: Optional[str] = None,
    ) -> Trajectory:
        """
        处理单个 session 的轨迹数据

        Args:
            session_id: 会话 ID
            tokenizer: Tokenizer 对象（传给 trajectory_construct_cls）
            answer: 可选的标准答案，用于奖励计算

        Returns:
            Trajectory: 标准化的轨迹对象

        Raises:
            ValueError: 如果没有找到 records 或返回的 Trajectory 无效
        """
        # 1. 从 TrajProxy 获取 records
        records = await self.request_repository.get_all_by_session(session_id)
        if not records:
            logger.warning(f"No records found for session {session_id}")
            raise ValueError(f"No records found for session {session_id}")

        # 2. 轨迹处理（聚合 + tokenize），返回 Trajectory（traj_reward 置空）
        trajectory = self.trajectory_construct_cls(session_id, records, tokenizer, answer)

        # 3. 验证必需字段
        if not trajectory.prompt_ids or not trajectory.response_ids:
            logger.warning(f"Empty prompt_ids or response_ids for session {session_id}")
            raise ValueError("trajectory_construct_cls must return Trajectory with non-empty prompt_ids and response_ids")

        # 4. 奖励计算，填充 traj_reward
        trajectory = self.reward_compute_cls(trajectory, answer)

        return trajectory
