"""
VerlConverter - 批量格式转换器

接收多条轨迹，批量转换为 verl DataProto 格式。
"""

import logging
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from tensordict import TensorDict
    from verl.protocol import DataProto

from .schema import Trajectory

logger = logging.getLogger(__name__)


class VerlConverter:
    """VerlConverter - 批量格式转换器

    输出 verl DataProto 格式。
    配置在初始化时设定，输出固定在 CPU。
    """

    def __init__(
        self,
        max_prompt_length: int,
        max_response_length: int,
        pad_token_id: int,
    ):
        """
        Args:
            max_prompt_length: 最大 prompt 长度（左填充到此长度）
            max_response_length: 最大 response 长度（右填充到此长度）
            pad_token_id: Padding token ID
        """
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.pad_token_id = pad_token_id

    def convert(
        self,
        trajectories: List[Trajectory],
    ) -> "DataProto":
        """
        将轨迹列表转换为 verl DataProto 格式

        Args:
            trajectories: 轨迹列表

        Returns:
            DataProto: verl 兼容的数据协议对象（固定在 CPU）

        Raises:
            ValueError: 如果 trajectories 为空
        """
        if not trajectories:
            logger.warning("Empty trajectories list")
            raise ValueError("trajectories list is empty")

        # 延迟导入，避免模块加载时依赖 verl
        from tensordict import TensorDict
        from verl.protocol import DataProto

        # 1. Padding
        prompts_batch = self._pad_left(
            [t.prompt_ids for t in trajectories],
            self.max_prompt_length, self.pad_token_id
        )
        responses_batch = self._pad_right(
            [t.response_ids for t in trajectories],
            self.max_response_length, self.pad_token_id
        )

        # 2. input_ids = prompts + responses
        input_ids = torch.cat([prompts_batch, responses_batch], dim=1)

        # 3. attention_mask
        attention_mask = self._build_attention_mask(
            prompts_batch, responses_batch, self.pad_token_id
        )

        # 4. position_ids (cumsum of attention_mask)
        position_ids = self._build_position_ids(attention_mask)

        # 5. response_mask (标记哪些 token 参与 loss 计算)
        # 优先使用 step_masks（多轮对话中区分历史和生成 token）
        # 如果 step_masks 不存在，使用简单的 padding mask
        if all(t.step_masks is not None for t in trajectories):
            response_mask = self._pad_right(
                [t.step_masks for t in trajectories],
                self.max_response_length, 0  # padding 值为 0
            )
        else:
            # fallback: 所有有效 token 都参与 loss
            response_mask = (responses_batch != self.pad_token_id).long()

        # 6. rewards (放在 response 最后一个有效 token 位置)
        token_level_rewards = self._build_rewards(
            [t.traj_reward for t in trajectories],
            [len(t.response_ids) for t in trajectories],
            self.max_response_length
        )
        token_level_scores = torch.zeros_like(token_level_rewards)  # 暂与 rewards 相同

        # 7. 构建 non_tensor_batch（用于 verl 训练流程）
        non_tensor_batch = {
            'uid': np.array([t.trajectory_id for t in trajectories]),
        }
        # 可选字段：ground_truth, data_source（从 metadata 获取）
        if all('ground_truth' in t.metadata for t in trajectories):
            non_tensor_batch['ground_truth'] = np.array([t.metadata['ground_truth'] for t in trajectories])
        if all('data_source' in t.metadata for t in trajectories):
            non_tensor_batch['data_source'] = np.array([t.metadata['data_source'] for t in trajectories])

        # 8. 组装 DataProto（使用 verl 原生字段命名）
        batch = TensorDict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'prompts': prompts_batch,
            'responses': responses_batch,
            'response_mask': response_mask,
            'token_level_rewards': token_level_rewards,
            'token_level_scores': token_level_scores,
        }, batch_size=len(trajectories))

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _pad_left(self, sequences: List[List[int]], max_len: int, pad_id: int) -> torch.Tensor:
        """左填充序列，超出长度时从左侧截断"""
        tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        padded = []
        for t in tensors:
            if len(t) < max_len:
                pad_size = max_len - len(t)
                t = torch.cat([torch.full((pad_size,), pad_id, dtype=torch.long), t])
            else:
                t = t[-max_len:]  # 从左侧截断
            padded.append(t)
        return torch.stack(padded)

    def _pad_right(self, sequences: List[List[int]], max_len: int, pad_id: int) -> torch.Tensor:
        """右填充序列，超出长度时从右侧截断"""
        tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        padded = []
        for t in tensors:
            if len(t) < max_len:
                pad_size = max_len - len(t)
                t = torch.cat([t, torch.full((pad_size,), pad_id, dtype=torch.long)])
            else:
                t = t[:max_len]  # 从右侧截断
            padded.append(t)
        return torch.stack(padded)

    def _build_attention_mask(
        self, prompts: torch.Tensor, responses: torch.Tensor, pad_id: int
    ) -> torch.Tensor:
        """构建 attention mask: 1 表示有效 token, 0 表示 padding"""
        prompts_mask = (prompts != pad_id).long()
        responses_mask = (responses != pad_id).long()
        return torch.cat([prompts_mask, responses_mask], dim=1)

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """计算 position ids: cumsum of attention_mask - 1"""
        return (attention_mask.cumsum(dim=1) - 1) * attention_mask

    def _build_rewards(
        self, rewards: List[Optional[float]], lengths: List[int], max_len: int
    ) -> torch.Tensor:
        """构建奖励 tensor，放在 response 最后一个有效 token 位置"""
        batch_size = len(rewards)
        reward_tensor = torch.zeros(batch_size, max_len, dtype=torch.float32)
        for i, (reward, length) in enumerate(zip(rewards, lengths)):
            if reward is not None and length > 0 and length <= max_len:
                reward_tensor[i, length - 1] = reward
        return reward_tensor
