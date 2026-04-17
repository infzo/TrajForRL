"""
默认轨迹构建函数

提供轨迹聚合和 token 处理的默认实现。

支持多轮对话拼接，正确构建 step_masks 区分历史 token 和生成 token。
参考 rllm/engine/agent_execution_engine.py:assemble_steps
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING

from traj_for_rl.schema import Trajectory

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def default_trajectory_construct_cls(
    session_id: str,
    records: List[Dict[str, Any]],
    tokenizer: "PreTrainedTokenizer",
    answer: Optional[str] = None,
) -> Trajectory:
    """默认轨迹处理器

    支持多轮对话拼接，正确构建 step_masks。

    多轮对话处理策略（参考 rllm assemble_steps）：
    1. prompt_ids = 第一条 record 的 token_ids（初始 prompt）
    2. response_ids = 所有轮次的 completion_ids + 增量 prompt
    3. step_masks = 标记哪些 token 参与 loss（只有 completion_ids 参与）

    Args:
        session_id: 会话 ID
        records: 多轮对话记录列表
                 每条 record 的 token_ids 包含完整历史（累积式）
        tokenizer: Tokenizer 对象
        answer: 标准答案（此默认实现不使用）

    Returns:
        Trajectory 对象（traj_reward 置空）

    Raises:
        ValueError: 如果 records 为空
    """
    if not records:
        raise ValueError(f"No records found for session {session_id}")

    # 单轮对话：直接使用，所有 response 都参与 loss
    if len(records) == 1:
        record = records[0]
        prompt_ids = record.get('token_ids')
        response_ids = record.get('response_ids')

        # 如果没有 token_ids，使用 tokenizer tokenize
        if prompt_ids is None:
            prompt_text = record.get('prompt_text', '')
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False) if prompt_text else []

        if response_ids is None:
            response_text = record.get('response_text', '')
            response_ids = tokenizer.encode(response_text, add_special_tokens=False) if response_text else []

        # 单轮对话：所有 response 都参与 loss
        step_masks = [1] * len(response_ids) if response_ids else None

        return Trajectory(
            trajectory_id=session_id,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            step_masks=step_masks,
            traj_reward=None,  # 先置空，由 reward_compute_cls 填充
            metadata={
                'model': record.get('model'),
                'prompt_tokens': record.get('prompt_tokens'),
                'completion_tokens': record.get('completion_tokens'),
            }
        )

    # 多轮对话：拼接所有轮次，构建 step_masks
    # 初始 prompt 来自第一条 record
    initial_prompt_ids = records[0].get('token_ids')
    if initial_prompt_ids is None:
        initial_prompt_text = records[0].get('prompt_text', '')
        initial_prompt_ids = tokenizer.encode(initial_prompt_text, add_special_tokens=False) if initial_prompt_text else []

    accumulated_sequence = initial_prompt_ids.copy()
    response_tokens = []
    step_masks = []

    for i, record in enumerate(records):
        current_prompt_ids = record.get('token_ids')
        current_response_ids = record.get('response_ids')

        # 如果没有 token_ids，使用 tokenizer tokenize
        if current_prompt_ids is None:
            prompt_text = record.get('prompt_text', '')
            current_prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False) if prompt_text else []

        if current_response_ids is None:
            response_text = record.get('response_text', '')
            current_response_ids = tokenizer.encode(response_text, add_special_tokens=False) if response_text else []

        if i == 0:
            # 第一轮：completion 参与 loss
            response_tokens.extend(current_response_ids)
            step_masks.extend([1] * len(current_response_ids))
            accumulated_sequence = current_prompt_ids + current_response_ids
        else:
            # 后续轮次：增量 prompt 不参与 loss，completion 参与 loss
            new_prompt_len = len(current_prompt_ids) - len(accumulated_sequence)
            response_tokens.extend(current_prompt_ids[len(accumulated_sequence):] + current_response_ids)
            step_masks.extend([0] * new_prompt_len + [1] * len(current_response_ids))
            accumulated_sequence = current_prompt_ids + current_response_ids

    return Trajectory(
        trajectory_id=session_id,
        prompt_ids=initial_prompt_ids,
        response_ids=response_tokens,
        step_masks=step_masks if step_masks else None,
        traj_reward=None,  # 先置空，由 reward_compute_cls 填充
        metadata={
            'model': records[-1].get('model'),
            'n_steps': len(records),
        }
    )
