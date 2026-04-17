"""
默认轨迹构建函数

提供轨迹聚合和 token 处理的默认实现。
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

    策略：取最后一条 record 作为完整轨迹。
    假设最后一条 record 已经包含完整的对话信息。
    优先使用已有的 token_ids，如果不存在则使用 tokenizer tokenize。

    Args:
        session_id: 会话 ID
        records: 多轮对话记录列表
        tokenizer: Tokenizer 对象
        answer: 标准答案（此默认实现不使用）

    Returns:
        Trajectory 对象（traj_reward 置空）

    Raises:
        ValueError: 如果 records 为空
    """
    if not records:
        raise ValueError(f"No records found for session {session_id}")

    # 取最后一条记录（包含完整对话）
    last_record = records[-1]

    # 优先使用已有的 token_ids
    prompt_ids = last_record.get('token_ids')
    response_ids = last_record.get('response_ids')

    # 如果没有 token_ids，使用 tokenizer tokenize
    if prompt_ids is None:
        prompt_text = last_record.get('prompt_text', '')
        if prompt_text:
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            prompt_ids = []

    if response_ids is None:
        response_text = last_record.get('response_text', '')
        if response_text:
            response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        else:
            response_ids = []

    return Trajectory(
        trajectory_id=session_id,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        traj_reward=None,  # 先置空，由 reward_compute_cls 填充
        metadata={
            'model': last_record.get('model'),
            'prompt_tokens': last_record.get('prompt_tokens'),
            'completion_tokens': last_record.get('completion_tokens'),
        }
    )
