"""
VAEEHandler 模块测试

测试覆盖：
- 正常流程：从 records 构建 Trajectory
- 边界条件：空 records、空 prompt_ids、空 response_ids
- tokenize 流程：使用 tokenizer 从文本生成 token_ids
- 自定义处理器：使用自定义 trajectory_construct_cls
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizer

from traj_for_rl.dataclasses import Trajectory
from traj_for_rl.vaee_handler import VAEEHandler
from traj_for_rl.processors.defaults import default_trajectory_construct_cls, default_reward_compute_cls


# ============ Fixtures ============

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer"""
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    tokenizer.encode = MagicMock(side_effect=lambda text, add_special_tokens=False: [ord(c) for c in text])
    return tokenizer


@pytest.fixture
def mock_request_repository():
    """Mock RequestRepository"""
    repo = AsyncMock()
    return repo


@pytest.fixture
def sample_records():
    """示例 records 数据"""
    return [
        {
            'token_ids': [1, 2, 3],
            'response_ids': [4, 5, 6],
            'model': 'test-model',
            'prompt_tokens': 3,
            'completion_tokens': 3,
        }
    ]


@pytest.fixture
def sample_records_with_text():
    """包含文本的 records 数据"""
    return [
        {
            'prompt_text': 'Hello',
            'response_text': 'World',
            'model': 'test-model',
        }
    ]


# ============ 正常流程测试 ============

class TestVAEEHandlerNormal:
    """正常流程测试"""

    @pytest.mark.asyncio
    async def test_handler_with_default_processor(self, mock_request_repository, mock_tokenizer, sample_records):
        """测试默认处理器 - 正常流程"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = sample_records

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute
        trajectory = await handler.process('test_session', mock_tokenizer)

        # Verify
        assert isinstance(trajectory, Trajectory)
        assert trajectory.trajectory_id == 'test_session'
        assert trajectory.prompt_ids == [1, 2, 3]
        assert trajectory.response_ids == [4, 5, 6]
        assert trajectory.traj_reward == 0.0  # 由 default_reward_compute_cls 填充
        assert trajectory.metadata['model'] == 'test-model'

    @pytest.mark.asyncio
    async def test_handler_with_tokenizer(self, mock_request_repository, mock_tokenizer, sample_records_with_text):
        """测试使用 tokenizer tokenize"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = sample_records_with_text

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute
        trajectory = await handler.process('test_session', mock_tokenizer)

        # Verify
        assert len(trajectory.prompt_ids) > 0  # tokenizer.encode 被调用
        assert len(trajectory.response_ids) > 0
        mock_tokenizer.encode.assert_called()

    @pytest.mark.asyncio
    async def test_handler_with_answer(self, mock_request_repository, mock_tokenizer, sample_records):
        """测试传入 answer 参数"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = sample_records

        # 自定义 reward_compute_cls 使用 answer
        def custom_reward_cls(trajectory: Trajectory, answer: Optional[str] = None) -> Trajectory:
            trajectory.traj_reward = 1.0 if answer == 'correct' else 0.0
            return trajectory

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=custom_reward_cls,
        )

        # Execute
        trajectory = await handler.process('test_session', mock_tokenizer, answer='correct')

        # Verify
        assert trajectory.traj_reward == 1.0


# ============ 边界条件测试 ============

class TestVAEEHandlerBoundary:
    """边界条件测试"""

    @pytest.mark.asyncio
    async def test_handler_empty_records(self, mock_request_repository, mock_tokenizer):
        """测试空 records - 应抛出 ValueError"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = []

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute & Verify
        with pytest.raises(ValueError, match="No records found for session"):
            await handler.process('test_session', mock_tokenizer)

    @pytest.mark.asyncio
    async def test_handler_empty_prompt_ids(self, mock_request_repository, mock_tokenizer):
        """测试空 prompt_ids - 应抛出 ValueError"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = [
            {'response_ids': [4, 5, 6]}  # 没有 token_ids 和 prompt_text
        ]

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute & Verify
        with pytest.raises(ValueError, match="non-empty prompt_ids"):
            await handler.process('test_session', mock_tokenizer)

    @pytest.mark.asyncio
    async def test_handler_empty_response_ids(self, mock_request_repository, mock_tokenizer):
        """测试空 response_ids - 应抛出 ValueError"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = [
            {'token_ids': [1, 2, 3]}  # 没有 response_ids 和 response_text
        ]

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute & Verify
        with pytest.raises(ValueError, match="non-empty response_ids"):
            await handler.process('test_session', mock_tokenizer)

    @pytest.mark.asyncio
    async def test_handler_none_traj_reward(self, mock_request_repository, mock_tokenizer, sample_records):
        """测试 traj_reward 为 None 时被默认填充"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = sample_records

        # 自定义 reward_compute_cls 不填充 reward
        def no_reward_cls(trajectory: Trajectory, answer: Optional[str] = None) -> Trajectory:
            return trajectory  # 不填充 traj_reward

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=no_reward_cls,
        )

        # Execute
        trajectory = await handler.process('test_session', mock_tokenizer)

        # Verify - traj_reward 保持 None（由上层处理）
        assert trajectory.traj_reward is None


# ============ 自定义处理器测试 ============

class TestVAEEHandlerCustomProcessor:
    """自定义处理器测试"""

    @pytest.mark.asyncio
    async def test_handler_custom_trajectory_construct(self, mock_request_repository, mock_tokenizer):
        """测试自定义 trajectory_construct_cls"""
        # Setup
        records = [
            {'token_ids': [1, 2, 3], 'response_ids': [4, 5, 6]},
            {'token_ids': [7, 8], 'response_ids': [9, 10]},
        ]
        mock_request_repository.get_all_by_session.return_value = records

        # 自定义：聚合所有 records
        def custom_construct_cls(
            session_id: str,
            records: List[Dict[str, Any]],
            tokenizer: PreTrainedTokenizer,
            answer: Optional[str] = None
        ) -> Trajectory:
            all_prompt_ids = []
            all_response_ids = []
            for r in records:
                all_prompt_ids.extend(r.get('token_ids', []))
                all_response_ids.extend(r.get('response_ids', []))
            return Trajectory(
                trajectory_id=session_id,
                prompt_ids=all_prompt_ids,
                response_ids=all_response_ids,
            )

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=custom_construct_cls,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute
        trajectory = await handler.process('test_session', mock_tokenizer)

        # Verify - 所有 records 被聚合
        assert trajectory.prompt_ids == [1, 2, 3, 7, 8]
        assert trajectory.response_ids == [4, 5, 6, 9, 10]

    @pytest.mark.asyncio
    async def test_handler_custom_reward_compute(self, mock_request_repository, mock_tokenizer, sample_records):
        """测试自定义 reward_compute_cls"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = sample_records

        def custom_reward_cls(trajectory: Trajectory, answer: Optional[str] = None) -> Trajectory:
            # 根据长度计算奖励
            trajectory.traj_reward = len(trajectory.response_ids) * 0.1
            trajectory.metadata['reward_type'] = 'length_based'
            return trajectory

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=custom_reward_cls,
        )

        # Execute
        trajectory = await handler.process('test_session', mock_tokenizer)

        # Verify
        assert trajectory.traj_reward == 0.3  # 3 tokens * 0.1
        assert trajectory.metadata['reward_type'] == 'length_based'


# ============ Metadata 传递测试 ============

class TestVAEEHandlerMetadata:
    """Metadata 传递测试"""

    @pytest.mark.asyncio
    async def test_handler_metadata_ground_truth(self, mock_request_repository, mock_tokenizer):
        """测试 ground_truth 在 metadata 中传递"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = [
            {
                'token_ids': [1, 2, 3],
                'response_ids': [4, 5, 6],
                'model': 'test-model',
            }
        ]

        def construct_with_ground_truth(
            session_id: str,
            records: List[Dict[str, Any]],
            tokenizer: PreTrainedTokenizer,
            answer: Optional[str] = None
        ) -> Trajectory:
            traj = default_trajectory_construct_cls(session_id, records, tokenizer, answer)
            if answer:
                traj.metadata['ground_truth'] = answer
            return traj

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=construct_with_ground_truth,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute
        trajectory = await handler.process('test_session', mock_tokenizer, answer='expected_answer')

        # Verify
        assert trajectory.metadata['ground_truth'] == 'expected_answer'

    @pytest.mark.asyncio
    async def test_handler_metadata_data_source(self, mock_request_repository, mock_tokenizer, sample_records):
        """测试 data_source 在 metadata 中传递"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = sample_records

        def construct_with_data_source(
            session_id: str,
            records: List[Dict[str, Any]],
            tokenizer: PreTrainedTokenizer,
            answer: Optional[str] = None
        ) -> Trajectory:
            traj = default_trajectory_construct_cls(session_id, records, tokenizer, answer)
            traj.metadata['data_source'] = 'gsm8k'
            return traj

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=construct_with_data_source,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute
        trajectory = await handler.process('test_session', mock_tokenizer)

        # Verify
        assert trajectory.metadata['data_source'] == 'gsm8k'


# ============ 多轮对话场景测试 ============

class TestVAEEHandlerMultiTurn:
    """多轮对话场景测试

    文档强调：一个 session 包含多条 records（多轮对话历史），
    default_trajectory_construct_cls 取最后一条 record 作为完整轨迹。
    """

    @pytest.mark.asyncio
    async def test_handler_multi_turn_default_processor(self, mock_request_repository, mock_tokenizer):
        """测试多轮对话 - 默认处理器取最后一条 record"""
        # Setup - 模拟多轮对话
        records = [
            {
                'token_ids': [1, 2],
                'response_ids': [3, 4],
                'model': 'test-model',
                'prompt_tokens': 2,
                'completion_tokens': 2,
            },
            {
                'token_ids': [5, 6, 7],
                'response_ids': [8, 9, 10],
                'model': 'test-model',
                'prompt_tokens': 3,
                'completion_tokens': 3,
            },
            {
                'token_ids': [11, 12, 13, 14],
                'response_ids': [15, 16, 17, 18],
                'model': 'test-model',
                'prompt_tokens': 4,
                'completion_tokens': 4,
            },
        ]
        mock_request_repository.get_all_by_session.return_value = records

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute
        trajectory = await handler.process('multi_turn_session', mock_tokenizer)

        # Verify - 取最后一条 record
        assert trajectory.trajectory_id == 'multi_turn_session'
        assert trajectory.prompt_ids == [11, 12, 13, 14]
        assert trajectory.response_ids == [15, 16, 17, 18]
        assert trajectory.metadata['prompt_tokens'] == 4
        assert trajectory.metadata['completion_tokens'] == 4

    @pytest.mark.asyncio
    async def test_handler_multi_turn_with_tokenizer(self, mock_request_repository, mock_tokenizer):
        """测试多轮对话 - 使用 tokenizer 从文本 tokenize"""
        # Setup
        records = [
            {
                'prompt_text': 'First turn',
                'response_text': 'First response',
                'model': 'test-model',
            },
            {
                'prompt_text': 'Second turn',
                'response_text': 'Second response',
                'model': 'test-model',
            },
        ]
        mock_request_repository.get_all_by_session.return_value = records

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute
        trajectory = await handler.process('multi_turn_session', mock_tokenizer)

        # Verify - 最后一条 record 被 tokenize
        assert len(trajectory.prompt_ids) > 0
        assert len(trajectory.response_ids) > 0
        # 验证 tokenizer.encode 被调用（mock 实现）
        mock_tokenizer.encode.assert_called()

    @pytest.mark.asyncio
    async def test_handler_multi_turn_custom_aggregation(self, mock_request_repository, mock_tokenizer):
        """测试多轮对话 - 自定义聚合所有 records"""
        # Setup
        records = [
            {'token_ids': [1, 2], 'response_ids': [3, 4]},
            {'token_ids': [5, 6], 'response_ids': [7, 8]},
            {'token_ids': [9, 10], 'response_ids': [11, 12]},
        ]
        mock_request_repository.get_all_by_session.return_value = records

        # 自定义：聚合所有 records
        def aggregate_all_construct_cls(
            session_id: str,
            records: List[Dict[str, Any]],
            tokenizer: PreTrainedTokenizer,
            answer: Optional[str] = None
        ) -> Trajectory:
            all_prompt_ids = []
            all_response_ids = []
            for r in records:
                all_prompt_ids.extend(r.get('token_ids', []))
                all_response_ids.extend(r.get('response_ids', []))
            return Trajectory(
                trajectory_id=session_id,
                prompt_ids=all_prompt_ids,
                response_ids=all_response_ids,
            )

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=aggregate_all_construct_cls,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute
        trajectory = await handler.process('aggregate_session', mock_tokenizer)

        # Verify - 所有 records 被聚合
        assert trajectory.prompt_ids == [1, 2, 5, 6, 9, 10]
        assert trajectory.response_ids == [3, 4, 7, 8, 11, 12]


# ============ 异常传播测试 ============

class TestVAEEHandlerException:
    """异常传播测试"""

    @pytest.mark.asyncio
    async def test_handler_repository_exception(self, mock_request_repository, mock_tokenizer):
        """测试 RequestRepository 抛出异常时正确传播"""
        # Setup
        mock_request_repository.get_all_by_session.side_effect = ConnectionError("Database connection failed")

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute & Verify
        with pytest.raises(ConnectionError, match="Database connection failed"):
            await handler.process('test_session', mock_tokenizer)

    @pytest.mark.asyncio
    async def test_handler_trajectory_construct_exception(self, mock_request_repository, mock_tokenizer):
        """测试 trajectory_construct_cls 抛出异常时正确传播"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = [
            {'token_ids': [1, 2, 3], 'response_ids': [4, 5, 6]}
        ]

        def failing_construct_cls(session_id, records, tokenizer, answer):
            raise RuntimeError("Construct failed")

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=failing_construct_cls,
            reward_compute_cls=default_reward_compute_cls,
        )

        # Execute & Verify
        with pytest.raises(RuntimeError, match="Construct failed"):
            await handler.process('test_session', mock_tokenizer)

    @pytest.mark.asyncio
    async def test_handler_reward_compute_exception(self, mock_request_repository, mock_tokenizer):
        """测试 reward_compute_cls 抛出异常时正确传播"""
        # Setup
        mock_request_repository.get_all_by_session.return_value = [
            {'token_ids': [1, 2, 3], 'response_ids': [4, 5, 6]}
        ]

        def failing_reward_cls(trajectory, answer):
            raise RuntimeError("Reward compute failed")

        handler = VAEEHandler(
            request_repository=mock_request_repository,
            trajectory_construct_cls=default_trajectory_construct_cls,
            reward_compute_cls=failing_reward_cls,
        )

        # Execute & Verify
        with pytest.raises(RuntimeError, match="Reward compute failed"):
            await handler.process('test_session', mock_tokenizer)
