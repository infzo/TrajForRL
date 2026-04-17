"""
VerlConverter 模块测试

测试覆盖：
- 正常转换：Trajectory 列表转 DataProto
- 填充测试：左填充 prompt、右填充 response
- 截断测试：超长 prompt/response 截断
- Reward 位置：验证 reward 在 response 最后有效 token 位置
- Tensor 值验证：验证 input_ids、attention_mask、position_ids、token_level_rewards 内部值
- Non-Tensor 字段：验证 uid、ground_truth、data_source
- 边界条件：空 response、None reward
"""

import pytest
import torch
import numpy as np

from traj_for_rl import Trajectory
from traj_for_rl.verl_converter import VerlConverter


# ============ Fixtures ============

@pytest.fixture
def converter():
    """创建测试用 Converter"""
    return VerlConverter(
        max_prompt_length=5,
        max_response_length=5,
        pad_token_id=0,
    )


@pytest.fixture
def converter_large():
    """创建大长度 Converter"""
    return VerlConverter(
        max_prompt_length=10,
        max_response_length=10,
        pad_token_id=0,
    )


@pytest.fixture
def sample_trajectories():
    """示例 trajectories"""
    return [
        Trajectory(
            trajectory_id='test_1',
            prompt_ids=[1, 2, 3],
            response_ids=[4, 5, 6],
            traj_reward=1.0,
        ),
        Trajectory(
            trajectory_id='test_2',
            prompt_ids=[7, 8],
            response_ids=[9, 10, 11, 12],
            traj_reward=0.5,
        ),
    ]


# ============ 正常转换测试 ============

class TestVerlConverterNormal:
    """正常转换测试"""

    def test_converter_normal(self, converter, sample_trajectories):
        """测试正常转换流程"""
        # Execute
        data_proto = converter.convert(sample_trajectories)

        # Verify batch size
        assert len(data_proto) == 2

        # Verify tensor shapes
        assert data_proto.batch['input_ids'].shape == (2, 10)  # 5 + 5
        assert data_proto.batch['prompts'].shape == (2, 5)
        assert data_proto.batch['responses'].shape == (2, 5)

        # Verify field names
        assert 'input_ids' in data_proto.batch
        assert 'attention_mask' in data_proto.batch
        assert 'position_ids' in data_proto.batch
        assert 'prompts' in data_proto.batch
        assert 'responses' in data_proto.batch
        assert 'response_mask' in data_proto.batch
        assert 'token_level_rewards' in data_proto.batch
        assert 'token_level_scores' in data_proto.batch

    def test_converter_non_tensor_batch(self, converter, sample_trajectories):
        """测试 non_tensor_batch 字段"""
        # Execute
        data_proto = converter.convert(sample_trajectories)

        # Verify uid
        assert 'uid' in data_proto.non_tensor_batch
        assert data_proto.non_tensor_batch['uid'][0] == 'test_1'
        assert data_proto.non_tensor_batch['uid'][1] == 'test_2'

    def test_converter_with_metadata(self, converter):
        """测试包含 ground_truth 和 data_source 的 metadata"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test_1',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6],
                traj_reward=1.0,
                metadata={'ground_truth': 'answer_1', 'data_source': 'gsm8k'},
            ),
            Trajectory(
                trajectory_id='test_2',
                prompt_ids=[7, 8],
                response_ids=[9, 10],
                traj_reward=0.5,
                metadata={'ground_truth': 'answer_2', 'data_source': 'math'},
            ),
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify ground_truth
        assert 'ground_truth' in data_proto.non_tensor_batch
        assert data_proto.non_tensor_batch['ground_truth'][0] == 'answer_1'
        assert data_proto.non_tensor_batch['ground_truth'][1] == 'answer_2'

        # Verify data_source
        assert 'data_source' in data_proto.non_tensor_batch
        assert data_proto.non_tensor_batch['data_source'][0] == 'gsm8k'
        assert data_proto.non_tensor_batch['data_source'][1] == 'math'

    def test_converter_empty_trajectories(self, converter):
        """测试空 trajectories - 应抛出 ValueError"""
        with pytest.raises(ValueError, match="trajectories list is empty"):
            converter.convert([])


# ============ 填充测试 ============

class TestVerlConverterPadding:
    """填充测试"""

    def test_converter_left_pad_prompt(self, converter):
        """测试 prompt 左填充"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: 左填充 prompt = [PAD, PAD, 1, 2, 3]
        expected_prompt = [0, 0, 1, 2, 3]
        assert data_proto.batch['prompts'][0].tolist() == expected_prompt

    def test_converter_right_pad_response(self, converter):
        """测试 response 右填充"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: 右填充 response = [4, 5, 6, PAD, PAD]
        expected_response = [4, 5, 6, 0, 0]
        assert data_proto.batch['responses'][0].tolist() == expected_response

    def test_converter_attention_mask(self, converter):
        """测试 attention_mask 构建"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify attention_mask
        # prompt: [PAD, PAD, 1, 2, 3] -> mask: [0, 0, 1, 1, 1]
        # response: [4, 5, 6, PAD, PAD] -> mask: [1, 1, 1, 0, 0]
        expected_mask = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        assert data_proto.batch['attention_mask'][0].tolist() == expected_mask

    def test_converter_response_mask(self, converter):
        """测试 response_mask 构建"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify response_mask: [1, 1, 1, 0, 0]
        expected_response_mask = [1, 1, 1, 0, 0]
        assert data_proto.batch['response_mask'][0].tolist() == expected_response_mask


# ============ position_ids 测试 ============

class TestVerlConverterPositionIds:
    """position_ids 测试"""

    def test_converter_position_ids_normal(self, converter):
        """测试正常 position_ids 计算"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify position_ids
        # attention_mask: [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        # cumsum - 1:     [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        # * mask:         [0, 0, 0, 1, 2, 3, 4, 5, 0, 0]
        expected_position_ids = [0, 0, 0, 1, 2, 3, 4, 5, 0, 0]
        assert data_proto.batch['position_ids'][0].tolist() == expected_position_ids

    def test_converter_position_ids_all_padding(self, converter):
        """测试全 padding 序列的 position_ids"""
        # Setup - 空 prompt 和 response 会被填充为全 PAD
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1],  # 最小有效 token
                response_ids=[2],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify - position_ids 从 0 开始
        # [PAD, PAD, PAD, PAD, 1, 2, PAD, PAD, PAD, PAD]
        # attention_mask: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        # position_ids:   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        position_ids = data_proto.batch['position_ids'][0].tolist()
        # 验证有效部分的 position 是连续的
        assert position_ids[4] == 0
        assert position_ids[5] == 1


# ============ 截断测试 ============

class TestVerlConverterTruncation:
    """截断测试"""

    def test_converter_truncate_prompt_left(self, converter):
        """测试 prompt 左截断"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8],  # 超过 max_prompt_length=5
                response_ids=[4, 5, 6],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: 左截断，取最后 5 个
        expected_prompt = [4, 5, 6, 7, 8]
        assert data_proto.batch['prompts'][0].tolist() == expected_prompt

    def test_converter_truncate_response_right(self, converter):
        """测试 response 右截断"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6, 7, 8, 9, 10],  # 超过 max_response_length=5
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: 右截断，取前 5 个
        expected_response = [4, 5, 6, 7, 8]
        assert data_proto.batch['responses'][0].tolist() == expected_response

    def test_converter_truncate_both(self, converter):
        """测试 prompt 和 response 都截断"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                response_ids=[9, 10, 11, 12, 13, 14, 15],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify
        assert data_proto.batch['prompts'][0].tolist() == [4, 5, 6, 7, 8]
        assert data_proto.batch['responses'][0].tolist() == [9, 10, 11, 12, 13]


# ============ Reward 位置测试 ============

class TestVerlConverterReward:
    """Reward 位置测试"""

    def test_converter_reward_position(self, converter):
        """测试 reward 在 response 最后有效 token 位置"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: response 有 3 个有效 token
        # reward 放在 index 2（最后一个有效 token）
        expected_rewards = [0.0, 0.0, 1.0, 0.0, 0.0]
        assert data_proto.batch['token_level_rewards'][0].tolist() == expected_rewards

    def test_converter_reward_different_lengths(self, converter):
        """测试不同长度 response 的 reward 位置"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test_1',
                prompt_ids=[1],
                response_ids=[4, 5, 6],
                traj_reward=1.0,
            ),
            Trajectory(
                trajectory_id='test_2',
                prompt_ids=[1],
                response_ids=[7, 8],
                traj_reward=0.5,
            ),
            Trajectory(
                trajectory_id='test_3',
                prompt_ids=[1],
                response_ids=[9],
                traj_reward=0.3,
            ),
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify
        rewards = data_proto.batch['token_level_rewards']

        # test_1: response 长度 3，reward 在 index 2
        assert rewards[0].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]

        # test_2: response 长度 2，reward 在 index 1
        assert rewards[1].tolist() == [0.0, 0.5, 0.0, 0.0, 0.0]

        # test_3: response 长度 1，reward 在 index 0
        assert rewards[2].tolist() == [0.3, 0.0, 0.0, 0.0, 0.0]

    def test_converter_reward_none(self, converter):
        """测试 traj_reward 为 None 时不放置 reward"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6],
                traj_reward=None,  # None reward
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: reward tensor 全为 0
        expected_rewards = [0.0, 0.0, 0.0, 0.0, 0.0]
        assert data_proto.batch['token_level_rewards'][0].tolist() == expected_rewards


# ============ 边界条件测试 ============

class TestVerlConverterBoundary:
    """边界条件测试"""

    def test_converter_empty_response(self, converter):
        """测试空 response - reward 不放置"""
        # 注意：VAEEHandler 会拦截空 response_ids
        # 这里测试 VerlConverter 对空 response 的处理
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4],  # 最小长度 1
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: reward 在 index 0
        assert data_proto.batch['token_level_rewards'][0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]

    def test_converter_single_token_response(self, converter):
        """测试单 token response"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify
        assert data_proto.batch['responses'][0].tolist() == [4, 0, 0, 0, 0]
        assert data_proto.batch['token_level_rewards'][0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]

    def test_converter_full_length_prompt(self, converter):
        """测试恰好满长度的 prompt（无填充）"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3, 4, 5],  # 恰好 max_prompt_length=5
                response_ids=[6, 7, 8],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: 无填充
        assert data_proto.batch['prompts'][0].tolist() == [1, 2, 3, 4, 5]
        assert data_proto.batch['attention_mask'][0][:5].tolist() == [1, 1, 1, 1, 1]

    def test_converter_full_length_response(self, converter):
        """测试恰好满长度的 response（无填充）"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6, 7, 8],  # 恰好 max_response_length=5
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify
        assert data_proto.batch['responses'][0].tolist() == [4, 5, 6, 7, 8]
        assert data_proto.batch['response_mask'][0].tolist() == [1, 1, 1, 1, 1]
        # reward 在最后一个有效 token (index 4)
        assert data_proto.batch['token_level_rewards'][0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


# ============ Tensor 值综合验证 ============

class TestVerlConverterTensorValues:
    """Tensor 值综合验证"""

    def test_converter_tensor_values_complete(self, converter):
        """综合验证所有 tensor 内部值"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test_complete',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify input_ids: [PAD, PAD, tok1, tok2, tok3, tok1, tok2, tok3, PAD, PAD]
        expected_input_ids = [0, 0, 1, 2, 3, 4, 5, 6, 0, 0]
        assert data_proto.batch['input_ids'][0].tolist() == expected_input_ids

        # Verify attention_mask: [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        expected_attention_mask = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        assert data_proto.batch['attention_mask'][0].tolist() == expected_attention_mask

        # Verify position_ids: [0, 0, 0, 1, 2, 3, 4, 5, 0, 0]
        expected_position_ids = [0, 0, 0, 1, 2, 3, 4, 5, 0, 0]
        assert data_proto.batch['position_ids'][0].tolist() == expected_position_ids

        # Verify token_level_rewards: [0, 0, 1.0, 0, 0]
        expected_rewards = [0.0, 0.0, 1.0, 0.0, 0.0]
        assert data_proto.batch['token_level_rewards'][0].tolist() == expected_rewards

        # Verify token_level_scores: 文档说"暂与 rewards 相同"
        assert data_proto.batch['token_level_scores'][0].tolist() == expected_rewards

    def test_converter_tensor_dtype(self, converter, sample_trajectories):
        """验证 tensor 数据类型"""
        # Execute
        data_proto = converter.convert(sample_trajectories)

        # Verify dtypes
        assert data_proto.batch['input_ids'].dtype == torch.long
        assert data_proto.batch['attention_mask'].dtype == torch.long
        assert data_proto.batch['position_ids'].dtype == torch.long
        assert data_proto.batch['prompts'].dtype == torch.long
        assert data_proto.batch['responses'].dtype == torch.long
        assert data_proto.batch['response_mask'].dtype == torch.long
        assert data_proto.batch['token_level_rewards'].dtype == torch.float32
        assert data_proto.batch['token_level_scores'].dtype == torch.float32

    def test_converter_tensor_device(self, converter, sample_trajectories):
        """验证 tensor 在 CPU 上"""
        # Execute
        data_proto = converter.convert(sample_trajectories)

        # Verify all tensors on CPU
        for key in data_proto.batch.keys():
            assert data_proto.batch[key].device.type == 'cpu'


# ============ 空 Response 边界测试 ============

class TestVerlConverterEmptyResponse:
    """空 Response 边界测试

    文档 Corner Case：response_length = 0 时跳过，不放置 reward。
    注意：VAEEHandler 会拦截空 response_ids，但 VerlConverter 应独立处理此边界。
    """

    def test_converter_zero_length_response(self, converter):
        """测试空 response 时 reward 不放置"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test_zero_response',
                prompt_ids=[1, 2, 3],
                response_ids=[],  # 空 response
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: response 全为 PAD
        assert data_proto.batch['responses'][0].tolist() == [0, 0, 0, 0, 0]
        # Verify: response_mask 全为 0
        assert data_proto.batch['response_mask'][0].tolist() == [0, 0, 0, 0, 0]
        # Verify: reward tensor 全为 0（因为 length = 0）
        assert data_proto.batch['token_level_rewards'][0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_converter_zero_length_response_batch(self, converter):
        """测试批量中包含空 response"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test_normal',
                prompt_ids=[1],
                response_ids=[2, 3],
                traj_reward=1.0,
            ),
            Trajectory(
                trajectory_id='test_zero_response',
                prompt_ids=[4],
                response_ids=[],  # 空 response
                traj_reward=0.5,
            ),
            Trajectory(
                trajectory_id='test_normal_2',
                prompt_ids=[5],
                response_ids=[6],
                traj_reward=0.3,
            ),
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify batch size
        assert len(data_proto) == 3

        # Verify: 第一个正常
        assert data_proto.batch['token_level_rewards'][0].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
        # Verify: 第二个空 response，reward 不放置
        assert data_proto.batch['token_level_rewards'][1].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
        # Verify: 第三个正常
        assert data_proto.batch['token_level_rewards'][2].tolist() == [0.3, 0.0, 0.0, 0.0, 0.0]


# ============ Non-Tensor 字段部分缺失测试 ============

class TestVerlConverterPartialMetadata:
    """Non-Tensor 字段部分缺失测试

    文档使用 all() 判断，当部分 trajectory 缺失字段时，该字段不应出现在 non_tensor_batch 中。
    """

    def test_converter_partial_ground_truth(self, converter):
        """测试部分 trajectory 缺失 ground_truth"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test_1',
                prompt_ids=[1],
                response_ids=[2],
                traj_reward=1.0,
                metadata={'ground_truth': 'answer_1'},
            ),
            Trajectory(
                trajectory_id='test_2',
                prompt_ids=[3],
                response_ids=[4],
                traj_reward=0.5,
                metadata={},  # 无 ground_truth
            ),
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: ground_truth 不应出现在 non_tensor_batch（因为 not all）
        assert 'ground_truth' not in data_proto.non_tensor_batch

        # Verify: uid 仍然存在
        assert 'uid' in data_proto.non_tensor_batch
        assert data_proto.non_tensor_batch['uid'][0] == 'test_1'
        assert data_proto.non_tensor_batch['uid'][1] == 'test_2'

    def test_converter_partial_data_source(self, converter):
        """测试部分 trajectory 缺失 data_source"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test_1',
                prompt_ids=[1],
                response_ids=[2],
                traj_reward=1.0,
                metadata={'data_source': 'gsm8k'},
            ),
            Trajectory(
                trajectory_id='test_2',
                prompt_ids=[3],
                response_ids=[4],
                traj_reward=0.5,
                metadata={'ground_truth': 'answer'},  # 有其他字段但无 data_source
            ),
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: data_source 不应出现在 non_tensor_batch
        assert 'data_source' not in data_proto.non_tensor_batch

    def test_converter_no_metadata_fields(self, converter):
        """测试所有 trajectory 都无 ground_truth/data_source"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test_1',
                prompt_ids=[1],
                response_ids=[2],
                traj_reward=1.0,
                metadata={},
            ),
            Trajectory(
                trajectory_id='test_2',
                prompt_ids=[3],
                response_ids=[4],
                traj_reward=0.5,
                metadata={},
            ),
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: 只有 uid
        assert 'uid' in data_proto.non_tensor_batch
        assert 'ground_truth' not in data_proto.non_tensor_batch
        assert 'data_source' not in data_proto.non_tensor_batch

    def test_converter_all_have_metadata_fields(self, converter):
        """测试所有 trajectory 都有 ground_truth/data_source"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test_1',
                prompt_ids=[1],
                response_ids=[2],
                traj_reward=1.0,
                metadata={'ground_truth': 'answer_1', 'data_source': 'gsm8k'},
            ),
            Trajectory(
                trajectory_id='test_2',
                prompt_ids=[3],
                response_ids=[4],
                traj_reward=0.5,
                metadata={'ground_truth': 'answer_2', 'data_source': 'math'},
            ),
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: 所有字段都存在
        assert 'uid' in data_proto.non_tensor_batch
        assert 'ground_truth' in data_proto.non_tensor_batch
        assert 'data_source' in data_proto.non_tensor_batch
        assert data_proto.non_tensor_batch['ground_truth'][0] == 'answer_1'
        assert data_proto.non_tensor_batch['ground_truth'][1] == 'answer_2'
        assert data_proto.non_tensor_batch['data_source'][0] == 'gsm8k'
        assert data_proto.non_tensor_batch['data_source'][1] == 'math'


# ============ 截断后 mask/position 验证测试 ============

class TestVerlConverterTruncationMask:
    """截断后的 attention_mask 和 position_ids 验证"""

    def test_converter_truncation_attention_mask(self, converter):
        """测试截断后的 attention_mask 正确性"""
        # Setup - prompt 超长
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8],  # 被截断为 [4,5,6,7,8]
                response_ids=[9, 10, 11],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: 截断后 prompt 无填充，全有效
        # prompt: [4, 5, 6, 7, 8] -> mask: [1, 1, 1, 1, 1]
        # response: [9, 10, 11, PAD, PAD] -> mask: [1, 1, 1, 0, 0]
        expected_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        assert data_proto.batch['attention_mask'][0].tolist() == expected_mask

    def test_converter_truncation_position_ids(self, converter):
        """测试截断后的 position_ids 正确性"""
        # Setup - response 超长
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6, 7, 8, 9, 10],  # 被截断为 [4,5,6,7,8]
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: position_ids 正确计算
        # prompt: [PAD, PAD, 1, 2, 3] -> mask: [0, 0, 1, 1, 1]
        # response: [4, 5, 6, 7, 8] -> mask: [1, 1, 1, 1, 1]
        # attention_mask: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        # position_ids: [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
        expected_position_ids = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
        assert data_proto.batch['position_ids'][0].tolist() == expected_position_ids

    def test_converter_both_truncation_mask_position(self, converter):
        """测试 prompt 和 response 都截断后的 mask 和 position"""
        # Setup
        trajectories = [
            Trajectory(
                trajectory_id='test',
                prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                response_ids=[9, 10, 11, 12, 13, 14, 15],
                traj_reward=1.0,
            )
        ]

        # Execute
        data_proto = converter.convert(trajectories)

        # Verify: 截断后无填充
        # prompt: [4, 5, 6, 7, 8] -> mask: [1, 1, 1, 1, 1]
        # response: [9, 10, 11, 12, 13] -> mask: [1, 1, 1, 1, 1]
        expected_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert data_proto.batch['attention_mask'][0].tolist() == expected_mask

        # position_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected_position_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert data_proto.batch['position_ids'][0].tolist() == expected_position_ids

        # reward 在截断后的最后一个有效 token (index 9)
        expected_rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        assert data_proto.batch['token_level_rewards'][0].tolist() == expected_rewards
