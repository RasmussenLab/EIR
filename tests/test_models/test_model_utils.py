from copy import deepcopy

import pytest
import torch
from torch import nn

from eir.models import model_training_utils
from eir.models.omics import models_cnn


@pytest.fixture
def create_test_util_model():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc_1 = nn.Linear(10, 10, bias=True)
            self.act_1 = nn.PReLU()
            self.bn_1 = nn.BatchNorm1d(10)

            self.fc_2 = nn.Linear(10, 10, bias=True)
            self.act_2 = nn.PReLU()
            self.bn_2 = nn.BatchNorm1d(10)

        def forward(self, x):
            return x

    model = TestModel()

    return model


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            {
                "size_w": 1000,
                "stride_w": 4,
                "first_stride_expansion_w": 1,
                "size_h": 1000,
                "stride_h": 4,
                "first_stride_expansion_h": 1,
                "cutoff": 128,
            },
            [2, 1],
        ),
        (
            {
                "size_w": 10000,
                "stride_w": 4,
                "first_stride_expansion_w": 1,
                "size_h": 10000,
                "stride_h": 4,
                "first_stride_expansion_h": 1,
                "cutoff": 128,
            },
            [2, 2],
        ),
        (
            {
                "size_w": 1e6,
                "stride_w": 4,
                "first_stride_expansion_w": 1,
                "size_h": 1e6,
                "stride_h": 4,
                "first_stride_expansion_h": 1,
                "cutoff": 128,
            },
            [2, 2, 2, 2],
        ),
        (
            {
                "size_w": 64,
                "stride_w": 2,
                "first_stride_expansion_w": 1,
                "size_h": 64,
                "stride_h": 2,
                "first_stride_expansion_h": 1,
                "cutoff": 128,
            },
            [2],
        ),
        (
            {
                "size_w": 128,
                "stride_w": 2,
                "first_stride_expansion_w": 1,
                "size_h": 128,
                "stride_h": 2,
                "first_stride_expansion_h": 1,
                "cutoff": 128,
            },
            [2, 1],
        ),
        (
            {
                "size_w": 32,
                "stride_w": 2,
                "first_stride_expansion_w": 1,
                "size_h": 32,
                "stride_h": 2,
                "first_stride_expansion_h": 1,
                "cutoff": 128,
            },
            [1],
        ),
    ],
)
def test_find_no_residual_blocks_needed(test_input, expected):
    assert models_cnn.auto_find_no_cnn_residual_blocks_needed(**test_input) == expected


def test_get_model_params(create_test_util_model):
    test_model = create_test_util_model

    weight_decay = 0.05
    model_params = model_training_utils.add_wd_to_model_params(
        model=test_model, wd=weight_decay
    )

    # BN has weight and bias, hence 6 [w] + 2 [b] + 2 = 10 parameter groups
    assert len(model_params) == 10

    for param_group in model_params:
        if param_group["params"].shape[0] == 1:
            assert param_group["weight_decay"] == 0.00
        else:
            assert param_group["weight_decay"] == 0.05


def set_up_stack_list_of_tensors_dicts_data():
    test_batch_base = {
        "Target_Column_1": torch.ones((5, 5)),
        "Target_Column_2": torch.ones((5, 5)) * 2,
    }

    test_list_of_batches = [deepcopy(test_batch_base) for _ in range(3)]

    for i in range(3):
        test_list_of_batches[i]["Target_Column_1"] *= i
        test_list_of_batches[i]["Target_Column_2"] *= i

    return test_list_of_batches


def test_stack_list_of_tensor_dicts():
    test_input = set_up_stack_list_of_tensors_dicts_data()

    test_output = model_training_utils._stack_list_of_batch_dicts(test_input)

    assert (test_output["Target_Column_1"][0] == 0.0).all()
    assert (test_output["Target_Column_1"][5] == 1.0).all()
    assert (test_output["Target_Column_1"][10] == 2.0).all()

    assert (test_output["Target_Column_2"][0] == 0.0).all()
    assert (test_output["Target_Column_2"][5] == 2.0).all()
    assert (test_output["Target_Column_2"][10] == 4.0).all()


def set_up_stack_list_of_output_tensor_dicts_data():
    test_batch_base = {
        "test_output": {
            "Target_Column_1": torch.ones((5, 5)),
            "Target_Column_2": torch.ones((5, 5)) * 2,
        }
    }

    test_list_of_batches = [deepcopy(test_batch_base) for _ in range(3)]

    for i in range(3):
        test_list_of_batches[i]["test_output"]["Target_Column_1"] *= i
        test_list_of_batches[i]["test_output"]["Target_Column_2"] *= i

    return test_list_of_batches


def test_stack_list_of_output_target_dicts():
    test_input = set_up_stack_list_of_output_tensor_dicts_data()

    test_output = model_training_utils._stack_list_of_output_target_dicts(test_input)

    assert (test_output["test_output"]["Target_Column_1"][0] == 0.0).all()
    assert (test_output["test_output"]["Target_Column_1"][5] == 1.0).all()
    assert (test_output["test_output"]["Target_Column_1"][10] == 2.0).all()

    assert (test_output["test_output"]["Target_Column_2"][0] == 0.0).all()
    assert (test_output["test_output"]["Target_Column_2"][5] == 2.0).all()
    assert (test_output["test_output"]["Target_Column_2"][10] == 4.0).all()
