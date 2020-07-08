from argparse import Namespace

import pytest
from human_origins_supervised.models import models, layers


def test_make_conv_layers():
    """
    Check that:
        - We have correct number of layers (+2 for first conv layer and
          self attn).
        - We start with correct block (first conv layer).
        - Self attention is the second last layer.

    """
    conv_layer_list = [1, 1, 1, 1]
    test_cl_args = Namespace(
        kernel_width=5,
        down_stride=4,
        target_width=int(8e5),
        rb_do=0.1,
        first_kernel_expansion=1,
        first_stride_expansion=5,
        first_channel_expansion=1,
        dilation_factor=1,
        channel_exp_base=5,
        sa=True,
    )
    conv_layers = models._make_conv_layers(conv_layer_list, test_cl_args)

    # account for first block, add +2 instead if using SA
    assert len(conv_layers) == len(conv_layer_list) + 2
    assert isinstance(conv_layers[0], layers.FirstBlock)
    assert isinstance(conv_layers[-2], layers.SelfAttention)


def get_test_module_dict_data():
    test_classes_dict = {"Origin": 10, "Height": 1, "BMI": 1}
    test_fc_in = 128

    return test_classes_dict, test_fc_in


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ({"dilation_factor": 2, "block_number": 1, "width": 1000}, 2),
        ({"dilation_factor": 4, "block_number": 2, "width": 1000}, 16),
        ({"dilation_factor": 5, "block_number": 5, "width": 100}, 25),
    ],
)
def test_get_cur_dilation(test_input, expected):
    test_dilation = models._get_cur_dilation(**test_input)

    assert test_dilation == expected


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "multi_task"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {  # Case 1: Check that we add and use extra inputs.
            "custom_cl_args": {
                "model_type": "cnn",
                "target_cat_columns": ["Origin"],
                "extra_con_columns": ["ExtraTarget"],
                "extra_cat_columns": ["OriginExtraCol"],
                "target_con_columns": ["Height"],
                "run_name": "extra_inputs",
            }
        }
    ],
    indirect=True,
)
def test_cnn_model(
    parse_test_cl_args, create_test_data, create_test_cl_args, create_test_model
):
    cnn_test_model = create_test_model

    assert isinstance(cnn_test_model.conv[0], models.FirstBlock)
    assert True


def test_mlp_model(parse_test_cl_args):
    # parse_test_cl_args to have access to input size for checking dimensions
    pass


def test_logistic_regression_model():
    pass


def test_linear_regression_model():
    pass
