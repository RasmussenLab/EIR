from unittest.mock import patch, MagicMock

import pytest
from torch import nn
from torch.optim import SGD
from torch.optim.adamw import AdamW
from torch.utils.data import WeightedRandomSampler, SequentialSampler, RandomSampler

import eir.data_load.data_augmentation
import eir.data_load.data_utils
import eir.models.model_setup
import eir.models.omics.omics_models
import eir.setup.config
import eir.setup.input_setup
import eir.setup.output_setup
import eir.train
from eir import train
from eir.models import MetaModel
from eir.models.model_setup import get_default_model_registry_per_input_type
from eir.models.omics.models_cnn import CNNModel
from eir.models.omics.models_linear import LinearModel
from eir.setup.config import Configs
from eir.setup.output_setup import set_up_outputs_for_training
from eir.setup.schemas import GlobalConfig
from eir.train_utils import optimizers


@patch("eir.train.utils.get_run_folder", autospec=True)
def test_prepare_run_folder_pass(patched_get_run_folder, tmp_path):

    # patch since we don't want to create run folders while testing
    patched_get_run_folder.return_value = tmp_path / "test_folder"
    train._prepare_run_folder("test_folder")

    assert (tmp_path / "test_folder").exists()


@patch("eir.train.utils.get_run_folder", autospec=True)
def test_prepare_run_folder_fail(patched_get_run_folder, tmp_path):

    patched_path = tmp_path / "test_folder"
    patched_get_run_folder.return_value = patched_path
    patched_path.mkdir()

    fake_file = patched_path / "train_average_history.log"
    fake_file.write_text("Disco Elysium")

    with pytest.raises(FileExistsError):
        train._prepare_run_folder("test_folder")


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Linear
        {
            "injections": {
                "global_configs": {
                    "lr": 1e-03,
                    "output_folder": "test_get_default_experiment",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_get_default_experiment(
    create_test_config: Configs,
):
    test_configs = create_test_config

    default_experiment = train.get_default_experiment(configs=test_configs, hooks=None)
    assert default_experiment.configs == test_configs

    assert default_experiment.hooks is None

    assert len(default_experiment.criteria) == 1
    assert isinstance(
        default_experiment.criteria["test_output"]["Origin"], nn.CrossEntropyLoss
    )

    assert len(default_experiment.inputs) == 1
    assert set(default_experiment.inputs.keys()) == {"test_genotype"}

    assert len(default_experiment.outputs) == 1
    assert default_experiment.outputs["test_output"].target_columns == {
        "cat": ["Origin"],
        "con": [],
    }


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Linear
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_get_train_sampler(create_test_data, create_test_datasets, create_test_config):

    test_config = create_test_config
    gc = test_config.global_config

    train_dataset, *_ = create_test_datasets
    gc.weighted_sampling_columns = ["test_output.Origin"]

    test_sampler = eir.data_load.data_utils.get_train_sampler(
        columns_to_sample=gc.weighted_sampling_columns, train_dataset=train_dataset
    )
    assert isinstance(test_sampler, WeightedRandomSampler)

    gc.weighted_sampling_columns = None
    test_sampler = eir.data_load.data_utils.get_train_sampler(
        columns_to_sample=gc.weighted_sampling_columns, train_dataset=train_dataset
    )
    assert test_sampler is None


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Linear
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_get_dataloaders(
    create_test_config: Configs, create_test_data, create_test_datasets
):
    test_config = create_test_config
    gc = test_config.global_config
    gc.weighted_sampling_columns = ["test_output.Origin"]

    train_dataset, valid_dataset = create_test_datasets
    train_sampler = eir.data_load.data_utils.get_train_sampler(
        columns_to_sample=gc.weighted_sampling_columns, train_dataset=train_dataset
    )

    train_dataloader, valid_dataloader = train.get_dataloaders(
        train_dataset, train_sampler, valid_dataset, gc.batch_size
    )

    assert train_dataloader.batch_size == gc.batch_size
    assert valid_dataloader.batch_size == gc.batch_size
    assert isinstance(train_dataloader.sampler, WeightedRandomSampler)
    assert isinstance(valid_dataloader.sampler, SequentialSampler)

    train_dataloader, valid_dataloader = train.get_dataloaders(
        train_dataset, None, valid_dataset, gc.batch_size
    )

    assert isinstance(train_dataloader.sampler, RandomSampler)
    assert isinstance(valid_dataloader.sampler, SequentialSampler)


def test_get_optimizer():
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(10, 10)

        def forward(self, x):
            pass

    model = FakeModel()

    gc_adamw = GlobalConfig(output_folder="test", optimizer="adamw")

    adamw_optimizer = optimizers.get_optimizer(
        model=model, loss_callable=lambda x: x, global_config=gc_adamw
    )
    assert isinstance(adamw_optimizer, AdamW)

    gc_sgdm = GlobalConfig(output_folder="test", optimizer="sgdm")
    sgdm_optimizer = optimizers.get_optimizer(
        model=model, loss_callable=lambda x: x, global_config=gc_sgdm
    )
    assert isinstance(sgdm_optimizer, SGD)
    assert sgdm_optimizer.param_groups[0]["momentum"] == 0.9


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "cnn"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    }
                ],
            }
        },
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    }
                ],
            }
        },
    ],
    indirect=True,
)
def test_get_model(create_test_config: Configs, create_test_labels):
    test_config = create_test_config
    gc = create_test_config.global_config
    target_labels = create_test_labels

    inputs_as_dict = eir.setup.input_setup.set_up_inputs_for_training(
        inputs_configs=create_test_config.input_configs,
        train_ids=tuple(create_test_labels.train_labels.keys()),
        valid_ids=tuple(create_test_labels.valid_labels.keys()),
        hooks=None,
    )

    default_registry = get_default_model_registry_per_input_type()

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        target_transformers=target_labels.label_transformers,
    )

    model = eir.models.model_setup.get_model(
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        fusion_config=test_config.fusion_config,
        global_config=gc,
        model_registry_per_input_type=default_registry,
    )

    assert len(test_config.input_configs) == 1
    omics_model_type = test_config.input_configs[0].model_config.model_type
    _check_model(model_type=omics_model_type, model=model)


def _check_model(model_type: str, model: nn.Module):

    if model_type == "cnn":
        assert isinstance(model, MetaModel)
        output_module = model.output_modules.test_output
        origin_module = output_module.multi_task_branches["Origin"]
        assert origin_module[-1][-1].out_features == 3

        height_module = output_module.multi_task_branches["Height"]
        assert height_module[-1][-1].out_features == 1

        assert isinstance(model.input_modules["test_genotype"], CNNModel)

    elif model_type == "linear":
        assert isinstance(model.input_modules["test_genotype"], LinearModel)


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "cnn"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    }
                ],
            }
        },
    ],
    indirect=True,
)
def test_get_criteria(
    create_test_config: Configs, create_test_labels: train.MergedTargetLabels
):
    target_labels = create_test_labels

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        target_transformers=target_labels.label_transformers,
    )

    test_criteria = train._get_criteria(outputs_as_dict=outputs_as_dict)
    for output_name, output_object in outputs_as_dict.items():
        for column_name in output_object.target_columns["con"]:
            assert test_criteria[output_name][column_name].func is train._calc_mse

        for column_name in output_object.target_columns["cat"]:
            assert isinstance(
                test_criteria[output_name][column_name], nn.CrossEntropyLoss
            )


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "binary",
            "modalities": ("omics",),
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Default optimizer setup
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "linear",
                        },
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
        # With gradient accumulation
        {
            "injections": {
                "global_configs": {"gradient_accumulation_steps": 4},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "linear",
                        },
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_hook_default_optimizer_backward(prep_modelling_test_configs):

    experiment, *_ = prep_modelling_test_configs

    experiment.__dict__["optimizer"] = MagicMock()

    state = {"iteration": 1, "loss": MagicMock()}
    num_test_steps = 12

    for i in range(num_test_steps):
        train.hook_default_optimizer_backward(experiment=experiment, state=state)
        state["iteration"] += 1

    grad_acc_steps = experiment.configs.global_config.gradient_accumulation_steps

    if grad_acc_steps:
        assert state["loss"].__truediv__.call_count == num_test_steps
        assert experiment.optimizer.step.call_count == num_test_steps // grad_acc_steps
    else:
        assert state["loss"].backward.call_count == num_test_steps
        assert experiment.optimizer.step.call_count == num_test_steps
