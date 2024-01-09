from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Sequence, Tuple

import pytest

from eir import train
from eir.setup.config import get_all_tabular_targets
from eir.setup.schemas import BasicPretrainedConfig
from eir.train_utils.step_logic import get_default_hooks
from tests.setup_tests.fixtures_create_configs import cleanup
from tests.setup_tests.fixtures_create_experiment import get_cur_modelling_test_config
from tests.test_modelling.test_modelling_utils import check_performance_result_wrapper

if TYPE_CHECKING:
    from tests.setup_tests.fixtures_create_experiment import ModelTestConfig


def _get_pre_trained_module_setup_parametrization() -> Dict:
    base = {
        "injections": {
            "global_configs": {
                "output_folder": "multi_task_multi_modal",
                "n_epochs": 1,
                "attribution_background_samples": 8,
                "sample_interval": 50,
                "checkpoint_interval": 50,
                "n_saved_models": 2,
                "compute_attributions": False,
            },
            "input_configs": [
                {
                    "input_info": {"input_name": "test_genotype"},
                    "model_config": {
                        "model_type": "cnn",
                        "model_init_config": {"l1": 1e-03},
                    },
                },
                {
                    "input_info": {"input_name": "test_sequence"},
                },
                {
                    "input_info": {"input_name": "test_bytes"},
                },
                {
                    "input_info": {"input_name": "test_image"},
                    "model_config": {
                        "model_init_config": {
                            "layers": [2],
                            "kernel_width": 2,
                            "kernel_height": 2,
                            "down_stride_width": 2,
                            "down_stride_height": 2,
                        },
                    },
                },
                {
                    "input_info": {"input_name": "test_tabular"},
                    "input_type_info": {
                        "input_cat_columns": ["OriginExtraCol"],
                        "input_con_columns": ["ExtraTarget"],
                    },
                    "model_config": {
                        "model_type": "tabular",
                        "model_init_config": {"l1": 1e-03},
                    },
                },
            ],
            "fusion_configs": {
                "model_type": "mlp-residual",
                "model_config": {
                    "fc_task_dim": 128,
                    "fc_do": 0.10,
                    "rb_do": 0.10,
                },
            },
            "output_configs": [
                {
                    "output_info": {"output_name": "test_output_tabular"},
                    "output_type_info": {
                        "target_cat_columns": ["Origin"],
                        "target_con_columns": ["Height"],
                    },
                },
                {
                    "output_info": {"output_name": "test_output_sequence"},
                },
                {
                    "output_info": {
                        "output_name": "test_output_array_cnn",
                    },
                    "model_config": {
                        "model_type": "cnn",
                        "model_init_config": {
                            "channel_exp_base": 3,
                            "allow_pooling": False,
                        },
                    },
                },
            ],
        },
    }
    return base


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi_task",
            "modalities": (
                "omics",
                "sequence",
                "image",
                "array",
            ),
            "extras": {"array_dims": 1},
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task",
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        _get_pre_trained_module_setup_parametrization(),
    ],
    indirect=True,
)
def test_pre_trained_module_setup(
    prep_modelling_test_configs: Tuple[train.Experiment, "ModelTestConfig"],
):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    _get_experiment_overloaded_for_pretrained_extractor(
        experiment=experiment,
        test_config=test_config,
        rename_pretrained_inputs=False,
    )

    _get_experiment_overloaded_for_pretrained_extractor(
        experiment=experiment,
        test_config=test_config,
        rename_pretrained_inputs=True,
    )

    _get_experiment_overloaded_for_pretrained_checkpoint(
        experiment=experiment,
        test_config=test_config,
    )

    _get_experiment_overloaded_for_pretrained_checkpoint(
        experiment=experiment,
        test_config=test_config,
        change_architecture=True,
    )

    experiment_overwritten = _add_new_feature_extractor_to_experiment(
        experiment=experiment
    )
    _get_experiment_overloaded_for_pretrained_extractor(
        experiment=experiment_overwritten,
        test_config=test_config,
        rename_pretrained_inputs=False,
    )

    _get_experiment_overloaded_for_pretrained_extractor(
        experiment=experiment_overwritten,
        test_config=test_config,
        rename_pretrained_inputs=True,
    )


def _add_new_feature_extractor_to_experiment(
    experiment: train.Experiment,
) -> train.Experiment:
    """
    Used to check that we can do a partial loading of pretrained feature extractors.
    """

    experiment_attrs = experiment.__dict__
    experiment_configs = deepcopy(experiment.configs)
    first_config_modified = deepcopy(experiment_configs.input_configs[0])

    extra_input_name = first_config_modified.input_info.input_name + "_copy"
    first_config_modified.input_info.input_name = extra_input_name

    inputs_configs = list(experiment_configs.input_configs)
    inputs_configs_with_extra = inputs_configs + [first_config_modified]
    experiment_attrs["configs"].input_configs = inputs_configs_with_extra

    experiment_overwritten = train.Experiment(**experiment_attrs)

    return experiment_overwritten


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi_task",
            "modalities": (
                "omics",
                "sequence",
                "image",
                "array",
            ),
            "extras": {"array_dims": 1},
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task",
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "output_folder": "multi_task_multi_modal",
                    "n_epochs": 3,
                    "attribution_background_samples": 8,
                    "sample_interval": 50,
                    "checkpoint_interval": 50,
                    "n_saved_models": 2,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_bytes"},
                    },
                    {
                        "input_info": {"input_name": "test_image"},
                        "model_config": {
                            "model_init_config": {
                                "layers": [2],
                                "kernel_width": 2,
                                "kernel_height": 2,
                                "down_stride_width": 2,
                                "down_stride_height": 2,
                            },
                        },
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {
                            "model_type": "tabular",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 128,
                        "fc_do": 0.10,
                        "rb_do": 0.10,
                    },
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    },
                    {
                        "output_info": {"output_name": "test_output_sequence"},
                    },
                    {
                        "output_info": {
                            "output_name": "test_output_array_cnn",
                        },
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "channel_exp_base": 3,
                                "allow_pooling": False,
                            },
                        },
                    },
                ],
            },
        }
    ],
    indirect=True,
)
def test_pre_training_and_loading(
    prep_modelling_test_configs: Tuple[train.Experiment, "ModelTestConfig"],
):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    (
        pretrained_experiment,
        pretrained_test_config,
    ) = _get_experiment_overloaded_for_pretrained_extractor(
        experiment=experiment,
        test_config=test_config,
        rename_pretrained_inputs=True,
    )

    train.train(experiment=pretrained_experiment)

    # Note we skip checking R2 for now as we patch the metrics in conftest.py
    # to check for both training and validation, but for now we will make do with
    # checking only the MCC for this
    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        max_thresholds=(0.5, 0.5),
        min_thresholds=(3.0, 3.0),
        con_metric=None,
    )

    (
        pretrained_checkpoint_experiment,
        pretrained_checkpoint_test_config,
    ) = _get_experiment_overloaded_for_pretrained_checkpoint(
        experiment=experiment, test_config=test_config
    )

    train.train(experiment=pretrained_checkpoint_experiment)

    check_performance_result_wrapper(
        outputs=pretrained_checkpoint_experiment.outputs,
        run_path=pretrained_checkpoint_test_config.run_path,
        max_thresholds=(0.85, 0.85),
        min_thresholds=(2.0, 2.0),
        con_metric=None,
    )


def _get_experiment_overloaded_for_pretrained_extractor(
    experiment: train.Experiment,
    test_config: "ModelTestConfig",
    rename_pretrained_inputs: bool,
    skip_pretrained_keys: Sequence[str] = tuple(),
) -> Tuple[train.Experiment, "ModelTestConfig"]:
    pretrained_configs = _get_input_configs_with_pretrained_modifications(
        run_path=test_config.run_path,
        pretrained_configs=experiment.configs,
        skip_pretrained_keys=skip_pretrained_keys,
        rename_pretrained_inputs=rename_pretrained_inputs,
    )

    pretrained_configs = _get_output_configs_with_pretrained_modifications(
        pretrained_configs=pretrained_configs,
        rename_pretrained_inputs=rename_pretrained_inputs,
    )

    pretrained_configs = _get_pretrained_config_with_modified_globals(
        pretrained_configs=pretrained_configs
    )

    run_path = Path(f"{pretrained_configs.global_config.output_folder}/")
    if run_path.exists():
        cleanup(run_path=run_path)

    default_hooks = get_default_hooks(configs=pretrained_configs)
    pretrained_experiment = train.get_default_experiment(
        configs=pretrained_configs, hooks=default_hooks
    )

    targets = get_all_tabular_targets(
        output_configs=pretrained_experiment.configs.output_configs
    )
    pretrained_test_config = get_cur_modelling_test_config(
        train_loader=pretrained_experiment.train_loader,
        global_config=pretrained_configs.global_config,
        tabular_targets=targets,
        output_configs=pretrained_experiment.configs.output_configs,
        input_names=pretrained_experiment.inputs.keys(),
    )

    return pretrained_experiment, pretrained_test_config


def _get_input_configs_with_pretrained_modifications(
    pretrained_configs: train.Configs,
    run_path: Path,
    rename_pretrained_inputs: bool,
    skip_pretrained_keys: Sequence[str] = tuple(),
) -> train.Configs:
    """
    `rename_pretrained_inputs`:
        To check that we can use the `load_module_name` argument, where we have
        a different name for the input module in the pretrained model than in the
        current model.
    """
    pretrained_configs = deepcopy(pretrained_configs)
    input_configs = pretrained_configs.input_configs
    saved_model_path = next((run_path / "saved_models").iterdir())

    input_configs_with_pretrained = []
    for cur_input_config in input_configs:
        cur_name = cur_input_config.input_info.input_name

        if cur_name in skip_pretrained_keys:
            continue

        cur_pretrained_config = BasicPretrainedConfig(
            model_path=str(saved_model_path), load_module_name=cur_name
        )
        cur_input_config.pretrained_config = cur_pretrained_config

        # Check that the names can differ
        if rename_pretrained_inputs:
            cur_input_config.input_info.input_name = cur_name + "_pretrained_module"

        input_configs_with_pretrained.append(cur_input_config)

    pretrained_configs.input_configs = input_configs_with_pretrained

    return pretrained_configs


def _get_output_configs_with_pretrained_modifications(
    pretrained_configs: train.Configs,
    rename_pretrained_inputs: bool,
) -> train.Configs:
    """
    This is needed because we are modifying the configs from a previous experiment
    directly, and we need to make sure that the output configs are also modified
    when because the assumption os that the input-output configs are in sync.
    """
    pretrained_configs = deepcopy(pretrained_configs)
    output_configs = pretrained_configs.output_configs

    output_configs_with_pretrained = []
    for output_config in output_configs:
        if output_config.output_info.output_type != "sequence":
            output_configs_with_pretrained.append(output_config)

        else:
            cur_name = output_config.output_info.output_name
            if rename_pretrained_inputs:
                output_config.output_info.output_name = cur_name + "_pretrained_module"

            output_configs_with_pretrained.append(output_config)

    pretrained_configs.output_configs = output_configs_with_pretrained

    return pretrained_configs


def _get_pretrained_config_with_modified_globals(
    pretrained_configs: train.Configs,
) -> train.Configs:
    pretrained_configs = deepcopy(pretrained_configs)

    pretrained_configs.global_config.n_epochs = 6
    pretrained_configs.global_config.sample_interval = 200
    pretrained_configs.global_config.checkpoint_interval = 200
    pretrained_configs.global_config.output_folder = (
        pretrained_configs.global_config.output_folder + "_with_pretrained"
    )

    return pretrained_configs


def _get_experiment_overloaded_for_pretrained_checkpoint(
    experiment: train.Experiment,
    test_config: "ModelTestConfig",
    change_architecture: bool = False,
) -> Tuple[train.Experiment, "ModelTestConfig"]:
    """
    :param change_architecture:
        If True, we will change the architecture of the model to be different from the
        original model. This is to test that we can partially load models with
        different shapes.
    """

    pretrained_configs = deepcopy(experiment.configs)
    saved_model_path = next((test_config.run_path / "saved_models").iterdir())

    pretrained_configs.global_config.n_epochs = 6
    pretrained_configs.global_config.pretrained_checkpoint = str(saved_model_path)
    pretrained_configs.global_config.sample_interval = 200
    pretrained_configs.global_config.checkpoint_interval = 200
    pretrained_configs.global_config.output_folder = (
        pretrained_configs.global_config.output_folder + "_with_pretrained_checkpoint"
    )

    if change_architecture:
        pretrained_configs.global_config.strict_pretrained_loading = False
        fus_task_dim = pretrained_configs.fusion_config.model_config.fc_task_dim
        pretrained_configs.fusion_config.model_config.fc_task_dim = fus_task_dim * 2

    run_path = Path(f"{pretrained_configs.global_config.output_folder}/")
    if run_path.exists():
        cleanup(run_path=run_path)

    default_hooks = get_default_hooks(configs=pretrained_configs)
    pretrained_experiment = train.get_default_experiment(
        configs=pretrained_configs, hooks=default_hooks
    )

    targets = get_all_tabular_targets(
        output_configs=pretrained_experiment.configs.output_configs
    )
    pretrained_test_config = get_cur_modelling_test_config(
        train_loader=pretrained_experiment.train_loader,
        global_config=pretrained_configs.global_config,
        output_configs=pretrained_experiment.configs.output_configs,
        tabular_targets=targets,
        input_names=pretrained_experiment.inputs.keys(),
    )

    return pretrained_experiment, pretrained_test_config
