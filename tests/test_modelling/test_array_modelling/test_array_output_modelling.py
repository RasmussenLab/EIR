from copy import deepcopy
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pytest
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error

from eir import train
from eir.train_utils.utils import seed_everything
from tests.conftest import should_skip_in_gha_macos
from tests.test_modelling.test_modelling_utils import check_performance_result_wrapper

if TYPE_CHECKING:
    from tests.setup_tests.fixtures_create_experiment import (
        ModelTestConfig,
        al_modelling_test_configs,
    )

seed_everything(seed=0)


def _get_output_array_data_parameters() -> Sequence[dict]:
    base = {
        "task_type": "multi",
        "modalities": ("array",),
        "extras": {"array_dims": np.nan},
        "split_to_test": True,
        "source": "FILL",
    }

    parameters = []

    for dims in [1, 2, 3]:
        cur_base = deepcopy(base)
        for source in ["local", "deeplake"]:
            cur_base["source"] = source
            cur_base["extras"]["array_dims"] = dims
            parameters.append(cur_base)

    return parameters


@pytest.mark.skipif(condition=should_skip_in_gha_macos(), reason="In GHA.")
@pytest.mark.parametrize(
    "create_test_data",
    _get_output_array_data_parameters(),
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "output_folder": "test_array_generation",
                    "n_epochs": 15,
                    "memory_dataset": True,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_array"},
                        "input_type_info": {
                            "normalization": "channel",
                        },
                        "model_config": {
                            "model_type": "lcl",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 3,
                                "attention_inclusion_cutoff": 128,
                            },
                        },
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {
                            "output_name": "test_output_array_lcl",
                        },
                        "model_config": {
                            "model_type": "lcl",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 3,
                                "direction": "up",
                                "cutoff": "auto",
                            },
                        },
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
        },
    ],
    indirect=True,
)
def test_array_output_modelling(
    prep_modelling_test_configs: "al_modelling_test_configs",
) -> None:
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        max_thresholds=(0.5, 0.5),
        min_thresholds=(1.5, 1.5),
    )

    _array_output_test_check_wrapper(experiment=experiment, test_config=test_config)


def _array_output_test_check_wrapper(
    experiment: train.Experiment,
    test_config: "ModelTestConfig",
    mse_threshold: float = 0.2,
    cosine_similarity_threshold: float = 0.6,
) -> None:
    output_configs = experiment.configs.output_configs

    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        output_type = output_config.output_info.output_type

        if output_type != "array":
            continue

        latest_sample = test_config.last_sample_folders[output_name][output_name]
        auto_folder = latest_sample / "auto"

        for f in auto_folder.iterdir():
            if f.suffix != ".npy":
                continue

            generated_array = np.load(str(f))

            index = f.name.split("_")[0]
            matching_input_folder = auto_folder / f"{index}_inputs"
            matching_input_array_file = matching_input_folder / "test_array.npy"
            matching_input_array = np.load(str(matching_input_array_file))

            mse = mean_squared_error(
                y_true=matching_input_array.ravel(),
                y_pred=generated_array.ravel(),
            )
            assert mse < mse_threshold

            # due to deeplake arrays not storing 0s but as very small numbers
            matching_input_array[matching_input_array < 1e-8] = 0.0

            cosine_similarity = 1 - cosine(
                u=matching_input_array.ravel().astype(np.float32),
                v=generated_array.ravel(),
            )
            assert cosine_similarity > cosine_similarity_threshold
