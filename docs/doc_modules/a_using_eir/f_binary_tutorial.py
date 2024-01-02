import base64
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from docs.doc_modules.deploy_experiments_utils import copy_inputs
from docs.doc_modules.deployment_experiments import AutoDocDeploymentInfo
from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.utils import add_model_path_to_command


def get_06_imdb_binary_run_1_transformer_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/a_using_eir/06_raw_bytes_tutorial/"

    conf_output_path = "eir_tutorials/a_using_eir/06_raw_bytes_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input.yaml",
        "--output_configs",
        f"{conf_output_path}/output.yaml",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/06_training_curve_ACC_transformer_1.pdf",
        ),
        (
            "training_curve_MCC",
            "figures/06_training_curve_MCC_transformer_1.pdf",
        ),
    ]

    data_output_path = Path(
        "eir_tutorials/a_using_eir/06_raw_bytes_tutorial/data/imdb.zip"
    )

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_tutorials/a_using_eir/06_raw_bytes_tutorial/",
                "-L",
                "3",
                "-I",
                "*.zip",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="SEQUENCE_BINARY_IMDB_1",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_06_imdb_binary_deploy_transformer_info() -> AutoDocDeploymentInfo:
    base_path = "docs/tutorials/tutorial_files/a_using_eir/06_raw_bytes_tutorial"

    model_path_placeholder = "FILL_MODEL"

    server_command = ["eirdeploy", "--model-path", model_path_placeholder]

    base = "eir_tutorials/a_using_eir/03_sequence_tutorial/data/IMDB/IMDB_Reviews"
    example_requests = [
        {"imdb_reviews_bytes_base_transformer": f"{base}/10021_2.txt"},
        {"imdb_reviews_bytes_base_transformer": f"{base}/10132_9.txt"},
    ]

    add_model_path = partial(
        add_model_path_to_command,
        run_path="eir_tutorials/tutorial_runs/a_using_eir/"
        "tutorial_06_imdb_sentiment_binary",
    )

    copy_inputs_to_deploy = (
        copy_inputs,
        {
            "example_requests": example_requests,
            "output_folder": str(Path(base_path) / "deploy_results"),
        },
    )

    ade = AutoDocDeploymentInfo(
        name="BYTES_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(add_model_path,),
        post_run_functions=(copy_inputs_to_deploy,),
        example_requests=example_requests,
        data_loading_function=_load_data_for_binary_deploy,
    )

    return ade


def _load_data_for_binary_deploy(data: dict[str, Any]) -> dict[str, Any]:
    loaded_data = {}
    for key, data_pointer in data.items():
        arr = np.fromfile(data_pointer, dtype="uint8")
        arr_bytes = arr.tobytes()
        base_64_encoded = base64.b64encode(arr_bytes).decode("utf-8")
        loaded_data[key] = base_64_encoded
    return loaded_data


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_06_imdb_binary_run_1_transformer_info()
    exp_2 = get_06_imdb_binary_deploy_transformer_info()

    return [exp_1, exp_2]
