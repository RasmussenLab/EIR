from pathlib import Path
from typing import Sequence, Literal, Optional, Dict, Callable

import torch.backends
import torch.cuda

from eir.setup.config import recursive_dict_replace


def get_test_base_global_init(
    allow_cuda: bool = True, allow_mps: bool = False
) -> Sequence[dict]:
    device = "cpu"
    if allow_cuda:
        device = "cuda" if torch.cuda.is_available() else device
    if allow_mps:
        device = "mps" if torch.backends.mps.is_available() else device

    global_inits = [
        {
            "output_folder": "runs/test_run",
            "plot_skip_steps": 0,
            "device": device,
            "compute_attributions": True,
            "attributions_every_sample_factor": 0,
            "attribution_background_samples": 64,
            "n_epochs": 12,
            "warmup_steps": 100,
            "lr": 2e-03,
            "optimizer": "adabelief",
            "lr_lb": 1e-05,
            "batch_size": 32,
            "valid_size": 0.05,
            "wd": 1e-03,
        }
    ]
    return global_inits


def get_test_inputs_inits(
    test_path: Path,
    input_config_dicts: Sequence[dict],
    split_to_test: bool,
    source: Literal["local", "deeplake"],
    extra_kwargs: Optional[dict] = None,
) -> Sequence[dict]:
    if extra_kwargs is None:
        extra_kwargs = {}

    inits = []

    base_func_map = get_input_test_init_base_func_map()
    for init_dict in input_config_dicts:
        cur_name = init_dict["input_info"]["input_name"]

        cur_base_func_keys = [i for i in base_func_map.keys() if cur_name.startswith(i)]
        assert len(cur_base_func_keys) == 1
        cur_base_func_key = cur_base_func_keys[0]

        cur_base_func = base_func_map.get(cur_base_func_key)
        cur_init_base = cur_base_func(
            init_dict=init_dict,
            test_path=test_path,
            split_to_test=split_to_test,
            source=source,
            extra_kwargs=extra_kwargs,
        )

        cur_init_injected = recursive_dict_replace(
            dict_=cur_init_base, dict_to_inject=init_dict
        )
        inits.append(cur_init_injected)

    return inits


def get_test_outputs_inits(
    test_path: Path, output_configs_dicts: Sequence[dict], split_to_test: bool
) -> Sequence[dict]:
    inits = []

    base_func_map = get_output_test_init_base_func_map()

    for init_dict in output_configs_dicts:
        cur_name = init_dict["output_info"]["output_name"]

        cur_base_func_keys = [i for i in base_func_map.keys() if cur_name == i]
        assert len(cur_base_func_keys) == 1
        cur_base_func_key = cur_base_func_keys[0]

        cur_base_func = base_func_map.get(cur_base_func_key)
        cur_init_base = cur_base_func(test_path=test_path, split_to_test=split_to_test)

        cur_init_injected = recursive_dict_replace(
            dict_=cur_init_base, dict_to_inject=init_dict
        )
        inits.append(cur_init_injected)

    return inits


def get_output_test_init_base_func_map() -> Dict[str, Callable]:
    mapping = {
        "test_output": get_test_base_output_inits,
        "test_output_copy": get_test_base_output_inits,
    }

    return mapping


def get_input_test_init_base_func_map() -> Dict[str, Callable]:
    mapping = {
        "test_genotype": get_test_omics_input_init,
        "test_tabular": get_test_tabular_input_init,
        "test_sequence": get_test_sequence_input_init,
        "test_bytes": get_test_bytes_input_init,
        "test_image": get_test_image_input_init,
        "test_array": get_test_array_input_init,
    }

    return mapping


def _inject_train_source_path(
    test_path: Path,
    source: Literal["local", "deeplake"],
    local_name: Literal["omics", "sequence", "image", "array"],
    split_to_test: bool,
) -> Path:
    if source == "local":
        input_source = test_path / local_name

        if split_to_test:
            input_source = input_source / "train_set"

    elif source == "deeplake":
        input_source = test_path / "deeplake"
        if split_to_test:
            input_source = test_path / "deeplake_train_set"

    else:
        raise ValueError(f"Source {source} not supported.")

    return input_source


def get_test_omics_input_init(
    test_path: Path,
    split_to_test: bool,
    init_dict: Dict,
    source: Literal["local", "deeplake"],
    *args,
    **kwargs,
) -> dict:
    input_source = _inject_train_source_path(
        test_path=test_path,
        source=source,
        local_name="omics",
        split_to_test=split_to_test,
    )

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_genotype",
            "input_type": "omics",
            "input_inner_key": "test_genotype",
        },
        "input_type_info": {
            "na_augment_perc": 0.10,
            "na_augment_prob": 0.80,
            "snp_file": str(test_path / "test_snps.bim"),
        },
        "model_config": {"model_type": "genome-local-net"},
    }

    if init_dict.get("input_type_info", {}).get("subset_snps_file", None) == "auto":
        subset_path = str(test_path / "test_subset_snps.txt")
        init_dict["input_type_info"]["subset_snps_file"] = subset_path

    return input_init_kwargs


def get_test_tabular_input_init(
    test_path: Path, split_to_test: bool, *args, **kwargs
) -> dict:
    input_source = test_path / "labels.csv"
    if split_to_test:
        input_source = test_path / "labels_train.csv"

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_tabular",
            "input_type": "tabular",
        },
        "input_type_info": {},
        "model_config": {"model_type": "tabular"},
    }

    return input_init_kwargs


def get_test_sequence_input_init(
    test_path: Path,
    split_to_test: bool,
    source: Literal["local", "deeplake"],
    extra_kwargs: dict,
    *args,
    **kwargs,
) -> dict:
    if extra_kwargs.get("sequence_csv_source", False):
        assert source == "local"
        name = "sequence.csv"
        if split_to_test:
            name = "sequence_train.csv"
        input_source = test_path / name
    else:
        input_source = _inject_train_source_path(
            test_path=test_path,
            source=source,
            local_name="sequence",
            split_to_test=split_to_test,
        )

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_sequence",
            "input_type": "sequence",
            "input_inner_key": "test_sequence",
        },
        "input_type_info": {
            "max_length": "max",
            "tokenizer_language": "en",
        },
        "model_config": {
            "model_type": "sequence-default",
            "embedding_dim": 64,
            "model_init_config": {
                "num_heads": 2,
                "num_layers": 1,
                "dropout": 0.10,
            },
        },
    }

    return input_init_kwargs


def get_test_bytes_input_init(
    test_path: Path, split_to_test: bool, *args, **kwargs
) -> Dict:
    input_source = test_path / "sequence"
    if split_to_test:
        input_source = input_source / "train_set"

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_sequence",
            "input_type": "bytes",
        },
        "input_type_info": {
            "max_length": 128,
        },
        "model_config": {
            "model_type": "sequence-default",
            "embedding_dim": 8,
            "window_size": 64,
        },
    }

    return input_init_kwargs


def get_test_image_input_init(
    test_path: Path,
    split_to_test: bool,
    source: Literal["local", "deeplake"],
    *args,
    **kwargs,
) -> Dict:
    input_source = _inject_train_source_path(
        test_path=test_path,
        source=source,
        local_name="image",
        split_to_test=split_to_test,
    )

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_image",
            "input_type": "image",
            "input_inner_key": "test_image",
        },
        "input_type_info": {
            "auto_augment": False,
            "size": (16,),
        },
        "model_config": {
            "model_type": "ResNet",
            "pretrained_model": False,
            "num_output_features": 128,
            "freeze_pretrained_model": False,
            "model_init_config": {
                "layers": [1, 1, 1, 1],
                "block": "BasicBlock",
            },
        },
    }

    return input_init_kwargs


def get_test_array_input_init(
    test_path: Path,
    split_to_test: bool,
    source: Literal["local", "deeplake"],
    *args,
    **kwargs,
) -> dict:
    input_source = _inject_train_source_path(
        test_path=test_path,
        source=source,
        local_name="array",
        split_to_test=split_to_test,
    )

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_array",
            "input_type": "array",
            "input_inner_key": "test_array",
        },
        "model_config": {"model_type": "cnn"},
    }

    return input_init_kwargs


def get_test_base_fusion_init(model_type: str) -> Sequence[dict]:
    if model_type == "identity":
        return [{}]
    elif model_type in ("default", "mgmoe"):
        return [
            {
                "model_config": {
                    "rb_do": 0.1,
                    "fc_do": 0.1,
                    "layers": [1],
                    "fc_task_dim": 128,
                }
            }
        ]
    else:
        raise ValueError()


def get_test_base_output_inits(test_path: Path, split_to_test: bool) -> Dict:
    label_file = test_path / "labels.csv"
    if split_to_test:
        label_file = test_path / "labels_train.csv"

    test_target_init_kwargs = {
        "output_info": {
            "output_name": "test_output",
            "output_type": "tabular",
            "output_source": str(label_file),
        },
        "output_type_info": {
            "target_cat_columns": ["Origin"],
        },
        "model_config": {
            "model_init_config": {
                "layers": [1],
                "fc_task_dim": 128,
            }
        },
    }

    return test_target_init_kwargs