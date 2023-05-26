import argparse
import types
from argparse import Namespace
from collections import Counter
from copy import copy
from dataclasses import dataclass
from typing import (
    Sequence,
    Iterable,
    Dict,
    List,
    Literal,
    Union,
    Tuple,
    Type,
    Any,
    Callable,
    overload,
    Generator,
    Mapping,
)

import configargparse
import yaml
from aislib.misc_utils import get_logger

from eir.models.array.array_models import (
    ArrayModelConfig,
    get_array_config_dataclass_mapping,
)
from eir.models.fusion.fusion_identity import IdentityConfig
from eir.models.fusion.fusion_mgmoe import MGMoEModelConfig
from eir.models.image.image_models import ImageModelConfig
from eir.models.layers import ResidualMLPConfig
from eir.models.omics.omics_models import (
    get_omics_config_dataclass_mapping,
    OmicsModelConfig,
)
from eir.models.output.linear import LinearOutputModuleConfig
from eir.models.output.mlp_residual import (
    ResidualMLPOutputModulelConfig,
)
from eir.models.output.output_module_setup import (
    TabularOutputModuleConfig,
    SequenceOutputModuleConfig,
)
from eir.models.output.sequence.sequence_output_modules import (
    TransformerSequenceOutputModuleConfig,
)
from eir.models.sequence.transformer_models import (
    BasicTransformerFeatureExtractorModelConfig,
    SequenceModelConfig,
)
from eir.models.tabular.tabular import TabularModelConfig, SimpleTabularModelConfig
from eir.setup import schemas
from eir.setup.config_setup_modules.config_setup_utils import (
    get_yaml_iterator_with_injections,
)
from eir.setup.config_setup_modules.output_config_setup_sequence import (
    get_configs_object_with_seq_output_configs,
)
from eir.train_utils.utils import configure_global_eir_logging

al_input_types = Union[
    schemas.OmicsInputDataConfig,
    schemas.TabularInputDataConfig,
    schemas.SequenceInputDataConfig,
    schemas.ByteInputDataConfig,
]

al_output_types = Union[schemas.TabularOutputTypeConfig]

al_output_types_schema_map = Dict[
    str, Union[Type[schemas.TabularOutputTypeConfig], Type]
]

al_output_module_config_class_getter = (
    Callable[[str], Union[schemas.al_output_module_configs_classes, Any]],
)

al_output_model_configs = Union[
    ResidualMLPOutputModulelConfig, LinearOutputModuleConfig, Any
]
al_output_model_init_map = Dict[str, al_output_model_configs]

logger = get_logger(name=__name__)


@dataclass
class DynamicOutputSetup:
    output_types_schema_map: al_output_types_schema_map
    output_module_config_class_getter: al_output_module_config_class_getter
    output_module_init_class_map: al_output_model_init_map


def get_configs():
    main_cl_args, extra_cl_args = get_main_cl_args()

    output_folder, log_level = get_output_folder_and_log_level_from_cl_args(
        main_cl_args=main_cl_args, extra_cl_args=extra_cl_args
    )

    configure_global_eir_logging(output_folder=output_folder, log_level=log_level)

    dynamic_output_setup = DynamicOutputSetup(
        output_types_schema_map=get_outputs_types_schema_map(),
        output_module_config_class_getter=get_output_module_config_class,
        output_module_init_class_map=get_output_config_type_init_callable_map(),
    )
    configs = generate_aggregated_config(
        cl_args=main_cl_args,
        extra_cl_args_overload=extra_cl_args,
        dynamic_output_setup=dynamic_output_setup,
    )

    configs_with_seq_outputs = get_configs_object_with_seq_output_configs(
        configs=configs,
    )

    return configs_with_seq_outputs


def get_main_cl_args() -> Tuple[argparse.Namespace, List[str]]:
    parser_ = get_main_parser()
    cl_args, extra_cl_args = parser_.parse_known_args()

    return cl_args, extra_cl_args


def get_output_folder_and_log_level_from_cl_args(
    main_cl_args: argparse.Namespace,
    extra_cl_args: List[str],
) -> Tuple[str, str]:
    global_configs = main_cl_args.global_configs

    output_folder = None
    for config in global_configs:
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)

            output_folder = config_dict.get("output_folder")
            log_level = config_dict.get("log_level", "info")

            if output_folder and log_level:
                break

    for arg in extra_cl_args:
        if "output_folder=" in arg:
            output_folder = arg.split("=")[1]
            break

    if output_folder is None:
        raise ValueError("Output folder not found in global configs.")

    return output_folder, log_level


def get_main_parser(
    output_nargs: Literal["+", "*"] = "+"
) -> configargparse.ArgumentParser:
    parser_ = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    parser_.add_argument(
        "--global_configs",
        nargs="+",
        type=str,
        required=True,
        help="Global .yaml configurations for the experiment.",
    )

    parser_.add_argument(
        "--input_configs",
        type=str,
        nargs="*",
        required=False,
        default=[],
        help="Input feature extraction .yaml configurations. "
        "Each configuration represents one input.",
    )

    parser_.add_argument(
        "--fusion_configs",
        type=str,
        nargs="*",
        required=False,
        default=[],
        help="Fusion .yaml configurations.",
    )

    parser_.add_argument(
        "--output_configs",
        type=str,
        nargs=output_nargs,
        required=True,
        help="Output .yaml configurations.",
    )

    return parser_


def _recursive_search(
    dict_: Mapping, target: Any, path=None
) -> Generator[Tuple[str, Any], None, None]:
    if not path:
        path = []

    for key, value in dict_.items():
        local_path = copy(path)
        local_path.append(key)
        if isinstance(value, Mapping):
            yield from _recursive_search(dict_=value, target=target, path=local_path)
        else:
            if value == target:
                yield local_path, value


@dataclass
class Configs:
    global_config: schemas.GlobalConfig
    input_configs: Sequence[schemas.InputConfig]
    fusion_config: schemas.FusionConfig
    output_configs: Sequence[schemas.OutputConfig]


def generate_aggregated_config(
    cl_args: Union[argparse.Namespace, types.SimpleNamespace],
    dynamic_output_setup: DynamicOutputSetup,
    extra_cl_args_overload: Union[List[str], None] = None,
    strict: bool = True,
) -> Configs:
    global_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.global_configs, extra_cl_args=extra_cl_args_overload
    )
    global_config = get_global_config(global_configs=global_config_iter)

    inputs_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.input_configs,
        extra_cl_args=extra_cl_args_overload,
    )
    input_configs = get_input_configs(input_configs=inputs_config_iter)

    fusion_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.fusion_configs,
        extra_cl_args=extra_cl_args_overload,
    )
    fusion_config = load_fusion_configs(fusion_configs=fusion_config_iter)

    output_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.output_configs,
        extra_cl_args=extra_cl_args_overload,
    )
    output_configs = load_output_configs(
        output_configs=output_config_iter,
        dynamic_output_setup=dynamic_output_setup,
    )

    if strict:
        _check_input_and_output_config_names(
            input_configs=input_configs, output_configs=output_configs
        )

    aggregated_configs = Configs(
        global_config=global_config,
        input_configs=input_configs,
        fusion_config=fusion_config,
        output_configs=output_configs,
    )

    return aggregated_configs


def get_global_config(global_configs: Iterable[dict]) -> schemas.GlobalConfig:
    combined_config = combine_dicts(dicts=global_configs)

    combined_config_namespace = Namespace(**combined_config)

    global_config_object = schemas.GlobalConfig(**combined_config_namespace.__dict__)

    global_config_object_mod = modify_global_config(global_config=global_config_object)

    return global_config_object_mod


def modify_global_config(global_config: schemas.GlobalConfig) -> schemas.GlobalConfig:
    gc_copy = copy(global_config)

    if gc_copy.valid_size > 1.0:
        gc_copy.valid_size = int(gc_copy.valid_size)

    return gc_copy


def get_input_configs(
    input_configs: Iterable[dict],
) -> Sequence[schemas.InputConfig]:
    config_objects = []

    for config_dict in input_configs:
        config_object = init_input_config(yaml_config_as_dict=config_dict)
        config_objects.append(config_object)

    _check_input_config_names(input_configs=config_objects)
    return config_objects


def _check_input_config_names(input_configs: Iterable[schemas.InputConfig]) -> None:
    names = []
    for config in input_configs:
        names.append(config.input_info.input_name)

    if len(set(names)) != len(names):
        counts = Counter(names)
        raise ValueError(
            f"Found duplicates of input names in input configs: {counts}. "
            f"Please make sure each input source has a unique input_name."
        )


def init_input_config(yaml_config_as_dict: Dict[str, Any]) -> schemas.InputConfig:
    cfg = yaml_config_as_dict

    input_info_object = schemas.InputDataConfig(**cfg["input_info"])

    input_schema_map = get_inputs_schema_map()
    input_type_info_class = input_schema_map[input_info_object.input_type]
    input_type_info_object = input_type_info_class(**cfg.get("input_type_info", {}))
    _validate_input_type_info_object(
        input_type_info_object=input_type_info_object,
        input_source=input_info_object.input_source,
    )

    model_config = set_up_input_feature_extractor_config(
        input_info_object=input_info_object,
        input_type_info_object=input_type_info_object,
        model_init_kwargs_base=cfg.get("model_config", {}),
    )

    pretrained_config = set_up_pretrained_config(
        pretrained_config_dict=cfg.get("pretrained_config", None)
    )

    interpretation_config = set_up_interpretation_config(
        input_type=input_info_object.input_type,
        interpretation_config_dict=cfg.get("interpretation_config", None),
    )

    input_config = schemas.InputConfig(
        input_info=input_info_object,
        input_type_info=input_type_info_object,
        pretrained_config=pretrained_config,
        model_config=model_config,
        interpretation_config=interpretation_config,
    )

    return input_config


def get_inputs_schema_map() -> (
    Dict[
        str,
        Union[
            Type[schemas.OmicsInputDataConfig],
            Type[schemas.TabularInputDataConfig],
            Type[schemas.SequenceInputDataConfig],
            Type[schemas.ByteInputDataConfig],
        ],
    ]
):
    mapping = {
        "omics": schemas.OmicsInputDataConfig,
        "tabular": schemas.TabularInputDataConfig,
        "sequence": schemas.SequenceInputDataConfig,
        "bytes": schemas.ByteInputDataConfig,
        "image": schemas.ImageInputDataConfig,
        "array": schemas.ArrayInputDataConfig,
    }

    return mapping


def _validate_input_type_info_object(
    input_type_info_object: al_input_types, input_source: str
) -> None:
    ito = input_type_info_object
    match ito:
        case schemas.TabularInputDataConfig():
            con = ito.input_con_columns
            cat = ito.input_cat_columns
            common = set(con).intersection(set(cat))
            if common:
                raise ValueError(
                    f"Found columns passed in both continuous and categorical inputs: "
                    f"{common}. In input source: {input_source}."
                    f"Please make sure that each column is only in one of the two, "
                    f"or create differently named copies of the columns."
                )


def set_up_input_feature_extractor_config(
    input_info_object: schemas.InputDataConfig,
    input_type_info_object: al_input_types,
    model_init_kwargs_base: Union[None, dict],
) -> schemas.al_feature_extractor_configs:
    input_type = input_info_object.input_type

    model_config_class = get_input_feature_extractor_config_class(input_type=input_type)

    model_type = model_init_kwargs_base.get("model_type", None)
    if not model_type:
        try:
            model_type = getattr(model_config_class, "model_type")
        except AttributeError:
            raise AttributeError(
                "Not model type specified in model config and could not find default "
                "value for '%s'.",
                input_type,
            )

        logger.info(
            "Input model type not specified in model configuration for input name "
            "'%s', attempting to grab default value.",
            input_info_object.input_name,
        )

    model_type_init_config = set_up_feature_extractor_init_config(
        input_info_object=input_info_object,
        input_type_info_object=input_type_info_object,
        model_init_kwargs_base=model_init_kwargs_base.get("model_init_config", {}),
        model_type=model_type,
    )

    common_kwargs = {
        "model_type": model_type,
        "model_init_config": model_type_init_config,
    }
    other_specific_kwargs = {
        k: v for k, v in model_init_kwargs_base.items() if k not in common_kwargs
    }
    model_config_kwargs = {**common_kwargs, **other_specific_kwargs}
    model_config = model_config_class(**model_config_kwargs)

    return model_config


def get_input_feature_extractor_config_class(
    input_type: str,
) -> schemas.al_feature_extractor_configs_classes:
    model_config_setup_map = get_input_feature_extractor_config_init_class_map()

    return model_config_setup_map.get(input_type)


def get_input_feature_extractor_config_init_class_map() -> (
    Dict[str, schemas.al_feature_extractor_configs_classes]
):
    mapping = {
        "tabular": TabularModelConfig,
        "omics": OmicsModelConfig,
        "sequence": SequenceModelConfig,
        "bytes": SequenceModelConfig,
        "image": ImageModelConfig,
        "array": ArrayModelConfig,
    }

    return mapping


def set_up_feature_extractor_init_config(
    input_info_object: schemas.InputDataConfig,
    input_type_info_object: al_input_types,
    model_init_kwargs_base: Union[None, dict],
    model_type: str,
) -> Dict:
    if getattr(input_type_info_object, "pretrained_model", None):
        return {}

    not_from_eir = get_is_not_eir_model_condition(
        input_info_object=input_info_object,
        model_type=model_type,
    )
    if not_from_eir:
        return model_init_kwargs_base

    if not model_init_kwargs_base:
        model_init_kwargs_base = {}

    model_init_kwargs = copy(model_init_kwargs_base)

    model_config_type_setup_hook = get_feature_extractor_config_type_setup_hook(
        input_type=input_info_object.input_type
    )
    if model_config_type_setup_hook:
        model_init_kwargs = model_config_type_setup_hook(
            init_kwargs=model_init_kwargs,
            input_info_object=input_info_object,
            input_type_info_object=input_type_info_object,
        )

    model_config_map = get_feature_extractor_config_type_init_callable_map()
    model_config_callable = model_config_map[model_type]

    model_config = model_config_callable(**model_init_kwargs)

    return model_config


def get_is_not_eir_model_condition(
    input_info_object: schemas.InputDataConfig, model_type: str
) -> bool:
    is_possibly_external = getattr(input_info_object, "input_type") in (
        "sequence",
        "bytes",
        "image",
    )
    is_unknown_sequence_model = model_type not in ("sequence-default",)
    not_from_eir = is_possibly_external and is_unknown_sequence_model
    return not_from_eir


@overload
def get_feature_extractor_config_type_setup_hook(
    input_type: str,
) -> Callable[[dict, schemas.InputDataConfig, al_input_types], dict]:
    ...


@overload
def get_feature_extractor_config_type_setup_hook(input_type: None) -> None:
    ...


def get_feature_extractor_config_type_setup_hook(input_type):
    model_config_setup_map = get_model_config_type_setup_hook_map()

    return model_config_setup_map.get(input_type, None)


def get_model_config_type_setup_hook_map():
    mapping = {
        "omics": set_up_config_object_init_kwargs_identity,
        "tabular": set_up_config_object_init_kwargs_identity,
        "sequence": set_up_config_object_init_kwargs_identity,
        "bytes": set_up_config_object_init_kwargs_identity,
        "image": set_up_config_object_init_kwargs_identity,
        "array": set_up_config_object_init_kwargs_identity,
    }

    return mapping


def set_up_config_object_init_kwargs_identity(
    init_kwargs: dict, *args, **kwargs
) -> dict:
    return init_kwargs


def get_feature_extractor_config_type_init_callable_map() -> Dict[str, Type]:
    omics_mapping = get_omics_config_dataclass_mapping()
    array_mapping = get_array_config_dataclass_mapping()
    other_mapping = {
        "tabular": SimpleTabularModelConfig,
        "sequence-default": BasicTransformerFeatureExtractorModelConfig,
    }

    mapping = omics_mapping | array_mapping | other_mapping
    return mapping


def set_up_pretrained_config(
    pretrained_config_dict: Union[None, Dict[str, Any]]
) -> Union[None, schemas.BasicPretrainedConfig]:
    if pretrained_config_dict is None:
        return None

    config_class = get_pretrained_config_class()
    if config_class is None:
        return None

    config_object = config_class(**pretrained_config_dict)

    return config_object


def get_pretrained_config_class() -> Type[schemas.BasicPretrainedConfig]:
    return schemas.BasicPretrainedConfig


def set_up_interpretation_config(
    input_type: str, interpretation_config_dict: Union[None, Dict[str, Any]]
) -> Union[None, schemas.BasicInterpretationConfig]:
    config_class = get_interpretation_config_class(input_type=input_type)
    if config_class is None:
        return None

    if interpretation_config_dict is None:
        interpretation_config_dict = {}

    config_object = config_class(**interpretation_config_dict)

    return config_object


def get_interpretation_config_class(
    input_type: str,
) -> Union[None, schemas.BasicInterpretationConfig]:
    mapping = get_interpretation_config_schema_map()

    return mapping.get(input_type, None)


def get_interpretation_config_schema_map() -> (
    Dict[str, Type[schemas.BasicInterpretationConfig]]
):
    mapping = {
        "sequence": schemas.BasicInterpretationConfig,
        "image": schemas.BasicInterpretationConfig,
    }

    return mapping


def load_fusion_configs(fusion_configs: Iterable[dict]) -> schemas.FusionConfig:
    combined_config = combine_dicts(dicts=fusion_configs)

    combined_config.setdefault("model_type", "default")
    combined_config.setdefault("model_config", {})

    fusion_model_type = combined_config["model_type"]

    model_dataclass_config_class = ResidualMLPConfig
    if fusion_model_type == "mgmoe":
        model_dataclass_config_class = MGMoEModelConfig
    elif fusion_model_type == "linear":
        model_dataclass_config_class = IdentityConfig

    fusion_config_kwargs = combined_config["model_config"]
    fusion_config = model_dataclass_config_class(**fusion_config_kwargs)

    fusion_config = schemas.FusionConfig(
        model_type=combined_config["model_type"], model_config=fusion_config
    )

    return fusion_config


def load_output_configs(
    output_configs: Iterable[dict],
    dynamic_output_setup: DynamicOutputSetup,
) -> Sequence[schemas.OutputConfig]:
    output_config_objects = []

    for config_dict in output_configs:
        config_object = init_output_config(
            yaml_config_as_dict=config_dict,
            dynamic_output_setup=dynamic_output_setup,
        )
        output_config_objects.append(config_object)

    _check_output_config_names(output_configs=output_config_objects)
    return output_config_objects


def _check_output_config_names(output_configs: Iterable[schemas.OutputConfig]) -> None:
    names = []
    for config in output_configs:
        names.append(config.output_info.output_name)

    if len(set(names)) != len(names):
        counts = Counter(names)
        raise ValueError(
            f"Found duplicates of input names in input configs: {counts}. "
            f"Please make sure each input source has a unique input_name."
        )


def init_output_config(
    yaml_config_as_dict: Dict[str, Any],
    dynamic_output_setup: DynamicOutputSetup,
) -> schemas.OutputConfig:
    cfg = yaml_config_as_dict
    ds = dynamic_output_setup

    output_info_object = schemas.OutputInfoConfig(**cfg["output_info"])

    output_schema_map = ds.output_types_schema_map
    output_type_info_class = output_schema_map[output_info_object.output_type]
    output_type_info_object = output_type_info_class(**cfg.get("output_type_info", {}))

    model_config = set_up_output_module_config(
        output_info_object=output_info_object,
        model_init_kwargs_base=cfg.get("model_config", {}),
        output_module_config_class_getter=ds.output_module_config_class_getter,
        output_module_init_class_map=ds.output_module_init_class_map,
    )

    output_config = schemas.OutputConfig(
        output_info=output_info_object,
        output_type_info=output_type_info_object,
        model_config=model_config,
        sampling_config=cfg.get("sampling_config", {}),
    )

    return output_config


def get_outputs_types_schema_map() -> (
    Dict[
        str,
        Type[schemas.TabularOutputTypeConfig] | Type[schemas.SequenceOutputTypeConfig],
    ]
):
    mapping = {
        "tabular": schemas.TabularOutputTypeConfig,
        "sequence": schemas.SequenceOutputTypeConfig,
    }

    return mapping


def get_output_module_config_class(
    output_type: str,
) -> schemas.al_output_module_configs_classes:
    model_config_setup_map = get_output_module_config_class_map()

    return model_config_setup_map.get(output_type)


def get_output_module_config_class_map() -> (
    Dict[str, schemas.al_output_module_configs_classes]
):
    mapping = {
        "tabular": TabularOutputModuleConfig,
        "sequence": SequenceOutputModuleConfig,
    }

    return mapping


def set_up_output_module_config(
    output_info_object: schemas.OutputInfoConfig,
    model_init_kwargs_base: Union[None, dict],
    output_module_config_class_getter: al_output_module_config_class_getter,
    output_module_init_class_map: al_output_model_init_map,
) -> schemas.al_output_module_configs:
    output_type = output_info_object.output_type

    model_config_class = output_module_config_class_getter(output_type=output_type)

    model_type = None
    if model_init_kwargs_base:
        model_type = model_init_kwargs_base.get("model_type", None)

    if not model_type:
        try:
            model_type = getattr(model_config_class, "model_type")
        except AttributeError:
            raise AttributeError(
                "Not model type specified in model config and could not find default "
                "value for '%s'.",
                output_type,
            )

        logger.info(
            "Output model type not specified in model configuration with name '%s', "
            "attempting to grab default value.",
            output_info_object.output_name,
        )

    model_type_config = set_up_output_module_init_config(
        model_init_kwargs_base=model_init_kwargs_base.get("model_init_config", {}),
        output_type=output_type,
        model_type=model_type,
        output_module_init_class_map=output_module_init_class_map,
    )

    common_kwargs = {"model_type": model_type, "model_init_config": model_type_config}
    other_specific_kwargs = {
        k: v for k, v in model_init_kwargs_base.items() if k not in common_kwargs
    }
    model_config_kwargs = {**common_kwargs, **other_specific_kwargs}
    model_config = model_config_class(**model_config_kwargs)

    return model_config


def set_up_output_module_init_config(
    model_init_kwargs_base: Union[None, dict],
    output_type: Literal["tabular", "sequence"],
    model_type: str,
    output_module_init_class_map: al_output_model_init_map,
) -> al_output_model_configs:
    if not model_init_kwargs_base:
        model_init_kwargs_base = {}

    model_init_kwargs = copy(model_init_kwargs_base)

    model_init_config_callable = output_module_init_class_map[output_type][model_type]

    model_init_config = model_init_config_callable(**model_init_kwargs)

    return model_init_config


def get_output_config_type_init_callable_map() -> Dict[str, Dict[str, Type]]:
    mapping = {
        "tabular": {
            "mlp_residual": ResidualMLPOutputModulelConfig,
            "linear": LinearOutputModuleConfig,
        },
        "sequence": {
            "sequence": TransformerSequenceOutputModuleConfig,
        },
    }

    return mapping


def load_configs_general(config_dict_iterable: Iterable[dict], cls: Type):
    config_objects = []

    for config_dict in config_dict_iterable:
        config_object = cls(**config_dict)
        config_objects.append(config_object)

    return config_objects


@dataclass
class TabularTargets:
    con_targets: Dict[str, Sequence[str]]
    cat_targets: Dict[str, Sequence[str]]

    def __len__(self):
        all_con = sum(len(i) for i in self.con_targets.values())
        all_cat = sum(len(i) for i in self.cat_targets.values())
        return all_con + all_cat


def get_all_tabular_targets(
    output_configs: Iterable[schemas.OutputConfig],
) -> TabularTargets:
    con_targets = {}
    cat_targets = {}

    for output_config in output_configs:
        if output_config.output_info.output_type != "tabular":
            continue

        output_name = output_config.output_info.output_name
        con_targets[output_name] = output_config.output_type_info.target_con_columns
        cat_targets[output_name] = output_config.output_type_info.target_cat_columns

    targets = TabularTargets(con_targets=con_targets, cat_targets=cat_targets)
    return targets


def _check_input_and_output_config_names(
    input_configs: Sequence[schemas.InputConfig],
    output_configs: Sequence[schemas.OutputConfig],
) -> None:
    input_names = set(i.input_info.input_name for i in input_configs)
    output_names = set(i.output_info.output_name for i in output_configs)

    intersection = output_names.intersection(input_names)
    if len(intersection) > 0:
        raise ValueError(
            "Found common names in input and output configs. Please ensure"
            " that there are no common names between the input and output configs."
            " Input config names: '%s'.\n"
            " Output config names: '%s'.\n",
            " Common names: '%s'.\n",
            input_names,
            output_names,
            intersection,
        )


def combine_dicts(dicts: Iterable[dict]) -> dict:
    combined_dict = {}

    for dict_ in dicts:
        combined_dict = {**combined_dict, **dict_}

    return combined_dict
