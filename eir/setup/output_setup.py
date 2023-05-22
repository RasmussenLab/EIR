from typing import Dict, Union, Callable, Any, Optional, TYPE_CHECKING

from aislib.misc_utils import get_logger

from eir.data_load.label_setup import (
    al_label_transformers,
)
from eir.setup import schemas
from eir.setup.output_setup_modules.sequence_output_setup import (
    set_up_sequence_output,
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    set_up_tabular_output,
    ComputedTabularOutputInfo,
)

if TYPE_CHECKING:
    from eir.setup.input_setup import al_input_objects_as_dict


logger = get_logger(name=__name__)

al_output_objects = ComputedTabularOutputInfo | ComputedSequenceOutputInfo
al_output_objects_as_dict = Dict[str, al_output_objects]


def set_up_outputs_for_training(
    output_configs: schemas.al_output_configs,
    input_objects: Optional["al_input_objects_as_dict"] = None,
    target_transformers: Optional[Dict[str, al_label_transformers]] = None,
) -> al_output_objects_as_dict:
    all_inputs = set_up_outputs_general(
        output_configs=output_configs,
        setup_func_getter=get_output_setup_function_for_train,
        setup_func_kwargs={
            "input_objects": input_objects,
            "target_transformers": target_transformers,
        },
    )

    return all_inputs


def set_up_outputs_general(
    output_configs: schemas.al_output_configs,
    setup_func_getter: Callable[
        [Union[schemas.OutputConfig, Any]], Callable[..., al_output_objects]
    ],
    setup_func_kwargs: Dict[str, Any],
) -> al_output_objects_as_dict:
    all_inputs = {}

    name_config_iter = get_output_name_config_iterator(output_configs=output_configs)

    for name, output_config in name_config_iter:
        setup_func = setup_func_getter(output_config=output_config)

        cur_output_data_config = output_config.output_info
        logger.info(
            "Setting up %s outputs '%s' from %s.",
            cur_output_data_config.output_name,
            cur_output_data_config.output_type,
            cur_output_data_config.output_source,
        )

        set_up_output = setup_func(output_config=output_config, **setup_func_kwargs)
        all_inputs[name] = set_up_output

    return all_inputs


def get_output_setup_function_for_train(
    output_config: schemas.OutputConfig,
) -> Callable[..., al_output_objects]:
    output_type = output_config.output_info.output_type

    mapping = get_output_setup_function_map()

    return mapping[output_type]


def get_output_setup_function_map() -> Dict[str, Callable[..., al_output_objects]]:
    setup_mapping = {
        "tabular": set_up_tabular_output,
        "sequence": set_up_sequence_output,
    }

    return setup_mapping


def get_output_name_config_iterator(output_configs: schemas.al_output_configs):
    """
    We do not allow '.' as it is used in the weighted sampling setup.
    """

    for output_config in output_configs:
        cur_input_data_config = output_config.output_info
        cur_name = cur_input_data_config.output_name

        if "." in cur_name:
            raise ValueError(
                "Having '.' in the output name is currently not supported. Got '%s'."
                "Kindly rename '%s' to not include any '.' symbols.",
                cur_name,
                cur_name,
            )

        yield cur_name, output_config
