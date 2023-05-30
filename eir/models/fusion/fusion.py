from functools import partial
from typing import Type, Union, Dict, Callable, NewType, Literal

import torch
from torch import nn
from aislib.misc_utils import get_logger

from eir.models.fusion import fusion_mgmoe, fusion_default, fusion_identity
from eir.models.layers import ResidualMLPConfig

ComputedType = NewType("Computed", torch.Tensor)
PassThroughType = NewType("PassThrough", Dict[str, torch.Tensor])

al_fusion_model = Literal["pass-through", "mlp-residual", "identity", "mgmoe"]
al_fused_features = Dict[str, ComputedType | PassThroughType]

logger = get_logger(name=__name__)


def pass_through_fuse(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return features


def get_fusion_modules(
    fusion_model_type: str,
    model_config: Union[
        ResidualMLPConfig, fusion_identity.IdentityConfig, fusion_mgmoe.MGMoEModelConfig
    ],
    modules_to_fuse: nn.ModuleDict,
    out_feature_per_feature_extractor: Dict[str, int],
    output_types: dict[str, Literal["tabular", "sequence"]],
) -> nn.ModuleDict:
    _check_fusion_modules(
        output_types=output_types, fusion_model_type=fusion_model_type
    )

    fusion_modules = nn.ModuleDict()

    fusion_in_dim = _get_fusion_input_dimension(modules_to_fuse=modules_to_fuse)

    if any(i for i in output_types.values() if i == "tabular"):
        fusion_class = get_fusion_class(fusion_model_type=fusion_model_type)
        computing_fusion_module = fusion_class(
            model_config=model_config,
            fusion_in_dim=fusion_in_dim,
            out_feature_per_feature_extractor=out_feature_per_feature_extractor,
        )
        fusion_modules["computed"] = computing_fusion_module

    if any(i for i in output_types.values() if i == "sequence"):
        pass_through_fusion_module = fusion_identity.IdentityFusionModel(
            model_config=model_config,
            fusion_in_dim=fusion_in_dim,
            fusion_callable=pass_through_fuse,
        )
        fusion_modules["pass-through"] = pass_through_fusion_module

    return fusion_modules


def _check_fusion_modules(
    output_types: dict[str, Literal["tabular", "sequence"]], fusion_model_type: str
) -> None:
    output_set = set(output_types.values())

    if output_set == {"sequence"} and fusion_model_type != "pass-through":
        raise ValueError(
            "When using only sequence outputs, only pass-through is supported."
            f"Got {fusion_model_type}. To use pass-through, "
            f"set fusion_model_type to 'pass-through'."
        )
    elif output_set == {"sequence", "tabular"} and fusion_model_type != "pass-through":
        logger.warning(
            "Note: When using both sequence and tabular outputs, "
            f"the fusion model type {fusion_model_type} is only applied to"
            "the tabular outputs. The fusion for sequence outputs are handled"
            "by the sequence output module itself, and the feature representations "
            "from the input modules are passed through directly to the output module."
        )
    elif output_set == {"tabular"} and fusion_model_type == "pass-through":
        raise ValueError(
            "When using only tabular outputs, pass-through is not supported."
            f"Got {fusion_model_type}. Kindly set the fusion_model_type "
            "to 'mlp-residual' 'mgmoe', or 'identity'."
        )


def _get_fusion_input_dimension(modules_to_fuse: nn.ModuleDict) -> int:
    fusion_in_dim = sum(i.num_out_features for i in modules_to_fuse.values())
    return fusion_in_dim


def get_fusion_class(
    fusion_model_type: str,
) -> Type[nn.Module] | Callable:
    if fusion_model_type == "mgmoe":
        return fusion_mgmoe.MGMoEModel
    elif fusion_model_type == "mlp-residual":
        return fusion_default.MLPResidualFusionModule
    elif fusion_model_type == "identity":
        return fusion_identity.IdentityFusionModel
    elif fusion_model_type == "pass-through":
        return partial(
            fusion_identity.IdentityFusionModel, fusion_callable=pass_through_fuse
        )
    raise ValueError(f"Unrecognized fusion model type: {fusion_model_type}.")
