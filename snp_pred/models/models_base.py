from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Union, Tuple, Dict, Callable, Iterable, Any

import torch
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from torch import nn

from snp_pred.data_load.datasets import al_num_classes
from snp_pred.models import extra_inputs_module
from snp_pred.models.extra_inputs_module import al_emb_lookup_dict

# type aliases

logger = get_logger(name=__name__, tqdm_compatible=True)


class ModelBase(nn.Module):
    def __init__(
        self,
        cl_args: Namespace,
        num_classes: al_num_classes,
        embeddings_dict: Union[al_emb_lookup_dict, None] = None,
        extra_continuous_inputs_columns: Union[List[str], None] = None,
    ):
        super().__init__()

        self.cl_args = cl_args
        self.num_classes = num_classes
        self.embeddings_dict = embeddings_dict
        self.extra_continuous_inputs_columns = extra_continuous_inputs_columns

        emb_total_dim = con_total_dim = 0
        if embeddings_dict:
            emb_total_dim = extra_inputs_module.attach_embeddings(self, embeddings_dict)
        if extra_continuous_inputs_columns:
            con_total_dim = len(self.extra_continuous_inputs_columns)

        self.fc_repr_and_extra_dim = cl_args.fc_repr_dim
        self.fc_task_dim = cl_args.fc_task_dim

        # TODO: Better to have this a method so fc_extra is explicitly defined?
        self.extra_dim = emb_total_dim + con_total_dim
        if emb_total_dim or con_total_dim:
            # we have a specific layer for fc_extra in case it's going straight
            # to bn or act, ensuring linear before
            self.fc_extra = nn.Linear(self.extra_dim, self.extra_dim, bias=False)
            self.fc_repr_and_extra_dim += self.extra_dim

    @property
    def fc_1_in_features(self) -> int:
        raise NotImplementedError


def merge_module_dicts(module_dicts: Tuple[nn.ModuleDict, ...]):
    def _check_inputs():
        assert all(i.keys() == module_dicts[0].keys() for i in module_dicts)

    _check_inputs()

    new_module_dicts = deepcopy(module_dicts)
    final_module_dict = nn.ModuleDict()

    keys = new_module_dicts[0].keys()
    for key in keys:
        final_module_dict[key] = nn.Sequential()

        for index, module_dict in enumerate(new_module_dicts):
            cur_module = module_dict[key]
            final_module_dict[key].add_module(str(index), cur_module)

    return final_module_dict


def construct_blocks(
    num_blocks: int, block_constructor: Callable, block_kwargs: Dict
) -> nn.Sequential:
    blocks = []
    for i in range(num_blocks):
        cur_block = block_constructor(**block_kwargs)
        blocks.append(cur_block)
    return nn.Sequential(*blocks)


def create_multi_task_blocks_with_first_adaptor_block(
    num_blocks: int,
    branch_names,
    block_constructor: Callable,
    block_constructor_kwargs: Dict,
    first_layer_kwargs_overload: Dict,
):

    adaptor_block = construct_multi_branches(
        branch_names=branch_names,
        branch_factory=construct_blocks,
        branch_factory_kwargs={
            "num_blocks": 1,
            "block_constructor": block_constructor,
            "block_kwargs": {**block_constructor_kwargs, **first_layer_kwargs_overload},
        },
    )

    if num_blocks == 1:
        return merge_module_dicts((adaptor_block,))

    blocks = construct_multi_branches(
        branch_names=branch_names,
        branch_factory=construct_blocks,
        branch_factory_kwargs={
            "num_blocks": num_blocks - 1,
            "block_constructor": block_constructor,
            "block_kwargs": {**block_constructor_kwargs},
        },
    )

    merged_blocks = merge_module_dicts((adaptor_block, blocks))

    return merged_blocks


@dataclass
class LayerSpec:
    name: str
    module: nn.Module
    module_kwargs: Dict


def get_basic_multi_branch_spec(in_features: int, out_features: int, dropout_p: float):
    base_spec = OrderedDict(
        {
            "fc_1_linear_1": (
                nn.Linear,
                {
                    "in_features": in_features,
                    "out_features": out_features,
                    "bias": False,
                },
            ),
            "fc_1_bn_1": (nn.BatchNorm1d, {"num_features": out_features}),
            "fc_1_act_1": (Swish, {}),
            "fc_1_do_1": (nn.Dropout, {"p": dropout_p}),
        }
    )

    return base_spec


def assert_module_dict_uniqueness(module_dict: Dict[str, nn.Sequential]):
    """
    We have this function as a safeguard to help us catch if we are reusing modules
    when they should not be (i.e. if splitting into multiple branches with same layers,
    one could accidentally reuse the instantiated nn.Modules across branches).
    """
    branch_ids = [id(sequential_branch) for sequential_branch in module_dict.values()]
    assert len(branch_ids) == len(set(branch_ids))

    module_ids = []
    for sequential_branch in module_dict.values():
        module_ids += [id(module) for module in sequential_branch.modules()]

    num_total_modules = len(module_ids)
    num_unique_modules = len(set(module_ids))
    assert num_unique_modules == num_total_modules


def construct_multi_branches(
    branch_names: Iterable[str],
    branch_factory: Callable[[Any], nn.Sequential],
    branch_factory_kwargs,
    extra_hooks: List[Callable] = (),
) -> nn.ModuleDict:

    branched_module_dict = nn.ModuleDict()
    for name in branch_names:

        cur_branch = branch_factory(**branch_factory_kwargs)
        assert callable(cur_branch)
        branched_module_dict[name] = cur_branch

    for hook in extra_hooks:
        branched_module_dict = hook(branched_module_dict)

    assert_module_dict_uniqueness(branched_module_dict)
    return branched_module_dict


def get_final_layer(in_features, num_classes):
    final_module_dict = nn.ModuleDict()

    for task, num_outputs in num_classes.items():
        cur_spec = OrderedDict(
            {
                "fc_final": (
                    nn.Linear,
                    {
                        "in_features": in_features,
                        "out_features": num_outputs,
                        "bias": True,
                    },
                )
            }
        )
        cur_module = initialize_modules_from_spec(spec=cur_spec)
        final_module_dict[task] = cur_module

    return final_module_dict


def compose_spec_creation_and_initalization(spec_func, **spec_kwargs):
    spec = spec_func(**spec_kwargs)
    module = initialize_modules_from_spec(spec=spec)
    return module


def initialize_modules_from_spec(
    spec: "OrderedDict[str, Tuple[nn.Module, Dict]]",
) -> nn.Sequential:

    module_dict = OrderedDict()
    for name, recipe in spec.items():
        module_class = recipe[0]
        module_args = recipe[1]

        module = initialize_module(module=module_class, module_args=module_args)

        module_dict[name] = module

    return nn.Sequential(module_dict)


def initialize_module(module: nn.Module, module_args: Dict) -> nn.Module:
    return module(**module_args)


def calculate_module_dict_outputs(
    input_: torch.Tensor, module_dict: nn.ModuleDict
) -> "OrderedDict[str, torch.Tensor]":
    final_out = OrderedDict()
    for target_column, linear_layer in module_dict.items():
        final_out[target_column] = linear_layer(input_)

    return final_out