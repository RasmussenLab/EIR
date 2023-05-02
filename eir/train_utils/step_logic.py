from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import (
    Dict,
    Union,
    List,
    Tuple,
    Iterable,
    Callable,
    Sequence,
    Any,
    Optional,
    TYPE_CHECKING,
)

import torch
from aislib.misc_utils import get_logger
from ignite.engine import Engine
from torch import nn, autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

from eir.data_load.data_augmentation import get_mix_data_hook, hook_mix_loss
from eir.data_load.data_utils import Batch
from eir.data_load.label_setup import al_all_column_ops
from eir.models import model_training_utils
from eir.models.tabular.tabular import get_tabular_inputs
from eir.setup import schemas
from eir.setup.config import Configs, get_all_tabular_targets
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.output_setup import al_output_objects_as_dict
from eir.train_utils.metrics import (
    get_uncertainty_loss_hook,
    hook_add_l1_loss,
    calculate_batch_metrics,
    add_loss_to_metrics,
    add_multi_task_average_metrics,
    aggregate_losses,
)
from eir.train_utils.optim import get_optimizer_backward_kwargs
from eir.train_utils.train_handlers import HandlerConfig

if TYPE_CHECKING:
    from eir.train import Experiment

al_training_labels_target = Dict[str, Dict[str, torch.Tensor]]
al_input_batch = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
al_ids = List[str]
al_dataloader_getitem_batch = Tuple[
    al_input_batch,
    al_training_labels_target,
    al_ids,
]

logger = get_logger(name=__name__)


def get_default_hooks(configs: Configs):
    step_func_hooks = _get_default_step_function_hooks(configs=configs)
    hooks_object = Hooks(step_func_hooks=step_func_hooks)

    return hooks_object


@dataclass
class Hooks:
    al_handler_attachers = Iterable[Callable[[Engine, HandlerConfig], Engine]]

    step_func_hooks: "StepFunctionHookStages"
    custom_column_label_parsing_ops: al_all_column_ops = None
    custom_handler_attachers: Union[None, al_handler_attachers] = None


def _get_default_step_function_hooks(configs: Configs):
    """
    TODO: Add validation, inspect that outputs have correct names.
    TODO: Refactor, split into smaller functions e.g. for L1, mixing and uncertainty.
    """

    init_kwargs = _get_default_step_function_hooks_init_kwargs(configs=configs)

    step_func_hooks = StepFunctionHookStages(**init_kwargs)

    return step_func_hooks


def _get_default_step_function_hooks_init_kwargs(
    configs: Configs,
) -> Dict[str, Sequence[Callable]]:
    init_kwargs = {
        "base_prepare_batch": [hook_default_prepare_batch],
        "post_prepare_batch": [],
        "model_forward": [hook_default_model_forward],
        "loss": [hook_default_per_target_loss],
        "optimizer_backward": [hook_default_optimizer_backward],
        "metrics": [hook_default_compute_metrics],
    }

    if configs.global_config.mixing_alpha:
        logger.debug(
            "Setting up hooks for mixing with with α=%.2g.",
            configs.global_config.mixing_alpha,
        )
        mix_hook = get_mix_data_hook(input_configs=configs.input_configs)

        init_kwargs["post_prepare_batch"].append(mix_hook)
        init_kwargs["loss"][0] = hook_mix_loss

    if _should_add_uncertainty_loss_hook(output_configs=configs.output_configs):
        uncertainty_hook = get_uncertainty_loss_hook(
            output_configs=configs.output_configs,
            device=configs.global_config.device,
        )
        init_kwargs["loss"].append(uncertainty_hook)

    init_kwargs["loss"].append(hook_default_aggregate_losses)

    init_kwargs = add_l1_loss_hook_if_applicable(
        step_function_hooks_init_kwargs=init_kwargs, configs=configs
    )

    grad_acc_steps = configs.global_config.gradient_accumulation_steps
    if grad_acc_steps and grad_acc_steps > 1:
        logger.debug(
            "Adding gradient accumulation hook with steps=%d.",
            configs.global_config.gradient_accumulation_steps,
        )
    init_kwargs["loss"].append(get_hook_iteration_counter())

    do_amp = configs.global_config.amp
    if do_amp:
        logger.debug("Setting up AMP training.")
        model_forward_with_amp_objects = [
            get_hook_amp_objects(device=configs.global_config.device)
        ] + init_kwargs["model_forward"]
        init_kwargs["model_forward"] = model_forward_with_amp_objects

    return init_kwargs


def _should_add_uncertainty_loss_hook(
    output_configs: Sequence[schemas.OutputConfig],
) -> bool:
    all_targets = get_all_tabular_targets(output_configs=output_configs)

    more_than_one_target = len(all_targets) > 1

    any_uncertainty_targets = any(
        hasattr(c.output_type_info, "uncertainty_weighted_mt_loss")
        and c.output_type_info.uncertainty_weighted_mt_loss
        for c in output_configs
    )
    return more_than_one_target and any_uncertainty_targets


def add_l1_loss_hook_if_applicable(
    step_function_hooks_init_kwargs: Dict[str, Sequence[Callable]],
    configs: Configs,
) -> Dict[str, List[Callable]]:
    input_l1 = any(
        getattr(input_config.model_config.model_init_config, "l1", None)
        for input_config in configs.input_configs
    )
    preds_l1 = getattr(configs.fusion_config.model_config, "l1", None)
    if input_l1 or preds_l1:
        logger.debug("Adding L1 loss hook.")
        step_function_hooks_init_kwargs["loss"].append(hook_add_l1_loss)

    return step_function_hooks_init_kwargs


@dataclass
class StepFunctionHookStages:
    al_hook = Callable[..., Dict]
    al_hooks = [Iterable[al_hook]]

    base_prepare_batch: al_hooks
    post_prepare_batch: al_hooks
    model_forward: al_hooks
    loss: al_hooks
    optimizer_backward: al_hooks
    metrics: al_hooks


def hook_default_prepare_batch(
    experiment: "Experiment",
    loader_batch: al_dataloader_getitem_batch,
    *args,
    **kwargs,
) -> Dict:
    batch = prepare_base_batch_default(
        loader_batch=loader_batch,
        input_objects=experiment.inputs,
        output_objects=experiment.outputs,
        model=experiment.model,
        device=experiment.configs.global_config.device,
    )

    state_updates = {"batch": batch}

    return state_updates


def prepare_base_batch_default(
    loader_batch: al_dataloader_getitem_batch,
    input_objects: al_input_objects_as_dict,
    output_objects: al_output_objects_as_dict,
    model: nn.Module,
    device: str,
) -> Batch:
    inputs, target_labels, ids = loader_batch

    inputs_prepared = _prepare_inputs_for_model(
        batch_inputs=inputs, input_objects=input_objects, model=model, device=device
    )

    if target_labels:
        target_labels = model_training_utils.parse_target_labels(
            output_objects=output_objects,
            device=device,
            labels=target_labels,
        )

    batch = Batch(
        inputs=inputs_prepared,
        target_labels=target_labels,
        ids=ids,
    )

    return batch


def _prepare_inputs_for_model(
    batch_inputs: Dict[str, Any],
    input_objects: al_input_objects_as_dict,
    model: nn.Module,
    device: str,
) -> Dict[str, torch.Tensor]:
    inputs_prepared = {}
    for input_name, input_object in input_objects.items():
        input_type = input_object.input_config.input_info.input_type

        match input_type:
            case "omics" | "image" | "array":
                cur_tensor = batch_inputs[input_name]
                cur_tensor = cur_tensor.to(device=device)
                cur_tensor = cur_tensor.to(dtype=torch.float32)

                inputs_prepared[input_name] = cur_tensor

            case "tabular":
                tabular_source_input = batch_inputs[input_name]
                for tabular_name, tensor in tabular_source_input.items():
                    cur_tensor = tensor.to(device=device)

                    if torch.is_floating_point(input=tensor):
                        cur_tensor = cur_tensor.to(dtype=torch.float32)

                    tabular_source_input[tabular_name] = cur_tensor

                tabular_input_type_info = input_object.input_config.input_type_info
                cat_columns = tabular_input_type_info.input_cat_columns
                con_columns = tabular_input_type_info.input_con_columns
                tabular = get_tabular_inputs(
                    input_cat_columns=cat_columns,
                    input_con_columns=con_columns,
                    tabular_model=getattr(model.input_modules, input_name),
                    tabular_input=tabular_source_input,
                    device=device,
                )
                inputs_prepared[input_name] = tabular

            case "sequence" | "bytes":
                cur_seq = batch_inputs[input_name]
                cur_seq = cur_seq.to(device=device)
                cur_module = getattr(model.input_modules, input_name)
                cur_module_embedding = cur_module.embedding
                cur_embedding = cur_module_embedding(input=cur_seq)
                inputs_prepared[input_name] = cur_embedding
            case _:
                raise ValueError(f"Unrecognized input type {input_name}.")

    return inputs_prepared


def hook_default_model_forward(
    experiment: "Experiment", state: Dict, batch: "Batch", *args, **kwargs
) -> Dict:
    inputs = batch.inputs

    context_manager = get_maybe_amp_context_manager_from_state(state=state)
    with context_manager:
        train_outputs = experiment.model(inputs=inputs)

    state_updates = {"model_outputs": train_outputs}

    return state_updates


def get_amp_context_manager(device_type: str) -> autocast:
    return autocast(device_type=device_type)


def hook_default_optimizer_backward(
    experiment: "Experiment", state: Dict, *args, **kwargs
) -> Dict:
    gc = experiment.configs.global_config
    optimizer_backward_kwargs = get_optimizer_backward_kwargs(
        optimizer_name=gc.optimizer
    )

    loss = maybe_scale_loss_with_grad_accumulation_steps(
        loss=state["loss"], grad_acc_steps=gc.gradient_accumulation_steps
    )

    amp_scaler = state.get("amp_scaler")
    loss = maybe_scale_loss_with_amp_scaler(
        do_amp=gc.amp, loss=loss, amp_scaler=amp_scaler, device=gc.device
    )

    loss.backward(**optimizer_backward_kwargs)

    maybe_apply_gradient_noise_to_model(
        model=experiment.model, gradient_noise=gc.gradient_noise
    )
    maybe_apply_gradient_clipping_to_model(
        model=experiment.model, gradient_clipping=gc.gradient_clipping
    )

    step_func = get_optimizer_step_func(
        do_amp=gc.amp,
        optimizer_step=experiment.optimizer.step,
        amp_scaler=amp_scaler,
        device=gc.device,
    )

    if should_perform_optimizer_step(
        iteration=state["iteration"],
        grad_acc_steps=gc.gradient_accumulation_steps,
    ):
        step_func()

    if gc.amp and gc.device != "cpu":
        amp_scaler.update()

    maybe_update_model_parameters_with_swa(
        n_iter_before_swa=gc.n_iter_before_swa,
        model=experiment.model,
        iteration=state["iteration"],
        sample_interval=gc.sample_interval,
    )

    return {}


def maybe_scale_loss_with_amp_scaler(
    do_amp: bool, loss: torch.Tensor, amp_scaler: Optional["GradScaler"], device: str
) -> torch.Tensor:
    if do_amp and device != "cpu":
        return amp_scaler.scale(loss)
    else:
        return loss


def maybe_scale_loss_with_grad_accumulation_steps(
    loss: torch.Tensor, grad_acc_steps: int
) -> torch.Tensor:
    if grad_acc_steps and grad_acc_steps > 1:
        return loss / grad_acc_steps
    else:
        return loss


def maybe_apply_gradient_noise_to_model(
    model: "nn.Module", gradient_noise: float
) -> None:
    if gradient_noise:
        for name, weight in model.named_parameters():
            weight.grad = weight.grad + torch.randn_like(weight.grad) * gradient_noise


def maybe_apply_gradient_clipping_to_model(
    model: "nn.Module", gradient_clipping: float
) -> None:
    if gradient_clipping:
        clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=gradient_clipping,
        )


def get_optimizer_step_func(
    do_amp: bool,
    optimizer_step: Callable,
    amp_scaler: Optional["GradScaler"],
    device: str,
) -> Callable:
    if do_amp and device != "cpu":
        return partial(amp_scaler.step, optimizer=optimizer_step)
    else:
        return optimizer_step


def should_perform_optimizer_step(iteration: int, grad_acc_steps: int) -> bool:
    if grad_acc_steps and grad_acc_steps > 1:
        return iteration % grad_acc_steps == 0
    else:
        return True


def maybe_update_model_parameters_with_swa(
    n_iter_before_swa: Optional[int],
    model: "nn.Module",
    iteration: int,
    sample_interval: int,
) -> None:
    should_not_be_called_ever = n_iter_before_swa is None
    if should_not_be_called_ever:
        return

    if iteration >= n_iter_before_swa and iteration % sample_interval == 0:
        model.update_parameters(model.module)


def hook_default_compute_metrics(
    experiment: "Experiment", batch: "Batch", state: Dict, *args, **kwargs
):
    train_batch_metrics = calculate_batch_metrics(
        outputs_as_dict=experiment.outputs,
        outputs=state["model_outputs"],
        labels=batch.target_labels,
        mode="train",
        metric_record_dict=experiment.metrics,
    )

    train_batch_metrics_w_loss = add_loss_to_metrics(
        outputs_as_dict=experiment.outputs,
        losses=state["per_target_train_losses"],
        metric_dict=train_batch_metrics,
    )

    train_batch_metrics_with_averages = add_multi_task_average_metrics(
        batch_metrics_dict=train_batch_metrics_w_loss,
        outputs_as_dict=experiment.outputs,
        loss=state["loss"].item(),
        performance_average_functions=experiment.metrics["averaging_functions"],
    )

    state_updates = {"metrics": train_batch_metrics_with_averages}

    return state_updates


def hook_default_per_target_loss(
    experiment: "Experiment", batch: "Batch", state: Dict, *args, **kwargs
) -> Dict:
    context_manager = get_maybe_amp_context_manager_from_state(state=state)
    with context_manager:
        per_target_train_losses = experiment.loss_function(
            inputs=state["model_outputs"], targets=batch.target_labels
        )

        state_updates = {"per_target_train_losses": per_target_train_losses}

    return state_updates


def hook_default_aggregate_losses(state: Dict, *args, **kwargs) -> Dict:
    context_manager = get_maybe_amp_context_manager_from_state(state=state)
    with context_manager:
        train_loss_avg = aggregate_losses(losses_dict=state["per_target_train_losses"])
        state_updates = {"loss": train_loss_avg}

    return state_updates


def get_maybe_amp_context_manager_from_state(
    state: Dict,
) -> Union[nullcontext, autocast]:
    context_manager = state.get("amp_context_manager", nullcontext())
    return context_manager


def get_hook_iteration_counter() -> Callable:
    iteration_count = 0

    def _counter_iterator(do_increment: bool = True, *args, **kwargs) -> Dict[str, int]:
        nonlocal iteration_count
        if do_increment:
            iteration_count += 1

        state_updates = {"iteration": iteration_count}
        return state_updates

    return _counter_iterator


def get_hook_amp_objects(device: str):
    device_type = "cpu" if device == "cpu" else "cuda"

    if device == "cpu":
        logger.warning("Using AMP is on a CPU, speedups will most likely be minimal.")

    scaler = None
    if device != "cpu":
        scaler = GradScaler()

    amp_context_manager = get_amp_context_manager(device_type=device_type)

    def _get_objects(*args, **kwargs) -> Dict[str, GradScaler]:
        state_updates = {
            "amp_context_manager": amp_context_manager,
        }
        if device != "cpu":
            state_updates["amp_scaler"] = scaler

        return state_updates

    return _get_objects


def hook_adjust_loss_for_gradient_accumulation(
    experiment: "Experiment", state: Dict, *args, **kwargs
) -> Dict:
    gradient_accumulation_steps = (
        experiment.configs.global_config.gradient_accumulation_steps
    )

    loss = state["loss"]
    loss_adjusted = loss / gradient_accumulation_steps

    state_updates = {"loss": loss_adjusted}

    return state_updates