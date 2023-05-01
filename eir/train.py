import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Union,
    Tuple,
    List,
    Dict,
    TYPE_CHECKING,
    Callable,
    Optional,
)

import torch
from aislib.misc_utils import ensure_path_exists
from aislib.misc_utils import get_logger
from ignite.engine import Engine
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from eir.data_load import data_utils
from eir.data_load import datasets
from eir.data_load.data_utils import get_train_sampler
from eir.data_load.label_setup import (
    split_ids,
    save_transformer_set,
)
from eir.experiment_io.experiment_io import (
    serialize_experiment,
    get_default_experiment_keys_to_serialize,
    serialize_all_input_transformers,
    serialize_chosen_input_objects,
)
from eir.models import al_meta_model
from eir.models.model_setup import get_model, get_default_model_registry_per_input_type
from eir.models.model_training_utils import run_lr_find
from eir.setup import schemas
from eir.setup.config import (
    get_configs,
    Configs,
)
from eir.setup.input_setup import al_input_objects_as_dict, set_up_inputs_for_training
from eir.setup.output_setup import (
    al_output_objects_as_dict,
    set_up_outputs_for_training,
)
from eir.target_setup.target_label_setup import (
    set_up_tabular_target_labels_wrapper,
    gather_all_ids_from_output_configs,
    read_manual_ids_if_exist,
    get_tabular_target_file_infos,
)
from eir.train_utils import distributed
from eir.train_utils import utils
from eir.train_utils.metrics import (
    calculate_prediction_losses,
    get_average_history_filepath,
    get_default_metrics,
)
from eir.train_utils.optim import (
    get_optimizer,
    maybe_wrap_model_with_swa,
)
from eir.train_utils.step_logic import (
    al_training_labels_target,
    get_default_hooks,
    Hooks,
)
from eir.train_utils.train_handlers import configure_trainer
from eir.train_utils.utils import (
    call_hooks_stage_iterable,
)

if TYPE_CHECKING:
    from eir.train_utils.metrics import (
        al_step_metric_dict,
        al_metric_record_dict,
    )

al_criteria = Dict[str, Dict[str, Union[nn.CrossEntropyLoss, nn.MSELoss]]]

utils.seed_everything()
logger = get_logger(name=__name__, tqdm_compatible=True)


def main():
    torch.backends.cudnn.benchmark = True

    configs = get_configs()

    configs, local_rank = distributed.maybe_initialize_distributed_environment(
        configs=configs
    )

    default_hooks = get_default_hooks(configs=configs)
    default_experiment = get_default_experiment(
        configs=configs,
        hooks=default_hooks,
    )

    run_experiment(experiment=default_experiment)


@dataclass(frozen=True)
class Experiment:
    configs: Configs
    inputs: al_input_objects_as_dict
    outputs: al_output_objects_as_dict
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    valid_dataset: torch.utils.data.Dataset
    model: al_meta_model
    optimizer: Optimizer
    criteria: al_criteria
    loss_function: Callable
    writer: SummaryWriter
    metrics: "al_metric_record_dict"
    hooks: Union[Hooks, None]


def get_default_experiment(
    configs: Configs,
    hooks: Union[Hooks, None] = None,
) -> "Experiment":
    run_folder = _prepare_run_folder(output_folder=configs.global_config.output_folder)

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=configs.output_configs
    )
    manual_valid_ids = read_manual_ids_if_exist(
        manual_valid_ids_file=configs.global_config.manual_valid_ids_file
    )

    train_ids, valid_ids = split_ids(
        ids=all_array_ids,
        valid_size=configs.global_config.valid_size,
        manual_valid_ids=manual_valid_ids,
    )

    logger.debug("Setting up target labels.")
    target_labels_info = get_tabular_target_file_infos(
        output_configs=configs.output_configs
    )

    custom_ops = hooks.custom_column_label_parsing_ops if hooks else None
    target_labels = set_up_tabular_target_labels_wrapper(
        tabular_target_file_infos=target_labels_info,
        custom_label_ops=custom_ops,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )
    save_transformer_set(
        transformers_per_source=target_labels.label_transformers, run_folder=run_folder
    )

    inputs_as_dict = set_up_inputs_for_training(
        inputs_configs=configs.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=hooks,
    )

    serialize_all_input_transformers(inputs_dict=inputs_as_dict, run_folder=run_folder)
    serialize_chosen_input_objects(inputs_dict=inputs_as_dict, run_folder=run_folder)

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=configs.output_configs,
        target_transformers=target_labels.label_transformers,
    )

    train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
        configs=configs,
        target_labels=target_labels,
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
    )

    train_sampler = get_train_sampler(
        columns_to_sample=configs.global_config.weighted_sampling_columns,
        train_dataset=train_dataset,
    )

    train_dloader, valid_dloader = get_dataloaders(
        train_dataset=train_dataset,
        train_sampler=train_sampler,
        valid_dataset=valid_dataset,
        batch_size=configs.global_config.batch_size,
        num_workers=configs.global_config.dataloader_workers,
    )

    default_registry = get_default_model_registry_per_input_type()

    model = get_model(
        global_config=configs.global_config,
        inputs_as_dict=inputs_as_dict,
        fusion_config=configs.fusion_config,
        outputs_as_dict=outputs_as_dict,
        model_registry_per_input_type=default_registry,
        model_registry_per_output_type={},
    )

    model = maybe_wrap_model_with_swa(
        n_iter_before_swa=configs.global_config.n_iter_before_swa,
        model=model,
        device=torch.device(configs.global_config.device),
    )

    criteria = _get_criteria(
        outputs_as_dict=outputs_as_dict,
    )

    writer = get_summary_writer(run_folder=run_folder)

    loss_func = _get_loss_callable(
        criteria=criteria,
    )

    optimizer = get_optimizer(
        model=model, loss_callable=loss_func, global_config=configs.global_config
    )

    metrics = get_default_metrics(
        target_transformers=target_labels.label_transformers,
        cat_averaging_metrics=configs.global_config.cat_averaging_metrics,
        con_averaging_metrics=configs.global_config.con_averaging_metrics,
    )

    experiment = Experiment(
        configs=configs,
        inputs=inputs_as_dict,
        outputs=outputs_as_dict,
        train_loader=train_dloader,
        valid_loader=valid_dloader,
        valid_dataset=valid_dataset,
        model=model,
        optimizer=optimizer,
        criteria=criteria,
        loss_function=loss_func,
        writer=writer,
        metrics=metrics,
        hooks=hooks,
    )

    return experiment


def _prepare_run_folder(output_folder: str) -> Path:
    run_folder = utils.get_run_folder(output_folder=output_folder)
    history_file = get_average_history_filepath(
        run_folder=run_folder, train_or_val_target_prefix="train_"
    )
    if history_file.exists():
        raise FileExistsError(
            f"There already exists a run with that name: {history_file}. Please choose "
            f"a different run name or delete the folder."
        )

    ensure_path_exists(path=run_folder, is_folder=True)

    return run_folder


def get_dataloaders(
    train_dataset: datasets.DatasetBase,
    train_sampler: Union[None, WeightedRandomSampler],
    valid_dataset: datasets.DatasetBase,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple:
    check_dataset_and_batch_size_compatiblity(
        dataset=train_dataset, batch_size=batch_size, name="Training"
    )

    check_dataset_and_batch_size_compatiblity(
        dataset=valid_dataset, batch_size=batch_size, name="Validation"
    )
    train_dloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    valid_dloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_dloader, valid_dloader


def check_dataset_and_batch_size_compatiblity(
    dataset: datasets.DatasetBase, batch_size: int, name: str = ""
):
    if len(dataset) < batch_size:
        raise ValueError(
            f"{name} dataset size ({len(dataset)}) can not be smaller than "
            f"batch size ({batch_size}). A fix can be increasing {name.lower()} sample "
            f"size or reducing the batch size. If predicting on few unknown samples, "
            f"a solution can be setting the batch size to 1 in the global configuration"
            f" passed to the prediction module. Future work includes making this "
            f"easier to work with."
        )


def _get_criteria(outputs_as_dict: al_output_objects_as_dict) -> al_criteria:
    criteria_dict = {}

    def get_criterion(
        column_type_: str, cat_label_smoothing_: float = 0.0
    ) -> Union[nn.CrossEntropyLoss, Callable]:
        if column_type_ == "con":
            assert cat_label_smoothing_ == 0.0
            return partial(_calc_mse, mse_loss_func=nn.MSELoss())
        elif column_type_ == "cat":
            return nn.CrossEntropyLoss(label_smoothing=cat_label_smoothing_)

    target_columns_gen = data_utils.get_output_info_generator(
        outputs_as_dict=outputs_as_dict
    )

    for output_name, column_type, column_name in target_columns_gen:
        label_smoothing = _get_label_smoothing(
            output_config=outputs_as_dict[output_name].output_config,
            column_type=column_type,
        )

        criterion = get_criterion(
            column_type_=column_type, cat_label_smoothing_=label_smoothing
        )

        if output_name not in criteria_dict:
            criteria_dict[output_name] = {}
        criteria_dict[output_name][column_name] = criterion

    return criteria_dict


def _get_label_smoothing(
    output_config: schemas.OutputConfig,
    column_type: str,
) -> float:
    if column_type == "con":
        return 0.0
    elif column_type == "cat":
        return output_config.output_type_info.cat_label_smoothing

    raise ValueError(f"Unknown column type: {column_type}")


def _calc_mse(input: torch.Tensor, target: torch.Tensor, mse_loss_func: nn.MSELoss):
    return mse_loss_func(input=input.squeeze(), target=target.squeeze())


def _get_loss_callable(criteria: al_criteria):
    single_task_loss_func = partial(calculate_prediction_losses, criteria=criteria)
    return single_task_loss_func


def get_summary_writer(run_folder: Path) -> SummaryWriter:
    log_dir = Path(run_folder / "tensorboard_logs")
    writer = SummaryWriter(log_dir=str(log_dir))

    return writer


def _log_model(
    model: nn.Module, verbose: bool = False, output_file: Optional[str] = None
) -> None:
    no_params = sum(p.numel() for p in model.parameters())
    no_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_summary = "Starting training with following model specifications:\n"
    model_summary += f"Total parameters: {format(no_params, ',.0f')}\n"
    model_summary += f"Trainable parameters: {format(no_trainable_params, ',.0f')}\n"

    logger.info(model_summary)

    if verbose:
        layer_summary = "\nModel layers:\n"
        for name, param in model.named_parameters():
            layer_summary += (
                f"Layer: {name},"
                f"Shape: {list(param.size())},"
                f"Parameters: {param.numel()},"
                f"Trainable: {param.requires_grad}\n"
            )
        logger.info(layer_summary)

    if output_file:
        with open(output_file, "w") as f:
            f.write(model_summary)
            if verbose:
                f.write(layer_summary)


def run_experiment(experiment: Experiment) -> None:
    _log_model(model=experiment.model)

    gc = experiment.configs.global_config

    run_folder = utils.get_run_folder(output_folder=gc.output_folder)
    keys_to_serialize = get_default_experiment_keys_to_serialize()
    serialize_experiment(
        experiment=experiment,
        run_folder=run_folder,
        keys_to_serialize=keys_to_serialize,
    )

    train(experiment=experiment)


def train(experiment: Experiment) -> None:
    exp = experiment
    gc = experiment.configs.global_config

    trainer = get_base_trainer(experiment=experiment)

    if gc.find_lr:
        logger.info("Running LR find and exiting.")
        run_lr_find(
            trainer_engine=trainer,
            train_dataloader=exp.train_loader,
            model=exp.model,
            optimizer=exp.optimizer,
            output_folder=utils.get_run_folder(output_folder=gc.output_folder),
        )
        sys.exit(0)

    trainer = configure_trainer(trainer=trainer, experiment=experiment)

    trainer.run(data=exp.train_loader, max_epochs=gc.n_epochs)


def get_base_trainer(experiment: Experiment) -> Engine:
    step_hooks = experiment.hooks.step_func_hooks

    def step(
        engine: Engine,
        loader_batch: Tuple[torch.Tensor, al_training_labels_target, List[str]],
    ) -> "al_step_metric_dict":
        """
        The output here goes to trainer.output.
        """
        experiment.model.train()
        experiment.optimizer.zero_grad()

        base_prepare_inputs_stage = step_hooks.base_prepare_batch
        state = call_hooks_stage_iterable(
            hook_iterable=base_prepare_inputs_stage,
            common_kwargs={"experiment": experiment, "loader_batch": loader_batch},
            state=None,
        )
        base_batch = state["batch"]

        post_prepare_inputs_stage = step_hooks.post_prepare_batch
        state = call_hooks_stage_iterable(
            hook_iterable=post_prepare_inputs_stage,
            common_kwargs={"experiment": experiment, "loader_batch": base_batch},
            state=state,
        )
        batch = state["batch"]

        model_forward_loss_stage = step_hooks.model_forward
        state = call_hooks_stage_iterable(
            hook_iterable=model_forward_loss_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        loss_stage = step_hooks.loss
        state = call_hooks_stage_iterable(
            hook_iterable=loss_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        optimizer_backward_stage = step_hooks.optimizer_backward
        state = call_hooks_stage_iterable(
            hook_iterable=optimizer_backward_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        metrics_stage = step_hooks.metrics
        state = call_hooks_stage_iterable(
            hook_iterable=metrics_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        return state["metrics"]

    trainer = Engine(process_function=step)

    return trainer


if __name__ == "__main__":
    main()
