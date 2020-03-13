import atexit
import json
from argparse import Namespace
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Callable, Union, Tuple, TYPE_CHECKING, Dict

import pandas as pd
from aislib.misc_utils import get_logger
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import RunningAverage
from torch.utils.tensorboard import SummaryWriter

from human_origins_supervised.data_load.data_utils import get_target_columns_generator
from human_origins_supervised.train_utils import H_PARAMS
from human_origins_supervised.train_utils.activation_analysis import (
    activation_analysis_handler,
)
from human_origins_supervised.train_utils.evaluation import validation_handler
from human_origins_supervised.train_utils.lr_scheduling import (
    get_eval_history_path,
    set_up_scheduler,
    attach_lr_scheduler,
    _calc_plateu_patience,
)
from human_origins_supervised.train_utils.metrics import get_train_metrics
from human_origins_supervised.train_utils.utils import (
    get_custom_module_submodule,
    get_run_folder,
)
from human_origins_supervised.visualization import visualization_funcs as vf
from human_origins_supervised.train_utils.metrics import (
    get_metrics_dataframes,
    get_best_average_performance,
    persist_metrics,
    get_metrics_files,
    read_metrics_history_file,
)

if TYPE_CHECKING:
    from human_origins_supervised.train import Config
    from human_origins_supervised.train_utils.metrics import al_step_metric_dict

# Aliases
al_get_custom_handles_return_value = Union[Tuple[Callable, ...], Tuple[None]]
al_get_custom_handlers = Callable[["HandlerConfig"], al_get_custom_handles_return_value]


logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class HandlerConfig:
    config: "Config"
    run_folder: Path
    run_name: str
    pbar: ProgressBar
    monitoring_metrics: List[Tuple[str, str]]


def _get_early_stopping_handler(trainer: Engine, handler_config: HandlerConfig):
    cl_args = handler_config.config.cl_args
    c = handler_config.config
    eval_history_fpath = get_eval_history_path(run_name=cl_args.run_name)

    def scoring_function(engine):
        eval_df = read_metrics_history_file(eval_history_fpath)
        latest_val_loss = eval_df["v_perf-average"].iloc[-1]
        return latest_val_loss

    plateau_steps = _calc_plateu_patience(
        steps_per_epoch=len(c.train_loader), sample_interval=cl_args.sample_interval
    )
    patience = plateau_steps + int(plateau_steps / 2)

    handler = EarlyStopping(
        patience=patience, score_function=scoring_function, trainer=trainer
    )
    handler.logger = logger

    return handler


def configure_trainer(trainer: Engine, config: "Config") -> Engine:

    cl_args = config.cl_args
    run_folder = Path("runs/", cl_args.run_name)
    pbar = ProgressBar()
    run_name = cl_args.run_name

    monitoring_metrics = _get_monitoring_metrics(target_columns=config.target_columns)

    handler_config = HandlerConfig(
        config=config,
        run_folder=run_folder,
        run_name=run_name,
        pbar=pbar,
        monitoring_metrics=monitoring_metrics,
    )

    for handler in validation_handler, activation_analysis_handler:
        trainer.add_event_handler(
            event_name=Events.ITERATION_COMPLETED(every=cl_args.sample_interval),
            handler=handler,
            handler_config=handler_config,
        )

        if (
            _do_run_completed_handler(
                iter_per_epoch=len(config.train_loader),
                n_epochs=cl_args.n_epochs,
                sample_interval=cl_args.sample_interval,
            )
            and not cl_args.early_stopping
        ):
            trainer.add_event_handler(
                event_name=Events.COMPLETED,
                handler=handler,
                handler_config=handler_config,
            )

    if cl_args.early_stopping:
        early_stopping_handler = _get_early_stopping_handler(
            trainer=trainer, handler_config=handler_config
        )
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=cl_args.sample_interval),
            early_stopping_handler,
        )

    if cl_args.lr_schedule != "same":
        lr_scheduler = set_up_scheduler(handler_config=handler_config)
        attach_lr_scheduler(engine=trainer, lr_scheduler=lr_scheduler, config=config)

    _attach_running_average_metrics(
        engine=trainer, monitoring_metrics=monitoring_metrics
    )
    if not cl_args.no_pbar:
        pbar.attach(engine=trainer, metric_names=["t_loss-average"])

    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=_log_stats_to_pbar,
        handler_config=handler_config,
    )

    if handler_config.run_name:
        trainer = _attach_run_event_handlers(
            trainer=trainer, handler_config=handler_config
        )

    return trainer


def _do_run_completed_handler(iter_per_epoch: int, n_epochs: int, sample_interval: int):
    """
    We need this function to avoid running handlers twice in cases where the total
    number of iterations has a remainder of 0 w.r.t. the sample interval.
    """
    if (iter_per_epoch * n_epochs) % sample_interval != 0:
        return True

    return False


def _get_monitoring_metrics(target_columns) -> List[Tuple[str, str]]:
    """
    The spec for the tuple here follows the metric dict spec, i.e. the tuple is:
    (column_name, metric).
    """
    target_columns_gen = get_target_columns_generator(target_columns)

    loss_average_metrics = tuple(["t_average", "t_loss-average"])
    perf_average_metrics = tuple(["t_average", "t_perf-average"])
    monitoring_metrics = [loss_average_metrics, perf_average_metrics]

    for column_type, column_name in target_columns_gen:

        cur_metrics = get_train_metrics(
            column_type=column_type, prefix=f"t_{column_name}"
        )

        for metric in cur_metrics:
            cur_tuple = tuple([column_name, metric])
            monitoring_metrics.append(cur_tuple)

    return monitoring_metrics


def _attach_running_average_metrics(
    engine: Engine, monitoring_metrics: List[Tuple[str, str]]
) -> None:
    """
    For each metric, we create an output_transform function that grabs the
    target variable from the output of the step function (which is a dict).

    Basically what we attach to the trainer operates on the output of the
    update / step function, that we pass to the Engine definition.

    We use a partial so each lambda has it's own metric variable (otherwise
    they all reference the same object as it gets overwritten).
    """
    for column_name, metric_name in monitoring_metrics:

        def output_transform(
            metric_dict_from_step: "al_step_metric_dict",
            column_name_key: str,
            metric_name_key: str,
        ) -> float:
            return metric_dict_from_step[column_name_key][metric_name_key]

        partial_func = partial(
            output_transform, column_name_key=column_name, metric_name_key=metric_name
        )

        RunningAverage(
            output_transform=partial_func, alpha=0.98, epoch_bound=False
        ).attach(engine, name=metric_name)


def _log_stats_to_pbar(engine: Engine, handler_config: HandlerConfig) -> None:
    log_string = f"[Epoch {engine.state.epoch}/{engine.state.max_epochs}]"

    key = "t_loss-average"
    value = engine.state.metrics[key]
    log_string += f" | {key}: {value:.4g}"

    handler_config.pbar.log_message(log_string)


def _attach_run_event_handlers(trainer: Engine, handler_config: HandlerConfig):
    c = handler_config.config
    cl_args = handler_config.config.cl_args

    checkpoint_handler = ModelCheckpoint(
        dirname=Path(handler_config.run_folder, "saved_models"),
        filename_prefix=Path(cl_args.run_name).name,
        create_dir=True,
        n_saved=100,
        save_as_state_dict=True,
    )

    _save_config(run_folder=handler_config.run_folder, cl_args=cl_args)

    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=cl_args.checkpoint_interval),
        handler=checkpoint_handler,
        to_save={"model": handler_config.config.model},
    )

    # *gotcha*: write_metrics needs to be attached before plot progress so we have the
    # last row when plotting
    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED,
        handler=_write_training_metrics_handler,
        handler_config=handler_config,
    )

    for plot_event in [
        Events.ITERATION_COMPLETED(every=cl_args.sample_interval),
        Events.COMPLETED,
    ]:
        if plot_event == Events.COMPLETED and not _do_run_completed_handler(
            iter_per_epoch=len(c.train_loader),
            n_epochs=cl_args.n_epochs,
            sample_interval=cl_args.sample_interval,
        ):
            continue

        trainer.add_event_handler(
            event_name=plot_event,
            handler=_plot_progress_handler,
            handler_config=handler_config,
        )

    if cl_args.custom_lib:
        custom_handlers = _get_custom_handlers(handler_config)
        trainer = _attach_custom_handlers(trainer, handler_config, custom_handlers)

    log_tb_hparams_on_exit_func = partial(
        add_hparams_to_tensorboard,
        h_params=H_PARAMS,
        config=handler_config.config,
        writer=handler_config.config.writer,
    )
    atexit.register(log_tb_hparams_on_exit_func)

    return trainer


def _save_config(run_folder: Path, cl_args: Namespace):
    with open(str(run_folder / "cl_args.json"), "w") as config_file:
        config_dict = vars(cl_args)
        json.dump(config_dict, config_file, sort_keys=True, indent=4)


def _write_training_metrics_handler(engine: Engine, handler_config: HandlerConfig):
    """
    Note that trainer.state.metrics contains the *running averages* we are interested
    in.

    The main "problem" here is that we lose the structure of the metrics dict that
    we get from `train_utils.metrics.calculate_batch_metrics`, so we have to
    filter all metrics for a given target column specifically, from the 1d array
    trainer.state.metrics gives us.
    """

    iteration = engine.state.iteration

    is_first_iteration = True if iteration == 1 else False

    running_average_metrics = _unflatten_engine_metrics_dict(
        step_base=engine.state.output, engine_metrics_dict=engine.state.metrics
    )
    persist_metrics(
        handler_config=handler_config,
        metrics_dict=running_average_metrics,
        iteration=iteration,
        write_header=is_first_iteration,
        prefixes={"metrics": "t_", "writer": "train"},
    )


def _unflatten_engine_metrics_dict(
    step_base: "al_step_metric_dict", engine_metrics_dict: Dict[str, float]
) -> "al_step_metric_dict":
    """
    We need this to streamline the 1D dictionary that comes from engine.state.metrics.
    """
    unflattened_dict = {}
    for column_name, column_metric_dict in step_base.items():
        unflattened_dict[column_name] = {}

        for column_metric_name in column_metric_dict.keys():
            eng_run_avg_value = engine_metrics_dict[column_metric_name]
            unflattened_dict[column_name][column_metric_name] = eng_run_avg_value

    return unflattened_dict


def _plot_progress_handler(engine: Engine, handler_config: HandlerConfig) -> None:
    cl_args = handler_config.config.cl_args

    # if no val data is available yet
    if engine.state.iteration < cl_args.sample_interval:
        return

    run_folder = get_run_folder(cl_args.run_name)

    for results_dir in (run_folder / "results").iterdir():
        target_column = results_dir.name

        train_history_df, valid_history_df = get_metrics_dataframes(
            results_dir=results_dir, target_string=target_column
        )

        vf.generate_all_training_curves(
            training_history_df=train_history_df,
            valid_history_df=valid_history_df,
            output_folder=results_dir,
            title_extra=target_column,
            plot_skip_steps=cl_args.plot_skip_steps,
        )

    train_avg_history_df, valid_avg_history_df = get_metrics_dataframes(
        results_dir=run_folder, target_string="average"
    )

    vf.generate_all_training_curves(
        training_history_df=train_avg_history_df,
        valid_history_df=valid_avg_history_df,
        output_folder=run_folder,
        title_extra="Multi Task Average",
        plot_skip_steps=cl_args.plot_skip_steps,
    )

    with open(Path(handler_config.run_folder, "model_info.txt"), "w") as mfile:
        mfile.write(str(handler_config.config.model))


def _get_custom_handlers(handler_config: "HandlerConfig"):
    custom_lib = handler_config.config.cl_args.custom_lib

    custom_handlers_module = get_custom_module_submodule(custom_lib, "custom_handlers")

    if not custom_handlers_module:
        return None

    if not hasattr(custom_handlers_module, "get_custom_handlers"):
        raise ImportError(
            f"'get_custom_handlers' function must be defined in "
            f"{custom_handlers_module} for custom handler attachment."
        )

    custom_handlers_getter = custom_handlers_module.get_custom_handlers
    custom_handlers = custom_handlers_getter(handler_config)

    return custom_handlers


def _attach_custom_handlers(trainer: Engine, handler_config, custom_handlers):
    if not custom_handlers:
        return trainer

    for custom_handler_attacher in custom_handlers:
        trainer = custom_handler_attacher(trainer, handler_config)

    return trainer


def add_hparams_to_tensorboard(
    h_params: List[str], config: "Config", writer: SummaryWriter
) -> None:

    logger.debug(
        "Exiting and logging best hyperparameters for best average loss "
        "to tensorboard."
    )

    c = config
    run_folder = get_run_folder(c.cl_args.run_name)

    metrics_files = get_metrics_files(
        target_columns=c.target_columns, run_folder=run_folder, target_prefix="v_"
    )

    try:
        best_overall_performance = get_best_average_performance(
            val_metrics_files=metrics_files, target_columns=c.target_columns
        )
        average_loss_file = metrics_files["v_average"]
        average_loss_df = pd.read_csv(average_loss_file)

    except FileNotFoundError as e:
        logger.debug(
            "Could not find %s at exit. " "Tensorboard hyper parameters not logged.",
            e.filename,
        )
        return

    h_param_dict = _generate_h_param_dict(cl_args=c.cl_args, h_params=h_params)
    min_loss = average_loss_df["v_loss-average"].min()

    writer.add_hparams(
        h_param_dict,
        {
            "v_loss-overall_min": min_loss,
            "best_overall_performance": best_overall_performance,
        },
    )


def _generate_h_param_dict(
    cl_args: Namespace, h_params: List[str]
) -> Dict[str, Union[str, float, int]]:

    h_param_dict = {}

    for param_name in h_params:
        param_value = getattr(cl_args, param_name)

        if isinstance(param_value, (tuple, list)):
            param_value = "_".join([str(p) for p in param_value])
        elif param_value is None:
            param_value = str(param_value)

        h_param_dict[param_name] = param_value

    return h_param_dict
