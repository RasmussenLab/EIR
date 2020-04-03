import argparse
import atexit
import uuid
from argparse import Namespace
from os.path import abspath
from os import cpu_count
from pathlib import Path
from functools import partial
from typing import Dict, Any, List

import numpy as np
from aislib.misc_utils import get_logger
from ax.core.experiment import Experiment
from ax.core.base_trial import TrialStatus
from ax.core.trial import BaseTrial
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.random import RandomModelBridge
from ax.plot.base import AxPlotConfig
from ax.plot.slice import plot_slice
from ax.plot.trace import optimization_trace_single_method
from ax.service.ax_client import AxClient
from plotly import offline
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS, TBXLogger
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.ax import AxSearch
from ray.tune.analysis import ExperimentAnalysis
from torch import cuda

from human_origins_supervised.data_load.datasets import merge_target_columns
from human_origins_supervised import train
from human_origins_supervised.train_utils.metrics import (
    get_best_average_performance,
    get_metrics_files,
)
from human_origins_supervised.train_utils.utils import get_run_folder

logger = get_logger(name=__name__, tqdm_compatible=True)

TRAIN_CL_BASE = {
    "act_classes": None,
    "b1": 0.9,
    "b2": 0.999,
    "batch_size": 32,
    "channel_exp_base": 5,
    "checkpoint_interval": 10000,
    "extra_con_columns": [],
    "custom_lib": None,
    "data_folder": abspath("None"),
    "debug": False,
    "device": "cuda:0" if cuda.is_available() else "cpu",
    "dilation_factor": 1,
    "down_stride": 4,
    "extra_cat_columns": [],
    "early_stopping": False,
    "fc_repr_dim": 64,
    "fc_task_dim": 32,
    "fc_do": 0.0,
    "find_lr": False,
    "first_kernel_expansion": 1,
    "first_stride_expansion": 1,
    "first_channel_expansion": 1,
    "get_acts": False,
    "gpu_num": "0",
    "kernel_width": 12,
    "label_file": abspath("None"),
    "lr": 1e-2,
    "lr_lb": 1e-5,
    "lr_schedule": "plateau",
    "memory_dataset": False,
    "model_type": "cnn",
    "multi_gpu": False,
    "no_pbar": True,
    "n_cpu": 8,
    "n_epochs": 20,
    "na_augment_perc": 0.0,
    "na_augment_prob": 0.0,
    "optimizer": "adamw",
    "plot_skip_steps": 50,
    "rb_do": 0.0,
    "resblocks": None,
    "run_name": "hyperopt_trial",
    "sa": False,
    "snp_file": None,
    "sample_interval": 200,
    "target_cat_columns": ["None"],
    "target_con_columns": [],
    "target_width": None,
    "valid_size": 0.05,
    "warmup_steps": "auto",
    "wd": 0.00,
    "weighted_sampling_column": None,
}

SEARCH_SPACE = [
    {"name": "batch_size", "type": "choice", "values": [16, 32], "is_ordered": True},
    {
        "name": "lr",
        "type": "range",
        "bounds": [1e-5, 0.5],
        "log_scale": True,
        "digits": 5,
    },
    {"name": "lr_schedule", "type": "choice", "values": ["cycle", "plateau"]},
    {"name": "optimizer", "type": "choice", "values": ["adamw", "sgdm"]},
    {
        "name": "fc_repr_dim",
        "type": "range",
        "bounds": [32, 512],
        "parameter_type": int,
    },
    {
        "name": "fc_task_dim",
        "type": "range",
        "bounds": [32, 128],
        "parameter_type": int,
    },
    {"name": "na_augment_perc", "type": "range", "bounds": [0.0, 0.5], "digits": 2},
    {"name": "na_augment_prob", "type": "range", "bounds": [0.0, 1.0], "digits": 2},
    {"name": "fc_do", "type": "range", "bounds": [0.0, 0.5], "digits": 2},
    {"name": "rb_do", "type": "range", "bounds": [0.0, 0.5], "digits": 2},
    {"name": "kernel_width", "type": "range", "bounds": [2, 20], "parameter_type": int},
    {
        "name": "first_kernel_expansion",
        "type": "range",
        "bounds": [1, 6],
        "parameter_type": int,
    },
    {
        "name": "dilation_factor",
        "type": "range",
        "bounds": [1, 6],
        "parameter_type": int,
    },
    {
        "name": "first_stride_expansion",
        "type": "range",
        "bounds": [1, 2],
        "parameter_type": int,
    },
    {"name": "down_stride", "type": "range", "bounds": [2, 10]},
    {
        "name": "channel_exp_base",
        "type": "range",
        "bounds": [2, 7],
        "parameter_type": int,
    },
    {"name": "wd", "type": "range", "bounds": [1e-5, 1e-1], "log_scale": True},
    {"name": "sa", "type": "choice", "values": [True, False]},
]


def _prep_train_cl_args_namespace(parametrization) -> Namespace:
    """
    We update the sample interval to make sure we sample based on number of samples
    seen.
    """
    config_ = {**TRAIN_CL_BASE, **parametrization}

    config_["sample_interval"] = int(
        32 / config_["batch_size"] * config_["sample_interval"]
    )

    current_run_name = config_["run_name"] + "_" + str(uuid.uuid4())
    config_["run_name"] = current_run_name

    cl_args = Namespace(**config_)

    return cl_args


def get_experiment_func():
    def run_experiment(parametrization):
        train_cl_args = _prep_train_cl_args_namespace(parametrization=parametrization)

        train.main(cl_args=train_cl_args)

        target_columns = merge_target_columns(
            target_cat_columns=train_cl_args.target_cat_columns,
            target_con_columns=train_cl_args.target_con_columns,
        )
        run_folder = get_run_folder(run_name=train_cl_args.run_name)
        metrics_files = get_metrics_files(
            target_columns=target_columns, run_folder=run_folder, target_prefix="v_"
        )

        best_performance = get_best_average_performance(
            val_metrics_files=metrics_files, target_columns=target_columns
        )
        logger.info("Best performance: %f", best_performance)
        tune.track.log(best_average_performance=best_performance)

        return best_performance

    return run_experiment


def _get_search_algorithm(
    output_folder: Path, num_gpus_per_trial: int, num_cpus_per_trial: int
):
    client = AxClient(enforce_sequential_optimization=False)

    snapshot_path = output_folder / "ax_client_snapshot.json"
    if snapshot_path.exists():
        logger.info(
            "Found snapshot file %s, resuming hyperoptimization from previous state.",
            snapshot_path,
        )
        client = client.load_from_json_file(filepath=str(snapshot_path))

    else:
        parameter_constraints = _get_parameter_constraints(search_space=SEARCH_SPACE)
        client.create_experiment(
            name="hyperopt_experiment",
            parameters=SEARCH_SPACE,
            objective_name="best_average_performance",
            minimize=False,
            parameter_constraints=parameter_constraints,
        )

    max_concurrent = _get_max_concurrent_runs(
        gpus_per_trial=num_gpus_per_trial, cpus_per_trial=num_cpus_per_trial
    )
    search_algorithm = AxSearch(client, max_concurrent=max_concurrent)

    return client, search_algorithm


def _get_parameter_constraints(search_space: List[Dict[str, Any]],):
    constraints = []
    hparams = [i["name"] for i in search_space]
    if "down_stride" in hparams and "kernel_width" in hparams:
        constraints.append("down_stride <= kernel_width")

    if "first_stride_expansion" in hparams and "first_kernel_expansion" in hparams:
        constraints.append("first_stride_expansion <= first_kernel_expansion")

    return constraints


def _get_max_concurrent_runs(gpus_per_trial: int, cpus_per_trial: int):
    num_cpus_total = cpu_count()
    max_concurrent_on_cpus = min(num_cpus_total // cpus_per_trial, 10)

    if gpus_per_trial == 0:
        logger.debug(
            "No GPUs allocated to each trial. Max concurrent trials set to %d"
            " based on %d available CPUs with %d CPUs per trial. Hardcoded max is 10.",
            max_concurrent_on_cpus,
            num_cpus_total,
            cpus_per_trial,
        )
        return max_concurrent_on_cpus

    num_gpus_total = cuda.device_count()
    max_concurrent_on_gpus = num_gpus_total // gpus_per_trial

    concurrent_bottleneck = min(max_concurrent_on_cpus, max_concurrent_on_gpus)
    logger.debug(
        "Max concurrent trials set to %d based on %d available GPUs with %d GPUs "
        "per trial and %d available CPUs with %d CPUs per trial. Harcoded max is 10.",
        concurrent_bottleneck,
        num_gpus_total,
        gpus_per_trial,
        num_cpus_total,
        cpus_per_trial,
    )

    return concurrent_bottleneck


def _get_scheduler(grace_period: int):
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="best_average_performance",
        mode="max",
        grace_period=grace_period,
    )
    return scheduler


def _hyperopt_run_finalizer(ax_client: AxClient, output_folder: Path) -> None:
    _analyse_ax_results(ax_client=ax_client, output_folder=output_folder)

    snapshot_outpath = output_folder / "ax_client_snapshot.json"
    ax_client.save_to_json_file(filepath=str(snapshot_outpath))


def _analyse_ax_results(ax_client: AxClient, output_folder: Path) -> None:
    _save_trials_plot(
        experiment_object=ax_client.experiment, outpath=output_folder / "trials.html"
    )

    model = ax_client.generation_strategy.model
    if not isinstance(model, RandomModelBridge):
        for param in SEARCH_SPACE:
            if param["type"] == "range":
                param_name = param["name"]
                cur_outpath = output_folder / f"{param_name}_slice.html"
                _save_slice_plot(
                    model_object=model,
                    parameter_name=param_name,
                    metric_name="best_average_performance",
                    outpath=cur_outpath,
                )


def _save_trials_plot(experiment_object: Experiment, outpath: Path):
    """
    We filter for completed trials only in case the program is terminated only. We do
    not want uncompleted trials to be considered because their objective_mean is None.
    """
    completed_trials = _get_completed_ax_trials(trials=experiment_object.trials)

    best_objectives = np.array(
        [[trial.objective_mean * 100 for trial in completed_trials.values()]]
    )
    best_objective_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="Classification Accuracy, %",
    )

    _save_plot_from_ax_plot_config(plot_config=best_objective_plot, outpath=outpath)


def _get_completed_ax_trials(trials: Dict[int, BaseTrial]):
    completed_trials = {
        k: v for k, v in trials.items() if v.status == TrialStatus.COMPLETED
    }

    return completed_trials


def _save_slice_plot(
    model_object: ModelBridge, parameter_name: str, metric_name: str, outpath: Path
) -> None:
    slice_object = plot_slice(
        model=model_object, param_name=parameter_name, metric_name=metric_name
    )

    _save_plot_from_ax_plot_config(plot_config=slice_object, outpath=outpath)


def _save_plot_from_ax_plot_config(plot_config: AxPlotConfig, outpath: Path) -> None:
    data = plot_config[0]["data"]
    layout = plot_config[0]["layout"]

    fig = {"data": data, "layout": layout}

    offline.plot(figure_or_data=fig, filename=str(outpath), auto_open=False)


def run_hyperopt(cl_args: Namespace) -> ExperimentAnalysis:
    output_folder = Path(cl_args.output_folder)
    experiment_func = get_experiment_func()

    ax_client, ax_search_algorithm = _get_search_algorithm(
        output_folder=output_folder,
        num_gpus_per_trial=cl_args.n_gpus_per_trial,
        num_cpus_per_trial=cl_args.n_cpus_per_trial,
    )
    atexit.register(
        partial(
            _hyperopt_run_finalizer, ax_client=ax_client, output_folder=output_folder
        )
    )

    scheduler = _get_scheduler(grace_period=cl_args.scheduler_grace_period)

    loggers = [i for i in DEFAULT_LOGGERS if i != TBXLogger]
    run_analysis = tune.run(
        run_or_experiment=experiment_func,
        local_dir=Path(cl_args.output_folder),
        loggers=loggers,
        scheduler=scheduler,
        search_alg=ax_search_algorithm,
        num_samples=cl_args.total_trials,
        resources_per_trial={
            "gpu": cl_args.n_gpus_per_trial,
            "cpu": cl_args.n_cpus_per_trial,
        },
    )

    return run_analysis


def _parse_cl_args(cl_args: Namespace) -> Namespace:
    if cuda.device_count() == 0:
        logger.debug("Setting n_gpus_per_trial to 0 since device cound is 0.")
        cl_args.n_gpus_per_trial = 0

    return cl_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--total_trials", type=int, default=20)

    parser.add_argument("--n_gpus_per_trial", type=int, default=1)

    parser.add_argument("--n_cpus_per_trial", type=int, default=8)

    parser.add_argument("--scheduler_grace_period", type=int, default=10)

    parser.add_argument("--output_folder", type=str, required=True)

    cur_cl_args = parser.parse_args()

    parsed_cl_args = _parse_cl_args(cl_args=cur_cl_args)

    analysis = run_hyperopt(cl_args=parsed_cl_args)
