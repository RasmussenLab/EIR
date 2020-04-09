import argparse
import atexit
import uuid
from argparse import Namespace
from functools import partial
from os import cpu_count
from os.path import abspath
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from aislib.misc_utils import get_logger
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.trial import BaseTrial
from ax.core.types import TParamValue
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.random import RandomModelBridge
from ax.plot.base import AxPlotConfig
from ax.plot.slice import plot_slice
from ax.plot.trace import optimization_trace_single_method
from ax.service.ax_client import AxClient
from plotly import offline
from ray import tune
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.logger import DEFAULT_LOGGERS, TBXLogger
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.ax import AxSearch
from torch import cuda
from yaml import load, Loader

from human_origins_supervised import train
from human_origins_supervised.data_load.datasets import merge_target_columns
from human_origins_supervised.train_utils.metrics import (
    get_best_average_performance,
    get_metrics_files,
)
from human_origins_supervised.train_utils.utils import get_run_folder

# aliases
al_search_space = List[Dict[str, Union[TParamValue, List[TParamValue]]]]

logger = get_logger(name=__name__, tqdm_compatible=True)


def _get_default_train_cl_args(train_config_file_for_hyperopt: str) -> Namespace:
    train_argument_parser = train.get_train_argument_parser()
    cl_args_base = train_argument_parser.parse_args(
        args=["--config_file", abspath(train_config_file_for_hyperopt)]
    )
    cl_args_base = train.modify_train_arguments(cl_args=cl_args_base)
    setattr(cl_args_base, "no_pbar", True)

    base_cl_args_with_abspaths = _convert_filepaths_to_abspaths(cl_args=cl_args_base)
    return base_cl_args_with_abspaths


def _convert_filepaths_to_abspaths(cl_args: Namespace) -> Namespace:
    """
    This is needed since ray is in a different context.
    """
    keys = ["custom_lib", "snp_file", "data_folder", "label_file"]

    for file_key in keys:
        cur_file_path = getattr(cl_args, file_key)

        if cur_file_path is not None:
            if not Path(cur_file_path).exists():
                raise FileNotFoundError(f"Could not find file: {cur_file_path}")

            setattr(cl_args, file_key, abspath(cur_file_path))

    return cl_args


def get_experiment_func(train_config_file_for_hyperopt: str):
    """
    The abspath is needed since ray creates its own context.
    """
    train_cl_args_base = _get_default_train_cl_args(
        train_config_file_for_hyperopt=abspath(train_config_file_for_hyperopt)
    )

    def run_experiment(parametrization):
        train_cl_args = _parametrize_base_cl_args(
            train_cl_args_base=train_cl_args_base, parametrization=parametrization
        )

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


def _parametrize_base_cl_args(
    train_cl_args_base: Namespace, parametrization: Dict
) -> Namespace:
    """
    We update the sample interval to make sure we sample based on number of samples
    seen.
    """
    config_ = {**vars(train_cl_args_base), **parametrization}

    config_["sample_interval"] = int(
        32 / config_["batch_size"] * config_["sample_interval"]
    )

    current_run_name = config_["run_name"] + "_" + str(uuid.uuid4())
    config_["run_name"] = current_run_name

    cl_args = Namespace(**config_)

    return cl_args


def _get_search_algorithm(
    search_space: al_search_space,
    output_folder: Path,
    num_gpus_per_trial: int,
    num_cpus_per_trial: int,
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
        parameter_constraints = _get_parameter_constraints(search_space=search_space)
        client.create_experiment(
            name="hyperopt_experiment",
            parameters=search_space,
            objective_name="best_average_performance",
            minimize=False,
            parameter_constraints=parameter_constraints,
        )

    max_concurrent = _get_max_concurrent_runs(
        gpus_per_trial=num_gpus_per_trial, cpus_per_trial=num_cpus_per_trial
    )
    search_algorithm = AxSearch(client, max_concurrent=max_concurrent)

    return client, search_algorithm


def _get_parameter_constraints(search_space: al_search_space):
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


def _hyperopt_run_finalizer(
    ax_client: AxClient, output_folder: Path, search_space: al_search_space
) -> None:
    _analyse_ax_results(
        ax_client=ax_client, output_folder=output_folder, search_space=search_space
    )

    snapshot_outpath = output_folder / "ax_client_snapshot.json"
    ax_client.save_to_json_file(filepath=str(snapshot_outpath))


def _analyse_ax_results(
    ax_client: AxClient, output_folder: Path, search_space: al_search_space
) -> None:
    _save_trials_plot(
        experiment_object=ax_client.experiment, outpath=output_folder / "trials.html"
    )

    model = ax_client.generation_strategy.model
    if not isinstance(model, RandomModelBridge):
        for param in search_space:
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
    experiment_func = get_experiment_func(
        train_config_file_for_hyperopt=abspath(cl_args.train_config_file)
    )

    search_space = _load_search_space_yaml_config(
        search_space_config_file_path=cl_args.search_space_file
    )
    ax_client, ax_search_algorithm = _get_search_algorithm(
        search_space=search_space,
        output_folder=output_folder,
        num_gpus_per_trial=cl_args.n_gpus_per_trial,
        num_cpus_per_trial=cl_args.n_cpus_per_trial,
    )
    atexit.register(
        partial(
            _hyperopt_run_finalizer,
            ax_client=ax_client,
            output_folder=output_folder,
            search_space=search_space,
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


def _load_search_space_yaml_config(
    search_space_config_file_path: str,
) -> al_search_space:
    stream = open(search_space_config_file_path, "r")
    search_space = load(stream=stream, Loader=Loader)
    breakpoint()

    return search_space


def _parse_cl_args(cl_args: Namespace) -> Namespace:
    if cuda.device_count() == 0:
        logger.debug("Setting n_gpus_per_trial to 0 since device count is 0.")
        cl_args.n_gpus_per_trial = 0

    return cl_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--total_trials", type=int, default=20)

    parser.add_argument("--n_gpus_per_trial", type=int, default=1)

    parser.add_argument("--n_cpus_per_trial", type=int, default=8)

    parser.add_argument("--scheduler_grace_period", type=int, default=10)

    parser.add_argument(
        "--search_space_file",
        type=str,
        default="config/base_search_space.yaml",
        help=".yaml file indicating the search space to use in the hyperparameter "
        "optimization, follows ax-platform format",
    )

    parser.add_argument(
        "--train_config_file",
        type=str,
        required=True,
        help="path to .yaml file specifying experiment specific settings to be used "
        "as CL arguments to train.py. This needs to contain at least data_folder, "
        "label_file, run_name and target column (con and/or cat) arguments as "
        "specified in train.py.",
    )

    parser.add_argument("--output_folder", type=str, required=True)

    cur_cl_args = parser.parse_args()

    parsed_cl_args = _parse_cl_args(cl_args=cur_cl_args)

    analysis = run_hyperopt(cl_args=parsed_cl_args)
