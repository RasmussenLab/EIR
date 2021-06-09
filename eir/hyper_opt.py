import argparse
import atexit
import json
import uuid
from argparse import Namespace
from functools import partial
from os import cpu_count
from os.path import abspath
from pathlib import Path
from typing import Dict, List, Union

import ray

# See: https://github.com/ray-project/ray/issues/6573
ray._private.services.address_to_ip = lambda x: "127.0.0.1"
ray.init()

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
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.ax import AxSearch
from torch import cuda
from yaml import load, Loader

from eir import train, configuration


# aliases
al_search_space = List[Dict[str, Union[TParamValue, List[TParamValue]]]]

logger = get_logger(name=__name__, tqdm_compatible=True)


def _get_default_train_cl_args(train_config_file_for_hyperopt: str) -> Namespace:

    train_argument_parser = configuration.get_train_argument_parser()
    cl_args_base = train_argument_parser.parse_args(
        args=["--config_file", abspath(train_config_file_for_hyperopt)]
    )
    cl_args_base = configuration.modify_train_arguments(cl_args=cl_args_base)
    setattr(cl_args_base, "no_pbar", True)

    base_cl_args_with_abspaths = _convert_filepaths_to_abspaths(cl_args=cl_args_base)
    return base_cl_args_with_abspaths


def _convert_filepaths_to_abspaths(cl_args: Namespace) -> Namespace:
    """
    This is needed since ray is in a different context.
    """
    single_file_keys = ["snp_file", "label_file"]
    for file_key in single_file_keys:
        cur_file_path = getattr(cl_args, file_key)

        if cur_file_path is not None:
            if not Path(cur_file_path).exists():
                raise FileNotFoundError(f"Could not find file: {cur_file_path}")

            setattr(cl_args, file_key, abspath(cur_file_path))

    multi_file_keys = ["omics_sources"]
    for files_key in multi_file_keys:
        cur_file_paths = getattr(cl_args, files_key)
        absolute_paths = []

        for cur_file_path in cur_file_paths:
            if not Path(cur_file_path).exists():
                raise FileNotFoundError(f"Could not find file: {cur_file_path}")

            absolute_paths.append(abspath(cur_file_path))

        setattr(cl_args, files_key, absolute_paths)

    return cl_args


def get_experiment_func(train_config_file_for_hyperopt: str, hyperopt_output_dir: str):
    """
    The abspath is needed since ray creates its own context.

    Note that the average performance at each step is logged to ray inside the
    evaluation handler.

    See:
    https://docs.ray.io/en/master/tune/api_docs/trainable.html#function-api-checkpointing
    for details about checkpoint_dir argument.

    """
    train_cl_args_base = _get_default_train_cl_args(
        train_config_file_for_hyperopt=abspath(train_config_file_for_hyperopt)
    )

    def run_experiment(parametrization, checkpoint_dir: str = hyperopt_output_dir):
        train_cl_args = _parametrize_base_cl_args(
            train_cl_args_base=train_cl_args_base, parametrization=parametrization
        )

        default_hooks = train.get_default_hooks(cl_args_=train_cl_args)
        default_config = train.get_default_config(
            cl_args=train_cl_args, hooks=default_hooks
        )

        train.run_experiment(cl_args=train_cl_args, config=default_config)

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
    objective_name: str,
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
            objective_name=objective_name,
            minimize=False,
            parameter_constraints=parameter_constraints,
        )

    max_concurrent = _get_max_concurrent_runs(
        gpus_per_trial=num_gpus_per_trial, cpus_per_trial=num_cpus_per_trial
    )
    search_algorithm = AxSearch(ax_client=client)
    search_algorithm = ConcurrencyLimiter(
        search_algorithm, max_concurrent=max_concurrent
    )

    return client, search_algorithm


def get_objective_name_from_cl_argument(search_objective_cla: str):
    if search_objective_cla == "best":
        return "best_overall_performance"
    return "latest_average_performance"


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
        metric="latest_average_performance",
        mode="max",
        grace_period=grace_period,
    )
    return scheduler


def _hyperopt_run_finalizer(
    ax_client: AxClient,
    output_folder: Path,
    search_space: al_search_space,
    objective_name: str,
) -> None:
    """
    TODO: Convert arguments to this function to object.
    """
    _analyse_ax_results(
        ax_client=ax_client,
        output_folder=output_folder,
        search_space=search_space,
        objective_name=objective_name,
    )

    snapshot_outpath = output_folder / "ax_client_snapshot.json"
    ax_client.save_to_json_file(filepath=str(snapshot_outpath))

    best_result_outpath = output_folder / "best_parameters.json"
    best_params, values = ax_client.get_best_parameters()
    result_dict = {"best_params": best_params, "best_params_performance": values}
    with open(str(best_result_outpath), "w") as best_param_outpath:
        json.dump(result_dict, best_param_outpath, sort_keys=True, indent=4)


def _analyse_ax_results(
    ax_client: AxClient,
    output_folder: Path,
    search_space: al_search_space,
    objective_name: str,
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
                    metric_name=objective_name,
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
    """
    NOTE: Currently this plot will fail if we only have one parameter in the search
    space, related to how AX computes the slice values in slice.slice_config_to_trace.
    """
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
        train_config_file_for_hyperopt=abspath(cl_args.train_config_file),
        hyperopt_output_dir=cl_args.output_folder,
    )

    search_space = _load_search_space_yaml_config(
        search_space_config_file_path=cl_args.search_space_file
    )

    objective_name = get_objective_name_from_cl_argument(
        search_objective_cla=cl_args.search_objective
    )

    ax_client, ax_search_algorithm = _get_search_algorithm(
        search_space=search_space,
        output_folder=output_folder,
        objective_name=objective_name,
        num_gpus_per_trial=cl_args.n_gpus_per_trial,
        num_cpus_per_trial=cl_args.n_cpus_per_trial,
    )

    atexit.register(
        partial(
            _hyperopt_run_finalizer,
            ax_client=ax_client,
            output_folder=output_folder,
            search_space=search_space,
            objective_name=objective_name,
        )
    )

    scheduler = _get_scheduler(grace_period=cl_args.scheduler_grace_period)

    loggers = [i for i in DEFAULT_LOGGERS if i != TBXLogger]
    run_analysis = tune.run(
        run_or_experiment=experiment_func,
        local_dir=cl_args.output_folder,
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
        "--search_objective",
        type=str,
        choices=["final", "best"],
        default="final",
        help="Whether the search algorithm uses the final performance "
        "(i.e. last evaluation) or the peak / best performance as the reference"
        "when choosing new parameters.",
    )

    parser.add_argument(
        "--train_config_file",
        type=str,
        required=True,
        help="path to .yaml file specifying experiment specific settings to be used "
        "as CL arguments to train.py. This needs to contain at least data_source, "
        "label_file, run_name and target column (con and/or cat) arguments as "
        "specified in train.py.",
    )

    parser.add_argument("--output_folder", type=str, required=True)

    cur_cl_args = parser.parse_args()

    parsed_cl_args = _parse_cl_args(cl_args=cur_cl_args)

    analysis = run_hyperopt(cl_args=parsed_cl_args)
