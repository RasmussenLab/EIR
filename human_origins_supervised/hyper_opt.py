import uuid
from argparse import Namespace
from pathlib import Path

import numpy as np
from aislib.misc_utils import get_logger
from ax.core.experiment import Experiment
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.random import RandomModelBridge
from ax.plot.base import AxPlotConfig
from ax.plot.slice import plot_slice
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from plotly import offline
from torch import cuda

from human_origins_supervised.data_load.datasets import merge_target_columns
from human_origins_supervised.train import main
from human_origins_supervised.train_utils.metrics import (
    get_best_average_performance,
    get_metrics_files,
)
from human_origins_supervised.train_utils.utils import get_run_folder

logger = get_logger(name=__name__, tqdm_compatible=True)

config_base = {
    "act_classes": None,
    "b1": 0.9,
    "b2": 0.999,
    "batch_size": 32,
    "channel_exp_base": 5,
    "checkpoint_interval": 100,
    "extra_con_columns": [],
    "custom_lib": None,
    "data_folder": "data/1240k_HO_2019/processed/encoded_outputs_uint8/2000/train/",
    "data_width": 1000,
    "debug": False,
    "device": "cuda:0" if cuda.is_available() else "cpu",
    "down_stride": 4,
    "extra_cat_columns": [],
    "fc_repr_dim": 64,
    "fc_task_dim": 32,
    "fc_do": 0.0,
    "find_lr": False,
    "first_kernel_expansion": 1,
    "first_stride_expansion": 1,
    "get_acts": False,
    "gpu_num": "0",
    "kernel_width": 12,
    "label_file": "data/1240k_HO_2019/processed/labels/labels_2000.csv",
    "lr": 1e-2,
    "lr_lb": 1e-5,
    "lr_schedule": "plateau",
    "memory_dataset": False,
    "model_type": "cnn",
    "multi_gpu": False,
    "n_cpu": 8,
    "n_epochs": 8,
    "na_augment_perc": 0.0,
    "na_augment_prob": 0.0,
    "optimizer": "adamw",
    "plot_skip_steps": 50,
    "rb_do": 0.0,
    "resblocks": None,
    "run_name": "test_run",
    "sa": False,
    "snp_file": "data/1240k_HO_2019/processed/parsed_files/2000/data_final.snp",
    "sample_interval": 100,
    "target_cat_columns": ["Origin"],
    "target_con_columns": [],
    "target_width": 1000,
    "valid_size": 0.05,
    "warmup_steps": 144,
    "wd": 0.00,
    "weighted_sampling_column": None,
}


def get_experiment_func(output_folder: Path):
    def run_experiment(parametrization):

        config_ = {**config_base, **parametrization}
        current_run_name = "test_run_" + str(uuid.uuid4())
        config_["run_name"] = str(output_folder / current_run_name)

        config = Namespace(**config_)

        main(cl_args=config)

        target_columns = merge_target_columns(
            target_cat_columns=config.target_cat_columns,
            target_con_columns=config.target_con_columns,
        )
        run_folder = get_run_folder(run_name=config.run_name)
        metrics_files = get_metrics_files(
            target_columns=target_columns, run_folder=run_folder, target_prefix="v_"
        )

        best_performance = get_best_average_performance(
            val_metrics_files=metrics_files, target_columns=target_columns
        )
        logger.info("Best performance: %f", best_performance)

        return best_performance

    return run_experiment


def _save_trials_plot(experiment_object: Experiment, outpath: Path):
    best_objectives = np.array(
        [[trial.objective_mean * 100 for trial in experiment_object.trials.values()]]
    )
    best_objective_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="Classification Accuracy, %",
    )

    _save_plot_from_ax_plot_config(plot_config=best_objective_plot, outpath=outpath)


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


if __name__ == "__main__":

    experiment_func = get_experiment_func(output_folder=Path("hyperopt"))

    parameters = [
        {
            "name": "batch_size",
            "type": "choice",
            "values": [16, 32, 64],
            "is_ordered": True,
        },
        {
            "name": "lr",
            "type": "range",
            "bounds": [1e-5, 0.5],
            "log_scale": True,
            "digits": 2,
        },
        {
            "name": "lr_schedule",
            "type": "choice",
            "values": ["cycle", "plateau", "same"],
        },
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
        {"name": "na_augment_perc", "type": "range", "bounds": [0.0, 0.9], "digits": 2},
        {"name": "na_augment_prob", "type": "range", "bounds": [0.0, 1.0], "digits": 2},
        {"name": "fc_do", "type": "range", "bounds": [0.0, 0.9], "digits": 2},
        {"name": "rb_do", "type": "range", "bounds": [0.0, 0.9], "digits": 2},
        {
            "name": "kernel_width",
            "type": "range",
            "bounds": [2, 20],
            "parameter_type": int,
        },
        {
            "name": "first_kernel_expansion",
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
            "bounds": [2, 6],
            "parameter_type": int,
        },
    ]

    total_trials = 20
    best_parameters, values, experiment, model = optimize(
        total_trials=total_trials,
        parameters=parameters,
        evaluation_function=experiment_func,
        parameter_constraints=[
            "first_stride_expansion <= first_kernel_expansion",
            "down_stride <= kernel_width",
        ],
        experiment_name="test",
        objective_name="best_average_performance",
    )

    logger.info("Best parameters: %s", best_parameters)

    _save_trials_plot(
        experiment_object=experiment, outpath=Path("runs/hyperopt/trials.html")
    )

    if total_trials > 5 and not isinstance(model, RandomModelBridge):
        for param in parameters:
            if param["type"] == "range":
                param_name = param["name"]
                _save_slice_plot(
                    model_object=model,
                    parameter_name=param_name,
                    metric_name="best_average_performance",
                    outpath=Path(f"runs/hyperopt/{param_name}_slice.html"),
                )
