import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, TYPE_CHECKING, List, Tuple, Callable, Union

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists, get_logger
from scipy.stats import pearsonr
from sklearn.metrics import (
    matthews_corrcoef,
    r2_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from torch.utils.tensorboard import SummaryWriter

from human_origins_supervised.data_load.data_utils import get_target_columns_generator

if TYPE_CHECKING:
    from human_origins_supervised.train import (  # noqa: F401
        al_criterions,
        al_averaging_functions_dict,
    )
    from human_origins_supervised.train_utils.train_handlers import HandlerConfig
    from human_origins_supervised.data_load.label_setup import (  # noqa: F401
        al_target_columns,
        al_label_transformers_object,
    )

# aliases
al_step_metric_dict = Dict[str, Dict[str, float]]
al_metric_record_dict = Dict[
    str, Union[Tuple["MetricRecord", ...], "al_averaging_functions_dict"]
]

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass()
class MetricRecord:
    name: str
    function: Callable
    only_val: bool = False
    minimize_goal: bool = False


def calculate_batch_metrics(
    target_columns: "al_target_columns",
    losses: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    mode: str,
    metric_record_dict: al_metric_record_dict,
) -> al_step_metric_dict:
    """
    """
    assert mode in ["val", "train"]

    target_columns_gen = get_target_columns_generator(target_columns)

    master_metric_dict = {}

    for column_type, column_name in target_columns_gen:
        cur_metric_dict = {}

        cur_metric_records: Tuple[MetricRecord, ...] = metric_record_dict[column_type]
        cur_outputs = outputs[column_name].detach().cpu().numpy()
        cur_labels = labels[column_name].cpu().numpy()

        for metric_record in cur_metric_records:

            if metric_record.only_val and mode == "train":
                continue

            cur_key = f"{column_name}_{metric_record.name}"
            cur_metric_dict[cur_key] = metric_record.function(
                outputs=cur_outputs, labels=cur_labels, column_name=column_name
            )

        cur_metric_dict[f"{column_name}_loss"] = losses[column_name].item()

        master_metric_dict[column_name] = cur_metric_dict

    return master_metric_dict


def add_multi_task_average_metrics(
    batch_metrics_dict: al_step_metric_dict,
    target_columns: "al_target_columns",
    loss: float,
    performance_average_functions: Dict[str, Callable[[al_step_metric_dict], float]],
):
    average_performance = average_performances_across_tasks(
        metric_dict=batch_metrics_dict,
        target_columns=target_columns,
        performance_calculation_functions=performance_average_functions,
    )
    batch_metrics_dict["average"] = {
        "loss-average": loss,
        "perf-average": average_performance,
    }

    return batch_metrics_dict


def average_performances_across_tasks(
    metric_dict: al_step_metric_dict,
    target_columns: "al_target_columns",
    performance_calculation_functions: Dict[
        str, Callable[[al_step_metric_dict], float]
    ],
) -> float:
    target_columns_gen = get_target_columns_generator(target_columns)

    all_metrics = []

    for column_type, column_name in target_columns_gen:
        cur_metric_func = performance_calculation_functions.get(column_type)

        metric_func_args = {"metric_dict": metric_dict, "column_name": column_name}
        cur_value = cur_metric_func(**metric_func_args)
        all_metrics.append(cur_value)

        all_metrics.append(cur_value)

    average = np.array(all_metrics).mean()

    return average


def calc_mcc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    pred = np.argmax(a=outputs, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mcc = matthews_corrcoef(labels, pred)

    return mcc


def calc_roc_auc_ovr(
    outputs: np.ndarray, labels: np.ndarray, average: str = "macro", *args, **kwargs
) -> float:
    assert average in ["micro", "macro"]

    if outputs.shape[1] > 2:
        labels = label_binarize(y=labels, classes=sorted(np.unique(labels)))
    else:
        outputs = outputs[:, 1]

    roc_auc = roc_auc_score(y_true=labels, y_score=outputs, average=average)
    return roc_auc


def calc_average_precision_ovr(
    outputs: np.ndarray, labels: np.ndarray, average: str = "macro", *args, **kwargs
) -> float:

    assert average in ["micro", "macro"]

    labels_bin = label_binarize(y=labels, classes=sorted(np.unique(labels)))
    if outputs.shape[1] == 2:
        outputs = outputs[:, 1]

    average_precision = average_precision_score(
        y_true=labels_bin, y_score=outputs, average=average
    )

    return average_precision


def calc_acc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    pred = np.argmax(outputs, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return accuracy


def calc_pcc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:

    if len(outputs) < 2:
        return 0.0

    pcc = pearsonr(x=labels.squeeze(), y=outputs.squeeze())[0]
    return pcc


def calc_r2(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:

    if len(outputs) < 2:
        return 0.0

    r2 = r2_score(y_true=labels, y_pred=outputs)
    return r2


def calc_rmse(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    target_transformers: Dict[str, StandardScaler],
    column_name: str,
    *args,
    **kwargs,
) -> float:
    cur_target_transformer = target_transformers[column_name]

    labels = cur_target_transformer.inverse_transform(labels).squeeze()
    preds = cur_target_transformer.inverse_transform(outputs).squeeze()

    rmse = np.sqrt(mean_squared_error(y_true=labels, y_pred=preds))
    return rmse


def calculate_losses(
    criterions: "al_criterions",
    labels: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    losses_dict = {}

    for target_column, criterion in criterions.items():
        cur_target_col_labels = labels[target_column]
        cur_target_col_outputs = outputs[target_column]
        losses_dict[target_column] = criterion(
            input=cur_target_col_outputs, target=cur_target_col_labels
        )

    return losses_dict


def aggregate_losses(losses_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    losses_values = list(losses_dict.values())
    average_loss = torch.mean(torch.stack(losses_values))

    return average_loss


def get_extra_loss_term_functions(model, l1_weight: float) -> List[Callable]:
    extra_loss_funcs = []

    def add_l1_loss(*args, **kwargs):
        l1_loss = torch.norm(model.l1_penalized_weights, p=1) * l1_weight
        return l1_loss

    if l1_weight > 0.0:
        if not hasattr(model, "l1_penalized_weights"):
            raise AttributeError(
                f"Model {model} does not have attribute 'l1_penalized_weights' which is"
                f" required to calculate L1 loss with passed in {l1_weight} L1 weight."
            )
        logger.debug(
            "Penalizing weights of shape %s with L1 loss with weight %f.",
            model.l1_penalized_weights.shape,
            l1_weight,
        )
        extra_loss_funcs.append(add_l1_loss)

    return extra_loss_funcs


def add_extra_losses(total_loss: torch.Tensor, extra_loss_functions: List[Callable]):
    """
    TODO: Possibly add inputs and labels as arguments here if needed later.
    """
    for loss_func in extra_loss_functions:
        total_loss += loss_func()

    return total_loss


def persist_metrics(
    handler_config: "HandlerConfig",
    metrics_dict: "al_step_metric_dict",
    iteration: int,
    write_header: bool,
    prefixes: Dict[str, str],
):

    hc = handler_config
    c = handler_config.config
    cl_args = c.cl_args

    metrics_files = get_metrics_files(
        target_columns=c.target_columns,
        run_folder=hc.run_folder,
        train_or_val_target_prefix=f"{prefixes['metrics']}",
    )

    if write_header:
        _ensure_metrics_paths_exists(metrics_files=metrics_files)

    for metrics_name, metrics_history_file in metrics_files.items():
        cur_metric_dict = metrics_dict[metrics_name]

        _add_metrics_to_writer(
            name=f"{prefixes['writer']}/{metrics_name}",
            metric_dict=cur_metric_dict,
            iteration=iteration,
            writer=c.writer,
            plot_skip_steps=cl_args.plot_skip_steps,
        )

        _append_metrics_to_file(
            filepath=metrics_history_file,
            metrics=cur_metric_dict,
            iteration=iteration,
            write_header=write_header,
        )


def get_metrics_files(
    target_columns: "al_target_columns",
    run_folder: Path,
    train_or_val_target_prefix: str,
) -> Dict[str, Path]:
    assert train_or_val_target_prefix in ["validation_", "train_"]

    all_target_columns = target_columns["con"] + target_columns["cat"]

    path_dict = {}
    for target_column in all_target_columns:
        cur_fname = train_or_val_target_prefix + target_column + "_history.log"
        cur_path = Path(run_folder, "results", target_column, cur_fname)
        path_dict[target_column] = cur_path

    average_loss_training_metrics_file = get_average_history_filepath(
        run_folder=run_folder, train_or_val_target_prefix=train_or_val_target_prefix
    )
    path_dict["average"] = average_loss_training_metrics_file

    return path_dict


def get_average_history_filepath(
    run_folder: Path, train_or_val_target_prefix: str
) -> Path:
    assert train_or_val_target_prefix in ["validation_", "train_"]
    metrics_file_path = run_folder / f"{train_or_val_target_prefix}average_history.log"
    return metrics_file_path


def _ensure_metrics_paths_exists(metrics_files: Dict[str, Path]) -> None:
    for path in metrics_files.values():
        ensure_path_exists(path)


def _add_metrics_to_writer(
    name: str,
    metric_dict: Dict[str, float],
    iteration: int,
    writer: SummaryWriter,
    plot_skip_steps: int,
    do_writing_divisor: int = 10,
) -> None:
    """
    We do %10 to reduce the amount of training data going to tensorboard, otherwise
    it slows down with many large experiments.

    TODO:
        Have better logic here – this can cause unexpected behavior where nothing
        gets written for the evaluation parts if the sample interval is not divisible
        by 10. Either remove the modulus or pass it in as an argument, then 10 for
        training.
    """

    if iteration >= plot_skip_steps and iteration % do_writing_divisor == 0:
        for metric_name, metric_value in metric_dict.items():
            cur_name = name + f"/{metric_name}"
            writer.add_scalar(
                tag=cur_name, scalar_value=metric_value, global_step=iteration
            )


def _append_metrics_to_file(
    filepath: Path, metrics: Dict[str, float], iteration: int, write_header=False
):
    with open(str(filepath), "a") as logfile:
        fieldnames = ["iteration"] + sorted(metrics.keys())
        writer = csv.DictWriter(logfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        dict_to_write = {**{"iteration": iteration}, **metrics}
        writer.writerow(dict_to_write)


def read_metrics_history_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col="iteration")

    return df


def get_metrics_dataframes(
    results_dir: Path, target_string: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_history_path = read_metrics_history_file(
        results_dir / f"train_{target_string}_history.log"
    )
    valid_history_path = read_metrics_history_file(
        results_dir / f"validation_{target_string}_history.log"
    )

    return train_history_path, valid_history_path
