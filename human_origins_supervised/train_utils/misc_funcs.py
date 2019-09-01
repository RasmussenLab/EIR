from typing import Dict

import torch
from sklearn.metrics import matthews_corrcoef, r2_score


def calc_multiclass_metrics(
    outputs: torch.Tensor, labels: torch.Tensor, prefix: str
) -> Dict[str, float]:
    _, pred = torch.max(outputs, 1)

    pred = pred.cpu().numpy()
    labels = labels.cpu().numpy()

    mcc = matthews_corrcoef(labels, pred)

    return {f"{prefix}_mcc": mcc}


def calc_regression_metrics(
    outputs: torch.Tensor, labels: torch.Tensor, prefix: str
) -> Dict[str, float]:
    train_pred = outputs.detach().cpu().numpy()
    train_labels = labels.cpu().numpy()

    r2 = r2_score(train_labels, train_pred)

    return {f"{prefix}_r2": r2}


def get_train_metrics(model_task):
    if model_task == "reg":
        return ["t_r2"]
    elif model_task == "cls":
        return ["t_mcc"]
