from pathlib import Path
from typing import Dict, Sequence, TYPE_CHECKING, List
from collections import defaultdict
from textwrap import wrap


import torch
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from snp_pred.interpretation.interpret_omics import _get_target_class_name

if TYPE_CHECKING:
    from snp_pred.train import Config
    from snp_pred.data_load.label_setup import al_label_transformers_object
    from snp_pred.interpretation.interpretation import SampleActivation


def analyze_tabular_input_activations(
    config: "Config",
    input_name: str,
    target_column_name: str,
    target_column_type: str,
    activation_outfolder: Path,
    all_activations: Sequence["SampleActivation"],
):

    tabular_model = config.model.modules_to_fuse[input_name]
    activation_tensor_slices = set_up_tabular_tensor_slices(
        cat_input_columns=config.cl_args.extra_cat_columns,
        con_input_columns=config.cl_args.extra_con_columns,
        embedding_module=tabular_model,
    )

    parsed_activations = parse_tabular_activations(
        activation_tensor_slices=activation_tensor_slices,
        all_activations=all_activations,
        input_name=input_name,
    )

    df_activations = get_tabular_activation_df(parsed_activations=parsed_activations)

    plot_tabular_activations(
        df_activations=df_activations, activation_outfolder=activation_outfolder
    )

    all_activations_class_stratified = stratify_activations_by_target_classes(
        all_activations=all_activations,
        target_transformer=config.target_transformers[target_column_name],
        target_column=target_column_name,
        column_type=target_column_type,
    )

    for class_name, class_activations in all_activations_class_stratified.items():

        cat_to_con_cutoff = get_cat_to_con_cutoff_from_slices(
            slices=activation_tensor_slices,
            cat_input_columns=config.cl_args.extra_cat_columns,
        )
        continuous_shap = _gather_continuous_shap_values(
            all_activations=class_activations,
            cat_to_con_cutoff=cat_to_con_cutoff,
            input_name=input_name,
        )
        continuous_inputs = _gather_continuous_inputs(
            all_activations=class_activations,
            cat_to_con_cutoff=cat_to_con_cutoff,
            con_names=config.cl_args.extra_con_columns,
            input_name=input_name,
        )
        plot_tabular_beeswarm(
            shap_values=continuous_shap,
            features=continuous_inputs,
            activation_outfolder=activation_outfolder,
            class_name=class_name,
        )


def set_up_tabular_tensor_slices(
    cat_input_columns: Sequence[str],
    con_input_columns: Sequence[str],
    embedding_module: torch.nn.Module,
) -> Dict[str, slice]:
    """
    TODO: Probably better to pass in Dict[column_name, nn.Embedding]
    """
    slices = {}

    current_index = 0
    for cat_column in cat_input_columns:
        cur_embedding = getattr(embedding_module, "embed_" + cat_column)
        cur_embedding_dim = cur_embedding.embedding_dim

        slice_start = current_index
        slice_end = current_index + cur_embedding_dim
        slices[cat_column] = slice(slice_start, slice_end)

        current_index = slice_end

    for con_column in con_input_columns:
        slice_start = current_index
        slice_end = slice_start + 1

        slices[con_column] = slice(slice_start, slice_end)
        current_index = slice_end

    return slices


def stratify_activations_by_target_classes(
    all_activations: Sequence["SampleActivation"],
    target_transformer: "al_label_transformers_object",
    target_column: str,
    column_type: str,
) -> Dict[str, Sequence["SampleActivation"]]:
    all_activations_class_stratified = defaultdict(list)

    for sample in all_activations:
        cur_label_name = _get_target_class_name(
            sample_label=sample.sample_info.target_labels[target_column],
            target_transformer=target_transformer,
            column_type=column_type,
            target_column_name=target_column,
        )
        all_activations_class_stratified[cur_label_name].append(sample)

    return all_activations_class_stratified


def get_cat_to_con_cutoff_from_slices(
    slices: Dict[str, slice], cat_input_columns: Sequence[str]
) -> int:
    cutoff = 0

    for cat_column in cat_input_columns:
        cur_cat_slice = slices[cat_column]
        slice_size = cur_cat_slice.stop - cur_cat_slice.start
        cutoff += slice_size

    return cutoff


def parse_tabular_activations(
    activation_tensor_slices: Dict,
    all_activations: Sequence["SampleActivation"],
    input_name: str,
) -> Dict[str, List[float]]:
    column_activations = {column: [] for column in activation_tensor_slices.keys()}

    for sample_activation in all_activations:
        sample_acts = sample_activation.sample_activations[input_name].squeeze()

        for column in activation_tensor_slices.keys():
            cur_slice = activation_tensor_slices[column]
            cur_column_activations = sample_acts[cur_slice].sum()
            column_activations[column].append(cur_column_activations)

    finalized_activations = {}

    for column, aggregated_activations in column_activations.items():
        mean_activation = np.abs(np.array(column_activations[column]).mean())
        finalized_activations[column] = [mean_activation]

    return finalized_activations


def get_tabular_activation_df(
    parsed_activations: Dict[str, List[float]]
) -> pd.DataFrame:
    df_activations = pd.DataFrame.from_dict(
        parsed_activations, orient="index", columns=["Shap_Value"]
    )
    df_activations = df_activations.sort_values(by="Shap_Value", ascending=False)

    return df_activations


def plot_tabular_activations(
    df_activations: pd.DataFrame, activation_outfolder: Path
) -> None:

    sns_plot = sns.barplot(
        x=df_activations["Shap_Value"],
        y=df_activations.index,
        palette="Blues_d",
    )
    plt.tight_layout()
    sns_figure = sns_plot.get_figure()
    sns_figure.set_size_inches(10, 0.5 * len(df_activations))
    sns_figure.savefig(
        str(activation_outfolder / "feature_importance.png"), bbox_inches="tight"
    )

    plt.close("all")


def _gather_continuous_inputs(
    all_activations: Sequence["SampleActivation"],
    cat_to_con_cutoff: int,
    con_names: Sequence[str],
    input_name: str,
) -> pd.DataFrame:
    con_inputs = []

    for sample in all_activations:
        cur_full_input = sample.sample_info.inputs[input_name]
        cur_con_input_part = cur_full_input.squeeze()[cat_to_con_cutoff:]
        con_inputs.append(cur_con_input_part)

    con_inputs_array = np.array([np.array(i.cpu()) for i in con_inputs])
    df = pd.DataFrame(con_inputs_array, columns=con_names)

    return df


def _gather_continuous_shap_values(
    all_activations: Sequence["SampleActivation"],
    cat_to_con_cutoff: int,
    input_name: str,
) -> np.ndarray:
    """
    We need to expand_dims in the case where we only have one input value, otherwise
    shap will not accept a vector.
    """
    con_acts = []

    for sample in all_activations:
        cur_full_input = sample.sample_activations[input_name]
        cur_con_input_part = cur_full_input.squeeze()[cat_to_con_cutoff:]
        con_acts.append(cur_con_input_part)

    array_squeezed = np.array(con_acts).squeeze()

    if len(array_squeezed.shape) == 1:
        return np.expand_dims(array_squeezed, 1)

    return array_squeezed


def plot_tabular_beeswarm(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    activation_outfolder: Path,
    class_name: str,
):

    _ = shap.summary_plot(
        shap_values=shap_values,
        features=features,
        show=False,
        title=class_name,
        max_display=20,
    )

    fig = plt.gcf()

    ax = _get_shap_main_axis(figure=fig)
    ax_wrapped_ytick_labels = _get_wrapped_ax_ytick_text(axis=ax)
    ax.set_yticklabels(ax_wrapped_ytick_labels)

    plt.title(class_name, fontsize=14)
    plt.yticks(fontsize=10, wrap=True)

    plt.tight_layout()
    fig.savefig(
        str(activation_outfolder / f"con_features_beeswarm_{class_name}.png"),
        bbox_inches="tight",
    )

    plt.close("all")


def _get_shap_main_axis(figure: plt.Figure):
    return figure.axes[0]


def _get_wrapped_ax_ytick_text(axis: plt.Axes):
    labels = [item.get_text() for item in axis.get_yticklabels()]
    labels_wrapped = ["\n".join(wrap(label, 20)) for label in labels]

    return labels_wrapped
