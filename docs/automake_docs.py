from itertools import chain
from typing import Iterable

from docs.doc_modules.a_using_eir import (
    a_basic_tutorial,
    b_tabular_tutorial,
    c_sequence_tutorial,
    d_pretrained_models_tutorial,
    e_image_tutorial,
    f_binary_tutorial,
    g_multimodal_tutorial,
    h_array_tutorial,
)
from docs.doc_modules.b_customizing_eir import a_customizing_fusion_tutorial
from docs.doc_modules.c_sequence_outputs import a_sequence_generation
from docs.doc_modules.experiments import AutoDocExperimentInfo, make_tutorial_data


def _get_a_using_eir_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_basic_tutorial.get_experiments()
    b_experiments = b_tabular_tutorial.get_experiments()
    c_experiments = c_sequence_tutorial.get_experiments()
    d_experiments = d_pretrained_models_tutorial.get_experiments()
    e_experiments = e_image_tutorial.get_experiments()
    f_experiments = f_binary_tutorial.get_experiments()
    g_experiments = g_multimodal_tutorial.get_experiments()
    h_experiments = h_array_tutorial.get_experiments()

    return chain(
        a_experiments,
        b_experiments,
        c_experiments,
        d_experiments,
        e_experiments,
        f_experiments,
        g_experiments,
        h_experiments,
    )


def _get_b_customizing_eir_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_customizing_fusion_tutorial.get_experiments()

    return chain(
        a_experiments,
    )


def _get_c_sequence_outputs_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_sequence_generation.get_experiments()

    return chain(
        a_experiments,
    )


if __name__ == "__main__":
    a_using_eir_experiments = _get_a_using_eir_experiments()
    b_customizing_eir_experiments = _get_b_customizing_eir_experiments()
    c_sequence_outputs_experiments = _get_c_sequence_outputs_experiments()

    experiment_iter = chain.from_iterable(
        [
            a_using_eir_experiments,
            c_sequence_outputs_experiments,
            b_customizing_eir_experiments,
        ]
    )
    for experiment in experiment_iter:
        make_tutorial_data(auto_doc_experiment_info=experiment)
