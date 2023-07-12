from copy import deepcopy
from pathlib import Path
from typing import Sequence, Union
from unittest.mock import patch

import numpy as np
import pytest
import torch

from eir.data_load.data_preparation_modules import prepare_omics


def test_prepare_genotype_array_train_mode():
    test_array = torch.zeros((4, 100), dtype=torch.uint8).detach().numpy()
    test_array_copy = deepcopy(test_array)

    prepared_array_train = prepare_omics.prepare_one_hot_omics_data(
        genotype_array=test_array,
        na_augment_perc=1.0,
        na_augment_prob=1.0,
        test_mode=False,
    )

    assert prepared_array_train != test_array
    assert (test_array_copy == test_array).all()

    assert (prepared_array_train[:, -1, :] == 1).all()


def test_prepare_genotype_array_test_mode():
    test_array = torch.zeros((1, 4, 100), dtype=torch.uint8).detach().numpy()
    test_array_copy = deepcopy(test_array)

    prepared_array_test = prepare_omics.prepare_one_hot_omics_data(
        genotype_array=test_array,
        na_augment_perc=1.0,
        na_augment_prob=1.0,
        test_mode=True,
    )
    assert prepared_array_test != test_array
    assert (test_array_copy == test_array).all()

    assert prepared_array_test.sum().item() == 0


@pytest.mark.parametrize(
    "subset_indices",
    [
        None,
        range(10),
        range(0, 50, 2),
        range(50, 100),
        range(0, 100, 2),
    ],
)
def test_load_omics_array_from_disk(subset_indices: Union[None, Sequence[int]]):
    test_arr = np.zeros((4, 100))
    test_arr[-1, :50] = 1
    test_arr[0, 50:] = 1

    with patch(
        "eir.data_load.data_preparation_modules.prepare_omics.np.load",
        return_value=test_arr,
        autospec=True,
    ):
        loaded = prepare_omics.omics_load_wrapper(
            input_source="fake",
            data_pointer=Path("fake"),
            subset_indices=subset_indices,
        )

    expected = test_arr
    if subset_indices is not None:
        expected = test_arr[:, subset_indices]

    assert (loaded == expected).all()
