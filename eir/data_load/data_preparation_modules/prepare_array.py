from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch

from eir.data_load.data_preparation_modules.common import _load_deeplake_sample
from eir.data_load.data_source_modules import deeplake_ops


def array_load_wrapper(
    data_pointer: Union[Path, int],
    input_source: str,
    deeplake_inner_key: Optional[str] = None,
) -> np.ndarray:
    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_inner_key is not None
        assert isinstance(data_pointer, int)
        array_data = _load_deeplake_sample(
            data_pointer=data_pointer,
            input_source=input_source,
            inner_key=deeplake_inner_key,
        )
    else:
        assert isinstance(data_pointer, Path)
        array_data = np.load(str(data_pointer))

    return array_data


def prepare_array_data(array_data: np.ndarray) -> torch.Tensor:
    """Enforce 3 dimensions for now."""

    tensor = torch.from_numpy(array_data).float()

    match len(tensor.shape):
        case 1:
            tensor = tensor.unsqueeze(dim=0).unsqueeze(dim=0)
        case 2:
            tensor = tensor.unsqueeze(dim=0)
        case 3:
            tensor = tensor
        case _:
            raise ValueError(
                f"Array has {len(tensor.shape)} dimensions, currently only "
                f"1, 2, or 3 are supported."
            )

    return tensor
