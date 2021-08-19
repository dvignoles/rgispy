"""Utility functions for operating on gridded datasets
"""

# Type Hints
from typing import Any, Generator

import numpy as np
from xarray.core.dataarray import DataArray as xarray_da


def non_nan_cells(data: np.ndarray) -> list[tuple[Any, ...]]:
    """Get indices of cells in grid which are non null/nan

    Args:
        data (np.ndarray): numpy ndarray containing nan and actual data

    Returns:
       list[tuple[int, int]]: list of index values
    """
    indices = [tuple(ind) for ind in np.argwhere(~np.isnan(data))]
    return indices


def get_non_nan_cells(data_array: xarray_da) -> Generator[xarray_da, None, None]:
    """Generator of cells in DataArray containing non-nan data

    Args:
        data_array (xarray DataArray): xarray DataArray

    Yields:
        xarray DataArray: DataArray at index of actual data
    """
    indices = non_nan_cells(data_array.data)
    for idx in indices:
        yield data_array[idx]


def count_non_nan(data: np.ndarray) -> int:
    """Count number of non nan cells in numpy matrix

    Args:
        data (np.ndarray): numpy array

    Returns:
        int: count of cells which are not nan
    """
    return np.count_nonzero(~np.isnan(data))
