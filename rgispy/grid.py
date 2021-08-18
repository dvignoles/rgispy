"""Utility functions for operating on gridded datasets
"""

import numpy as np
import xarray as xr


def non_nan_cells(data: np.ndarray):
    """Get indices of cells in grid which are non null/nan

    Args:
        data (np.ndarray): numpy ndarray containing nan and actual data

    Returns:
        np.ndarray: array of index values
    """
    indices = [tuple(ind) for ind in np.argwhere(~np.isnan(data))]
    return indices


def get_non_nan_cells(data_array: xr.core.dataarray.DataArray):
    """Generator of cells in DataArray containing non-nan data

    Args:
        data_array (xr.core.dataarray.DataArray): xarray DataArray

    Yields:
        xr.core.dataarray.DataArray: DataArray at index of actual data
    """
    indices = non_nan_cells(data_array.data)
    for idx in indices:
        yield data_array[idx]


def count_non_nan(data):
    """Count number of non nan cells in numpy matrix

    Args:
        data (np.ndarray): numpy array

    Returns:
        (int): count of cells which are not nan
    """
    return np.count_nonzero(~np.isnan(data))
