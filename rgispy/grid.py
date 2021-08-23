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


def get_buffer_cells(
    lati: int, loni: int, buffer: int, xmax: int, ymax: int
) -> list[tuple[int, int]]:
    """Get a list of cells surrounding a given index of a cell in a certain number of cells buffer.
    Example: Get all cells within a 5 cell radius of cell at (latitiude_index i, longitude_index j)

    Args:
        lati (int): index of x coord
        loni (int): index of y coord
        buffer (int): number of cells to buffer by
        xmax (int): maximum x index
        ymax (int): maxium y index

    Returns:
        list[tuple[int, int]]: list of cell index tuples
    """
    xstart = lati - buffer
    xend = lati + buffer
    ystart = loni - buffer
    yend = loni + buffer

    # border cases
    if xstart < 0:
        xstart = 0
    if xend >= xmax:
        xend = xmax - 1
    if ystart < 0:
        ystart = 0
    if yend >= ymax:
        yend = ymax - 1

    buffer_cells = []
    for x in range(xstart, xend + 1):
        for y in range(ystart, yend + 1):
            buffer_cells.append((x, y))

    return buffer_cells
