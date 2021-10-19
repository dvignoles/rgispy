import datetime
import os
import subprocess as sp
from io import StringIO
from math import isnan
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import rasterio.crs as crs
import rioxarray  # noqa
import xarray as xa

# Type Hints
from xarray.core.dataset import Dataset as xarray_ds

from .grid import non_nan_cells

if "GHAASDIR" in os.environ:
    Dir2Ghaas = os.environ["GHAASDIR"]
else:
    Dir2Ghaas = "/usr/local/share/ghaas"


# Define the variables and their save structure
OUT_ENCODING = {
    "ID": {"dtype": "int32", "scale_factor": 1, "_FillValue": -9999, "zlib": True},
    "ToCell": {"dtype": "int8", "scale_factor": 1, "_FillValue": -9999, "zlib": True},
    "FromCell": {"dtype": "int8", "scale_factor": 1, "_FillValue": -9999, "zlib": True},
    "Order": {"dtype": "int8", "scale_factor": 1, "_FillValue": -9999, "zlib": True},
    "BasinID": {"dtype": "int16", "scale_factor": 1, "_FillValue": -9999, "zlib": True},
    "BasinCells": {
        "dtype": "int16",
        "scale_factor": 1,
        "_FillValue": -9999,
        "zlib": True,
    },
    "Travel": {"dtype": "int16", "scale_factor": 1, "_FillValue": -9999, "zlib": True},
    "CellArea": {
        "dtype": "int16",
        "scale_factor": 0.0001,
        "_FillValue": -9999,
        "zlib": True,
    },
    "CellLength": {
        "dtype": "int16",
        "scale_factor": 0.001,
        "_FillValue": -9999,
        "zlib": True,
    },
    "SubbasinArea": {
        "dtype": "int16",
        "scale_factor": 0.0001,
        "_FillValue": -9999,
        "zlib": True,
    },
    "SubbasinLength": {
        "dtype": "int16",
        "scale_factor": 0.001,
        "_FillValue": -9999,
        "zlib": True,
    },
}
# Out Attributes
OUT_ATTR = {
    "ID": {"description": "CellID of WBM network"},
    "ToCell": {
        "description": "Network flow direction",
        "values": "0-No flow, 1-E, 2-SE,4-S,8-SW,16-W,32-NW,64-N,128-NE",
    },
    "FromCell": {
        "description": "Incoming flow direction",
        "values": "Sum of direction values following the codes shown in the ToCell values",
    },
    "Order": {"description": "River order at cell"},
    "BasinID": {"description": "ID of river basins in WBM network"},
    "BasinCells": {"description": "Number of grid cell of upstream basin"},
    "Travel": {"description": "Number of grid cell downstream to the basin outlet"},
    "CellArea": {"description": "Cell cartesian area", "units": "km2 - Square km"},
    "CellLength": {"description": "Cell cartesian flow length", "units": "km"},
    "SubbasinArea": {
        "description": "Area of upstream basin",
        "units": "km2 - Square km",
    },
    "SubbasinLength": {
        "description": "Mainstem length upstream following the larger catchment area (e.g. the mainstem length follows the Missouri in the Mississippi basin)",
        "units": "km",
    },
}


def LoadDBCells(in_gdbn):
    # Adds the X and Y columns to the DBCells table and loads the network
    # DBCells table into a Pandas DataFrame

    cmd1 = [Dir2Ghaas + "/bin/tblAddXY", "-a", "DBCells", in_gdbn, "-"]
    cmd2 = [Dir2Ghaas + "/bin/rgis2table", "-a", "DBCells", "-"]

    ps = sp.Popen(cmd1, stdout=sp.PIPE)
    output = sp.check_output(cmd2, stdin=ps.stdout)
    ps.wait()

    df_out = pd.read_csv(StringIO(output.decode()), sep="\t")

    return df_out


def gdbn_to_netcdf(in_gdbn: Path, out_netcdf: Path, project: str = "") -> Path:

    """Convert .gdbn rgis network to netcdf network compatible with rgispy

    Raises:
        Exception: unable to encode maximum value
        Exception: unable to encode maximum value

    Returns:
        Path: Path to created netcdf network
    """
    # We define the CRS for the output NetCDF
    wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
    crs4326 = crs.CRS.from_wkt(wkt)

    # We read the data (can take a long time)
    dfNet = LoadDBCells(in_gdbn)

    # Create and empty xarray DataSet to add the xarrays as variables
    ds = xa.Dataset()

    # We process each variable individually (e.g. ToCell, Order etc.)
    for var, encoding in OUT_ENCODING.items():
        print("Processing variable {}".format(var))
        # Extract the column that we want from the pandas dataframe
        # the column correcponded to the name of the variable
        # We then pivot it using the coordinates columns as rows (Ys) and
        # columns (Xs) headers
        df_pv = dfNet[[var, "CellXCoord", "CellYCoord"]].pivot(
            index="CellYCoord", columns="CellXCoord"
        )

        # Make sure that we have the smallest data type needed
        max = dfNet[var].max() / encoding["scale_factor"]
        min = dfNet[var].min()
        if min >= 0:  # if all values are positive, we can use unsigned integers
            if max < 255:
                OUT_ENCODING[var]["dtype"] = "uint8"
                OUT_ENCODING[var]["_FillValue"] = 255
            elif max < 65535:
                OUT_ENCODING[var]["dtype"] = "uint16"
                OUT_ENCODING[var]["_FillValue"] = 65535
            elif max < 4294967295:
                OUT_ENCODING[var]["dtype"] = "uint32"
                OUT_ENCODING[var]["_FillValue"] = 4294967295
            elif max < 18446744073709551615:
                OUT_ENCODING[var]["dtype"] = "uint64"
                OUT_ENCODING[var]["_FillValue"] = 18446744073709551615
            else:
                raise Exception(
                    "Max value: {}... Unable to code data to unsigned int type!".format(
                        max
                    )
                )
        else:  # otherwise we use signed integers
            if max <= 127:
                OUT_ENCODING[var]["dtype"] = "int8"
                OUT_ENCODING[var]["_FillValue"] = -128
            elif max <= 32767:
                OUT_ENCODING[var]["dtype"] = "int16"
                OUT_ENCODING[var]["_FillValue"] = -32768
            elif max <= 2147483647:
                OUT_ENCODING[var]["dtype"] = "int32"
                OUT_ENCODING[var]["_FillValue"] = -2147483648
            elif max <= 9223372036854775807:
                OUT_ENCODING[var]["dtype"] = "int64"
                OUT_ENCODING[var]["_FillValue"] = -9223372036854775808
            else:
                raise Exception(
                    "Max value: {}... Unable to code data to signed int type!".format(
                        max
                    )
                )
        # Note that int numpy arrays cannot have nan... (so we convert first to -9999)
        # No longer needed, the encoding on save takes care of it (good to keep it here
        # for future reference)
        # df_pv=df_pv.fillna(-9999).astype("int32")

        # And now we can create the xarray
        da = xa.DataArray(
            data=df_pv.values,
            dims=["lat", "lon"],
            coords=[
                dfNet["CellYCoord"].sort_values(ascending=True).unique(),
                dfNet["CellXCoord"].sort_values(ascending=True).unique(),
            ],
        )
        # And add the CRS (we use the rioxarray package...)
        da.rio.set_spatial_dims(y_dim="lat", x_dim="lon", inplace=True)
        da.rio.write_crs(crs4326, inplace=True)
        da.assign_attrs(OUT_ATTR[var])
        # We add the xarray to the xarray dataset
        ds[var] = da

    # Once all the variables are loaded into the xarray dataset we add
    # dataset attributes
    ds = ds.assign_attrs(
        {
            "WBM_network": in_gdbn.__str__(),
            "project": project,
            "crs": "+init=epsg:4326",
            "creation_date": "{}".format(datetime.datetime.now()),
        }
    )
    # And we save the xarray dataset as a netCDF
    ds.to_netcdf(out_netcdf, encoding=OUT_ENCODING)
    return out_netcdf


def next_cell(cell_index: tuple, to_cell_code: int) -> Optional[tuple[int, int]]:
    """Get index of next cell in network

    Args:
        cell_index (tuple): index of cell to get next_cell of
        to_cell_code (int): ToCell encoding value representing direction of next cell

    Returns:
        tuple: index of next_cell in flow direction
    """
    assert to_cell_code in [
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ], "to cell must be valid direction encoding"
    # cell_index = (lat, lon)
    # No Flow
    if to_cell_code == 0:
        return None
    # East
    elif to_cell_code == 1:
        return (cell_index[0], cell_index[1] + 1)
    # SE
    elif to_cell_code == 2:
        return (cell_index[0] - 1, cell_index[1] + 1)
    # S
    elif to_cell_code == 4:
        return (cell_index[0] - 1, cell_index[1])
    # SW
    elif to_cell_code == 8:
        return (cell_index[0] - 1, cell_index[1] - 1)
    # W
    elif to_cell_code == 16:
        return (cell_index[0], cell_index[1] - 1)
    # NW
    elif to_cell_code == 32:
        return (cell_index[0] + 1, cell_index[1] - 1)
    # N
    elif to_cell_code == 64:
        return (cell_index[0] + 1, cell_index[1])
    # NE
    elif to_cell_code == 128:
        return (cell_index[0] + 1, cell_index[1] + 1)
    else:
        return None


def lookup_cellid(lon: float, lat: float, network: xarray_ds) -> int:
    """Get cellid of snapped coordinate pair

    Args:
        lon (float): longitude
        lat (float): latitude
        network (xarray_ds): network xarray dataset

    Returns:
        int: cellid of coordinate
    """
    cellid = network["ID"].sel(lat=lat, lon=lon).data.tolist()
    return int(cellid)


def get_basin_mouth(network: xarray_ds, cell_idx: tuple) -> tuple[Any, tuple[Any, ...]]:
    """Get the mouth of the basin for a particular network cell by recursing through network.

    Args:
        network (xarray Dataset): xarray Dataset of network created via this module
        cell_idx (tuple): tuple representing index of cell to find basin mouth of

    Returns:
        (tuple): (basinid, cell_idx) tuple of basinid and the terminal mouth cell_idx
    """
    basin = network["BasinID"][cell_idx].data.tolist()
    tocell = int(network["ToCell"][cell_idx].data.tolist())
    next_cell_idx = next_cell(cell_idx, tocell)

    next_cell_id = network["ID"][next_cell_idx].data.tolist()
    next_cell_basin = network["BasinID"][next_cell_idx].data.tolist()

    if next_cell_basin == basin and not isnan(next_cell_id):
        return get_basin_mouth(network, next_cell_idx)  # type: ignore
    else:
        return basin, cell_idx


def get_all_basin_mouth(network: xarray_ds) -> list[tuple[Any, tuple[Any, ...]]]:  # type: ignore[return]
    """Get list of all basin mouths for network

    Args:
        network (xarray Dataset): xarray Dataset of network created via this module

    Returns:
        List[tuple]: list of tuples of form (basinid, cell_idx)
    """
    all_valid_starts = non_nan_cells(network["ID"].data)
    unique_basins = set(np.unique(network["BasinID"]))
    assert len(unique_basins) > 1, "must have at least 1 non-nan basin"

    basin_mouths = []
    for cell_idx in all_valid_starts:

        # we're done if the only unique basin is "nan"
        if len(unique_basins) > 1:
            basinid = network["BasinID"][cell_idx].data.tolist()

            # avoid duplicating efforts for same basin cells
            if basinid in unique_basins:
                basin_mouths.append(get_basin_mouth(network, cell_idx))
                unique_basins.remove(basinid)
        else:
            return basin_mouths
