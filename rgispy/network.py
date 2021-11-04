import datetime
import gzip
import os
import struct
import subprocess as sp
from io import StringIO
from math import isnan
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

# Type Hints
from xarray.core.dataset import Dataset as xarray_ds

from .grid import non_nan_cells

if "GHAASDIR" in os.environ:
    Dir2Ghaas = os.environ["GHAASDIR"]
else:
    Dir2Ghaas = "/usr/local/share/ghaas"

# EPSG:4326
WKT = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'

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


def _ReadDBLayers(inFile):
    cmd = [Dir2Ghaas + "/bin/rgis2table", "-a", "DBLayers", inFile]
    proc = sp.Popen(cmd, stdout=sp.PIPE)  # , shell=True) +inFile
    data1 = StringIO(bytearray(proc.stdout.read()).decode("utf-8"))
    Layers = pd.read_csv(
        data1,
        sep="\t",
        dtype={
            "ID": "int",
            "RowNum": "int",
            "ColNum": "int",
            "ValueType": "int",
            "ValueSize": "int",
            "CellWidth": "float",
            "CellHeight": "float",
        },
    )
    Layers["ID"] = Layers["ID"] - 1

    return Layers


def _LoadGeo(name, compressed):
    if compressed:
        ifile = gzip.open(name, "rb")
    else:
        ifile = open(name, "rb")
    ifile.seek(40)
    LLx = struct.unpack("d", ifile.read(8))[0]
    LLy = struct.unpack("d", ifile.read(8))[0]
    ifile.read(8)
    titleLen = struct.unpack("h", ifile.read(2))[0]
    title = ifile.read(titleLen).decode()
    MetaData = {"title": title}
    ifile.read(9)
    docLen = struct.unpack("h", ifile.read(2))[0]
    _ = ifile.read(docLen).decode()
    ifile.read(25)
    readMore = True
    while readMore:
        infoLen = struct.unpack("h", ifile.read(2))[0]
        infoRec = ifile.read(infoLen).decode()
        if infoRec == "Data Records":
            readMore = False
            break
        ifile.read(1)
        valLen = struct.unpack("h", ifile.read(2))[0]
        if valLen == 44:
            ifile.read(26)
        elif valLen == 48:
            ifile.read(30)
        valLen = struct.unpack("h", ifile.read(2))[0]
        valRec = ifile.read(valLen).decode()
        MetaData[infoRec.lower()] = valRec
        ifile.read(1)

    return LLx, LLy, MetaData


def get_network_meta(gdbn):
    llx, lly, metadata = _LoadGeo(gdbn, True)
    layers = _ReadDBLayers(gdbn)

    meta = dict(
        llx=llx,
        lly=lly,
        col_num=layers[["ColNum"]].values.tolist()[0][0],
        row_num=layers[["RowNum"]].values.tolist()[0][0],
        cell_width=layers[["CellWidth"]].values.tolist()[0][0],
        cell_height=layers[["CellHeight"]].values.tolist()[0][0],
    )
    return meta


def GetRounding(inArray):
    for r in range(1, 20):
        good = True
        prev = 0
        for i in range(0, len(inArray)):
            curr = round(inArray[i], ndigits=r)
            if curr == prev:
                good = False
                break
            prev = curr
        if good:
            return r


def calc_rounded_coords(gdbn):
    meta = get_network_meta(gdbn)
    inX = meta["llx"] + meta["cell_width"] / 2.0
    inY = meta["lly"] + meta["cell_height"] / 2.0
    xcoords = [inX + meta["cell_width"] * float(n) for n in range(0, meta["col_num"])]
    ycoords = [inY + meta["cell_height"] * float(n) for n in range(0, meta["row_num"])]

    roundX = GetRounding(xcoords)
    roundY = GetRounding(ycoords)

    round_ycoords = [round(y, ndigits=roundY) for y in ycoords]
    round_xcoords = [round(x, ndigits=roundX) for x in xcoords]

    shape = (meta["row_num"], meta["col_num"])
    da_template = xr.DataArray(
        np.zeros(shape=shape),
        dims=["lat", "lon"],
        coords={"lat": round_ycoords, "lon": round_xcoords},
        name="da_template",
    )

    return xcoords, roundX, ycoords, roundY, da_template


def get_dbcells_component_da(da_template, calc_xcoords, calc_ycoords, dbcells, name):
    piv = dbcells[[name, "CellXCoord", "CellYCoord"]].pivot(
        index="CellYCoord", columns="CellXCoord"
    )

    # shape of dbcells
    da_temp = xr.DataArray(
        data=piv.values,
        dims=["lat", "lon"],
        coords=[
            dbcells["RoundCellYCoord"].sort_values(ascending=True).unique(),
            dbcells["RoundCellXCoord"].sort_values(ascending=True).unique(),
        ],
        name="da_temp",
    )

    # merge with desired template shape
    da = xr.merge([da_template, da_temp], join="left")["da_temp"]

    # dbcells rounded_coord -> dbcells actual coord
    x_cell_lookup = pd.DataFrame(
        {
            "CellXCoord": dbcells.CellXCoord.unique(),
        },
        index=dbcells.RoundCellXCoord.unique(),
    ).sort_index()
    y_cell_lookup = pd.DataFrame(
        {
            "CellYCoord": dbcells.CellYCoord.unique(),
        },
        index=dbcells.RoundCellYCoord.unique(),
    ).sort_index()

    # calculcated rounded coord -> calculated actual coord
    x_fill_lookup = pd.DataFrame(
        {
            "CellXCoord": calc_xcoords,
        },
        index=da_template.lon.data,
    ).sort_index()
    y_fill_lookup = pd.DataFrame(
        {
            "CellYCoord": calc_ycoords,
        },
        index=da_template.lat.data,
    ).sort_index()

    def _lookup_final_x(x):
        try:
            coord = x_cell_lookup.loc[x, "CellXCoord"]
            return coord
        except KeyError:
            return x_fill_lookup.loc[x, "CellXCoord"]

    def _lookup_final_y(y):
        try:
            coord = y_cell_lookup.loc[y, "CellYCoord"]
            return coord
        except KeyError:
            return y_fill_lookup.loc[y, "CellYCoord"]

    # dbcells actual coord if it maps to a rounded calculated coord else the actual calculated coord
    final_x = x_fill_lookup.index.map(_lookup_final_x).values
    final_y = y_fill_lookup.index.map(_lookup_final_y).values

    da["lat"] = final_y
    da["lon"] = final_x
    print(name)

    return da


def get_encoding(min_val, max_val):
    if min_val >= 0:  # if all values are positive, we can use unsigned integers
        if max_val < 255:
            return "uint8", 255
        elif max_val < 65535:
            return "uint16", 65535
        elif max_val < 4294967295:
            return "uint32", 4294967295
        elif max_val < 18446744073709551615:
            return "uint64", 18446744073709551615
        else:
            raise Exception(
                "max_val value: {}... Unable to code data to unsigned int type!".format(
                    max_val
                )
            )
    else:  # otherwise we use signed integers
        if max_val <= 127:
            return "int8", -128
        elif max_val <= 32767:
            return "int16", -32768
        elif max_val <= 2147483647:
            return "int32", -2147483648
        elif max_val <= 9223372036854775807:
            return "int64", -9223372036854775808
        else:
            raise Exception(
                "max_val value: {}... Unable to code data to signed int type!".format(
                    max_val
                )
            )


def gdbn_to_netcdf(gdbn: Path, out_netcdf: Path, project: str = "") -> Path:

    """Convert .gdbn rgis network to netcdf network compatible with rgispy

    Raises:
        Exception: unable to encode maximum value
        Exception: unable to encode maximum value

    Returns:
        Path: Path to created netcdf network
    """
    import rasterio.crs as crs
    import rioxarray  # noqa

    ds = xr.Dataset()
    crs4326 = crs.CRS.from_wkt(WKT)

    dbcells = LoadDBCells(gdbn)
    xcoords, roundX, ycoords, roundY, da_template = calc_rounded_coords(gdbn)

    dbcells["RoundCellXCoord"] = dbcells.CellXCoord.map(
        lambda x: round(x, ndigits=roundX)
    )
    dbcells["RoundCellYCoord"] = dbcells.CellYCoord.map(
        lambda y: round(y, ndigits=roundY)
    )

    for var, encoding in OUT_ENCODING.items():
        max_val = dbcells[var].max() / encoding["scale_factor"]
        min_val = dbcells[var].min()

        # TODO: this without relying on a global dict modification <:l
        dtype, fill = get_encoding(min_val, max_val)
        OUT_ENCODING[var]["dtype"] = dtype
        OUT_ENCODING[var]["_FillValue"] = fill

        da = get_dbcells_component_da(da_template, xcoords, ycoords, dbcells, var)

        da.rio.set_spatial_dims(y_dim="lat", x_dim="lon", inplace=True)
        da.rio.write_crs(crs4326, inplace=True)
        da.assign_attrs(OUT_ATTR[var])
        ds[var] = da

    ds = ds.assign_attrs(
        {
            "WBM_network": gdbn.__str__(),
            "project": project,
            "crs": "+init=epsg:4326",
            "creation_date": "{}".format(datetime.datetime.now()),
        }
    )

    # TODO: decouple ds creation from netcdf saving
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
