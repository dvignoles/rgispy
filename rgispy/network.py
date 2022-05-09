import datetime
import gzip
import os
import struct
import subprocess as sp
from io import StringIO
from math import ceil, isnan
from pathlib import Path

# Type Hints
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine

from .grid import count_non_nan, non_nan_cells

# type aliases
numeric = Union[int, float, np.number]
array = Union[list, tuple, np.ndarray, pd.Series]
file_loc = Union[str, Path]
xarray_ds = xr.core.dataset.Dataset
xarray_da = xr.core.dataarray.DataArray


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


def get_encoding(min_val: numeric, max_val: numeric) -> tuple[str, int]:
    """Get best encoding type and fill number.

    Args:
        min_val (numeric): minimum value of data to encode
        max_val (numeric): maximum value of data to encode
    Returns:
        tuple[str, int]: (encoding name, fill value)
    """
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


def load_dbcells(in_gdbn: file_loc) -> pd.DataFrame:
    """Get the 'dbcells' table from a network containing cells and their attributes.

    Args:
        in_gdbn (file_loc): gdbn network file

    Returns:
        pd.DataFrame: table of cells
    """

    cmd1 = [Dir2Ghaas + "/bin/tblAddXY", "-a", "DBCells", in_gdbn, "-"]
    cmd2 = [Dir2Ghaas + "/bin/rgis2table", "-a", "DBCells", "-"]

    ps = sp.Popen(cmd1, stdout=sp.PIPE)
    output = sp.check_output(cmd2, stdin=ps.stdout)
    ps.wait()

    df_out = pd.read_csv(StringIO(output.decode()), sep="\t")

    return df_out


def load_dblayers(inFile: file_loc) -> pd.DataFrame:
    """Read in 'dblayers' info from rgis file. Contains georeferencing informatin.

    Args:
        inFile (file_loc): rgis compatible file

    Returns:
        pd.DataFrame: table with number of rows, cols, cell width etc.
    """
    cmd = [Dir2Ghaas + "/bin/rgis2table", "-a", "DBLayers", inFile]
    proc = sp.Popen(cmd, stdout=sp.PIPE)  # , shell=True) +inFile
    data1 = StringIO(bytearray(proc.stdout.read()).decode("utf-8"))  # type: ignore
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


def _load_geo(name: file_loc, compressed: bool) -> tuple[numeric, numeric, dict]:
    """Get origin and metadata from rgis file.

    Args:
        name (file_loc): rgis compatible file
        compressed (bool): whether or not gzip compressed

    Returns:
        tuple[numeric, numeric, dict]: lower left x, lower left y, metadata
    """
    if compressed:
        ifile = gzip.open(name, "rb")
    else:
        ifile = open(name, "rb")  # type: ignore
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


def get_meta(gdbn: file_loc) -> dict[str, numeric]:
    """Get geospatial meta data from rgis network.

    Args:
        gdbn (file_loc): gbdn rgis network

    Returns:
        dict: dict(llx, lly, col_num, row_num, cell_width, cell_height)
    """
    llx, lly, _ = _load_geo(gdbn, True)
    layers = load_dblayers(gdbn)
    cell_width = layers[["CellWidth"]].values.tolist()[0][0]
    cell_height = layers[["CellHeight"]].values.tolist()[0][0]

    aff = str(Affine.from_gdal(llx, cell_width, 0.0, lly, 0.0, cell_height).to_gdal())

    meta = dict(
        llx=llx,
        lly=lly,
        col_num=layers[["ColNum"]].values.tolist()[0][0],
        row_num=layers[["RowNum"]].values.tolist()[0][0],
        cell_width=cell_width,
        cell_height=cell_height,
        affine=aff,
    )
    return meta


def get_round_coords(
    dbcells_x: array, dbcells_y: array, meta: dict[str, numeric]
) -> dict[str, dict[str, array]]:
    """Get coordinates (rounded and unrounded) from dbcells and based on dblayers meta.

    Args:
        dbcells_x (array): unique x coordinates from dbcells
        dbcells_y (array): unique y coordinates from dbcells
        meta (dict[str, numeric]):

    Returns:
        dict[str, dict[str, array]]: [description]
    """

    inX = meta["llx"] + meta["cell_width"] / 2.0
    inY = meta["lly"] + meta["cell_height"] / 2.0
    x_calc = [inX + meta["cell_width"] * float(n) for n in range(0, meta["col_num"])]  # type: ignore
    y_calc = [inY + meta["cell_height"] * float(n) for n in range(0, meta["row_num"])]  # type: ignore

    def _get_round(dbcells_arr, calc_arr):
        for i in range(1, 10):

            # ceil function minimizes risk of rounding mismatch due to floating point precision
            # ie. decimals ending in 5 round down becasue they are internally represented 0.5 ~= .49999999
            def func(x):
                return ceil(round(x, i) * 100)

            dbcell_round = list(map(func, dbcells_arr))
            calc_round = list(map(func, calc_arr))

            # check for rounded dbcells coords not in rounded calc coords
            not_in = []
            for coord in dbcell_round:
                if coord not in calc_round:
                    not_in.append(coord)
                    break

            if len(not_in) == 0:
                # confirm num unique remains same in rounded calc set
                if len(set(calc_round)) == len(set(calc_arr)):
                    return dbcell_round, calc_round
        return None, None

    y_true_round, y_calc_round = _get_round(dbcells_y, y_calc)
    x_true_round, x_calc_round = _get_round(dbcells_x, x_calc)
    assert (
        x_true_round is not None and x_calc_round is not None
    ), "X rounding procedure failed"
    assert (
        y_true_round is not None and y_calc_round is not None
    ), "Y rounding procedure failed"

    coords = {
        "x": {
            "true": dbcells_x,
            "true_round": x_true_round,
            "calc": x_calc,
            "calc_round": x_calc_round,
        },
        "y": {
            "true": dbcells_y,
            "true_round": y_true_round,
            "calc": y_calc,
            "calc_round": y_calc_round,
        },
    }
    return coords


def _get_dbcells_component_da(
    da_template: xarray_da,
    coords: dict,
    dbcells: pd.DataFrame,
    name: str,
) -> xarray_da:
    """Get the DataArray for an attribute of a network (ID, cell_area, order, etc.) using the 'round-matching approach'.

    Args:
        da_template (xarray_da): empty dataarray with correct shape (from dblayers) and 'calculated' x/y coordinates
        coords (dict): output of get_round_coords (true, true_round, calc, calc_round)
        dbcells (pd.DataFrame): dbcells table from load_db_cells
        name (str): column of dbcells

    Returns:
        xarray_da: [description]
    """
    piv = dbcells[[name, "CellXCoord", "CellYCoord"]].pivot(
        index="CellYCoord", columns="CellXCoord"
    )

    # shape of dbcells
    da_temp = xr.DataArray(
        data=piv.values,
        dims=["lat", "lon"],
        coords=[
            sorted(coords["y"]["true_round"]),  # type: ignore
            sorted(coords["x"]["true_round"]),  # type: ignore
        ],
        name="da_temp",
    )

    # merge with desired template shape
    da = xr.merge([da_template, da_temp], join="left")["da_temp"]

    assert count_non_nan(piv.values) == count_non_nan(
        da.data
    ), "mismatch between rounded coords"

    # dbcells rounded_coord -> dbcells actual coord
    x_cell_lookup = pd.DataFrame(
        {
            "CellXCoord": coords["x"]["true"],
        },
        index=coords["x"]["true_round"],
    ).sort_index()
    y_cell_lookup = pd.DataFrame(
        {
            "CellYCoord": coords["y"]["true"],
        },
        index=coords["y"]["true_round"],
    ).sort_index()

    # calculcated rounded coord -> calculated actual coord
    x_fill_lookup = pd.DataFrame(
        {
            "CellXCoord": coords["x"]["calc"],
        },
        index=coords["x"]["calc_round"],
    ).sort_index()
    y_fill_lookup = pd.DataFrame(
        {
            "CellYCoord": coords["y"]["calc"],
        },
        index=coords["y"]["calc_round"],
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

    return da


def gdbn_to_netcdf_alt(gdbn: Path, out_netcdf: Path, project: str = "") -> Path:

    """Convert .gdbn rgis network to netcdf network compatible with rgispy with alternate 'round-matching' method. Needed when dimensions of dbcells
    do not match true diemensions of network.

    round-matching pseudo algorithm
    1. Get coords from dbcells (true)
    2. Calculate expected coords based on origin, dimensions, and cell size (calculated)
    3. Round true and calculated with maintaining coordinate uniqueness
    4. Match Rounded true with Rounded calcualte.
    5. Where There is not a match use calculated coord. Where there is a match, use true coord.

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

    dbcells = load_dbcells(gdbn)
    meta = get_meta(gdbn)

    # Get all coordinate values (true, true_round, calc, calc_round)
    coords = get_round_coords(
        dbcells.CellXCoord.unique(), dbcells.CellYCoord.unique(), meta
    )

    # Template output for matching process
    shape = (meta["row_num"], meta["col_num"])
    da_template = xr.DataArray(
        np.zeros(shape=shape),  # type: ignore
        dims=["lat", "lon"],
        coords={"lat": coords["y"]["calc_round"], "lon": coords["x"]["calc_round"]},
        name="da_template",
    )

    for var, encoding in OUT_ENCODING.items():
        max_val = dbcells[var].max() / encoding["scale_factor"]
        min_val = dbcells[var].min()

        # TODO: this without relying on a global dict modification <:l
        dtype, fill = get_encoding(min_val, max_val)
        OUT_ENCODING[var]["dtype"] = dtype
        OUT_ENCODING[var]["_FillValue"] = fill

        da = _get_dbcells_component_da(da_template, coords, dbcells, var)

        da.rio.set_spatial_dims(y_dim="lat", x_dim="lon", inplace=True)
        da.rio.write_crs(crs4326, inplace=True)
        da.assign_attrs(OUT_ATTR[var])
        ds[var] = da

    attrs = {
        "WBM_network": gdbn.__str__(),
        "project": project,
        "crs": "+init=epsg:4326",
        "creation_date": "{}".format(datetime.datetime.now()),
    }

    for k in meta.keys():
        attrs[k] = meta[k]  # type: ignore

    ds = ds.assign_attrs(attrs)
    # TODO: decouple ds creation from netcdf saving
    ds.to_netcdf(out_netcdf, encoding=OUT_ENCODING)
    return out_netcdf


def gdbn_to_netcdf_base(in_gdbn: Path, out_netcdf: Path, project: str = "") -> Path:

    """Convert .gdbn rgis network to netcdf network compatible with rgispy
    Raises:
        Exception: unable to encode maximum value
        Exception: unable to encode maximum value
    Returns:
        Path: Path to created netcdf network
    """
    import rasterio.crs as crs
    import rioxarray  # noqa

    # We define the CRS for the output NetCDF
    wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
    crs4326 = crs.CRS.from_wkt(wkt)

    meta = get_meta(in_gdbn)

    # We read the data (can take a long time)
    dfNet = load_dbcells(in_gdbn)

    # Create and empty xarray DataSet to add the xarrays as variables
    ds = xr.Dataset()

    # We process each variable individually (e.g. ToCell, Order etc.)
    for var, encoding in OUT_ENCODING.items():
        # Extract the column that we want from the pandas dataframe
        # the column correcponded to the name of the variable
        # We then pivot it using the coordinates columns as rows (Ys) and
        # columns (Xs) headers
        df_pv = dfNet[[var, "CellXCoord", "CellYCoord"]].pivot(
            index="CellYCoord", columns="CellXCoord"
        )

        # Make sure that we have the smallest data type needed
        max_val = dfNet[var].max() / encoding["scale_factor"]
        min_val = dfNet[var].min()

        # TODO: this without relying on a global dict modification <:l
        dtype, fill = get_encoding(min_val, max_val)
        OUT_ENCODING[var]["dtype"] = dtype
        OUT_ENCODING[var]["_FillValue"] = fill

        da = xr.DataArray(
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
    attrs = {
        "WBM_network": in_gdbn.__str__(),
        "project": project,
        "crs": "+init=epsg:4326",
        "creation_date": "{}".format(datetime.datetime.now()),
    }

    for k in meta.keys():
        attrs[k] = meta[k]  # type: ignore

    ds = ds.assign_attrs(attrs)
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
