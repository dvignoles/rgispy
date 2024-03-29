import datetime
import gzip
import subprocess as sp
from ctypes import Structure
from ctypes import Union as c_Union
from ctypes import c_char, c_double, c_int, c_short
from os import PathLike
from pathlib import Path

# typing hints
from typing import IO, BinaryIO, Generator, List, Optional, Union, no_type_check

import numpy as np
import pandas as pd
import xarray as xa

# Define the variables and their save structure
OutEncoding = {
    "Discharge": {
        "dtype": "int32",
        "scale_factor": 0.0001,
        "_FillValue": -9999,
        "zlib": True,
    },
    "Evapotranspiration": {
        "dtype": "int32",
        "scale_factor": 1,
        "_FillValue": -9999,
        "zlib": True,
    },
    "RainPET": {"dtype": "int32", "scale_factor": 1, "_FillValue": -9999, "zlib": True},
    "RelativeSoilMoisture": {
        "dtype": "int32",
        "scale_factor": 1,
        "_FillValue": -9999,
        "zlib": True,
    },
    "RiverDepth": {
        "dtype": "int32",
        "scale_factor": 1,
        "_FillValue": -9999,
        "zlib": True,
    },
    "RiverTemperature": {
        "dtype": "int32",
        "scale_factor": 0.01,
        "_FillValue": -9999,
        "zlib": True,
    },
    "RiverWidth": {
        "dtype": "int32",
        "scale_factor": 1,
        "_FillValue": -9999,
        "zlib": True,
    },
    "Runoff": {"dtype": "int32", "scale_factor": 1, "_FillValue": -9999, "zlib": True},
    "SnowPack": {
        "dtype": "int32",
        "scale_factor": 1,
        "_FillValue": -9999,
        "zlib": True,
    },
    "SoilMoisture": {
        "dtype": "int32",
        "scale_factor": 1,
        "_FillValue": -9999,
        "zlib": True,
    },
    "WetBulbTemp": {
        "dtype": "int32",
        "scale_factor": 1,
        "_FillValue": -9999,
        "zlib": True,
    },
}

# The following are two data structures converted from the RGIS C code,
# and are used to parse the GDBC file header


class MFmissing(c_Union):
    _fields_ = [("Int", c_int), ("Float", c_double)]


class MFdsHeader(Structure):
    _fields_ = [
        ("Swap", c_short),
        ("Type", c_short),
        ("ItemNum", c_int),
        ("Missing", MFmissing),
        ("Date", c_char * 24),
    ]


def get_date_format(time_step):

    time_step = time_step.lower()
    if time_step == "daily":
        return "%Y-%m-%d"
    elif time_step == "monthly":
        return "%Y-%m"
    elif time_step == "annual":
        return "%Y"
    elif time_step == "alt":
        return None
    elif time_step == "dlt":
        return None


def headDS(ifile, time_step):

    forty = ifile.read(40)
    dump40 = MFdsHeader.from_buffer_copy(forty)
    Type = dump40.Type
    # Type values 5:short,6:long,7:float,8:double
    if Type > 6:
        NoData = dump40.Missing.Float
    else:
        NoData = dump40.Missing.Int

    npType = _npType(Type)

    date_raw = dump40.Date.decode()
    time_step = time_step.lower()
    date_format = get_date_format(time_step)

    if date_format is not None:
        Date = datetime.datetime.strptime(date_raw, date_format)
    else:
        Date = date_raw

    Items = dump40.ItemNum

    return Type, npType, NoData, Date, Items


def recordDS(ifile, items, npType, skip=True):
    # Skip the 40 bytes of the record header
    if skip:
        _ = ifile.read(40)
    bytes = items * npType(1).itemsize
    RecordData = np.frombuffer(ifile.read(bytes), dtype=npType)
    return RecordData


def _npType(nType: int) -> type:
    """Translate GDBC data type codes into standard numpy types

    Args:
        nType (int): gdbc data type code

    Raises:
        Exception: Unknown data type code

    Returns:
        (numpy type): np.int16, np.int32, np.float32, np.float64
    """
    # nType values: 5=short,6=long,7=float,8=double
    if nType == 5:
        return np.int16
    elif nType == 6:
        return np.int32
    elif nType == 7:
        return np.float32
    elif nType == 8:
        return np.float64
    else:
        raise Exception("Unknown value format: type {}".format(nType))


def n_records(year: Optional[int], time_step: str) -> int:
    """Get number of expected records in datastream based on time_step

    Args:
        year (Optional[int]): year of datastream file
        time_step (str): annual, monthly, or daily

    Returns:
        int: number of records (ex: 365 for daily non-leap year datastream)
    """

    time_step = time_step.lower()
    assert time_step in [
        "annual",
        "monthly",
        "daily",
        "alt",
        "dlt",
    ], "time_step must be annual monthly or daily"

    if time_step == "annual":
        return 1
    elif time_step == "monthly":
        return 12
    elif time_step == "daily":
        p = pd.Period("{}-01-01".format(year))
        if p.is_leap_year:
            days = 366
        else:
            days = 365
        return days
    # Long term average files
    elif time_step == "alt":
        return 1
    elif time_step == "dlt":
        return 365

    # Here just for explicitness when running type-checker
    return 0


def gdbc_to_ds_buffer(gdbc: Path, network: Path) -> Optional[IO[bytes]]:
    """Get buffered fileobject of datastream from gdbc using network gdbn as template
    via rgis2ds command

    Args:
        gdbc (Path): gdbc file path
        network (Path): gdbn file path

    Returns:
        BinaryIO: buffered reader of output datastream
    """
    cmd = "/usr/local/share/ghaas/bin/rgis2ds --template {network} {gdbc}".format(
        network=network, gdbc=gdbc
    ).split()
    p = sp.Popen(cmd, stdout=sp.PIPE)

    return p.stdout


# type checking kind of a mess with this function, it's handled by try/except
@no_type_check
def get_true_datastream(
    file_in: Union[BinaryIO, PathLike[str]]
) -> Union[BinaryIO, gzip.GzipFile]:
    """Get file descriptor of either datastream (gds) gzip compressed datastream (gz)
    or stdin buffer or rgis2ds stdout

    Args:
        file_in (file like): either file like object or path like

    Returns:
       BinaryIO: either file_in or gzip reader
    """

    def _is_compressed(file_name):
        extension = file_name.split(".", 1)[-1]
        assert extension in [
            "gds.gz",
            "ds.gz",
            "gds",
            "ds",
        ], "extension must be either gz or gds or ds"
        if extension in ["gds.gz", "ds.gz"]:
            return True
        elif extension in ["gds", "ds"]:
            return False

    try:
        # if str or Path object this will work
        file_in = Path(file_in).resolve()
        if _is_compressed(file_in.name):
            return gzip.open(file_in)
        else:
            return open(file_in, "rb")
    except TypeError:
        # File object
        datastream_path = file_in.name

        if datastream_path != "<stdin>":
            # case when not stdin but not named file
            try:
                datastream_name = Path(datastream_path).name
            except TypeError:
                return file_in

            if _is_compressed(datastream_name):
                datastream = gzip.open(datastream_name)
                file_in.close()
                return datastream
            else:
                return file_in
        else:
            # return stdin ensuring it is binary buffer
            try:
                file_buf = file_in.buffer
                return file_buf
            except AttributeError:
                # already a buffer
                return file_in


def _gen_date_cols(time_step, year):
    time_step = time_step.lower()
    valid_time_steps = [
        "annual",
        "monthly",
        "daily",
        "alt",
        "dlt",
    ]
    assert time_step in valid_time_steps, "time_step must be in {}".format(
        valid_time_steps
    )

    freq_d = {
        "annual": "YS",
        "monthly": "MS",
        "daily": "D",
    }
    if time_step in ["daily", "monthly", "annual"]:

        date_cols = pd.date_range(
            start="1/1/{}".format(year),
            end="12/31/{}".format(year),
            freq=freq_d[time_step],
        )
        return date_cols

    # Long Term Average files
    elif time_step == "alt":
        date_cols = [
            "XXXX",
        ]
        return date_cols

    elif time_step in [
        "dlt",
    ]:
        dummy_dates = pd.date_range(start="1/1/2001", end="12/31/2001", freq="D")
        year_mon = [(str(d.month).zfill(2), str(d.day).zfill(2)) for d in dummy_dates]
        if time_step == "dlt":
            return ["XXXX--{}-{}".format(m, d) for m, d in year_mon]


def get_masks(mask_ds, mask_layers, output_dir, year, time_step):

    masks = []
    for m in mask_layers:
        Mask = mask_ds[m].data
        MaskType = mask_ds[m].attrs["Type"]
        MaskValues = Mask.flatten()
        MaskValues = MaskValues[~np.isnan(MaskValues)].astype("int")
        MaskValues = list(set(MaskValues))

        OutputPath = output_dir.joinpath(m, time_step.capitalize())
        OutputPath.mkdir(exist_ok=True, parents=True)

        time_step = time_step.lower()
        date_cols = _gen_date_cols(time_step, year)

        if MaskType == "Polygon":
            dfOut = pd.DataFrame(
                index=MaskValues,
            )
        elif MaskType == "Point":
            dfOut = pd.DataFrame(index=MaskValues, columns=date_cols)

        masks.append((m, Mask, MaskType, MaskValues, OutputPath, dfOut))

    return masks


def iter_ds(
    file_buf: BinaryIO, mask_id: np.ndarray, year: Optional[int], time_step: str
) -> Generator[tuple[np.ndarray, datetime.datetime], None, None]:
    """Generator of (ndarray, datetime) record tuples over datastream file object

    Args:
        file_buf (BinaryIO): file object of datastream file (output from get_true_datastream)
        mask_id (np.ndarray): mask['ID'].data from mask xarray
        year (Optional[int]): year of datastream
        time_step (str): annual, monthly, or daily
    Yields:
        Generator[tuple[np.ndarray, datetime.datetime]]: (data, datetime) record pairs
    """
    cell_id = np.nan_to_num(mask_id, copy=True, nan=0.0).astype("int32")
    nRecords = n_records(year, time_step)

    rgisType, npType, NoData, Date, Cells = headDS(file_buf, time_step)

    for day in range(0, nRecords):
        if day != 0:
            _, _, _, Date, _ = headDS(file_buf, time_step)

        Data = recordDS(file_buf, Cells, npType, skip=False)
        # We add a NoData entry at the beginning of the data array, so that
        # ID = 0 (e.g. the NoData values of the rgis network) will map to NoData...
        Data = np.insert(Data, 0, NoData)

        Data = Data[cell_id.flatten()].reshape(cell_id.shape)
        if rgisType <= 6:
            _ = Data.astype("float")
        Data[Data == NoData] = np.nan

        yield Data, Date


def iter_gdbc(
    gdbc: Path, network: Path, mask_id: np.ndarray, year: Optional[int], time_step: str
) -> Generator[tuple[np.ndarray, datetime.datetime], None, None]:
    """Wrapper of iter_ds for gdbc via rgis2ds

    Args:
        gdbc (Path): gdbc file path
        network (Path): gdbn file path
        mask_id (np.ndarray): mask['ID'].data from mask xarray
        year (Optional[int]): year of datastream
        time_step (str): annual, monthly, or daily
    Yields:
        Generator[tuple[np.ndarray, datetime.datetime]]: (data, datetime) record pairs
    """
    file_buf = gdbc_to_ds_buffer(gdbc, network)
    try:
        for data, date in iter_ds(file_buf, mask_id, year, time_step):  # type: ignore
            yield data, date
    except GeneratorExit:
        file_buf.close()  # type: ignore


def sample_ds(
    mask_nc: Path,
    file_in: Union[
        BinaryIO,
        Path,
    ],
    mask_layers: List[str],
    output_dir: Path,
    year: Optional[int],
    variable: str,
    time_step: str,
    csv_name: Union[str, Path] = None,
    cell_area: np.ndarray = None,
) -> None:

    """Sample a datastream using a netcdf mask

    Args:
        mask_nc (Path): netcdf mask file
        file_in (Union[BinaryIO, Path]): datastream file object or pathlike
        mask_layers (List[str]): list of masks from mask_nc to sample with
        output_dir (Path): directory of output
        year (Optional[int]): year of datastream file
        variable (str): variable of datastream file (ie. Discharge, Temperature..)
        time_step (str): annual, monthly, daily, alt, or dlt
        csv_name (Union[str, Path]): Name of resulting sampled csv
        cell_area (Optional[np.ndarray]): Cell Area grid corresponding to mask. Needed for polygon masks to calculate weighted zonal mean.
    """

    file_buf = get_true_datastream(file_in)

    # set up masks
    mask_ds = xa.open_dataset(mask_nc)
    masks = get_masks(mask_ds, mask_layers, output_dir, year, time_step)

    for Data, Date in iter_ds(file_buf, mask_ds["ID"].data, year, time_step):
        # each loop is one day in the case of daily time_step
        for m, Mask, MaskType, MaskValues, OutputPath, dfOut in masks:
            if MaskType == "Polygon":
                assert (
                    cell_area is not None
                ), "Cell Area required to weight polygon averages"

                # exclude nan values from mean/min/max determination
                mData = np.ma.MaskedArray(
                    Data, mask=np.isnan(Data)
                )  # type: np.ma.MaskedArray

                date_format = get_date_format(time_step)
                if date_format is not None:
                    # daily / monthly annual
                    dstr = Date.strftime(date_format)
                else:
                    # alt - XXXX, dlt - XXXX-01-01
                    dstr = Date  # type: ignore

                dfOut["mean_{}".format(dstr)] = [
                    # Mean weighted by cell area
                    np.ma.average(mData[Mask == i], weights=cell_area[Mask == i])
                    for i in MaskValues
                ]
                dfOut["min_{}".format(dstr)] = [
                    np.ma.min(mData[Mask == i]) for i in MaskValues
                ]
                dfOut["max_{}".format(dstr)] = [
                    np.ma.max(mData[Mask == i]) for i in MaskValues
                ]
            elif MaskType == "Point":
                dfOut[Date] = pd.DataFrame(
                    data=Data[~np.isnan(Mask)],
                    index=Mask[~np.isnan(Mask)].astype("int"),
                    columns=["values"],
                )

    for m, _, _, _, OutputPath, dfOut in masks:
        # provide default csv name
        year_name = str(year)
        if year is None:
            year_name = time_step
        if csv_name is None:
            csv_f = OutputPath.joinpath("{}_{}.csv".format(variable, year_name))
        else:
            csv_f = OutputPath.joinpath(csv_name)

        dfOut.to_csv(csv_f)


def sample_gdbc(
    mask_nc: Path,
    file_path: Path,
    network: Path,
    mask_layers: List[str],
    output_dir: Path,
    year: Optional[int],
    variable: str,
    time_step: str,
    csv_name: str = None,
    cell_area: np.ndarray = None,
) -> None:

    """Sample a gdbc rgis grid using a netcdf mask

    Args:
        mask_nc (Path): netcdf mask file
        file_in (Path]): .gdbc or gdbc.gz path
        network (Path): gdbn rgis network
        mask_layers (List[str]): list of masks from mask_nc to sample with
        output_dir (Path): directory of output
        year (Optional[int]): year of datastream file
        variable (str): variable of datastream file (ie. Discharge, Temperature..)
        time_step (str): annual, monthly, daily, alt, or dlt
        csv_name (str): Name of resulting sampled csv
        cell_area (Optional[np.ndarray]): Cell Area grid corresponding to mask. Needed for polygon masks to calculate weighted zonal mean.
    """

    assert "gdbn" in network.name.split(".", 1)[-1], "Network must be gdbn"
    ds = gdbc_to_ds_buffer(file_path, network)
    sample_ds(mask_nc, ds, mask_layers, output_dir, year, variable, time_step, csv_name=csv_name, cell_area=cell_area)  # type: ignore
