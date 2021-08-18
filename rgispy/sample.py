import datetime
import subprocess as sp
from ctypes import Structure, Union, c_char, c_double, c_int, c_short

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


class MFmissing(Union):
    _fields_ = [("Int", c_int), ("Float", c_double)]


class MFdsHeader(Structure):
    _fields_ = [
        ("Swap", c_short),
        ("Type", c_short),
        ("ItemNum", c_int),
        ("Missing", MFmissing),
        ("Date", c_char * 24),
    ]


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

    if time_step == "daily":
        date_format = "%Y-%m-%d"
    elif time_step == "monthly":
        date_format = "%Y-%m"
    else:
        date_format = "%Y"

    Date = datetime.datetime.strptime(dump40.Date.decode(), date_format)

    Items = dump40.ItemNum

    return Type, npType, NoData, Date, Items


def recordDS(ifile, items, npType, skip=True):
    # Skip the 40 bytes of the record header
    if skip:
        _ = ifile.read(40)
    bytes = items * npType(1).itemsize
    RecordData = np.frombuffer(ifile.read(bytes), dtype=npType)
    return RecordData


def _npType(nType):
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


def n_records(year, time_step):
    """Get number of expected records in datastream based on time_step

    Args:
        year (int): year of datastream file
        time_step (str): annual, monthly, or daily

    Returns:
        (int): number of records (ex: 365 for daily non-leap year datastream)
    """
    assert time_step in [
        "annual",
        "monthly",
        "daily",
    ], "time_step must be annual monthly or daily"

    if time_step == "annual":
        return 1
    elif time_step == "monthly":
        return 12
    else:
        p = pd.Period("{}-01-01".format(year))
        if p.is_leap_year:
            days = 366
        else:
            days = 365
        return days


def gdbc_to_ds_buffer(gdbc, network):
    """Get buffered fileobject of datastream from gdbc using network gdbn as template via rgis2ds command

    Args:
        gdbc (Path): gdbc file path
        network (Path): gdbn file path

    Returns:
        (io.BufferedReader): buffered reader of output datastream
    """
    cmd = "rgis2ds --template {network} {gdbc}".format(
        network=network, gdbc=gdbc
    ).split()
    p = sp.Popen(cmd, stdout=sp.PIPE)

    return p.stdout


def get_masks(mask_ds, mask_layers, output_dir, year, time_step):

    masks = []
    for m in mask_layers:
        Mask = mask_ds[m].data
        MaskType = mask_ds[m].attrs["Type"]
        MaskValues = Mask.flatten()
        MaskValues = MaskValues[~np.isnan(MaskValues)].astype("int")
        MaskValues = list(set(MaskValues))

        OutputPath = output_dir.joinpath(m)
        OutputPath.mkdir(exist_ok=True)

        if time_step == "daily":
            date_cols = pd.date_range(
                start="1/1/{}".format(year), end="12/31/{}".format(year), freq="D"
            )
        elif time_step == "monthly":
            date_cols = pd.date_range(
                start="1/1/{}".format(year), end="12/31/{}".format(year), freq="MS"
            )
        else:
            date_cols = pd.date_range(
                start="1/1/{}".format(year), end="12/31/{}".format(year), freq="YS"
            )

        if MaskType == "Polygon":
            dfOut = pd.DataFrame(index=MaskValues, columns=date_cols)
        elif MaskType == "Point":
            dfOut = pd.DataFrame(index=MaskValues, columns=date_cols)

        masks.append((m, Mask, MaskType, MaskValues, OutputPath, dfOut))

        return masks


def sample_ds(mask_nc, datastream, mask_layers, output_dir, year, variable, time_step):

    # set up masks
    mask_ds = xa.open_dataset(mask_nc)
    CellID = np.nan_to_num(mask_ds["ID"].data, copy=True, nan=0.0).astype("int32")
    nRecords = n_records(year, time_step)

    inFileID = datastream
    rgisType, npType, NoData, Date, Cells = headDS(inFileID, time_step)

    masks = get_masks(mask_ds, mask_layers, output_dir, year, time_step)

    for day in range(0, nRecords):
        if day != 0:
            dummy1, dummy1, dummy1, Date, dummy1 = headDS(inFileID, time_step)

        Data = recordDS(inFileID, Cells, npType, skip=False)
        # We add a NoData entry at the beginning of the data array, so that
        # ID = 0 (e.g. the NoData values of the rgis network) will map to NoData...
        Data = np.insert(Data, 0, NoData)
        Data = Data[CellID.flatten()].reshape(CellID.shape)
        if rgisType <= 6:
            _ = Data.astype("float")
        Data[Data == NoData] = np.nan
        # And now we can calculate the statistics for this layer (one day) and add it to the
        # output dataframe as an additional column...

        for m, Mask, MaskType, MaskValues, OutputPath, dfOut in masks:
            if MaskType == "Polygon":
                # TODO deal with polygon masks
                pass
                # For instance getting the mean of the variable for each
                # region:
                dfOut["mean_{}".format(Date.strftime("%Y-%m-%d"))] = [
                    Data[Mask == i].mean() for i in MaskValues
                ]
                # or the sum:
                dfOut["sum_{}".format(Date.strftime("%Y-%m-%d"))] = [
                    Data[Mask == i].sum() for i in MaskValues
                ]
            elif MaskType == "Point":
                dfOut[Date] = pd.DataFrame(
                    data=Data[~np.isnan(Mask)],
                    index=Mask[~np.isnan(Mask)].astype("int"),
                    columns=["values"],
                )

    for m, _, _, _, OutputPath, dfOut in masks:
        dfOut.to_csv(OutputPath.joinpath("{}_{}.csv".format(variable, year)))
