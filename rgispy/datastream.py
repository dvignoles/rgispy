import datetime
import gzip
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


def openDS(name, compressed=False):
    if compressed:
        ifile = gzip.open(name, "rb")  # as ifile
    else:
        ifile = open(name, "rb")  # as ifile:
    return ifile


def openDSbuffred(name, buffer_size, compressed=False):
    if compressed:
        ifile = gzip.open(name, "rb")  # as ifile
    else:
        ifile = open(name, "rb", buffer=buffer_size)  # as ifile:
    return ifile


def headDS(ifile):

    forty = ifile.read(40)
    dump40 = MFdsHeader.from_buffer_copy(forty)
    Type = dump40.Type
    # Type values 5:short,6:long,7:float,8:double
    if Type > 6:
        NoData = dump40.Missing.Float
    else:
        NoData = dump40.Missing.Int

    npType = _npType(Type)

    # print(dump40.Date.decode())

    Date = datetime.datetime.strptime(dump40.Date.decode(), "%Y-%m-%d")

    Items = dump40.ItemNum

    return Type, npType, NoData, Date, Items


def recordDS(ifile, items, npType, skip=True):
    # Skip the 40 bytes of the record header
    if skip:
        _ = ifile.read(40)
    bytes = items * npType(1).itemsize
    RecordData = np.frombuffer(ifile.read(bytes), dtype=npType)
    return RecordData


# This is translation of the GDBC data type codes into standard
# Numpy type codes
def _npType(nType):
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


def nDays(year):
    p = pd.Period("{}-01-01".format(year))
    if p.is_leap_year:
        days = 366
    else:
        days = 365
    return days


def sample_ds(mask_nc, datastream, mask_layers, output_dir, year, variable):

    # set up masks
    dsMasks = xa.open_dataset(mask_nc)
    CellID = np.nan_to_num(dsMasks["ID"].data, copy=True, nan=0.0).astype("int32")
    nRecords = nDays(year)

    inFileID = datastream
    rgisType, npType, NoData, Date, Cells = headDS(inFileID)

    masks = []
    for m in mask_layers:
        Mask = dsMasks[m].data
        MaskType = dsMasks[m].attrs["Type"]
        MaskValues = Mask.flatten()
        MaskValues = MaskValues[~np.isnan(MaskValues)].astype("int")
        MaskValues = list(set(MaskValues))

        OutputPath = output_dir.joinpath(m)
        OutputPath.mkdir(exist_ok=True)

        if MaskType == "Polygon":
            dfOut = pd.DataFrame(index=MaskValues)
        elif MaskType == "Point":
            dfOut = pd.DataFrame(
                index=MaskValues,
                columns=pd.date_range(
                    start="1/1/{}".format(year), end="1/08/{}".format(year), freq="D"
                ),
            )

        masks.append((m, Mask, MaskType, MaskValues, OutputPath, dfOut))

    # bRecord = 40 + Cells * npType(1).itemsize
    for day in range(0, nRecords):
        # if day % 20 == 0:
        #     print(day,end='')
        if day != 0:
            dummy1, dummy1, dummy1, Date, dummy1 = headDS(inFileID)
            # Data = recordDS(inFileID, Cells, npType)
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
