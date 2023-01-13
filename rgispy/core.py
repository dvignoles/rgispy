"""RGIS Rcommand wrapper functions"""

import gzip
import os
import shutil
import struct
import subprocess as sp
import tempfile
import typing
import warnings
from ctypes import Structure
from ctypes import Union as c_Union
from ctypes import c_char, c_double, c_int, c_short
from datetime import datetime
from functools import reduce
from io import BufferedIOBase, BytesIO, StringIO
from os import PathLike
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from . import util

# network fields (added to table by pntSTNChar)
CHAR_FIELDS = [
    "CellID",
    "BasinID",
    "BasinName",
    "Order",
    "Color",
    "NumberOfCells",
    "STNMainstemLength",
    "STNInterStationArea",
    "NextStation",
    "STNCatchmentArea",
]


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


# types of file objects supported
ftype = typing.Union[bytes, str, Path, PathLike, BufferedIOBase]


def _check_ftype_code(f: ftype) -> int:
    """Check if bytes, path, or file descriptor"""

    # bytes
    if isinstance(f, bytes) or f is None:
        return 1
    # Path
    if isinstance(f, PathLike) or isinstance(f, str):
        return 2
    # list[Path]
    # some commands take multiple files as input arguments
    if isinstance(f, list) or isinstance(f, tuple):
        is_path_list = all([isinstance(p, Path) for p in f])
        if is_path_list:
            return 4
        else:
            return -1
    # binary file descriptor
    if isinstance(f, BufferedIOBase):
        return 3

    return -1


def _do_rgiscmd(cmd_configured, finput, foutput):
    """Run rgis command as subprocess on variable file like types
    (bytes, path, file descriptors)"""
    finput_code = _check_ftype_code(finput)
    foutput_code = _check_ftype_code(foutput)

    assert finput_code != -1, f"Unsupported input {type(finput)}"
    assert foutput_code != -1, f"Unsupported output {type(foutput)}"
    assert foutput_code != 4, "Unsupported output list of paths"

    # bytes/bytes
    if finput_code == 1 and foutput_code == 1:
        p = sp.run(cmd_configured.split(), input=finput, stdout=sp.PIPE)
        return p.stdout
    # bytes/Path
    if finput_code == 1 and foutput_code == 2:
        cmd_configured += f" - {foutput}"
        p = sp.run(cmd_configured.split(), input=finput)
        return foutput
    # bytes/fd
    if finput_code == 1 and foutput_code == 3:
        foutput.seek(0)
        p = sp.run(cmd_configured.split(), input=finput, stdout=foutput)
        foutput.flush()
        foutput.seek(0)
        return foutput
    # Path/bytes
    if finput_code == 2 and foutput_code == 1:
        cmd_configured += f" {finput}"
        p = sp.run(cmd_configured.split(), stdout=sp.PIPE)
        return p.stdout
    # Path/Path
    if finput_code == 2 and foutput_code == 2:
        cmd_configured += f" {finput} {foutput}"
        sp.run(cmd_configured.split())
        return foutput
    # Path/fd
    if finput_code == 2 and foutput_code == 3:
        foutput.seek(0)
        cmd_configured += f" {finput}"
        p = sp.run(cmd_configured.split(), stdout=foutput)
        foutput.flush()
        foutput.seek(0)
        return foutput
    # fd/bytes
    if finput_code == 3 and foutput_code == 1:
        finput.seek(0)
        p = sp.Popen(cmd_configured.split(), stdin=sp.PIPE, stdout=sp.PIPE)
        shutil.copyfileobj(finput, p.stdin)
        p.wait()
        finput.seek(0)
        return p.stdout.read()
    # fd/Path
    if finput_code == 3 and foutput_code == 2:
        finput.seek(0)
        cmd_configured += f" - {foutput}"
        p = sp.Popen(cmd_configured.split(), stdin=sp.PIPE)
        shutil.copyfileobj(finput, p.stdin)
        p.wait()
        finput.seek(0)
        return foutput
    # fd/fd
    if finput_code == 3 and foutput_code == 3:
        finput.seek(0)
        foutput.seek(0)
        p = sp.Popen(cmd_configured.split(), stdin=sp.PIPE, stdout=foutput)
        shutil.copyfileobj(finput, p.stdin)
        p.wait()
        finput.seek(0)
        foutput.flush()
        foutput.seek(0)
        return foutput
    # list[Path]/Path
    if finput_code == 4 and foutput_code == 2:
        finput_str = " ".join([str(p) for p in finput])
        cmd_configured += f" {finput_str} {foutput}"
        sp.run(cmd_configured.split())
        return foutput
    # list[Path]/fd
    if finput_code == 4 and foutput_code == 3:
        finput_str = " ".join([str(p) for p in finput])
        foutput.seek(0)
        cmd_configured += f" {finput_str}"
        p = sp.run(cmd_configured.split(), stdout=foutput)
        foutput.flush()
        foutput.seek(0)
        return foutput
    # list[Path]/bytes
    if finput_code == 4 and foutput_code == 1:
        finput_str = " ".join([str(p) for p in finput])
        cmd_configured += f" {finput_str}"
        p = sp.run(cmd_configured.split(), stdout=sp.PIPE)
        return p.stdout


def _clean_flags(flags):
    """Extract from RgisFile types if necessary"""
    # TODO: if RgisFile is not a path, save to temporary path for command usage
    if flags is None:
        return None

    cleaned = []
    for i, f in enumerate(flags):
        if isinstance(f[1], RgisFile):
            cleaned.append((f[0], f[1]._fref))
        else:
            cleaned.append((f[0], f[1]))

    return cleaned


def _clean_path_list(path_list):
    """Clean list of path-likes into list of existing paths"""
    clean = [Path(p) for p in path_list]
    exists = all([p.exists() for p in clean])
    assert exists

    return clean


def _assert_extension(rgis_path: Path):

    # extract from rgis objects
    if isinstance(rgis_path, RgisFile):
        rgis_path = rgis_path._fref
    rgis_path = Path(rgis_path)

    valid = [".gdbp", ".gdbd", ".gdbc", ".gdbt", ".gdbl", ".gdbn", ".ds", ".gds"]
    valid_gzip = [v + ".gz" for v in valid]
    valid += valid_gzip
    assert rgis_path.suffix != "", f"{rgis_path.name} must have file extension"
    assert (
        "." + rgis_path.name.split(".", 1)[-1] in valid
    ), f"{rgis_path.name} must have extension in {valid}"


def _assert_ftype(fref):
    assert (
        _check_ftype_code(fref) != -1
    ), f"{type(fref)} is invalid initialization type. Must initialize with instance"
    "of bytes, Path, or BufferedIOBase."


def _validate_fref(fref, check_exists=True):
    _assert_ftype(fref)
    ftype_code = _check_ftype_code(fref)
    if ftype_code == 2:
        # ensure Path type and real
        fref = Path(fref)
        _assert_extension(fref)
        if check_exists:
            assert fref.exists(), f"{fref} does not exist"
    return fref, ftype_code


def _guess_rgis_class(rgis_file: Path):
    if ".gdbt" in rgis_file.suffixes:
        return RgisTable
    if ".gdbp" in rgis_file.suffixes:
        return RgisPoint
    if ".gdbd" in rgis_file.suffixes:
        return RgisPolygon
    if ".gdbl" in rgis_file.suffixes:
        return RgisLine
    if ".gdbn" in rgis_file.suffixes:
        return RgisNetwork
    if (".gds" in rgis_file.suffixes) or (".ds" in rgis_file.suffixes):
        return RgisDataStream
    return None


def _as_rgis_file(rgis_file: Path, ghaas_bin=None, scratch_dir=None):
    guess_class = _guess_rgis_class(rgis_file)
    assert guess_class is not None, "No recognized rgis file extension for {rgis_file}"
    return guess_class(rgis_file, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)


class Rgis:
    def __init__(self, ghaas_bin=None, scratch_dir=None):
        if ghaas_bin is None:
            self.ghaas_bin = Path("/usr/local/share/ghaas/bin")
        else:
            self.ghaas_bin = Path(ghaas_bin)

        # where to put tempfiles
        if scratch_dir is None:
            if "SCRATCH" in os.environ:
                scratch_dir = Path(os.environ["SCRATCH"])
                assert scratch_dir.exists(), f"{scratch_dir} must exist"
            else:
                scratch_dir = Path(tempfile._get_default_tempdir())
                warnings.warn(
                    f"Default system temporary directory  {scratch_dir} will be used"
                    "for temporary file storage."
                    "Set scratch_dir or $SCRATCH to override."
                )
        else:
            scratch_dir = Path(scratch_dir)
            assert scratch_dir.exists(), f"{scratch_dir} must exist"

        # rgispy_tmp sub dir
        self.scratch_root = scratch_dir
        if "rgispy_tmp" not in scratch_dir.parts:
            self.scratch_dir = scratch_dir.joinpath("rgispy_tmp")
            self.scratch_dir.mkdir(exist_ok=True)
        else:
            self.scratch_dir = scratch_dir

    def run_rgiscmd(self, cmd, finput, flags=None, foutput=None):
        cmd_path = self.ghaas_bin.joinpath(cmd)

        flags = _clean_flags(flags)
        flags_str = (
            reduce(lambda x, y: f"{x} {y[0]} {y[1]}", flags, "")
            if flags is not None
            else None
        )

        # extract from rgis objects
        if isinstance(finput, RgisFile):
            finput = finput._fref
        if isinstance(foutput, RgisFile):
            foutput = foutput._fref

        if flags_str is not None:
            cmd_configured = f"{cmd_path} {flags_str}"
        else:
            cmd_configured = f"{cmd_path}"
        return _do_rgiscmd(cmd_configured, finput, foutput)

    def _temp_rgisfile(self, name=None, suffix=None):

        prefix = "rgispy_" if name is None else f"rgispy_{name}_"
        _temp = tempfile.NamedTemporaryFile(
            prefix=prefix, dir=self.scratch_dir, suffix=suffix
        )
        return _temp

    def rgis2mapper(self, network, sampler, mapper):
        flags = [
            ("--domain", f"{network}"),
        ]
        mapper_out = self.run_rgiscmd(
            "rgis2mapper", sampler, foutput=mapper, flags=flags
        )

        return mapper_out

    def rgis2domain(self, rgisdata, output_domain_ds):
        domain = self.run_rgiscmd("rgis2domain", rgisdata, foutput=output_domain_ds)
        return domain

    def rgis2ds(self, rgisdata, network, output_ds):
        flags = [
            ("--template", f"{network}"),
        ]
        self.run_rgiscmd("rgis2ds", rgisdata, flags=flags, foutput=output_ds)
        return RgisDataStream(output_ds)

    def rgis2netcdf(self, rgisdata, netcdf: Path):
        self.run_rgiscmd("rgis2netcdf", rgisdata, foutput=netcdf)


class TableMixin:
    def tbl_add_xy(self, xfield="XCoord", yfield="YCoord", table="DBItems"):
        flags = [("-a", table), ("-x", xfield), ("-y", yfield)]
        new_rf = self.run_rgiscmd("tblAddXY", self._fref, flags=flags)
        self.set_fref(new_rf)

    def tbl_redef_field(self, field, rename, table="DBItems"):
        flags = [("-a", table), ("-f", field), ("-r", rename)]
        new_rf = self.run_rgiscmd("tblRedefField", self._fref, flags=flags)
        self.set_fref(new_rf)

    def tbl_delete_field(self, field, table="DBItems"):
        flags = [("-a", table), ("-f", field), ("-a", table)]
        new_rf = self.run_rgiscmd("tblDeleteField", self._fref, flags=flags)
        self.set_fref(new_rf)

    def tbl_join_tables(
        self,
        dataset: Path,
        out_dataset: Path,
        relate_table=None,
        join_table=None,
        relate_field=None,
        join_field=None,
    ):

        dataset = Path(dataset)
        out_dataset = Path(out_dataset)
        _assert_extension(dataset)
        _assert_extension(out_dataset)

        flags = [
            ("-a", dataset),
        ]

        if relate_table is not None:
            flags.append(("-e", relate_table))
        if join_table is not None:
            flags.append(("-o", join_table))
        if relate_field is not None:
            flags.append(("-r", relate_field))
        if join_field is not None:
            flags.append(("-j", join_field))

        self.run_rgiscmd("tblJoinTables", self._fref, flags=flags, foutput=out_dataset)

        return _as_rgis_file(
            out_dataset, ghaas_bin=self.ghaas_bin, scratch_dir=self.scratch_dir
        )


class RgisFile(Rgis, TableMixin):
    def set_fref(self, new_fref):
        self._fref, self._ftype_code = _validate_fref(new_fref)

    def __init__(self, fref: ftype, ghaas_bin=None, scratch_dir=None):
        super().__init__(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
        self.set_fref(fref)

    def _is_compressed(self):
        assert (
            self._ftype_code == 2
        ), "Cannot determine if in memory objects are compressed"
        # NOTE: You probably can by examining the start of the stream
        if self._fref.name.endswith(".gz"):
            return True
        else:
            return False

    def _to_buffer(
        self,
    ):
        """Convert self from path, gzipped path, or in memory bytes
        to file descriptor.
        """

        assert self._ftype_code in [1, 2, 3]

        # bytes
        if self._ftype_code == 1:
            self.set_fref(BytesIO(self._fref))
            return True
        # pathlike
        elif self._ftype_code == 2:
            if self._is_compressed():
                self.set_fref(gzip.open(self._fref, mode="rb"))
            else:
                self.set_fref(open(self._fref, "rb"))
            return True
        # buffer
        elif self._ftype_code == 3:
            return True

    def load(self):
        """Load into memory if file path"""
        if self._ftype_code != 2:
            warnings.warn("Source already loaded or reference to file descriptor.")
        else:
            _assert_extension(self._fref)
            if self._fref.suffix == ".gz":
                with gzip.open(self._rfef, "rb") as f:
                    self.set_fref(f.read())
            else:
                with open(self._rfef, "rb") as f:
                    self.set_fref(f.read())

    def _save_file(self, full_path, gzipped=False, replace_path=False):
        _assert_extension(full_path)

        # check doesn't already exist OR if ok to replace
        assert (not full_path.exists()) or (
            replace_path
        ), f"{full_path} exists, set replace_path=True if you wish to overwrite it"

        if not full_path.parent.exists():
            full_path.parent.mkdir(parents=True)

        if self._ftype_code == 1:
            if gzipped:
                with gzip.open(full_path, mode="wb") as f:
                    f.write(self._fref)
            else:
                with open(full_path, mode="wb") as f:
                    f.write(self._fref)
        return full_path

    def to_file(self, file_path, gzipped=False, replace_path=False):
        # only relevant for in memory files
        assert self._ftype_code != 3, "Source is already file descriptor"
        assert self._ftype_code != 2, f"Source already exists at {self._fref}"

        file_path = Path(file_path)
        output_dir = file_path.parent
        extension = "".join(file_path.suffixes)
        name = file_path.stem.split(".")[0]
        full_name = f"{name}{extension}"

        # overwrite flag if supplied .gz
        if extension.endswith(".gz"):
            if gzipped is False:
                warnings.warn(
                    ".gz file extension supplied with gzipped=False. gzip compression \
                will be used anyway"
                )

            gzipped = True

        if gzipped and (not full_name.endswith(".gz")):
            full_name += ".gz"

        full_path = output_dir.joinpath(full_name)

        self._save_file(file_path, replace_path=replace_path, gzipped=gzipped)
        return _as_rgis_file(
            full_path, ghaas_bin=self.ghaas_bin, scratch_dir=self.scratch_dir
        )


class RgisTable(RgisFile, TableMixin):
    def __init__(
        self, fref: ftype, table_type: str = None, ghaas_bin=None, scratch_dir=None
    ):
        super().__init__(fref, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

        self.table_type = table_type if table_type is not None else "DBItems"
        flags = [("-a", table_type)] if table_type is not None else None
        # save table as bytes
        self.table = self.run_rgiscmd("rgis2table", fref, flags=flags)

    def df(self):
        df = pd.read_csv(StringIO(self.table.decode()), sep="\t")

        # ensure cellids are integers
        cellid_floats = [
            c for c in df.columns if "cellid" in c.lower() if df[c].dtype == float
        ]
        for c in cellid_floats:
            df.loc[:, c] = df[c].astype(pd.Int64Dtype())

        return df

    @classmethod
    def from_df(cls, df: pd.DataFrame, ghaas_bin=None, scratch_dir=None):
        buf = StringIO()
        df.to_csv(buf, sep="\t")

        # probably not a clean method but it's simple
        rgis = Rgis(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
        table = rgis.run_rgiscmd(
            "table2rgis",
            buf.getvalue().encode(),
        )

        return cls(table, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)


class RgisLayer(RgisFile, TableMixin):
    """RgisLayers consist of points/lines/zones/grids + one or more tables"""

    def _load_meta(self):
        """Get origin and metadata from rgis file binary"""
        self._to_buffer()
        ifile = self._fref
        ifile.seek(40)
        llx = struct.unpack("d", ifile.read(8))[0]
        lly = struct.unpack("d", ifile.read(8))[0]
        ifile.read(8)
        titleLen = struct.unpack("h", ifile.read(2))[0]
        title = ifile.read(titleLen).decode()
        meta = {"title": title, "llx": llx, "lly": lly}
        ifile.read(9)
        doc_len = struct.unpack("h", ifile.read(2))[0]
        _ = ifile.read(doc_len).decode()
        ifile.read(25)
        read_more = True
        while read_more:
            infoLen = struct.unpack("h", ifile.read(2))[0]
            infoRec = ifile.read(infoLen).decode()
            if infoRec == "Data Records":
                read_more = False
                break
            ifile.read(1)
            valLen = struct.unpack("h", ifile.read(2))[0]
            if valLen == 44:
                ifile.read(26)
            elif valLen == 48:
                ifile.read(30)
            valLen = struct.unpack("h", ifile.read(2))[0]
            valRec = ifile.read(valLen).decode()
            meta[infoRec.lower()] = valRec
            ifile.read(1)

        meta = pd.DataFrame.from_records(
            [
                meta,
            ]
        )
        meta.loc[:, "ID"] = 1
        return meta

    def to_table(self, table_type="DBItems", output=None) -> RgisTable:
        table = RgisTable(
            self._fref,
            table_type,
            ghaas_bin=self.ghaas_bin,
            scratch_dir=self.scratch_dir,
        )
        return table

    def to_netcdf(self, netcdf: Path):
        return self.run_rgiscmd("rgis2netcdf", self._fref, foutput=netcdf)


class DBItemsMixin:
    def db_items(
        self,
    ) -> RgisTable:
        return self.to_table(table_type="DBItems")


class DBLayersMixin:
    def db_layers(
        self,
    ) -> RgisTable:
        return self.to_table(table_type="DBLayers")

    def _load_geo(self):
        meta = self._load_meta()
        dbl = self.db_layers().df()
        df = pd.merge(meta, dbl, left_on="ID", right_on="ID")
        keep = [
            "geodomain",
            "llx",
            "lly",
            "RowNum",
            "ColNum",
            "ValueType",
            "ValueSize",
            "CellWidth",
            "CellHeight",
        ]
        return df[keep]

    def affine(self):
        """Return affine transformation of network in gdal format

        Transform will match with flipped output (np.flipud(arr)) of RgisDataStream.iter()
        """
        geo = self._load_geo()
        # rgis reports cell dimensions with precision 6
        uly = round(
            geo.lly.tolist()[0]
            + (geo.RowNum.tolist()[0] * abs(geo.CellHeight.tolist()[0])),
            6,
        )

        ulx = round(geo.llx.tolist()[0], 6)
        aff = (
            ulx,
            geo.CellWidth.tolist()[0],
            0.0,
            uly,
            0.0,
            -1 * geo.CellHeight.tolist()[0],
        )
        return aff


# TODO: Add XY to dbcells by default?
class DBCellsMixin:
    def db_cells(
        self,
    ) -> RgisTable:
        return self.to_table(table_type="DBCells")

    _dbcells_cols = {
        "ID": {"dtype": "int32", "scale_factor": 1, "_FillValue": -9999, "zlib": True},
        "ToCell": {
            "dtype": "int8",
            "scale_factor": 1,
            "_FillValue": -9999,
            "zlib": True,
        },
        "FromCell": {
            "dtype": "int8",
            "scale_factor": 1,
            "_FillValue": -9999,
            "zlib": True,
        },
        "Order": {
            "dtype": "int8",
            "scale_factor": 1,
            "_FillValue": -9999,
            "zlib": True,
        },
        "BasinID": {
            "dtype": "int16",
            "scale_factor": 1,
            "_FillValue": -9999,
            "zlib": True,
        },
        "BasinCells": {
            "dtype": "int16",
            "scale_factor": 1,
            "_FillValue": -9999,
            "zlib": True,
        },
        "Travel": {
            "dtype": "int16",
            "scale_factor": 1,
            "_FillValue": -9999,
            "zlib": True,
        },
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

    def cells2np(self, dbcell_columns=None):

        """Return DBCells columns as numpy ndarray in correct grid shape"""

        if dbcell_columns is None:
            dbcell_columns = list(self._dbcells_cols.keys())

        self.tbl_add_xy(table="DBCells", xfield="CellXCoord", yfield="CellYCoord")
        net = self.db_cells().df()
        x = net["CellXCoord"].sort_values(ascending=True).unique()
        y = net["CellYCoord"].sort_values(ascending=True).unique()

        assert all(
            [c in self._dbcells_cols.keys() for c in dbcell_columns]
        ), f"Columns must be in {self._dbcells_cols}"

        result = {"x": x, "y": y}
        for col in dbcell_columns:
            df_pv = net[[col, "CellXCoord", "CellYCoord"]].pivot(
                index="CellYCoord", columns="CellXCoord"
            )

            result[col] = df_pv.values
        return result


class RgisPoint(RgisLayer, DBItemsMixin):
    def __init__(self, fref: ftype, ghaas_bin=None, scratch_dir=None):
        super().__init__(fref, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

    @classmethod
    def from_df(cls, df, xcol, ycol, ghaas_bin=None, scratch_dir=None):
        tbl = RgisTable.from_df(df, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
        flags = [("-x", xcol), ("-y", ycol)]
        rgis = Rgis(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
        gdbp = rgis.run_rgiscmd("tblConv2Point", tbl._fref, flags=flags)
        return cls(gdbp)

    def gdf(self, x="XCoord", y="YCoord", table="DBItems"):
        self.tbl_add_xy(xfield=x, yfield=y, table=table)
        df = self.db_items().df()
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(x=df[x], y=df[y]), crs=4326
        )
        return gdf

    def pnt_stn_char(self, network, suffix=None):

        flags = [
            ("-n", network),
        ]
        new_rf = self.run_rgiscmd("pntSTNChar", self._fref, flags=flags)
        self.set_fref(new_rf)

        if suffix is not None:
            for c in CHAR_FIELDS:
                self.tbl_redef_field(c, f"{c}{suffix}")

    def pnt_stn_coord(
        self, network, field=None, tolerance=None, radius=None, cfield="SubbasinArea"
    ):
        flags = [
            ("-M", "fixed"),
            ("-c", cfield),
            ("--network", network),
        ]

        if field is not None:
            flags.append(("--field", field))

        if tolerance is not None:
            flags.append(("--tolerance", tolerance))

        if radius is not None:
            flags.append(("--radius", radius))

        new_rf = self.run_rgiscmd("pntSTNCoord", self._fref, flags=flags)
        self.set_fref(new_rf)


class RgisPolygon(RgisLayer, DBItemsMixin, DBLayersMixin):
    def __init__(self, fref: ftype, ghaas_bin=None, scratch_dir=None):
        super().__init__(fref, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

    @classmethod
    def from_gdf():
        # see rgisConverters.sh
        # gdal_rasterize -> grdImport tbl2rgis?
        # I think a gdbd is just a gdbc with grouped cells
        pass

    def to_gdf():
        # rgis2ascii -> gdal_translate (tif) -> gdal_polygonize (rasterio ?)
        pass


class RgisLine(RgisLayer, DBItemsMixin, DBLayersMixin):
    def __init__(self, fref: ftype, ghaas_bin=None, scratch_dir=None):
        super().__init__(fref, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)


class RgisNetwork(RgisLayer, DBItemsMixin, DBLayersMixin, DBCellsMixin):
    def __init__(self, fref: ftype, ghaas_bin=None, scratch_dir=None):
        super().__init__(fref, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

    def netCells2Grid(self, fieldname, out_grid=None):
        flags = [
            ("-f", fieldname),
        ]

        output = self.run_rgiscmd(
            "netCells2Grid", self._fref, foutput=out_grid, flags=flags
        )
        return RgisGrid(output)


class RgisDataStream(RgisFile):
    def __init__(self, fref: ftype, ghaas_bin=None, scratch_dir=None):
        super().__init__(fref, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

    def _head(self, time_step):
        self._to_buffer()

        forty = self._fref.read(40)
        dump40 = MFdsHeader.from_buffer_copy(forty)
        _type = dump40.Type
        # Type values 5:short,6:long,7:float,8:double
        if _type > 6:
            na = dump40.Missing.Float
        else:
            na = dump40.Missing.Int

        np_type = _npType(_type)

        date_raw = dump40.Date.decode()
        time_step = time_step.lower()
        date_format = util.get_date_format(time_step)

        if date_format is not None:
            dt = datetime.strptime(date_raw, date_format)
        else:
            dt = date_raw

        items = dump40.ItemNum

        return _type, np_type, na, dt, items

    def _record(self, items, np_type, skip=True):
        self._to_buffer()

        # Skip the 40 bytes of the record header
        if skip:
            _ = self._fref.read(40)
        num_bytes = items * np_type(1).itemsize
        record_data = np.frombuffer(self._fref.read(num_bytes), dtype=np_type)
        return record_data

    def iter(self, year, time_step, network=None, cell_id=None):
        """Generator of (ndarray, datetime) record tuples over datastream file object

        Args:
        Yields:
            Generator[tuple[np.ndarray, datetime.datetime]]:
            (datetime, data) record pairs
        """

        assert (network is not None) or (
            cell_id is not None
        ), "Must specify network or cellid numpy grid"

        self._to_buffer()

        if cell_id is None:
            if not isinstance(network, RgisNetwork):
                network = RgisNetwork(network)

            cells = network.cells2np(
                [
                    "ID",
                ]
            )
            cell_id = cells["ID"]

        cell_id_0 = np.nan_to_num(cell_id, copy=True, nan=0.0).astype("int32")

        rgis_type, np_type, no_data, dt, _cells = self._head(time_step)

        for day in range(0, util.n_records(year, time_step)):
            if day != 0:
                _, _, _, dt, _ = self._head(time_step)

            data = self._record(_cells, np_type, skip=False)
            # We add a NoData entry at the beginning of the data array, so that
            # ID = 0 (e.g. the NoData values of the rgis network) will map to NoData...
            data = np.insert(data, 0, no_data)

            data = data[cell_id_0.flatten()].reshape(cell_id_0.shape)
            if rgis_type <= 6:
                _ = data.astype("float")
            data[data == no_data] = np.nan

            yield dt, data

    def dsAggregate(self, step, out_ds, aggregate="avg"):
        aggs = ["avg", "sum"]
        steps = ["day", "month", "year"]
        assert aggregate.lower() in aggs, f"aggregate must be on of {aggs}"
        assert step.lower() in steps, f"aggregate must be on of {steps}"

        flags = [
            ("--step", step),
            ("--aggregate", aggregate),
        ]

        self._to_buffer()
        self.run_rgiscmd("dsAggregate", self._fref, foutput=out_ds, flags=flags)
        return RgisDataStream(
            out_ds, ghaas_bin=self.ghaas_bin, scratch_dir=self.scratch_dir
        )

    def dsSampling(self, domain_file, mapper_file, title="rgispy_sampling"):
        flags = [
            ("--domainfile", domain_file),
            ("--mapper", mapper_file),
            ("--title", title),
        ]

        _sample = self.run_rgiscmd("dsSampling", self._fref, flags=flags)
        sample_table = RgisTable(
            _sample, ghaas_bin=self.ghaas_bin, scratch_dir=self.scratch_dir
        )
        return sample_table

    def to_rgis(self, template, output_grid, title=None, subject=None, domain=None):
        self._to_buffer()
        flags = [
            ("--template", template),
        ]
        if title is not None:
            flags.append(("--title", title))
        else:
            flags.append(("--title", "rgispy_ds2rgis"))

        if subject is not None:
            flags.append(("--subject", subject))

        if domain is not None:
            flags.append(("--domain", domain))

        self.run_rgiscmd("ds2rgis", self._fref, foutput=output_grid, flags=flags)
        return RgisGrid(output_grid)


class RgisGrid(RgisLayer, DBLayersMixin, DBItemsMixin):
    def __init__(self, fref: ftype, ghaas_bin=None, scratch_dir=None):
        super().__init__(fref, ghaas_bin, scratch_dir)

    def grdTSAgg(self, agg: str, step: str, out_gdbc_gz: Path):
        """Aggregate GDBC To coarser temporal resolution

        Args:
            gdbc (Path): RGIS Grid Coverage (gdbc[.gz])
            agg (str): [avg|max|min|sum] aggregation method
            step (str): [year|month|day|hour] temporal step to aggregate to
            out_gdbc_gz (Path): RGIS Grid Coverge (gdbc[.gz])
        """

        flags = [
            ("-a", agg),
            ("-e", step),
        ]

        output = self.run_rgiscmd(
            "grdTSAggr", self._fref, foutput=out_gdbc_gz, flags=flags
        )
        return RgisGrid(output)

    def grdBoxAggr(self, out_gdbc: Path, box_size: int, weight="area"):
        """'Upscale' RGIS Grid Coverage to lower resolution (ie 03min -> 06min)

        Args:
            gdbc (Path): RGIS Grid Coverage (gdbc[.gz]) input
            out_gdbc (Path): RGIS Grid Coverage (gdbc[.gz]) output
            box_size (int): Number of input grid cells per output grid cell
            weight (str, optional): [area|cellnum]. How to weight "upscaled" averages.
              Defaults to "area".
        """
        assert weight in ["area", "cellnum"]

        flags = [
            ("-z", box_size),
            ("-w", weight),
        ]

        output = self.run_rgiscmd(
            "grdBoxAggr", self._fref, foutput=out_gdbc, flags=flags
        )

        return RgisGrid(output)

    def grdZoneStats(self, gdbd: Path) -> pd.DataFrame:
        """Sample Zonal Statistics RGIS Grid Coverage  (gdbc[.gz]) with RGIS Polygon
         Coverage (gdbd[.gz])

        Args:
            gdbc (Path): RGIS Grid Coverage (gdbc[.gz])
            gdbd (Path): RGIS Polygon Coverage (gdbd[.gz])

        Returns:
            pd.DataFrame: Zonal Statistcs DataFrame
        """
        flags = [
            ("-z", gdbd),
        ]

        output = self.run_rgiscmd("grdZoneStats", self._fref, flags=flags)
        df = RgisTable(output).df()

        time_cols = df.WeightLayerName.str.split("-", expand=True).astype(int)
        df.loc[:, "Year"] = time_cols[0]
        df.loc[:, "Month"] = time_cols[1]
        df.loc[:, "Day"] = 1
        df.loc[:, "Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df.set_index(["ID", "Date"], inplace=True)
        df.sort_index(inplace=True)
        return df

    def grdCycleMean(
        self, number, output_grd=None, offset=None, title="rgispy_grdCycleMean"
    ):
        flags = [("-t", title), ("-n", number)]

        if offset is not None:
            flags.append(("-o", offset))

        output = self.run_rgiscmd(
            "grdCycleMean", self._fref, foutput=output_grd, flags=flags
        )
        return RgisGrid(output, ghaas_bin=self.ghaas_bin, scratch_dir=self.scratch_root)

    def grdExtractLayers(
        self, first, last, output_grd=None, title="rgispy_grdExtractLayers"
    ):
        flags = [("-t", title), ("-f", first), ("-l", last)]

        output = self.run_rgiscmd(
            "grdExtractLayers", self._fref, foutput=output_grd, flags=flags
        )
        return RgisGrid(output, ghaas_bin=self.ghaas_bin, scratch_dir=self.scratch_root)

    def grdRenameLayers(self, renames, output_grd=None, title="rgispy_grdRenameLayers"):
        flags = [
            ("-t", title),
        ]
        for i, r in enumerate(renames):
            flags.append(("-r", f"{i+1} {r}"))

        output = self.run_rgiscmd(
            "grdRenameLayers", self._fref, foutput=output_grd, flags=flags
        )
        return RgisGrid(output, ghaas_bin=self.ghaas_bin, scratch_dir=self.scratch_root)

    def grdDateLayers(
        self,
        step=None,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        number=None,
        output_grd=None,
        title="rgispy_grdRenameLayers",
    ):
        flags = [
            ("-t", title),
        ]

        if step is not None:
            flags.append(("--step", step))
        if year is not None:
            flags.append(("--year", year))
        if month is not None:
            flags.append(("--month", month))
        if day is not None:
            flags.append(("--day", day))
        if hour is not None:
            flags.append(("--hour", hour))
        if minute is not None:
            flags.append(("--minute", minute))
        if number is not None:
            flags.append(("--number", number))

        output = self.run_rgiscmd(
            "grdDateLayers", self._fref, foutput=output_grd, flags=flags
        )
        return RgisGrid(output, ghaas_bin=self.ghaas_bin, scratch_dir=self.scratch_root)

    def to_datastream(self, network, datastream=None):

        flags = []
        if isinstance(network, RgisNetwork):
            assert (
                network._ftype_code == 2
            ), "network object must be initialized as on disk Path"
            flags.append(("--template", f"{network._fref}"))
        else:
            flags.append(("--template", f"{network}"))

        output = self.run_rgiscmd(
            "rgis2ds", self._fref, flags=flags, foutput=datastream
        )
        return RgisDataStream(
            output, ghaas_bin=self.ghaas_bin, scratch_dir=self.scratch_root
        )


# For commands that don't make sense to shoehorn into object oriented model
class RgisCalculate(Rgis):
    def __init__(self, ghaas_bin=None, scratch_dir=None):
        super().__init__(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

    def _clean_expr(self, expr: str) -> str:
        """remove tabs, new lines from string"""
        import re

        expr = expr.replace("\n", " ")
        expr = expr.replace("\t", " ")
        expr = re.sub(r"\s+", " ", expr)
        return expr

    def _grdCalculate(
        self,
        expression: str,
        stdout: int,
        extent: Path,
        interpolate="surface",
        var_exprs: list[tuple[str]] = None,
        title: str = None,
        subject: str = None,
        domain: str = None,
        verbose: bool = False,
    ) -> sp.CompletedProcess:
        """Perform raster calculation operation on RGIS coverages

        Args:
            expression (str): String expression ie 'x.gdbc + y.gdbc'
                or 'x.gdbc == nodata ? y.gdbc : xgdbc'
            stdout (int): subprocess stdout encoding ex: sp.PIPE
            extent (Path): RGIS coverage to use as bounding box of output
            interpolate (str, optional): [surface|flat] interpolation method.
                Defaults to "surface".
            var_exprs (list[tuple[str]], optional): sub expressions
                to use as variables in main expression. Defaults to None.
            title (str, optional): RGIS metadata title. Defaults to None.
            subject (str, optional): RIGS metadata subject. Defaults to None.
            domain (str, optional): RGIS metadata domain. Defaults to None.
            verbose (bool, optional): Prints output command. Defaults to False.

        Returns:
            sp.CompletedProcess
        """

        cmd_path = self.ghaas_bin.joinpath("grdCalculate")
        cmd = f"{cmd_path} -x {str(extent)} -n {interpolate}".split()

        if title is not None:
            cmd += f"-t {title}".split()

        if subject is not None:
            cmd += f"-u {subject}".split()

        if domain is not None:
            cmd += f"-d {domain}".split()

        if var_exprs is not None:
            for ve in var_exprs:
                cmd.append("-r")
                cmd.append(ve[0])
                cmd.append(self._clean_expr(ve[1]))  # type: ignore

        cmd.append("-c")
        cmd.append(self._clean_expr(expression))

        if verbose:
            print(" ".join(cmd))

        ps = sp.run(cmd, stdout=stdout)
        return ps

    def grdCalculate(self, expression: str, output: Path, **kwargs) -> Path:
        """_summary_

        Args:
            expression (str): String expression ie 'x.gdbc + y.gdbc'
                or 'x.gdbc == nodata ? y.gdbc : xgdbc'
            output_gz (Path): Output gdbc.gz path
            extent (Path): RGIS coverage to use as bounding box of output
            interpolate (str, optional): [surface|flat] interpolation method.
                Defaults to "surface".
            var_exprs (list[tuple[str]], optional): sub expressions to use
                as variables in main expression. Defaults to None.
            title (str, optional): RGIS metadata title. Defaults to None.
            subject (str, optional): RIGS metadata subject. Defaults to None.
            domain (str, optional): RGIS metadata domain. Defaults to None.
            verbose (bool, optional): Prints output command. Defaults to False.

        Returns:
            Path: output_gz
        """
        if not output.parent.exists():
            output.parent.mkdir(parents=True)

        ps = self._grdCalculate(expression, sp.PIPE, **kwargs)

        if output.name.endswith(".gz"):
            with gzip.open(output, "wb") as f:
                f.write(ps.stdout)
        else:
            with open(output, "wb") as f:
                f.write(ps.stdout)
        return output

    def grdAppendLayers(
        self, grids: list[Path], output_grd=None, title="rgispy_grdAppendLayers"
    ):
        flags = [
            ("-t", title),
        ]

        if output_grd is not None:
            flags.append(("-o", str(output_grd)))
            foutput = None
        else:
            foutput = output_grd

        grids = _clean_path_list(grids)

        output = self.run_rgiscmd(
            "grdAppendLayers", grids, foutput=foutput, flags=flags
        )

        if output_grd is not None:
            # list[path] with output as flag (-o) will return None from run_rgiscommand
            # easier just to workaround than handle edge case
            return RgisGrid(
                output_grd, ghaas_bin=self.ghaas_bin, scratch_dir=self.scratch_root
            )
        else:
            return RgisGrid(
                output, ghaas_bin=self.ghaas_bin, scratch_dir=self.scratch_root
            )
