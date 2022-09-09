"""RGIS Rcommand wrapper functions"""

import gzip
import subprocess as sp
import typing
import warnings
from functools import reduce
from io import BufferedIOBase, StringIO
from os import PathLike
from pathlib import Path

import pandas as pd

GIGABYTE = 1000000000

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
    # binary file descriptor
    if isinstance(f, BufferedIOBase):
        return 3

    return -1


def _do_rgiscmd(cmd_configured, finput, foutput):
    """Run rgis command as subprocess on variable file like types
    (bytes, path, file descriptors)"""
    finput_code = _check_ftype_code(finput)
    foutput_code = _check_ftype_code(foutput)

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
        return None
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
        return None
    # fd/bytes
    if finput_code == 3 and foutput_code == 1:
        finput.seek(0)
        p = sp.run(cmd_configured.split(), stdin=finput, stdout=sp.PIPE)
        return p.stdout
    # fd/Path
    if finput_code == 3 and foutput_code == 2:
        finput.seek(0)
        cmd_configured += f" - {foutput}"
        p = sp.run(cmd_configured.split(), stdin=finput)
        return foutput
    # fd/fd
    if finput_code == 3 and foutput_code == 3:
        finput.seek(0)
        foutput.seek(0)
        p = sp.run(cmd_configured.split(), stdin=finput, stdout=foutput)
        return None


class Rgis:
    def __init__(self, ghaas_bin=None):
        if ghaas_bin is None:
            self.ghaas_bin = Path("/usr/local/share/ghaas/bin")
        else:
            self.ghaas_bin = Path(ghaas_bin)

    def run_rgiscmd(self, cmd, finput, flags=None, foutput=None):
        cmd_path = self.ghaas_bin.joinpath(cmd)
        flags_str = (
            reduce(lambda x, y: f"{x} {y[0]} {y[1]}", flags, "")
            if flags is not None
            else None
        )

        if flags_str is not None:
            cmd_configured = f"{cmd_path} {flags_str}"
        else:
            cmd_configured = f"{cmd_path}"
        return _do_rgiscmd(cmd_configured, finput, foutput)

    def assert_extension(self, rgis_path):
        valid = [".gdbp", ".gdbd", ".gdbc", ".gdbt", ".gdbl", ".ds", ".gds"]
        valid_gzip = [v + ".gz" for v in valid]
        valid += valid_gzip
        assert rgis_path.suffix != "", f"{rgis_path.name} must have file extension"
        assert (
            rgis_path.name.split(".", 1)[-1] in valid
        ), f"{rgis_path.name} must have extension in {valid}"

    def assert_ftype(self, fref):
        assert (
            _check_ftype_code(fref) != -1
        ), f"{type(fref)} is invalid initialization type. Must initialize with instance"
        "of bytes, Path, or BufferedIOBase."

    def validate_fref(self, fref):
        self.assert_ftype(fref)
        self._ftype_code = _check_ftype_code(fref)
        if self._ftype_code == 2:
            # ensure Path type and real
            fref_path = Path(fref)
            self.assert_extension(fref)
            assert fref_path.exists(), f"{fref_path} does not exist"
            return fref_path
        else:
            return fref


class RgisTable(Rgis):
    def __init__(self, fref: ftype, table_type: str = None, ghaas_bin=None):
        super().__init__(ghaas_bin)
        fref = self.validate_fref(fref)
        self.table_type = table_type if table_type is not None else "DBItems"
        flags = [("-a", table_type)] if table_type is not None else None
        # save table as bytes
        self.table = self.run_rgiscmd("rgis2table", fref, flags=flags)

    def df(self):
        df = pd.read_csv(StringIO(self.table.decode()), sep="\t")
        return df

    @classmethod
    def from_df(cls, df: pd.DataFrame, ghaas_bin=None):
        buf = StringIO()
        df.to_csv(buf, sep="\t")

        # probably not a clean method but it's simple
        rgis = Rgis(ghaas_bin=ghaas_bin)
        table = rgis.run_rgiscmd(
            "table2rgis",
            buf.getvalue().encode(),
        )

        return cls(table)


class RgisFile(Rgis):
    def set_fref(self, new_fref):
        self._fref = self.validate_fref(new_fref)

    def __init__(self, fref: ftype, ghaas_bin=None):
        super().__init__(ghaas_bin)
        self.set_fref(fref)

    def to_table(self, table_type="DBItems", output=None) -> RgisTable:
        table = RgisTable(self._fref, table_type, ghaas_bin=self.ghaas_bin)
        return table

    def tbl_add_xy(self, xfield="XCoord", yfield="YCoord", table="DBItems"):
        flags = [("-a", table), ("-x", xfield), ("-y", yfield)]
        new_rf = self.run_rgiscmd("tblAddXY", self._fref, flags=flags)
        self.set_fref(new_rf)

    def tbl_redef_field(self, field, rename, table="DBItems"):
        flags = [("-a", table), ("-f", field), ("-r", rename)]
        new_rf = self.run_rgiscmd("tblRedefField", self._fref, flags=flags)
        self.set_fref(new_rf)

    def load(self):
        """Load into memory if file path"""
        if self._ftype_code != 2:
            warnings.warn("Source already loaded or reference to file descriptor.")
        else:
            self.assert_extension(self._fref)
            if self._fref.suffix == ".gz":
                with gzip.open(self._rfef, "rb") as f:
                    self.set_fref(f.read())
            else:
                with open(self._rfef, "rb") as f:
                    self.set_fref(f.read())

    def _to_file(
        self, output_dir_path, name, extension, gzipped=True, replace_path=False
    ):
        # only relevant for in memory files
        assert self._ftype_code != 3, "Source is already file descriptor"
        assert self._ftype_code != 2, f"Source already exists at {self._fref}"

        full_name = f"{name}.{extension}"

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

        full_path = Path(output_dir_path).joinpath(full_name)
        self.assert_extension(full_path)

        assert (full_path.exists()) and (
            not replace_path
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


class RgisPoint(RgisFile):
    def __init__(self, fref: ftype, ghaas_bin=None):
        super().__init__(fref, ghaas_bin)

    def db_items(
        self,
    ) -> RgisTable:
        return self.to_table(table_type="DBItems")

    @classmethod
    def from_df(cls, df, xcol, ycol, ghaas_bin=None):
        tbl = RgisTable.from_df(df, ghaas_bin=ghaas_bin)
        flags = [("--xcoord", xcol), ("--ycoord", ycol)]
        rgis = Rgis(ghaas_bin=ghaas_bin)
        gdbp = rgis.run_rgiscmd("tblAddXY", tbl._fref, flags=flags)
        return cls(gdbp)

    def pnt_stn_char(self, network, suffix=None):

        flags = [
            ("-n", f"{network}"),
        ]
        new_rf = self.run_rgiscmd("pntSTNChar", self._fref, flags=flags)
        self.set_fref(new_rf)

        if suffix is not None:

            char_fields = [
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

            for c in char_fields:
                self.tbl_redef_field(c, f"{c}{suffix}")

    def pnt_stn_coord(self, network, field, tolerance, radius, cfield="SubbasinArea"):
        flags = [
            ("-M", "fixed"),
            ("--tolerance", tolerance),
            ("--radius", radius),
            ("--field", field),
            ("-c", cfield),
            ("--network", network),
        ]
        new_rf = self.run_rgiscmd("pntSTNCoord", self._fref, flags=flags)
        self.set_fref(new_rf)


class RgisGrid(RgisFile):
    # Idea: If not output files supplied to grd functions, use namedTemporaryFile
    pass


class RgisDataStream(Rgis):
    pass


class RgisNetwork:
    pass


class RgisPolygon:
    pass


class RgisOld:
    def __init__(self, ghaas_bin=None):
        if ghaas_bin is None:
            self.ghaas_bin = Path("/usr/local/share/ghaas/bin")
        else:
            self.ghaas_bin = Path(ghaas_bin)

    def grdTSAgg(
        self, gdbc: Path, agg: str, step: str, out_gdbc_gz: Path
    ) -> sp.CompletedProcess:
        """Aggregate GDBC To coarser temporal resolution

        Args:
            gdbc (Path): RGIS Grid Coverage (gdbc[.gz])
            agg (str): [avg|max|min|sum] aggregation method
            step (str): [year|month|day|hour] temporal step to aggregate to
            out_gdbc_gz (Path): RGIS Grid Coverge (gdbc[.gz])

        Returns:
            sp.CompletedProcess
        """

        cmd_path = self.ghaas_bin.joinpath("grdTSAggr")
        cmd = f"{cmd_path} -a {agg} -e {step} {str(gdbc)} {str(out_gdbc_gz)}".split()
        return sp.run(cmd)

    def grdBoxAggr(
        self, gdbc: Path, out_gdbc: Path, box_size: int, weight="area"
    ) -> sp.CompletedProcess:
        """'Upscale' RGIS Grid Coverage to lower resolution (ie 03min -> 06min)

        Args:
            gdbc (Path): RGIS Grid Coverage (gdbc[.gz]) input
            out_gdbc (Path): RGIS Grid Coverage (gdbc[.gz]) output
            box_size (int): Number of input grid cells per output grid cell
            weight (str, optional): [area|cellnum]. How to weight "upscaled" averages.  Defaults to "area".

        Returns:
            sp.CompletedProcess:
        """
        assert weight in ["area", "cellnum"]

        cmd_path = self.ghaas_bin.joinpath("grdBoxAggr")
        cmd = f"{cmd_path} -z {box_size} -w {weight} {gdbc} {out_gdbc}".split()

        p = sp.run(cmd)
        return p

    def grdZoneStats(self, gdbc: Path, gdbd: Path) -> pd.DataFrame:
        """Sample Zonal Statistics RGIS Grid Coverage  (gdbc[.gz]) with RGIS Polygon Coverage (gdbd[.gz])

        Args:
            gdbc (Path): RGIS Grid Coverage (gdbc[.gz])
            gdbd (Path): RGIS Polygon Coverage (gdbd[.gz])

        Returns:
            pd.DataFrame: Zonal Statistcs DataFrame
        """
        cmd_path = self.ghaas_bin.joinpath("grdZoneStats")
        cmd = f"{cmd_path} -z {gdbd} {gdbc}".split()
        p = sp.Popen(cmd, stdout=sp.PIPE)
        output, _ = p.communicate()
        df = self.rgis2df(output)

        time_cols = df.WeightLayerName.str.split("-", expand=True).astype(int)
        df.loc[:, "Year"] = time_cols[0]
        df.loc[:, "Month"] = time_cols[1]
        df.loc[:, "Day"] = 1
        df.loc[:, "Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df.set_index(["ID", "Date"], inplace=True)
        df.sort_index(inplace=True)
        return df

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
            expression (str): String expression ie 'x.gdbc + y.gdc' or 'x.gdbc == nodata ? y.gdbc : xgdbc'
            stdout (int): subprocess stdout encoding ex: sp.PIPE
            extent (Path): RGIS coverage to use as bounding box of output
            interpolate (str, optional): [surface|flat] interpolation method. Defaults to "surface".
            var_exprs (list[tuple[str]], optional): sub expressions to use as variables in main expression. Defaults to None.
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

    def grdCalculate(self, expression: str, output_gz: Path, **kwargs) -> Path:
        """_summary_

        Args:
            expression (str): String expression ie 'x.gdbc + y.gdc' or 'x.gdbc == nodata ? y.gdbc : xgdbc'
            output_gz (Path): Output gdbc.gz path
            extent (Path): RGIS coverage to use as bounding box of output
            interpolate (str, optional): [surface|flat] interpolation method. Defaults to "surface".
            var_exprs (list[tuple[str]], optional): sub expressions to use as variables in main expression. Defaults to None.
            title (str, optional): RGIS metadata title. Defaults to None.
            subject (str, optional): RIGS metadata subject. Defaults to None.
            domain (str, optional): RGIS metadata domain. Defaults to None.
            verbose (bool, optional): Prints output command. Defaults to False.

        Returns:
            Path: output_gz
        """
        if not output_gz.parent.exists():
            output_gz.parent.mkdir(parents=True)

        ps = self._grdCalculate(expression, sp.PIPE, **kwargs)
        with gzip.open(output_gz, "wb") as f:
            f.write(ps.stdout)
        return output_gz

    def netCells2Grid(self, network, fieldname):
        cmd_path = self.ghaas_bin.joinpath("netCells2Grid")
        cmd = f"{cmd_path} -f {fieldname} {network}".split()
        return sp.run(cmd)

    def rgis2mapper(self, network, sampler, mapper):
        cmd_path = self.ghaas_bin.joinpath("rgis2mapper")
        cmd = f"{cmd_path} --domain {network} {sampler} {mapper}".split()
        return sp.run(cmd)

    def dsAggregate(self, in_ds, out_ds, step, aggregate="avg"):
        aggs = ["aggregate", "sum"]
        steps = ["day", "month", "year"]
        assert aggregate.lower() in aggs, f"aggregate must be on of {aggs}"
        assert step.lower() in steps, f"aggregate must be on of {steps}"
        cmd_path = self.ghaas_bin.joinpath("dsAggregate")
        cmd = f"{cmd_path} --step {step} --aagregate {aggregate} {in_ds} {out_ds}"

        return sp.run(cmd.split())

    def dsSampling(self, ds, domain_file, mapper_file, output, title="rgispy_sampling"):
        cmd_path = self.ghaas_bin.joinpath("dsSampling")
        cmd = f"{cmd_path} --domainfile {domain_file} --mapper {mapper_file} --title {title} {ds}"
        ps = sp.run(cmd.split(), stdout=output)
        return ps

    def rgis2domain(self, network, output_domain_ds):
        cmd_path = self.ghaas_bin.joinpath("rgis2domain")
        cmd = f"{cmd_path} {network} {output_domain_ds}"
        return sp.run(cmd.split())
