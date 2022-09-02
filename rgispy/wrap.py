"""RGIS Rcommand wrapper functions"""

import gzip
import subprocess as sp
import tempfile
from io import StringIO
from pathlib import Path

import pandas as pd

GIGABYTE = 1000000000


class Rgis:
    def __init__(self, ghaas_bin=None):
        if ghaas_bin is None:
            self.ghaas_bin = Path("/usr/local/share/ghaas/bin")
        else:
            self.ghaas_bin = Path(ghaas_bin)

    def table2rgis(self, df: pd.DataFrame) -> bytes:
        """Convert DataFrame to RGIS table (gdbt)

        Args:
            df (pd.DataFrame): pandas DataFrame

        Returns:
            bytes: In memory gdbt file
        """
        buf = StringIO()
        df.to_csv(buf, sep="\t")

        cmd_path = self.ghaas_bin.joinpath("table2rgis")
        cmd = f"{cmd_path}".split()
        p = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE)
        output, err = p.communicate(buf.getvalue().encode())

        return output

    def tblAddXY(
        self, gdbp_bytes: bytes, x: str = "XCoord", y: str = "YCoord", table="DBItems"
    ) -> bytes:
        """Add Coordinates to RGIS point coverage

        Args:
            gdbp_bytes (bytes): In memory rgis table / point coverage
            x (str, optional): X Name. Defaults to "XCoord".
            y (str, optional): Y Name. Defaults to "YCoord".

        Returns:
            bytes: In memory RGIS file
        """
        cmd_path = self.ghaas_bin.joinpath("tblAddXY")
        cmd = f"{cmd_path} -x {x} -y {y} -a {table}".split()
        p = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE)
        output, _ = p.communicate(gdbp_bytes)
        return output

    def tblConv2Point(self, gdbt_bytes: bytes, xfield: str, yfield: str) -> bytes:
        """Convert RGIS table to Point Coverage

        Args:
            gdbt_bytes (bytes): in memory rgis table
            xfield (str): Name of x coordinate in table
            yfield (str): Name of y coordinate in table

        Returns:
            bytes: In memory RGIS point coverage
        """
        cmd_path = self.ghaas_bin.joinpath("tblConv2Point")
        cmd = f"{cmd_path} --xcoord {xfield} --ycoord {yfield}".split()

        p = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE)

        output, _ = p.communicate(
            gdbt_bytes,
        )
        return output

    def tblRedefField(self, gdbp_bytes: bytes, old: str, new: str) -> bytes:
        """Rename RGIS Table Field name

        Args:
            gdbp_bytes (bytes): In Memory RGIS table like
            old (str): old field name
            new (str): new field name

        Returns:
            bytes: in Memory RGIS table like
        """
        cmd_path = self.ghaas_bin.joinpath("tblRedefField")
        cmd = f"{cmd_path} -f {old} -r {new}".split()
        p = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE)
        output, _ = p.communicate(gdbp_bytes)
        return output

    def pntSTNChar(self, gdbp_bytes: bytes, network: Path, suffix: str = None) -> bytes:
        """Gather RGIS network attributes and add to point coverage

        Args:
            gdbp_bytes (bytes): RGIS Point Coverage
            network (Path): RGIS Network Coverage (gdbn)
            suffix (str, optional): Suffix to add to new columns. Defaults to None.

        Returns:
            bytes: In Memory RGIS Point Coverage
        """
        cmd_path = self.ghaas_bin.joinpath("pntSTNChar")
        cmd = f"{cmd_path} -n {network}".split()
        p = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE)
        output, _ = p.communicate(gdbp_bytes)

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
                output = self.tblRedefField(output, c, f"{c}{suffix}")

        return output

    def rgis2df(
        self, rgis_bytes: bytes, table="DBItems", file_obj=False
    ) -> pd.DataFrame:
        """Convert in Memroy RGIS table/points to pandas DataFrame using rgis2table

        Args:
            rgis_bytes (bytes): In Memory RGIS table like

        Returns:
            pd.DataFrame: pandas DataFrame
        """
        cmd_path = self.ghaas_bin.joinpath("rgis2table")
        cmd = f"{cmd_path} -a {table}".split()

        if file_obj:
            rgis_bytes.seek(0)
            with tempfile.SpooledTemporaryFile(mode="w+", max_size=GIGABYTE) as f:
                p = sp.run(cmd, stdin=rgis_bytes, stdout=f)
                f.seek(0)
                df = pd.read_csv(f, sep="\t")
                return df
        else:
            p = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE)
            output, _ = p.communicate(rgis_bytes)
            df = pd.read_csv(StringIO(output.decode()), sep="\t")
            return df

    def pntSTNCoord(
        self,
        gdbp_bytes: bytes,
        network: Path,
        field: str,
        tolerance: int,
        radius: int,
        cfield: str = "SubbasinArea",
    ) -> bytes:
        """Snap Coordinates of RGIS Point Coverage to different network.
        By default the comparison field is compared to SubbasinArea.

        Args:
            gdbp_bytes (bytes): In Memory RGIS Point Coverage
            network (Path): RGIS network coverage (gdbn)
            field (str): Field of point coverage to use for comparison.
            tolerance (int): Tolerance in percent for field comparison.
            radius (int): Radius in KM to search for cell match.
            cfield (str, optional): Field of Network to compare to field of Point Coverage. Defaults to "SubbasinArea".

        Returns:
            bytes: In Memory RGIS Point Coverage
        """
        cmd_path = self.ghaas_bin.joinpath("pntSTNCoord")
        cmd = f"{cmd_path} -M fixed --tolerance {tolerance} --radius {radius} --field {field} -c {cfield} --network {network}".split()
        p = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE)
        output, _ = p.communicate(gdbp_bytes)
        return output

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
