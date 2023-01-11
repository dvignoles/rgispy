import multiprocessing as mp
import tempfile
from pathlib import Path

import click
import pandas as pd

from ..core import Rgis, RgisDataStream


def _get_mapper(network: Path, sampler: dict, ghaas_bin=None, scratch_dir=None):
    """Get sampler mapper as named tempfile"""

    sampler_path = sampler["file"]
    sampler_type = sampler["type"]

    rgis = Rgis(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
    mapper_name = sampler_path.stem.split(".")[0]
    mapper = rgis._temp_rgisfile(name=f"{mapper_name}_mapper", suffix=".mapper")
    rgis.rgis2mapper(network, sampler_path, Path(mapper.name))
    mapper.flush()
    return {"type": sampler_type, "mapper": mapper}


def _prepare_data_file(data_file, network=None, ghaas_bin=None, scratch_dir=None):
    """Prepare datastream for sampling. If gdbc, convert to datastream tempfile"""
    ds_suf = [
        ".ds",
        ".gds",
        ".ds.gz",
        ".gds.gz",
    ]
    grd_suf = [".gdbc", ".gdbc.gz"]
    valid = ds_suf + grd_suf

    data_suf = "." + data_file.name.split(".", 1)[-1]

    assert data_suf in valid, f"{data_file.name} must have extension in {valid}"

    if data_suf in ds_suf:
        ds = RgisDataStream(data_file, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
        return ds, None
    else:
        assert network is not None, "must supply network to sample gdbc"
        rgis = Rgis(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

        name = data_file.stem.split(".")[0]
        temp_ds = rgis._temp_rgisfile(name=name, suffix=".ds")
        try:
            ds = rgis.rgis2ds(data_file, network, Path(temp_ds.name))
            temp_ds.flush()
            return ds, temp_ds
        except Exception:
            temp_ds.close()


def _do_sample_file(
    data, domain_file, mappers, network, var_name=None, ghaas_bin=None, scratch_dir=None
):
    """Run dsSampling with all mappers"""
    # RgisDataStream, NamedTemporaryFile
    ds, temp_ds = _prepare_data_file(
        data, network=network, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir
    )
    try:
        for name, info in mappers.items():
            mapper = info["file"]
            mtype = info["type"]

            table = ds.dsSampling(domain_file, mapper)

            if mtype == "point":
                df = table.df()[["SampleID", "Date", "Value"]]
                if var_name is not None:
                    df.rename(columns={"Value": var_name}, inplace=True)
            else:
                df = table.df()[
                    [
                        "SampleID",
                        "Date",
                        # "ZoneArea",
                        "Mean",
                        "Minimum",
                        "Maximum",
                        # "StdDev",
                    ]
                ]
                if var_name is not None:
                    df.rename(
                        columns={
                            "Mean": f"{var_name}ZonalMean",
                            "Minimum": f"{var_name}ZonalMin",
                            "Maximum": f"{var_name}ZonalMax",
                        },
                        inplace=True,
                    )

            df.set_index(["SampleID", "Date"], inplace=True)
            yield name, df
    finally:
        # ensure any temp files cleaned up
        if temp_ds is not None:
            temp_ds.close()


def _do_sample_to_tempfile(
    data, domain_file, mappers, network, var_name=None, ghaas_bin=None, scratch_dir=None
):
    for name, df in _do_sample_file(
        data,
        domain_file,
        mappers,
        network,
        var_name=var_name,
        ghaas_bin=ghaas_bin,
        scratch_dir=scratch_dir,
    ):
        tf = tempfile.NamedTemporaryFile(
            prefix=name + "_", suffix="_output_tmp.csv", dir=scratch_dir
        )
        df.to_csv(tf.name)
        tf.flush()

        # cleanup tempfile manually later
        yield name, Path(tf.name)


def _do_sample_file_mp(
    data,
    temp_name,
    domain_file,
    mappers,
    network,
    var_name,
    output_dir,
    ghaas_bin=None,
    scratch_dir=None,
    ts_aggregate=False,
    compress=False,
    accum_vars=["Runoff", "Precipitation"],
):
    """Wrap _do_sample_file generator for multiprocessing simplification"""
    output = {}
    if scratch_dir is None:
        scratch_dir = Path(tempfile._get_default_tempdir())

    for name, df in _do_sample_file(
        data,
        domain_file,
        mappers,
        network,
        var_name=var_name,
        ghaas_bin=ghaas_bin,
        scratch_dir=scratch_dir,
    ):
        mapper_name_short = name.split("_")[1]
        csv_name = f"{mapper_name_short}_{temp_name}.csv"
        if compress:
            csv_name += ".gz"

        csv_path = output_dir.joinpath(csv_name)
        df.to_csv(csv_path)

        output[name] = {var_name: {"dTS": csv_path}}
        if ts_aggregate:
            monthly, annual = _agg_ts(df, accum_vars=accum_vars)
            mts = output_dir.joinpath(csv_name.replace("dTS", "mTS"))
            ats = output_dir.joinpath(csv_name.replace("dTS", "aTS"))
            monthly.to_csv(mts)
            annual.to_csv(ats)
            output[name][var_name]["mTS"] = mts
            output[name][var_name]["aTS"] = ats

    return output


def _process_mp_output(var_dfs, completed_procs):
    """Process completed processes into var_dfs dictionary template"""
    for p in completed_procs:
        output = p.get()
        for mapper, var_dict in output.items():
            for var_name, var_tf in var_dict.items():
                for ts, out in var_tf.items():
                    var_dfs[mapper][var_name][ts].append(out)
    return var_dfs


def _mp_engine(
    workers,
    data,
    domain,
    mappers,
    network,
    samplers,
    output_dir,
    ghaas_bin=None,
    scratch_dir=None,
    ts_aggregate=False,
    compress=False,
    accum_vars=["Runoff", "Precipitation"],
):
    """Run sampling either as standalone process or in pool configuration"""
    zone_dfs = {
        mapper_name: {v: {"dTS": [], "mTS": [], "aTS": []} for v in data["zone"].keys()}
        for mapper_name in samplers.keys()
        if samplers[mapper_name]["type"] == "zone"
    }
    point_dfs = {
        mapper_name: {
            v: {"dTS": [], "mTS": [], "aTS": []} for v in data["point"].keys()
        }
        for mapper_name in samplers.keys()
        if samplers[mapper_name]["type"] == "point"
    }
    var_dfs = {**zone_dfs, **point_dfs}
    zone_mappers = {k: v for k, v in mappers.items() if v["type"] == "zone"}
    point_mappers = {k: v for k, v in mappers.items() if v["type"] == "point"}

    # TODO: By dividing into zone/point separate calls to _do_sample_file_mp, gdbc files can be converted to datastream
    # multiple times (once for point samplers, once for zone samplers)
    # Should re-factor to ensure this does not happen

    procs = []
    with mp.get_context("spawn").Pool(workers) as pool:
        try:
            if len(zone_mappers) > 0:
                for var_key, var_ds_group in data["zone"].items():
                    for ds in var_ds_group:
                        tf = f"{ds.name.split('.')[0]}"
                        procs.append(
                            pool.apply_async(
                                _do_sample_file_mp,
                                args=(
                                    ds,
                                    tf,
                                    domain,
                                    zone_mappers,
                                    network,
                                    var_key,
                                    output_dir,
                                ),
                                kwds=dict(
                                    ghaas_bin=ghaas_bin,
                                    scratch_dir=scratch_dir,
                                    ts_aggregate=ts_aggregate,
                                    compress=compress,
                                    accum_vars=accum_vars,
                                ),
                            )
                        )
            if len(point_mappers) > 0:
                for var_key, var_ds_group in data["point"].items():
                    for ds in var_ds_group:
                        tf = f"{ds.name.split('.')[0]}"
                        procs.append(
                            pool.apply_async(
                                _do_sample_file_mp,
                                args=(
                                    ds,
                                    tf,
                                    domain,
                                    point_mappers,
                                    network,
                                    var_key,
                                    output_dir,
                                ),
                                kwds=dict(
                                    ghaas_bin=ghaas_bin,
                                    scratch_dir=scratch_dir,
                                    ts_aggregate=ts_aggregate,
                                    compress=compress,
                                    accum_vars=accum_vars,
                                ),
                            )
                        )
            pool.close()
            pool.join()

            var_dfs = _process_mp_output(var_dfs, procs)
            return var_dfs
        except KeyboardInterrupt:
            pool.terminate()


def _collect_sample(
    data: dict[str, list[Path]],  # {var_name: [ds1980, ds1981, ...]}
    network: Path,
    samplers: dict[str, Path],  # {mapper_name: {type: <point or zone>, file: <path>}}
    output_dir: Path,
    workers=1,
    ghaas_bin=None,
    scratch_dir=None,
    ts_aggregate=False,
    compress=False,
    accum_vars=["Runoff", "Precipitation"],
):
    """Setup domain & mappers and call sampling driver"""
    rgis = Rgis(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

    domain = rgis._temp_rgisfile(name="domain", suffix=".ds")
    mappers = {
        mapper_name: _get_mapper(
            network, info, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir
        )
        for mapper_name, info in samplers.items()
    }
    mappers_fname = {
        k: {"type": mapper["type"], "file": Path(mapper["mapper"].name)}
        for k, mapper in mappers.items()
    }

    try:
        rgis.rgis2domain(network, Path(domain.name))
        domain.flush()
        var_dfs = _mp_engine(
            workers,
            data,
            domain.name,
            mappers_fname,
            network,
            samplers,
            output_dir,
            ghaas_bin=ghaas_bin,
            scratch_dir=scratch_dir,
            ts_aggregate=ts_aggregate,
            compress=compress,
            accum_vars=accum_vars,
        )
    finally:
        domain.close()
        for m in mappers.values():
            m["mapper"].close()
        # ensures everything cleaned up in any forked processes

    return var_dfs


def _validate_inputs(data):
    assert "zone" in data.keys() or "point" in data.keys()

    if "zone" in data.keys():
        zone_var_lens = [len(d) for d in data["zone"].values()]
        assert len(set(zone_var_lens)) in [
            0,
            1,
        ], "Must supply same number of files for all variables"
    if "point" in data.keys():
        point_var_lens = [len(d) for d in data["point"].values()]
        assert len(set(point_var_lens)) in [
            0,
            1,
        ], "Must supply same number of files for all variables"


def _guess_ts(date_series):
    if date_series.str.match(r"^\d{4}-\d{2}-\d{2}$").all():
        return "d"
    if date_series.str.match(r"^\d{4}-\d{2}$").all():
        return "m"
    if date_series.str.match(r"^\d{4}$").all():
        return "a"
    raise ValueError


def _dt_format(ts):
    """Get date format string for daily/monthly/annual"""
    assert ts in ["d", "m", "a"]

    if ts == "d":
        return "%Y-%m-%d"
    if ts == "m":
        return "%Y-%m"
    if ts == "a":
        return "%Y"


def _cleanup_agg(df):
    """Fix indexes and remove garbage columns"""
    # handle case where idx somehow sneak into index and columns
    if "SampleID" in df.index.names and "SampleID" in df.columns:
        df = df[[c for c in df.columns if c != "SampleID"]]

    if "Date" in df.index.names and "Date" in df.columns:
        df = df[[c for c in df.columns if c != "Date"]]

    df = (
        df.reset_index()
        .sort_values(["Date", "SampleID"])
        .set_index(["SampleID", "Date"])
    )

    df = df[
        [
            c
            for c in df.columns
            if c
            not in ["SampleID", "index", "level_0", "Year", "Month", "Day", "pddate"]
        ]
    ]

    return df


def _agg_accumulate(df, vars, ts, agg="mean"):
    for v in vars:
        assert v in df.columns

    assert agg in ["sum", "mean", "min", "max"], "must be sum / mean / min / max"

    df = df.reset_index()
    df.loc[:, "pddate"] = pd.to_datetime(df.Date, format=_dt_format(ts))
    # subset and ensure no dup columns
    cols = list(set(["SampleID", "Date", "pddate"] + vars))
    avg_vars_df = df[cols]

    if ts == "d":

        monthly_g = avg_vars_df.groupby(
            [
                avg_vars_df.SampleID,
                avg_vars_df.pddate.dt.year,
                avg_vars_df.pddate.dt.month,
            ]
        )
        # calls mean/sum/min/max
        monthly = getattr(monthly_g, agg)()
        monthly.index.rename(["SampleID", "Year", "Month"], inplace=True)
        monthly.reset_index(inplace=True)
        monthly.loc[:, "Date"] = (
            monthly.Year.astype(str) + "-" + monthly.Month.astype(str).str.zfill(2)
        )

        monthly = _cleanup_agg(monthly)
        return monthly

    if ts == "m":
        ann_cols = [c for c in avg_vars_df.columns if c not in ["Month", "Date"]]
        annual_g = avg_vars_df[ann_cols].groupby(
            [
                avg_vars_df.SampleID,
                avg_vars_df.pddate.dt.year,
            ]
        )
        # calls mean/sum/min/max
        annual = getattr(annual_g, agg)()
        # index after groupby is SampleID, Year
        annual.index.rename(["SampleID", "Date"], inplace=True)
        annual = _cleanup_agg(annual)
        return annual


def _split_agg_accum(df, sum_vars, ts):
    """Do combination of avg/sum/min/max montly/annual aggregations"""
    has_avg = False
    has_sum = False
    has_min = False
    has_max = False

    min_vars = [c for c in df.columns if "ZonalMin" in c]
    max_vars = [c for c in df.columns if "ZonalMax" in c]

    # daily means need to be summed for Precipitation/Runoff etc.
    more_sum_vars = []
    for c in df.columns:
        if "ZonalMean" in c:
            if any([s in c for s in sum_vars]):
                if c not in sum_vars:
                    more_sum_vars.append(c)
    sum_vars = list(sum_vars) + more_sum_vars

    # all non summation aggregations are average aggregations
    avg_vars = [
        c
        for c in df.columns
        if c not in ["SampleID", "Date"] + sum_vars + min_vars + max_vars
    ]

    # not all sum vars may necessarily be in df
    true_sum_vars = [c for c in sum_vars if c in df.columns]

    if len(avg_vars) > 0:
        has_avg = True

    if len(true_sum_vars) > 0:
        has_sum = True

    if len(min_vars) > 0:
        has_min = True

    if len(max_vars) > 0:
        has_max = True

    assert has_avg or has_sum or has_max or has_min

    if has_avg:
        avg_agg = _agg_accumulate(df, avg_vars, ts, agg="mean")
    else:
        avg_agg = None

    if has_sum:
        sum_agg = _agg_accumulate(df, true_sum_vars, ts, agg="sum")
    else:
        sum_agg = None

    if has_min:
        min_agg = _agg_accumulate(df, min_vars, ts, agg="min")
    else:
        min_agg = None

    if has_max:
        max_agg = _agg_accumulate(df, max_vars, ts, agg="max")
    else:
        max_agg = None

    final = None
    for agg_df in [avg_agg, sum_agg, min_agg, max_agg]:
        if agg_df is not None:
            if final is None:
                final = agg_df
            else:
                final = final.merge(agg_df, left_index=True, right_index=True)

    return final


def _agg_ts(sample_df, accum_vars=["Runoff", "Precipitation"]):
    sample_df = sample_df.reset_index()
    current = _guess_ts(sample_df.Date)

    if current == "d":

        # daily -> monthly
        monthly = _split_agg_accum(sample_df, accum_vars, current)

        # monthly -> annual
        annual = _split_agg_accum(monthly, accum_vars, "m")
        return monthly, annual
    elif current == "m":
        raise ValueError("Unimplemented")
    elif current == "a":
        raise ValueError("Unimplemented")


def _collect_samplers(samplers_list):
    """Organize samplers into dict specifying if point or zone"""
    out = {}
    for s in samplers_list:
        # xor
        assert (".gdbp" in s.suffixes) ^ (".gdbd" in s.suffixes)
        stype = "point" if ".gdbp" in s.suffixes else "zone"
        out[s.stem.split(".")[0]] = {"type": stype, "file": s}
    return out


def sample(
    data: dict[str, list[Path]],  # {var_name: [ds1980, ds1981, ...]}
    network: Path,
    samplers: list[Path],
    output_dir: Path,
    workers=1,
    ghaas_bin=None,
    scratch_dir=None,
    ts_aggregate=False,
    compress=False,
    accum_vars=["Runoff", "Precipitation"],
):
    _validate_inputs(data)
    samplers_dict = _collect_samplers(samplers)

    var_csvs = _collect_sample(
        data,
        network,
        samplers_dict,
        output_dir,
        workers=workers,
        ghaas_bin=ghaas_bin,
        scratch_dir=scratch_dir,
        ts_aggregate=ts_aggregate,
        compress=compress,
        accum_vars=accum_vars,
    )
    return var_csvs


def _group_ds_byvar(all_gds, gdbc=False):
    all_vars = set()

    all_domain = set()
    all_exp = set()
    all_res = set()
    gds_groups = {}

    for gds in all_gds:
        split = gds.name.split("_")

        if gdbc:
            all_domain.add(split[0])
            all_exp.add(split[2])
            all_res.add(split[3])
            variable = split[1]
        else:
            all_domain.add(split[0])
            all_exp.add(split[3])
            all_res.add(split[4])
            variable = split[2]

        if variable in all_vars:
            gds_groups[variable].append(gds)
        else:
            gds_groups[variable] = [
                gds,
            ]
            all_vars.add(variable)

    # ensure groups sorted
    for k in gds_groups.keys():
        gds_groups[k] = sorted(gds_groups[k])

    assert len(all_exp) == 1, f"Multiple experiments detected, {all_exp}"
    assert len(all_res) == 1, f"Multiple resolutions detected, {all_res}"
    assert len(all_domain) == 1, f"Multiple domains detected, {all_domain}"

    return gds_groups, all_exp.pop(), all_res.pop(), all_domain.pop()


def _filter_vars(gds_groups, variables):
    """Filter out datastream groups to include only outputs in variables"""
    for v in variables:
        assert v in gds_groups.keys(), f"Variable {v} not found in {gds_groups.keys()}"
    all_v = list(gds_groups.keys())
    for v in all_v:
        if v not in variables:
            gds_groups.pop(v)
    return gds_groups


def _dry_summary(
    zone_groups,
    point_groups,
    domain,
    exp,
    res,
    network,
    samplers,
    workers,
    ts_aggregate,
    accum_vars,
):
    true_accum_vars = list(
        set(
            [v for v in accum_vars if v in zone_groups.keys()]
            + [v for v in accum_vars if v in point_groups.keys()]
        )
    )

    sum_agg_vars = ", ".join(true_accum_vars) if len(true_accum_vars) > 0 else ""

    summary = f"""
        Workers: {workers}
        Aggregate Monthly/Annual: {ts_aggregate}

        Domain: {domain}
        Resolution: {res}
        Experiment: {exp}
        Network: {network.name}

        Point Variables: {", ".join([v for v in point_groups.keys()])}
        Zone Variables: {", ".join([v for v in zone_groups.keys()])}
        Sum Aggregation Variables: {sum_agg_vars}
        Samplers: {", ".join([s.name for s in samplers])}
    """
    print(summary)


def _rename(mapper, csv_name, gz=True):
    name = f"{mapper.replace('_Static', '')}" + "_"
    name += csv_name.split("_")[-4] + "_"
    name += csv_name.split("_")[-1]
    if gz:
        name += ".gz"
    return name


def _prep_inputs(
    dsdir,
    itype="ds",
    outputs_only=True,
    variables=[],
    filters=[],
):
    assert itype in ["ds", "gdbc"], "must supply directory of datastream (ds) or gdbc"

    if isinstance(dsdir, Path) and dsdir.is_dir():
        if itype == "ds":
            all_dts = dsdir.glob("*dTS*.gds*")
        else:
            all_dts = dsdir.rglob("*dTS*.gdbc*")
    else:  # list of files
        assert all([p.is_file() for p in dsdir]), "Must be list of path objects"
        all_dts = sorted(dsdir)

    if len(filters) > 0:
        for fil in filters:
            all_dts = filter(lambda x: fil.lower() in x.name.lower(), all_dts)

    if outputs_only:
        all_dts = filter(lambda x: "output" in x.name.lower(), all_dts)

    all_dts = sorted(all_dts)
    is_gdbc = True if itype == "gdbc" else False
    gds_groups, exp, res, domain = _group_ds_byvar(all_dts, gdbc=is_gdbc)

    # filter vars if specified
    if len(variables) > 0:
        gds_groups = _filter_vars(gds_groups, variables)

    return gds_groups, exp, res, domain


def _sample_dir(
    directory,
    directory_type,
    network,
    samplers,
    output_dir,
    workers=1,
    ts_aggregate=True,
    outputs_only=True,
    ghaas_bin=None,
    scratch_dir=None,
    variables=[],
    accum_vars=["Runoff", "Precipitation"],
    compress=False,
    filters=[],
):
    gds_groups, exp, res, domain = _prep_inputs(
        directory,
        itype=directory_type,
        outputs_only=outputs_only,
        variables=variables,
        filters=filters,
    )

    # set up final output dir
    output_subdir = output_dir.joinpath(f"{domain}_{exp}_{res}")
    if not output_subdir.exists():
        output_subdir.mkdir(parents=True)

    # for testing
    # gds_groups = {k: v[0:2] for k, v in gds_groups.items()}

    # includes Flux variables (not sure if correct?)
    zone_groups = {
        k: v
        for k, v in gds_groups.items()
        if (not k.startswith("River"))
        and (not k.startswith("Accum"))
        and (not k.startswith("Reservoir"))
    }

    # will still sample reservoir columns at non reservoir samplers
    point_groups = {
        k: v
        for k, v in gds_groups.items()
        if (k.startswith("River"))
        or (k.startswith("Accum"))
        or ("Flux" in k)
        or (k.startswith("Reservoir"))
    }

    # We still want to look at variables like AirTemperature at single Cells
    for k, v in zone_groups.items():
        if k not in point_groups:
            point_groups[k] = v

    groups = {"zone": zone_groups, "point": point_groups}

    _dry_summary(
        zone_groups,
        point_groups,
        domain,
        exp,
        res,
        network,
        samplers,
        workers,
        ts_aggregate,
        accum_vars,
    )

    rgis = Rgis(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
    temp_dir = tempfile.TemporaryDirectory(prefix="sample_", dir=rgis.scratch_dir)

    var_csvs = sample(
        groups,
        network,
        samplers,
        output_subdir,
        workers=workers,
        ghaas_bin=ghaas_bin,
        scratch_dir=Path(temp_dir.name),
        accum_vars=accum_vars,
        ts_aggregate=ts_aggregate,
        compress=compress,
    )
    temp_dir.cleanup()
    return var_csvs


def sample_wbm_dsdir(
    dsdir,
    network,
    samplers,
    output_dir,
    workers=1,
    ts_aggregate=True,
    outputs_only=True,
    ghaas_bin=None,
    scratch_dir=None,
    variables=[],
    accum_vars=["Runoff", "Precipitation"],
    compress=False,
    filters=[],
):
    return _sample_dir(
        dsdir,
        "ds",
        network,
        samplers,
        output_dir,
        workers=workers,
        ts_aggregate=ts_aggregate,
        outputs_only=outputs_only,
        ghaas_bin=ghaas_bin,
        scratch_dir=scratch_dir,
        variables=variables,
        accum_vars=accum_vars,
        compress=compress,
        filters=filters,
    )


def sample_wbm_gdbcdir(
    dsdir,
    network,
    samplers,
    output_dir,
    workers=1,
    ts_aggregate=True,
    outputs_only=False,
    ghaas_bin=None,
    scratch_dir=None,
    variables=[],
    accum_vars=["Runoff", "Precipitation"],
    compress=False,
    filters=[],
):
    return _sample_dir(
        dsdir,
        "gdbc",
        network,
        samplers,
        output_dir,
        workers=workers,
        ts_aggregate=ts_aggregate,
        outputs_only=outputs_only,
        ghaas_bin=ghaas_bin,
        scratch_dir=scratch_dir,
        variables=variables,
        accum_vars=accum_vars,
        compress=compress,
        filters=filters,
    )


@click.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "-d",
    "--outputdirectory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path.cwd(),
)
@click.option(
    "-v",
    "--variable",
    type=click.STRING,
    multiple=True,
    default=[],
    help="If specified, filter to these variables",
)
@click.option(
    "-s",
    "--sampler",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    multiple=True,
)
@click.option(
    "-f",
    "--filter",
    type=click.STRING,
    multiple=True,
    default=[],
    help="File name must contain filter str (case insenstive)",
)
@click.option(
    "-n",
    "--network",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=Path
    ),  # TODO: infer option for res
)
@click.option("-w", "--workers", type=int, default=1)
@click.option(
    "-a",
    "--accum-var",
    type=click.STRING,
    multiple=True,
    default=["Runoff", "Precipitation"],
)
@click.option(
    "-t",
    "--aggregatetime",
    is_flag=True,
    default=False,
    show_default=True,
    help="Create monthly and annual results from daily",
)
@click.option(
    "-z",
    "--gzipped",
    is_flag=True,
    default=False,
    show_default=True,
    help="compress csvs with gzip",
)
@click.option(
    "-g",
    "--gdbc",
    is_flag=True,
    default=False,
    show_default=True,
    help="Directory contains gdbc, not datastreams",
)
def wbm_sample(
    directory,
    outputdirectory,
    variable,
    sampler,
    filter,
    network,
    workers,
    accum_var,
    aggregatetime,
    gzipped,
    gdbc,
):
    if not gdbc:
        sample_wbm_dsdir(
            directory,
            network,
            sampler,
            outputdirectory,
            workers=workers,
            variables=variable,
            accum_vars=accum_var,
            ts_aggregate=aggregatetime,
            compress=gzipped,
            filters=filter,
        )
    else:
        sample_wbm_gdbcdir(
            directory,
            network,
            sampler,
            outputdirectory,
            workers=workers,
            variables=variable,
            accum_vars=accum_var,
            ts_aggregate=aggregatetime,
            compress=gzipped,
            filters=filter,
        )
