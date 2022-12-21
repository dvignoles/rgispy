"""Perform long term averages sum of rgis products"""

import multiprocessing as mp
import tempfile
from collections import OrderedDict
from pathlib import Path

from .. import util
from ..core import RgisCalculate, RgisDataStream, RgisGrid


def _ltmean(
    cycle_num: int,
    rgis_results_dir: Path,
    output_path,
    ghaas_bin=None,
    scratch_dir=None,
):
    rcalc = RgisCalculate(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
    grids = sorted(rgis_results_dir.glob("*.gdbc*"))

    # tempfile
    _app_tmp = rcalc._temp_rgisfile("append_temp", suffix=".gdbc")

    app_grid = rcalc.grdAppendLayers(grids, output_grd=Path(_app_tmp.name))
    _app_tmp.flush()

    ltmean = app_grid.grdCycleMean(cycle_num, output_grd=output_path)

    # cleanup
    _app_tmp.close()

    return ltmean


def lt_annual(annual_rgis: Path, output_path: Path, ghaas_bin=None, scratch_dir=None):
    return _ltmean(
        1, annual_rgis, output_path, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir
    )


def lt_monthly(monthly_rgis, output_path, ghaas_bin=None, scratch_dir=None):
    return _ltmean(
        12, monthly_rgis, output_path, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir
    )


def _summer_avg(monthly_rgis_grid, year, output_grid, ghaas_bin=None, scratch_dir=None):
    grid = RgisGrid(monthly_rgis_grid)
    # intermediate grids
    _sumsub = grid._temp_rgisfile(f"summer_subset_{year}", suffix=".gdbc")

    sumsub = grid.grdExtractLayers(
        first=f"{year}-05", last=f"{year}-08", output_grd=Path(_sumsub.name)
    )
    _ = sumsub.grdCycleMean(1, output_grd=output_grid)

    _sumsub.close()
    return output_grid


def lt_summer(monthly_rgis, output_path, workers=4, ghaas_bin=None, scratch_dir=None):
    rcalc = RgisCalculate(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
    grids = sorted(monthly_rgis.glob("*.gdbc*"))
    min_year = 9999
    max_year = -9999
    # extract summer months and take mean for each year
    with tempfile.TemporaryDirectory(
        dir=rcalc.scratch_dir, prefix="lt_summer_avg"
    ) as _tdir:
        tdir = Path(_tdir)
        procs = []
        summer_grids = []
        with mp.get_context("spawn").Pool(workers) as pool:
            for g in grids:
                year = util.get_year(g.name)
                if year < min_year:
                    min_year = year
                if year > max_year:
                    max_year = year

                _summean = tdir.joinpath(f"summermean_{year}.gdbc")
                p = pool.apply_async(
                    _summer_avg,
                    args=(g, year, _summean),
                    kwds={"ghaas_bin": ghaas_bin, "scratch_dir": scratch_dir},
                )
                procs.append(p)
                # _summer_avg(g, year, _summean, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
                summer_grids.append(_summean)
            try:
                [p.wait() for p in procs]
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                pool.terminate()
        # join all year-summer means and take long term mean
        _summeanall = tdir.joinpath("summer_mean_all.gdbc")
        summer_all = rcalc.grdAppendLayers(summer_grids, output_grd=_summeanall)
        summer_all_mean = summer_all.grdCycleMean(1, output_grd=output_path)
        summer_all_mean = summer_all_mean.grdRenameLayers(
            [
                f"LTSummerMean_{min_year}_{max_year}",
            ],
            title=f"Long_Term_Summer_Average_{min_year}_{max_year}",
            output_grd=output_path,
        )
        return summer_all_mean


def _dlt_renames(leap_year=True):
    if leap_year:
        dummy = util._gen_date_cols("daily", 2000)
    else:
        dummy = util._gen_date_cols("daily", 1999)

    return [d.strftime("XXXX-%m-%d") for d in dummy]


def _extract_layer(grid, first, last, new_grid, ghaas_bin=None, scratch_dir=None):
    g = RgisGrid(grid, scratch_dir=scratch_dir, ghaas_bin=ghaas_bin)
    g.grdExtractLayers(first=first, last=last, output_grd=new_grid)
    return new_grid


def _mean_layers(grds, append, mean, ghaas_bin=None, scratch_dir=None):
    rcalc = RgisCalculate(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
    day_append = rcalc.grdAppendLayers(grds, output_grd=append)
    day_append.grdCycleMean(1, output_grd=mean)
    return mean


def _get_grids(
    wbm_files, template=None, convert_gds=True, ghaas_bin=None, scratch_dir=None
):
    temporary = False
    exts = util._unique_extenions_files(wbm_files)
    assert len(exts) == 1, "Can only support files with one common file extension"
    assert exts[0] in [".gdbc", ".gds", ".ds"]
    if exts[0] == ".gdbc":
        return temporary, wbm_files
    else:
        if convert_gds:
            temporary = True
            assert (
                template is not None
            ), "Must provide network template to convert datastream to gdbc on the fly"
            temp_gdbcs = []
            for child in wbm_files:
                ds = RgisDataStream(child, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
                name = child.name.split(".")[0]
                _gdbc = ds._temp_rgisfile(name=name, suffix=".gdbc")
                _ = ds.to_rgis(template, Path(_gdbc.name))
                temp_gdbcs.append(_gdbc)
                # TODO
                break
            return temporary, temp_gdbcs
        else:
            print("set convert_gds=True to convert datastreams to grids on the fly")


def lt_daily(
    daily_rgis_files: list[Path],
    output_path,
    workers=4,
    template=None,
    convert_gds=False,
    ghaas_bin=None,
    scratch_dir=None,
):
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    # lt average including feb 29
    output_split = output_path.name.split(".", 1)
    output_name = output_split[0]
    output_ext = "." + output_split[1]
    output_leap = output_path.parent.joinpath(output_name + "_leap" + output_ext)

    temporary, _grids = _get_grids(
        daily_rgis_files,
        template=template,
        convert_gds=convert_gds,
        ghaas_bin=ghaas_bin,
        scratch_dir=scratch_dir,
    )
    grids = _grids if not temporary else [Path(c.name) for c in _grids]

    rcalc = RgisCalculate(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

    with tempfile.TemporaryDirectory(dir=rcalc.scratch_dir, prefix="lt_daily") as _tdir:
        tdir = Path(_tdir)

        # includs feb 29
        daily_bin = OrderedDict()

        years = []
        any_leap = False
        # extract layers into daily bins [1...366]
        with mp.get_context("spawn").Pool(workers) as pool:
            procs = []
            for g in grids:
                year = util.get_year(g.name)
                assert year not in years, f"cannot have multiple of same year {year}"
                years.append(year)

                # track presence of leap year in bins
                if year % 4 == 0:
                    any_leap = True

                layers = util._gen_date_cols("daily", year)
                for layer_id in layers:
                    date_str = layer_id.strftime("%Y-%m-%d")
                    dlt_bin = layer_id.strftime("XXXX-%m-%d")
                    day = layer_id.day_of_year

                    day_dir = tdir.joinpath(dlt_bin)
                    if not day_dir.exists():
                        day_dir.mkdir(parents=True)

                    if dlt_bin in daily_bin.keys():
                        new_grd = day_dir.joinpath(f"{date_str}.gdbc")
                        p = pool.apply_async(
                            _extract_layer,
                            args=(g, date_str, date_str, new_grd),
                            kwds={"ghaas_bin": ghaas_bin, "scratch_dir": scratch_dir},
                        )
                        procs.append(p)
                        daily_bin[dlt_bin].append(new_grd)

                    else:
                        new_grd = day_dir.joinpath(f"{date_str}.gdbc")
                        p = pool.apply_async(
                            _extract_layer,
                            args=(g, date_str, date_str, new_grd),
                            kwds={"ghaas_bin": ghaas_bin, "scratch_dir": scratch_dir},
                        )
                        procs.append(p)
                        daily_bin[dlt_bin] = [
                            new_grd,
                        ]

            try:
                [p.wait() for p in procs]
            except KeyboardInterrupt:
                pool.terminate()

            # average bins into single layer lt daily means
            dlts = []
            dlts_leap = []
            procs = []
            for day, grds in daily_bin.items():
                _day_append = tdir.joinpath(day, f"{day}_append.gdbc")
                _day_mean = tdir.joinpath(day, f"{day}_mean.gdbc")

                p = pool.apply_async(
                    _mean_layers,
                    args=(grds, _day_append, _day_mean),
                    kwds={"ghaas_bin": ghaas_bin, "scratch_dir": scratch_dir},
                )
                procs.append(p)

                if day != "XXXX-02-29":
                    dlts.append(_day_mean)

                if any_leap:
                    dlts_leap.append(_day_mean)

            try:
                [p.wait() for p in procs]
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                pool.terminate()

        # combine daily mean layers into 1 year of dLT
        dlt = rcalc.grdAppendLayers(dlts, output_grd=output_path)
        renames = _dlt_renames(leap_year=False)
        dlt = dlt.grdRenameLayers(renames, output_grd=output_path)

        if any_leap:
            dlt_leap = rcalc.grdAppendLayers(dlts_leap, output_grd=output_leap)
            renames_leap = _dlt_renames(leap_year=True)
            dlt_leap = dlt_leap.grdRenameLayers(renames_leap, output_grd=output_leap)
            return dlt, dlt_leap

        # clean up on the fly grids if necessary
        if temporary:
            for child in _grids:
                child.close()

        return dlt
