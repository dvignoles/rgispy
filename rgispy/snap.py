"""Snapping utilities"""

import typing
from functools import partial
from numbers import Number
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from geopy.distance import geodesic

from .wrap import Rgis

xarray_ds = xr.core.dataset.Dataset
xarray_da = xr.core.dataarray.DataArray
coordinate = tuple[float, float]


def get_cell(
    coord: tuple[float, float], cellid_da: xarray_da
) -> tuple[typing.Any, tuple[float, float]]:
    """Get cell value and coordinate for given point

    Args:
        coord (tuple[float, float]): (lon, lat) pair
        cellid_da (xr.DataArray): DataArray of cellid

    Returns:
        tuple[int, tuple[float, float]] -> cell value, (cell lon, cell lat)
    """
    match = cellid_da.sel(lon=coord[0], lat=coord[1], method="nearest")

    # Extract lon value and index
    lon = match.lon.data.tolist()

    # Extract lat value and index
    lat = match.lat.data.tolist()

    match_val = int(match.data.tolist())
    return match_val, (lon, lat)


def add_cellid(
    gdf: gpd.GeoDataFrame, cellid_da: xarray_da, suffix=None
) -> gpd.GeoDataFrame:
    """Add cellid information to a Geodataframe of points

    Args:
        gdf (gpd.GeoDataFrame): geodataframe with points geometry column
        cellid_da (xarray_da): xr.DataArray of cellid values
        suffix (_type_, optional): Suffix for new columns added. Defaults to None.

    Returns:
        gpd.GeoDataFrame: Input gdf with additional cellid columns
    """

    def _do_mapping(row):
        coord = (row.geometry.x, row.geometry.y)
        cellid, cellid_coord = get_cell(coord, cellid_da)

        row.loc[f"cellid{suffix}"] = cellid
        row.loc[f"xcoord{suffix}"] = cellid_coord[0]
        row.loc[f"ycoord{suffix}"] = cellid_coord[1]

        return row

    result = gdf.apply(_do_mapping, axis=1, result_type="expand")
    return result


def naive_guess(lon, lat, dr):
    match = dr.sel(lon=lon, lat=lat, method="nearest")

    # Extract lon value and index
    lon = match.lon

    # Extract lat value and index
    lat = match.lat

    match_val = match.data.tolist()
    return match_val, (lat, lon)


def assert_coords(lons, lats, dr):
    # confirm required dimensions
    assert "lat" in dr.dims, "must have lat dimension"
    assert "lon" in dr.dims, "must have lon dimension"

    if lons is None:
        lons = list(dr.lon.values)
    assert len(lons) == dr.shape[1]

    if lats is None:
        lats = list(dr.lat.values)
    assert len(lats) == dr.shape[0]

    return lons, lats


def get_buffer_indices(
    coord: coordinate,
    value_da: xarray_da,
    radius: int = 1,
) -> typing.Generator[tuple[int, int], None, None]:
    """Get values in radius of coord given from value_da

    Args:
        coord (tuple[float, float]): coord at center of cell radius
        value_da (xarray_da): dataarray to extract values from in radius at coord
        radius (int, optional): cell radisu. Defaults to 1.

    Returns:
        xarray_da: subset of value_da in radius of coord
    """
    lons, lats = assert_coords(None, None, value_da)

    # maximum indexes
    loni_max = value_da.shape[1]
    lati_max = value_da.shape[0]

    naive_val, naive_latlon = naive_guess(coord[0], coord[1], value_da)

    lati_naive = lats.index(naive_latlon[0])
    loni_naive = lons.index(naive_latlon[1])

    # ensure indexes within bounds
    def _check_bounds(ind, bound_ind, greater_than=True):
        if greater_than:
            if ind >= bound_ind:
                return ind
            else:
                return bound_ind
        else:
            if ind <= bound_ind:
                return ind
            else:
                return bound_ind

    lati_start = _check_bounds(lati_naive - radius, 0)
    lati_end = _check_bounds(lati_naive + radius + 1, lati_max, greater_than=False)
    loni_start = _check_bounds(loni_naive - radius, 0)
    loni_end = _check_bounds(loni_naive + radius + 1, loni_max, greater_than=False)

    lons_ind = range(loni_start, loni_end)
    lats_ind = range(lati_start, lati_end)

    for lati in lats_ind:
        for loni in lons_ind:
            yield lati, loni


def get_buffer_values(
    coord: coordinate,
    value_da: xarray_da,
    radius: int = 1,
) -> xarray_da:
    """Get values in radius of coord given from value_da

    Args:
        coord (tuple[float, float]): coord at center of cell radius
        value_da (xarray_da): dataarray to extract values from in radius at coord
        radius (int, optional): cell radisu. Defaults to 1.

    Returns:
        xarray_da: subset of value_da in radius of coord
    """

    indices = list(get_buffer_indices(coord, value_da, radius))
    lats_ind = list(map(lambda x: x[0], indices))
    lons_ind = list(map(lambda x: x[1], indices))

    in_radius = value_da.isel(lon=lons_ind, lat=lats_ind)
    return in_radius


def symmetric_dif(x, y, absolute=False):
    if (x + y) == 0:
        return 1
    else:
        dif = (x - y) / (x + y)
        if absolute:
            return abs(dif)
        return dif


abs_symmetric_dif = partial(symmetric_dif, absolute=True)


def cartesian_distance(x1, y1, x2, y2):
    d = (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** (1 / 2)
    return d


snap_pair = tuple[str, float, xarray_da]


def ensure_iter(x):
    if not hasattr(x, "__iter__"):
        return [x]

    else:
        return x


def ensure_list(x) -> list:
    if type(x) is not list:
        return [
            x,
        ]
    else:
        return x


def clean_df(df):
    real_cols = [c for c in df.columns if "unnamed:" not in c.lower()]
    remove_cols = [c + ".1" for c in real_cols]
    keep_cols = [c for c in real_cols if c not in remove_cols]

    return df[keep_cols]


def pre_snap_stats(
    coord: coordinate,
    target: snap_pair,
    supplements,
    radius: int = 1,
    tolerances: list[int] = [5, 10, 15, 20, 25],
) -> dict:
    """Generate statistics for possible snap candidates in radius of coord

    Args:
        coord (tuple[float, float]): coord at center of cell radisu
        snap_from_val (float): value we are snapping against
        target_da (xarray_da): target grid for snapping
        radius (int, optional): cell radius. Defaults to 1.
        tolerances (list[int], optional): Absolute symmetric difference tolerance
                                          thresholds. Defaults to [5, 10, 15, 20, 25].

    Returns:
        dict: {'best_dif': ... , 'best_val': ..., }
    """
    target_id, target_from_val, target_da = target
    lons, lats = assert_coords(None, None, target_da)
    indices = list(get_buffer_indices(coord, target_da, radius=radius))
    lati_in_radius = list(map(lambda x: x[0], indices))
    loni_in_radius = list(map(lambda x: x[1], indices))

    in_radius = target_da.isel(lon=loni_in_radius, lat=lati_in_radius).data.flatten()

    sym_difs = [
        abs_symmetric_dif(target_from_val, snap_to_val) for snap_to_val in in_radius
    ]
    sym_difs_abs = [abs(x) for x in sym_difs]

    best_dif = min(sym_difs_abs)
    best_ind = sym_difs_abs.index(best_dif)
    best_val = in_radius[best_ind]

    stats = {
        f"{target_id}_source_val": target_from_val,
        f"{target_id}_snap_dif": best_dif,
        f"{target_id}_snap_val": best_val,
    }

    # target snap evaluation
    tolerances = ensure_iter(tolerances)

    for t in tolerances:
        in_t = True if (best_dif <= t / 100) else False
        stats[f"{target_id}_in_{t}%"] = in_t

    # suplemmental lookups
    supplements = ensure_list(supplements)
    for sup_id, sup_from_val, sup_da in supplements:
        sup_in_radius = sup_da.isel(
            lon=loni_in_radius, lat=lati_in_radius
        ).data.flatten()
        sup_best_val = sup_in_radius[best_ind]

        stats[f"{sup_id}_from_val"] = sup_from_val
        stats[f"{sup_id}_snap_val"] = sup_best_val

    return stats


def _default_choice(sym_dif, best):
    if sym_dif < best["best_dif"]:
        return True
    return False


def _prioritize_distance_choice(coordi_naive, coordi_cand, sym_dif, best):
    """choose closest cell within radius that meets tolerance criteria"""

    # case already handled above
    dist = cartesian_distance(*coordi_naive, *coordi_cand)

    # initilization case TODO: VERIFY THIS MAKE SENSE
    if not best["best_in_tolerance"]:
        return True, dist
    else:
        if (dist <= best["best_distance"]) and (sym_dif < best["best_dif"]):
            return True, dist

    return False, dist


def comparison_guess(
    coord: coordinate,
    cell_radius: int,
    comparison_da: list[xarray_da],
    comparison_value: list[Number],
    tolerance: float = 0.05,
    adjust_outside_tolerance=False,
    weights=None,
    lons=None,
    lats=None,
) -> tuple[bool, tuple[int], float, tuple[typing.Any]]:
    """Move location to best match betweeen comparison_da and comparison_value within
       radius and tolerance.  If Multiple comparison factors are inputted, apply weight
       to each factor.

    Args:
        lon (float): Longitude
        lat (_type_): Latitude
        cell_radius (int): Maximum cell radius within comparison_da to move coordinate
        comparison_da (list[xr_da]): One or more grids of comparison factors
        comparison_value (list[Number]): Source attribute value to compare is
                                         comparison_da
        tolerance (float, optional): Maximum tolerance for comparison difference
                                    (0.05 -> 5%). Defaults to None.
        adjust_outside_tolerance= (bool, optional): If no matches within tolerance,
                                             move to best available cell in radius.
        weights (list[Number], optional): Weights of comparison factors if using
                                          :nmultiple. Defaults to None.
        lons (_type_, optional): All possible snapped longitudes. Defaults to None.
        lats (_type_, optional): All possible snapped latitudes. Defaults to None.

    Returns:
        tuple:  (is_naive, snapped_indexes, symmetric_difference, snapped_values)
    """

    comparison_da = ensure_list(comparison_da)
    comparison_value = ensure_list(comparison_value)

    # handle single factor vs multi
    assert len(comparison_da) == len(comparison_value)
    if len(comparison_da) > 1:
        assert (
            weights is not None
        ), "Must supply weights list if using more than 1 comparison factor"
        assert len(comparison_da) == len(
            weights
        ), "Must supply same number of weights as comparison factors"
    else:
        weights = [
            1,
        ]

    # extract coords
    lons, lats = assert_coords(lons, lats, comparison_da[0])

    # get native cell values
    naive = {}
    naive_dif_sum = 0
    naive_values = []
    for i, comp in enumerate(zip(comparison_da, comparison_value, weights)):
        naive_val, naive_latlon = naive_guess(*coord, comp[0])
        naive_values.append(naive_val)
        naive_dif = abs_symmetric_dif(naive_val, comp[1])

        naive_dif_sum += naive_dif * comp[2]
        naive[i] = {
            "naive_val": naive_val,
            "naive_dif": naive_dif,
            "naive_latlon": naive_latlon,
        }

    naive_dif = naive_dif_sum / sum(weights)

    lati_naive = lats.index(naive[0]["naive_latlon"][0])
    loni_naive = lons.index(naive[0]["naive_latlon"][1])
    naive_coord_index = (lati_naive, loni_naive)

    # save current best option, initialized w/ naive case
    best = dict(
        best_coord_index=naive_coord_index,
        best_values=naive_values,
        best_dif=naive_dif,
        best_distance=0,
        best_is_naive=True,
        best_in_tolerance=True if naive_dif <= tolerance else False,
    )

    # if naive cell is in tolerance, that's the answer
    if best["best_in_tolerance"]:
        return (
            best["best_is_naive"],
            best["best_coord_index"],
            best["best_dif"],
            best["best_values"],
        )

    # Check all cells in buffer for best match
    for lati_cand, loni_cand in get_buffer_indices(
        coord, comparison_da[0], radius=cell_radius
    ):
        coord_index = (lati_cand, loni_cand)
        lati_cand, loni_cand = coord_index
        # naive case already handled by intialization
        is_naive_cand = True if coord_index == naive_coord_index else False
        if is_naive_cand:
            continue

        # extract values and assess symmetric dif for candidate
        cand_values = []
        sym_dif_sum = 0
        for da, comp_val, w in zip(comparison_da, comparison_value, weights):
            cand_val = da[coord_index].data.tolist()
            cand_values.append(cand_val)
            sym_dif_sum += abs_symmetric_dif(cand_val, comp_val) * w

        sym_dif = sym_dif_sum / sum(weights)

        if (sym_dif < tolerance) or adjust_outside_tolerance:
            # choose cell within radius that with closest symmetric difference
            if _default_choice(sym_dif, best):
                best["best_coord_index"] = coord_index
                best["best_dif"] = sym_dif
                best["best_values"] = cand_values
                best["best_in_tolerance"] = (
                    True if best["best_dif"] <= tolerance else False
                )
                best["best_is_naive"] = False

    return (
        best["best_is_naive"],
        best["best_coord_index"],
        best["best_dif"],
        best["best_values"],
    )


def do_snap(
    source_coord,
    target: snap_pair,
    supplements: typing.Union[snap_pair, list[snap_pair], None] = None,
    radius=1,
    tolerance=0.05,
    adjust_outside_tolerance=False,
    target_suffix="target",
    source_suffix="source",
):
    x, y = source_coord
    snap_key, source_from_val, snap_da = target

    da = ensure_list(snap_da)
    cvals = ensure_list(source_from_val)

    is_naive, idx, dif, values = comparison_guess(
        source_coord,
        radius,
        da,
        cvals,
        tolerance=tolerance,
        adjust_outside_tolerance=adjust_outside_tolerance,
    )

    target_cell = snap_da[idx]
    target_lon = target_cell.lon.data.tolist()
    target_lat = target_cell.lat.data.tolist()

    result = {
        f"xCoord{source_suffix}": source_coord[0],
        f"yCoord{source_suffix}": source_coord[1],
        f"{snap_key}{source_suffix}": source_from_val,
        f"xCoord{target_suffix}": target_lon,
        f"yCoord{target_suffix}": target_lat,
        f"{snap_key}{target_suffix}": values[0],
        "NetSymmetricDifference": dif,
        "is_naive": is_naive,
    }

    if supplements is not None:
        supplements = ensure_list(supplements)
        for sup_key, sup_from_val, sup_da in supplements:
            sup_val = sup_da[idx].data.tolist()
            result[f"{sup_key}{source_suffix}"] = (
                sup_from_val if sup_from_val is not None else np.nan
            )
            result[f"{sup_key}{target_suffix}"] = sup_val

    return result


def assert_gdf(gdf, columns):
    # check 4326I

    # check desired columns present
    return gdf


def snap_gdf(
    gdf,
    target_col,
    supplement_cols,
    radius=1,
    tolerance=0.05,
    adjust_outside_tolerance=False,
    source_suffix="source",
    target_suffix="target",
    passthrough_cols=None,
    post_report=True,
):

    gdf = assert_gdf(gdf, columns=gdf.columns)

    if passthrough_cols is not None:
        passthrough_cols = ensure_list(passthrough_cols)
    else:
        passthrough_cols = [c for c in gdf.columns if c != "geometry"]

    def _apply_snap(rec):
        coord = (rec.geometry.x, rec.geometry.y)
        target = (target_col[0], rec[target_col[1]], target_col[2])

        result = {}
        if passthrough_cols is not None:
            for p in passthrough_cols:
                result[p] = rec[p]

        kwargs = dict()
        if supplement_cols is not None:
            supplements = ensure_list(supplement_cols)
            sups = [(sup[0], rec[sup[1]], sup[2]) for sup in supplements]
            kwargs["supplements"] = sups

        kwargs["radius"] = radius
        kwargs["tolerance"] = tolerance
        kwargs["adjust_outside_tolerance"] = adjust_outside_tolerance
        kwargs["target_suffix"] = target_suffix
        kwargs["source_suffix"] = source_suffix

        snap_func = partial(do_snap, **kwargs)
        snap_result = snap_func(coord, target)
        result.update(snap_result)

        return result

    targetx = f"xCoord{target_suffix}"
    targety = f"yCoord{target_suffix}"
    snap_results = gdf.apply(_apply_snap, axis=1, result_type="expand")
    snap_results = gpd.GeoDataFrame(
        snap_results,
        geometry=gpd.points_from_xy(
            x=snap_results[targetx], y=snap_results[targety], crs=4326
        ),
    )

    # remove garbage columns
    snap_results = clean_df(snap_results)
    if post_report:
        report = snap_post_report(
            snap_results,
            source_suffix=source_suffix,
            target_suffix=target_suffix,
            factors=[target_col[0]],
            tolerance=tolerance,
        )
        return snap_results, report

    return snap_results


def add_network_info(df, xcol: str, ycol: str, gdbn: Path, suffix=None):
    if suffix is None:
        suffix = gdbn.name.split(".")[0].split("_")[-2]

    rgis = Rgis()
    gdbt = rgis.table2rgis(df)
    gdbp_from = rgis.tblConv2Point(gdbt, xcol, ycol)
    gdbp_from_char = rgis.pntSTNChar(gdbp_from, gdbn, suffix=suffix)
    df_char = rgis.rgis2df(gdbp_from_char)

    return df_char


def _snap_distance(
    row,
    x_orig="xCoord30sec",
    y_orig="yCoord30sec",
    x_new="xCoord01min",
    y_new="yCoord01min",
):
    orig = (row[y_orig], row[x_orig])
    new = (row[y_new], row[x_new])
    dist = float(geodesic(orig, new).km)

    return dist


def snap_post_report(
    snap_df,
    source_suffix="30sec",
    target_suffix="01min",
    factors=[
        "STNCatchmentArea",
    ],
    tolerance=0.05,
):
    df = snap_df.copy()

    _snap_distance_foo = partial(
        _snap_distance,
        x_orig=f"xCoord{source_suffix}",
        y_orig=f"yCoord{source_suffix}",
        x_new=f"xCoord{target_suffix}",
        y_new=f"yCoord{target_suffix}",
    )

    df["snap_distance_km"] = df.apply(_snap_distance_foo, axis=1, result_type="reduce")
    metrics = {}

    metrics["count"] = len(df)
    metrics["count_adjusted"] = len(df[df["is_naive"] == False])  # noqa: E712
    metrics["count_outside_tolerance"] = len(
        df[df["NetSymmetricDifference"] > tolerance]
    )

    for f in factors:

        def _add_symdif(row):
            return symmetric_dif(
                row[f"{f}{source_suffix}"], row[f"{f}{target_suffix}"], absolute=True
            )

        df[f"{f}_symdif"] = df.apply(_add_symdif, axis=1, result_type="reduce")
        metrics[f"count_outside_tolerance_{f}"] = len(
            df[df[f"{f}_symdif"].abs() > tolerance]
        )

        fd = df[f"{f}_symdif"].abs().describe()
        metrics[f"{f}_symdif_mean"] = fd["mean"]
        metrics[f"{f}_symdif_min"] = fd["min"]
        metrics[f"{f}_symdif_25%"] = fd["25%"]
        metrics[f"{f}_symdif_50%"] = fd["50%"]
        metrics[f"{f}_symdif_75%"] = fd["75%"]
        metrics[f"{f}_symdif_max"] = fd["max"]

        # overall factor correlation
        metrics[f"{f}_corr"] = (
            df[[f + source_suffix, f + target_suffix]]
            .corr()
            .loc[f + source_suffix, f + target_suffix]
        )

    sdd = df["snap_distance_km"].describe()
    metrics["snap_distance_km_mean"] = sdd["mean"]
    metrics["snap_distance_km_std"] = sdd["std"]
    metrics["snap_distance_km_min"] = sdd["min"]
    metrics["snap_distance_km_25%"] = sdd["25%"]
    metrics["snap_distance_km_50%"] = sdd["50%"]
    metrics["snap_distance_km_75%"] = sdd["75%"]
    metrics["snap_distance_km_max"] = sdd["max"]

    return metrics


def get_over_under(
    df,
    sym_dif_col,
    tolerance,
):
    """rows of dataframe outside tolerance post snap"""

    over = df[df[sym_dif_col] > tolerance]

    under = df[df[sym_dif_col] < (tolerance * -1)]

    return over, under
