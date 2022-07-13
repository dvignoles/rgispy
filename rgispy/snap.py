"""Snapping utilities"""

import typing

import geopandas as gpd
import xarray as xr

xarray_ds = xr.core.dataset.Dataset
xarray_da = xr.core.dataarray.DataArray


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


def get_buffer_values(
    coord: tuple[float, float],
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

    in_radius = value_da.isel(
        lon=range(loni_start, loni_end), lat=range(lati_start, lati_end)
    )
    return in_radius


def symmetric_dif(x, y):
    if (x + y) == 0:
        return 1
    else:
        return (x - y) / (x + y)


def pre_snap_stats(
    coord: tuple[float, float],
    snap_from_val: float,
    target_da: xarray_da,
    radius: int = 1,
    tolerances: list[int] = [5, 10, 15, 20, 25],
) -> dict:
    """Generate statistics for possible snap candidates in radius

    Args:
        coord (tuple[float, float]): coord at center of cell radisu
        snap_from_val (float): value we are snapping against
        target_da (xarray_da): target grid for snapping
        radius (int, optional): cell radius. Defaults to 1.
        tolerances (list[int], optional): Absolute symmetric difference tolerance thresholds. Defaults to [5, 10, 15, 20, 25].

    Returns:
        dict: {'best_dif': ... , 'best_val': ..., }
    """
    lons, lats = assert_coords(None, None, target_da)
    in_radius = get_buffer_values(coord, target_da, radius=radius)
    in_radius_flat = in_radius.data.flatten()

    sym_difs = [
        symmetric_dif(snap_from_val, snap_to_val) for snap_to_val in in_radius_flat
    ]
    sym_difs_abs = [abs(x) for x in sym_difs]

    best_dif = min(sym_difs_abs)
    best_val = in_radius_flat[sym_difs_abs.index(best_dif)]

    stats = {"source_val": snap_from_val, "best_dif": best_dif, "best_val": best_val}
    for t in tolerances:
        in_t = True if (best_dif <= t / 100) else False
        stats[f"in_{t}%"] = in_t

    return stats
