"""Snapping utilities"""

import geopandas as gpd
import xarray as xr

xarray_ds = xr.core.dataset.Dataset
xarray_da = xr.core.dataarray.DataArray


def get_cellid(
    coord: tuple[float, float], cellid_da: xarray_da
) -> tuple[int, tuple[float, float]]:
    """Get cellid number and coordinate for given point

    Args:
        coord (tuple[float, float]): (lon, lat) pair
        cellid_da (xr.DataArray): DataArray of cellid

    Returns:
        tuple[int, tuple[float, float]] -> cellid, (lon, lat)
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
        cellid, cellid_coord = get_cellid(coord, cellid_da)

        row.loc[f"cellid{suffix}"] = cellid
        row.loc[f"xcoord{suffix}"] = cellid_coord[0]
        row.loc[f"ycoord{suffix}"] = cellid_coord[1]

        return row

    result = gdf.apply(_do_mapping, axis=1, result_type="expand")
    return result
