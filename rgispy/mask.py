import datetime

# Type Hints
from io import StringIO

import numpy as np
import rasterio as rio
from affine import Affine
from xarray.core.dataarray import DataArray as xarray_da
from xarray.core.dataset import Dataset as xarray_ds

from .grid import count_non_nan, get_buffer_cells, non_nan_cells
from .network import get_all_basin_mouth


def get_empty_da(da_template: xarray_da, da_name: str) -> xarray_da:
    """Get an DataArray of nan of the same shape as another DataArray

    Args:
        da_template (xarray_da): DataArray to use as template
        da_name (str): Name of new DataArray

    Returns:
        xarray_da: DataArray with shape of da_template containing only np.nan
    """
    da = da_template.copy().rename(da_name)
    da.values[:] = np.nan
    return da


def get_basin_mouth_mask(network: xarray_ds) -> xarray_da:
    """From a given rgispy network, generate a mask of all the basin mouth cells

    Args:
        network (xarray DataSet): network with ID DataArray

    Returns:
        xarray DataArray: mask DataArray of basin mouths
    """
    basin_mouths = get_all_basin_mouth(network)

    mask = get_empty_da(network["ID"], "BasinMouth")

    for _, cell_idx in basin_mouths:
        cellid = network["ID"][cell_idx].data.tolist()
        mask[cell_idx] = cellid
    mask = mask.assign_attrs({"description": "Mouth cells of basins", "Type": "Point"})
    return mask


def get_mask_ds(network: xarray_ds) -> xarray_ds:
    """Get a xarray dataset skeleton for setting mask variables to

    Args:
        network (xarray Dataset): xarray Dataset of network

    Returns:
        xarray Dataset: xarray Dataset with appropriate skeleton for given network
    """
    todrop = list(network.data_vars.keys())
    if "spatial_ref" in todrop:
        todrop.remove("spatial_ref")
    if "ID" in todrop:
        todrop.remove("ID")

    mask_ds = network.copy().drop(todrop)
    mask_ds = mask_ds.assign_attrs(
        {
            "How to use stored Attribute Tables": "Import as pandas dataframe using command: pd.read_csv(StringIO(<xArray>.attrs['Attribute Table']),sep='\t')",
            "creation_date": "{}".format(datetime.datetime.now()),
            "crs": "+init=epsg:4326",
        }
    )
    return mask_ds


def mask_set_attrs(
    mask,
    description="",
    wbm_filename="",
    wbm_fieldname="",
    processing_type="",
    mask_type="Point",
):
    mask.attrs["description"] = description
    mask.attrs["WBM_filename"] = wbm_filename
    mask.attrs["WBM_fieldname"] = wbm_fieldname
    mask.attrs["Processing type"] = processing_type
    mask.attrs["Type"] = mask_type

    mask.attrs["Number Cells Occupied"] = count_non_nan(mask.data)
    if mask_type.lower() == "polygon":
        mask.attrs["Number Polygons"] = len(np.unique(mask.data[~np.isnan(mask.data)]))

    return mask


def mask_set_att_table(mask, att_table):
    mask.attrs[
        "How to use stored Attribute Tables"
    ] = "Import as pandas dataframe using command: pd.read_csv(StringIO(<xArray>.attrs['Attribute Table']),sep='\t')"
    out_buf = StringIO()
    att_table.to_csv(out_buf, sep="\t", index=False)
    mask.attrs["Attribute Table"] = out_buf.getvalue()
    return mask


def get_point_mask_from_df(
    df,
    network,
    description="",
    values="WBM IDs for layer",
    wbm_filename="",
    wbm_fieldname="",
    processing_type="",
    mask_type="Point",
):
    cellids = df.CellID.unique()

    mask = network.where(network["ID"].isin(cellids))["ID"]
    mask = mask_set_attrs(
        mask,
        description=description,
        wbm_filename=wbm_filename,
        wbm_fieldname=wbm_fieldname,
        processing_type=processing_type,
        mask_type=mask_type,
    )
    mask = mask_set_att_table(mask, df)
    return mask


def gdf_mask(gdf, network_xr) -> np.ndarray:

    assert 0 not in gdf.ID, "ID cannot be zero indexed"
    shp_shapes = gdf[["ID", "geometry"]].to_records(index=False)

    cell_sum = 0
    mask_shape = network_xr["ID"].shape
    aff = Affine.from_gdal(*eval(network_xr.affine))

    # zeros used to additively create mask
    final_mask = np.zeros(mask_shape)
    for ind, shape in shp_shapes:
        shp_mask = rio.features.geometry_mask(
            [
                shape,
            ],
            mask_shape,
            aff,
            all_touched=False,
            invert=True,
        )
        cell_sum += count_non_nan(network_xr["ID"].data[shp_mask])

        mask = np.zeros(mask_shape)
        mask[shp_mask] = ind
        final_mask = np.sum([final_mask, mask], axis=0)

    # set remaining zeros to nan
    final_mask[final_mask == 0] = np.nan
    assert cell_sum == count_non_nan(
        final_mask
    ), f"{cell_sum} != {count_non_nan(final_mask)}"

    return final_mask


def get_polygon_mask_from_gdf(
    gdf,
    network_xr,
    description="",
    values="WBM IDs for layer",
    wbm_filename="",
    wbm_fieldname="",
    processing_type="",
    mask_type="Polygon",
):

    mask_nd = gdf_mask(gdf, network_xr)
    mask = network_xr["ID"].copy()
    mask.data = mask_nd

    mask = mask_set_attrs(
        mask,
        description=description,
        wbm_filename=wbm_filename,
        wbm_fieldname=wbm_fieldname,
        processing_type=processing_type,
        mask_type=mask_type,
    )

    non_geom_cols = [c for c in gdf.columns if c != "geometry"]
    mask = mask_set_att_table(mask, gdf[non_geom_cols])
    return mask


def _mask_buffer_single(network, buffer, mask, lati, loni):
    def _lookup_cellid(lati, loni):
        cellid = network["ID"][lati, loni].data.tolist()
        return cellid

    lat_max, lon_max = mask.shape
    buffer_cells = get_buffer_cells(lati, loni, buffer, lat_max, lon_max)
    for i, j in buffer_cells:
        cellid = _lookup_cellid(i, j)
        mask[i, j] = cellid

    return mask


def mask_buffer(point_mask_da, network, buffer, mask_name=None):
    da = point_mask_da.copy()
    grid_indexes = non_nan_cells(da.data)
    mask = da.data
    for latlon in grid_indexes:
        mask = _mask_buffer_single(network, buffer, mask, latlon[0], latlon[1])

    da.data = mask

    if mask_name:
        da = da.rename(mask_name)
    else:
        new_name = da.name + "_Buffer{}".format(buffer)
        da = da.rename(new_name)

    new_desc = da.description + " with {} cell buffer".format(buffer)
    da.attrs.update(description=new_desc)
    return da
