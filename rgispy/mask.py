import datetime

import numpy as np

from .network import get_all_basin_mouth


def get_basin_mouth_mask(network):
    basin_mouths = get_all_basin_mouth(network)

    mask = network["ID"].copy().rename("BasinMouth")
    mask.values[:] = np.nan

    for _, cell_idx in basin_mouths:
        cellid = network["ID"][cell_idx].data.tolist()
        mask[cell_idx] = cellid
    mask = mask.assign_attrs({"description": "Mouth cells of basins", "Type": "Point"})
    return mask


def get_mask_ds(network):
    """Get a xarray dataset skeleton for setting mask variables to

    Args:
        network (xarray.core.dataset.Dataset): xarray Dataset of network

    Returns:
        (xarray.core.dataset.Dataset): xarray Dataset with appropriate skeleton for given network
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
        }
    )
    return mask_ds
