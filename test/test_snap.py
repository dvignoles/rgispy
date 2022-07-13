from pathlib import Path

import geopandas as gpd
import pytest
import xarray as xr

from rgispy import snap


@pytest.fixture
def network():
    ds = xr.open_dataset(
        Path(__file__).parent.joinpath("fixtures/CONUS_Network_15min.nc")
    )
    return ds


@pytest.fixture
def points():
    gdf = gpd.read_file(Path(__file__).parent.joinpath("fixtures/points.geojson"))
    return gdf


def test_get_cellid(network):
    assert snap.get_cell((-104.519, 33.291), network["ID"])[0] == 10518
    assert snap.get_cell((-91.7681, 36.1459), network["ID"])[0] == 3005
    assert snap.get_cell((-120.216, 50.152), network["ID"])[0] == 11013


def test_add_cellid(network, points):
    gdf = snap.add_cellid(points, network["ID"], suffix="_test")

    assert gdf.iloc[0].cellid_test == 12548
    assert gdf.iloc[1].cellid_test == 2596
    assert gdf.iloc[2].cellid_test == 12606


def test_get_buffer_values(network):
    coord = (-95.835, 41.357)
    da = network["SubbasinArea"]

    # int just to circumvent floating point error
    desired = [
        int(x)
        for x in set(
            [
                860494.56,
                1104102.5,
                582.149,
                579.9257,
                2312.9995,
                577.6913,
                856441.75,
                579.9257,
                242443.7,
            ]
        )
    ]
    result = [
        int(x) for x in set(snap.get_buffer_values(coord, da, radius=1).data.flatten())
    ]

    assert desired == result


def test_pre_snap_stats(network):
    stats = snap.pre_snap_stats(
        (-95.835, 41.357), 2300, network["SubbasinArea"], radius=1
    )
    assert int(stats["best_val"]) == 2312
