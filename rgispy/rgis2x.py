"""Conversion functions for rgis formats (gdbc, gdz)"""

import gzip
import shutil
import subprocess as sp
import tempfile
from calendar import isleap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from rasterio.transform import from_bounds

from .sample import get_date_format, get_true_datastream, iter_ds


def gdsgz_to_rgis(gdsgz, gdbn, output_dir):
    cmd = "ds2rgis --title title123 --template {}".format(gdbn).split()
    gdbc_name = gdsgz.name.split(".")[0] + ".gdbc"
    gdbc = Path(output_dir).joinpath(gdbc_name)

    with open(gdbc, "wb") as output_file:
        p = sp.Popen(cmd, stdin=sp.PIPE, stdout=output_file)

        with gzip.open(gdsgz, "rb") as f:
            with p.stdin:
                shutil.copyfileobj(f, p.stdin)

        p.wait()

    return gdbc


def gen_date_layers(year, time_domain):
    time_domain = time_domain.lower()
    assert time_domain in ["annual", "monthly", "daily"]
    freq_d = {
        "annual": "YS",
        "monthly": "MS",
        "daily": "D",
    }
    date_cols = pd.date_range(
        start="1/1/{}".format(year),
        end="12/31/{}".format(year),
        freq=freq_d[time_domain],
    )

    if time_domain == "daily":
        return [datetime.strftime(dt, "%Y-%m-%d") for dt in date_cols]
    elif time_domain == "monthly":
        return [datetime.strftime(dt, "%Y-%m") for dt in date_cols]
    else:
        return [datetime.strftime(dt, "%Y") for dt in date_cols]


def gdbc_to_ascii(gdbc, output_dir, year, time_domain):
    date_layers = gen_date_layers(year, time_domain)
    gdbc_name = Path(gdbc).name.split(".")[0]
    asc_files = []
    for dl in date_layers:
        asc = Path(output_dir).joinpath(gdbc_name + "_" + dl + ".asc")
        cmd = "rgis2ascii -l {} {} {}".format(dl, gdbc, asc).split()
        sp.run(cmd)
        asc_files.append(asc)
    return asc_files


def asc_to_geotiff(asc_files, output_dir):

    asc_file_names = [str(asc) for asc in asc_files]

    asc_dir = Path(asc_files[0]).parent
    VRT = str(asc_dir.joinpath("OutputImage.vrt"))
    gdal.BuildVRT(VRT, asc_file_names, separate=True, outputSRS="EPSG:4326")

    # set dates as band descriptions
    InputImage = gdal.Open(VRT, gdal.GA_Update)
    for i, asc in enumerate(asc_files):
        dt_layer = asc.name.split(".")[0].split("_")[-1]
        band = InputImage.GetRasterBand(i + 1)
        band.SetDescription(dt_layer)

    # convert vrt to geotiff
    output_tif_name = asc_files[0].name.split(".")[0].rsplit("_", 1)[0] + ".tif"
    output_tif = Path(output_dir).joinpath(output_tif_name)
    gdal.Translate(
        str(output_tif),
        InputImage,
        format="GTiff",
        creationOptions=["COMPRESS:DEFLATE", "TILED:YES"],
    )
    del InputImage  # close the VRT

    return output_tif


def gdbc_to_geotiff(gdbc, year, time_domain, output_dir, tmp_dir_root="/tmp"):
    tmp_dir = tempfile.TemporaryDirectory(dir=tmp_dir_root, prefix="rgispy_")
    asc_files = gdbc_to_ascii(gdbc, tmp_dir.name, year, time_domain)
    output_tif = asc_to_geotiff(asc_files, output_dir)

    tmp_dir.cleanup()
    return output_tif


def build_overviews(tif, resampling="AVERAGE"):

    image = gdal.Open(str(tif), 1)

    # create levels until smaller than 256x tile
    largest_dimension = max(image.RasterXSize, image.RasterYSize)
    levels = []
    level = 1
    while largest_dimension > 256:
        level *= 2
        levels.append(level)
        largest_dimension = largest_dimension / 2

    image.BuildOverviews(resampling, levels)

    del image


def get_affine(network):

    west = network.lon[0].data.tolist()
    east = network.lon[-1].data.tolist()
    north = network.lat[-1].data.tolist()
    south = network.lat[0].data.tolist()

    # adjust for half pixel difference between cell corners and xarray coords (center)
    xsize_adj = abs(west - network.lon[1].data.tolist()) / 2
    ysize_adj = abs(north - network.lat[-2].data.tolist()) / 2

    west -= xsize_adj
    east += xsize_adj
    north += ysize_adj
    south -= ysize_adj

    height = network.lat.shape[0]
    width = network.lon.shape[0]

    transform = from_bounds(west, south, east, north, width, height)
    return transform


def gdsgz_to_geotiff(gdsgz, network, year, time_domain, output_dir=Path.cwd()):
    time_domain = time_domain.lower()
    assert time_domain in ["annual", "monthly", "daily"]
    date_format = get_date_format(time_domain)
    count = 1 if time_domain == "annual" else 12
    if time_domain == "daily":
        count = 366 if isleap(year) else 365

    tif_name = gdsgz.name.split(".")[0] + ".tif"
    tif_path = output_dir.joinpath(tif_name)

    height = network.lat.shape[0]
    width = network.lon.shape[0]
    transform = get_affine(network)

    with get_true_datastream(gdsgz) as f:
        for i, data_date in enumerate(
            iter_ds(f, network["ID"].data, year, time_domain)
        ):
            arr, date = data_date
            arr = np.flipud(arr)
            if i == 0:
                assert arr.shape == (height, width)
                # if tiff total size > 4gb, need to use BIGTIFF=YES GTiff Creation option
                byte_size_est = arr.size * arr.itemsize * count * 1.1
                big_tiff = "YES" if byte_size_est > (4 * 10 ** 9) else "NO"

                tif = rasterio.open(
                    tif_path,
                    "w",
                    driver="GTiff",
                    height=arr.shape[0],
                    width=arr.shape[1],
                    count=count,
                    nodata=-9999,
                    dtype=str(arr.dtype),
                    crs="EPSG:4326",
                    transform=transform,
                    tiled=True,
                    compress="DEFLATE",
                    BIGTIFF=big_tiff,
                )

            tif.write_band(i + 1, arr)

            # must deal with lt dates later
            tif.set_band_description(i + 1, date.strftime(date_format))
        tif.close()


def gds_to_geotiff_mosaic(
    gds,
    network_id,
    year,
    time_domain,
    transform,
    output_dir=Path.cwd(),
    overviews=True,
    overview_resample="AVERAGE",
):
    time_domain = time_domain.lower()
    assert time_domain in ["annual", "monthly", "daily"]
    date_format = get_date_format(time_domain)

    with get_true_datastream(gds) as f:
        for i, data_date in enumerate(iter_ds(f, network_id, year, time_domain)):
            arr, date = data_date
            arr = np.flipud(arr)

            date_id = date.strftime("%Y%m%d")
            tif_name = gds.name.split(".")[0][:-4] + date_id + ".tif"
            output_dir_year = output_dir.joinpath(str(year))
            if not output_dir_year.exists():
                output_dir_year.mkdir(parents=True)

            tif_path = output_dir_year.joinpath(tif_name)

            tif = rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype=str(arr.dtype),
                crs="EPSG:4326",
                transform=transform,
                tiled=True,
                compress="DEFLATE",
            )
            tif.write_band(1, arr)
            # must deal with lt dates later
            tif.set_band_description(1, date.strftime(date_format))
            tif.close()

            if overviews:
                build_overviews(tif_path, resampling=overview_resample)
