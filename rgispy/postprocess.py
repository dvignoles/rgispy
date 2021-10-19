from pathlib import Path

import pandas as pd
import xarray as xr

from .network import lookup_cellid


def join_sampled_files(part_files: list[Path]) -> pd.DataFrame:
    """Join directory of sampled files into one dataframe in wide pivot format.

    Args:
        part_files (list[Path]): Directory containing csvs (Discharge_1990.csv, Discharge_1991.csv ...)

    Returns:
        pd.DataFrame: pandas dataframe in wide form (columns are dates)
    """

    part_dfs = [pd.read_csv(f, index_col=0) for f in part_files]
    df = pd.concat(part_dfs, axis=1)
    df.index.names = [
        "cellid",
    ]
    return df


def join_sampled_dir(sample_dir):
    return join_sampled_files(sorted(sample_dir.iterdir()))


def stack_sampled_df(
    sampled_df: pd.DataFrame, variable: str, index_name="cellid"
) -> pd.DataFrame:
    """Stack pivoted dataframe (wide form to long form).

    Args:
        sampled_df (pd.DataFrame): Ouput of join_sampled_files
        variable (str): Name of variable column (Discharge, Temperature, etc)
        index_name (str, optional): Name of primary index. Defaults to 'cellid'.

    Returns:
        pd.DataFrame
    """
    # stack
    df = sampled_df.stack().to_frame()
    df.index = df.index.set_names([index_name, "date"])
    df = df.rename(columns={0: variable})

    # fix index
    df.reset_index(inplace=True)
    df["date"] = pd.to_datetime(df.date)
    df.set_index([index_name, "date"], inplace=True)
    df = df.sort_index()
    return df


def georeference_sampled(
    sampled_df: pd.DataFrame, network: xr.core.dataset.Dataset
) -> pd.DataFrame:
    """Add latitude and longitude columns to sampled dataframe based on network cellid

    Args:
        sampled_df (pd.DataFrame): Dataframe with cellid index
        network (xr.core.dataset.Dataset): Network xarray dataset

    Returns:
        pd.DataFrame: sampled_df with lat/lon coords appended
    """

    network_id = network["ID"].where(network["ID"].isin(sampled_df.index), drop=True)

    coords = []
    for cell_id in sampled_df.index.get_level_values("cellid").unique():
        cell = network_id.where(
            network_id.isin(
                [
                    cell_id,
                ]
            ),
            drop=True,
        )
        lon = cell.lon.data.tolist()[0]
        lat = cell.lat.data.tolist()[0]
        coords.append((cell_id, lon, lat))

    coord_df = pd.DataFrame(coords, columns=["cellid", "longitude", "latitude"])
    coord_df.set_index("cellid", inplace=True)

    df = pd.concat([sampled_df, coord_df], axis=1)

    return df


def add_cellid(
    df: pd.DataFrame,
    network: xr.core.dataset.Dataset,
    x_coord="longitude",
    y_coord="latitude",
) -> pd.DataFrame:
    """Add cellid column to a dataframe with network snapped x/y coordinates

    Args:
        df (pd.DataFrame): DataFrame with x and y coordiante columns
        network (xr.core.dataset.Dataset): Network xarray dataset
        x_coord (str, optional): Column name of x coord value. Defaults to 'longitude'.
        y_coord (str, optional): Column name of y coord value. Defaults to 'latitude'.

    Returns:
        pd.DataFrame: df with cellid column appended
    """
    assert (x_coord in df.columns) and (y_coord in df.columns)

    def _lookup_cellid(row):
        lon = row[x_coord]
        lat = row[y_coord]
        return lookup_cellid(lon, lat, network)

    df["cellid"] = df.apply(_lookup_cellid, axis=1)
    return df


def add_sampleid(sampled_df: pd.DataFrame, sampler_df: pd.DataFrame) -> pd.DataFrame:
    """Add sampleid based on 'id' of sampling feature

    Args:
        sampled_df (pd.DataFrame): sampled dataframe with cellid as index
        sampler_df (pd.DataFrame): dataframe of sampling features with 'id' column

    Returns:
        [type]: [description]
    """

    assert "cellid" in sampled_df.index.names
    sampler_df = sampler_df.reset_index()
    assert "cellid" in sampler_df.columns
    assert "id" in sampler_df.columns
    sampler_df.set_index("cellid", inplace=True)

    cellid_df = sampler_df[
        [
            "id",
        ]
    ]

    # join on cellid indexes
    df = sampled_df.join(cellid_df)
    df.rename(columns={"id": "sampleid"}, inplace=True)
    return df


def normalize_sampled_dir(
    sample_dir: Path, variable: str, sampler_ref: pd.Dataframe
) -> pd.DataFrame:
    """Convert wide form cellid indexed sampled dataframe to long form with 'sampleid' corresponding to id of sampling feature.

    Args:
        sample_dir (Path): Directory of sampled files
        variable (str): name of variable (Discharge, Runoff, ... etc)
        sampler_ref (pd.Dataframe): DataFrame of sampling attribute (Guages, Dams, Country, ... etc) containing 'id' and 'cellid' columns

    Returns:
        pd.DataFrame: Normalized dataframe indexed by (sampleid, date) where sampleid corresponds to id in sampler_ref
    """
    df = join_sampled_dir(sample_dir)
    df = stack_sampled_df(df, variable=variable)
    df = add_sampleid(df, sampler_ref)

    df.reset_index(inplace=True)
    df.set_index(["sampleid", "date"], inplace=True)
    df.drop("cellid", axis=1, inplace=True)
    return df
