from pathlib import Path

import pandas as pd
import xarray as xr

from .network import lookup_cellid


def join_sampled_files(part_files: list[Path]) -> pd.DataFrame:
    """Join list of sampled files into one dataframe in wide pivot format.

    Args:
        part_files (list[Path]): list-like of sampled files (Discharge_1990.csv, Discharge_1991.csv ...)

    Returns:
        pd.DataFrame: pandas dataframe in wide form (columns are dates)
    """

    part_dfs = [pd.read_csv(f, index_col=0) for f in part_files]
    df = pd.concat(part_dfs, axis=1)
    df.index.names = [
        "cellid",
    ]
    return df


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


def _lower_cols(df, copy=True):

    if copy:
        df = df.copy()
    df.columns = map(str.lower, df.columns)

    return df


def add_sampleid(sampled_df: pd.DataFrame, sampler_df: pd.DataFrame) -> pd.DataFrame:
    """Add sampleid based on 'id' of sampling feature

    Args:
        sampled_df (pd.DataFrame): sampled dataframe with cellid as index
        sampler_df (pd.DataFrame): dataframe of sampling features with 'id' column

    Returns:
        [type]: [description]
    """

    sampled_df = _lower_cols(sampled_df, copy=False)
    assert "cellid" in sampled_df.index.names
    sampler_df = sampler_df.reset_index()

    sampler_df = _lower_cols(sampler_df)
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

    # re-index
    df.reset_index(inplace=True)
    df.drop("cellid", axis=1, inplace=True)

    # if dataframe is not stacked, there is no date column
    if "date" in df.columns:
        df.set_index(["sampleid", "date"], inplace=True)
    else:
        df.set_index("sampleid", inplace=True)

    df.sort_index(inplace=True)
    return df


def normalize_sampled_files(
    sampled_files: list[Path], variable: str, sampler_ref: pd.DataFrame
) -> pd.DataFrame:
    """Convert wide form cellid indexed sampled dataframe to long form with 'sampleid' corresponding to id of sampling feature.

    Args:
        sample_files (list like): iterable of sampld files
        variable (str): name of variable (Discharge, Runoff, ... etc)
        sampler_ref (pd.Dataframe): DataFrame of sampling attribute (Guages, Dams, Country, ... etc) containing 'id' and 'cellid' columns

    Returns:
        pd.DataFrame: Normalized dataframe indexed by (sampleid, date) where sampleid corresponds to id in sampler_ref
    """

    df = join_sampled_files(sampled_files)
    df = stack_sampled_df(df, variable=variable)
    df = add_sampleid(df, sampler_ref)

    return df


def get_sampled_row(sampled_file: Path, row_to_keep: int) -> pd.DataFrame:
    """Read in specific row of sampled csv, without loading in other rows.

    Returns:
        pd.Dataframe: 1 row dataframe
    """
    rows_to_keep = [0, row_to_keep]
    df = pd.read_csv(sampled_file, header=0, skiprows=lambda x: x not in rows_to_keep)
    df.rename(columns={"Unnamed: 0": "cellid"}, inplace=True)
    df.set_index("cellid", inplace=True)
    return df


def get_row_df(sampled_files: list[Path], row_num: int) -> pd.DataFrame:
    """Read in time series DataFrame for a specific row of data across all files. Other rows are skipped.

    Args:
        sampled_files (list[Path]): list of csvs
        row_num (int): row number, 1 refers to first non-header row

    Returns:
        pd.DataFrame: 1 row DataFrame w/ all timestamps in sampled_files
    """
    dfs = [get_sampled_row(f, row_num) for f in sampled_files]
    df = pd.concat(dfs, axis=1)
    return df


def get_sampled_df_byattr(
    sampled_files: list[Path],
    df_ref: pd.DataFrame,
    identifier_name: str,
    identifier_value,
    stacked: bool = False,
    variable: str = "value",
    normalize: bool = False,
) -> pd.DataFrame:
    """Get DataFrame based on value of a specific sampling attribute.

    Ex: By NIDID of a dam (identifier_name='NIDID' , identifier_value=<nidid>)

    Args:
        sampled_files (list[Path]): list of csvs
        df_ref (pd.DataFrame): dataframe to reference sampling attributes
        identifier_name (str): column of df_ref
        identifier_value ([type]): value of identifer_name to filter by
        stacked (bool, optional): Convert DataFrame from wide form to long form. Defaults to False.
        variable (str, optional): Name of data variable (ie Discharge). Only matters if stacked=True. Defaults to "value".
        normalize (bool, optional): Convert cellid index to sampleid index (sampleid = df_ref.ID). Defaults to False.

    Returns:
        pd.DataFrame: pandas dataframe
    """
    # determine cellid based on def_ref and identifier
    cellid = int(df_ref[df_ref[identifier_name] == identifier_value].CellID.tolist()[0])
    test_file = pd.read_csv(sampled_files[0], header=0)
    test_file.index += 1
    test_file = test_file.rename(columns={"Unnamed: 0": "cellid"})

    # determine number of row to sample
    sample_row = test_file[test_file.cellid == cellid].index.tolist()[0]

    df = get_row_df(sampled_files, sample_row)

    # transforms
    if stacked:
        df = stack_sampled_df(df, variable, index_name="cellid")

    if normalize:
        df = add_sampleid(df, df_ref)

    return df
