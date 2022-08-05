from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import postgres, util


def read_multires_gdbp(top_level_dir: Path, join_col: str = None, id_col: str = "id"):
    """Read in directory of related rgis point coverages (gdbp) and normalize
    based on resolution.

    Args:
        top_level_dir (Path): directory of feature
        join_col (str, optional): unique identifer column. Defaults to None.
        id_col (str, optional): default primary key column. Defaults to "id".

    Returns:
        df, [gdf]: primary table, [resolution specific tables, ...]
    """
    all_gdf = util.gdbp_allres_to_gdf(top_level_dir)
    consistent, inconsistent = util.split_consistent_dfs(
        all_gdf, join_col=join_col, id_col=id_col
    )

    return consistent, inconsistent


def get_rgis_feature_gdfs(top_level_dir, join_col, real_x_col, real_y_col):
    real_x_col = util.clean_col_name(real_x_col)
    real_y_col = util.clean_col_name(real_y_col)
    base, res_dependent = read_multires_gdbp(
        top_level_dir, join_col=join_col, id_col="id"
    )
    base = gpd.GeoDataFrame(
        base,
        geometry=gpd.points_from_xy(x=base[real_x_col], y=base[real_y_col], crs=4326),
    )

    return base, res_dependent


def pg_create_rgis_feature_tables(
    con,
    top_level_dir: Path,
    join_col: str,
    base_table_name: str,
    wbm_table_name: str,
    real_x_col: str,
    real_y_col: str,
):
    """Create normalized postgres tables for rgis point coverage multi resolution set

    Args:
        con (sqlAlchemy Connection)
        top_level_dir (Path): directory of feature
        join_col (str): dataset specific identifiet (grand_id, usgs_site_no, etc...)
        base_table_name (str): name for feature table
        wbm_table_name (str): name for wbm relationship table
        real_x_col (str): name of column with in situ longitude
        real_y_col (str): name of column with in situ latitude
    """

    #  Normalize into main table -> wbm tables dataframes
    base, res_dependent = get_rgis_feature_gdfs(
        top_level_dir, join_col, real_x_col, real_y_col
    )

    # add to postgres with appropriate relations and indices
    con.execute(f"DROP TABLE IF EXISTS {base_table_name} CASCADE;")
    base.to_postgis(
        base_table_name, con, if_exists="replace", schema="public", index=False
    )
    postgres.set_primary_key(con, base_table_name, key_col="id")
    postgres.create_index(con, base_table_name, join_col, unique=True)

    res_dependent[0].to_postgis(
        wbm_table_name, con, if_exists="replace", schema="public", index=False
    )
    postgres.set_primary_key_auto(con, wbm_table_name, key_col="id")
    postgres.set_foreign_key(con, wbm_table_name, "relate_id", base_table_name, "id")
    postgres.create_index(con, wbm_table_name, ["wbm_res", "sample_id"])
    postgres.create_index(con, wbm_table_name, join_col)

    for gdf in res_dependent[1:]:
        gdf.to_postgis(
            wbm_table_name, con, if_exists="append", schema="public", index=False
        )


def pg_upsert_rgis_feature_tables(
    con,
    top_level_dir: Path,
    join_col: str,
    base_table_name: str,
    wbm_table_name: str,
    real_x_col: str,
    real_y_col: str,
):
    #  Normalize into main table -> wbm tables dataframes
    base, res_dependent = get_rgis_feature_gdfs(
        top_level_dir, join_col, real_x_col, real_y_col
    )

    postgres.upsert_df(
        con,
        base,
        base_table_name,
        [
            join_col,
        ],
    )

    res_unique_cols = [join_col, "wbm_res"]
    res_dependent = pd.concat(res_dependent)
    postgres.upsert_df(con, res_dependent, wbm_table_name, res_unique_cols)
