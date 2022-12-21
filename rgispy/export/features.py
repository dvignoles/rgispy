from pathlib import Path

import geopandas as gpd
import pandas as pd

from .. import core, util
from . import postgres


def _correct_id_col(df, expected_id_col):
    # capitalization issues
    if expected_id_col not in df.columns:
        if expected_id_col.islower():
            if expected_id_col.upper() in df.columns:
                expected_id_col = expected_id_col.upper()
        if expected_id_col.isupper():
            if expected_id_col.lower() in df.columns:
                expected_id_col = expected_id_col.lower()
    return expected_id_col


def gdbp_to_gdf(wbm_gdbp: Path, normalize_col_names=True):
    gdbp = core.RgisPoint(wbm_gdbp)

    res = util.get_res(wbm_gdbp.name)
    xcoord = f"x_coord_{res}"
    ycoord = f"y_coord_{res}"
    gdf = gdbp.gdf(x=xcoord, y=ycoord)

    if normalize_col_names:
        gdf.rename(columns=util.clean_col_names(gdf.columns), inplace=True)

    gdf.loc[:, "wbm_res"] = res
    keep_coords = [
        "xcoordorig",
        "ycoordorig",
        "x_coordorig",
        "y_coordorig",
        "x_coord_orig",
        "y_coord_orig",
        "xcoord_orig",
        "ycoord_orig",
    ]
    keep_cols = [c for c in util.filter_extra_coords(gdf.columns, keep_coords)]
    keep_cols = [c for c in util.filter_garbage_cols(keep_cols)]
    gdf = gdf[keep_cols]
    return gdf


def gdbp_allres_to_gdf(feature_dir: Path, normalize_col_names=True):
    feature_files = sorted(feature_dir.rglob("*.gdbp*"))
    gdfs = [
        gdbp_to_gdf(f, normalize_col_names=normalize_col_names) for f in feature_files
    ]
    # sort from highest res to lowest (lower res have less features usually)
    gdfs = sorted(gdfs, key=lambda x: len(x), reverse=True)
    return gdfs


def read_multires_gdbp(
    top_level_dir: Path,
    join_col: str = None,
    normalize_col_names=True,
    passthrough_both=[],
    passthrough_res=[],
):
    """Read in directory of related rgis point coverages (gdbp) and normalize
    based on resolution.

    Args:
        top_level_dir (Path): directory of feature
        join_col (str, optional): unique identifer column. Defaults to None.

    Returns:
        df, [gdf]: primary table, [resolution specific tables, ...]
    """
    all_gdf = util.gdbp_allres_to_gdf(
        top_level_dir, normalize_col_names=normalize_col_names
    )

    consistent, inconsistent = util.split_consistent_dfs(
        all_gdf,
        join_col=join_col,
        passthrough_both=passthrough_both,
        passthrough_res=passthrough_res,
    )

    return consistent, inconsistent


def get_rgis_feature_gdfs(
    top_level_dir,
    join_col,
    real_x_col,
    real_y_col,
    normalize_col_names=True,
    passthrough_both=[],
    passthrough_res=[],
):

    if normalize_col_names:
        real_x_col = util.clean_col_name(real_x_col)
        real_y_col = util.clean_col_name(real_y_col)
    base, res_dependent = read_multires_gdbp(
        top_level_dir,
        join_col=join_col,
        normalize_col_names=normalize_col_names,
        passthrough_both=passthrough_both,
        passthrough_res=passthrough_res,
    )
    base = gpd.GeoDataFrame(
        base,
        geometry=gpd.points_from_xy(x=base[real_x_col], y=base[real_y_col], crs=4326),
    )

    return base, res_dependent


def gdf_to_gdbp(
    gdf,
    gdbp_path: Path,
    replace_path=False,
):
    gdf.loc[:, "geom_x"] = gdf.geometry.x
    gdf.loc[:, "geom_y"] = gdf.geometry.y

    # ensure cellid columns saved as integers not floats
    cellid_floats = [
        c for c in gdf.columns if "cellid" in c.lower() if gdf[c].dtype == float
    ]
    for c in cellid_floats:
        gdf.loc[:, c] = gdf[c].astype(pd.Int64Dtype())

    gdbp = core.RgisPoint.from_df(gdf, "geom_x", "geom_y")
    gdf.drop(["geom_x", "geom_y"], axis=1, inplace=True)
    gdbp.tbl_delete_field(
        "geom_x",
    )
    gdbp.tbl_delete_field(
        "geom_y",
    )

    gdbp.to_file(
        gdbp_path,
        gzipped=True,
        replace_path=replace_path,
    )

    return gdbp_path


def _pg_create_rgis_feature_tables(
    con,
    base,
    res_dependent,
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

    # add to postgres with appropriate relations and indices
    con.execute(f"DROP TABLE IF EXISTS {base_table_name} CASCADE;")

    # exclude existing id/ID
    base_cols = [c for c in base.columns if c not in ["id", "ID"]]

    base[base_cols].to_postgis(
        base_table_name, con, if_exists="replace", schema="public", index=False
    )
    postgres.set_primary_key(con, base_table_name, key_col=join_col)

    # exclude existing id/ID
    res_cols = [c for c in res_dependent[0].columns if c not in ["id", "ID"]]
    res_dependent[0][res_cols].to_postgis(
        wbm_table_name, con, if_exists="replace", schema="public", index=False
    )
    postgres.set_primary_key_auto(con, wbm_table_name, key_col="id")
    postgres.set_foreign_key(
        con, wbm_table_name, "relate_id", base_table_name, join_col
    )

    for gdf in res_dependent[1:]:
        gdf[res_cols].to_postgis(
            wbm_table_name, con, if_exists="append", schema="public", index=False
        )


def pg_create_rgis_feature_tables(
    con,
    top_level_dir: Path,
    join_col: str,
    base_table_name: str,
    wbm_table_name: str,
    real_x_col: str,
    real_y_col: str,
    normalize_col_names=True,
    passthrough_both=[],
    passthrough_res=[],
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
        top_level_dir,
        join_col,
        real_x_col,
        real_y_col,
        normalize_col_names=normalize_col_names,
        passthrough_both=passthrough_both,
        passthrough_res=passthrough_res,
    )

    return _pg_create_rgis_feature_tables(
        con,
        base,
        res_dependent,
        join_col,
        base_table_name,
        wbm_table_name,
        real_x_col,
        real_y_col,
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
        top_level_dir, join_col, real_x_col, real_y_col, id_col="id"
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
