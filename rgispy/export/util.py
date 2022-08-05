import gzip
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ..wrap import Rgis

RES_REGEX_DELIM = re.compile(r"(\B\d{1,2}((min\B){1}|(sec\B){1}|(m\d{2}s\B){1}){1})")
RES_REGEX_NODELIM = re.compile(r"(\d{1,2}((min){1}|(sec){1}|(m\d{2}s){1}){1})")


def get_res(wbm_filename):
    # matches (01min, 2m30s, 30sec)
    match = RES_REGEX_DELIM.search(wbm_filename).group()
    assert match != "", f"no match found in {wbm_filename}"
    return match


def clean_col_name(col):
    def _camel_to_snake(col):
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", col)
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

        return name

    def _rename_conflicts(col):
        # if col.lower() == "id":
        #     return "sample_id"
        # else:
        #     return col
        return col

    def _final(col):
        col = _camel_to_snake(col)
        col = _rename_conflicts(col)
        return col

    return _final(col)


def clean_col_names(columns):

    col_renames = {col: clean_col_name(col) for col in columns}
    return col_renames


def filter_extra_coords(columns, keep_cols):
    def _do_filter(col):
        col = col.lower()
        if "coord" in col:
            if col in keep_cols:
                return True
            else:
                return False
        else:
            return True

    return list(filter(_do_filter, columns))


def gdbp_to_gdf(wbm_gdbp: Path):
    res = get_res(wbm_gdbp.name)
    rgis = Rgis()
    if wbm_gdbp.suffix == ".gz":
        fo = gzip.open(wbm_gdbp, "rb")
    else:
        fo = open(wbm_gdbp, "rb")
    contents = fo.read()
    xcoord = f"x_coord_{res}"
    ycoord = f"y_coord_{res}"
    df = rgis.rgis2df(rgis.tblAddXY(contents, x=xcoord, y=ycoord))
    fo.close()

    df.rename(columns=clean_col_names(df.columns), inplace=True)
    df.loc[:, "wbm_res"] = res
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            x=df[xcoord],
            y=df[ycoord],
            crs=4326,
        ),
    )
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
    keep_cols = [c for c in filter_extra_coords(df.columns, keep_coords)]
    gdf = gdf[keep_cols]
    return gdf


def gdbp_allres_to_gdf(feature_dir: Path):
    feature_files = sorted(feature_dir.rglob("*.gdbp*"))
    gdfs = [gdbp_to_gdf(f) for f in feature_files]
    # sort from highest res to lowest (lower res have less features usually)
    gdfs = sorted(gdfs, key=lambda x: len(x), reverse=True)
    return gdfs


def _check_dfs_ordered(dfs):
    dfs = sorted(dfs, key=lambda x: len(x), reverse=True)
    lengths = set(map(lambda df: len(df.columns), dfs))
    assert len(lengths) == 1, "inconsistent columns between dataframes"

    return dfs


def separate_consistent_cols(
    dfs,
    passthrough_both=["id", "sampleid", "sample_id"],
    skip=[
        "geometry",
    ],
):
    """distinguish between columns which are subsets of higher res dataframe and those that are unique to resolution"""
    dfs = _check_dfs_ordered(dfs)
    consistent_cols = []
    for col in dfs[0].columns:
        if col.lower() in skip:
            continue
        col_contained = []
        for i in range(0, len(dfs) - 1):
            s0 = dfs[0][col]
            s1 = dfs[i + 1][col]
            contained = s1.isin(s0).all()
            col_contained.append(contained)

        if all(col_contained):
            consistent_cols.append(col)

    consistent = [
        c for c in dfs[0].columns if (c in consistent_cols) or (c in passthrough_both)
    ]
    inconsistent = [
        c for c in dfs[0].columns if (c not in consistent) or (c in passthrough_both)
    ]

    return consistent, inconsistent


def split_consistent_dfs(dfs, join_col=None, id_col="id"):
    dfs = _check_dfs_ordered(dfs)
    passthrough = ["id", "sampleid", "sample_id"]
    if join_col is not None:
        passthrough.append(join_col)

    consistent, inconsistent = separate_consistent_cols(
        dfs, passthrough_both=passthrough
    )
    res_agnostic_df = dfs[0][consistent]
    res_dependent_dfs = [df[inconsistent] for df in dfs]

    # set up relation
    if join_col is not None:

        def _lookup_relateid(to_id):
            relateid = res_agnostic_df[res_agnostic_df[join_col] == to_id][
                id_col
            ].tolist()[0]
            return relateid

        def _add_relateid(df):
            relate_series = df[join_col].map(
                _lookup_relateid,
            )
            return relate_series

        pd.options.mode.chained_assignment = None
        for df in res_dependent_dfs:
            df.rename(columns={"id": "sample_id"}, inplace=True)
            df.loc[:, "relate_id"] = _add_relateid(df)
        pd.options.mode.chained_assignment = "warn"

    return res_agnostic_df, res_dependent_dfs
