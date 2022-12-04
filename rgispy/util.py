"""Standalone internal utility functions"""
import re
from pathlib import Path

import pandas as pd

RES_REGEX_DELIM = re.compile(r"(\B\d{1,2}((min\B){1}|(sec\B){1}|(m\d{2}s\B){1}){1})")
RES_REGEX_NODELIM = re.compile(r"(\d{1,2}((min){1}|(sec){1}|(m\d{2}s){1}){1})")

# matches 4 digit year after [ats, mts, dts, alt, mlt, dlt]
YEAR_REGEX_DELIM = re.compile(r"(?<=([amd](TS|ts|lt|LT)))\d{4}")

# assumes variable prefeix by 'Output_'
VARIABLE_REGEX_DELIM = re.compile(r"(?<=(Output_))[^_]+")


def get_res(wbm_filename):
    # matches (01min, 2m30s, 30sec)
    match = RES_REGEX_DELIM.search(wbm_filename).group()
    assert match != "", f"no match found in {wbm_filename}"
    return match


def get_year(wbm_filename):
    match = YEAR_REGEX_DELIM.search(wbm_filename).group()
    assert match != "", f"no match found in {wbm_filename}"
    match = int(match)
    return match


def get_ds_variable(wbm_ds_filename):
    """Get variable name of datastream Output"""
    match = VARIABLE_REGEX_DELIM.search(wbm_ds_filename).group()
    assert match != "", f"no match found in {wbm_ds_filename}"
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


def filter_garbage_cols(cols):
    def _filter_garb(col):
        if "." in col:
            if col.split(".")[0] in cols:
                return False
            else:
                return True
        elif col.startswith("Unnamed:"):
            return False
        else:
            return True

    return list(filter(_filter_garb, cols))


def _check_dfs_ordered(dfs):
    dfs = sorted(dfs, key=lambda x: len(x), reverse=True)
    lengths = set(map(lambda df: len(df.columns), dfs))
    assert len(lengths) == 1, "inconsistent columns between dataframes"

    return dfs


def separate_consistent_cols(
    dfs,
    passthrough_both=["id", "sampleid", "sample_id"],
    passthrough_res=[],
    skip=[
        "geometry",
    ],
):
    """distinguish between columns which are subsets of higher res
    dataframe and those that are unique to resolution"""
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
        c
        for c in dfs[0].columns
        if ((c in consistent_cols) or (c in passthrough_both))
        and (c not in passthrough_res)
    ]
    inconsistent = [
        c
        for c in dfs[0].columns
        if (c not in consistent) or (c in passthrough_both) or (c in passthrough_res)
    ]

    return consistent, inconsistent


def split_consistent_dfs(dfs, join_col=None, passthrough_both=[], passthrough_res=[]):
    dfs = _check_dfs_ordered(dfs)
    passthrough = ["id", "sampleid", "sample_id", "ID", "SampleID"]
    for c in passthrough_both:
        if c not in passthrough:
            passthrough.append(c)

    if join_col is not None:
        passthrough.append(join_col)

    consistent, inconsistent = separate_consistent_cols(
        dfs, passthrough_both=passthrough, passthrough_res=passthrough_res
    )
    res_agnostic_df = dfs[0][consistent]
    res_dependent_dfs = [df[inconsistent] for df in dfs]

    # set up relation
    if join_col is not None:

        def _lookup_relateid(to_id):
            relateid = res_agnostic_df[res_agnostic_df[join_col] == to_id][
                join_col
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
            # moved to relate_id
            if join_col in df.columns:
                df.drop(join_col, axis=1, inplace=True)

        pd.options.mode.chained_assignment = "warn"

    return res_agnostic_df, res_dependent_dfs


def n_records(year: int, time_step: str) -> int:
    """Get number of expected records in datastream based on time_step

    Args:
        year (int): year of datastream file
        time_step (str): annual, monthly, or daily

    Returns:
        int: number of records (ex: 365 for daily non-leap year datastream)
    """

    time_step = time_step.lower()
    assert time_step in [
        "annual",
        "monthly",
        "daily",
        "alt",
        "dlt",
    ], "time_step must be annual monthly or daily"

    if time_step == "annual":
        return 1
    elif time_step == "monthly":
        return 12
    elif time_step == "daily":
        p = pd.Period("{}-01-01".format(year))
        if p.is_leap_year:
            days = 366
        else:
            days = 365
        return days
    # Long term average files
    elif time_step == "alt":
        return 1
    elif time_step == "dlt":
        return 365


def get_date_format(time_step):

    time_step = time_step.lower()
    if time_step == "daily":
        return "%Y-%m-%d"
    elif time_step == "monthly":
        return "%Y-%m"
    elif time_step == "annual":
        return "%Y"
    elif time_step == "alt":
        return None
    elif time_step == "dlt":
        return None


def _gen_date_cols(time_step, year):
    time_step = time_step.lower()
    valid_time_steps = [
        "annual",
        "monthly",
        "daily",
        "alt",
        "dlt",
    ]
    assert time_step in valid_time_steps, "time_step must be in {}".format(
        valid_time_steps
    )

    freq_d = {
        "annual": "YS",
        "monthly": "MS",
        "daily": "D",
    }
    if time_step in ["daily", "monthly", "annual"]:

        date_cols = pd.date_range(
            start="1/1/{}".format(year),
            end="12/31/{}".format(year),
            freq=freq_d[time_step],
        )
        return date_cols

    # Long Term Average files
    elif time_step == "alt":
        date_cols = [
            "XXXX",
        ]
        return date_cols

    elif time_step in [
        "dlt",
    ]:
        dummy_dates = pd.date_range(start="1/1/2001", end="12/31/2001", freq="D")
        year_mon = [(str(d.month).zfill(2), str(d.day).zfill(2)) for d in dummy_dates]
        if time_step == "dlt":
            return ["XXXX--{}-{}".format(m, d) for m, d in year_mon]


def get_encoding(min_val, max_val) -> tuple[str, int]:
    """Get best encoding type and fill number.

    Args:
        min_val (numeric): minimum value of data to encode
        max_val (numeric): maximum value of data to encode
    Returns:
        tuple[str, int]: (encoding name, fill value)
    """
    if min_val >= 0:  # if all values are positive, we can use unsigned integers
        if max_val < 255:
            return "uint8", 255
        elif max_val < 65535:
            return "uint16", 65535
        elif max_val < 4294967295:
            return "uint32", 4294967295
        elif max_val < 18446744073709551615:
            return "uint64", 18446744073709551615
        else:
            raise Exception(
                "max_val value: {}... Unable to code data to unsigned int type!".format(
                    max_val
                )
            )
    else:  # otherwise we use signed integers
        if max_val <= 127:
            return "int8", -128
        elif max_val <= 32767:
            return "int16", -32768
        elif max_val <= 2147483647:
            return "int32", -2147483648
        elif max_val <= 9223372036854775807:
            return "int64", -9223372036854775808
        else:
            raise Exception(
                "max_val value: {}... Unable to code data to signed int type!".format(
                    max_val
                )
            )


def _unique_extenions_dir(dir):
    """Return unique extensions of files within directory excluding .gz"""
    unq = []
    for child in Path(dir).iterdir():
        if child.is_file():
            for ext in child.suffixes:
                if ext not in [".gz", ".log"]:
                    unq.append(ext)
    unq = list(set(unq))
    return unq


def _unique_extenions_files(files):
    """Return unique extensions of files within directory excluding .gz"""
    unq = []
    for child in files:
        child = Path(child)
        if child.is_file():
            for ext in child.suffixes:
                if ext not in [".gz", ".log"]:
                    unq.append(ext)
    unq = list(set(unq))
    return unq
