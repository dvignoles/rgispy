"""Routines to make Rgis Grid Coverages (gdbc) from other data types"""
from pathlib import Path

from ..core import RgisCalculate, RgisNetwork, RgisTable


def _join_dbcells(
    df, network, output, cellid_field="cell_id", ghaas_bin=None, scratch_dir=None
):
    """Join dataframe to network dbcells table.

    Args:
        df (pd.DataFrame): Dataframe with column containing cellid for network
        network (Path): network gdbn
        output (Path): gdbn file with dbcells table joined on df
        cellid_field (str, optional): Column/Index name of df containing
            network cellid numbers. Defaults to "cell_id".
        ghaas_bin (Path, optional): rgis ghaas bin directory. Defaults to None.
        scratch_dir (Path, optional): temporary file storage directroy.
            Defaults to None.

    Returns:
        Path: location of new gdbn file with joined dbcells table
    """

    assert (
        df.index.name == cellid_field or cellid_field in df.columns
    ), f"{cellid_field} must be in df"
    if not isinstance(network, RgisNetwork):
        network = RgisNetwork(network, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

    # convert dataframe to gdbt table
    tbl = RgisTable.from_df(df, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
    _tmp_tbl = tbl._temp_rgisfile(suffix=".gdbt")
    tbl.to_file(Path(_tmp_tbl.name), replace_path=True)

    # join temp gdbt on network dbcells table
    out = network.tbl_join_tables(
        Path(_tmp_tbl.name),
        out_dataset=output,
        relate_table="DBCells",
        join_field=cellid_field,
    )
    _tmp_tbl.close()
    return out


def dbcells_to_grid(
    network,
    col,
    output_grid,
    extent=None,
    na_override=None,
    ghaas_bin=None,
    scratch_dir=None,
):
    """Output column of dbcells table as gdbc grid

    Args:
        network (Path): Path to network gdbn with dbcells table
        col (str): column of dbcells table
        output_grid (Path): output gdbc[.gz] file
        extent (Path, optional): extent of output_grd for grdCalculate.
            Defaults to input network.
        na_override (_type_, optional): Replacement value for na values in valid
            network cellids. Defaults to None.
        ghaas_bin (Path, optional): rgis ghaas bin directory. Defaults to None.
        scratch_dir (Path, optional): temporary file storage directroy.
             Defaults to None.
    """
    if not isinstance(network, RgisNetwork):
        network = RgisNetwork(network, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

    if extent is None:
        extent = network._fref

    if na_override is not None:
        # output column to temporary grid
        _tmp_grid = network._temp_rgisfile(suffix=".gdbc")
        network.netCells2Grid(col, out_grid=Path(_tmp_grid.name))

        # create reference to network na vs not na ``
        _netref = network._temp_rgisfile(suffix=".gdbc")
        network.netCells2Grid("BasinID", out_grid=Path(_netref.name))

        # fill in na with override value if in valid network cell
        expr = (
            f"{_netref.name} == nodata ? nodata"
            f" : ( {_tmp_grid.name} == nodata ? {na_override} : {_tmp_grid.name} )"
        )
        rcalc = RgisCalculate(ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)
        rcalc.grdCalculate(expr, output_grid, extent=extent)

        _tmp_grid.close()
        _netref.close()
    else:
        network.netCells2Grid(col, out_grid=output_grid)
    return output_grid


def cellattr_to_grid(
    df,
    col,
    network,
    output,
    cellid_field="cell_id",
    na_override=None,
    ghaas_bin=None,
    scratch_dir=None,
):
    """Convert dataframe column referenced by cellid number to gdbc grid

    Args:
        df (pd.DataFrame): Dataframe with network cellid reference for records
        col (str): Feature to output to grid
        network (Path): gdbn network
        output (Path): output gdbc[.gz] grid
        cellid_field (str, optional): DataFrame column or index containing
            network celllid. Defaults to "cell_id".
        na_override (_type_, optional): Replacement value for na values in valid
            network cellids. Defaults to None.
        ghaas_bin (Path, optional): rgis ghaas bin directory. Defaults to None.
        scratch_dir (Path, optional): temporary file storage directroy.
            Defaults to None.
    Returns:
        output: output gdbc[.gz] grid
    """
    if not isinstance(network, RgisNetwork):
        network = RgisNetwork(network, ghaas_bin=ghaas_bin, scratch_dir=scratch_dir)

    _cfeat_gdbn = network._temp_rgisfile(suffix=".gdbn")
    _ = _join_dbcells(
        df,
        network,
        Path(_cfeat_gdbn.name),
        cellid_field=cellid_field,
        ghaas_bin=ghaas_bin,
        scratch_dir=scratch_dir,
    )
    _ = dbcells_to_grid(
        Path(_cfeat_gdbn.name),
        col,
        output,
        extent=network._fref,
        na_override=na_override,
        ghaas_bin=ghaas_bin,
        scratch_dir=scratch_dir,
    )

    _cfeat_gdbn.close()
    return output
