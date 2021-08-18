import argparse
from pathlib import Path

from ..sample import gdbc_to_ds_buffer, sample_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gdbc", nargs="?", type=Path, help="rgis gdbc grid file")
    parser.add_argument(
        "--network",
        "-n",
        nargs="?",
        type=Path,
        help="rgis gdbn network corresponding to gdbc",
        required=True,
    )
    parser.add_argument(
        "--mask_nc", "-m", nargs="?", type=Path, help="netcdf mask file", required=True
    )
    parser.add_argument(
        "--mask_layers",
        "-l",
        nargs="+",
        default=[],
        help="Which mask layers to sample by",
        required=True,
    )
    parser.add_argument(
        "--outdir",
        "-d",
        nargs="?",
        type=Path,
        help="Directory to output to",
        required=True,
    )
    parser.add_argument(
        "--year", "-y", nargs="?", help="Year of datastream file", required=True
    )
    parser.add_argument(
        "--variable", "-v", nargs="?", help="Variable of datastream file", required=True
    )
    parser.add_argument(
        "--timestep", "-t", nargs="?", help="annual, monthly, or daily", required=True
    )
    args = parser.parse_args()

    datastream = gdbc_to_ds_buffer(args.gdbc, args.network)
    sample_ds(
        args.mask_nc,
        datastream,
        args.mask_layers,
        args.outdir,
        args.year,
        args.variable,
        args.timestep,
    )
