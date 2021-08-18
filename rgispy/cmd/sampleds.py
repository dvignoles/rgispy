import argparse
import gzip
import sys
from pathlib import Path

from ..sample import sample_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datastream", nargs="?", type=argparse.FileType("rb"), default=sys.stdin.buffer
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

    # deal with gzip compressed and stdin variations
    datastream_name = args.datastream.name
    if datastream_name != "<stdin>":
        extension = datastream_name.split(".")[-1]
        assert extension in [
            "gz",
            "gds",
            "ds",
        ], "extension must be either gz or gds or ds"
        if extension == "gz":
            datastream = gzip.open(datastream_name)
            args.datastream.close()
        elif extension in ["gds", "ds"]:
            datastream = args.datastream
    else:
        datastream = args.datastream

    sample_ds(
        args.mask_nc,
        datastream,
        args.mask_layers,
        args.outdir,
        args.year,
        args.variable,
        args.timestep,
    )
