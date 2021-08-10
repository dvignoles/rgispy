import argparse
import sys
from pathlib import Path

from ..datastream import sample_ds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("mask_nc", nargs="?", type=Path, help="netcdf mask file")

    parser.add_argument(
        "--mask_layers", nargs="+", default=[], help="Which mask layers to sample by"
    )

    parser.add_argument(
        "datastream", nargs="?", type=argparse.FileType("rb"), default=sys.stdin.buffer
    )

    parser.add_argument("--outdir", nargs="?", type=Path, help="Directory to output to")

    parser.add_argument("--year", nargs="?", help="Year of datastream file")
    parser.add_argument("--variable", nargs="?", help="Variable of datastream file")

    args = parser.parse_args()
    sample_ds(
        args.mask_nc,
        args.datastream,
        args.mask_layers,
        args.outdir,
        args.year,
        args.variable,
    )
