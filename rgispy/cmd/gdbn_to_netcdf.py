import argparse
from pathlib import Path

from ..network import gdbn_to_netcdf


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "gdbn",
        nargs="?",
        type=Path,
        help="netcdf mask file",
    )

    parser.add_argument(
        "output_netcdf",
        nargs="?",
        type=Path,
    )
    parser.add_argument(
        "--project",
        nargs="?",
        help="project name to embed as attr",
        default="",
        required=False,
    )

    args = parser.parse_args()
    gdbn_to_netcdf(args.gdbn, args.output_netcdf, project=args.project)
