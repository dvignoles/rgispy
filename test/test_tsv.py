from pathlib import Path

import pytest

from rgispy import tsv


@pytest.fixture
def tsv_gz_file():
    tsv_gz = Path(__file__).parent.joinpath("fixtures/ats.tsv.gz")
    assert tsv_gz.exists()
    return tsv_gz


def test_read_tsv_raw(tsv_gz_file):
    contents = tsv.read_tsv_raw(tsv_gz_file)
    # number expected lines
    assert len(contents.split("\n")) == 16442


def test_read_tsv(tsv_gz_file):
    df = tsv.read_tsv(tsv_gz_file)
    assert len(df.SampleID.unique()) == 411
    assert len(df.columns) == 7
