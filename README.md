# rgispy

## Installation
python 3.9 required

The easiest way to get going is with conda. Nobody wants to fuss about with GDAL. 

```sh

create -n rgispy numpy pandas GDAL xarray rasterio rioxarray netcdf4 geopandas sqlalchemy geoalchemy2 psycopg2

pip install git+git://github.com/dvignoles/rgispy@main
```

## Developer Setup
```sh
# Setup up base environment like above

pip install -e .
# Install dependencies
pip install -r requirements.dev

# Setup pre-commit and pre-push hooks
pre-commit install -t pre-commit
pre-commit install -t pre-push
```

## Credits
This package was created with Cookiecutter and the [sourcery-ai/python-best-practices-cookiecutter](https://github.com/sourcery-ai/python-best-practices-cookiecutter) project template.
