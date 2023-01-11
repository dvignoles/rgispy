# rgispy

## Installation
python >= 3.9 required

[RGIS](https://github.com/bmfekete/RGIS) required

```sh
git clone https://github.com/bmfekete/RGIS /tmp/RGIS \
 && /tmp/RGIS/install.sh /usr/local/share \
 && rm -rf /tmp/RGIS
```

The easiest way to get going is with conda to circumvent any gdal dependency issues. 

```sh

create -n rgispy python=3.10 gdal

pip install git+https://github.com/dvignoles/rgispy@main
```

## Developer Setup
To develop this code base further, first set up installation as described above. Then install the developer dependencies which enfore code quality and formatting. 

```sh
# Install rgispy in edittable mode
pip install -e .
# Install dependencies
pip install -r requirements.dev
# Setup pre-commit and pre-push hooks
pre-commit install -t pre-commit
pre-commit install -t pre-push
```

## Credits
This package was created with Cookiecutter and the [sourcery-ai/python-best-practices-cookiecutter](https://github.com/sourcery-ai/python-best-practices-cookiecutter) project template.
