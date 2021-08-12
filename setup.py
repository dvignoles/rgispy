from setuptools import find_packages, setup

setup(
    name="rgispy",
    version="0.1",
    description="WBM model output processing",
    url="https://github.com/dvignoles/rgispy",
    author="Daniel Vignoles",
    author_email="dvignoles@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "xarray", "netcdf4"],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "sampleDs=rgispy.cmd.sampleds:main",
        ],
    },
)
