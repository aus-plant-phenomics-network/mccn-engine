# mccn-engine

MCCN-Engine is a python library for loading and combining STAC described asset, generated using the [stac_generator](https://aus-plant-phenomics-network.github.io/stac-generator/), into an [xarray](https://docs.xarray.dev/en/stable/) datacube.

## Installation

Install from PyPi:

```bash
pip install mccn-engine
```

## For developers:

The MCCN-Engine repository uses `pdm` for dependency management. Please [install](https://pdm-project.org/en/latest/#installation) pdm before running the comands below.

Installing dependencies:

```bash
pdm install
```

Run tests:

```bash
make test
```

Lint:

```bash
make lint
```
