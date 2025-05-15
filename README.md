# MCCN-Engine

<!-- markdownlint-disable -->
<p align="center">
  <!-- github-banner-start -->
  <!-- github-banner-end -->
</p>
<!-- markdownlint-restore -->

<div align="center">

<!-- prettier-ignore-start -->

| Project |     | Status|
|---------|:----|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CI/CD   |     | [![CI](https://github.com/aus-plant-phenomics-network/mccn-engine/actions/workflows/github-actions.yml/badge.svg)](https://github.com/aus-plant-phenomics-network/mccn-engine/actions/workflows/github-actions.yml) [![documentation](https://github.com/aus-plant-phenomics-network/mccn-engine/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/aus-plant-phenomics-network/mccn-engine/actions/workflows/pages/pages-build-deployment) |

<!-- prettier-ignore-end -->
</div>

<hr>

Check out the [documentation 📚](https://aus-plant-phenomics-network.github.io/mccn-engine/)

Also check out the [case studies](.https://github.com/aus-plant-phenomics-network/mccn-case-studies) notebooks.

## About


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
