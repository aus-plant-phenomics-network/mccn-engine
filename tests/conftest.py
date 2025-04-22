import json
from pathlib import Path
from typing import cast

import pystac
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from stac_generator.core import StacCollectionConfig
from stac_generator.factory import StacGeneratorFactory

from mccn.extent import GeoBoxBuilder
from tests.utils import RASTER_FIXTURE_PATH, VECTOR_FIXTURE_PATH


def load_collection(path: Path | str) -> pystac.Collection:
    with Path(path).open("r") as file:
        config = json.load(file)
    for i in range(len(config)):
        config[i]["location"] = Path(config[i]["location"]).absolute().as_uri()
    factory = StacGeneratorFactory.get_stac_generator(
        config, StacCollectionConfig(id="Collection")
    )
    return factory.create_collection()


# RASTER FIXTURES


@pytest.fixture(scope="module")
def year_dsm() -> xr.Dataset:
    return xr.open_dataset(RASTER_FIXTURE_PATH / "year_dsm.cd")


@pytest.fixture(scope="module")
def month_dsm() -> xr.Dataset:
    return xr.open_dataset(RASTER_FIXTURE_PATH / "month_dsm.cd")


@pytest.fixture(scope="module")
def day_dsm() -> xr.Dataset:
    return xr.open_dataset(RASTER_FIXTURE_PATH / "day_dsm.cd")


@pytest.fixture(scope="module")
def hour_dsm() -> xr.Dataset:
    return xr.open_dataset(RASTER_FIXTURE_PATH / "hour_dsm.cd")


@pytest.fixture(scope="module")
def dsm_collection() -> pystac.Collection:
    return load_collection(RASTER_FIXTURE_PATH / "dsm_config.json")


@pytest.fixture(scope="module")
def dsm_items(dsm_collection: pystac.Collection) -> list[pystac.Item]:
    return list(dsm_collection.get_items(recursive=True))


@pytest.fixture(scope="module")
def multibands_collection() -> pystac.Collection:
    return load_collection(RASTER_FIXTURE_PATH / "multibands_config.json")


@pytest.fixture(scope="module")
def multibands_items(multibands_collection: pystac.Collection) -> list[pystac.Item]:
    return list(multibands_collection.get_items(recursive=True))


# AREA FIXTURES


@pytest.fixture(scope="module")
def area_collection() -> pystac.Collection:
    return load_collection(VECTOR_FIXTURE_PATH / "area_config.json")


@pytest.fixture(scope="module")
def area_items(area_collection: pystac.Collection) -> list[pystac.Item]:
    return list(area_collection.get_items(recursive=True))


@pytest.fixture(scope="module")
def dsm_geobox(dsm_collection: pystac.Collection) -> GeoBox:
    return GeoBoxBuilder.from_collection(dsm_collection, (100, 100))


@pytest.fixture(scope="module")
def multiband_geobox(multibands_collection: pystac.Collection) -> GeoBox:
    return GeoBoxBuilder.from_collection(multibands_collection, (100, 100))


@pytest.fixture(scope="module")
def dsm_bottom_right_geobox(dsm_collection: pystac.Collection) -> GeoBox:
    return GeoBoxBuilder.from_item(
        cast(pystac.Item, dsm_collection.get_item("bottom-right-16")), (100, 100)
    )


@pytest.fixture(scope="module")
def dsm_top_right_geobox(dsm_collection: pystac.Collection) -> GeoBox:
    return GeoBoxBuilder.from_item(
        cast(pystac.Item, dsm_collection.get_item("top-right-16")), (100, 100)
    )


@pytest.fixture(scope="module")
def area_geobox(area_collection: pystac.Collection) -> GeoBox:
    return GeoBoxBuilder.from_collection(area_collection, (100, 100))
