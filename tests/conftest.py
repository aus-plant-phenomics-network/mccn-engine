import json
from pathlib import Path
from typing import cast

import pystac
import pytest
from odc.geo.geobox import GeoBox
from stac_generator import StacGeneratorFactory
from stac_generator.core import StacCollectionConfig

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


@pytest.fixture(scope="module")
def dsm_collection() -> pystac.Collection:
    return load_collection(RASTER_FIXTURE_PATH / "dsm_config.json")


@pytest.fixture(scope="module")
def multibands_collection() -> pystac.Collection:
    return load_collection(RASTER_FIXTURE_PATH / "multibands_config.json")


@pytest.fixture(scope="module")
def area_collection() -> pystac.Collection:
    return load_collection(VECTOR_FIXTURE_PATH / "area_config.json")


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
