import pytest

from mccn.extent import GeoBoxBuilder


@pytest.mark.parametrize(
    "collection, items",
    [
        ("dsm_collection", "dsm_items"),
        ("multibands_collection", "multibands_items"),
        ("area_collection", "area_items"),
    ],
)
def test_expects_collection_geobox_and_all_item_geobox_to_be_the_same(
    collection: str,
    items: str,
    request: pytest.FixtureRequest,
) -> None:
    collection_fx = request.getfixturevalue(collection)
    items_fx = request.getfixturevalue(items)
    collection_geobox = GeoBoxBuilder.from_collection(collection_fx, 100)
    item_geobox = GeoBoxBuilder.from_items(items_fx, 100)
    assert collection_geobox == item_geobox
