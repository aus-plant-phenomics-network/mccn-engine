import pystac
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from mccn.client import MCCN

MASK_NAME = "MASK"

ID_MASK_MAP = {
    "point_cook_mask": [1],
    "hoppers_crossing_name": [2],
    "werribee_crime": [3],
    "sunbury_crime": [4, 5],
    "sunbury_population": [4, 5],
}


def verify_mask_value(
    ds: xr.Dataset,
    request: pytest.FixtureRequest,
) -> None:
    targets = ds.attrs[MASK_NAME].values()
    for target in targets:
        ref_ds: xr.Dataset = request.getfixturevalue(target)
        for date in ref_ds["time"].values:  # type: ignore[operator]
            mask = ref_ds.sel(time=date, method="nearest")[MASK_NAME] > 0
            data = ds.sel(time=date, method="nearest")[MASK_NAME].where(mask, drop=True)
            assert data.mean() in ID_MASK_MAP[target]


def verify_layer_value(
    ds: xr.Dataset,
    request: pytest.FixtureRequest,
    field: str,
    targets: set[str],
) -> None:
    for target in targets:
        ref_ds: xr.Dataset = request.getfixturevalue(target)
        for date in ref_ds["time"].values:  # type: ignore[operator]
            mask = ref_ds.sel(time=date, method="nearest")[field] != 0
            data = ds.sel(time=date, method="nearest")[field].where(mask, drop=True)
            ref_data = ds.sel(time=date, method="nearest")[field].where(mask, drop=True)
            assert (data.fillna(0) == ref_data.fillna(0)).all()


@pytest.fixture(scope="module")
def point_cook_mask(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
) -> xr.Dataset:
    client = MCCN(
        items=[item for item in area_items if item.id == "point_cook_mask"],
        geobox=area_geobox,
        mask_name=MASK_NAME,
    )
    return client.load_vector()


@pytest.fixture(scope="module")
def hoppers_crossing_name(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
) -> xr.Dataset:
    client = MCCN(
        items=[item for item in area_items if item.id == "hoppers_crossing_name"],
        geobox=area_geobox,
        mask_name=MASK_NAME,
    )
    return client.load_vector()


@pytest.fixture(scope="module")
def werribee_crime(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
) -> xr.Dataset:
    client = MCCN(
        items=[item for item in area_items if item.id == "werribee_crime"],
        geobox=area_geobox,
        mask_name=MASK_NAME,
    )
    return client.load_vector()


@pytest.fixture(scope="module")
def sunbury_crime(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
) -> xr.Dataset:
    client = MCCN(
        items=[item for item in area_items if item.id == "sunbury_crime"],
        geobox=area_geobox,
        mask_name=MASK_NAME,
    )
    return client.load_vector()


@pytest.fixture(scope="module")
def sunbury_population(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
) -> xr.Dataset:
    client = MCCN(
        items=[item for item in area_items if item.id == "sunbury_population"],
        geobox=area_geobox,
        mask_name=MASK_NAME,
    )
    return client.load_vector()


@pytest.mark.parametrize(
    "bands, use_all_vectors",
    [
        (None, True),
        (None, False),
        ({"name"}, True),
        ({"name"}, False),
        ({"name", "lga_name"}, True),
        ({"name", "lga_name"}, False),
        ({"non_matching"}, True),
        ({"non_matching"}, False),
        ({"name", "non_matching"}, True),
        ({"name", "non_matching"}, False),
    ],
    ids=[
        "None-True",
        "None-False",
        "name-True",
        "name-False",
        "name+lga_name-True",
        "name+lga_name-False",
        "non_matching-True",
        "non_matching-False",
        "name+non_matching-True",
        "name+non_matching-False",
    ],
)
def test_given_mask_only_load_only_mask(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
    bands: set[str] | None,
    use_all_vectors: bool,
    request: pytest.FixtureRequest,
) -> None:
    client = MCCN(
        items=area_items,
        geobox=area_geobox,
        mask_only=True,
        mask_name=MASK_NAME,
        bands=bands,
        use_all_vectors=use_all_vectors,
    )
    ds = client.load_vector()
    assert len(ds.data_vars) == 1
    assert MASK_NAME in ds.data_vars
    map_targets = set(ds.attrs[MASK_NAME].values())
    items = {item.id for item in area_items}
    assert map_targets == items
    verify_mask_value(ds, request)


@pytest.mark.parametrize(
    "bands, exp",
    [
        (
            None,
            {
                "lga_name": {"werribee_crime", "sunbury_crime"},
                "crime_incidents": {"werribee_crime", "sunbury_crime"},
                "crime_rate": {"werribee_crime", "sunbury_crime"},
                "population": {"sunbury_population"},
            },
        ),
        (
            {"population"},
            {
                "population": {"sunbury_population"},
            },
        ),
        (
            {"population", "lga_name"},
            {
                "lga_name": {"werribee_crime", "sunbury_crime"},
                "population": {"sunbury_population"},
            },
        ),
        (
            {"population", "lga_name", "non_matching"},
            {
                "lga_name": {"werribee_crime", "sunbury_crime"},
                "population": {"sunbury_population"},
            },
        ),
        (
            {"population", "lga_name", "crime_rate", "crime_incidents"},
            {
                "lga_name": {"werribee_crime", "sunbury_crime"},
                "crime_incidents": {"werribee_crime", "sunbury_crime"},
                "crime_rate": {"werribee_crime", "sunbury_crime"},
                "population": {"sunbury_population"},
            },
        ),
    ],
    ids=[
        "None",
        "population",
        "population+lga_name",
        "population+lga_name+non_matching",
        "population+lga_name+crime_rate+crime_incidents",
    ],
)
def test_given_bands_and_use_all_vectors_TRUE_load_masks_and_matched_layers(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
    bands: set[str] | None,
    exp: dict[str, set[str]],
    request: pytest.FixtureRequest,
) -> None:
    client = MCCN(
        items=area_items,
        geobox=area_geobox,
        mask_only=False,
        mask_name=MASK_NAME,
        bands=bands,
        use_all_vectors=True,
    )
    ds = client.load_vector()
    assert len(ds.attrs[MASK_NAME]) == 5
    verify_mask_value(ds, request)
    for field, target in exp.items():
        verify_layer_value(ds, request, field, target)
