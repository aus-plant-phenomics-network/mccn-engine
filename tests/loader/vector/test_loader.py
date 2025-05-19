import pickle
from pathlib import Path
from typing import Generator, cast

import numpy as np
import pandas as pd
import pystac
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from mccn._types import TimeGroupby
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
    targets = [k for k in ds.attrs[MASK_NAME].values() if k != "nodata"]
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


def compare_matching_legend(ds: xr.Dataset, ref_cube: xr.Dataset) -> None:
    for var in ref_cube.data_vars:
        for time in ref_cube["time"]:
            ds_values = ds.sel(time=time)[var].values
            ref_values = np.copy(ref_cube.sel(time=time)[var].values)
            if var in ds.attrs:
                ds_map = {v: k for k, v in ds.attrs[var].items()}
                ref_map = {v: k for k, v in ref_cube.attrs[var].items()}
                legend = {v1: ds_map[k1] for k1, v1 in ref_map.items() if k1 in ds_map}
                masks = {v2: ref_values == v1 for v1, v2 in legend.items()}
                for k, m in masks.items():
                    ref_values[m] = k
            # Compare on the same mask values
            np.testing.assert_almost_equal(
                ds_values[ref_values != 0], ref_values[ref_values != 0]
            )


@pytest.fixture(scope="module")
def point_cook_mask(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
) -> xr.Dataset:
    client = MCCN(
        items=[item for item in area_items if item.id == "point_cook_mask"],
        geobox=area_geobox,
        mask_name=MASK_NAME,
        combine_mask=True,
    )
    return client.load()


@pytest.fixture(scope="module")
def hoppers_crossing_name(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
) -> xr.Dataset:
    client = MCCN(
        items=[item for item in area_items if item.id == "hoppers_crossing_name"],
        geobox=area_geobox,
        mask_name=MASK_NAME,
        combine_mask=True,
    )
    return client.load()


@pytest.fixture(scope="module")
def werribee_crime(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
) -> xr.Dataset:
    client = MCCN(
        items=[item for item in area_items if item.id == "werribee_crime"],
        geobox=area_geobox,
        mask_name=MASK_NAME,
        combine_mask=True,
    )
    return client.load()


@pytest.fixture(scope="module")
def sunbury_crime(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
) -> xr.Dataset:
    client = MCCN(
        items=[item for item in area_items if item.id == "sunbury_crime"],
        geobox=area_geobox,
        mask_name=MASK_NAME,
        combine_mask=True,
    )
    return client.load()


@pytest.fixture(scope="module")
def sunbury_population(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
) -> xr.Dataset:
    client = MCCN(
        items=[item for item in area_items if item.id == "sunbury_population"],
        geobox=area_geobox,
        mask_name=MASK_NAME,
        combine_mask=True,
    )
    return client.load()


@pytest.fixture(scope="module")
def point_cook_fx() -> Generator[xr.Dataset, None, None]:
    with open(
        "tests/loader/vector/fixture/reference_cube/point_cook_mask.pkl", "rb"
    ) as file:
        yield pickle.load(file)


@pytest.fixture(scope="module")
def point_cook_global_fx() -> Generator[xr.Dataset, None, None]:
    with open(
        "tests/loader/vector/fixture/reference_cube/point_cook_mask_global.pkl", "rb"
    ) as file:
        yield pickle.load(file)


@pytest.fixture(scope="module")
def hoppers_crossing_fx() -> Generator[xr.Dataset, None, None]:
    with open(
        "tests/loader/vector/fixture/reference_cube/hoppers_crossing_name.pkl", "rb"
    ) as file:
        yield pickle.load(file)


@pytest.fixture(scope="module")
def hoppers_crossing_global_fx() -> Generator[xr.Dataset, None, None]:
    with open(
        "tests/loader/vector/fixture/reference_cube/hoppers_crossing_name_global.pkl",
        "rb",
    ) as file:
        yield pickle.load(file)


@pytest.fixture(scope="module")
def werribee_crime_fx() -> Generator[xr.Dataset, None, None]:
    with open(
        "tests/loader/vector/fixture/reference_cube/werribee_crime.pkl", "rb"
    ) as file:
        yield pickle.load(file)


@pytest.fixture(scope="module")
def werribee_crime_global_fx() -> Generator[xr.Dataset, None, None]:
    with open(
        "tests/loader/vector/fixture/reference_cube/werribee_crime_global.pkl", "rb"
    ) as file:
        yield pickle.load(file)


@pytest.fixture(scope="module")
def sunbury_crime_fx() -> Generator[xr.Dataset, None, None]:
    with open(
        "tests/loader/vector/fixture/reference_cube/sunbury_crime.pkl", "rb"
    ) as file:
        yield pickle.load(file)


@pytest.fixture(scope="module")
def sunbury_crime_global_fx() -> Generator[xr.Dataset, None, None]:
    with open(
        "tests/loader/vector/fixture/reference_cube/sunbury_crime_global.pkl", "rb"
    ) as file:
        yield pickle.load(file)


@pytest.fixture(scope="module")
def sunbury_population_fx() -> Generator[xr.Dataset, None, None]:
    with open(
        "tests/loader/vector/fixture/reference_cube/sunbury_population.pkl", "rb"
    ) as file:
        yield pickle.load(file)


@pytest.fixture(scope="module")
def sunbury_population_global_fx() -> Generator[xr.Dataset, None, None]:
    with open(
        "tests/loader/vector/fixture/reference_cube/sunbury_population_global.pkl", "rb"
    ) as file:
        yield pickle.load(file)


@pytest.fixture(scope="module")
def global_fx() -> Generator[xr.Dataset, None, None]:
    with open("tests/loader/vector/fixture/reference_cube/global.pkl", "rb") as file:
        yield pickle.load(file)


# Test load combined mask
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
        combine_mask=True,
    )
    ds = client.load()
    assert len(ds.data_vars) == 1
    assert MASK_NAME in ds.data_vars
    map_targets = {k for k in ds.attrs[MASK_NAME].values() if k != "nodata"}
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
        combine_mask=True,
    )
    ds = client.load()
    assert len(ds.attrs[MASK_NAME]) == 6
    verify_mask_value(ds, request)
    for field, target in exp.items():
        verify_layer_value(ds, request, field, target)


# Test vector load against reference asset - mask only
def test_load_point_cook(
    area_collection: pystac.Collection,
    point_cook_geobox: GeoBox,
    point_cook_fx: xr.Dataset,
) -> None:
    client = MCCN(
        items=[cast(pystac.Item, area_collection.get_item("point_cook_mask"))],
        geobox=point_cook_geobox,
    )
    ds = client.load()
    xr.testing.assert_equal(ds, point_cook_fx)


# Test vector load against reference asset - mask + vector attributes
def test_load_hoppers_crossing(
    area_collection: pystac.Collection,
    hoppers_crossing_geobox: GeoBox,
    hoppers_crossing_fx: xr.Dataset,
) -> None:
    client = MCCN(
        items=[cast(pystac.Item, area_collection.get_item("hoppers_crossing_name"))],
        geobox=hoppers_crossing_geobox,
    )
    ds = client.load()
    xr.testing.assert_equal(ds, hoppers_crossing_fx)


# Test vector load against reference asset - mask + vector attribute + join asset
def test_load_werribee_crime(
    area_collection: pystac.Collection,
    werribee_geobox: GeoBox,
    werribee_crime_fx: xr.Dataset,
) -> None:
    client = MCCN(
        items=[cast(pystac.Item, area_collection.get_item("werribee_crime"))],
        geobox=werribee_geobox,
    )
    ds = client.load()
    xr.testing.assert_equal(ds, werribee_crime_fx)


def test_load_sunbury_crime(
    area_collection: pystac.Collection,
    sunbury_geobox: GeoBox,
    sunbury_crime_fx: xr.Dataset,
) -> None:
    client = MCCN(
        items=[cast(pystac.Item, area_collection.get_item("sunbury_crime"))],
        geobox=sunbury_geobox,
    )
    ds = client.load()
    xr.testing.assert_equal(ds, sunbury_crime_fx)


def test_load_sunbury_population(
    area_collection: pystac.Collection,
    sunbury_geobox: GeoBox,
    sunbury_population_fx: xr.Dataset,
) -> None:
    client = MCCN(
        items=[cast(pystac.Item, area_collection.get_item("sunbury_population"))],
        geobox=sunbury_geobox,
    )
    ds = client.load()
    xr.testing.assert_equal(ds, sunbury_population_fx)


def test_load_global_fx(
    area_collection: pystac.Collection, area_geobox: GeoBox, global_fx: xr.Dataset
) -> None:
    client = MCCN(collection=area_collection, geobox=area_geobox)
    ds = client.load()
    xr.testing.assert_equal(ds, global_fx)


# Test filter by dates
# Set filtering dates to constrain to specific areas, then compare the resulting dataset and the referenced dataset of the area
@pytest.mark.parametrize(
    "start, end, ref_cubes, exp_time",
    [
        # Filter None, None - load everything
        (
            None,
            None,
            [
                "point_cook_global_fx",
                "hoppers_crossing_global_fx",
                "werribee_crime_global_fx",
                "sunbury_crime_global_fx",
                "sunbury_population_global_fx",
            ],
            [
                "2016-10-01T00:00:00",
                "2016-10-02T00:00:00",
                "2017-10-01T00:00:00",
                "2017-10-02T00:00:00",
                "2021-01-01T00:00:00",
                "2026-01-01T00:00:00",
            ],
        ),
        # Filter date cover whole range - load everything
        (
            "2013-01-01",
            "2030-01-01",
            [
                "point_cook_global_fx",
                "hoppers_crossing_global_fx",
                "werribee_crime_global_fx",
                "sunbury_crime_global_fx",
                "sunbury_population_global_fx",
            ],
            [
                "2016-10-01T00:00:00",
                "2016-10-02T00:00:00",
                "2017-10-01T00:00:00",
                "2017-10-02T00:00:00",
                "2021-01-01T00:00:00",
                "2026-01-01T00:00:00",
            ],
        ),
        # Filter only point cook
        (
            None,
            "2016-10-01T12:00:00Z",
            ["point_cook_global_fx"],
            [
                "2016-10-01T00:00:00",
            ],
        ),
        # Filter only hoppers crossing
        (
            "2016-10-01T12:00:00Z",
            "2016-10-02T12:00:00Z",
            ["hoppers_crossing_global_fx"],
            [
                "2016-10-02T00:00:00",
            ],
        ),
        # Filter both point cook and hoppers crossing
        (
            None,
            "2016-10-02T12:00:00Z",
            ["point_cook_global_fx", "hoppers_crossing_global_fx"],
            [
                "2016-10-01T00:00:00",
                "2016-10-02T00:00:00",
            ],
        ),
    ],
)
def test_load_cube_by_date_positive_test(
    start: str | None,
    end: str | None,
    ref_cubes: list[str],
    exp_time: list[str],
    area_collection: pystac.Collection,
    area_geobox: GeoBox,
    request: pytest.FixtureRequest,
) -> None:
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    client = MCCN(
        collection=area_collection, geobox=area_geobox, start_ts=start_ts, end_ts=end_ts
    )
    ds = client.load()
    ts = pd.to_datetime(exp_time)
    assert np.all(ts == ds["time"].values)
    for ref_cube_fx in ref_cubes:
        ref_cube = request.getfixturevalue(ref_cube_fx)
        compare_matching_legend(ds, ref_cube)


@pytest.mark.parametrize(
    "start, end, exp_time",
    [
        (None, None, ["2021-01-01T00:00", "2026-01-01T00:00"]),
        (
            "2021-01-01T00:00",
            "2026-01-01T00:00",
            ["2021-01-01T00:00", "2026-01-01T00:00"],
        ),
        (None, "2015-01-01", []),
        ("2025-01-01", None, ["2026-01-01T00:00"]),
    ],
)
def test_filter_timeseries(
    start: str | None,
    end: str | None,
    exp_time: list[str],
    area_geobox: GeoBox,
    area_collection: pystac.Collection,
) -> None:
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    client = MCCN(
        items=[
            item
            for item in area_collection.get_items(recursive=True)
            if item.id == "sunbury_population"
        ],
        start_ts=start_ts,
        end_ts=end_ts,
        geobox=area_geobox,
    )
    ds = client.load()
    ts = pd.to_datetime(exp_time)
    assert np.all(ts == ds["time"].values)


# Test filter by location
# Set the filtering geobox to constrain to specific areas, then compare the resulting dataset and the referenced dataset of the area
@pytest.mark.parametrize(
    "geobox_fx, cube_fx",
    [
        ("point_cook_geobox", "point_cook_fx"),
        ("hoppers_crossing_geobox", "hoppers_crossing_fx"),
        ("werribee_geobox", "werribee_crime_fx"),
        ("sunbury_geobox", "sunbury_crime_fx"),
        ("sunbury_geobox", "sunbury_population_fx"),
    ],
)
def test_load_cube_by_area(
    area_collection: pystac.Collection,
    geobox_fx: str,
    cube_fx: str,
    request: pytest.FixtureRequest,
) -> None:
    geobox = request.getfixturevalue(geobox_fx)
    client = MCCN(collection=area_collection, geobox=geobox)
    ds = client.load()
    ref_cube = request.getfixturevalue(cube_fx)
    compare_matching_legend(ds, ref_cube)


# Test filter by band name
# By default (use_all_vectors = True), even shape files with non matching bands will be loaded
# Vectors with matching bands will load geometry + bands
# Vectors with non matching band will load geometry only
@pytest.mark.parametrize(
    "bands, exp_bands",
    [
        # Bands = None - load everything
        (
            None,
            {
                "area_sqkm",
                "lga_name",
                "hoppers_crossing_name",
                "werribee_crime",
                "sunbury_crime",
                "sunbury_population",
                "crime_incidents",
                "crime_rate",
                "name",
                "point_cook_mask",
                "population",
            },
        ),
        # Band = name - load all geometries + name layer
        (
            {"name"},
            {
                "name",
                "point_cook_mask",
                "hoppers_crossing_name",
                "werribee_crime",
                "sunbury_crime",
                "sunbury_population",
            },
        ),
        # Band = name + area_sqkm - load all geometries + name + area_sqkm
        (
            {"name", "area_sqkm"},
            {
                "name",
                "area_sqkm",
                "point_cook_mask",
                "hoppers_crossing_name",
                "werribee_crime",
                "sunbury_crime",
                "sunbury_population",
            },
        ),
        # Non matching band - load all geometries + matching bands
        (
            {"name", "area_sqkm", "red"},
            {
                "name",
                "area_sqkm",
                "point_cook_mask",
                "hoppers_crossing_name",
                "werribee_crime",
                "sunbury_crime",
                "sunbury_population",
            },
        ),
    ],
)
def test_filter_bands_use_all_vectors_read_geometry_layers(
    bands: None | set[str],
    area_geobox: GeoBox,
    area_collection: pystac.Collection,
    exp_bands: set[str],
) -> None:
    client = MCCN(
        collection=area_collection,
        geobox=area_geobox,
        bands=bands,
    )
    ds = client.load()
    assert set(ds.data_vars.keys()) == exp_bands


@pytest.mark.parametrize(
    "bands, exp_bands",
    [
        # None - load everything
        (
            None,
            {
                "area_sqkm",
                "lga_name",
                "hoppers_crossing_name",
                "werribee_crime",
                "sunbury_crime",
                "sunbury_population",
                "crime_incidents",
                "crime_rate",
                "name",
                "point_cook_mask",
                "population",
            },
        ),
        # Name - point cook mask not loaded
        (
            {"name"},
            {
                "name",
                "hoppers_crossing_name",
                "werribee_crime",
                "sunbury_crime",
                "sunbury_population",
            },
        ),
        # Name and area - point cook not loaded
        (
            {"name", "area_sqkm"},
            {
                "name",
                "area_sqkm",
                "hoppers_crossing_name",
                "werribee_crime",
                "sunbury_crime",
                "sunbury_population",
            },
        ),
        # Population - only sunbury population loaded
        # Note here name is also loaded since it's the join column left right
        (
            {"population"},
            {
                "name",  # Join column
                "population",
                "sunbury_population",
            },
        ),
        # Population + name - everything except point cook mask
        # Note here name is also loaded since it's the join column left right
        (
            {"population", "name"},
            {
                "name",  # Join column
                "population",
                "hoppers_crossing_name",
                "werribee_crime",
                "sunbury_crime",
                "sunbury_population",
            },
        ),
        # Non matching band
        (
            {"name", "area_sqkm", "red"},
            {
                "name",
                "area_sqkm",
                "hoppers_crossing_name",
                "werribee_crime",
                "sunbury_crime",
                "sunbury_population",
            },
        ),
    ],
)
def test_filter_bands_not_read_all_vectors_read_matching_layers(
    bands: None | set[str],
    area_geobox: GeoBox,
    area_collection: pystac.Collection,
    exp_bands: set[str],
) -> None:
    client = MCCN(
        collection=area_collection,
        geobox=area_geobox,
        bands=bands,
        use_all_vectors=False,
    )
    ds = client.load()
    assert set(ds.data_vars.keys()) == exp_bands


# Test groupby
@pytest.mark.parametrize(
    "groupby, exp_time",
    [
        (
            "day",
            [
                "2016-10-01T00:00:00",
                "2016-10-02T00:00:00",
                "2017-10-01T00:00:00",
                "2017-10-02T00:00:00",
                "2021-01-01T00:00:00",
                "2026-01-01T00:00:00",
            ],
        ),
        (
            "month",
            [
                "2016-10-01T00:00:00",
                "2017-10-01T00:00:00",
                "2021-01-01T00:00:00",
                "2026-01-01T00:00:00",
            ],
        ),
        (
            "year",
            [
                "2016-01-01T00:00:00",
                "2017-01-01T00:00:00",
                "2021-01-01T00:00:00",
                "2026-01-01T00:00:00",
            ],
        ),
    ],
)
def test_groupby(
    groupby: TimeGroupby,
    exp_time: list[str],
    area_geobox: GeoBox,
    area_collection: pystac.Collection,
) -> None:
    client = MCCN(collection=area_collection, geobox=area_geobox, time_groupby=groupby)
    ds = client.load()
    ts = pd.to_datetime(exp_time)
    assert np.all(ts == ds["time"].values)


# Test set dtype
@pytest.mark.parametrize("dtype", ["int32", "float32", "float64"])
def test_set_dtype(
    area_geobox: GeoBox, area_collection: pystac.Collection, dtype: str
) -> None:
    client = MCCN(
        collection=area_collection, geobox=area_geobox, dtype={"population": dtype}
    )
    ds = client.load()
    assert ds["population"].dtype == dtype


# Test serialisation
def test_serialise(
    area_collection: pystac.Collection, area_geobox: GeoBox, tmp_path: Path
) -> None:
    client = MCCN(collection=area_collection, geobox=area_geobox)
    ds = client.load()
    path = tmp_path / "area.cd"
    client.to_netcdf(ds, path)
    ref_ds = client.from_netcdf(path)
    xr.testing.assert_equal(ds, ref_ds)
