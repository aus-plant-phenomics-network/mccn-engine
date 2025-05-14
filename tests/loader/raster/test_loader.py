from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
import xarray as xr

from mccn.client import MCCN
from tests.utils import RASTER_FIXTURE_PATH

if TYPE_CHECKING:
    from typing import Callable

    import pystac
    from odc.geo.geobox import GeoBox

    from mccn._types import TimeGroupby

X_COORD, Y_COORD, T_COORD = "X", "Y", "T"


def test_cube_axis_renamed(
    dsm_collection: pystac.Collection, dsm_geobox: GeoBox
) -> None:
    engine = MCCN(
        collection=dsm_collection,
        geobox=dsm_geobox,
        x_dim=X_COORD,
        y_dim=Y_COORD,
        t_dim=T_COORD,
    )
    ds = engine.load()
    assert X_COORD in ds.dims
    assert Y_COORD in ds.dims
    assert T_COORD in ds.dims
    assert len(ds.dims) == 3


@pytest.fixture(scope="module")
def year_dsm_loaded(
    dsm_collection: pystac.Collection,
    dsm_geobox: GeoBox,
) -> xr.Dataset:
    engine = MCCN(
        collection=dsm_collection,
        geobox=dsm_geobox,
        time_groupby="year",
        t_dim=T_COORD,
    )
    return engine.load()


@pytest.fixture(scope="module")
def month_dsm_loaded(
    dsm_collection: pystac.Collection,
    dsm_geobox: GeoBox,
) -> xr.Dataset:
    engine = MCCN(
        collection=dsm_collection,
        geobox=dsm_geobox,
        time_groupby="month",
        t_dim=T_COORD,
    )
    return engine.load()


@pytest.fixture(scope="module")
def day_dsm_loaded(
    dsm_collection: pystac.Collection,
    dsm_geobox: GeoBox,
) -> xr.Dataset:
    engine = MCCN(
        collection=dsm_collection,
        geobox=dsm_geobox,
        time_groupby="day",
        t_dim=T_COORD,
    )
    return engine.load()


@pytest.fixture(scope="module")
def hour_dsm_loaded(
    dsm_collection: pystac.Collection,
    dsm_geobox: GeoBox,
) -> xr.Dataset:
    engine = MCCN(
        collection=dsm_collection,
        geobox=dsm_geobox,
        time_groupby="hour",
        t_dim=T_COORD,
    )
    return engine.load()


@pytest.fixture(scope="module")
def top_left_dsm_loaded(
    multibands_collection: pystac.Collection, multiband_geobox: GeoBox
) -> xr.Dataset:
    client = MCCN(
        items=[
            item
            for item in multibands_collection.get_items(recursive=True)
            if item.id == "dsm"
        ],
        geobox=multiband_geobox,
    )
    return client.load()


@pytest.fixture(scope="module")
def top_left_rgb_loaded(
    multibands_collection: pystac.Collection, multiband_geobox: GeoBox
) -> xr.Dataset:
    client = MCCN(
        items=[
            item
            for item in multibands_collection.get_items(recursive=True)
            if item.id == "rgb"
        ],
        geobox=multiband_geobox,
    )
    return client.load()


@pytest.fixture(scope="module")
def top_left_ms_loaded(
    multibands_collection: pystac.Collection, multiband_geobox: GeoBox
) -> xr.Dataset:
    client = MCCN(
        items=[
            item
            for item in multibands_collection.get_items(recursive=True)
            if item.id == "rgb-alias"
        ],
        geobox=multiband_geobox,
    )
    return client.load()


@pytest.fixture(scope="module")
def multibands_ds() -> xr.Dataset:
    return xr.open_dataset(RASTER_FIXTURE_PATH / "reference_cube/multibands.cd")


# Groupby Feature testing
@pytest.mark.parametrize(
    "groupby,exp_ts",
    [
        (
            "year",
            [
                pd.Timestamp("2015-01-01T00:00:00"),
                pd.Timestamp("2016-01-01T00:00:00"),
            ],
        ),
        (
            "month",
            [
                pd.Timestamp("2015-10-01T00:00:00"),
                pd.Timestamp("2015-11-01T00:00:00"),
                pd.Timestamp("2016-10-01T00:00:00"),
                pd.Timestamp("2016-11-01T00:00:00"),
            ],
        ),
        (
            "day",
            [
                pd.Timestamp("2015-10-01T00:00:00"),
                pd.Timestamp("2015-10-02T00:00:00"),
                pd.Timestamp("2015-11-01T00:00:00"),
                pd.Timestamp("2015-11-02T00:00:00"),
                pd.Timestamp("2016-10-01T00:00:00"),
                pd.Timestamp("2016-10-02T00:00:00"),
                pd.Timestamp("2016-11-01T00:00:00"),
                pd.Timestamp("2016-11-02T00:00:00"),
            ],
        ),
        (
            "hour",
            [
                pd.Timestamp("2015-10-01T12:00:00"),
                pd.Timestamp("2015-10-02T12:00:00"),
                pd.Timestamp("2015-11-01T10:00:00"),
                pd.Timestamp("2015-11-02T10:00:00"),
                pd.Timestamp("2016-10-01T12:00:00"),
                pd.Timestamp("2016-10-02T12:00:00"),
                pd.Timestamp("2016-11-01T10:00:00"),
                pd.Timestamp("2016-11-02T10:00:00"),
            ],
        ),
    ],
    ids=["year", "month", "day", "hour"],
)
def test_raster_groupby(
    groupby: TimeGroupby,
    exp_ts: list[pd.Timestamp],
    request: pytest.FixtureRequest,
) -> None:
    ds = request.getfixturevalue(f"{groupby}_dsm_loaded")
    # Verify dates
    assert len(ds[T_COORD]) == len(exp_ts)  # 2 Years - 2015 and 2016
    timestamps = pd.DatetimeIndex(ds[T_COORD].values)
    assert all(timestamps == exp_ts)
    # Compare against ref ds
    ref_ds = request.getfixturevalue(f"{groupby}_dsm")
    xr.testing.assert_equal(ds, ref_ds)


@pytest.mark.parametrize(
    "filter_logic, index",
    [
        (lambda x: x.datetime < pd.Timestamp("2016-01-01T00:00:00Z"), 0),
        (lambda x: x.datetime > pd.Timestamp("2016-01-01T00:00:00Z"), 1),
    ],
    ids=["2015", "2016"],
)
def test_raster_year(
    year_dsm_loaded: xr.Dataset,
    dsm_items: list[pystac.Item],
    dsm_geobox: GeoBox,
    filter_logic: Callable,
    index: int,
) -> None:
    ds = year_dsm_loaded

    # Prepare ref clients - 2015
    ref_items = list(filter(filter_logic, dsm_items))
    ref_client = MCCN(items=ref_items, geobox=dsm_geobox, time_groupby="year")
    ref_ds = ref_client.load()

    # Compare values
    diff = ds["dsm"].values[index, :, :] - ref_ds["dsm"].values[0, :, :]
    assert diff.max() == 0


@pytest.mark.parametrize(
    "filter_logic, index",
    [
        (lambda x: x.datetime < pd.Timestamp("2015-11-01T00:00:00Z"), 0),  # 2015-10-01
        (
            lambda x: pd.Timestamp("2015-11-01T00:00:00Z")
            < x.datetime
            < pd.Timestamp("2016-01-01T00:00:00Z"),
            1,
        ),
        (
            lambda x: pd.Timestamp("2016-01-01T00:00:00Z")
            < x.datetime
            < pd.Timestamp("2016-11-01T00:00:00Z"),
            2,
        ),
        (
            lambda x: pd.Timestamp("2016-11-01T00:00:00Z") < x.datetime,
            3,
        ),
    ],
    ids=["2015-10", "2015-11", "2016-10", "2016-11"],
)
def test_raster_month(
    dsm_items: list[pystac.Item],
    dsm_geobox: GeoBox,
    month_dsm_loaded: xr.Dataset,
    filter_logic: Callable,
    index: int,
) -> None:
    ds = month_dsm_loaded
    # Prepare ref clients - 2015
    ref_items = list(filter(filter_logic, dsm_items))
    ref_client = MCCN(items=ref_items, geobox=dsm_geobox, time_groupby="month")
    ref_ds = ref_client.load()

    # Compare values
    diff = ds["dsm"].values[index, :, :] - ref_ds["dsm"].values[0, :, :]
    assert diff.max() == 0


# Filter by date Feature Testing
# DSM Dates are
# 2015-10-01, 2015-10-02, 2016-11-01, 2016-11-02
# Groupby year: 2015-01-01, 2016-01-01
# Groupby month: 2015-10-01, 2015-11-01, 2016-10-01, 2016-11-01
@pytest.mark.parametrize(
    "start,end,groupby,exp",
    [
        # No filter
        (
            None,
            None,
            "year",
            ["2015-01-01T00:00:00", "2016-01-01T00:00:00"],
        ),
        # (2016-01-01,)
        ("2016-01-01T00:00:00Z", None, "year", ["2016-01-01T00:00:00"]),
        # (,2016-01-01)
        (None, "2016-01-01T00:00:00Z", "year", ["2015-01-01T00:00:00"]),
        # (2015-11-01, 2016-01-01)
        (
            "2015-11-01T00:00:00Z",
            "2016-01-01T00:00:00Z",
            "month",
            ["2015-11-01T00:00:00"],
        ),
        # (2015-11-01, 2016-10-30)
        (
            "2015-11-01T00:00:00Z",
            "2016-10-30T00:00:00Z",
            "month",
            ["2015-11-01T00:00:00", "2016-10-01T00:00:00"],
        ),
    ],
)
def test_raster_timeslicing(
    start: str | None,
    end: str | None,
    groupby: TimeGroupby,
    exp: list[pd.Timestamp],
    dsm_collection: pystac.Collection,
    dsm_geobox: GeoBox,
) -> None:
    client = MCCN(
        collection=dsm_collection,
        geobox=dsm_geobox,
        start_ts=start,
        end_ts=end,
        time_groupby=groupby,
    )
    ds = client.load()
    assert all(pd.DatetimeIndex(ds["time"].values) == [pd.Timestamp(t) for t in exp])


# Filter by band
# Refers to "tests/loader/raster/fixture/multibands_config.json"
# Items - (name, common name):
# dsm - dsm
# rgb - red, green, blue
# ms-rgb: (ms_red, red), (ms_green, green), (ms_blue, blue)
@pytest.mark.parametrize(
    "bands, exp",
    [
        # Set filter bands = None will load everything
        (
            None,
            {
                "dsm": "top_left_dsm_loaded",
                "red": "top_left_rgb_loaded",
                "green": "top_left_rgb_loaded",
                "blue": "top_left_rgb_loaded",
                "ms-red": "top_left_ms_loaded",
                "ms-green": "top_left_ms_loaded",
                "ms-blue": "top_left_ms_loaded",
            },
        ),
        # Set dsm - only get dsm layer
        (
            {"dsm"},
            {
                "dsm": "top_left_dsm_loaded",
            },
        ),
        # Set dsm and red - get dsm and red + ms-red
        (
            {"dsm", "red"},
            {
                "dsm": "top_left_dsm_loaded",
                "red": "top_left_rgb_loaded",  # Note red and ms-red are overlapping here
            },
        ),
        # Set dsm and ms-red - get dsm and ms-red
        (
            {"dsm", "ms-red"},
            {
                "dsm": "top_left_dsm_loaded",
                "ms-red": "top_left_ms_loaded",
            },
        ),
        # Non existent band - get None
        (
            {"non-matching"},
            {},
        ),
        # Non existent + existent bands
        (
            {"non-matching", "ms-blue", "green"},
            {
                "ms-blue": "top_left_ms_loaded",
                "green": "top_left_rgb_loaded",
            },
        ),
    ],
)
def test_raster_band_filter(
    bands: set[str] | None,
    exp: dict[str, str],
    multibands_collection: pystac.Collection,
    multiband_geobox: GeoBox,
    request: pytest.FixtureRequest,
) -> None:
    client = MCCN(
        collection=multibands_collection,
        geobox=multiband_geobox,
        bands=bands,
    )
    ds = client.load()
    assert set(exp.keys()) == set(ds.data_vars.keys())
    for k, fixture_name in exp.items():
        ref_ds = request.getfixturevalue(fixture_name)
        xr.testing.assert_equal(ds[k], ref_ds[k])


@pytest.mark.parametrize(
    "bands, exp",
    [
        (
            None,
            {"dsm", "red", "green", "blue", "ms-red", "ms-green", "ms-blue"},
        ),
        (
            {"dsm"},
            {"dsm"},
        ),
        (
            {"dsm", "red"},
            {"dsm", "red"},
        ),
        (
            {"dsm", "ms-red"},
            {"dsm", "ms-red"},
        ),
        (
            {"non-matching"},
            set(),
        ),
        (
            {"non-matching", "ms-blue", "green"},
            {"ms-blue", "green"},
        ),
    ],
)
def test_raster_band_filter_ref_against_file(
    bands: set[str] | None,
    exp: set[str],
    multibands_collection: pystac.Collection,
    multiband_geobox: GeoBox,
    multibands_ds: xr.Dataset,
) -> None:
    client = MCCN(
        collection=multibands_collection,
        geobox=multiband_geobox,
        bands=bands,
    )
    ds = client.load()
    assert exp == set(ds.data_vars.keys())
    for k in exp:
        xr.testing.assert_equal(ds[k], multibands_ds[k])


# Filter based on bbox
# multiband collection are all top-left items
def test_raster_geobox_filter_top_right_multibands_collection_expects_none(
    multibands_collection: pystac.Collection, dsm_top_right_geobox: GeoBox
) -> None:
    client = MCCN(
        collection=multibands_collection,
        geobox=dsm_top_right_geobox,
    )
    ds = client.load()
    assert len(ds.data_vars) == 0


# Should load all bands
def test_raster_geobox_filter_bottom_right_multibands_collection_expects_non_null(
    multibands_collection: pystac.Collection, dsm_bottom_right_geobox: GeoBox
) -> None:
    client = MCCN(
        collection=multibands_collection,
        geobox=dsm_bottom_right_geobox,
    )
    ds = client.load()
    assert len(ds.data_vars) == 7


# Set dtype for band
def test_raster_set_dtype(
    multibands_collection: pystac.Collection, dsm_bottom_right_geobox: GeoBox
) -> None:
    dtype_map = {"dsm": "float32", "red": "float64", "green": "int32"}
    client = MCCN(
        collection=multibands_collection,
        geobox=dsm_bottom_right_geobox,
        bands={"dsm", "red", "green"},
        dtype=dtype_map,
    )
    ds = client.load()
    for k, v in dtype_map.items():
        assert ds[k].dtype == v
