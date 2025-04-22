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
        x_coord=X_COORD,
        y_coord=Y_COORD,
        t_coord=T_COORD,
    )
    ds = engine.load_raster()
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
        t_coord=T_COORD,
    )
    return engine.load_raster()


@pytest.fixture(scope="module")
def month_dsm_loaded(
    dsm_collection: pystac.Collection,
    dsm_geobox: GeoBox,
) -> xr.Dataset:
    engine = MCCN(
        collection=dsm_collection,
        geobox=dsm_geobox,
        time_groupby="month",
        t_coord=T_COORD,
    )
    return engine.load_raster()


@pytest.fixture(scope="module")
def day_dsm_loaded(
    dsm_collection: pystac.Collection,
    dsm_geobox: GeoBox,
) -> xr.Dataset:
    engine = MCCN(
        collection=dsm_collection,
        geobox=dsm_geobox,
        time_groupby="day",
        t_coord=T_COORD,
    )
    return engine.load_raster()


@pytest.fixture(scope="module")
def hour_dsm_loaded(
    dsm_collection: pystac.Collection,
    dsm_geobox: GeoBox,
) -> xr.Dataset:
    engine = MCCN(
        collection=dsm_collection,
        geobox=dsm_geobox,
        time_groupby="hour",
        t_coord=T_COORD,
    )
    return engine.load_raster()


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
    return client.load_raster()


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
    return client.load_raster()


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
    return client.load_raster()


@pytest.fixture(scope="module")
def multibands_ds() -> xr.Dataset:
    return xr.open_dataset(RASTER_FIXTURE_PATH / "multibands.cd")


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
def test_raster_generation_expects_correct_time_rounded_ts(
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
def test_raster_year_generation_expects_full_matching(
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
    ref_ds = ref_client.load_raster()

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
def test_raster_month_generation_expects_full_matching(
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
    ref_ds = ref_client.load_raster()

    # Compare values
    diff = ds["dsm"].values[index, :, :] - ref_ds["dsm"].values[0, :, :]
    assert diff.max() == 0


# FILTER BY DATE FEATURE TESTING
@pytest.mark.parametrize(
    "start,end,groupby,exp",
    [
        (
            None,
            None,
            "year",
            ["2015-01-01T00:00:00", "2016-01-01T00:00:00"],
        ),
        ("2016-01-01T00:00:00Z", None, "year", ["2016-01-01T00:00:00"]),
        (None, "2016-01-01T00:00:00Z", "year", ["2015-01-01T00:00:00"]),
        (
            "2015-11-01T00:00:00Z",
            "2016-01-01T00:00:00Z",
            "month",
            ["2015-11-01T00:00:00"],
        ),
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
    ds = client.load_raster()
    assert all(pd.DatetimeIndex(ds["time"].values) == [pd.Timestamp(t) for t in exp])


# FILTER BY BAND
@pytest.mark.parametrize(
    "bands, exp",
    [
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
        (
            {"dsm"},
            {
                "dsm": "top_left_dsm_loaded",
            },
        ),
        (
            {"dsm", "red"},
            {
                "dsm": "top_left_dsm_loaded",
                "red": "top_left_rgb_loaded",
            },
        ),
        (
            {"dsm", "ms-red"},
            {
                "dsm": "top_left_dsm_loaded",
                "ms-red": "top_left_ms_loaded",
            },
        ),
        (
            {"non-matching"},
            {},
        ),
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
    ds = client.load_raster()
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
    ds = client.load_raster()
    assert exp == set(ds.data_vars.keys())
    for k in exp:
        xr.testing.assert_equal(ds[k], multibands_ds[k])
