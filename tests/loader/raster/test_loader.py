from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from mccn.client import MCCN

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
    dsm_collection: pystac.Collection,
    dsm_geobox: GeoBox,
    groupby: TimeGroupby,
    exp_ts: list[pd.Timestamp],
) -> None:
    engine = MCCN(
        collection=dsm_collection,
        geobox=dsm_geobox,
        time_groupby=groupby,
        t_coord=T_COORD,
    )
    ds = engine.load_raster()

    # Verify dates
    assert len(ds[T_COORD]) == len(exp_ts)  # 2 Years - 2015 and 2016
    timestamps = pd.DatetimeIndex(ds[T_COORD].values)
    assert all(timestamps == exp_ts)


@pytest.mark.parametrize(
    "filter_logic, index",
    [
        (lambda x: x.datetime < pd.Timestamp("2016-01-01T00:00:00Z"), 0),
        (lambda x: x.datetime > pd.Timestamp("2016-01-01T00:00:00Z"), 1),
    ],
    ids=["2015", "2016"],
)
def test_raster_year_generation_expects_full_matching(
    dsm_items: list[pystac.Item],
    dsm_geobox: GeoBox,
    filter_logic: Callable,
    index: int,
) -> None:
    groupby: TimeGroupby = "year"
    client = MCCN(
        items=dsm_items, geobox=dsm_geobox, time_groupby=groupby, t_coord=T_COORD
    )
    ds = client.load_raster()

    # Prepare ref clients - 2015
    ref_items = list(filter(filter_logic, dsm_items))
    ref_client = MCCN(items=ref_items, geobox=dsm_geobox, time_groupby=groupby)
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
    filter_logic: Callable,
    index: int,
) -> None:
    groupby: TimeGroupby = "month"
    client = MCCN(
        items=dsm_items, geobox=dsm_geobox, time_groupby=groupby, t_coord=T_COORD
    )
    ds = client.load_raster()

    # Prepare ref clients - 2015
    ref_items = list(filter(filter_logic, dsm_items))
    ref_client = MCCN(items=ref_items, geobox=dsm_geobox, time_groupby=groupby)
    ref_ds = ref_client.load_raster()

    # Compare values
    diff = ds["dsm"].values[index, :, :] - ref_ds["dsm"].values[0, :, :]
    assert diff.max() == 0
