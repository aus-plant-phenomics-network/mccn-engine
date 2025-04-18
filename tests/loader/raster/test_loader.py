import datetime
from typing import cast

import pandas as pd
import pystac
import pytest
from odc.geo.geobox import GeoBox

from mccn._types import CubeConfig
from mccn.loader.raster.config import RasterLoadConfig
from mccn.loader.raster.loader import read_raster_asset


@pytest.fixture()
def cube_config() -> CubeConfig:
    return CubeConfig()


def test_year_raster_generation(
    dsm_collection: pystac.Collection,
    cube_config: CubeConfig,
) -> None:
    items = list(dsm_collection.get_items(recursive=True))
    ds = read_raster_asset(
        items,
        None,
        None,
        cube_config=cube_config,
        raster_config=RasterLoadConfig(groupby="year"),
    )
    assert cube_config.x_coord in ds.dims
    assert cube_config.y_coord in ds.dims
    assert cube_config.t_coord in ds.dims

    # Verify dates
    assert len(ds[cube_config.t_coord]) == 2  # 2 Years - 2015 and 2016
    timestamps = [pd.Timestamp(ts) for ts in ds[cube_config.t_coord].values]
    years = [ts.year for ts in timestamps]
    assert 2015 in years
    assert 2016 in years

    # Verify layers aggregated correctly - 2016 data should be the same as that of ds

    ref_ds = read_raster_asset(
        [item for item in items if cast(datetime.datetime, item.datetime).year >= 2016],
        None,
        None,
        cube_config=cube_config,
        raster_config=RasterLoadConfig(groupby="year"),
    )
    assert len(ref_ds[cube_config.t_coord]) == 1  # 2 Years - 2015 and 2016

    ds_data = ds["data"][0, :, :].values
    ref_data = ref_ds["data"][0, :, :].values
    diff = ds_data - ref_data
    assert diff.sum() < 0.05


def test_month_raster_generation(
    dsm_collection: pystac.Collection,
    dsm_geobox: GeoBox,
    cube_config: CubeConfig,
) -> None:
    items = list(dsm_collection.get_items(recursive=True))
    items_2016 = [
        item for item in items if cast(datetime.datetime, item.datetime).year >= 2016
    ]
    ds = read_raster_asset(
        items_2016,
        None,
        dsm_geobox,
        cube_config=cube_config,
        raster_config=RasterLoadConfig("month"),
    )
    assert cube_config.x_coord in ds.dims
    assert cube_config.y_coord in ds.dims
    assert cube_config.t_coord in ds.dims

    # Verify dates
    assert len(ds[cube_config.t_coord]) == 2  # 2 months - 2016-10 and 2016-11
    timestamps = [pd.Timestamp(ts) for ts in ds[cube_config.t_coord].values]
    years = [ts.strftime("%Y-%m") for ts in timestamps]
    assert "2016-10" in years
    assert "2016-11" in years

    # Compare against 2016/10 cube
    ref_ds = read_raster_asset(
        items=[
            item
            for item in items_2016
            if cast(datetime.datetime, item.datetime).month < 11
        ],
        bands=None,
        geobox=dsm_geobox,
        cube_config=cube_config,
        raster_config=RasterLoadConfig(groupby="year"),
    )
    ds_data = ds["data"].values[0, :, :]
    ref_data = ref_ds["data"].values[0, :, :]
    diff = ds_data - ref_data
    assert diff.sum() <= 0.05
