from __future__ import annotations

from typing import TYPE_CHECKING

import odc.stac
import pystac
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odc.geo.geobox import GeoBox


def stac_load_raster(
    items: Sequence[pystac.Item],
    geobox: GeoBox | None,
    bands: str | Sequence[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
) -> xr.Dataset:
    """Loader raster data

    Provides a thin wrapper for `odc.stac.load`. Also rename the `x`, `y` and `time` dimensions
    to input values `x_col`, `y_col`, and `t_col` for subsequent merging

    :param items: list of stac Item that has raster as the source asset
    :type items: Sequence[pystac.Item]
    :param geobox: object that defines the spatial extent, shape, and crs of the output dataset
    :type geobox: GeoBox | None
    :param bands: selected bands to read from source tiff. If None, read all bands, defaults to None
    :type bands: str | Sequence[str] | None, optional
    :param x_col: renamed x coordinate, defaults to "x"
    :type x_col: str, optional
    :param y_col: renamed y coordinate, defaults to "y"
    :type y_col: str, optional
    :param t_col: renamed time coordinate, defaults to "time"
    :type t_col: str, optional
    :return: raster data as xarray.Dataset
    :rtype: xr.Dataset
    """
    ds = odc.stac.load(items, bands, geobox=geobox)
    # NOTE: odc stac load uses odc.geo.xr.xr_coords to set dimension name
    # it either uses latitude/longitude or y/x depending on the underlying crs
    # so there is no proper way to know which one it uses aside from trying
    if "latitude" in ds.dims and "longitude" in ds.dims:
        ds = ds.rename({"longitude": x_col, "latitude": y_col})
    elif "x" in ds.dims and "y" in ds.dims:
        ds = ds.rename({"x": x_col, "y": y_col})
    if "time" in ds.dims:
        ds = ds.rename({"time": t_col})
    return ds
