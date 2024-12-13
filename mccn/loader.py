from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import geopandas as gpd
import odc.stac
import pystac
import pystac_client
import xarray as xr
from odc.geo.xr import xr_coords

from mccn.utils import (
    ASSET_KEY,
    BBOX_TOL,
    get_item_href,
    groupby_field,
    groupby_id,
    point_data_to_xarray,
    process_groupby,
    read_point_asset,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odc.geo.geobox import GeoBox

    from mccn._types import GroupbyOption, InterpMethods, MergeMethods


def stac_load_point(
    items: Sequence[pystac.Item],
    geobox: GeoBox,
    asset_key: str | Mapping[str, str] = ASSET_KEY,
    fields: Sequence[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
    merge_method: MergeMethods = "mean",
    interp_method: InterpMethods | None = "nearest",
    tol: float = BBOX_TOL,
) -> xr.Dataset:
    frames = []
    for item in items:
        frame = read_point_asset(item, fields, asset_key, t_col)
        if frame is not None:  # Can be None if does not contain required column
            frame = frame.to_crs(geobox.crs)
            # Process groupby - i.e. average out over depth, duplicate entries, etc
            merged = process_groupby(
                frame, geobox, x_col, y_col, t_col, merge_method, tol
            )
            frames.append(
                point_data_to_xarray(merged, geobox, x_col, y_col, t_col, interp_method)
            )
    return xr.merge(frames)


def stac_load_vector(
    items: Sequence[pystac.Item],
    geobox: GeoBox,
    groupby: GroupbyOption = "id",
    fields: Sequence[str] | dict[str, Sequence[str]] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    asset_key: str = ASSET_KEY,
    alias_renaming: dict[str, tuple[str, str]] | None = None,
) -> xr.Dataset:
    data = {}
    for item in items:
        gdf = gpd.read_file(item.assets[asset_key].href)
        gdf = gdf.to_crs(geobox.crs)
        data[item.id] = gdf
    coords = xr_coords(geobox, dims=[y_col, x_col])
    if groupby == "id":
        ds_data, ds_attrs = groupby_id(data, geobox, fields, x_col, y_col)
    elif groupby == "field":
        ds_data, ds_attrs = groupby_field(
            data, geobox, fields, alias_renaming, x_col, y_col
        )
    else:
        raise ValueError(
            f"Invalid groupby option: {groupby}. Supported operations include `id`, `field`."
        )
    return xr.Dataset(ds_data, coords, ds_attrs)


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


class Loader:
    def __init__(
        self,
        collection: str,
        geobox: GeoBox,
        asset_key: str | Mapping[str, str] = ASSET_KEY,
        fields: Sequence[str] | None = None,
        x_col: str = "x",
        y_col: str = "y",
        t_col: str = "time",
        merge_method: MergeMethods = "mean",
        interp_method: InterpMethods = "nearest",
        tol: float = BBOX_TOL,
    ) -> None:
        self.collection = collection
        self.geobox = geobox
        self.asset_key = asset_key
        self.fields = fields
        self.x_col = x_col
        self.y_col = y_col
        self.t_col = t_col
        self.merge_method = merge_method
        self.interp_method = interp_method
        self.tol = tol
        self.point_items: list[pystac.Item] = []
        self.vector_items: list[pystac.Item] = []
        self.raster_items: list[pystac.Item] = []

    def process_collection(self) -> None:
        self.client = pystac_client.Client.open(self.collection)
        for item in self.client.get_all_items():
            href = get_item_href(item, self.asset_key)
            if href.endswith("csv"):
                self.point_items.append(item)
            elif href.endswith("geotiff") or href.endswith("tiff"):
                self.raster_items.append(item)
            else:
                self.vector_items.append(item)

    def load(self) -> xr.Dataset:
        ds = []
        if self.point_items:
            ds.append(
                stac_load_point(
                    self.point_items,
                    self.geobox,
                    self.asset_key,
                    self.fields,
                    self.x_col,
                    self.y_col,
                    self.t_col,
                    self.merge_method,
                    self.interp_method,
                    self.tol,
                )
            )
        if self.vector_items:
            ds.append(stac_load_vector(self.vector_items, self.geobox, self.asset_key))
        if self.raster_items:
            ds.append(
                stac_load_raster(
                    self.raster_items,
                    self.geobox,
                    self.fields,
                    self.x_col,
                    self.y_col,
                    self.t_col,
                )
            )
        return xr.merge(ds)
