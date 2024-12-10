from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import geopandas as gpd
import odc.stac
import pystac
import pystac_client
import rasterio
import rioxarray
import xarray as xr
from rasterio import MemoryFile, features

from mccn.utils import (
    ASSET_KEY,
    BBOX_TOL,
    get_item_href,
    item_in_geobox,
    merge_frames,
    point_data_to_xarray,
    read_point_asset,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odc.geo.geobox import GeoBox

    from mccn._types import InterpMethods, MergeMethods


def stac_load_point(
    items: Sequence[pystac.Item],
    geobox: GeoBox,
    asset_key: str | Mapping[str, str] = ASSET_KEY,
    fields: Sequence[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
    merge_method: MergeMethods = "mean",
    interp_method: InterpMethods = "nearest",
    tol: float = BBOX_TOL,
) -> xr.Dataset:
    frames = []
    for item in items:
        # Only read items that are in geobox and have required columns
        if item_in_geobox(item, geobox):
            frame = read_point_asset(item, fields, asset_key, x_col, y_col, t_col)
            if frame is not None:  # Can be None if does not contain required column
                frames.append(frame)
    merged = merge_frames(frames, geobox, x_col, y_col, t_col, merge_method, tol)
    return point_data_to_xarray(merged, geobox, x_col, y_col, interp_method)


def stac_load_vector(
    items: Sequence[pystac.Item],
    geobox: GeoBox,
    asset_key: str | Mapping[str, str] = ASSET_KEY,
) -> xr.Dataset:
    """
    Load several STAC :class: `~pystac.item.Item` objects (from the same or similar collections)
    as an :class: `xarray.Dataset`.

    This method takes STAC objects describing vector assets (shapefile, geojson) and rasterises
    them using the provided Geobox parameter.
    :param items: Iterable of STAC :class `~pystac.item.Item` to load.
    :param gbox: Allows to specify exact region/resolution/projection using
       :class:`~odc.geo.geobox.GeoBox` object
    :return: Xarray datacube with rasterised vector layers.
    """
    # A temporary raster file is built in memory using attributes of the Geobox.
    with MemoryFile().open(
        driver="GTiff",
        crs=geobox.crs,
        transform=geobox.transform,
        dtype=rasterio.uint8,  # Results in uint8 dtype in xarray.DataArray
        count=len(items),
        width=geobox.width,
        height=geobox.height,
    ) as memfile:
        for index, item in enumerate(items):
            vector_filepath = get_item_href(item, asset_key)
            vector = gpd.read_file(vector_filepath)
            # Reproject polygons into the target CRS
            vector = vector.to_crs(geobox.crs)
            geom = [shapes for shapes in vector.geometry]
            # Rasterise polygons with 1 if centre of pixel inside polygon, 0 otherwise
            rasterized = features.rasterize(
                geom,
                out_shape=geobox.shape,
                fill=0,
                out=None,
                transform=geobox.transform,
                all_touched=False,
                default_value=1,  # 1 for boolean mask
                dtype=None,
            )
            # Raster bands are 1-indexed
            memfile.write(rasterized, index + 1)

    # The temporary raster file can then be read into xarray like a normal raster file.
    xx = rioxarray.open_rasterio(memfile.name)
    # TODO: Label the layers in the datacube
    return xx  # type: ignore[return-value]


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
