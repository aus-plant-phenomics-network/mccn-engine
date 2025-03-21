from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Mapping, cast

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.xr import xr_coords
from rasterio.features import rasterize

from mccn.loader.utils import (
    ASSET_KEY,
    bbox_from_geobox,
    get_item_crs,
    get_item_href,
    get_required_columns,
)

if TYPE_CHECKING:
    import pystac
    from odc.geo.geobox import GeoBox

JOIN_VECTOR_KEY = "join_attribute_vector"
JOIN_FILE_KEY = "join_field"


def update_attr_legend(
    attr_dict: dict[str, Any], layer_name: str, field: str, frame: gpd.GeoDataFrame
) -> None:
    if pd.api.types.is_string_dtype(frame[field]):
        cat_map = {name: index for index, name in enumerate(frame[field].unique())}
        attr_dict[layer_name] = cat_map
        frame[field] = frame[field].map(cat_map)


def groupby_id(
    data: Mapping[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    fields: Sequence[str] | dict[str, Sequence[str]] | None = None,
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Load as pure mask
    if fields is None:
        return {
            key: (
                [y_col, x_col],
                rasterize(
                    value.geometry,
                    geobox.shape,
                    transform=geobox.transform,
                    masked=True,
                ),
            )
            for key, value in data.items()
        }, {}
    # Load as attribute per layer
    # Prepare field for each layer
    item_fields = {}
    if isinstance(fields, dict):
        if set(data.keys()).issubset(set(fields.keys())):
            raise ValueError(
                f"Vector Loader: when groupby id and field is provided as a dictionary, its key must be a superset of ids of all vector items in the collection. {set(data.keys()) - set(fields.keys())}"
            )
        item_fields = fields
    else:
        item_fields = {k: fields for k in data.keys()}

    ds_data = {}
    ds_attrs: dict[str, Any] = {"legend": {}}
    # Field per layer
    for k, frame in data.items():
        for field in item_fields[k]:
            layer_name = f"{k}_{field}"
            update_attr_legend(ds_attrs["legend"], layer_name, field, frame)
            # Build legend mapping for categorical encoding of values
            ds_data[layer_name] = (
                [y_col, x_col],
                rasterize(
                    (
                        (geom, value)
                        for geom, value in zip(frame.geometry, frame[field])
                    ),
                    geobox.shape,
                    transform=geobox.transform,
                ),
            )
    return ds_data, ds_attrs


def groupby_field(
    data: Mapping[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    fields: Sequence[str],
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if fields is None:
        raise ValueError("When groupby field, fields parameter must not be None")
    gdf = pd.concat(data.values())
    ds_data = {}
    ds_attrs: dict[str, Any] = {"legend": {}}
    for field in fields:
        update_attr_legend(ds_attrs["legend"], field, field, gdf)
        ds_data[field] = (
            [y_col, x_col],
            rasterize(
                ((geom, value) for geom, value in zip(gdf.geometry, gdf[field])),
                out_shape=geobox.shape,
                transform=geobox.transform,
            ),
        )
    return ds_data, ds_attrs


def read_vector_file(
    item: pystac.Item,
    geobox: GeoBox,
    asset_key: str | Mapping[str, str] = ASSET_KEY,
    bands: Sequence[str] | None = None,
    band_preprocessing: dict[str, Callable] | None = None,
    band_renaming: dict[str, str] | None = None,
) -> gpd.GeoDataFrame | None:
    location = get_item_href(item, asset_key)
    layer = item.properties.get("layer", None)

    # Prepare requested bands
    # If vector file has join file - will also be requesting the join key
    requested_bands = copy.copy(set(bands)) if bands is not None else None
    if JOIN_VECTOR_KEY in item.properties and item.properties[JOIN_FILE_KEY]:
        if requested_bands is None:
            requested_bands = {item.properties[JOIN_FILE_KEY]}
        else:
            requested_bands.add(item.properties[JOIN_FILE_KEY])

    columns = get_required_columns(item, bands, band_renaming)
    # Specific bands requested and no matching column found -> filter out the items
    if not columns and bands:
        return None
    crs = get_item_crs(item)
    bbox = bbox_from_geobox(geobox, crs)
    gdf = gpd.read_file(location, layer=layer, bbox=bbox, columns=columns)
    gdf = gdf.to_crs(geobox.crs)
    # Apply transformation to each band
    if band_preprocessing:
        for band, transform in band_preprocessing.items():
            if band in gdf.columns:
                gdf[band] = gdf[band].apply(transform)
    # Rename bands
    if band_renaming:
        gdf.rename(columns=band_renaming, inplace=True)
    return gdf


def stac_load_vector(
    items: Sequence[pystac.Item],
    geobox: GeoBox,
    bands: Sequence[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    asset_key: str | Mapping[str, str] = ASSET_KEY,
    band_preprocessing: dict[str, Callable] | None = None,
    band_renaming: dict[str, str] | None = None,
) -> xr.Dataset:
    data = {}
    for item in items:
        data[item.id] = read_vector_file(
            item, geobox, asset_key, bands, band_preprocessing, band_renaming
        )
    coords = xr_coords(geobox, dims=(y_col, x_col))

    ds_data, ds_attrs = groupby_field(
        data, geobox, cast(Sequence[str], bands), x_col, y_col
    )
    return xr.Dataset(ds_data, coords, ds_attrs)
