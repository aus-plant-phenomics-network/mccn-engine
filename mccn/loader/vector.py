from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Mapping, cast

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.xr import xr_coords
from rasterio.features import rasterize

from mccn.loader.utils import ASSET_KEY, get_item_href

if TYPE_CHECKING:
    import pystac
    from odc.geo.geobox import GeoBox

    from mccn._types import GroupbyOption


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
    alias_renaming: Mapping[str, dict[str, str]] | None = None,
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if fields is None:
        raise ValueError("When groupby field, fields parameter must not be None")
    # Rename columns based on alias map
    if alias_renaming:
        for item_id, renaming_dict in alias_renaming.items():
            data[item_id].rename(columns=renaming_dict, inplace=True)
    if isinstance(fields, str):
        fields = [fields]
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


def stac_load_vector(
    items: Sequence[pystac.Item],
    geobox: GeoBox,
    groupby: GroupbyOption = "id",
    fields: Sequence[str] | dict[str, Sequence[str]] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    asset_key: str | Mapping[str, str] = ASSET_KEY,
    alias_renaming: dict[str, dict[str, str]] | None = None,
) -> xr.Dataset:
    data = {}
    for item in items:
        location = get_item_href(item, asset_key)
        layer = item.properties.get("layer", None)
        gdf = gpd.read_file(location, layer=layer)
        gdf = gdf.to_crs(geobox.crs)
        data[item.id] = gdf
    coords = xr_coords(geobox, dims=(y_col, x_col))
    if groupby == "id":
        ds_data, ds_attrs = groupby_id(data, geobox, fields, x_col, y_col)
    elif groupby == "field":
        ds_data, ds_attrs = groupby_field(
            data, geobox, cast(Sequence[str], fields), alias_renaming, x_col, y_col
        )
    else:
        raise ValueError(
            f"Invalid groupby option: {groupby}. Supported operations include `id`, `field`."
        )
    return xr.Dataset(ds_data, coords, ds_attrs)
