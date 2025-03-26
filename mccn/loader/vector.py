from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Mapping, cast

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.xr import xr_coords
from rasterio.features import rasterize

from mccn._types import ParsedVector
from mccn.loader.base import Loader
from mccn.loader.utils import (
    ASSET_KEY,
    bbox_from_geobox,
    get_item_crs,
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


def groupby_field(
    data: Mapping[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    fields: Sequence[str],
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[dict[str, Any], dict[str, Any]]:
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


class VectorLoader(Loader[ParsedVector]):
    def load(self):
        data = {}
        bands = set()
        for item in self.items:
            bands.update(item.load_bands)
            bands.update(item.load_aux_bands)
            data[item.item.id] = self.load_item(item)
        coords = xr_coords(
            self.filter_config.geobox,
            dims=(self.cube_config.y_coord, self.cube_config.x_coord),
        )
        ds_data, ds_attrs = groupby_field(
            data,
            self.filter_config.geobox,
            bands,
            self.cube_config.x_coord,
            self.cube_config.y_coord,
        )

        return xr.Dataset(ds_data, coords, ds_attrs)

    def load_item(self, item: ParsedVector) -> gpd.GeoDataFrame:
        bbox = bbox_from_geobox(self.filter_config.geobox, item.crs)
        # Load main item
        gdf = gpd.read_file(
            item.location,
            bbox=bbox,
            columns=item.load_bands,
            layer=item.config.layer,
        )
        # Load aux df
        if item.load_aux_bands:
            aux_df = pd.read_csv(item.config.join_file, usecols=item.load_aux_bands)
            # Join dfs
            gdf = pd.merge(
                gdf,
                aux_df,
                left_on=item.config.join_attribute_vector,
                right_on=item.config.join_field,
            )
        # Convert CRS
        gdf.to_crs(self.filter_config.geobox.crs, inplace=True)
        # Process
        return self.apply_process(gdf, self.process_config)
