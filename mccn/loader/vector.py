from __future__ import annotations

from typing import TYPE_CHECKING, Any, Hashable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.xr import xr_coords
from rasterio.features import rasterize

from mccn._types import ParsedVector
from mccn.loader.base import Loader
from mccn.loader.utils import (
    bbox_from_geobox,
)

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox


def update_attr_legend(
    attr_dict: dict[str, Any], layer_name: str, field: str, frame: gpd.GeoDataFrame
) -> None:
    if not pd.api.types.is_numeric_dtype(frame[field]):
        cat_map = {
            name: index for index, name in enumerate(frame[field].unique(), start=1)
        }
        attr_dict[layer_name] = cat_map
        frame[field] = frame[field].map(cat_map)


def groupby_id(
    data: Mapping[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    coords: Mapping[Hashable, xr.DataArray],
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
) -> xr.Dataset:
    # Load as pure mask
    dss = []
    for key, value in data.items():
        dates = sorted(value[t_col].unique())
        raster = []
        for date in dates:
            raster.append(
                rasterize(
                    value[value[t_col] == date].geometry,
                    (geobox.shape),
                    transform=geobox.transform,
                    masked=True,
                )
            )
        ds_data = np.stack(raster, axis=0)
        ds = xr.Dataset({key: ([t_col, y_col, x_col], ds_data)}, coords=coords)
        ds[t_col] = dates
        dss.append(ds)
    return xr.merge(dss)


def groupby_field(
    data: Mapping[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    fields: set[str],
    coords: Mapping[Hashable, xr.DataArray],
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
) -> xr.Dataset:
    if not data:
        return xr.Dataset()
    gdf = pd.concat(data.values())
    ds_data = {}
    ds_attrs: dict[str, Any] = {"legend": {}}
    dates = sorted(gdf[t_col].unique())  # Date attributes
    for field in fields:
        update_attr_legend(ds_attrs["legend"], field, field, gdf)
        raster = []
        for date in dates:
            raster.append(
                rasterize(
                    (
                        (geom, value)
                        for geom, value in zip(
                            gdf[gdf[t_col] == date].geometry,
                            gdf[gdf[t_col] == date][field],
                        )
                    ),
                    out_shape=geobox.shape,
                    transform=geobox.transform,
                ),
            )

        ds_data[field] = ([t_col, y_col, x_col], np.stack(raster, axis=0))
    ds = xr.Dataset(ds_data, attrs=ds_attrs, coords=coords)
    ds[t_col] = dates
    return ds


class VectorLoader(Loader[ParsedVector]):
    def load(self) -> xr.Dataset:
        data = {}
        bands = set()  # All bands available in vector collection
        mask_only = set()  # Set of items to be loaded as mask only
        for item in self.items:
            if not item.load_aux_bands and not item.load_bands:
                mask_only.add(item.item.id)
            bands.update(item.load_bands)
            bands.update(item.load_aux_bands)
            # Remove date column - already renamed
            if item.config.join_T_column:
                bands.remove(item.config.join_T_column)
            data[item.item.id] = self.load_item(item)
        coords = xr_coords(
            self.filter_config.geobox,
            dims=(self.cube_config.y_coord, self.cube_config.x_coord),
        )
        # Load vector with attributes
        attr_data = groupby_field(
            data={k: v for k, v in data.items() if k not in mask_only},
            geobox=self.filter_config.geobox,
            fields=bands,
            x_col=self.cube_config.x_coord,
            y_col=self.cube_config.y_coord,
            t_col=self.cube_config.t_coord,
            coords=coords,
        )
        # Load mask only vectors:
        mask_data = groupby_id(
            data={k: v for k, v in data.items() if k in mask_only},
            geobox=self.filter_config.geobox,
            x_col=self.cube_config.x_coord,
            y_col=self.cube_config.y_coord,
            t_col=self.cube_config.t_coord,
            coords=coords,
        )

        # Combine attribute + mask
        return xr.merge([attr_data, mask_data])

    def load_item(self, item: ParsedVector) -> gpd.GeoDataFrame:
        date_col = None
        bbox = bbox_from_geobox(self.filter_config.geobox, item.crs)
        # Load main item
        gdf = gpd.read_file(
            item.location,
            bbox=bbox,
            columns=list(item.load_bands),
            layer=item.config.layer,
        )
        # Load aux df
        if item.load_aux_bands:
            if item.config.join_T_column:
                date_col = item.config.join_T_column
                aux_df = pd.read_csv(
                    item.config.join_file,
                    usecols=item.load_aux_bands,
                    parse_dates=[item.config.join_T_column],
                    date_format=item.config.date_format,
                )
            else:
                aux_df = pd.read_csv(
                    item.config.join_file,
                    usecols=item.load_aux_bands,
                )
            # Join dfs
            gdf = pd.merge(
                gdf,
                aux_df,
                left_on=item.config.join_attribute_vector,
                right_on=item.config.join_field,
            )
        # Convert CRS
        gdf.to_crs(self.filter_config.geobox.crs, inplace=True)
        # Process date
        if not date_col:
            gdf[self.cube_config.t_coord] = item.item.datetime
        else:
            gdf.rename(columns={date_col: self.cube_config.t_coord}, inplace=True)

        # Process
        return self.apply_process(gdf, self.process_config)
