from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Hashable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize as _rasterize
from stac_generator.core.base.utils import (
    calculate_timezone,
    read_join_asset,
    read_vector_asset,
)

from mccn._types import Number_T, ParsedVector
from mccn.config import CubeConfig, FilterConfig, ProcessConfig
from mccn.loader.base import Loader
from mccn.loader.utils import (
    bbox_from_geobox,
    select_by_key,
    update_attr_legend,
)
from mccn.loader.vector.config import RasterizeConfig, VectorLoadConfig

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox


def field_rasterize(
    gdf: gpd.GeoDataFrame,
    field: str,
    dates: pd.Series,
    geobox: GeoBox,
    rasterize_config: RasterizeConfig,
    x_coord: str,
    y_coord: str,
    t_coord: str,
) -> tuple[tuple[str, str, str], np.ndarray]:
    raster: list[np.ndarray] = []
    for date in dates:
        mask = (gdf[t_coord] == date) & (~gdf[field].isna())
        raster.append(
            _rasterize(
                (
                    (geom, value)
                    for geom, value in zip(
                        gdf[mask].geometry,
                        gdf[mask][field],
                    )
                ),
                out_shape=geobox.shape,
                transform=geobox.transform,
                **asdict(rasterize_config),
            ),
        )

    # Stack all date layers to datacube and return result
    ds_data = np.stack(raster, axis=0)
    return (
        t_coord,
        y_coord,
        x_coord,
    ), ds_data


def rasterize(
    data: Mapping[str, gpd.GeoDataFrame],
    coords: Mapping[Hashable, xr.DataArray],
    geobox: GeoBox,
    fields: set[str],
    vector_config: VectorLoadConfig,
    x_coord: str,
    y_coord: str,
    t_coord: str,
    mask_name: str,
    nodata: Number_T | Mapping[str, Number_T],
    nodata_fallback: Number_T,
    categorical_encoding_start: int,
) -> xr.Dataset:
    if not data:
        return xr.Dataset()
    ds_data = {}
    ds_attrs: dict[str, dict[str, Any]] = {}

    # Add mask layer to field to load
    fields.add(mask_name)
    # Make mask attrs
    ds_attrs[mask_name] = {}
    # Assign mapping id to each df
    for idx, (k, v) in enumerate(data.items(), start=categorical_encoding_start):
        v[mask_name] = idx
        ds_attrs[mask_name][str(idx)] = k

    # Concatenate
    gdf = pd.concat(data.values())

    # Prepare dates
    dates = pd.Series(sorted(gdf[t_coord].unique()))

    # Rasterise field by field
    for field in fields:
        update_attr_legend(
            ds_attrs,
            field,
            gdf,
            categorical_encoding_start,
            nodata,
            nodata_fallback,
        )
        rasterize_config = select_by_key(
            field, vector_config.rasterize_config, RasterizeConfig()
        )

        ds_data[field] = field_rasterize(
            gdf,
            field,
            dates,
            geobox,
            rasterize_config,
            x_coord,
            y_coord,
            t_coord,
        )
    ds = xr.Dataset(ds_data, coords=coords, attrs=ds_attrs)
    ds[t_coord] = dates.values
    return ds


def read_asset(
    item: ParsedVector,
    geobox: GeoBox,
    t_coord: str,
    period: str | None,
) -> gpd.GeoDataFrame:
    """Load a single vector item

    Load vector asset. If a join asset is provided, will load the
    join asset and perform a join operation on common column (Inner Join)
    Convert all datetime to UTC but strip off timezone information

    Args:
        item (ParsedVector): parsed vector item
        geobox (GeoBox): target geobox
        t_coord (str): name of the time dimension if valid

    Returns:
        gpd.GeoDataFrame: vector geodataframe
    """
    date_col = item.config.join_config.date_column if item.config.join_config else None
    # Prepare geobox for filtering
    bbox = bbox_from_geobox(geobox, item.crs)
    # Load main item
    gdf = read_vector_asset(
        item.location,
        bbox,
        list(item.load_bands),
        item.config.layer,
    )
    # Load aux df
    if item.load_aux_bands and item.config.join_config:
        join_config = item.config.join_config
        tzinfo = calculate_timezone(gdf.to_crs(4326).geometry)
        aux_df = read_join_asset(
            join_config.file,
            join_config.right_on,
            join_config.date_format,
            join_config.date_column,
            item.load_aux_bands,
            tzinfo,
        )
        # Join dfs
        gdf = pd.merge(
            gdf,
            aux_df,
            left_on=item.config.join_config.left_on,
            right_on=item.config.join_config.right_on,
        )
    # Convert CRS
    gdf.to_crs(geobox.crs, inplace=True)
    # Process date
    if date_col and date_col in item.load_aux_bands:
        gdf.rename(columns={date_col: t_coord}, inplace=True)
    else:
        gdf[t_coord] = item.item.datetime

    # Convert to UTC and remove timezone info
    gdf[t_coord] = gdf[t_coord].dt.tz_convert("utc").dt.tz_localize(None)
    # Prepare groupby for efficiency
    # Need to remove timezone information. Xarray time does not use tz
    if period is not None:
        gdf[t_coord] = gdf[t_coord].dt.to_period(period).dt.start_time
    return gdf


class VectorLoader(Loader[ParsedVector]):
    """
    Vector STAC loader

    Similar to other item loaders, each band is loaded with dimension (time, y, x)
    Time is derived from the asset (mainly the external asset that joins with the main vector file) if valid (join_T_column is present),
    or from item's datetime field otherwise.

    Vectors can be loaded as masks (if no column_info is described in STAC) or as attribute/band layer. If an external asset (join_file) is
    described in STAC, an inner join operation will join the vector file's join_vector_attribute with the external asset's join_field to produce
    a join frame whose attributes will be loaded as band/variable in the datacube.

    Masks can be loaded in two modes - groupby field and groupby id. If masks are grouped by
    field, all masks are loaded to a single MASKS layer with dimension (time, y, x).
    If masks are grouped by id, each item is loaded as an independent mask with layer name being
    the item's id. This parameter can be updated using load_config.

    Users can control the dimension of the cube by updating cube_config parameter, control the renaming and preprocessing of fields by updating
    process_config, and control the rasterize operation using load_config.

    """

    def __init__(
        self,
        items: Sequence[ParsedVector],
        filter_config: FilterConfig,
        cube_config: CubeConfig | None = None,
        process_config: ProcessConfig | None = None,
        load_config: VectorLoadConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.load_config = load_config if load_config else VectorLoadConfig()
        super().__init__(items, filter_config, cube_config, process_config, **kwargs)

    def _collect_fields_to_load(self) -> set[str]:
        if self.filter_config.mask_only:
            return set()
        fields = set()
        for item in self.items:
            fields.update(item.load_bands)
            fields.update(item.load_aux_bands)
            if item.config.join_config and item.config.join_config.date_column:
                fields.remove(item.config.join_config.date_column)
        return fields

    def _load(self) -> xr.Dataset:
        data = {}  # Mapping of item id to geodataframe
        fields = self._collect_fields_to_load()

        # Prepare items
        for item in self.items:
            item_id = item.item.id
            data[item_id] = self.apply_process(
                read_asset(
                    item,
                    self.filter_config.geobox,
                    self.cube_config.t_coord,
                    self.process_config.period,
                ),
                self.process_config,
            )
        return rasterize(
            data=data,
            coords=self.coords,
            geobox=self.filter_config.geobox,
            fields=fields,
            vector_config=self.load_config,
            x_coord=self.cube_config.x_coord,
            y_coord=self.cube_config.y_coord,
            t_coord=self.cube_config.t_coord,
            mask_name=self.cube_config.mask_name,
            nodata=self.process_config.nodata,
            nodata_fallback=self.process_config.nodata_fallback,
            categorical_encoding_start=self.process_config.categorical_encoding_start,
        )
