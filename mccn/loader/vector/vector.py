from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize

from mccn._types import CubeConfig, FilterConfig, ParsedVector, ProcessConfig
from mccn.loader.base import Loader
from mccn.loader.utils import (
    bbox_from_geobox,
)

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox

    from mccn.loader.vector.config import VectorLoadConfig, VectorRasterizeConfig


def update_attr_legend(
    attr_dict: dict[str, Any], field: str, frame: gpd.GeoDataFrame
) -> None:
    """Update attribute dict with legend for non numeric fields.

    If the field is non-numeric - i.e. string, values will be categoricalised
    i.e. 1, 2, 3, ...
    The mapping will be updated in attr_dict under field name

    Args:
        attr_dict (dict[str, Any]): attribute dict
        field (str): field name
        frame (gpd.GeoDataFrame): input data frame
    """
    if not pd.api.types.is_numeric_dtype(frame[field]):
        cat_map = {
            name: index for index, name in enumerate(frame[field].unique(), start=1)
        }
        attr_dict[field] = {v: k for k, v in cat_map.items()}
        frame[field] = frame[field].map(cat_map)


def field_rasterize(
    gdf: gpd.GeoDataFrame,
    field: str,
    dates: pd.Series,
    geobox: GeoBox,
    rasterize_config: VectorRasterizeConfig,
    cube_config: CubeConfig,
) -> tuple[tuple[str, str, str], np.ndarray]:
    raster: list[np.ndarray] = []
    for date in dates:
        raster.append(
            rasterize(
                (
                    (geom, value)
                    for geom, value in zip(
                        gdf[gdf[cube_config.t_coord] == date].geometry,
                        gdf[gdf[cube_config.t_coord] == date][field],
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
        cube_config.t_coord,
        cube_config.y_coord,
        cube_config.x_coord,
    ), ds_data


def groupby(
    data: Mapping[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    fields: set[str],
    load_mask_only: bool,
    mask_layer_name: str,
    mask_value_start: int,
    cube_config: CubeConfig,
    rasterize_config: VectorRasterizeConfig,
) -> xr.Dataset:
    if not data:
        return xr.Dataset()
    ds_data = {}
    ds_attrs: dict[str, dict[str, Any]] = {}

    # if load as mask - load mask only
    if load_mask_only:
        fields.clear()

    # Add mask layer to field to load
    fields.add(mask_layer_name)
    # Make mask attrs
    ds_attrs[mask_layer_name] = {}
    # Assign mapping id to each df
    for idx, (k, v) in enumerate(data.items(), start=mask_value_start):
        v[mask_layer_name] = idx
        ds_attrs[mask_layer_name][str(idx)] = k

    # Concatenate
    gdf = pd.concat(data.values())
    # Prepare dates
    dates = pd.Series(sorted(gdf[cube_config.t_coord].unique()))

    # Rasterise field by field
    for field in fields:
        update_attr_legend(ds_attrs, field, gdf)
        ds_data[field] = field_rasterize(
            gdf, field, dates, geobox, rasterize_config, cube_config
        )
    ds = xr.Dataset(ds_data, coords=geobox.coordinates, attrs=ds_attrs)
    ds[cube_config.t_coord] = dates.values
    return ds


def read_vector_asset(
    item: ParsedVector, geobox: GeoBox, t_coord: str = "time"
) -> gpd.GeoDataFrame:
    """Load a single vector item

    Load vector asset. If a join asset is provided, will load the
    join asset and perform a join operation on common column (Inner Join)

    Args:
        item (ParsedVector): parsed vector item
        geobox (GeoBox): target geobox
        t_coord (str): name of the time dimension if valid

    Returns:
        gpd.GeoDataFrame: vector geodataframe
    """
    date_col = None
    # Prepare geobox for filtering
    bbox = bbox_from_geobox(geobox, item.crs)
    # Load main item
    gdf = gpd.read_file(
        item.location,
        bbox=bbox,
        columns=list(item.load_bands),
        layer=item.config.layer,
    )
    # Load aux df
    if item.load_aux_bands:
        if item.config.join_T_column and item.config.join_file:
            date_col = item.config.join_T_column
            aux_df = pd.read_csv(
                item.config.join_file,
                usecols=list(item.load_aux_bands),
                parse_dates=[item.config.join_T_column],
                date_format=cast(str, item.config.date_format),
            )
        else:
            aux_df = pd.read_csv(
                cast(str, item.config.join_file),
                usecols=list(item.load_aux_bands),
            )
        # Join dfs
        gdf = pd.merge(
            gdf,
            aux_df,
            left_on=item.config.join_attribute_vector,
            right_on=item.config.join_field,
        )
    # Convert CRS
    gdf.to_crs(geobox.crs, inplace=True)
    # Process date
    if not date_col:
        gdf[t_coord] = item.item.datetime
    else:
        gdf.rename(columns={date_col: t_coord}, inplace=True)
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

    def 

    def _load(self) -> xr.Dataset:
        data = {}  # Mapping of item id to geodataframe
        bands = set()  # All bands available in vector collection
        mask_only = set()  # Set of items to be loaded as mask only

        # Prepare items
        for item in self.items:
            if not item.load_aux_bands and not item.load_bands:
                mask_only.add(item.item.id)
            bands.update(item.load_bands)
            bands.update(item.load_aux_bands)
            # Remove date column - not a variable to load
            if item.config.join_T_column:
                bands.remove(item.config.join_T_column)
            data[item.item.id] = self.apply_process(
                read_vector_asset(
                    item, self.filter_config.geobox, self.cube_config.t_coord
                ),
                self.process_config,
            )
