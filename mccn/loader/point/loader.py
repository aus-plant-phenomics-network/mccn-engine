from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

import geopandas as gpd
import pandas as pd
import xarray as xr
from stac_generator.core.base.utils import read_point_asset

from mccn._types import CRS_T
from mccn.loader.base import Loader
from mccn.loader.point.config import PointLoadConfig
from mccn.loader.utils import update_attr_legend
from mccn.parser import ParsedPoint

if TYPE_CHECKING:
    from mccn._types import MergeMethod_Map_T
    from mccn.config import (
        CubeConfig,
        FilterConfig,
        ProcessConfig,
    )


class PointLoader(Loader[ParsedPoint]):
    """Point Loader

    The loading process comprises of:
    - Loading point data as GeoDataFrame from asset's location
    - Aggregating data by (time, y, x) or (time, y, x, z) depending on whether use_z is set
    - Interpolating point data into the geobox

    Note:
    - Aggregation is necessary for removing duplicates, either intentional or unintentional. For
    instance, we may not want to use depth value in a soil dataset. In that case, aggregation with
    mean will average soil traits over different depths.

    Caveats:
    - Point data bands should contain numeric values only - aggregation does not work with non-numeric data.
    - Interpolating into a geobox grid may lead to fewer values. This is the case of falling through the mesh.

    """

    def __init__(
        self,
        items: Sequence[ParsedPoint],
        filter_config: FilterConfig,
        cube_config: CubeConfig | None = None,
        process_config: ProcessConfig | None = None,
        load_config: PointLoadConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.load_config = load_config if load_config else PointLoadConfig()
        self.attr_map: dict[str, Any] = {}
        super().__init__(items, filter_config, cube_config, process_config, **kwargs)

    def _load(self) -> xr.Dataset:
        return xr.Dataset()
        # frames = []
        # for item in self.items:
        #     frames.append(self.load_item(item))
        # return xr.merge(frames)


def read_asset(
    item: ParsedPoint,
    t_coord: str,
    z_coord: str,
    crs: CRS_T,
) -> gpd.GeoDataFrame:
    # Read csv
    frame = read_point_asset(
        src_path=item.location,
        X_coord=item.config.X,
        Y_coord=item.config.Y,
        epsg=cast(int, item.crs.to_epsg()),
        T_coord=item.config.T,
        date_format=item.config.date_format,
        Z_coord=item.config.Z,
        columns=list(item.load_bands),
        timezone=item.config.timezone,
    )
    # Prepare rename dict
    rename_dict = {}
    if item.config.T:
        rename_dict[item.config.T] = t_coord
    else:  # If point data does not contain date - set datecol using item datetime
        frame[t_coord] = item.item.datetime
    if item.config.Z:
        rename_dict[item.config.Z] = z_coord

    # Convert to geobox crs
    frame = frame.to_crs(crs)

    # Rename indices
    frame.rename(columns=rename_dict, inplace=True)
    # Drop X and Y columns since we will repopulate them after changing crs
    frame.drop(columns=[item.config.X, item.config.Y], inplace=True)

    # Convert datetime to UTC and remove timezone information
    frame[t_coord] = frame[t_coord].dt.tz_convert("utc").dt.tz_localize(None)
    return frame


def groupby(
    frame: gpd.GeoDataFrame,
    x_coord: str,
    y_coord: str,
    t_coord: str,
    z_coord: str,
    use_z: bool,
    period: str | None,
    merge_method: MergeMethod_Map_T,
) -> gpd.GeoDataFrame:
    # Prepare groupby for efficiency
    # Need to remove timezone information. Xarray time does not use tz
    if period is not None:
        frame[t_coord] = frame[t_coord].dt.to_period(period).dt.start_time
    # Prepare attr legend
    ds_attrs: dict[str, dict[str, Any]] = {}
    # Prepare groupby indices
    frame[x_coord] = frame.geometry.x
    frame[y_coord] = frame.geometry.y
    group_index = [
        t_coord,
        y_coord,
        x_coord,
    ]
    if use_z:
        group_index.append(z_coord)

    # Excluding bands - bands excluded from aggregation
    excluding_bands = set(
        [
            x_coord,
            y_coord,
            t_coord,
            "geometry",
        ]
    )
    if use_z:
        if z_coord not in frame.columns:
            raise ValueError("No altitude column found but use_z expected")
        excluding_bands.add(z_coord)

    # Build categorical encoding
    non_numeric_bands = set()
    for band in frame.columns:
        if band not in excluding_bands and not pd.api.types.is_numeric_dtype(
            frame[band]
        ):
            excluding_bands.add(band)
            non_numeric_bands.add(band)
            update_attr_legend(
                ds_attrs,
                band,
                frame,
            )

    # Prepare aggregation method
    bands = [name for name in frame.columns if name not in excluding_bands]

    # band_map determines replacement strategy for each band when there is a conflict
    band_map = (
        {band: merge_method[band] for band in bands if band in merge_method}
        if isinstance(merge_method, dict)
        else {band: merge_method for band in bands}
    )
    for band in non_numeric_bands:
        band_map[band] = "replace"

    # Groupby + Aggregate
    if use_z:
        return frame.groupby(group_index).agg(band_map)
    # If don't use_z but z column is present -> Drop it
    grouped = frame.groupby(group_index).agg(band_map)
    if z_coord in frame.columns:
        grouped.drop(columns=[z_coord], inplace=True)
    return grouped
