from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, cast

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.xr import xr_coords
from stac_generator.core.point.generator import read_csv

from mccn._types import ParsedPoint
from mccn.loader.base import Loader

if TYPE_CHECKING:
    from mccn._types import (
        CubeConfig,
        FilterConfig,
        InterpMethods,
        MergeMethods,
        ProcessConfig,
    )


@dataclass
class PointLoadConfig:
    """Point load config - determines how point data should be aggregated and interpolated"""

    nodata: int | float = 0
    """No data value"""
    interp: InterpMethods | None = "nearest"
    """Interpolation Mode"""
    agg_method: MergeMethods = "mean"
    """Merge method for aggregation"""


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
        self.attr_map = {}
        super().__init__(items, filter_config, cube_config, process_config, **kwargs)

    def _load(self) -> xr.Dataset:
        frames = []
        for item in self.items:
            frames.append(self.load_item(item))
        return xr.merge(frames)

    def load_item(self, item: ParsedPoint) -> xr.Dataset:
        # Read csv
        frame = read_csv(
            src_path=item.location,
            X_coord=item.config.X,
            Y_coord=item.config.Y,
            epsg=cast(int, item.crs.to_epsg()),
            T_coord=item.config.T,
            date_format=item.config.date_format,
            Z_coord=item.config.Z,
            columns=list(item.load_bands),
        )
        # Prepare rename dict
        rename_dict = {}
        if item.config.T:
            rename_dict[item.config.T] = self.cube_config.t_coord
        else:  # If point data does not contain date - set datecol using item datetime
            frame[self.cube_config.t_coord] = item.item.datetime
        if item.config.Z:
            rename_dict[item.config.Z] = self.cube_config.z_coord

        # Apply transformation
        frame = self.apply_process(frame, self.process_config)

        # Rename indices
        frame.rename(columns=rename_dict, inplace=True)
        # Drop X and Y columns since we will repopulate them after changing crs
        frame.drop(columns=[item.config.X, item.config.Y], inplace=True)

        # Convert to geobox crs
        frame = frame.to_crs(self.filter_config.geobox.crs)
        # Process groupby - i.e. average out over depth, duplicate entries, etc
        merged = self.groupby(
            frame=frame,
            item=item,
        )
        ds = self.to_xarray(merged)
        # Fill nodata
        ds = ds.fillna(self.load_config.nodata)
        ds.attrs = self.attr_map
        return ds

    def groupby(
        self,
        frame: gpd.GeoDataFrame,
        item: ParsedPoint,
    ) -> gpd.GeoDataFrame:
        frame[self.cube_config.x_coord] = frame.geometry.x
        frame[self.cube_config.y_coord] = frame.geometry.y
        group_index = [
            self.cube_config.t_coord,
            self.cube_config.y_coord,
            self.cube_config.x_coord,
        ]
        if self.cube_config.use_z:
            group_index.append(self.cube_config.z_coord)

        # Excluding bands - bands excluded from aggregation
        excluding_bands = set(
            [
                self.cube_config.x_coord,
                self.cube_config.y_coord,
                self.cube_config.t_coord,
                "geometry",
            ]
        )
        if self.cube_config.use_z:
            if self.cube_config.z_coord not in frame.columns:
                raise ValueError("No altitude column found but use_z expected")
            excluding_bands.add(self.cube_config.z_coord)

        # Build categorical encoding
        non_numeric_bands = set()
        for band in frame.columns:
            if band not in excluding_bands and not pd.api.types.is_numeric_dtype(
                frame[band]
            ):
                excluding_bands.add(band)
                non_numeric_bands.add(band)
                # Build attr map and categorically encode data
                if band not in self.attr_map:
                    self.attr_map[band] = {}
                curr_idx = (
                    max(self.attr_map[band].values()) + 1
                    if self.attr_map[band].values()
                    else 1
                )
                uniques = {k: 1 for k in frame[band].unique()}
                for key in uniques:
                    if key not in self.attr_map[band]:
                        self.attr_map[band][key] = curr_idx
                        curr_idx += 1
                    uniques[key] = self.attr_map[band][key]
                frame[band] = frame[band].map(uniques)

        # Prepare aggregation method
        bands = [name for name in frame.columns if name not in excluding_bands]

        # band_map determines replacement strategy for each band when there is a conflict
        band_map = (
            {
                band: self.load_config.agg_method[band]
                for band in bands
                if band in self.load_config.agg_method
            }
            if isinstance(self.load_config.agg_method, dict)
            else {band: self.load_config.agg_method for band in bands}
        )
        for band in non_numeric_bands:
            band_map[band] = "first"

        # Groupby + Aggregate
        if self.cube_config.use_z:
            return frame.groupby(group_index).agg(band_map)
        # If don't use_z but z column is present -> Drop it
        grouped = frame.groupby(group_index).agg(band_map)
        if item.config.Z:
            grouped.drop(columns=[self.cube_config.z_coord], inplace=True)
        return grouped

    def to_xarray(self, frame: pd.DataFrame) -> xr.Dataset:
        ds: xr.Dataset = frame.to_xarray()
        # Sometime to_xarray bugs out with datetime index so need explicit conversion
        ds[self.cube_config.t_coord] = pd.DatetimeIndex(
            ds.coords[self.cube_config.t_coord].values
        )
        # For debugging purpose
        if self.load_config.interp is None:
            return ds

        coords_ = xr_coords(
            self.filter_config.geobox,
            dims=(self.cube_config.y_coord, self.cube_config.x_coord),
        )

        ds = ds.assign_coords(spatial_ref=coords_["spatial_ref"])
        coords = {
            self.cube_config.y_coord: coords_[self.cube_config.y_coord],
            self.cube_config.x_coord: coords_[self.cube_config.x_coord],
        }
        return ds.interp(coords=coords, method=self.load_config.interp)
