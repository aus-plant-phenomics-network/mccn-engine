from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, cast

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.xr import xr_coords
from stac_generator.core.point.generator import read_csv

from mccn.loader.base import CubeConfig, FilterConfig, Loader
from mccn.parser import ParsedPoint

if TYPE_CHECKING:
    from mccn._types import InterpMethods, MergeMethods


@dataclass
class PointLoadConfig:
    interp: InterpMethods | None = None
    agg_method: MergeMethods = "mean"


class PointLoader(Loader[ParsedPoint]):
    def __init__(
        self,
        items: Sequence[ParsedPoint],
        filter_config: FilterConfig,
        cube_config: CubeConfig | None = None,
        load_config: PointLoadConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.load_config = load_config if load_config else PointLoadConfig()
        super().__init__(items, filter_config, cube_config, **kwargs)

    def load(self) -> xr.Dataset:
        frames = []
        for item in self.items:
            frames.append(self.load_item(item))
        return xr.merge(frames)

    def load_item(self, item: ParsedPoint) -> xr.Dataset:
        # Load item to gdf
        frame = self.load_asset(
            item=item,
            cube_config=self.cube_config,
        )
        # Convert to geobox crs
        frame = frame.to_crs(self.filter_config.geobox.crs)
        # Process groupby - i.e. average out over depth, duplicate entries, etc
        merged = self.groupby(
            frame=frame,
            cube_config=self.cube_config,
            load_config=self.load_config,
        )
        return self.to_xarray(
            merged,
            self.filter_config,
            self.cube_config,
            self.load_config,
        )

    @staticmethod
    def load_asset(
        item: ParsedPoint,
        cube_config: CubeConfig,
    ) -> gpd.GeoDataFrame:
        config = item.config
        location = item.location
        columns = item.load_bands
        crs = item.crs
        # Read csv
        gdf = read_csv(
            src_path=location,
            X_coord=config.X,
            Y_coord=config.Y,
            epsg=cast(int, crs.to_epsg()),
            T_coord=config.T,
            date_format=config.date_format,
            Z_coord=config.Z,
            columns=columns,
        )
        # Prepare rename dict for indices
        rename_dict = {}
        if config.T:
            rename_dict[config.T] = cube_config.t_coord
        else:  # If point data does not contain date - set datecol using item datetime
            gdf[cube_config.t_coord] = item.item.datetime
        if config.Z:
            rename_dict[config.Z] = cube_config.z_coord
        # Rename indices
        gdf.rename(columns=rename_dict, inplace=True)
        # Drop X and Y columns since we will repopulate them after changing crs
        gdf.drop(columns=[config.X, config.Y], inplace=True)
        return gdf

    @staticmethod
    def groupby(
        frame: gpd.GeoDataFrame,
        cube_config: CubeConfig,
        load_config: PointLoadConfig,
    ) -> gpd.GeoDataFrame:
        frame[cube_config.x_coord] = frame.geometry.x
        frame[cube_config.y_coord] = frame.geometry.y

        # Prepare aggregation method
        excluding_bands = set(
            [cube_config.x_coord, cube_config.y_coord, cube_config.t_coord, "geometry"]
        )
        if cube_config.use_z:
            if cube_config.z_coord not in frame.columns:
                raise ValueError("No altitude column found but use_z expected")
            excluding_bands.add(cube_config.z_coord)

        bands = [name for name in frame.columns if name not in excluding_bands]

        # band_map determines replacement strategy for each band when there is a conflict
        band_map = (
            {
                band: load_config.agg_method[band]
                for band in bands
                if band in load_config.agg_method
            }
            if isinstance(load_config.agg_method, dict)
            else {band: load_config.agg_method for band in bands}
        )

        # Groupby + Aggregate
        if cube_config.use_z:
            return frame.groupby(
                [
                    cube_config.t_coord,
                    cube_config.y_coord,
                    cube_config.x_coord,
                    cube_config.z_coord,
                ]
            ).agg(band_map)
        return frame.groupby(
            [cube_config.t_coord, cube_config.y_coord, cube_config.x_coord]
        ).agg(band_map)

    @staticmethod
    def to_xarray(
        frame: pd.DataFrame,
        filter_config: FilterConfig,
        cube_config: CubeConfig,
        load_config: PointLoadConfig,
    ) -> xr.Dataset:
        ds: xr.Dataset = frame.to_xarray()
        # Sometime to_xarray bugs out with datetime index so need explicit conversion
        ds[cube_config.t_coord] = pd.DatetimeIndex(
            ds.coords[cube_config.t_coord].values
        )
        if load_config.interp is None:
            return ds

        coords_ = xr_coords(
            filter_config.geobox, dims=(cube_config.y_coord, cube_config.x_coord)
        )

        ds = ds.assign_coords(spatial_ref=coords_["spatial_ref"])
        coords = {
            cube_config.y_coord: coords_[cube_config.y_coord],
            cube_config.x_coord: coords_[cube_config.x_coord],
        }
        return ds.interp(coords=coords, method=load_config.interp)
