from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

import geopandas as gpd
import xarray as xr
from stac_generator.core.base.utils import read_point_asset

from mccn._types import CRS_T
from mccn.drawer import Canvas, Rasteriser
from mccn.loader.base import Loader
from mccn.loader.point.config import PointLoadConfig
from mccn.parser import ParsedPoint

if TYPE_CHECKING:
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
        self.canvas = Canvas.from_items(
            self.items,
            self.cube_config.x_dim,
            self.cube_config.y_dim,
            self.cube_config.t_dim,
            self.cube_config.spatial_ref_dim,
            self.filter_config.geobox,
            self.process_config.period,
            self.process_config.dtype,
            self.process_config.dtype_fallback,
            self.process_config.nodata,
            self.process_config.nodata_fallback,
            self.process_config.merge_method,
            self.process_config.merge_method_fallback,
        )
        self.rasteriser = Rasteriser(self.canvas, self.load_config.radius)

    def _load(self) -> xr.Dataset:
        if not self.items:
            return xr.Dataset()
        for item in self.items:
            df = read_asset(
                item,
                self.cube_config.x_dim,
                self.cube_config.y_dim,
                self.cube_config.t_dim,
                self.cube_config.z_dim,
                cast(CRS_T, self.filter_config.geobox.crs),
            )
            df = self.apply_process(df, self.process_config)
            self.rasteriser.rasterise(df, item.load_bands)
        return self.rasteriser.compile()


def read_asset(
    item: ParsedPoint,
    x_dim: str,
    y_dim: str,
    t_dim: str,
    z_dim: str,
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
        rename_dict[item.config.T] = t_dim
    else:  # If point data does not contain date - set datecol using item datetime
        frame[t_dim] = item.item.datetime
    if item.config.Z:
        rename_dict[item.config.Z] = z_dim
    # Drop X and Y columns since we will repopulate them after changing crs
    frame.drop(columns=[item.config.X, item.config.Y], inplace=True)
    # Convert to geobox crs
    frame = frame.to_crs(crs)
    # Rename indices
    frame.rename(columns=rename_dict, inplace=True)
    # X, Y columns:
    frame[x_dim] = frame.geometry.x
    frame[y_dim] = frame.geometry.y
    # Convert datetime to UTC and remove timezone information
    frame[t_dim] = frame[t_dim].dt.tz_convert("utc").dt.tz_localize(None)
    return frame
