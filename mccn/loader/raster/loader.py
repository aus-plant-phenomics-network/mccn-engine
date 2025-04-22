from __future__ import annotations

import collections
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import odc.stac
import pandas as pd
import xarray as xr

from mccn._types import ParsedRaster
from mccn.loader.base import Loader
from mccn.loader.raster.config import RasterLoadConfig, set_groupby

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pystac
    from odc.geo.geobox import GeoBox
    from odc.stac._stac_load import GroupbyCallback

    from mccn.config import CubeConfig, FilterConfig, ProcessConfig


logger = logging.getLogger(__name__)


class RasterLoader(Loader[ParsedRaster]):
    """Loader for raster asset

    Is an adapter for odc.stac.load
    """

    def __init__(
        self,
        items: Sequence[ParsedRaster],
        filter_config: FilterConfig,
        cube_config: CubeConfig | None = None,
        process_config: ProcessConfig | None = None,
        load_config: RasterLoadConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.load_config = load_config if load_config else RasterLoadConfig()
        super().__init__(items, filter_config, cube_config, process_config, **kwargs)
        self.groupby = set_groupby(self.process_config.time_groupby)
        self.period = self.process_config.period

    def _load(self) -> xr.Dataset:
        band_map = groupby_bands(self.items)
        ds = []
        for band_info, band_items in band_map.items():
            try:
                item_ds = read_asset(
                    items=band_items,
                    geobox=self.filter_config.geobox,
                    bands=band_info,
                    x_coord=self.cube_config.x_coord,
                    y_coord=self.cube_config.y_coord,
                    t_coord=self.cube_config.t_coord,
                    raster_config=self.load_config,
                    period=self.period,
                    groupby=self.groupby,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Fail to load items: {[item.id for item in band_items]} with bands: {band_info}"
                ) from e
            item_ds = self.apply_process(item_ds, self.process_config)
            ds.append(item_ds)
        return xr.merge(ds)


def groupby_bands(
    items: Sequence[ParsedRaster],
) -> dict[tuple[str, ...], list[pystac.Item]]:
    """Partition items into groups based on bands that will be loaded to dc

    Items that have the same bands will be put under the same group - i.e loaded together

    Args:
        items (Sequence[ParsedRaster]): ParsedRaster item

    Returns:
        dict[tuple[str, ...], list[pystac.Item]]: mapping between bands and items
    """
    result = collections.defaultdict(list)
    for item in items:
        result[tuple(sorted(item.load_bands))].append(item.item)
    return result


def read_asset(
    items: Sequence[pystac.Item],
    bands: tuple[str, ...] | None,
    geobox: GeoBox | None,
    x_coord: str,
    y_coord: str,
    t_coord: str,
    groupby: str | GroupbyCallback,
    period: str | None,
    raster_config: RasterLoadConfig,
) -> xr.Dataset:
    ds = odc.stac.load(
        items,
        bands,
        geobox=geobox,
        groupby=groupby,
        **asdict(raster_config),
    )
    # NOTE: odc stac load uses odc.geo.xr.xr_coords to set dimension name
    # it either uses latitude/longitude or y/x depending on the underlying crs
    # so there is no proper way to know which one it uses aside from trying
    if "latitude" in ds.dims and "longitude" in ds.dims:
        ds = ds.rename({"longitude": x_coord, "latitude": y_coord})
    elif "x" in ds.dims and "y" in ds.dims:
        ds = ds.rename({"x": x_coord, "y": y_coord})
    if "time" in ds.dims:
        ds = ds.rename({"time": t_coord})
    if period is not None:
        ts = pd.DatetimeIndex(ds[t_coord].values)
        ds = ds.assign_coords({t_coord: ts.to_period(period).start_time})
    return ds
