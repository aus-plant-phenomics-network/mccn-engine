from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping

import odc.stac
import xarray as xr

from mccn._types import ParsedRaster
from mccn.loader.base import Loader

if TYPE_CHECKING:
    from collections.abc import Sequence
    from concurrent.futures import ThreadPoolExecutor

    import pystac
    from numpy.typing import DTypeLike
    from odc.geo.geobox import GeoBox

    from mccn._types import CubeConfig, FilterConfig, ProcessConfig


logger = logging.getLogger(__name__)


@dataclass
class RasterLoadConfig:
    resampling: str | Mapping[str, str] | None = None
    chunks: Mapping[str, int | Literal["auto"]] | None = None
    pool: ThreadPoolExecutor | int | None = None
    dtype: DTypeLike | Mapping[str, DTypeLike] = None


class RasterLoader(Loader[ParsedRaster]):
    def __init__(
        self,
        items: Sequence[ParsedRaster],
        filter_config: FilterConfig,
        cube_config: CubeConfig | None = None,
        load_config: RasterLoadConfig | None = None,
        process_config: ProcessConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.load_config = load_config if load_config else RasterLoadConfig()
        super().__init__(items, filter_config, cube_config, process_config, **kwargs)

    @staticmethod
    def groupby_bands(
        items: Sequence[ParsedRaster],
    ) -> dict[tuple[str, ...], list[pystac.Item]]:
        result = collections.defaultdict(list)
        for item in items:
            result[tuple(sorted(item.load_bands))].append(item.item)
        return result

    def load(self) -> xr.Dataset:
        band_map = self.groupby_bands(self.items)
        ds = []
        for band_info, band_items in band_map.items():
            item_ds = _odc_load_wrapper(
                band_items,
                self.filter_config.geobox,
                bands=band_info,
                x_col=self.cube_config.x_coord,
                y_col=self.cube_config.y_coord,
                t_col=self.cube_config.t_coord,
            )
            item_ds = self.apply_process(item_ds, self.process_config)
            ds.append(item_ds)
        return xr.merge(ds, compat="equals")


def _odc_load_wrapper(
    items: Sequence[pystac.Item],
    geobox: GeoBox | None,
    bands: str | Sequence[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
) -> xr.Dataset:
    ds = odc.stac.load(items, bands, geobox=geobox)
    # NOTE: odc stac load uses odc.geo.xr.xr_coords to set dimension name
    # it either uses latitude/longitude or y/x depending on the underlying crs
    # so there is no proper way to know which one it uses aside from trying
    if "latitude" in ds.dims and "longitude" in ds.dims:
        ds = ds.rename({"longitude": x_col, "latitude": y_col})
    elif "x" in ds.dims and "y" in ds.dims:
        ds = ds.rename({"x": x_col, "y": y_col})
    if "time" in ds.dims:
        ds = ds.rename({"time": t_col})
    return ds
