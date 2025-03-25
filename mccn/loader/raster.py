from __future__ import annotations

import collections
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Mapping

import odc.stac
import pystac
import xarray as xr
from numpy.typing import DTypeLike

from mccn.loader.base import CubeConfig, FilterConfig, Loader, ProcessConfig
from mccn.parser import ParsedItem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odc.geo.geobox import GeoBox


logger = logging.getLogger(__name__)


@dataclass
class RasterLoadConfig:
    resampling: str | Mapping[str, str] | None = None
    chunks: Mapping[str, int | Literal["auto"]] | None = None
    pool: ThreadPoolExecutor | int | None = None
    dtype: DTypeLike | Mapping[str, DTypeLike] = None


class RasterLoader(Loader):
    def __init__(
        self,
        items: list[ParsedItem],
        filter_config: FilterConfig,
        cube_config: CubeConfig | None = None,
        load_config: RasterLoadConfig | None = None,
        **kwargs,
    ):
        self.load_config = load_config if load_config else RasterLoadConfig()
        super().__init__(items, filter_config, cube_config, **kwargs)

    @staticmethod
    def groupby_bands(items: list[ParsedItem]) -> dict[set[str], list[pystac.Item]]:
        result = collections.defaultdict(list)
        for item in items:
            result[item.load_bands].append(item)
        return result

    def load(self) -> xr.Dataset:
        band_map = self.groupby_bands(self.items)
        ds = []
        for band_info, band_items in band_map.items():
            ds.append(
                _odc_load_wrapper(
                    band_items,
                    self.filter_config.geobox,
                    bands=band_info,
                    x_col=self.cube_config.x_coord,
                    y_col=self.cube_config.y_coord,
                    z_col=self.cube_config.z_coord,
                )
            )
        return xr.merge(ds, compat="equals")


def _odc_load_wrapper(
    items: Sequence[pystac.Item],
    geobox: GeoBox | None,
    bands: str | Sequence[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
    process_bands: dict[str, Callable] | None = None,
    rename_bands: dict[str, str] | None = None,
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
    # Process variable
    if process_bands:
        for k, fn in process_bands.items():
            if k in ds.data_vars.keys():
                ds[k] = xr.apply_ufunc(fn, ds[k])
    # Rename variable
    if rename_bands and set(rename_bands.keys()) & set(ds.data_vars.keys()):
        ds = ds.rename_vars(rename_bands)
    return ds
