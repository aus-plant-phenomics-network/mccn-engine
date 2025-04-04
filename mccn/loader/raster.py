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
    """Load config for raster asset. Parameters come from odc.stac.load"""

    resampling: str | dict[str, str] | None = None
    chunks: dict[str, int | Literal["auto"]] | None = None
    pool: ThreadPoolExecutor | int | None = None
    dtype: DTypeLike | Mapping[str, DTypeLike] = None


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

    def _load(self) -> xr.Dataset:
        band_map = groupby_bands(self.items)
        ds = []
        for band_info, band_items in band_map.items():
            try:
                item_ds = read_raster_asset(
                    band_items,
                    self.filter_config.geobox,
                    bands=band_info,
                    x_col=self.cube_config.x_coord,
                    y_col=self.cube_config.y_coord,
                    t_col=self.cube_config.t_coord,
                    resampling=self.load_config.resampling,
                    chunks=self.load_config.chunks,
                    pool=self.load_config.pool,
                    dtype=self.load_config.dtype,
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


def read_raster_asset(
    items: Sequence[pystac.Item],
    geobox: GeoBox | None,
    bands: str | Sequence[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
    resampling: str | dict[str, str] | None = None,
    chunks: dict[str, int | Literal["auto"]] | None = None,
    pool: ThreadPoolExecutor | int | None = None,
    dtype: DTypeLike | Mapping[str, DTypeLike] = None,
) -> xr.Dataset:
    """Wrapper for odc.stac.load

    Also perform cube axis renaming for consistency

    Args:
        items (Sequence[pystac.Item]): sequence of items to be loaded
        geobox (GeoBox | None): target geobox
        bands (str | Sequence[str] | None, optional): bands to load. Defaults to None.
        x_col (str, optional): name of x dimension of cube. Defaults to "x".
        y_col (str, optional): name of y dimension of cube. Defaults to "y".
        t_col (str, optional): name of time dimension of cube. Defaults to "time".
        resampling (str | Mapping[str, str] | None, optional): resampling method from odc.stac.load . Defaults to None.
        chunks (Mapping[str, int  |  Literal[&quot;auto&quot;]] | None, optional): chunk parameter from odc.stac.load . Defaults to None.
        pool (ThreadPoolExecutor | int | None, optional): pool paramter from odc.stac.load. Defaults to None.
        dtype (DTypeLike | Mapping[str, DTypeLike], optional): dtype from odc.stac.load. Defaults to None.

    Returns:
        xr.Dataset: loaded dataset
    """
    ds = odc.stac.load(
        items,
        bands,
        geobox=geobox,
        resampling=resampling,
        chunks=chunks,
        pool=pool,
        dtype=dtype,
    )
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
