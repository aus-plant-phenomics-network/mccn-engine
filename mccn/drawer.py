from __future__ import annotations

import abc
from typing import Any, Callable, Sequence, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import DTypeLike
from odc.geo.geobox import GeoBox

from mccn._types import (
    DType_Map_T,
    Dtype_T,
    MergeMethod_Map_T,
    MergeMethod_T,
    Nodata_Map_T,
    Nodata_T,
    Number_T,
)
from mccn.loader.utils import (
    coords_from_geobox,
    get_neighbor_mask,
    mask_aggregate,
    query_by_key,
    query_if_null,
)
from mccn.parser import ParsedItem


class Drawer(abc.ABC):
    def __init__(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        t_coords: np.ndarray,
        shape: tuple[int, int, int],
        dtype: Dtype_T = "float64",
        nodata: Nodata_T = 0,
        **kwargs: Any,
    ) -> None:
        # Set up xarray dimensions and shape
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.t_coords = t_coords
        self.shape = shape

        # Set up drawer parameters
        self.dtype = dtype
        self.nodata = nodata

        # Date index for quick query
        self.t_map = {value: index for index, value in enumerate(self.t_coords)}

        # Post init hooks
        self.data = self.alloc()
        self.__post_init__(kwargs)

    def _alloc(self, dtype: DTypeLike, fill_value: Nodata_T) -> np.ndarray:
        return np.full(shape=self.shape, fill_value=fill_value, dtype=dtype)

    def alloc(self) -> np.ndarray:
        return self._alloc(self.dtype, self.nodata)

    def t_index(self, t_value: Any) -> int:
        if t_value in self.t_map:
            return self.t_map[t_value]
        raise KeyError(f"Invalid time value: {t_value}")

    def __post_init__(self, kwargs: Any) -> None: ...

    def draw(self, t_value: Any, layer: np.ndarray) -> None:
        t_index = self.t_index(t_value)
        valid_mask = self.valid_mask(layer)
        nodata_mask = self.nodata_mask(t_index)
        self._draw(t_index, layer, valid_mask, nodata_mask)

    @abc.abstractmethod
    def _draw(
        self,
        index: int,
        layer: np.ndarray,
        valid_mask: Any,
        nodata_mask: Any,
    ) -> None:
        self.data[index][nodata_mask & valid_mask] = layer[nodata_mask & valid_mask]

    def nodata_mask(self, t_index: int) -> Any:
        return self.data[t_index] == self.nodata

    def valid_mask(self, layer: np.ndarray) -> Any:
        return (layer != self.nodata) & ~(np.isnan(layer))


class SumDrawer(Drawer):
    def _draw(
        self,
        index: int,
        layer: np.ndarray,
        valid_mask: Any,
        nodata_mask: Any,
    ) -> None:
        super()._draw(index, layer, valid_mask, nodata_mask)
        self.data[index][valid_mask & ~nodata_mask] += layer[valid_mask & ~nodata_mask]


class MinMaxDrawer(Drawer):
    def __init__(self, is_max: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.is_max = is_max
        self.op = np.maximum if is_max else np.minimum

    def _draw(
        self,
        index: int,
        layer: np.ndarray,
        valid_mask: Any,
        nodata_mask: Any,
    ) -> None:
        super()._draw(index, layer, valid_mask, nodata_mask)
        data = self.data[index]
        data = self.op(layer, data, out=data, where=valid_mask & ~nodata_mask)
        self.data[index] = data


class MinDrawer(MinMaxDrawer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(is_max=False, **kwargs)


class MaxDrawer(MinMaxDrawer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(is_max=True, **kwargs)


class MeanDrawer(Drawer):
    def __post_init__(self, kwargs: Any) -> None:
        self.count = self._alloc("int", 0)

    def _draw(
        self,
        index: int,
        layer: np.ndarray,
        valid_mask: Any,
        nodata_mask: Any,
    ) -> None:
        data = self.data[index]
        count = self.count[index]
        data[count > 0] = data[count > 0] * count[count > 0]
        data[nodata_mask & valid_mask] = layer[nodata_mask & valid_mask]
        data[~nodata_mask & valid_mask] += layer[~nodata_mask & valid_mask]
        count[valid_mask] += 1
        data[count > 0] = data[count > 0] / count[count > 0]
        self.count[index] = count
        self.data[index] = data


class ReplaceDrawer(Drawer):
    def _draw(
        self,
        index: int,
        layer: np.ndarray,
        valid_mask: Any,
        nodata_mask: Any,
    ) -> None:
        self.data[index][valid_mask] = layer[valid_mask]


DRAWERS: dict[MergeMethod_T | str, type[Drawer]] = {
    "mean": MeanDrawer,
    "max": MaxDrawer,
    "min": MinDrawer,
    "replace": ReplaceDrawer,
    "sum": SumDrawer,
}


class Canvas:
    def __init__(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        t_coords: np.ndarray,
        x_dim: str = "x",
        y_dim: str = "y",
        t_dim: str = "t",
        spatial_ref: xr.DataArray | None = None,
        spatial_ref_dim: str = "spatial_ref",
        dtype: DType_Map_T = None,
        dtype_fallback: Dtype_T = "float64",
        nodata: Nodata_Map_T = 0,
        nodata_fallback: Nodata_T = 0,
        is_sorted: bool = False,
        merge: MergeMethod_Map_T = None,
        merge_fallback: MergeMethod_T = "replace",
    ) -> None:
        self.spatial_ref = spatial_ref
        self.x_coords = self._sort_coord(x_coords, is_sorted)
        self.y_coords = self._sort_coord(y_coords, is_sorted)
        self.t_coords = self._sort_coord(t_coords, is_sorted)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.t_dim = t_dim
        self.spatial_ref_dim = spatial_ref_dim
        # Cube parameters
        self.shape = (len(self.t_coords), len(self.y_coords), len(self.x_coords))
        self.dims = (self.t_dim, self.y_dim, self.x_dim)
        self.coords = {
            self.y_dim: self.y_coords,
            self.x_dim: self.x_coords,
            self.spatial_ref_dim: self.spatial_ref,
            self.t_dim: self.t_coords,
        }
        self.dtype = dtype
        self.dtype_fallback = dtype_fallback
        self.nodata = nodata
        self.nodata_fallback = nodata_fallback
        self.is_sorted = is_sorted
        self.merge = merge
        self.merge_fallback = merge_fallback
        self._drawers: dict[str, Drawer] = {}

    def _sort_coord(self, coords: np.ndarray, is_sorted: bool) -> np.ndarray:
        if not is_sorted:
            coords = np.sort(coords)
        return coords

    def has_band(self, band: str) -> bool:
        return band in self._drawers

    def add_band(
        self,
        band: str,
        merge: MergeMethod_T | None = None,
        dtype: Dtype_T | None = None,
        nodata: Nodata_T | None = None,
    ) -> None:
        if band not in self._drawers:
            _method = query_if_null(merge, band, self.merge, self.merge_fallback)
            _nodata = query_if_null(nodata, band, self.nodata, self.nodata_fallback)
            _dtype = cast(
                Dtype_T, query_if_null(dtype, band, self.dtype, self.dtype_fallback)
            )
            handler = DRAWERS[_method]
            self._drawers[band] = handler(
                self.x_coords,
                self.y_coords,
                self.t_coords,
                self.shape,
                _dtype,
                _nodata,
            )

    def draw(self, t_value: Any, band: str, layer: np.ndarray) -> None:
        if band not in self._drawers:
            raise KeyError(f"Unallocated band: {band}")
        drawer = self._drawers[band]
        drawer.draw(t_value, layer)

    def compile(self, attrs: dict[str, Any]) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                band: (self.dims, drawer.data) for band, drawer in self._drawers.items()
            },
            coords=self.coords,
            attrs=attrs,
        )

    @classmethod
    def from_items(
        cls,
        items: Sequence[ParsedItem],
        x_dim: str,
        y_dim: str,
        t_dim: str,
        spatial_ref_dim: str,
        geobox: GeoBox,
        period: str | None,
        dtype: DType_Map_T,
        dtype_fallback: Dtype_T,
        nodata: Nodata_Map_T,
        nodata_fallback: Nodata_T,
        merge: MergeMethod_Map_T,
        merge_fallback: MergeMethod_T,
    ) -> Canvas:
        coords = coords_from_geobox(geobox, x_dim, y_dim)
        x_coords = coords[x_dim].values
        y_coords = coords[y_dim].values
        spatial_ref = coords[spatial_ref_dim]
        # Build t_coords
        timestamps = []
        for item in items:
            timestamps.extend(item.item.properties["timestamps"])
        # Remove tzinfo - everything should be utc by default
        time_index = pd.Series(pd.to_datetime(timestamps).tz_localize(None))
        # Convert to period for groupby
        if period is not None:
            time_index = time_index.dt.to_period(period).dt.start_time
        t_coords = np.sort(time_index.unique())

        return Canvas(
            x_coords,
            y_coords,
            t_coords,
            x_dim,
            y_dim,
            t_dim,
            spatial_ref,
            spatial_ref_dim,
            dtype,
            dtype_fallback,
            nodata,
            nodata_fallback,
            True,
            merge,
            merge_fallback,
        )


class Rasteriser:
    def __init__(
        self,
        canvas: Canvas,
        radius: Number_T = 1.0,
        categorical_prefix: str = "__cat_",
    ) -> None:
        self.attrs: dict[str, dict[int, Any]] = {}
        self.rev_attrs: dict[str, dict[Any, int]] = {}
        self.canvas = canvas
        self.x_dim = canvas.x_dim
        self.y_dim = canvas.y_dim
        self.t_dim = canvas.t_dim
        self.nodata = canvas.nodata
        self.nodata_fallback = canvas.nodata_fallback
        self.dims = (self.x_dim, self.y_dim)
        self.gx = self.canvas.x_coords
        self.gy = self.canvas.y_coords
        self.radius = radius
        self.categorical_prefix = categorical_prefix

    def is_categorical(self, series: pd.Series) -> bool:
        return not pd.api.types.is_numeric_dtype(series)

    def encode(self, series: pd.Series, nodata: int, band: str) -> pd.Series:
        curr = 0 if nodata != 0 else 1
        if band not in self.attrs:
            self.attrs[band] = {nodata: "nodata"}
            self.rev_attrs[band] = {"nodata": nodata}
        # Update attr map and rev attrs map
        for name in series.unique():
            if curr == nodata:
                curr += 1
            if name not in self.rev_attrs[band]:
                self.attrs[band][curr] = name
                self.rev_attrs[band][name] = curr
                curr += 1
        # Attr dict - mapped value -> original
        series = series.map(self.rev_attrs[band])
        return series

    def handle_categorical(self, series: pd.Series, band: str) -> tuple[pd.Series, str]:
        nodata = int(query_by_key(band, self.nodata, self.nodata_fallback))
        band = self.categorical_prefix + band
        series = self.encode(series, nodata, band)
        if not self.canvas.has_band(band):
            self.canvas.add_band(band, merge="replace", dtype="int8", nodata=nodata)
        return series, band

    def handle_numeric(self, series: pd.Series, band: str) -> tuple[pd.Series, str]:
        if not self.canvas.has_band(band):
            self.canvas.add_band(band)
        return series, band

    def rasterise_band(
        self,
        data: gpd.GeoDataFrame,
        band: str,
    ) -> None:
        for date in data[self.t_dim].unique():
            series = data.loc[data[self.t_dim] == date, [*self.dims, band]]
            series = series.drop_duplicates().dropna()
            dims = series.loc[:, [*self.dims]].values
            band_series = series.loc[:, band]
            if self.is_categorical(band_series):
                band_series, band = self.handle_categorical(band_series, band)
                op: Callable = np.nanmax
            else:
                band_series, band = self.handle_numeric(band_series, band)
                op = np.nanmean
            mask = get_neighbor_mask(self.gx, self.gy, dims, self.radius)
            raster = mask_aggregate(band_series.values, mask, op)
            self.canvas.draw(date, band, raster)

    def rasterise(self, data: pd.DataFrame, bands: set[str]) -> None:
        for band in bands:
            self.rasterise_band(data, band)

    def compile(self) -> xr.Dataset:
        return self.canvas.compile(self.attrs)
