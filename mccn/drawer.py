from __future__ import annotations

import abc
from typing import Any, cast

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike

from mccn._types import (
    DType_Map_T,
    Dtype_T,
    MergeMethod_Map_T,
    MergeMethod_T,
    Nodata_Map_T,
    Nodata_T,
)
from mccn.loader.utils import select_by_key


class Drawer(abc.ABC):
    def __init__(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        t_coords: np.ndarray,
        shape: tuple[int, int, int],
        dtype: Dtype_T = "float32",
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


DRAWERS: dict[MergeMethod_T, type[Drawer]] = {
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
        bands: set[str],
        x_dim: str = "x",
        y_dim: str = "y",
        t_dim: str = "t",
        dtype: DType_Map_T = "float32",
        nodata: Nodata_Map_T = 0,
        nodata_fallback: Nodata_T = 0,
        is_sorted: bool = False,
        merge_method: MergeMethod_Map_T = "replace",
        merge_method_fallback: MergeMethod_T = "replace",
    ) -> None:
        self.x_coords = self.sort_coord(x_coords, is_sorted)
        self.y_coords = self.sort_coord(y_coords, is_sorted)
        self.t_coords = self.sort_coord(t_coords, is_sorted)
        self.bands = bands
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.t_dim = t_dim
        # Cube parameters
        self.shape = (len(self.t_coords), len(self.x_coords), len(self.y_coords))
        self.dims = (self.t_dim, self.x_dim, self.y_dim)
        self.coords = {
            self.t_dim: self.t_coords,
            self.x_dim: self.x_coords,
            self.y_dim: self.y_coords,
        }
        self.dtype = dtype
        self.nodata = nodata
        self.nodata_fallback = nodata_fallback
        self.is_sorted = is_sorted
        self.merge_method = merge_method
        self.merge_method_fallback = merge_method_fallback
        self._drawers = self._init_drawers()

    def sort_coord(self, coords: np.ndarray, is_sorted: bool) -> np.ndarray:
        if not is_sorted:
            coords = np.sort(coords)
        return coords

    def _get_draw_handler(self, method: MergeMethod_T) -> type[Drawer]:
        if method in DRAWERS:
            return DRAWERS[method]
        raise KeyError(f"Invalid merge method: {method}")

    def _init_drawers(self) -> dict[str, Drawer]:
        drawers = {}
        for band in self.bands:
            method = select_by_key(band, self.merge_method, self.merge_method_fallback)
            handler = self._get_draw_handler(cast(MergeMethod_T, method))
            dtype = select_by_key(band, self.dtype, "float32")  # type: ignore[arg-type]
            nodata = select_by_key(band, self.nodata, self.nodata_fallback)
            drawers[band] = handler(
                self.x_coords, self.y_coords, self.t_coords, self.shape, dtype, nodata
            )
        return drawers

    def get_band_drawer(self, band: str) -> Drawer:
        if band not in self._drawers:
            raise KeyError(f"Uninitialised band: {band}")
        return self._drawers[band]

    def draw(self, t_value: Any, band: str, layer: np.ndarray) -> None:
        drawer = self.get_band_drawer(band)
        drawer.draw(t_value, layer)

    def build_cube(self, attrs: dict[str, Any]) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                band: (self.dims, self._drawers[band].data) for band in self.bands
            },
            coords=self.coords,
            attrs=attrs,
        )
