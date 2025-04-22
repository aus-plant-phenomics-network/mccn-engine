from __future__ import annotations

import abc
from typing import Any, Mapping

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike

from mccn._types import Number_T
from mccn.loader.utils import select_by_key


class Drawer(abc.ABC):
    def __init__(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        t_coords: np.ndarray,
        bands: set[str],
        x_dim: str = "x",
        y_dim: str = "y",
        t_dim: str = "t",
        dtype: Mapping[str, DTypeLike] | DTypeLike = "float32",
        nodata: Number_T | Mapping[str, Number_T] = 0,
        nodata_fallback: Number_T = 0,
        is_sorted: bool = False,
        **kwargs: Any,
    ) -> None:
        # Set up xarray dimensions and shape
        self.x_coords = self.sort_coord(x_coords, is_sorted)
        self.y_coords = self.sort_coord(y_coords, is_sorted)
        self.t_coords = self.sort_coord(t_coords, is_sorted)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.t_dim = t_dim
        self.dims = (self.t_dim, self.x_dim, self.y_dim)
        self.coords = {
            self.t_dim: self.t_coords,
            self.x_dim: self.x_coords,
            self.y_dim: self.y_coords,
        }
        self.shape = (len(self.t_coords), len(self.x_coords), len(self.y_coords))

        # Set up drawer parameters
        self.bands = bands
        self.dtype = dtype
        self.nodata = nodata
        self.nodata_fallback = nodata_fallback

        # Set up attributes
        self.t_map = {value: index for index, value in enumerate(self.t_coords)}
        self.attrs: dict[str, Any] = {}

        # Post init hooks
        self.data = self.alloc_layers()
        self.__post_init__(kwargs)

    def sort_coord(self, coords: np.ndarray, is_sorted: bool) -> np.ndarray:
        if not is_sorted:
            coords = np.sort(coords)
        return coords

    def alloc_layer(self, dtype: DTypeLike, nodata: Number_T) -> np.ndarray:
        return np.full(
            shape=self.shape,
            fill_value=nodata,
            dtype=dtype,
        )

    def alloc_layers(self) -> dict[str, np.ndarray]:
        return {
            band: self.alloc_layer(
                dtype=select_by_key(band, self.dtype, "float32"),  # type: ignore[arg-type]
                nodata=select_by_key(band, self.nodata, self.nodata_fallback),
            )
            for band in self.bands
        }

    def update_attrs(self, attrs: dict[str, Any]) -> None:
        self.attrs.update(attrs)

    def t_index(self, t_value: Any) -> int:
        if t_value in self.t_map:
            return self.t_map[t_value]
        raise KeyError(f"Invalid time value: {t_value}")

    def __post_init__(self, kwargs: Any) -> None: ...

    def draw(
        self, t_value: Any, band: str, layer: np.ndarray, is_categorical: bool
    ) -> None:
        t_index = self.t_index(t_value)
        if not is_categorical:
            nodata = select_by_key(band, self.nodata, self.nodata_fallback)
            self._draw(t_index, band, layer, nodata)
        else:
            self.replace(t_index, band, layer)

    @abc.abstractmethod
    def _draw(
        self, t_index: int, band: str, layer: np.ndarray, nodata: Number_T
    ) -> None:
        raise NotImplementedError

    def replace(self, t_index: int, band: str, layer: np.ndarray) -> None:
        mask = (
            layer != select_by_key(band, self.nodata, self.nodata_fallback)
        ) & ~np.isnan(layer)
        self.data[band][t_index][mask] = layer[mask]

    def build(self) -> xr.Dataset:
        return xr.Dataset(
            data_vars={key: (self.dims, value) for key, value in self.data.items()},
            coords=self.coords,
            attrs=self.attrs,
        )


class SumDrawer(Drawer):
    def _draw(
        self, t_index: int, band: str, layer: np.ndarray, nodata: Number_T
    ) -> None:
        data = self.data[band][t_index]
        layer_mask = (layer != nodata) & ~np.isnan(layer)
        nodata_mask = data == nodata
        data_mask = ~nodata_mask
        data[nodata_mask & layer_mask] = layer[nodata_mask & layer_mask]
        data[data_mask & layer_mask] += layer[data_mask & layer_mask]
        self.data[band][t_index] = data


class MinMaxDrawer(Drawer):
    def __init__(self, is_max: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.is_max = is_max
        self.op = np.maximum if is_max else np.minimum

    def _draw(
        self, t_index: int, band: str, layer: np.ndarray, nodata: Number_T
    ) -> None:
        old_layer = self.data[band][t_index]
        overwrite_mask = (old_layer == nodata) & ~np.isnan(layer) & (layer != nodata)
        old_layer = np.where(overwrite_mask, layer, old_layer)
        op_mask = (layer != nodata) & (old_layer != nodata) & ~np.isnan(layer)
        self.data[band][t_index] = self.op(
            layer, old_layer, out=old_layer, where=op_mask
        )


class MinDrawer(MinMaxDrawer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(is_max=False, **kwargs)


class MaxDrawer(MinMaxDrawer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(is_max=True, **kwargs)


class MeanDrawer(Drawer):
    def __post_init__(self, kwargs: Any) -> None:
        self.count = {band: self.alloc_layer("int", 0) for band in self.bands}

    def _draw(
        self, t_index: int, band: str, layer: np.ndarray, nodata: Number_T
    ) -> None:
        data = self.data[band][t_index]
        count = self.count[band][t_index]
        data[count > 0] = data[count > 0] * count[count > 0]

        mask = (layer != nodata) & ~(np.isnan(layer))
        data_mask = data != nodata
        nodata_mask = ~data_mask
        data[nodata_mask & mask] = layer[nodata_mask & mask]
        data[data_mask & mask] += layer[data_mask & mask]
        count[mask] += 1
        data[count > 0] = data[count > 0] / count[count > 0]
        self.count[band][t_index] = count
        self.data[band][t_index] = data


class ReplaceDrawer(Drawer):
    def _draw(
        self, t_index: int, band: str, layer: np.ndarray, nodata: Number_T
    ) -> None:
        self.replace(t_index, band, layer)


DRAWERS = {
    "mean": MeanDrawer,
    "max": MaxDrawer,
    "min": MinDrawer,
    "replace": ReplaceDrawer,
    "sum": SumDrawer,
}
