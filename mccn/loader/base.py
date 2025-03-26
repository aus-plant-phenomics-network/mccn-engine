from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar, overload

import pandas as pd
import xarray as xr

from mccn._types import CubeConfig, ParsedItem, ProcessConfig

if TYPE_CHECKING:
    from mccn._types import FilterConfig

T = TypeVar("T", bound=ParsedItem)


class Loader(abc.ABC, Generic[T]):
    def __init__(
        self,
        items: Sequence[T],
        filter_config: FilterConfig,
        cube_config: CubeConfig | None = None,
        process_config: ProcessConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.items = items
        self.filter_config = filter_config
        self.cube_config = cube_config if cube_config else CubeConfig()
        self.process_config = process_config if process_config else ProcessConfig()

    @abc.abstractmethod
    def load(self) -> xr.Dataset:
        raise NotADirectoryError

    @overload
    @staticmethod
    def apply_process(
        data: pd.DataFrame, process_config: ProcessConfig
    ) -> pd.DataFrame: ...

    @overload
    @staticmethod
    def apply_process(
        data: xr.Dataset, process_config: ProcessConfig
    ) -> xr.Dataset: ...

    @staticmethod
    def apply_process(
        data: pd.DataFrame | xr.Dataset, process_config: ProcessConfig
    ) -> pd.DataFrame | xr.Dataset:
        if isinstance(data, pd.DataFrame):
            # Transform
            if process_config.process_bands:
                for key, fn in process_config.process_bands.items():
                    if key in data.columns:
                        data[key] = data[key].apply(fn)
            # Rename bands
            if process_config.rename_bands:
                data.rename(columns=process_config.rename_bands, inplace=True)
            return data
        if isinstance(data, xr.Dataset):
            # Process variable
            if process_config.process_bands:
                for k, fn in process_config.process_bands.items():
                    if k in data.data_vars.keys():
                        data[k] = xr.apply_ufunc(fn, data[k])
            # Rename variable
            if process_config.rename_bands and set(
                process_config.rename_bands.keys()
            ) & set(data.data_vars.keys()):
                data = data.rename_vars(process_config.rename_bands)
            return data
