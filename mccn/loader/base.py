from __future__ import annotations

import abc
from typing import (
    Any,
    Generic,
    Mapping,
    Sequence,
    TypeVar,
)

import pandas as pd
import xarray as xr

from mccn.config import CubeConfig, FilterConfig, ProcessConfig
from mccn.drawer import Rasteriser
from mccn.loader.utils import query_by_key
from mccn.parser import ParsedItem

T = TypeVar("T", bound=ParsedItem)


class Loader(abc.ABC, Generic[T]):
    def __init__(
        self,
        items: Sequence[T],
        rasteriser: Rasteriser,
        filter_config: FilterConfig,
        cube_config: CubeConfig | None = None,
        process_config: ProcessConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Base STAC Item loader class

        Produces an xarray Dataset with layer (variables) of dimension (time, y, x) or (time, y, x, z) if a valid altitude
        dimension is found. Loader is further divided into Point, Vector and Raster for each data type.

        Config parameters:
        - filter_config: item filter parameters - date time, bands, geobox
        - cube_config: cube dimension parameters - name of x, y, z, t coordinates, whether to use z axis
        - process_config: preprocessing parameters - whether to rename a band or transform a band prior to loading

        Supports the following methods
        - load: load valid items to an xarray Dataset
        - apply_processing: process the load cube based on parameters from process_config

        Args:
            items (Sequence[T]): parsedItems
            filter_config (FilterConfig): datacube filter config
            cube_config (CubeConfig | None, optional): datacube dimension config. Defaults to None.
            process_config (ProcessConfig | None, optional): data cube processing config. Defaults to None.
        """
        self.items = items
        self.rasteriser = rasteriser
        self.filter_config = filter_config
        self.cube_config = cube_config if cube_config else CubeConfig()
        self.process_config = process_config if process_config else ProcessConfig()
        self.__post_init__()

    def __post_init__(self) -> None: ...

    @abc.abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    def apply_filter(
        self,
        data: xr.Dataset | pd.DataFrame,
        filter_config: FilterConfig,
        cube_config: CubeConfig,
        is_frame: bool,
    ) -> xr.Dataset:
        # Filter based on dates and geobox
        if not is_frame:
            data = data.sel(
                {
                    cube_config.t_dim: slice(
                        filter_config.start_no_tz, filter_config.end_no_tz
                    )
                }
            )
        else:
            mask = pd.Series(True, index=data.index)
            if filter_config.start_no_tz:
                mask &= data[cube_config.t_dim] >= filter_config.start_no_tz
            if filter_config.end_no_tz:
                mask &= data[cube_config.t_dim] <= filter_config.end_no_tz
            data = data[mask]
        return data

    def apply_process(
        self,
        data: pd.DataFrame | xr.Dataset,
        bands: set[str] | Sequence[str],
    ) -> pd.DataFrame | xr.Dataset:
        if isinstance(data, pd.DataFrame):
            is_frame = True
        elif isinstance(data, xr.Dataset):
            is_frame = False
        else:
            raise ValueError(
                f"Expeting data to be a dataframe or a dataset: {type(data)}"
            )
        data = self.transform(data, is_frame)
        data = self.rename(data, is_frame)
        data = self.fillna(data, bands)
        data = self.apply_filter(data, self.filter_config, self.cube_config, is_frame)
        return data

    def rename(
        self,
        data: pd.DataFrame | xr.Dataset,
        is_frame: bool,
    ) -> pd.DataFrame | xr.Dataset:
        if self.process_config.rename_bands:
            rename_bands: Mapping[str, str] = {
                k: v for k, v in self.process_config.rename_bands.items() if k in data
            }
            data = (
                data.rename(columns=rename_bands)
                if is_frame
                else data.rename_vars(rename_bands)
            )
        return data

    def transform(
        self,
        data: pd.DataFrame | xr.Dataset,
        is_frame: bool,
    ) -> pd.DataFrame | xr.Dataset:
        # Transform
        if self.process_config.process_bands:
            for key, fn in self.process_config.process_bands.items():
                if key in data:
                    data[key] = (
                        data[key].apply(fn)
                        if is_frame
                        else xr.apply_ufunc(fn, data[key])
                    )
        return data

    def fillna(
        self,
        data: pd.DataFrame | xr.Dataset,
        bands: set[str] | Sequence[str],
    ) -> pd.DataFrame | xr.Dataset:
        for band in bands:
            nodata = query_by_key(
                band, self.process_config.nodata, self.process_config.nodata_fallback
            )
            data[band] = data[band].fillna(nodata)
        return data
