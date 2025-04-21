from __future__ import annotations

import abc
from typing import (
    Any,
    Generic,
    Hashable,
    Sequence,
    TypeVar,
    overload,
)

import pandas as pd
import xarray as xr
from odc.geo.xr import xr_coords

from mccn._types import ParsedItem
from mccn.config import CubeConfig, FilterConfig, ProcessConfig

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
        self.filter_config = filter_config
        self.cube_config = cube_config if cube_config else CubeConfig()
        self.process_config = process_config if process_config else ProcessConfig()

    @property
    def coords(self) -> dict[Hashable, xr.DataArray]:
        """Coordinates of the datacube in y, x coordinates.

        Returns:
            dict[Hashable, xr.DataArray]: coordinate dict
        """
        return xr_coords(
            self.filter_config.geobox,
            dims=(self.cube_config.y_coord, self.cube_config.x_coord),
        )

    def load(self) -> xr.Dataset:
        ds = self._load()
        if ds:
            return self.apply_filter(
                ds,
                self.filter_config,
                self.cube_config,
            )
        return ds

    @abc.abstractmethod
    def _load(self) -> xr.Dataset:
        raise NotImplementedError

    @staticmethod
    def apply_filter(
        data: xr.Dataset,
        filter_config: FilterConfig,
        cube_config: CubeConfig,
    ) -> xr.Dataset:
        # Filter based on dates and geobox
        data = data.sel(
            {
                cube_config.t_coord: slice(
                    filter_config.start_no_tz, filter_config.end_no_tz
                )
            }
        )
        return data

    @overload
    @staticmethod
    def apply_process(
        data: xr.Dataset, process_config: ProcessConfig
    ) -> xr.Dataset: ...

    @overload
    @staticmethod
    def apply_process(
        data: pd.DataFrame, process_config: ProcessConfig
    ) -> pd.DataFrame: ...

    @staticmethod
    def apply_process(
        data: pd.DataFrame | xr.Dataset, process_config: ProcessConfig
    ) -> pd.DataFrame | xr.Dataset:
        """Apply a rename and process operation on input data.

        Acceptable data types are pandas.DataFrame, geopandas.GeoDataFrame, and xarray.Dataset

        Args:
            data (pd.DataFrame | xr.Dataset): input data
            process_config (ProcessConfig): process configuration

        Raises:
            ValueError: data type is invalid (not pandas.DataFrame or xarray.Dataset)

        Returns:
            pd.DataFrame | xr.Dataset: processed data
        """
        if isinstance(data, pd.DataFrame):
            # Transform
            if process_config.process_bands:
                for key, fn in process_config.process_bands.items():
                    if key in data.columns:
                        data[key] = data[key].apply(fn)
            # Rename bands
            if process_config.rename_bands:
                data.rename(columns=process_config.rename_bands, inplace=True)
            # Fillnan
            data.fillna(process_config.nodata, inplace=True)
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
            # Fillnan
            data = data.fillna(process_config.nodata)
            return data
        raise ValueError(f"Expeting data to be a dataframe or a dataset: {type(data)}")
