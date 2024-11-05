import abc
from collections.abc import Iterable, Sequence
from typing import Generic, TypeVar, cast

import geopandas as gpd
import pandas as pd
import pystac
import xarray as xr
from odc.geo.geobox import GeoBox
from pyproj import CRS
from stac_generator.csv.utils import read_csv, to_gdf

from mccn.geobox import BBox_T


class StacExtensionError(Exception): ...


T = TypeVar("T")


class Pipeline(abc.ABC, Generic[T]):
    def __init__(
        self,
        items: Iterable[pystac.Item],
        geobox: GeoBox,
        interp: str | None = None,
        fields: Sequence[str] | dict[str, Sequence[str]] | None = None,
    ) -> None:
        self._items = items
        self._geobox = geobox
        self._interp = interp
        self._fields = fields

    def execute(self) -> xr.Dataset:
        items = self.load_items(self._items, self._fields)
        items = self.query_bbox(
            items,
            cast(CRS, self._geobox.crs),
            (
                self._geobox.boundingbox.left,
                self._geobox.boundingbox.bottom,
                self._geobox.boundingbox.right,
                self._geobox.boundingbox.top,
            ),
        )
        ds = self.to_xarray(items)
        return self.interp(ds, self._geobox, self._interp)

    @abc.abstractmethod
    @staticmethod
    def load_item(
        item: pystac.Item,
        fields: Sequence[str] | None = None,
    ) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    @staticmethod
    def load_items(
        item: Iterable[pystac.Item],
        fields: Sequence[str] | dict[str, Sequence[str]] | None = None,
    ) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    @staticmethod
    def query_bbox(src: T, crs: CRS | None = None, bbox: BBox_T | None = None) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    @staticmethod
    def to_xarray(src: T) -> xr.Dataset:
        raise NotImplementedError

    @abc.abstractmethod
    @staticmethod
    def interp(src: xr.Dataset, geobox: GeoBox, method: str | None) -> xr.Dataset:
        raise NotImplementedError


class CsvPipeline(Pipeline[gpd.GeoDataFrame]):
    @staticmethod
    def load_item(
        item: pystac.Item,
        fields: Sequence[str] | None = None,
    ) -> gpd.GeoDataFrame:
        try:
            # Read in the raw csv file
            location = item.assets["source"].href
            X_name, Y_name = item.properties["X"], item.properties["Y"]
            prop_columns = item.properties.get("column_info", None)
            columns = fields if fields else prop_columns
            epsg = (
                item.properties.get("proj:epsg")
                if item.properties.get("proj:epsg")
                else item.properties.get("epsg")
            )
            epsg = cast(int, epsg)
            gdf = to_gdf(
                read_csv(
                    location,
                    X_name,
                    Y_name,
                    item.properties.get("T"),
                    item.properties.get("date_format", "ISO8601"),
                    columns,
                    item.properties.get("groupby"),
                ),
                X_name,
                Y_name,
                epsg,
            )
            gdf.drop(columns=[X_name, Y_name], inplace=True)
            return gdf
        except KeyError as e:
            raise StacExtensionError(
                "Key expected in stac item property not found"
            ) from e

    @staticmethod
    def load_items(
        items: Iterable[pystac.Item],
        fields: Sequence[str] | dict[str, Sequence[str]] | None = None,
    ) -> gpd.GeoDataFrame:
        df_list = []
        for item in items:
            if isinstance(fields, dict):
                item_df = CsvPipeline.load_item(item, fields.get(item.id))
            else:
                item_df = CsvPipeline.load_item(item, fields)
            df_list.append(item_df)
        return pd.concat(df_list)

    @staticmethod
    def query_bbox(
        src: gpd.GeoDataFrame,
        crs: CRS | None = None,
        bbox: BBox_T | None = None,
    ) -> gpd.GeoDataFrame:
        if crs:
            src = src.to_crs(crs)
        if bbox:
            x_slice = slice(bbox[0], bbox[2])
            y_slice = slice(bbox[1], bbox[3])
            src = src.cx[x_slice, y_slice]
        return src

    @staticmethod
    def to_xarray(src: gpd.GeoDataFrame) -> xr.Dataset:
        src["X"] = src["geometry"].apply(lambda item: item.x)
        src["Y"] = src["geometry"].apply(lambda item: item.y)
        src.drop(columns="geometry")
        src.set_index(["X", "Y"], inplace=True, drop=True)
        return src.to_xarray()  # type: ignore[no-any-return]
