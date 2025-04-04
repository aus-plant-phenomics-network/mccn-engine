from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import pandas as pd
import pystac
import pystac_client
import xarray as xr
from rasterio.enums import MergeAlg
from stac_generator import StacGeneratorFactory
from stac_generator.core import PointConfig, RasterConfig, VectorConfig

from mccn._types import (
    BBox_T,
    CubeConfig,
    FilterConfig,
    ParsedItem,
    ParsedPoint,
    ParsedRaster,
    ParsedVector,
    ProcessConfig,
)
from mccn.extent import GeoBoxBuilder
from mccn.loader.point import PointLoadConfig, PointLoader
from mccn.loader.raster import RasterLoadConfig, RasterLoader
from mccn.loader.utils import bbox_from_geobox, get_item_crs
from mccn.loader.vector import VectorLoadConfig, VectorLoader

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from numpy.typing import DTypeLike
    from odc.geo.geobox import GeoBox

    from mccn._types import InterpMethods, MergeMethods


def _parse_vector(
    config: VectorConfig,
    location: str,
    bbox: BBox_T,
    start: datetime.datetime,
    end: datetime.datetime,
    item: pystac.Item,
) -> ParsedVector:
    """
    Parse vector Item

    list the following
        crs - based on stac item projection
        bands - attributes described by column_info
        aux_bands - attributes of the external aslist that joins with vector file - i.e. join_column_info
    """
    crs = get_item_crs(item)
    bands = set([band["name"] for band in config.column_info])
    aux_bands = (
        set([band["name"] for band in config.join_column_info])
        if config.join_column_info
        else set()
    )
    return ParsedVector(
        location=location,
        bbox=bbox,
        start=start,
        end=end,
        config=config,
        item=item,
        bands=bands,
        load_bands=bands,
        aux_bands=aux_bands,
        load_aux_bands=aux_bands,
        crs=crs,
    )


def _parse_raster(
    config: RasterConfig,
    location: str,
    bbox: BBox_T,
    start: datetime.datetime,
    end: datetime.datetime,
    item: pystac.Item,
) -> ParsedRaster:
    """
    Parse Raster Item

    list the following:
        bands - bands described in band_info
        alias - eo:bands' common names
    """
    bands = set([band["name"] for band in config.band_info])
    alias = set(
        [
            band["common_name"]
            for band in config.band_info
            if band.get("common_name", None)
        ]
    )
    return ParsedRaster(
        location=location,
        bbox=bbox,
        start=start,
        end=end,
        config=config,
        item=item,
        bands=bands,
        load_bands=bands,
        alias=alias,
    )


def _parse_point(
    config: PointConfig,
    location: str,
    bbox: BBox_T,
    start: datetime.datetime,
    end: datetime.datetime,
    item: pystac.Item,
) -> ParsedPoint:
    """
    Parse point Item

    list the following
        crs - based on stac item projection
        bands - attributes described by column_info
    """
    bands = set([band["name"] for band in config.column_info])
    crs = get_item_crs(item)
    return ParsedPoint(
        location=location,
        bbox=bbox,
        start=start,
        end=end,
        config=config,
        item=item,
        bands=bands,
        load_bands=bands,
        crs=crs,
    )


def parse_item(item: pystac.Item) -> ParsedItem:
    """Parse a pystac.Item to a matching ParsedItem

    A ParsedItem contains attributes acquired from STAC metadata
    and stac_generator config that makes it easier to load aslist
    into the data cube

    Args:
        item (pystac.Item): Intial pystac Item

    Raises:
        ValueError: no stac_generator config found or config is not acceptable

    Returns:
        ParsedItem: one of ParsedVector, ParsedRaster, and ParsedPoint
    """
    config = StacGeneratorFactory.extract_item_config(item)
    location = config.location
    bbox = cast(BBox_T, item.bbox)
    start = cast(
        datetime.datetime,
        (
            pd.Timestamp(item.properties["start_datetime"])
            if "start_datetime" in item.properties
            else item.datetime
        ),
    )
    end = cast(
        datetime.datetime,
        (
            pd.Timestamp(item.properties["end_datetime"])
            if "end_datetime" in item.properties
            else item.datetime
        ),
    )
    if isinstance(config, PointConfig):
        return _parse_point(config, location, bbox, start, end, item)
    if isinstance(config, VectorConfig):
        return _parse_vector(config, location, bbox, start, end, item)
    if isinstance(config, RasterConfig):
        return _parse_raster(config, location, bbox, start, end, item)
    raise ValueError(f"Invalid config type: {type(config)}")


def bbox_filter(item: ParsedItem | None, bbox: BBox_T | None) -> ParsedItem | None:
    """Filter item based on bounding box

    If item is None or if item is outside of the bounding box, returns None
    Otherwise return the item

    Args:
        item (ParsedItem | None): parsed Item, nullable
        bbox (BBox_T | None): target bbox

    Returns:
        ParsedItem | None: filter result
    """
    if item and bbox:
        if (
            item.bbox[0] > bbox[2]
            or bbox[0] > item.bbox[2]
            or item.bbox[1] > bbox[3]
            or bbox[1] > item.bbox[3]
        ):
            return None
    return item


def date_filter(
    item: ParsedItem | None,
    start_dt: datetime.datetime | None,
    end_dt: datetime.datetime | None,
) -> ParsedItem | None:
    """Filter item by date

    If item is None or item's start and end timestamps are outside the range specified
    by start_dt and end_dt, return None. Otherwise, return the original item

    Args:
        item (ParsedItem | None): parsed item
        start_dt (datetime.datetime | None): start date
        end_dt (datetime.datetime | None): end date

    Returns:
        ParsedItem | None: filter result
    """
    if item:
        if (start_dt and start_dt > item.end) or (end_dt and end_dt < item.start):
            return None
    return item


def band_filter(
    item: ParsedItem | None, bands: Sequence[str] | None
) -> ParsedItem | None:
    """Parse and filter an item based on requested bands

    If the bands parameter is None or empty, all items' bands should be loaded. For
    point and raster data, the loaded bands are columns/attributes described
    in column_info and band_info. For raster data, the loaded bands are columns
    described in column_info and join_column_info is not null.

    If the bands parameter is not empty, items that contain any sublist of the requested bands
    are selected for loading. Items with no overlapping band will not be loaded.
    For point, the filtering is based on item.bands (columns described in column_info).
    For raster, the filtering is based on item.bands (columns described in band_info) and
    item.alias (list of potential alias). For vector, the filtering is based on item.bands
    and item.aux_bands (columns described in join_column_info).

    Selected items will have item.load_bands updated as the (list) intersection
    between item.bands and bands (same for item.aux_bands and item.load_aux_bands).
    For vector, if aux_bands are not null (columns will need to be read from the external aslist),
    join_vector_attribute and join_field will be added to item.load_bands and item.load_aux_bands.
    This means that to perform a join, the join columns must be loaded for both the vector aslist
    and the external aslist.

    Args:
        item (ParsedItem | None): parsed item, can be none
        bands (Sequence[str] | None): requested bands, can be none

    Returns:
        ParsedItem | None: parsed result
    """
    if item and bands:
        item.load_bands = set([band for band in bands if band in item.bands])
        # If vector - check if bands to be loaded are from joined_file - i.e. aux_bands
        if isinstance(item, ParsedVector):
            item.load_aux_bands = set(
                [band for band in bands if band in item.aux_bands]
            )
        # If raster - check if bands to be loaded are an alias
        if isinstance(item, ParsedRaster):
            alias = set([band for band in bands if band in item.alias])
            item.load_bands.update(alias)
        # If both load_band and load_aux_bands empty - return None
        if not item.load_bands and not (
            hasattr(item, "load_aux_bands") and item.load_aux_bands
        ):
            return None
    # If item is a vector - ensure that join attribute and join column are loaded
    if isinstance(item, ParsedVector) and item.load_aux_bands:
        item.load_aux_bands.add(cast(str, item.config.join_field))
        if item.config.join_T_column:
            item.load_aux_bands.add(item.config.join_T_column)
        item.load_bands.add(cast(str, item.config.join_attribute_vector))
    return item


class Parser:
    def __init__(
        self, filter_config: FilterConfig, collection: pystac.Collection
    ) -> None:
        self.collection = collection
        self.items = collection.get_items(recursive=True)
        self.filter_config = filter_config
        self.bbox = bbox_from_geobox(self.filter_config.geobox)
        self._point_items = list()
        self._vector_items = list()
        self._raster_items = list()

    @property
    def point(self) -> list[ParsedPoint]:
        return self._point_items

    @property
    def vector(self) -> list[ParsedVector]:
        return self._vector_items

    @property
    def raster(self) -> list[ParsedRaster]:
        return self._raster_items

    def __call__(self) -> None:
        for item in self.items:
            self.parse(item)

    def parse(self, item: pystac.Item) -> None:
        parsed_item = parse_item(item)
        parsed_item = bbox_filter(parsed_item, self.bbox)
        parsed_item = date_filter(
            parsed_item, self.filter_config.start_ts, self.filter_config.end_ts
        )
        parsed_item = band_filter(parsed_item, self.filter_config.bands)
        if parsed_item:
            if isinstance(parsed_item.config, VectorConfig):
                self._vector_items.append(parsed_item)
            elif isinstance(parsed_item.config, RasterConfig):
                self._raster_items.append(parsed_item)
            elif isinstance(parsed_item.config, PointConfig):
                self._point_items.append(parsed_item)
            else:
                raise ValueError(
                    f"Invalid item type - none of raster, vector or point: {type(parsed_item.config)}"
                )


class MCCN:
    def __init__(
        self,
        endpoint: str,
        collection_id: str,
        shape: int | tuple[int, int] | None = None,
        # Filter config
        geobox: GeoBox | None = None,
        start_ts: datetime.datetime | None = None,
        end_ts: datetime.datetime | None = None,
        bands: Sequence[str] | None = None,
        filter_config: FilterConfig | None = None,
        # Cube config
        x_coord: str = "x",
        y_coord: str = "y",
        t_coord: str = "time",
        z_coord: str = "z",
        use_z: bool = False,
        cube_config: CubeConfig | None = None,
        # Process Config
        rename_bands: Mapping[str, str] | None = None,
        process_bands: Mapping[str, Callable] | None = None,
        process_config: ProcessConfig | None = None,
        # Point Load Config
        point_nodata: int | float = 0,
        interp: InterpMethods | None = "nearest",
        agg_method: MergeMethods = "mean",
        point_load_config: PointLoadConfig | None = None,
        # Vector Load Config
        fill: int = 0,
        all_touched: bool = False,
        vector_nodata: Any | None = None,
        merge_alg: Literal["REPLACE", "ADD"] | MergeAlg = MergeAlg.replace,
        vector_dtype: Any | None = None,
        groupby: Literal["id", "field"] = "id",
        vector_load_config: VectorLoadConfig | None = None,
        # Raster Load Config
        resampling: str | dict[str, str] | None = None,
        chunks: dict[str, int | Literal["auto"]] | None = None,
        pool: ThreadPoolExecutor | int | None = None,
        raster_dtype: DTypeLike | Mapping[str, DTypeLike] = None,
        raster_load_config: RasterLoadConfig | None = None,
    ) -> None:
        self.collection = self.get_collection(endpoint, collection_id)
        # Prepare configs
        self.geobox = self.get_geobox(self.collection, geobox, shape)
        self.filter_config = (
            filter_config
            if filter_config
            else FilterConfig(
                geobox=self.geobox,
                start_ts=start_ts,
                end_ts=end_ts,
                bands=set(bands) if bands else None,
            )
        )
        self.filter_config.geobox = self.geobox
        self.cube_config = (
            cube_config
            if cube_config
            else CubeConfig(
                x_coord=x_coord,
                y_coord=y_coord,
                z_coord=z_coord,
                t_coord=t_coord,
                use_z=use_z,
            )
        )
        self.process_config = (
            process_config
            if process_config
            else ProcessConfig(rename_bands=rename_bands, process_bands=process_bands)
        )
        self.point_load_config = (
            point_load_config
            if point_load_config
            else PointLoadConfig(
                interp=interp, agg_method=agg_method, nodata=point_nodata
            )
        )
        self.vector_load_config = (
            vector_load_config
            if vector_load_config
            else VectorLoadConfig(
                fill=fill,
                all_touched=all_touched,
                nodata=vector_nodata,
                merge_alg=merge_alg,
                dtype=vector_dtype,
                groupby=groupby,
            )
        )
        self.raster_load_config = (
            raster_load_config
            if raster_load_config
            else RasterLoadConfig(
                resampling=resampling, chunks=chunks, pool=pool, dtype=raster_dtype
            )
        )
        self.parser = Parser(self.filter_config, self.collection)
        self.parser()
        self._point_loader = None
        self._vector_loader = None
        self._raster_loader = None

    @property
    def point_loader(self) -> PointLoader:
        if not self._point_loader:
            self._point_loader = PointLoader(
                list(self.parser.point),
                self.filter_config,
                self.cube_config,
                self.process_config,
                self.point_load_config,
            )
        return self._point_loader

    @property
    def vector_loader(self) -> VectorLoader:
        if not self._vector_loader:
            self._vector_loader = VectorLoader(
                list(self.parser.vector),
                self.filter_config,
                self.cube_config,
                self.process_config,
                self.vector_load_config,
            )
        return self._vector_loader

    @property
    def raster_loader(self) -> RasterLoader:
        if not self._raster_loader:
            self._raster_loader = RasterLoader(
                list(self.parser.raster),
                self.filter_config,
                self.cube_config,
                self.process_config,
                self.raster_load_config,
            )
        return self._raster_loader

    def get_collection(
        self,
        endpoint: str,
        collection_id: str,
    ) -> pystac.Collection:
        if endpoint.startswith("http"):
            res = pystac_client.Client.open(endpoint)
            return res.get_collection(collection_id)
        return pystac.Collection.from_file(endpoint)

    def get_geobox(
        self,
        collection: pystac.Collection,
        geobox: GeoBox | None = None,
        shape: int | tuple[int, int] | None = None,
    ) -> GeoBox:
        if geobox is not None:
            return geobox
        if shape is None:
            raise ValueError(
                "If geobox is not defined, shape must be provided to calculate geobox from collection"
            )
        return GeoBoxBuilder.from_collection(collection, shape)

    def load_point(self) -> xr.Dataset:
        return self.point_loader.load()

    def load_vector(self) -> xr.Dataset:
        return self.vector_loader.load()

    def load_raster(self) -> xr.Dataset:
        return self.raster_loader.load()

    def load(self) -> xr.Dataset:
        return xr.merge(
            [self.load_point(), self.load_vector(), self.load_raster()],
            combine_attrs="drop_conflicts",
        )
