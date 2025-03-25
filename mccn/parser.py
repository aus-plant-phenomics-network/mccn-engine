import datetime
from dataclasses import Field, dataclass
from typing import cast

import pandas as pd
import pystac
from pyproj import CRS
from stac_generator import StacGeneratorFactory
from stac_generator.core import PointConfig, RasterConfig, SourceConfig, VectorConfig

from mccn._types import BBox_T
from mccn.loader.utils import get_item_crs


@dataclass(kw_only=True)
class ParsedItem:
    location: str
    bbox: BBox_T
    start: datetime.datetime
    end: datetime.datetime
    config: SourceConfig
    item: pystac.Item
    bands: set[str]
    load_bands: set[str] = Field(default_factory=set)


@dataclass(kw_only=True)
class ParsedPoint(ParsedItem):
    crs: CRS


@dataclass(kw_only=True)
class ParsedVector(ParsedItem):
    aux_bands: set[str] = Field(default_factory=set)
    load_aux_bands: set[str] = Field(default_factory=set)


@dataclass(kw_only=True)
class ParsedRaster(ParsedItem):
    alias: set[str] = Field(default_factory=set)


def _parse_vector(
    config: VectorConfig,
    location: str,
    bbox: BBox_T,
    start: datetime.datetime,
    end: datetime.datetime,
    item: pystac.Item,
) -> ParsedVector:
    bands = set([band["name"] for band in config.column_info])
    aux_bands = set([band["name"] for band in config.join_column_info])
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
    )


def _parse_raster(
    config: RasterConfig,
    location: str,
    bbox: BBox_T,
    start: datetime.datetime,
    end: datetime.datetime,
    item: pystac.Item,
) -> ParsedRaster:
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
    config = StacGeneratorFactory.extract_item_config(item)
    location = config.location
    bbox = cast(BBox_T, item.bbox)
    start = (
        pd.Timestamp(item.properties["start_datetime"])
        if "start_datetime" in item.properties
        else item.datetime
    )
    end = (
        pd.Timestamp(item.properties["end_datetime"])
        if "end_datetime" in item.properties
        else item.datetime
    )
    if isinstance(config, PointConfig):
        return _parse_point(config, location, bbox, start, end, item)
    if isinstance(config, VectorConfig):
        return _parse_vector(config, location, bbox, start, end, item)
    return _parse_raster(config, location, bbox, start, end, item)


def bbox_filter(item: ParsedItem | None, bbox: BBox_T | None) -> ParsedItem | None:
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
    if item:
        if (start_dt and start_dt > item.end) or (end_dt and end_dt < item.start):
            return None
    return item


# def band_filter(item: ParsedItem | None, bands: Sequence[str] | None)->ParsedItem | None:
#     if item and bands:
#         if isinstance(item, ParsedRaster):
#     return item
