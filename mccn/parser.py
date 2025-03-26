import datetime
from typing import Sequence, cast

import pandas as pd
import pystac
from stac_generator import StacGeneratorFactory
from stac_generator.core import PointConfig, RasterConfig, VectorConfig

from mccn._types import (
    BBox_T,
    ParsedItem,
    ParsedPoint,
    ParsedRaster,
    ParsedVector,
)
from mccn.loader.utils import get_item_crs


def _parse_vector(
    config: VectorConfig,
    location: str,
    bbox: BBox_T,
    start: datetime.datetime,
    end: datetime.datetime,
    item: pystac.Item,
) -> ParsedVector:
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


def band_filter(
    item: ParsedItem | None, bands: Sequence[str] | None
) -> ParsedItem | None:
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
        item.load_bands.add(cast(str, item.config.join_attribute_vector))
    return item
