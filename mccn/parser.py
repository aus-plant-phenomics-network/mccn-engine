from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import cast

import pandas as pd
import pystac
from stac_generator.core import PointConfig, RasterConfig, VectorConfig
from stac_generator.factory import StacGeneratorFactory

from mccn._types import (
    BBox_T,
    ParsedItem,
    ParsedPoint,
    ParsedRaster,
    ParsedVector,
)
from mccn.config import FilterConfig
from mccn.loader.utils import bbox_from_geobox, get_item_crs


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
        set([band["name"] for band in config.join_config.column_info])
        if config.join_config
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
    start = (
        pd.Timestamp(item.properties["start_datetime"])
        if "start_datetime" in item.properties
        else pd.Timestamp(item.datetime)
    )
    end = (
        pd.Timestamp(item.properties["end_datetime"])
        if "end_datetime" in item.properties
        else pd.Timestamp(item.datetime)
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


def _filter_point(
    item: ParsedPoint,
    bands: Sequence[str] | set[str] | None,
) -> ParsedPoint | None:
    if not bands:
        return item
    item.load_bands = set([band for band in bands if band in item.bands])
    if item.load_bands:
        return item
    return None


def update_load_aux_bands(item: ParsedVector) -> ParsedVector:
    """Add join columns (left on, right on) and date column (if provided) into bands to load
    Only runs if there is at least one data bands (none of the above) to be loaded from join file
    """
    if item.config.join_config and item.load_aux_bands:
        join_config = item.config.join_config
        item.load_bands.add(join_config.left_on)
        item.load_aux_bands.add(join_config.right_on)
        if join_config.date_column:
            item.load_aux_bands.add(join_config.date_column)
    return item


def _filter_vector(
    item: ParsedVector,
    bands: Sequence[str] | set[str] | None,
    filter_config: FilterConfig,
) -> ParsedVector | None:
    # handle mask only -> Don't load any band
    if filter_config.mask_only:
        item.load_aux_bands.clear()
        item.load_bands.clear()
        return item
    # handle bands being None - load everything
    if not bands:
        return update_load_aux_bands(item)
    # Filter bands
    item.load_aux_bands = set([band for band in bands if band in item.aux_bands])
    item.load_bands = set([band for band in bands if band in item.bands])
    item = update_load_aux_bands(item)
    # If use all vectors -> load all items
    if filter_config.use_all_vectors:
        return item
    # If use only matching vectors -> remove items that don't match any band
    if item.load_bands or item.load_aux_bands:
        return item
    return None


def _filter_raster(
    item: ParsedRaster,
    bands: Sequence[str] | set[str] | None,
) -> ParsedRaster | None:
    if not bands:
        return item
    item.load_bands = set([band for band in bands if band in item.bands])
    alias = set([band for band in bands if band in item.alias])
    item.load_bands.update(alias)
    if item.load_bands:
        return item
    return None


def band_filter(
    item: ParsedItem | None,
    bands: Sequence[str] | set[str] | None,
    filter_config: FilterConfig,
) -> ParsedItem | None:
    """Parse and filter an item based on requested bands

    If the bands parameter is None or empty, all items' bands should be loaded. For
    point and raster data, the loaded bands are columns/attributes described
    in column_info and band_info. For raster data, the loaded bands are columns
    described in column_info and join_config.column_info.

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
    if not item:
        return None
    if isinstance(item, ParsedPoint):
        return _filter_point(item, bands)
    if isinstance(item, ParsedVector):
        return _filter_vector(item, bands, filter_config)
    if isinstance(item, ParsedRaster):
        return _filter_raster(item, bands)
    raise ValueError(f"Invalid item type: {type(item)}")


class Parser:
    """
    Parser collects metadata from pystac.Item for efficient loading. Also pre-filters
    items based on filter_config.

    The following basic filters are applied:
    - Date Filter - based on (start_ts, end_ts)
    - Bounding Box Filter - based on bbox in wgs 84
    - Band Filter - based on band information:
        If filtering band is None, all bands and columns from all assets will be loaded
        If filtering band is provided,
            - Raster - filter based on item's bands' name and common names
            - Vector - filter based on vector attributes (column_info) and join file attributes (join_column_info)
            - Point - filter based on point attributes (column_info)
    """

    def __init__(
        self,
        filter_config: FilterConfig,
        items: Sequence[pystac.Item] | None = None,
    ) -> None:
        self.items = list(items) if items else []
        self.filter_config = filter_config
        self.bands = self.filter_config.bands
        self.bbox = bbox_from_geobox(self.filter_config.geobox)
        self._point_items: list[ParsedPoint] = list()
        self._vector_items: list[ParsedVector] = list()
        self._raster_items: list[ParsedRaster] = list()

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
        parsed_item: ParsedItem | None
        parsed_item = parse_item(item)
        parsed_item = bbox_filter(parsed_item, self.bbox)
        parsed_item = date_filter(
            parsed_item,
            self.filter_config.start_utc,
            self.filter_config.end_utc,
        )
        parsed_item = band_filter(parsed_item, self.bands, self.filter_config)
        # Categorise parsed items
        if parsed_item:
            if isinstance(parsed_item.config, VectorConfig):
                self._vector_items.append(cast(ParsedVector, parsed_item))
            elif isinstance(parsed_item.config, RasterConfig):
                self._raster_items.append(cast(ParsedRaster, parsed_item))
            elif isinstance(parsed_item.config, PointConfig):
                self._point_items.append(cast(ParsedPoint, parsed_item))
            else:
                raise ValueError(
                    f"Invalid item type - none of raster, vector or point: {type(parsed_item.config)}"
                )
