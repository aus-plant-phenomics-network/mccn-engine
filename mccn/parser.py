import datetime
from typing import Mapping, Sequence, cast

import pandas as pd
import pystac
from odc.geo.geobox import GeoBox
from stac_generator import StacGeneratorFactory
from stac_generator.core import PointConfig, RasterConfig, SourceConfig, VectorConfig

from mccn._types import BBox_T
from mccn.loader.utils import bbox_from_geobox


class ParsedItem:
    def __init__(
        self,
        location: str,
        bbox: BBox_T,
        start: datetime.datetime,
        end: datetime.datetime,
        bands: set[str],
        aux_bands: set[str],
        config: SourceConfig,
        item: pystac.Item,
    ) -> None:
        self.location = location
        self.bbox = bbox
        self.start = start
        self.end = end
        self.bands = bands
        self.aux_bands = aux_bands
        self.config = config
        self.item = item
        # By default - load all visible bands and aux bands
        self.load_bands = bands
        self.load_aux_bands = aux_bands

    @classmethod
    def from_item(cls, item: pystac.Item) -> "ParsedItem":
        config = StacGeneratorFactory.extract_item_config(item)
        location = config.location
        bbox = cast(BBox_T, item.bbox)
        start = pd.Timestamp(item.properties.get("start_datetime", item.datetime))
        end = pd.Timestamp(item.properties.get("end_datetime", item.datetime))
        bands = set()
        aux_bands = set()

        if isinstance(config, PointConfig | VectorConfig):
            for field in config.column_info:
                bands.add(field["name"])
        if isinstance(config, VectorConfig) and config.join_column_info:
            for field in config.join_column_info:
                aux_bands.add(field["name"])
        if isinstance(config, RasterConfig):
            for band in config.band_info:
                bands.add(band["name"])
        return ParsedItem(
            location=location,
            bbox=bbox,
            start=start,
            end=end,
            bands=bands,
            aux_bands=aux_bands,
            config=config,
            item=item,
        )


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
    item: ParsedItem | None,
    bands: Sequence[str] | None = None,
) -> ParsedItem | None:
    if item and bands:
        load_bands = set([band for band in bands if band in item.bands])
        load_aux_bands = set([band for band in bands if band in item.aux_bands])
        if not (load_bands or load_aux_bands):
            return None
        item.load_bands = load_bands
        item.load_aux_bands = load_aux_bands
    return item


def filter_collection(
    collection: pystac.Collection,
    geobox: GeoBox | None = None,
    start_dt: datetime.datetime | None = None,
    end_dt: datetime.datetime | None = None,
    bands: Sequence[str] | None = None,
    band_mapping: Mapping[str, str] | None = None,
) -> list[ParsedItem]:
    parsed_items = [ParsedItem.from_item(item) for item in collection.get_all_items()]
    bbox = bbox_from_geobox(geobox) if geobox else None
    result = []
    for item in parsed_items:
        item = bbox_filter(item, bbox)
        item = date_filter(item, start_dt, end_dt)
        item = band_filter(item, bands, band_mapping)
        if item:
            result.append(item)
    return result


def process_band_and_band_mapping(
    bands: Sequence[str] | None, band_rename: Mapping[str, str]
) -> tuple[set[str] | None, set[str] | None, Mapping[str, str]]:
    if not bands:
        return None, None, band_rename
    # Avoid cyclic renaming
    for v in band_rename.values():
        if v in band_rename:
            raise ValueError(
                f"Chained renaming is not allowed - mapped value: {v} is a mapped key"
            )
    # Ensure that renamed bands must be loaded
    for k, v in band_rename.items():
        if v not in bands:
            raise ValueError(f"Mapped band from {k} to {v} must be loaded")
    filter_bands = set(bands).copy()
    for k in band_rename:
        filter_bands.add(k)
    return filter_bands, set(bands), band_rename
