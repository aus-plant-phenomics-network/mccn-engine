import abc
import datetime
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, cast

import pandas as pd
import pystac
from odc.geo.geobox import GeoBox
from stac_generator import StacGeneratorFactory
from stac_generator.core import PointConfig, RasterConfig, SourceConfig, VectorConfig
from stac_generator.core.base import SourceConfig

from mccn._types import BBox_T


def bbox_filter(item: pystac.Item | None, bbox: BBox_T) -> pystac.Item | None:
    if item:
        ibox = cast(BBox_T, item.bbox)
        if not (
            ibox[0] > bbox[2]
            or bbox[0] > ibox[2]
            or ibox[1] > bbox[3]
            or bbox[1] > ibox[3]
        ):
            return item
    return None


def date_filter(
    item: pystac.Item | None,
    start_dt: datetime.datetime | None,
    end_dt: datetime.datetime | None,
) -> pystac.Item | None:
    if item:
        istart_dt = pd.Timestamp(item.properties.get("start_datetime", item.datetime))
        iend_dt = pd.Timestamp(item.properties.get("end_datetime", item.datetime))
        if (start_dt and start_dt > iend_dt) or (end_dt and end_dt < istart_dt):
            return None
    return item


def band_filter(
    item: pystac.Item | None,
    bands_to_load: list[str],
    bands: Sequence[str] | None = None,
    band_mapping: Mapping[str, str] | None = None,
) -> pystac.Item | None:
    if not item:
        return None
    config = StacGeneratorFactory.extract_item_config(item)

    def filter_point(config: PointConfig):
        item_bands = [band["name"] for band in config.column_info]
    
    def filter_vector(config: VectorConfig): 

    def filter_raster(config: RasterConfig):
        

    return None
