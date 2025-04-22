from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Mapping, TypeVar

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from pyproj.transformer import Transformer

from mccn._types import BBox_T, Number_T

if TYPE_CHECKING:
    import pystac
    from odc.geo.geobox import GeoBox

ASSET_KEY = "data"
BBOX_TOL = 1e-10


class StacExtensionError(Exception): ...


logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_crs_transformer(src: CRS, dst: CRS) -> Transformer:
    """Cached method for getting pyproj.Transformer object

    Args:
        src (CRS): source crs
        dst (CRS): destition crs

    Returns:
        Transformer: transformer object
    """
    return Transformer.from_crs(src, dst, always_xy=True)


@lru_cache(maxsize=None)
def bbox_from_geobox(geobox: GeoBox, crs: CRS | str | int = 4326) -> BBox_T:
    """Generate a bbox from a geobox

    Args:
        geobox (GeoBox): source geobox which might have a different crs
        crs (CRS | str | int, optional): target crs. Defaults to 4326.

    Returns:
        BBox_T: bounds of the geobox in crs
    """
    if isinstance(crs, str | int):
        crs = CRS.from_epsg(crs)
    transformer = get_crs_transformer(geobox.crs, crs)
    bbox = list(geobox.boundingbox)
    left, bottom = transformer.transform(bbox[0], bbox[1])
    right, top = transformer.transform(bbox[2], bbox[3])
    return left, bottom, right, top


def get_item_crs(item: pystac.Item) -> CRS:
    """Extract CRS information from a STAC Item.

    For the best result, item should be generated using the
    projection extension (stac_generator does this by default).
    This method will look up proj:wkt2 (wkt2 string - the best option), proj:code,
    proj:projjson, proj:epsg, then epsg. An error is raised if none of the key
    is found.

    Args:
        item (pystac.Item): STAC Item with proj extension applied to properties

    Raises:
        StacExtensionError: ill-formatted proj:projjson
        StacExtensionError: no suitable proj key is found in item's properties

    Returns:
        CRS: CRS of item
    """
    if "proj:wkt2" in item.properties:
        return CRS(item.properties.get("proj:wkt2"))
    elif "proj:code" in item.properties:
        return CRS(item.properties.get("proj:code"))
    elif "proj:projjson" in item.properties:
        try:
            return CRS(json.loads(item.properties.get("proj:projjson")))  # type: ignore[arg-type]
        except json.JSONDecodeError as e:
            raise StacExtensionError("Invalid projjson encoding in STAC config") from e
    elif "proj:epsg" in item.properties:
        logger.warning(
            "proj:epsg is deprecated in favor of proj:code. Please consider using proj:code, or if possible, the full wkt2 instead"
        )
        return CRS(int(item.properties.get("proj:epsg")))  # type: ignore[arg-type]
    elif "epsg" in item.properties:
        return CRS(int(item.properties.get("epsg")))  # type: ignore[arg-type]
    else:
        raise StacExtensionError("Missing CRS information in item properties")


T = TypeVar("T")


class UNSET_T: ...


UNSET = UNSET_T()


def select_by_key(
    key: str,
    value: T | Mapping[str, T],
    fallback_value: T | UNSET_T = UNSET,
) -> T:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        if isinstance(fallback_value, UNSET_T):
            raise KeyError("Undefined fallback value")
        return fallback_value
    return value


def update_attr_legend(
    attr_dict: dict[str, Any],
    field: str,
    frame: gpd.GeoDataFrame,
    start: int = 1,
    nodata: Number_T | Mapping[str, Number_T] = 0,
    nodata_fallback: Number_T = 0,
) -> None:
    """Update attribute dict with legend for non numeric fields.

    If the field is non-numeric - i.e. string, values will be categoricalised
    i.e. 1, 2, 3, ...
    The mapping will be updated in attr_dict under field name

    Args:
        attr_dict (dict[str, Any]): attribute dict
        field (str): field name
        frame (gpd.GeoDataFrame): input data frame
        start (int): starting value
    """
    nodata_value = select_by_key(field, nodata, nodata_fallback)
    if not pd.api.types.is_numeric_dtype(frame[field]):
        curr = start
        cat_map = {}
        # Category map - original -> mapped value
        for name in frame[field].unique():
            if name != nodata_value and not pd.isna(name):
                cat_map[name] = curr
                curr += 1
        # Attr dict - mapped value -> original
        attr_dict[field] = {v: k for k, v in cat_map.items()}
        frame[field] = frame[field].map(cat_map)
