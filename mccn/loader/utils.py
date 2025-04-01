from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from pyproj import CRS
from pyproj.transformer import Transformer

from mccn._types import BBox_T

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
