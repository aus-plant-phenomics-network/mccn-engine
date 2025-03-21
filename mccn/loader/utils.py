from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import TYPE_CHECKING, Mapping, Sequence
from warnings import warn

from pyproj import CRS
from pyproj.transformer import Transformer

from mccn._types import BBox_T

if TYPE_CHECKING:
    import pystac
    from odc.geo.geobox import GeoBox


class StacExtensionError(Exception): ...


ASSET_KEY = "data"
BBOX_TOL = 1e-10


def get_item_href(
    item: pystac.Item,
    asset_key: str | Mapping[str, str],
) -> str:
    """Get the href of the primary asset.

    While the general STAC metadata allows for multiple asset per item, in MCCN, we
    assume that there is only one primary asset whose metadata is described by the item.
    This function loads the primary asset described by asset key.

    Args:
        item (pystac.Item): item
        asset_key (str | Mapping[str, str]): if str, the identifier of the primary asset. If dict, the key
        is the item's id and value is the identifier of the primary asset associated with item id.

    Raises:
        KeyError: asset key dict does not contain the current item's id
        TypeError: invalid type for asset_key. Expects str or dict

    Returns:
        str: the primary asset's href
    """
    if isinstance(asset_key, str):
        return item.assets[asset_key].href
    elif isinstance(asset_key, dict):
        if item.id not in asset_key:
            raise KeyError(f"Asset key map does not have entry for item: {item.id}")
        return item.assets[asset_key[item.id]].href
    raise TypeError(
        f"Invalid type for asset key: {type(asset_key)}. Accepts either a string or a mapping"
    )


def get_item_crs(item: pystac.Item) -> CRS:
    """Extract CRS information from item properties.

    This will first look for CRS information encoded as proj extension, in the following order:
    `proj:code, proj:wkt2, proj:projjson, proj:epsg`

    Args:
        item (pystac.Item): stac item

    Raises:
        StacExtensionError: Invalid format for proj:projjson
        StacExtensionError: no crs description available

    Returns:
        CRS: crs of the current item
    """
    if "proj:code" in item.properties:
        return CRS(item.properties.get("proj:code"))
    elif "proj:wkt2" in item.properties:
        return CRS(item.properties.get("proj:wkt2"))
    elif "proj:projjson" in item.properties:
        try:
            return CRS(json.loads(item.properties.get("proj:projjson")))  # type: ignore[arg-type]
        except json.JSONDecodeError as e:
            raise StacExtensionError("Invalid projjson encoding in STAC config") from e
    elif "proj:epsg" in item.properties:
        warn(
            "proj:epsg is deprecated in favor of proj:code. Please consider using proj:code, or if possible, the full wkt2 instead"
        )
        return CRS(int(item.properties.get("proj:epsg")))  # type: ignore[arg-type]
    elif "epsg" in item.properties:
        return CRS(int(item.properties.get("epsg")))  # type: ignore[arg-type]
    else:
        raise StacExtensionError("Missing CRS information in item properties")


def get_required_columns(
    item: pystac.Item,
    bands: Sequence[str] | None = None,
    band_renaming: Mapping[str, str] | None = None,
) -> list[str]:
    """Get a list of fields to be read from a point/vector asset.

    If column_info is not described in item properties, return empty list.
    If bands is None, return all bands described in column_info
    If band_renaming is not None, replace every band in bands with their band_renaming's mapping target.
    Return bands in column_info that are also requested in bands keyword.

    Args:
        item (pystac.Item): stac item
        bands (Sequence[str] | None, optional): requested bands. Defaults to None.
        band_renaming (Mapping[str, str] | None, optional): renaming map. Defaults to None.

    Returns:
        list[str]: bands to be read from the current item
    """
    # No column info described - return empty list
    if "column_info" not in item.properties:
        return []
    # No requested bands - load all columns
    if not bands:
        return [column["name"] for column in item.properties["column_info"]]
    # If band_renaming is provided, also replace the requested band with the version that will be renamed
    # i.e. if requested band is height_m, but we need to rename height_cm to height_m, the requested band will include
    # just height_cm
    requested_bands = copy.copy(set(bands))
    if band_renaming:
        for k, v in band_renaming.items():
            if v in bands:
                requested_bands.add(k)
    return [
        column["name"]
        for column in item.properties["column_info"]
        if column["name"] in requested_bands
    ]


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
def bbox_from_geobox(geobox: GeoBox, crs: CRS) -> BBox_T:
    transformer = get_crs_transformer(geobox.crs, crs)
    bbox = list(geobox.boundingbox)
    left, bottom = transformer.transform(bbox[0], bbox[1])
    right, top = transformer.transform(bbox[2], bbox[3])
    return left, bottom, right, top
