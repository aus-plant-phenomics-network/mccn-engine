from __future__ import annotations

import json
from typing import TYPE_CHECKING, Mapping
from warnings import warn

from pyproj import CRS

if TYPE_CHECKING:
    import pystac


class StacExtensionError(Exception): ...


ASSET_KEY = "data"
BBOX_TOL = 1e-10

# Data format Agnostic Utilities


def get_item_href(
    item: pystac.Item,
    asset_key: str | Mapping[str, str],
) -> str:
    """Get the href of the source asset in item.

    Source asset is the source file which the stac item primarily describes. This
    is to differentiate from other assets which might serve as summary for the source
    asset. The source asset should be indexed by asset key, which can either be a string,
    or a dictionary that maps item id to source asset key.

    :param item: stac item
    :type item: pystac.Item
    :param asset_key: if provided as a string, will be the key of the source asset in item.Assets.
    if provided as a dictionary, must contain an entry for the current item with item.id as key and
    asset key as value.
    :type asset_key: str | Mapping[str, str]
    :raises KeyError: if asset_key is provided as a dictionary but does not contain item.id
    :raises TypeError: if asset_key is neither a string or a dictionary
    :return: the source asset href
    :rtype: str
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

    If no proj extension fields is found, will attempt to look for the field `epsg` in properties.

    :param item: stac item metadata
    :type item: pystac.Item
    :raises StacExtensionError: if proj:projjson is provided but with an invalid format
    :raises StacExtensionError: if there is no crs field in the metadata
    :return: CRS information
    :rtype: CRS
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
