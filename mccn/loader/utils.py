from __future__ import annotations

import json
from typing import TYPE_CHECKING, Mapping, cast
from warnings import warn

import geopandas as gpd
from pandas.api.types import is_datetime64_any_dtype
from pyproj import CRS, Transformer

from mccn._types import BBox_T

if TYPE_CHECKING:
    import pystac
    from odc.geo.geobox import GeoBox

    from mccn.extent import TimeBox


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


def convert_bbox_to_target_crs(bbox: BBox_T, src: CRS, target: CRS) -> BBox_T:
    """Convert bbox from one crs to another

    :param bbox: bounding box
    :type bbox: tuple[float, float, float, float]
    :param src: crs of the bounding box
    :type src: CRS
    :param target: target crs
    :type target: CRS
    :return: the bounding box in target CRS
    :rtype: tuple[float, float, float, float]
    """
    if src == target:
        return bbox
    transformer = Transformer.from_crs(src, target, always_xy=True)
    left, bottom = transformer.transform(bbox[0], bbox[1])
    right, top = transformer.transform(bbox[2], bbox[3])
    return left, bottom, right, top


def item_in_geobox(item: pystac.Item, geobox: GeoBox) -> bool:
    """Check if item is contained in geobox

    :param item: pystac Item. Must have bbox field and crs information in properties either as proj:epsg or epsg
    :type item: pystac.Item
    :param geobox: geobox containing bounding box and crs information
    :type geobox: GeoBox
    :return: whether some areas of item overlap with the bbox of geobox
    :rtype: bool
    """
    item_bbox = convert_bbox_to_target_crs(
        cast(BBox_T, item.bbox), get_item_crs(item), cast(CRS, geobox.crs)
    )
    gbbox = geobox.boundingbox
    if (
        item_bbox[0] > gbbox[2]
        or gbbox[0] > item_bbox[2]
        or item_bbox[1] > gbbox[3]
        or gbbox[1] > item_bbox[3]
    ):
        return False
    return True


def query_timebox(
    frame: gpd.GeoDataFrame, timebox: TimeBox, t_col: str = "T"
) -> gpd.GeoDataFrame:
    """Constrain frame to time region specified by `timebox` and converts timezone to
    that of `timebox`.

    This method works on both point GeoDataFrame and vector GeoDataFrame

    :param frame: input dataframe. Dataframe must contain `t_col` of datetime type with valid timezone info.
    :type frame: gpd.GeoDataFrame
    :param timebox: time extent defines start date, end date, frequency and timezone information
    :type timebox: TimeBox
    :param t_col: name of time column in frame, defaults to "T"
    :type t_col: str, optional
    :raises KeyError: if `t_col` is not in frame
    :raises TypeError: if dtype of `t_col` is not any datetime type
    :raises ValueError: if `t_col` is of datatime type but no timezone info is present
    :return: frame confined to time extent defined by timebox
    :rtype: gpd.GeoDataFrame
    """
    if t_col not in frame.columns:
        raise KeyError(f"No time column {t_col} in dataframe")
    if not is_datetime64_any_dtype(frame[t_col]):
        raise TypeError(f"Column {t_col} is not of datetime type")
    if frame[t_col].dt.tz is None:
        raise ValueError(f"Date column {t_col} does not have timezone info.")

    # Convert to timezone specified in timebox
    # Note `tz_convert` converts one tz to another
    # `tz_localize` embeds tz information to a date column w/o tz
    frame[t_col] = frame[t_col].dt.tz_convert(timebox.tz)
    return frame[(frame[t_col] >= timebox.start) and (frame[t_col] <= timebox.end)]
