from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Hashable, Mapping, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
from odc.geo.xr import xr_coords
from pyproj import CRS
from pyproj.transformer import Transformer

from mccn._types import BBox_T, Nodata_Map_T, Nodata_T

if TYPE_CHECKING:
    import pystac
    import xarray as xr
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


def query_by_key(
    key: str,
    query_src: T | Mapping[str, T] | None,
    query_fallback: T,
) -> T:
    if isinstance(query_src, Mapping):
        return query_src.get(key, query_fallback)
    return query_src if query_src is not None else query_fallback


def query_if_null(
    value: T | None,
    key: str,
    query_src: T | Mapping[str, T] | None,
    query_fallback: T,
) -> T:
    if value is None:
        return query_by_key(key, query_src, query_fallback)
    return value


def update_attr_legend(
    attr_dict: dict[str, Any],
    field: str,
    frame: gpd.GeoDataFrame,
    start: int = 1,
    nodata: Nodata_Map_T = 0,
    nodata_fallback: Nodata_T = 0,
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
    nodata_value = query_by_key(field, nodata, nodata_fallback)
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


def get_neighbor_mask(
    gx: np.ndarray,
    gy: np.ndarray,
    points: np.ndarray,
    radius: float,
) -> Any:
    """Determine for each grid point the nearest neighbors based on the
    radius cut-off.

    Produce a mask matrix of shape (|gx|, |gy|, |points|), where |gx|,
    |gy|, |points| are the length of gx, gy, and points. Each mask entry
    index by x and y - i.e. mask[x,y,:] is a boolean array of the same length
    as points, where if mask[x,y,i] = True, then point[i] satisfies the condition
    to be the neighbor of (gx[x],gy[y]). For two points to be considered neighbors,
    their L2 distance must be less than radius.

    Args:
        gx (np.ndarray): grid x coordinates. Shape (|gx|,)
        gy (np.ndarray): grid y coordinates. Shape (|gy|,)
        points (np.ndarray): points coordinates. Shape (|points|, 2)
        radius (float): cut-off radius. Radius unit is based on the units of
        gx, gy, and points. Note for WGS84, radius of 0.05 degree ~ 5.5km

    Returns:
        Any: mask array. Shape (|gx|, |gy|, |points|)
    """
    # Build meshgrid and compute distance matrix
    grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")
    grid_coords = np.stack((grid_x, grid_y), axis=-1)
    distances = np.linalg.norm(grid_coords[..., np.newaxis, :] - points, axis=-1)
    # Render mask based on radius
    return distances < radius


def mask_aggregate(values: np.ndarray, mask: np.ndarray, op: Callable) -> np.ndarray:
    """Aggregate values using neighbor mask

    Neighbor mask parameter mask is obtained from the function `get_neighbor_mask`.
    Mask (|gx|, |gy|, |point|) describes the neighbor mask for a grid of dimenion (|gx|, |gy|).
    For each grid point, aggregation is performed by applying mean over True values.

    Args:
        values (np.ndarray): value array. Shape (|points|,)
        mask (np.ndarray): neighbor mask. Shape(|gx|, |gy|, |points|)

    Returns:
        np.ndarray: aggregation result. Shape (|gx|, |gy|)
    """
    # Apply mask - non masked values assigned nan to make calc simpler
    # masked_temp - (|gx|, |gy|, |points|)
    layer = np.full(mask.shape[:2], fill_value=np.nan)
    if mask.shape[2] == 0:
        return layer
    masked_temp = np.where(mask, values, np.nan)
    valid_mask = np.sum(mask, axis=2) > 0
    layer[valid_mask] = op(masked_temp[valid_mask], axis=1)
    return layer


def coords_from_geobox(
    geobox: GeoBox,
    x_dim: str,
    y_dim: str,
) -> dict[Hashable, xr.DataArray]:
    return xr_coords(
        geobox,
        dims=(y_dim, x_dim),
    )
