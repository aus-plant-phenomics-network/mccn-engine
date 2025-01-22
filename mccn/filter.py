from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pystac
from odc.geo.geobox import GeoBox
from pyproj import Transformer
from pyproj.crs.crs import CRS

from mccn._types import BBox_T
from mccn.loader.utils import ASSET_KEY, get_item_href


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


# TODO: filter based on fields and band information
class CollectionFilter:
    def __init__(
        self,
        collection: pystac.Collection,
        geobox: GeoBox,
        asset_key: str | Mapping[str, str] = ASSET_KEY,
    ) -> None:
        self.collection = collection
        self.geo_bbox = convert_bbox_to_target_crs(
            cast(BBox_T, geobox.boundingbox), cast(CRS, geobox.crs), CRS(4326)
        )
        self.asset_key = asset_key
        self._raster_items: list[pystac.Item] = []
        self._vector_items: list[pystac.Item] = []
        self._point_items: list[pystac.Item] = []
        self.classify()

    @property
    def raster(self) -> list[pystac.Item]:
        return self._raster_items

    @property
    def vector(self) -> list[pystac.Item]:
        return self._vector_items

    @property
    def point(self) -> list[pystac.Item]:
        return self._point_items

    @staticmethod
    def item_in_bbox(item: pystac.Item, bbox: BBox_T) -> bool:
        ibox = cast(BBox_T, item.bbox)
        if (
            ibox[0] > bbox[2]
            or bbox[0] > ibox[2]
            or ibox[1] > bbox[3]
            or bbox[1] > ibox[3]
        ):
            return False
        return True

    def classify(self) -> None:
        items = self.collection.get_items(recursive=True)
        for item in items:
            if not CollectionFilter.item_in_bbox(item, self.geo_bbox):
                continue
            href = get_item_href(item, self.asset_key)
            if href.endswith("tif") or href.endswith("geotif"):
                self._raster_items.append(item)
            elif href.endswith("csv") or href.endswith("txt"):
                self._point_items.append(item)
            elif (
                href.endswith("zip")
                or href.endswith("geojson")
                or href.endswith("json")
                or href.endswith("gpkg")
                or href.endswith("shp")
            ):
                self._vector_items.append(item)
            else:
                raise ValueError(f"Invalid item extension: {item}")
