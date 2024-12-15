from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast

import pystac
from odc.geo.geobox import GeoBox
from odc.geo.types import Shape2d

from mccn._types import BBox_T

if TYPE_CHECKING:
    from affine import Affine
    from odc.geo.geobox import GeoBox, GeoboxAnchor
    from odc.geo.types import XY, Resolution, Shape2d

    from mccn._types import CRS_T, AnchorPos_T, BBox_T


# Placeholder - might be useful for extracting geobox from all items in a collection
class GeoBoxBuilder:
    """Utility class to build odc.geo.GeoBox.

    GeoBox is a bounding box that knows its coordinate reference system and has a grid shape defined in X/Y orientation.
    To build a GeoBox, you will need to supply the crs (either at construction or with chaining), and either

    - The bounding box and one of:
        - The shape of the output geobox in terms of WxH
        - The resolution of the output geobox in terms of WxH where the unit is determined from crs
    - The affine transformation that maps grid points to CRS unit and:
        - The shape of the output geobox in terms of WxH

    To control pixel snapping, you can pass the parameters anchor and tol to the constructor

    ## Usage
    Building a geobox from a bounding box and crs="EPSG:4326" for region [ 138.5930197, -34.9260189, 138.5942712, -34.9242038 ]
    with shape being 100x100

    ```
    builder = (
        GeoBoxBuilder(crs=4326)
        .set_bbox([138.5930197, -34.9260189, 138.5942712, -34.9242038])
        .set_shape(100, 100)
    )

    geobox = builder.build()
    ```

    Building a geobox from a bounding box and crs="EPSG:4326" for region [ 138.5930197, -34.9260189, 138.5942712, -34.9242038 ] with resolution
    being 0.00036302 x 0.00036302 in degree unit. This should yield the same geobox as the previous example.

    ```
    builder = (
        GeoBoxBuilder(crs=4326)
        .set_box([138.5930197, -34.9260189, 138.5942712, -34.9242038])
        .set_resolution(0.00036302, 0.00036302)
    )

    geobox = builder.build()
    ```

    Building a geobox when the affine transformation is known - i.e. based on `rasterio.ReaderDriver` object
    ```
    from affine import Affine

    builder = (
        GeoBoxBuilder(crs=4326)
        .set_bbox([138.5930197, -34.9260189, 138.5942712, -34.9242038])
        .set_transform(
            Affine.from_gdal(
                0.000363020000000347,
                0.0,
                138.59268654013246,
                0.0,
                -0.000363020000000347,
                -34.92397608003338,
            )
        )
    )
    geobox = builder.build()
    ```

    :param crs: coordinate reference system information. Accepts one of - WKT string, EPSG code (int), or ```pyproj.crs.crs.CRS``` object
    :type crs: CRS_T
    :param tol: fraction of a pixel that can be ignored. Bounding box of the geobox is allowed to be smaller than the supplied bbox by that amount. Defaults to 0.01
    :type tol: float, optional
    :param anchor: anchoring value, defaults to "default"
    :type anchor: AnchorPos_T, optional
    """

    def __init__(
        self,
        crs: CRS_T,
        tol: float = 1e-3,
        anchor: AnchorPos_T = "default",
    ) -> None:
        self._crs = crs
        self._shape: Shape2d | None = None
        self._resolution: Resolution | None = None
        self._bbox: BBox_T | None = None
        self._transform: Affine = None
        self._tol = tol
        self._anchor: GeoboxAnchor = "default"
        if isinstance(anchor, tuple):
            if len(anchor) != 2:
                raise ValueError("Expect 2-tuple for anchor")
            self._anchor = XY(anchor[0], anchor[1])
        else:
            self._anchor = anchor

    def set_crs(self, crs: CRS_T) -> Self:
        """Overwrite CRS value

        :param crs: crs value
        :type crs: CRS_T
        :return: the current builder for method chaining
        :rtype: Self
        """
        self._crs = crs
        return self

    def set_shape(self, x: int, y: int) -> Self:
        """Set geobox's shape (WxH). For building a geobox from a
        bounding box (transformation is not known), one of shape or resolution
        needs to be provided prior to building the geobox. For building a geobox
        from a known affine transformation, only shape is required.

        :param x: width
        :type x: int
        :param y: height
        :type y: int
        :return: the current builder for method chaining
        :rtype: Self
        """
        self._shape = Shape2d(x, y)
        return self

    def set_resolution(self, x: float, y: float) -> Self:
        """Set geobox's resolution (WxH). For building a geobox from a
        bounding box (transformation is not known), one of shape or resolution
        needs to be provided prior to building the geobox.

        :param x: resolution in the x (W) dimension
        :type x: float
        :param y: resolution in the y (H) dimension
        :type y: float
        :return: the current builder for method chaining
        :rtype: Self
        """
        self._resolution = Resolution(x, y)
        return self

    def set_bbox(self, bbox: BBox_T) -> Self:
        """Set bbox.

        :param bbox: bbox in format [left, bottom, right, top] or [minX, minY, maxX, maxY]
        :type bbox: BBox_T
        :return: the current builder for method chaining
        :rtype: Self
        """
        self._bbox = (
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
        )
        return self

    def set_transformation(self, transform: Affine) -> Self:
        """Set the affine transformation that maps grid pixel to crs value

        :param transform: affine transformation
        :type transform: Affine
        :return: the current builder for method chaining
        :rtype: Self
        """
        self._transform = transform
        return self

    def build(self) -> GeoBox:
        """Build the geobox object based on given parameters"""
        if self._transform and self._shape:
            return GeoBox(self._shape, self._transform, self._crs)
        if self._bbox is None:
            raise ValueError(
                "Bounding Box expected. Use `set_bbox` method to specify the bounding box!"
            )
        if self._resolution is None and self._shape is None:
            raise ValueError(
                "Expect either resolution or shape to be set. Use `set_resolution` or `set_shape` to specify the shape of the geobox."
            )
        return GeoBox.from_bbox(
            bbox=self._bbox,
            crs=self._crs,
            shape=self._shape,
            resolution=self._resolution,
            anchor=self._anchor,
            tol=self._tol,
        )

    @classmethod
    def from_collection(
        cls, collection: pystac.Collection, shape: int | tuple[int, int]
    ) -> GeoBox:
        builder = GeoBoxBuilder(
            crs=4326
        )  # Note that collection bbox info is always 4326
        bbox = collection.extent.spatial.bboxes[0]  # First bbox is the enclosing bbox
        if isinstance(shape, int):
            shape = (shape, shape)
        builder = builder.set_bbox(bbox=cast(BBox_T, bbox)).set_shape(*shape)
        return builder.build()
