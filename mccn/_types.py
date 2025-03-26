from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from pyproj.crs.crs import CRS

if TYPE_CHECKING:
    import datetime
    from collections.abc import Mapping

    import pystac
    from odc.geo.geobox import GeoBox
    from stac_generator.core import (
        PointConfig,
        RasterConfig,
        SourceConfig,
        VectorConfig,
    )


InterpMethods = (
    Literal["linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"]
    | Literal["barycentric", "krogh", "pchip", "spline", "akima", "makima"]
)
_MergeMethods = (
    Literal[
        "add", "replace", "min", "max", "median", "mean", "sum", "prod", "var", "std"
    ]
    | Callable[[np.ndarray], float]
)
MergeMethods = _MergeMethods | dict[str, _MergeMethods]

BBox_T = tuple[float, float, float, float]
CRS_T = str | int | CRS
AnchorPos_T = Literal["center", "edge", "floating", "default"] | tuple[float, float]
GroupbyOption = Literal["id", "field"]


@dataclass(kw_only=True)
class ParsedItem:
    location: str
    bbox: BBox_T
    start: datetime.datetime
    end: datetime.datetime
    config: SourceConfig
    item: pystac.Item
    bands: set[str]
    load_bands: set[str] = field(default_factory=set)


@dataclass(kw_only=True)
class ParsedPoint(ParsedItem):
    crs: CRS
    config: PointConfig


@dataclass(kw_only=True)
class ParsedVector(ParsedItem):
    crs: CRS
    aux_bands: set[str] = field(default_factory=set)
    load_aux_bands: set[str] = field(default_factory=set)
    config: VectorConfig


@dataclass(kw_only=True)
class ParsedRaster(ParsedItem):
    alias: set[str] = field(default_factory=set)
    config: RasterConfig


@dataclass
class FilterConfig:
    geobox: GeoBox
    start_ts: datetime.datetime | None = None
    end_ts: datetime.datetime | None = None
    bands: set[str] | None = None


@dataclass
class CubeConfig:
    x_coord: str = "lon"
    """Name of the x coordinate in the datacube"""
    y_coord: str = "lat"
    """Name of the y coordinate in the datacube"""
    t_coord: str = "time"
    """Name of the time coordinate in the datacube"""
    z_coord: str = "alt"
    """Name of the altitude coordinate in the datacube"""
    use_z: bool = False
    """Whether to use the altitude coordinate as an axis"""


@dataclass
class ProcessConfig:
    rename_bands: Mapping[str, str] | None = None
    """Mapping between original to renamed bands"""
    process_bands: Mapping[str, Callable] | None = None
    """Mapping between band name and transformation to be applied to the band"""
