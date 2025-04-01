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
    """Data asset href"""
    bbox: BBox_T
    """Data asset bbox - in WGS84"""
    start: datetime.datetime
    """Data asset start_datetime. Defaults to item.datetime if item.start_datetime is null"""
    end: datetime.datetime
    """Data asset end_datetime. Defaults to item.datetime if item.end_datetime is null"""
    config: SourceConfig
    """STAC Generator config - used for loading data into datacube"""
    item: pystac.Item
    """Reference to the actual STAC Item"""
    bands: set[str]
    """Bands (or columns) described in the Data asset"""
    load_bands: set[str] = field(default_factory=set)
    """Bands (or columns) to be loaded into the datacube from the Data asset"""


@dataclass(kw_only=True)
class ParsedPoint(ParsedItem):
    crs: CRS
    """Data asset's CRS"""
    config: PointConfig
    """STAC Generator config - point type"""


@dataclass(kw_only=True)
class ParsedVector(ParsedItem):
    crs: CRS
    """Data asset's CRS"""
    aux_bands: set[str] = field(default_factory=set)
    """Bands (or columns) described in the join file - (external property file linked to the vector asset)"""
    load_aux_bands: set[str] = field(default_factory=set)
    """Bands (or columns) to be loaded into the datacube from the join file - i.e. external asset"""
    config: VectorConfig
    """STAC Generator config - vector type"""


@dataclass(kw_only=True)
class ParsedRaster(ParsedItem):
    alias: set[str] = field(default_factory=set)
    """Band aliasing - derived from eobands common name"""
    config: RasterConfig
    """STAC Generator config - raster type"""


@dataclass
class FilterConfig:
    """The config that describes the extent of the cube"""

    geobox: GeoBox
    """Spatial extent"""
    start_ts: datetime.datetime | None = None
    """Temporal extent - start"""
    end_ts: datetime.datetime | None = None
    """Temporal extent - end"""
    bands: set[str] | None = None
    """Bands to be loaded"""


@dataclass
class CubeConfig:
    """The config that describes the datacube coordinates"""

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
    """The config that describes data transformation and column renaming before data is loaded to the final datacube"""

    rename_bands: Mapping[str, str] | None = None
    """Mapping between original to renamed bands"""
    process_bands: Mapping[str, Callable] | None = None
    """Mapping between band name and transformation to be applied to the band"""
