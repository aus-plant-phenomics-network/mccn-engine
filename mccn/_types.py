from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import pandas as pd
from pyproj.crs.crs import CRS

if TYPE_CHECKING:
    import pystac
    from stac_generator.core import (
        PointConfig,
        RasterConfig,
        SourceConfig,
        VectorConfig,
    )

TimeGroupby = Literal["time", "day", "hour", "minute", "year", "month"]
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
Number_T = int | float
Resolution_T = Number_T | tuple[Number_T, Number_T]
Shape_T = int | tuple[int, int]
BBox_T = tuple[float, float, float, float]
CRS_T = str | int | CRS
AnchorPos_T = Literal["center", "edge", "floating", "default"] | tuple[float, float]


@dataclass(kw_only=True)
class ParsedItem:
    location: str
    """Data asset href"""
    bbox: BBox_T
    """Data asset bbox - in WGS84"""
    start: pd.Timestamp
    """Data asset start_datetime. Defaults to item.datetime if item.start_datetime is null"""
    end: pd.Timestamp
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
