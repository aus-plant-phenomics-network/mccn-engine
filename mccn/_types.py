from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from numpy.typing import DTypeLike
from pyproj.crs.crs import CRS

TimeGroupby = Literal["time", "day", "hour", "minute", "year", "month"]

MergeMethod_T = Literal["replace", "min", "max", "mean", "sum"]
MergeMethod_Map_T = MergeMethod_T | Mapping[str, MergeMethod_T] | None

Number_T = int | float

Nodata_T = Number_T
Nodata_Map_T = Number_T | Mapping[str, Number_T] | None

Resolution_T = Number_T | tuple[Number_T, Number_T]
Shape_T = int | tuple[int, int]

Dtype_T = DTypeLike
DType_Map_T = Dtype_T | Mapping[str, Dtype_T]

BBox_T = tuple[float, float, float, float]
CRS_T = str | int | CRS
AnchorPos_T = Literal["center", "edge", "floating", "default"] | tuple[float, float]
