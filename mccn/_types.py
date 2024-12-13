from typing import Callable, Literal

import numpy as np
from pyproj.crs.crs import CRS

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
