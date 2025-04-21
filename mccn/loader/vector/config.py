from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from rasterio.enums import MergeAlg


@dataclass
class RasterizeConfig:
    """Parameters to be passed to `rasterio.features.rasterize` function"""

    fill: int = 0
    """Used as fill value for all areas not covered by input geometries."""
    all_touched: bool = False
    """If True, all pixels touched by geometries will be burned in. If false, only pixels whose center is within the polygon or that are selected by Bresenham's line algorithm will be burned in."""
    nodata: Any | None = 0
    """nodata value to use in output file or masked array."""
    merge_alg: Literal["REPLACE", "ADD"] | MergeAlg = "REPLACE"
    """Merge algorithm to use. One of: MergeAlg.replace (default): the new value will overwrite the existing value. MergeAlg.add: the new value will be added to the existing raster."""
    dtype: Any | None = None
    """Used as data type for results, if out is not provided."""

    def __post_init__(self) -> None:
        # Instantiate merge_alg enum from string
        if isinstance(self.merge_alg, str):
            if self.merge_alg.lower() == "replace":
                self.merge_alg = MergeAlg.replace
            elif self.merge_alg.lower() == "add":
                self.merge_alg = MergeAlg.add
            else:
                raise ValueError(
                    f"Invalid merge alg value: expects replace or add, received: {self.merge_alg}"
                )


@dataclass
class VectorLoadConfig:
    rasterize_config: RasterizeConfig | Mapping[str, RasterizeConfig] = field(
        default_factory=RasterizeConfig
    )
    """Parameters to be passed to `rasterio.features.rasterize`"""
