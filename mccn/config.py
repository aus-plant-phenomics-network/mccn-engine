from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from mccn._types import MergeMethods, Number_T, TimeGroupby

if TYPE_CHECKING:
    import datetime
    from collections.abc import Mapping

    from odc.geo.geobox import GeoBox


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
    z_coord: str = "z"
    """Name of the z coordinate"""
    use_z: bool = False
    """Whether to use z coordinate"""


@dataclass
class ProcessConfig:
    """The config that describes data transformation and column renaming before data is loaded to the final datacube"""

    rename_bands: Mapping[str, str] | None = None
    """Mapping between original to renamed bands"""
    process_bands: Mapping[str, Callable] | None = None
    """Mapping between band name and transformation to be applied to the band"""
    nodata: Number_T | Mapping[str, Number_T] = 0
    """Value used to represent nodata value. Will also be used for filling nan data"""
    nodata_fallback: Number_T = 0
    """Value used for nodata when nodata is specified as as dict"""
    categorical_encoding_start: int = 1
    """The smallest legend value when converting from categorical values to numerical"""
    time_groupby: TimeGroupby = "time"
    """Time groupby value"""
    merge_alg: MergeMethods = "replace"
    """How to resolve merging for layer aggregation"""

    def __post_init__(self) -> None:
        if (
            self.categorical_encoding_start == self.nodata
            or self.categorical_encoding_start == self.nodata_fallback
        ):
            raise ValueError(
                "nodata value in categorical_encoding_start value must be different"
            )

    @property
    def period(self) -> str | None:
        match self.time_groupby:
            case "minute":
                return "min"
            case "hour":
                return "h"
            case "day":
                return "D"
            case "month":
                return "M"
            case "year":
                return "Y"
            case _:
                return None
