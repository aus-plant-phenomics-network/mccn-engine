from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

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


@dataclass
class ProcessConfig:
    """The config that describes data transformation and column renaming before data is loaded to the final datacube"""

    rename_bands: Mapping[str, str] | None = None
    """Mapping between original to renamed bands"""
    process_bands: Mapping[str, Callable] | None = None
    """Mapping between band name and transformation to be applied to the band"""
