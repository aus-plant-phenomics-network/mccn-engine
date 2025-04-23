from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Callable

import pandas as pd

from mccn._types import (
    DType_Map_T,
    Dtype_T,
    MergeMethod_Map_T,
    MergeMethod_T,
    Nodata_Map_T,
    Nodata_T,
    TimeGroupby,
)

if TYPE_CHECKING:
    import datetime
    from collections.abc import Mapping

    from odc.geo.geobox import GeoBox


@dataclass
class FilterConfig:
    """The config that describes the extent of the cube"""

    geobox: GeoBox
    """Spatial extent"""
    start_ts: pd.Timestamp | None = None
    """Temporal extent - start"""
    end_ts: pd.Timestamp | None = None
    """Temporal extent - end"""
    bands: set[str] | None = None
    """Bands to be loaded"""
    mask_only: bool = False
    """If true, will only load the mask layer for vector dtype"""
    use_all_vectors: bool = True
    """When loading masks, only use band matching vector items or all. Default to False (use all vectors for MASK layer)"""

    @cached_property
    def start_utc(self) -> pd.Timestamp:
        return self.to_timestamp(self.start_ts)

    @cached_property
    def start_no_tz(self) -> pd.Timestamp:
        return self.to_timestamp(self.start_ts, utc=False)

    @cached_property
    def end_utc(self) -> pd.Timestamp:
        return self.to_timestamp(self.end_ts)

    @cached_property
    def end_no_tz(self) -> pd.Timestamp:
        return self.to_timestamp(self.end_ts, utc=False)

    @staticmethod
    def to_timestamp(
        ts: datetime.datetime | pd.Timestamp | str | None, utc: bool = True
    ) -> pd.Timestamp | None:
        if not ts:
            return None
        # Try parsing timestamp information
        try:
            ts = pd.Timestamp(ts)
        except Exception as e:
            raise ValueError(f"Invalid timestamp value: {ts}") from e

        if ts.tzinfo is not None:
            ts = ts.tz_convert("utc")
        else:
            ts = ts.tz_localize("utc")
        return ts if utc else ts.tz_localize(None)


@dataclass
class CubeConfig:
    """The config that describes the datacube coordinates"""

    x_dim: str = "lon"
    """Name of the x coordinate in the datacube"""
    y_dim: str = "lat"
    """Name of the y coordinate in the datacube"""
    t_dim: str = "time"
    """Name of the time coordinate in the datacube"""
    z_dim: str = "z"
    """Name of the z coordinate"""
    spatial_ref_dim: str = "spatial_ref"
    use_z: bool = False
    """Whether to use z coordinate"""
    mask_name: str = "__MASK__"
    """Name of the mask layer"""


@dataclass
class ProcessConfig:
    """The config that describes data transformation and column renaming before data is loaded to the final datacube"""

    rename_bands: Mapping[str, str] | None = None
    """Mapping between original to renamed bands"""
    process_bands: Mapping[str, Callable] | None = None
    """Mapping between band name and transformation to be applied to the band"""
    nodata: Nodata_Map_T = None
    """Value used to represent nodata value. Will also be used for filling nan data"""
    nodata_fallback: Nodata_T = 0
    """Value used for nodata when nodata is specified as as dict"""
    time_groupby: TimeGroupby = "time"
    """Time groupby value"""
    merge_method: MergeMethod_Map_T = None
    merge_method_fallback: MergeMethod_T = "replace"
    dtype: DType_Map_T = None
    dtype_fallback: Dtype_T = "float64"

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
