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
    """Config used for item filtering"""

    geobox: GeoBox
    """Spatial extent"""
    start_ts: pd.Timestamp | None = None
    """Temporal extent - start"""
    end_ts: pd.Timestamp | None = None
    """Temporal extent - end"""
    bands: set[str] | None = None
    """Bands to be loaded"""
    mask_only: bool = False
    """If true, will only load the mask layers for vector assets. Any described vectors' columns will not be loaded."""
    use_all_vectors: bool = True
    """If use_all_vector is False, only items with column_info matching filtered bands will be loaded. Otherwise, load all vectors."""

    @cached_property
    def start_utc(self) -> pd.Timestamp | None:
        """Start time in UTC tzinfo"""
        return self.to_timestamp(self.start_ts)

    @cached_property
    def start_no_tz(self) -> pd.Timestamp | None:
        """Start time not tz-aware"""
        return self.to_timestamp(self.start_ts, utc=False)

    @cached_property
    def end_utc(self) -> pd.Timestamp | None:
        """End time UTC tzinfo"""
        return self.to_timestamp(self.end_ts)

    @cached_property
    def end_no_tz(self) -> pd.Timestamp | None:
        """End time not tz aware"""
        return self.to_timestamp(self.end_ts, utc=False)

    @staticmethod
    def to_timestamp(
        ts: datetime.datetime | pd.Timestamp | str | None,
        utc: bool = True,
    ) -> pd.Timestamp | None:
        """Convert datetime object to timestamps

        Timestamps that are not tz-awared (has no tzinfo) will be assigned utc. Otherwise,
        timestamps will be converted to utc.

        If utc is True, will keep tzinfo as UTC, otherwise return a utc timestamps that has no
        tzinfo

        Args:
            ts (datetime.datetime | pd.Timestamp | str | None): python datetime object
            utc (bool, optional): whether to localise to utc. Defaults to True.

        Raises:
            ValueError: if unable to convert ts to timestamp

        Returns:
            pd.Timestamp | None: output
        """
        if not ts:
            return None
        # Try parsing timestamp information
        try:
            ts = pd.Timestamp(ts)
        except Exception as e:
            raise ValueError(f"Invalid timestamp value: {ts}") from e
        # Localise to utc if not tzaware, then convert to UTC
        if ts.tzinfo is not None:
            ts = ts.tz_convert("utc")
        else:
            ts = ts.tz_localize("utc")
        # keep utc tz else set None
        return ts if utc else ts.tz_localize(None)


@dataclass
class CubeConfig:
    """Config that describes the datacube coordinates"""

    x_dim: str = "lon"
    """Name of the x coordinate in the datacube"""
    y_dim: str = "lat"
    """Name of the y coordinate in the datacube"""
    t_dim: str = "time"
    """Name of the time coordinate in the datacube"""
    z_dim: str = "z"
    """Name of the z coordinate"""
    spatial_ref_dim: str = "spatial_ref"
    """Spatial ref dimension - shows EPSG code"""
    use_z: bool = False
    """Whether to use z coordinate. Currently has no effect"""
    mask_name: str = "__MASK__"
    """Name of the mask layer, only comes in effect when combine_mask is True"""
    combine_mask: bool = False
    """Whether to combine all geometry layers to a single mask layer"""


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
    dtype_fallback: Dtype_T = "float32"

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
