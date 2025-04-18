from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor
    from typing import Literal, Mapping

    import pystac
    from numpy.typing import DTypeLike
    from odc.stac._stac_load import GroupbyCallback
    from odc.stac.model import ParsedItem

    from mccn._types import TimeGroupby


def _groupby_year(item: pystac.Item, parsed: ParsedItem, index: int) -> str:
    """Group item by year"""
    return parsed.nominal_datetime.strftime("%Y")


def _groupby_month(item: pystac.Item, parsed: ParsedItem, index: int) -> str:
    """Group item by year and month"""
    return parsed.nominal_datetime.strftime("%Y-%m")


def _groupby_day(item: pystac.Item, parsed: ParsedItem, index: int) -> str:
    """Group item by year, month, day"""
    return parsed.nominal_datetime.strftime("%Y-%m-%d")


def _groupby_hour(item: pystac.Item, parsed: ParsedItem, index: int) -> str:
    """Group item by year, month, day, hour"""
    return parsed.nominal_datetime.strftime("%Y-%m-%dT%H")


def _groupby_minute(item: pystac.Item, parsed: ParsedItem, index: int) -> str:
    """Group item by year, month, day, hour, minute"""
    return parsed.nominal_datetime.strftime("%Y-%m-%dT%H:%M")


@dataclass
class RasterLoadConfig:
    """Load config for raster asset. Parameters come from odc.stac.load"""

    groupby: TimeGroupby | GroupbyCallback | str = "time"
    resampling: str | dict[str, str] | None = None
    chunks: dict[str, int | Literal["auto"]] | None = None
    pool: ThreadPoolExecutor | int | None = None
    dtype: DTypeLike | Mapping[str, DTypeLike] = None

    def __post_init__(self) -> None:
        self.set_groupby()

    def set_groupby(self) -> None:
        match self.groupby:
            case "time":
                self.groupby = "time"
            case "year":
                self.groupby = _groupby_year
            case "month":
                self.groupby = _groupby_month
            case "day":
                self.groupby = _groupby_day
            case "hour":
                self.groupby = _groupby_hour
            case "minute":
                self.groupby = _groupby_minute
