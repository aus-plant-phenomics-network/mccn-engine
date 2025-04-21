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


def set_groupby(
    groupby: TimeGroupby | str | GroupbyCallback,
) -> str | GroupbyCallback:
    match groupby:
        case "year":
            return _groupby_year
        case "month":
            return _groupby_month
        case "day":
            return _groupby_day
        case "hour":
            return _groupby_hour
        case "minute":
            return _groupby_minute
        case _:
            return groupby


@dataclass
class RasterLoadConfig:
    """Load config for raster asset. Parameters come from odc.stac.load"""

    resampling: str | dict[str, str] | None = None
    chunks: dict[str, int | Literal["auto"]] | None = None
    pool: ThreadPoolExecutor | int | None = None
    dtype: DTypeLike | Mapping[str, DTypeLike] = None
