from typing import Any, Callable

import pystac
import pytest
from odc.stac.model import ParsedItem

from mccn.loader.raster.config import (
    _groupby_day,
    _groupby_hour,
    _groupby_minute,
    _groupby_month,
    _groupby_year,
    set_groupby,
)
from tests.utils import RASTER_FIXTURE_PATH


@pytest.fixture(scope="module")
def parsed_item_fx() -> tuple[pystac.Item, ParsedItem]:
    # Datetime is 2021-02-21T09:10:17Z
    item = pystac.Item.from_file(RASTER_FIXTURE_PATH / "L2A_PVI.json")
    from odc.stac._mdtools import parse_items

    parsed_item = list(parse_items([item]))[0]
    return item, parsed_item


@pytest.mark.parametrize(
    "fn,exp_value",
    [
        (_groupby_day, "2021-02-21"),
        (_groupby_hour, "2021-02-21T09"),
        (_groupby_minute, "2021-02-21T09:10"),
        (_groupby_month, "2021-02"),
        (_groupby_year, "2021"),
    ],
    ids=["day", "hour", "minute", "month", "year"],
)
def test_groupby_result(
    fn: Callable, exp_value: str, parsed_item_fx: tuple[pystac.Item, ParsedItem]
) -> None:
    actual = fn(*parsed_item_fx, 0)
    assert actual == exp_value


@pytest.mark.parametrize(
    "param,exp_value",
    [
        ("time", "time"),
        ("id", "id"),
        ("day", _groupby_day),
        ("hour", _groupby_hour),
        ("minute", _groupby_minute),
        ("month", _groupby_month),
        ("year", _groupby_year),
        (_groupby_day, _groupby_day),
    ],
    ids=["time", "id", "day", "hour", "minute", "month", "year", "callable"],
)
def test_set_groupby(param: Any, exp_value: str | Callable) -> None:
    value = set_groupby(param)
    assert value == exp_value
