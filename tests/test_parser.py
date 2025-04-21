from typing import Sequence

import pandas as pd
import pystac
import pytest
from odc.geo.geobox import GeoBox

from mccn._types import ParsedItem, ParsedVector
from mccn.config import FilterConfig
from mccn.parser import Parser


def get_items_id(items: Sequence[ParsedItem]) -> set[str]:
    return set([item.item.id for item in items])


def run_parser_test(parser: Parser, exp: set[str], attr: str) -> None:
    parser()
    assert len(parser.__getattribute__(attr)) == len(exp)
    assert get_items_id(parser.__getattribute__(attr)) == exp


def get_combined_load_bands(item: ParsedVector) -> set[str]:
    result = set()
    result.update(item.load_bands)
    result.update(item.load_aux_bands)
    return result


@pytest.mark.parametrize(
    "start_ts, end_ts, exp",
    [
        (
            "2016-01-01T00:00:00Z",
            None,
            {"top-left-16", "top-right-16", "bottom-left-16", "bottom-right-16"},
        ),
        (
            None,
            "2016-01-01T00:00:00Z",
            {"top-left-15", "top-right-15", "bottom-left-15", "bottom-right-15"},
        ),
        (
            "2015-11-01T00:00:00Z",
            "2016-01-01T00:00:00Z",
            {"bottom-left-15", "bottom-right-15"},
        ),
    ],
)
def test_time_filter(
    dsm_items: list[pystac.Item],
    dsm_geobox: GeoBox,
    start_ts: str | None,
    end_ts: str | None,
    exp: set[str],
) -> None:
    # No time filter, should load 8 items
    parser = Parser(
        FilterConfig(
            geobox=dsm_geobox,
            start_ts=pd.Timestamp(start_ts) if start_ts else None,
            end_ts=pd.Timestamp(end_ts) if end_ts else None,
        ),
        dsm_items,
    )
    run_parser_test(parser, exp, "raster")


@pytest.mark.parametrize(
    "geobox, exp",
    [
        (
            "dsm_geobox",
            {
                "bottom-left-16",
                "bottom-left-15",
                "bottom-right-15",
                "bottom-right-16",
                "top-left-16",
                "top-left-15",
                "top-right-15",
                "top-right-16",
            },
        ),
        (
            "dsm_top_right_geobox",
            {
                "bottom-right-15",
                "bottom-right-16",
                "top-right-15",
                "top-right-16",
            },
        ),
        (
            "dsm_bottom_right_geobox",
            {
                "bottom-left-16",
                "bottom-left-15",
                "bottom-right-15",
                "bottom-right-16",
                "top-left-16",
                "top-left-15",
                "top-right-15",
                "top-right-16",
            },
        ),
    ],
)
def test_geobox_filter(
    dsm_items: list[pystac.Item],
    geobox: str,
    exp: set[str],
    request: pytest.FixtureRequest,
) -> None:
    geobox_fx = request.getfixturevalue(geobox)
    parser = Parser(FilterConfig(geobox=geobox_fx), dsm_items)
    run_parser_test(parser, exp, "raster")


@pytest.mark.parametrize(
    "bands, exp_load_band",
    [
        (
            None,
            {
                "dsm": {"dsm"},
                "rgb": {"red", "green", "blue"},
                "rgb-alias": {"ms-red", "ms-green", "ms-blue"},
            },
        ),
        (
            {"red", "green", "blue"},
            {
                "rgb": {"red", "green", "blue"},
                "rgb-alias": {"red", "green", "blue"},
            },
        ),
        (
            {"ms-red", "ms-green", "ms-blue"},
            {"rgb-alias": {"ms-red", "ms-green", "ms-blue"}},
        ),
        (
            {"ms-red", "green", "ms-blue"},
            {"rgb-alias": {"ms-red", "green", "ms-blue"}, "rgb": {"green"}},
        ),
        ({"ms-red", "dsm"}, {"rgb-alias": {"ms-red"}, "dsm": {"dsm"}}),
        ({"dsm"}, {"dsm": {"dsm"}}),
    ],
    ids=[
        "None",
        "rgb",
        "ms-red, ms-green, ms-blue",
        "ms-red, green, blue",
        "ms-red, dsm",
        "dsm",
    ],
)
def test_raster_band_filter(
    multibands_items: list[pystac.Item],
    multiband_geobox: GeoBox,
    bands: set[str] | None,
    exp_load_band: dict[str, set[str]],
) -> None:
    parser = Parser(
        FilterConfig(bands=bands, geobox=multiband_geobox), multibands_items
    )
    parser()
    assert len(parser.raster) == len(exp_load_band)
    for item in parser.raster:
        assert item.item.id in exp_load_band
        assert item.load_bands == exp_load_band[item.item.id]


@pytest.mark.parametrize(
    "bands,exp",
    [
        (
            None,
            {
                "point-cook-mask": set(),
                "hoppers-crossing-name": {"name"},
                "werribee-crime": {
                    "name",
                    "area_sqkm",
                    "lga_name",
                    "crime_incidents",
                    "crime_rate",
                },
                "sunbury-crime": {
                    "name",
                    "area_sqkm",
                    "lga_name",
                    "crime_incidents",
                    "crime_rate",
                },
                "sunbury-population": {"name", "area_sqkm", "population", "date"},
            },
        ),
        (
            {"name"},
            {
                "point-cook-mask": set(),
                "hoppers-crossing-name": {"name"},
                "werribee-crime": {
                    "name",
                },
                "sunbury-crime": {
                    "name",
                },
                "sunbury-population": {"name", "date"},
            },
        ),
        (
            {"area_sqkm"},
            {
                "point-cook-mask": set(),
                "hoppers-crossing-name": set(),
                "werribee-crime": {
                    "area_sqkm",
                },
                "sunbury-crime": {
                    "area_sqkm",
                },
                "sunbury-population": {"area_sqkm"},
            },
        ),
        (
            {"lga_name"},
            {
                "point-cook-mask": set(),
                "hoppers-crossing-name": set(),
                "werribee-crime": {"lga_name", "name"},
                "sunbury-crime": {"lga_name", "name"},
                "sunbury-population": set(),
            },
        ),
        (
            {"name", "area_sqkm"},
            {
                "point-cook-mask": set(),
                "hoppers-crossing-name": {"name"},
                "werribee-crime": {"name", "area_sqkm"},
                "sunbury-crime": {"name", "area_sqkm"},
                "sunbury-population": {"name", "date", "area_sqkm"},
            },
        ),
    ],
    ids=["None-all", "name", "area_sqkm", "lga_name", "name, area_sqkm"],
)
def test_vector_band_filter(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
    bands: set[str] | None,
    exp: dict[str, set[str]],
) -> None:
    parser = Parser(FilterConfig(bands=bands, geobox=area_geobox), area_items)
    parser()
    assert len(parser.vector) == len(exp)
    for item in parser.vector:
        item_id = item.item.id
        assert item_id in exp
        assert get_combined_load_bands(item) == exp[item_id]
