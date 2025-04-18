from typing import Sequence

import pandas as pd
import pystac
import pytest
from odc.geo.geobox import GeoBox

from mccn._types import FilterConfig, ParsedItem
from mccn.parser import Parser


def get_items_id(items: Sequence[ParsedItem]) -> set[str]:
    return set([item.item.id for item in items])


def run_parser_test(parser: Parser, exp: set[str]) -> None:
    parser()
    assert len(parser.raster) == len(exp)
    assert get_items_id(parser.raster) == exp


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
    dsm_collection: pystac.Collection,
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
        dsm_collection,
    )
    run_parser_test(parser, exp)


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
    dsm_collection: pystac.Collection,
    geobox: str,
    exp: set[str],
    request: pytest.FixtureRequest,
) -> None:
    geobox_fx = request.getfixturevalue(geobox)
    parser = Parser(FilterConfig(geobox=geobox_fx), dsm_collection)
    run_parser_test(parser, exp)


@pytest.mark.parametrize(
    "bands,exp",
    [
        (None, {"dsm", "rgb", "rgb-alias"}),
        ({"red", "green", "blue"}, {"rgb", "rgb-alias"}),
        ({"ms-red", "ms-green", "ms-blue"}, {"rgb-alias"}),
        ({"ms-red", "green", "ms-blue"}, {"rgb-alias", "rgb"}),
        ({"ms-red", "dsm"}, {"rgb-alias", "dsm"}),
        ({"dsm"}, {"dsm"}),
    ],
    ids=["None-all", "rgb+ms", "ms-only", "ms+rgb", "ms+dsm", "dsm-only"],
)
def test_band_filter(
    multibands_collection: pystac.Collection,
    multiband_geobox: GeoBox,
    bands: set[str] | None,
    exp: set[str],
) -> None:
    parser = Parser(
        FilterConfig(bands=bands, geobox=multiband_geobox), multibands_collection
    )
    run_parser_test(parser, exp)
    raster = parser.raster
    if bands:
        for item in raster:
            assert item.load_bands.issubset(bands)
