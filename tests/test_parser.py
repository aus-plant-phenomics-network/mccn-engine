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
        (
            {"red", "green", "blue", "non_matching"},
            {
                "rgb": {"red", "green", "blue"},
                "rgb-alias": {"red", "green", "blue"},
            },
        ),
        ({"dsm", "non_matching"}, {"dsm": {"dsm"}}),
        ({"non_matching"}, set()),
    ],
    ids=[
        "None",
        "rgb",
        "ms-red, ms-green, ms-blue",
        "ms-red, green, blue",
        "ms-red, dsm",
        "dsm",
        "red, green, blue, non_matching",
        "dsm, non_matching",
        "non_matching",
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
    "bands, mask_only, use_all_vectors, exp",
    [
        (
            None,
            False,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": {"name"},
                "werribee_crime": {
                    "name",
                    "area_sqkm",
                    "lga_name",
                    "crime_incidents",
                    "crime_rate",
                },
                "sunbury_crime": {
                    "name",
                    "area_sqkm",
                    "lga_name",
                    "crime_incidents",
                    "crime_rate",
                },
                "sunbury_population": {"name", "area_sqkm", "population", "date"},
            },
        ),
        (
            None,
            True,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            None,
            False,
            False,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": {"name"},
                "werribee_crime": {
                    "name",
                    "area_sqkm",
                    "lga_name",
                    "crime_incidents",
                    "crime_rate",
                },
                "sunbury_crime": {
                    "name",
                    "area_sqkm",
                    "lga_name",
                    "crime_incidents",
                    "crime_rate",
                },
                "sunbury_population": {"name", "area_sqkm", "population", "date"},
            },
        ),
        (
            None,
            True,
            False,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"name"},
            False,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": {"name"},
                "werribee_crime": {
                    "name",
                },
                "sunbury_crime": {
                    "name",
                },
                "sunbury_population": {"name", "date"},
            },
        ),
        (
            {"name"},
            True,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"name"},
            False,
            False,
            {
                "hoppers_crossing_name": {"name"},
                "werribee_crime": {
                    "name",
                },
                "sunbury_crime": {
                    "name",
                },
                "sunbury_population": {"name", "date"},
            },
        ),
        (
            {"name"},
            True,
            False,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"area_sqkm"},
            False,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": {
                    "area_sqkm",
                },
                "sunbury_crime": {
                    "area_sqkm",
                },
                "sunbury_population": {"area_sqkm"},
            },
        ),
        (
            {"area_sqkm"},
            True,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"area_sqkm"},
            False,
            False,
            {
                "werribee_crime": {
                    "area_sqkm",
                },
                "sunbury_crime": {
                    "area_sqkm",
                },
                "sunbury_population": {"area_sqkm"},
            },
        ),
        (
            {"area_sqkm"},
            True,
            False,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"population"},
            False,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": {"population", "date", "name"},
            },
        ),
        (
            {"population"},
            True,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"population"},
            False,
            False,
            {
                "sunbury_population": {"population", "date", "name"},
            },
        ),
        (
            {"population"},
            True,
            False,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"lga_name"},
            False,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": {"lga_name", "name"},
                "sunbury_crime": {"lga_name", "name"},
                "sunbury_population": set(),
            },
        ),
        (
            {"lga_name"},
            True,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"lga_name"},
            False,
            False,
            {
                "werribee_crime": {"lga_name", "name"},
                "sunbury_crime": {"lga_name", "name"},
            },
        ),
        (
            {"lga_name"},
            True,
            False,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"name", "area_sqkm"},
            False,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": {"name"},
                "werribee_crime": {"name", "area_sqkm"},
                "sunbury_crime": {"name", "area_sqkm"},
                "sunbury_population": {"name", "date", "area_sqkm"},
            },
        ),
        (
            {"name", "area_sqkm"},
            True,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"name", "area_sqkm"},
            False,
            False,
            {
                "hoppers_crossing_name": {"name"},
                "werribee_crime": {"name", "area_sqkm"},
                "sunbury_crime": {"name", "area_sqkm"},
                "sunbury_population": {"name", "date", "area_sqkm"},
            },
        ),
        (
            {"name", "area_sqkm"},
            True,
            False,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"non_matching"},
            False,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"non_matching"},
            True,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        ({"non_matching"}, False, False, []),
        (
            {"non_matching"},
            True,
            False,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"name", "non_matching"},
            False,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": {"name"},
                "werribee_crime": {
                    "name",
                },
                "sunbury_crime": {
                    "name",
                },
                "sunbury_population": {"name", "date"},
            },
        ),
        (
            {"name", "non_matching"},
            True,
            True,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
        (
            {"name", "non_matching"},
            False,
            False,
            {
                "hoppers_crossing_name": {"name"},
                "werribee_crime": {
                    "name",
                },
                "sunbury_crime": {
                    "name",
                },
                "sunbury_population": {"name", "date"},
            },
        ),
        (
            {"name", "non_matching"},
            True,
            False,
            {
                "point_cook_mask": set(),
                "hoppers_crossing_name": set(),
                "werribee_crime": set(),
                "sunbury_crime": set(),
                "sunbury_population": set(),
            },
        ),
    ],
    ids=[
        "None-False-True",
        "None-True-True",
        "None-False-False",
        "None-True-False",
        "name-False-True",
        "name-True-True",
        "name-False-False",
        "name-True-False",
        "area-False-True",
        "area-True-True",
        "area-False-False",
        "area-True-False",
        "population-False-True",
        "population-True-True",
        "population-False-False",
        "population-True-False",
        "lga_name-False-True",
        "lga_name-True-True",
        "lga_name-False-False",
        "lga_name-True-False",
        "name+area-False-True",
        "name+area-True-True",
        "name+area-False-False",
        "name+area-True-False",
        "non_matching-False-True",
        "non_matching-True-True",
        "non_matching-False-False",
        "non_matching-True-False",
        "name+non_matching-False-True",
        "name+non_matching-True-True",
        "name+non_matching-False-False",
        "name+non_matching-True-False",
    ],
)
def test_vector_band_filter(
    area_items: list[pystac.Item],
    area_geobox: GeoBox,
    bands: set[str] | None,
    mask_only: bool,
    use_all_vectors: bool,
    exp: dict[str, set[str]],
) -> None:
    parser = Parser(
        FilterConfig(
            bands=bands,
            geobox=area_geobox,
            mask_only=mask_only,
            use_all_vectors=use_all_vectors,
        ),
        area_items,
    )
    parser()
    assert len(parser.vector) == len(exp)
    for item in parser.vector:
        item_id = item.item.id
        assert item_id in exp
        assert get_combined_load_bands(item) == exp[item_id]
