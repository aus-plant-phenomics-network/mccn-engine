import numpy as np
import pytest

from mccn.drawer import MaxDrawer, MeanDrawer, MinDrawer, ReplaceDrawer, SumDrawer


@pytest.mark.parametrize(
    "x, y, t, x_shape, y_shape, t_shape, nodata, bands",
    [
        # Standard case
        ("x", "y", "t", 10, 5, 3, -1, {"b1", "b2"}),
        # Single point in each dimension
        ("x", "y", "t", 1, 1, 1, 0, {"b1"}),
        # Large dimensions
        ("x", "y", "t", 100, 50, 30, -9999, {"b1", "b2", "b3"}),
        # Non-default dimension names
        ("longitude", "latitude", "time", 12, 8, 4, 0, {"band1", "band2"}),
        # Non-default dimension names
        ("longitude", "latitude", "time", 12, 8, 4, -9999, {"band1", "band2"}),
        # Negative nodata value
        ("x", "y", "t", 15, 10, 5, -5, {"temperature"}),
        # No bands
        ("x", "y", "t", 7, 3, 2, -1, set()),
        # Edge case: no time points
        ("x", "y", "t", 5, 5, 0, -1, {"b1"}),
        # Edge case: no x or y coordinates
        ("x", "y", "t", 0, 0, 10, 999, {"band1"}),
        # All zero values in the coordinates
        ("x", "y", "t", 0, 0, 0, 0, {"b1"}),
        # Different nodata values for each band
        ("x", "y", "t", 10, 5, 3, -999, {"b1", "b2", "b3"}),
    ],
)
def test_build_dimensions_and_coords(
    x: str,
    y: str,
    t: str,
    x_shape: int,
    y_shape: int,
    t_shape: int,
    nodata: int,
    bands: set[str],
) -> None:
    # Setup
    x_coords = np.arange(x_shape)
    y_coords = np.arange(y_shape)
    t_coords = np.arange(t_shape)
    bands = bands
    nodata = nodata

    drawer = SumDrawer(
        x_coords=x_coords,
        y_coords=y_coords,
        t_coords=t_coords,
        x_dim=x,
        y_dim=y,
        t_dim=t,
        bands=bands,
        nodata=nodata,
    )

    # Call build
    result = drawer.build()

    # Validate dimensions
    assert result.sizes == {t: len(t_coords), x: len(x_coords), y: len(y_coords)}
    assert list(result.sizes.keys()) == [t, x, y]

    # Validate coordinates
    assert np.array_equal(result.coords[t].values, t_coords)
    assert np.array_equal(result.coords[x].values, x_coords)
    assert np.array_equal(result.coords[y].values, y_coords)

    # Check that the bands are actually data vars
    assert bands == result.data_vars.keys()

    # Validate data is filled with nodata
    for band in bands:
        assert np.all(result[band].values == nodata)


@pytest.fixture()
def nonzero_sum_drawer() -> SumDrawer:
    return SumDrawer(
        x_coords=np.arange(3),
        y_coords=np.arange(4),
        t_coords=np.arange(2),
        bands={"b0"},
        nodata=-9999,
    )


@pytest.fixture()
def nonzero_min_drawer() -> MinDrawer:
    return MinDrawer(
        x_coords=np.arange(3),
        y_coords=np.arange(4),
        t_coords=np.arange(2),
        bands={"b0"},
        nodata=-9999,
    )


@pytest.mark.parametrize(
    "draw_sequence, exp_result",
    [
        # Single draw with no valid values
        (
            [
                np.array(
                    [
                        [np.nan, -9999, np.nan, -9999],
                        [np.nan, -9999, np.nan, -9999],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
            ],
            np.array(
                [
                    [-9999, -9999, -9999, -9999],
                    [-9999, -9999, -9999, -9999],
                    [-9999, -9999, -9999, -9999],
                ]
            ),
        ),
        # Multiple draws with mixed valid and invalid values
        (
            [
                np.array(
                    [
                        [1, 2, -9999, np.nan],
                        [4, 5, -9999, np.nan],
                        [7, 8, -9999, np.nan],
                    ]
                ),
                np.array(
                    [
                        [np.nan, 2, 3, 4],
                        [-9999, 5, 6, -9999],
                        [7, 8, 9, -9999],
                    ]
                ),
            ],
            np.array(
                [
                    [1, 4, 3, 4],
                    [4, 10, 6, -9999],
                    [14, 16, 9, -9999],
                ]
            ),
        ),
        # Single draw fully overwrites the nodata layer
        (
            [
                np.array(
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                    ]
                ),
            ],
            np.array(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                ]
            ),
        ),
        # Three sequential draws with compounding values
        (
            [
                np.array(
                    [
                        [1, np.nan, -9999, 4],
                        [-9999, 6, np.nan, 8],
                        [9, 10, -9999, 12],
                    ]
                ),
                np.array(
                    [
                        [1, 2, 3, 4],
                        [5, -9999, 7, 8],
                        [9, 10, -9999, 12],
                    ]
                ),
                np.array(
                    [
                        [np.nan, 2, 3, 4],
                        [5, 6, -9999, 8],
                        [9, 10, 11, np.nan],
                    ]
                ),
            ],
            np.array(
                [
                    [2, 4, 6, 12],
                    [10, 12, 7, 24],
                    [27, 30, 11, 24],
                ]
            ),
        ),
        # Draw with all `nodata` values followed by valid layer
        (
            [
                np.array(
                    [
                        [-9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999],
                    ]
                ),
                np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            ],
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
        ),
        # 4 draw sequences
        (
            [
                np.array(
                    [[1, -9999, 3, np.nan], [4, -9999, 6, 8], [-9999, 10, 12, 14]]
                ),
                np.array(
                    [[2, 3, -9999, 5], [-9999, 7, np.nan, 9], [10, -9999, 13, 15]]
                ),
                np.array(
                    [[np.nan, 4, 5, 6], [7, np.nan, -9999, 10], [11, 12, -9999, 16]]
                ),
                np.array([[3, -9999, 7, 8], [-9999, 11, 13, np.nan], [14, 15, 17, 18]]),
            ],
            np.array([[6, 7, 15, 19], [11, 18, 19, 27], [35, 37, 42, 63]]),
        ),
        # 5 draw sequences
        (
            [
                np.array([[1, 2, -9999, np.nan], [4, 5, 6, 7], [8, 9, -9999, 11]]),
                np.array([[2, 3, 4, 5], [-9999, 7, 8, 9], [10, np.nan, 12, 13]]),
                np.array([[np.nan, 4, 5, -9999], [7, 8, 9, 10], [11, 12, 13, 14]]),
                np.array([[3, 4, 5, 6], [7, np.nan, 8, 9], [-9999, 11, 12, 13]]),
                np.array([[4, 5, 6, 7], [8, 9, np.nan, -9999], [10, 11, 12, 13]]),
            ],
            np.array([[10, 18, 20, 18], [26, 29, 31, 35], [39, 43, 49, 64]]),
        ),
        # 15 draw sequences
        (
            [
                np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
                np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
                np.array([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]),
                np.array([[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
                np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
                np.array([[6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17]]),
                np.array([[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]]),
                np.array([[8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]]),
                np.array([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]),
                np.array([[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]),
                np.array([[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]]),
                np.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]),
                np.array([[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]),
                np.array([[14, 15, 16, 17], [18, 19, 20, 21], [22, 23, 24, 25]]),
                np.array([[15, 16, 17, 18], [19, 20, 21, 22], [23, 24, 25, 26]]),
            ],
            np.array(
                [[120, 135, 150, 165], [180, 195, 210, 225], [240, 255, 270, 285]]
            ),
        ),
        # 4 draw sequences with alternating nodata and valid values
        (
            [
                np.array(
                    [[1, -9999, np.nan, 4], [-9999, 6, 7, np.nan], [9, 10, -9999, 12]]
                ),
                np.array([[2, np.nan, 3, -9999], [4, 5, np.nan, 6], [-9999, 8, 9, 10]]),
                np.array([[np.nan, 2, 3, 4], [5, np.nan, 7, 8], [9, 10, np.nan, 12]]),
                np.array([[4, -9999, 5, 6], [-9999, 7, 8, 9], [10, np.nan, 11, 12]]),
            ],
            np.array(
                [
                    [7.0, 2.0, 11.0, 14.0],
                    [9.0, 18.0, 22.0, 23.0],
                    [28.0, 28.0, 20.0, 46.0],
                ]
            ),
        ),
        # 5 draw sequences with all valid values
        (
            [
                np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
                np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
                np.array([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]),
                np.array([[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
                np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
            ],
            np.array([[15, 20, 25, 30], [35, 40, 45, 50], [55, 60, 65, 70]]),
        ),
        # 15 sequences alternating valid, nodata, and NaN
        (
            [
                np.array([[1, 2, np.nan, 4], [5, -9999, 7, 8], [9, 10, 11, 12]]),
                np.array([[2, -9999, 3, 4], [-9999, 7, 8, 9], [10, 11, -9999, 13]]),
                np.array([[np.nan, 4, 5, -9999], [7, 8, 9, 10], [11, 12, 13, 14]]),
                np.array([[4, 5, 6, np.nan], [8, 9, -9999, 11], [12, 13, 14, -9999]]),
                np.array([[5, 6, np.nan, 7], [9, 10, 11, 12], [-9999, 13, 14, 15]]),
                np.array([[6, 7, 8, 9], [10, np.nan, 12, 13], [14, -9999, 15, 16]]),
                np.array([[np.nan, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]]),
                np.array([[8, 9, 10, 11], [-9999, 14, 15, 16], [17, 18, np.nan, 19]]),
                np.array([[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, -9999]]),
                np.array([[11, 12, 13, np.nan], [15, -9999, 17, 18], [19, 20, 21, 22]]),
                np.array([[12, 13, -9999, 15], [16, 17, 18, 19], [20, np.nan, 21, 22]]),
                np.array([[13, np.nan, 15, 16], [17, 18, 19, -9999], [21, 22, 23, 24]]),
                np.array([[14, 15, 16, 17], [-9999, 19, 20, 21], [22, 23, np.nan, 24]]),
                np.array([[15, 16, 17, 18], [19, np.nan, 21, 22], [-9999, 23, 24, 25]]),
                np.array([[16, 17, np.nan, 19], [20, 21, 22, 23], [24, 25, 26, -9999]]),
            ],
            np.array(
                [[117, 125, 114, 143], [151, 150, 208, 213], [212, 225, 219, 224]]
            ),
        ),
    ],
)
def test_given_draw_options_sum_drawer_expects_correct_draw_value(
    draw_sequence: list[np.ndarray],
    exp_result: np.ndarray,
    nonzero_sum_drawer: SumDrawer,
) -> None:
    for sequence in draw_sequence:
        nonzero_sum_drawer.draw(0, "b0", sequence, False)
    assert np.array_equal(nonzero_sum_drawer.data["b0"][0], exp_result)


@pytest.mark.parametrize(
    "draw_sequence, exp_result",
    [
        # Single draw with no valid values
        (
            [
                np.array(
                    [
                        [np.nan, -9999, np.nan, -9999],
                        [np.nan, -9999, np.nan, -9999],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
            ],
            np.array(
                [
                    [-9999, -9999, -9999, -9999],
                    [-9999, -9999, -9999, -9999],
                    [-9999, -9999, -9999, -9999],
                ]
            ),
        ),
        # Single draw with valid values replacing nodata
        (
            [
                np.array(
                    [
                        [1, 2, -9999, np.nan],
                        [-9999, 5, 6, np.nan],
                        [7, 8, -9999, np.nan],
                    ]
                ),
            ],
            np.array(
                [
                    [1, 2, -9999, -9999],
                    [-9999, 5, 6, -9999],
                    [7, 8, -9999, -9999],
                ]
            ),
        ),
        # Multiple draws with mixed valid and invalid values
        (
            [
                np.array(
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                    ]
                ),
                np.array(
                    [
                        [0, 3, 2, -9999],
                        [np.nan, 4, -9999, 7],
                        [8, np.nan, 10, 11],
                    ]
                ),
            ],
            np.array(
                [
                    [0, 2, 2, 4],
                    [5, 4, 7, 7],
                    [8, 10, 10, 11],
                ]
            ),
        ),
        # Sequential draws with compounding minimum values
        (
            [
                np.array([[5, 4, -9999, 3], [2, 1, -9999, np.nan], [6, 5, 4, 3]]),
                np.array([[3, 3, 2, -9999], [np.nan, 0, -9999, 4], [2, 3, 3, np.nan]]),
                np.array([[np.nan, 1, -9999, 5], [1, 2, 0, 3], [1, np.nan, 2, 2]]),
            ],
            np.array([[3, 1, 2, 3], [1, 0, 0, 3], [1, 3, 2, 2]]),
        ),
        # Draw with all `nodata` values followed by valid layer
        (
            [
                np.array(
                    [
                        [-9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999],
                    ]
                ),
                np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            ],
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
        ),
        # Four draws with a mix of nodata, nan, and valid values
        (
            [
                np.array(
                    [[5, 10, np.nan, -9999], [3, 6, -9999, -9999], [2, np.nan, 7, 8]]
                ),
                np.array([[4, 9, -9999, np.nan], [np.nan, 5, 8, -9999], [1, 8, 6, 7]]),
                np.array([[3, 12, 4, -9999], [-9999, 4, 7, np.nan], [0, np.nan, 5, 6]]),
                np.array(
                    [
                        [np.nan, 8, -9999, 4],
                        [2, np.nan, -9999, 5],
                        [np.nan, 2, np.nan, 5],
                    ]
                ),
            ],
            np.array([[3, 8, 4, 4], [2, 4, 7, 5], [0, 2, 5, 5]]),
        ),
        # Five draws with compounding minimums
        (
            [
                np.array([[10, -9999, 7, 6], [5, -9999, np.nan, 8], [4, 3, -9999, 2]]),
                np.array([[8, 9, 5, -9999], [4, 6, 7, 3], [2, -9999, 8, 1]]),
                np.array([[9, 7, -9999, 4], [6, 3, 2, -9999], [3, 5, 7, -9999]]),
                np.array([[7, 6, np.nan, 3], [-9999, np.nan, 1, 2], [1, 4, 6, 0]]),
                np.array([[6, 5, 4, 3], [5, 4, 3, 2], [0, 1, 2, 1]]),
            ],
            np.array([[6, 5, 4, 3], [4, 3, 1, 2], [0, 1, 2, 0]]),
        ),
        # Sequence with no changes after the initial valid draw
        (
            [
                np.array([[7, 6, 5, 4], [3, 2, 1, 0], [9, 8, 7, 6]]),
                np.array(
                    [
                        [np.nan, -9999, np.nan, np.nan],
                        [-9999, np.nan, np.nan, -9999],
                        [-9999, -9999, np.nan, -9999],
                    ]
                ),
                np.array(
                    [
                        [-9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999],
                    ]
                ),
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
            ],
            np.array([[7, 6, 5, 4], [3, 2, 1, 0], [9, 8, 7, 6]]),
        ),
        # Five sequential draws replacing nodata and selecting minimum
        (
            [
                np.array([[10, 15, -9999, 9], [8, -9999, 7, 12], [11, 14, -9999, 13]]),
                np.array([[9, 14, -9999, 8], [7, 10, 6, 11], [10, 13, -9999, 12]]),
                np.array([[8, 13, 7, 7], [6, 9, 5, 10], [9, 12, 8, 11]]),
                np.array([[7, 12, 6, 6], [5, 8, 4, 9], [8, 11, 7, 10]]),
                np.array([[6, 11, 5, 5], [4, 7, 3, 8], [7, 10, 6, 9]]),
            ],
            np.array([[6, 11, 5, 5], [4, 7, 3, 8], [7, 10, 6, 9]]),
        ),
        (
            [
                np.array(
                    [[10, -9999, 5, np.nan], [8, 6, -9999, 9], [11, -9999, 7, 12]]
                ),
                np.array(
                    [[9, 15, -9999, 8], [7, np.nan, 4, -9999], [np.nan, 14, 6, 10]]
                ),
                np.array([[np.nan, 12, 3, 6], [5, 10, -9999, 7], [8, -9999, 5, 9]]),
                np.array([[7, np.nan, 2, -9999], [4, 9, 8, 6], [-9999, 13, np.nan, 8]]),
                np.array([[np.nan, 11, 1, 5], [3, 8, 7, -9999], [7, 12, 4, np.nan]]),
                np.array([[6, 10, np.nan, 4], [2, 7, -9999, 5], [6, 11, 3, 7]]),
                np.array([[5, 9, 0, 3], [np.nan, 6, 7, 4], [5, 10, 2, 6]]),
                np.array([[4, 8, np.nan, 2], [1, 5, -9999, np.nan], [4, 9, 1, 5]]),
                np.array([[3, 7, -9999, 1], [0, 4, 6, 3], [3, 8, 0, 4]]),
                np.array([[2, 6, 4, 0], [-9999, 3, 5, 2], [2, 7, -9999, 3]]),
                np.array([[1, 5, np.nan, -9999], [np.nan, 2, 4, 1], [1, 6, 5, -9999]]),
                np.array([[0, 4, 3, 2], [1, 3, -9999, 0], [np.nan, 5, 4, 2]]),
                np.array([[np.nan, 3, 2, 1], [0, -9999, 3, np.nan], [0, 4, 3, 1]]),
                np.array([[2, 2, 1, 0], [np.nan, 4, 2, -9999], [1, 3, -9999, 0]]),
                np.array([[np.nan, 1, 0, np.nan], [0, 3, 1, 0], [0, 2, 1, np.nan]]),
            ],
            np.array([[0, 1, 0, 0], [0, 2, 1, 0], [0, 2, 0, 0]]),
        ),  # Fifteen draws with varying nodata, nan, and valid values
        (
            [
                np.array(
                    [[10, -9999, 5, np.nan], [8, 6, -9999, 9], [11, -9999, 7, 12]]
                ),
                np.array(
                    [[9, 15, -9999, 8], [7, np.nan, 4, -9999], [np.nan, 14, 6, 10]]
                ),
                np.array([[np.nan, 12, 3, 6], [5, 10, -9999, 7], [8, -9999, 5, 9]]),
                np.array([[7, np.nan, 2, -9999], [4, 9, 8, 6], [-9999, 13, np.nan, 8]]),
                np.array([[np.nan, 11, 1, 5], [3, 8, 7, -9999], [7, 12, 4, np.nan]]),
                np.array([[6, 10, np.nan, 4], [2, 7, -9999, 5], [6, 11, 3, 7]]),
                np.array([[5, 9, 0, 3], [np.nan, 6, 7, 4], [5, 10, 2, 6]]),
                np.array([[4, 8, np.nan, 2], [1, 5, -9999, np.nan], [4, 9, 1, 5]]),
                np.array([[3, 7, -9999, 1], [0, 4, 6, 3], [3, 8, 0, 4]]),
                np.array([[2, 6, 4, 0], [-9999, 3, 5, 2], [2, 7, -9999, 3]]),
                np.array([[1, 5, np.nan, -9999], [np.nan, 2, 4, 1], [1, 6, 5, -9999]]),
                np.array([[0, 4, 3, 2], [1, 3, -9999, 0], [np.nan, 5, 4, 2]]),
                np.array([[np.nan, 3, 2, 1], [0, -9999, 3, np.nan], [0, 4, 3, 1]]),
                np.array([[2, 2, 1, 0], [np.nan, 4, 2, -9999], [1, 3, -9999, 0]]),
                np.array([[np.nan, 1, 0, np.nan], [0, 3, 1, 0], [0, 2, 1, np.nan]]),
            ],
            np.array([[0, 1, 0, 0], [0, 2, 1, 0], [0, 2, 0, 0]]),
        ),
    ],
)
def test_given_draw_options_min_drawer_expects_correct_draw_value(
    draw_sequence: list[np.ndarray],
    exp_result: np.ndarray,
    nonzero_min_drawer: MinDrawer,
) -> None:
    for sequence in draw_sequence:
        nonzero_min_drawer.draw(0, "b0", sequence, False)
    assert np.array_equal(nonzero_min_drawer.data["b0"][0], exp_result)


@pytest.fixture()
def max_drawer() -> MaxDrawer:
    return MaxDrawer(
        x_coords=np.arange(3),
        y_coords=np.arange(4),
        t_coords=np.arange(2),
        bands={"b0"},
        nodata=-9999,
    )


@pytest.mark.parametrize(
    "draw_sequence, exp_result",
    [
        (
            [
                np.array(
                    [[1, -9999, np.nan, 4], [-9999, 6, 7, np.nan], [9, 10, -9999, 12]]
                ),
                np.array([[2, np.nan, 3, -9999], [4, 5, np.nan, 6], [-9999, 8, 9, 10]]),
            ],
            np.array([[2, -9999, 3, 4], [4, 6, 7, 6], [9, 10, 9, 12]]),
        ),
        (
            [
                np.array([[np.nan, 2, 3, 4], [5, np.nan, 7, 8], [9, 10, np.nan, 12]]),
                np.array([[4, -9999, 5, 6], [-9999, 7, 8, 9], [10, np.nan, 11, 12]]),
            ],
            np.array([[4, 2, 5, 6], [5, 7, 8, 9], [10, 10, 11, 12]]),
        ),
        # 4 Sequences Test Case
        (
            [
                np.array(
                    [[1, -9999, np.nan, 4], [-9999, 6, 7, np.nan], [9, 10, -9999, 12]]
                ),
                np.array([[2, np.nan, 3, -9999], [4, 5, np.nan, 6], [-9999, 8, 9, 10]]),
                np.array([[3, 2, 3, 4], [5, np.nan, 7, 8], [9, 10, 11, 12]]),
                np.array([[4, 5, 5, 6], [6, 7, 8, 9], [10, 11, 12, 13]]),
            ],
            np.array([[4, 5, 5, 6], [6, 7, 8, 9], [10, 11, 12, 13]]),
        ),
        # 5 Sequences Test Case
        (
            [
                np.array(
                    [[1, -9999, np.nan, 4], [-9999, 6, 7, np.nan], [9, 10, -9999, 12]]
                ),
                np.array([[2, np.nan, 3, -9999], [4, 5, np.nan, 6], [-9999, 8, 9, 10]]),
                np.array([[np.nan, 2, 3, 4], [5, np.nan, 7, 8], [9, 10, np.nan, 12]]),
                np.array([[4, -9999, 5, 6], [-9999, 7, 8, 9], [10, np.nan, 11, 12]]),
                np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
            ],
            np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
        ),
        # 15 Sequences Test Case
        (
            [
                np.array(
                    [[1, -9999, np.nan, 4], [-9999, 6, 7, np.nan], [9, 10, -9999, 12]]
                ),
                np.array([[2, np.nan, 3, -9999], [4, 5, np.nan, 6], [-9999, 8, 9, 10]]),
                np.array([[3, 2, 3, 4], [5, np.nan, 7, 8], [9, 10, 11, 12]]),
                np.array([[4, 5, 5, 6], [6, 7, 8, 9], [10, 11, 12, 13]]),
                np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
                np.array([[6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17]]),
                np.array([[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]]),
                np.array([[8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]]),
                np.array([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]),
                np.array([[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]),
                np.array([[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]]),
                np.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]),
                np.array([[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]),
                np.array([[14, 15, 16, 17], [18, 19, 20, 21], [22, 23, 24, 25]]),
                np.array([[15, 16, 17, 18], [19, 20, 21, 22], [23, 24, 25, 26]]),
            ],
            np.array([[15, 16, 17, 18], [19, 20, 21, 22], [23, 24, 25, 26]]),
        ),
        # Edge Case: Sequences with only nodata values
        (
            [
                np.array(
                    [
                        [-9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999],
                    ]
                ),
                np.array(
                    [
                        [-9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999],
                        [-9999, -9999, -9999, -9999],
                    ]
                ),
            ],
            np.array(
                [
                    [-9999, -9999, -9999, -9999],
                    [-9999, -9999, -9999, -9999],
                    [-9999, -9999, -9999, -9999],
                ]
            ),
        ),
        # Edge Case: Sequences with no nodata, only non-nodata values
        (
            [
                np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
                np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
            ],
            np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
        ),
        # Case with alternating nodata and valid values
        (
            [
                np.array([[1, -9999, 3, -9999], [5, -9999, 7, 8], [9, 10, -9999, 12]]),
                np.array([[-9999, 2, -9999, 4], [6, 7, 8, 9], [10, 11, -9999, 13]]),
                np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            ],
            np.array([[1, 2, 3, 4], [6, 7, 8, 9], [10, 11, 11, 13]]),
        ),
        # 3 Sequences with increasing values
        (
            [
                np.array(
                    [[1, 2, np.nan, 4], [-9999, 6, 7, np.nan], [9, 10, -9999, 12]]
                ),
                np.array([[2, np.nan, 3, -9999], [4, 5, np.nan, 6], [-9999, 8, 9, 10]]),
                np.array([[np.nan, 2, 3, 4], [5, np.nan, 7, 8], [9, 10, np.nan, 12]]),
            ],
            np.array([[2, 2, 3, 4], [5, 6, 7, 8], [9, 10, 9, 12]]),
        ),
        # Mixed Nodata and Large Values
        (
            [
                np.array(
                    [
                        [1, 2, -9999, 4],
                        [-9999, 6, 7, np.nan],
                        [9, 10, -9999, 12],
                    ]
                ),
                np.array(
                    [
                        [1000, np.nan, 1000, -9999],
                        [1000, 1000, np.nan, 1000],
                        [-9999, 1000, 1000, 1000],
                    ]
                ),
                np.array(
                    [
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ),
            ],
            np.array(
                [
                    [1000, 6, 1000, 8],
                    [1000, 1000, 11, 1000],
                    [13, 1000, 1000, 1000],
                ]
            ),
        ),
        # Edge Case: Sequence with NaN values
        (
            [
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
            ],
            np.array(
                [
                    [-9999, -9999, -9999, -9999],
                    [-9999, -9999, -9999, -9999],
                    [-9999, -9999, -9999, -9999],
                ]
            ),
        ),
        # Sequence with descending values
        (
            [
                np.array([[10, 9, 8, 7], [6, 5, 4, 3], [2, 1, 0, -1]]),
                np.array([[9, 8, 7, 6], [5, 4, 3, 2], [1, 0, -1, -2]]),
                np.array([[8, 7, 6, 5], [4, 3, 2, 1], [0, -1, -2, -3]]),
            ],
            np.array([[10, 9, 8, 7], [6, 5, 4, 3], [2, 1, 0, -1]]),
        ),
        # Sequence with 5 draws and non-overlapping values
        (
            [
                np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
                np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
                np.array([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]),
                np.array([[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
                np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
            ],
            np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
        ),
        # Sequence with 5 draws and random values
        (
            [
                np.array([[1, 2, 3, -9999], [5, -9999, 7, 8], [9, 10, -9999, 12]]),
                np.array([[-9999, 2, 3, 4], [6, 7, -9999, 8], [9, -9999, 11, 12]]),
                np.array([[1, -9999, 3, 4], [5, 6, 7, -9999], [9, 10, 11, 12]]),
                np.array([[2, -9999, 3, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
                np.array([[3, -9999, 4, 6], [7, 8, 9, 10], [11, 12, 13, 14]]),
            ],
            np.array([[3, 2, 4, 6], [7, 8, 9, 10], [11, 12, 13, 14]]),
        ),
    ],
)
def test_given_draw_options_expects_correct_max_draw_value(
    draw_sequence: list[np.ndarray],
    exp_result: np.ndarray,
    max_drawer: MaxDrawer,
) -> None:
    for sequence in draw_sequence:
        max_drawer.draw(0, "b0", sequence, False)
    assert np.array_equal(max_drawer.data["b0"][0], exp_result)


@pytest.fixture()
def replace_drawer() -> ReplaceDrawer:
    return ReplaceDrawer(
        x_coords=np.arange(3),
        y_coords=np.arange(4),
        t_coords=np.arange(2),
        bands={"b0"},
        nodata=-9999,
    )


@pytest.mark.parametrize(
    "draw_sequence, exp_result",
    [
        (
            [
                np.array(
                    [
                        [1, -9999, np.nan, 4],
                        [-9999, 6, 7, np.nan],
                        [9, 10, -9999, 12],
                    ]
                ),
                np.array(
                    [
                        [2, np.nan, 3, -9999],
                        [4, 5, np.nan, 6],
                        [-9999, 8, 9, 10],
                    ]
                ),
            ],
            np.array(
                [
                    [2, -9999, 3, 4],
                    [4, 5, 7, 6],
                    [9, 8, 9, 10],
                ]
            ),
        ),
        (
            [
                np.array([[np.nan, 2, 3, 4], [5, np.nan, 7, 8], [9, 10, np.nan, 12]]),
                np.array([[4, -9999, 5, 6], [-9999, 7, 8, 9], [10, np.nan, 11, 12]]),
            ],
            np.array([[4, 2, 5, 6], [5, 7, 8, 9], [10, 10, 11, 12]]),
        ),
        # Single replacement where only non-nodata, non-NaN values are replaced
        (
            [
                np.array(
                    [[1, -9999, np.nan, 4], [-9999, 6, 7, np.nan], [9, 10, -9999, 12]]
                ),
                np.array(
                    [[-9999, 2, 3, -9999], [4, np.nan, 5, 6], [-9999, 8, 9, -9999]]
                ),
            ],
            np.array([[1, 2, 3, 4], [4, 6, 5, 6], [9, 8, 9, 12]]),
        ),
        # Three layers where values progressively replace nodata or existing values
        (
            [
                np.array(
                    [[1, 2, np.nan, -9999], [-9999, 6, 7, np.nan], [9, 10, -9999, 12]]
                ),
                np.array(
                    [[-9999, -9999, 3, 4], [5, np.nan, -9999, 6], [13, -9999, 14, 15]]
                ),
                np.array(
                    [[16, 17, -9999, 18], [19, 20, 21, 22], [-9999, 23, 24, -9999]]
                ),
            ],
            np.array([[16, 17, 3, 18], [19, 20, 21, 22], [13, 23, 24, 15]]),
        ),
        # Four-layer replacement with alternating nodata and values
        (
            [
                np.array(
                    [[1, -9999, 3, np.nan], [5, 6, -9999, 8], [-9999, 10, 11, 12]]
                ),
                np.array(
                    [[-9999, 14, np.nan, 16], [-9999, 18, 19, 20], [21, 22, -9999, 24]]
                ),
                np.array(
                    [[25, np.nan, -9999, 26], [27, 28, 29, np.nan], [-9999, 30, 31, 32]]
                ),
                np.array(
                    [[33, 34, 35, -9999], [36, 37, np.nan, 38], [39, 40, 41, -9999]]
                ),
            ],
            np.array([[33, 34, 35, 26], [36, 37, 29, 38], [39, 40, 41, 32]]),
        ),
        # Fifteen-layer scenario with mixed replacements
        (
            [
                np.array([[1, 2, 3, 4], [-9999, 6, 7, 8], [9, 10, -9999, 12]]),
                np.array(
                    [[13, 14, 15, -9999], [16, np.nan, 18, 19], [20, 21, 22, -9999]]
                ),
                np.array([[23, 24, -9999, 25], [26, 27, 28, 29], [-9999, 30, 31, 32]]),
                np.array([[33, 34, 35, 36], [37, -9999, 39, 40], [41, 42, -9999, 43]]),
                np.array(
                    [[44, np.nan, 46, -9999], [47, 48, 49, 50], [51, 52, 53, -9999]]
                ),
                np.array([[54, 55, 56, 57], [-9999, 58, 59, 60], [61, np.nan, 63, 64]]),
                np.array([[65, 66, 67, -9999], [68, 69, 70, 71], [72, 73, 74, 75]]),
                np.array([[76, 77, np.nan, 78], [79, 80, 81, -9999], [82, 83, 84, 85]]),
                np.array([[86, 87, 88, 89], [90, np.nan, 92, 93], [-9999, 94, 95, 96]]),
                np.array(
                    [[97, 98, 99, -9999], [100, 101, 102, 103], [104, 105, np.nan, 106]]
                ),
                np.array(
                    [
                        [107, 108, -9999, 109],
                        [110, 111, 112, -9999],
                        [113, 114, 115, 116],
                    ]
                ),
                np.array(
                    [
                        [117, 118, 119, 120],
                        [-9999, 121, 122, 123],
                        [124, np.nan, 125, 126],
                    ]
                ),
                np.array(
                    [[127, 128, 129, 130], [131, 132, 133, -9999], [134, 135, 136, 137]]
                ),
                np.array(
                    [
                        [138, 139, -9999, 140],
                        [141, 142, 143, 144],
                        [-9999, 145, 146, 147],
                    ]
                ),
                np.array(
                    [[148, 149, 150, 151], [152, 153, 154, -9999], [155, 156, 157, 158]]
                ),
            ],
            np.array(
                [[148, 149, 150, 151], [152, 153, 154, 144], [155, 156, 157, 158]]
            ),
        ),
    ],
)
def test_given_draw_options_expects_correct_replace_draw_value(
    draw_sequence: list[np.ndarray],
    exp_result: np.ndarray,
    replace_drawer: ReplaceDrawer,
) -> None:
    for sequence in draw_sequence:
        replace_drawer.draw(0, "b0", sequence, False)
    assert np.array_equal(replace_drawer.data["b0"][0], exp_result)


@pytest.fixture()
def nonzero_mean_drawer() -> MeanDrawer:
    return MeanDrawer(
        x_coords=np.arange(3),
        y_coords=np.arange(4),
        t_coords=np.arange(2),
        bands={"b0"},
        nodata=-9999,
    )


@pytest.mark.parametrize(
    "draw_sequence, exp_result",
    [
        # Single layer mean calculation
        (
            [
                np.array(
                    [[1, -9999, np.nan, 4], [-9999, 6, 7, np.nan], [9, 10, -9999, 12]]
                ),
            ],
            np.array([[1, -9999, -9999, 4], [-9999, 6, 7, -9999], [9, 10, -9999, 12]]),
        ),
        # Two layers mean calculation
        (
            [
                np.array(
                    [[1, 2, np.nan, 4], [5, -9999, 7, np.nan], [9, 10, -9999, 12]]
                ),
                np.array(
                    [[2, np.nan, 3, 5], [-9999, np.nan, 8, 9], [10, 11, -9999, 13]]
                ),
            ],
            np.array([[1.5, 2, 3, 4.5], [5, -9999, 7.5, 9], [9.5, 10.5, -9999, 12.5]]),
        ),
        # Three layers mean calculation
        (
            [
                np.array([[1, 2, 3, 4], [-9999, 6, 7, 8], [9, 10, -9999, 12]]),
                np.array([[2, -9999, 4, 5], [6, -9999, 8, 9], [10, 11, 12, 13]]),
                np.array([[3, 4, -9999, np.nan], [7, 8, 9, 10], [-9999, 12, 13, 14]]),
            ],
            np.array([[2, 3, 3.5, 4.5], [6.5, 7, 8, 9], [9.5, 11, 12.5, 13]]),
        ),
        # Longer sequence (4 layers)
        (
            [
                np.array([[1, -9999, np.nan, 4], [-9999, 6, 7, 8], [9, 10, -9999, 12]]),
                np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
                np.array([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]),
                np.array([[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
            ],
            np.array([[2.5, 4, 5, 5.5], [7, 7.5, 8.5, 9.5], [10.5, 11.5, 13, 13.5]]),
        ),
        # Long sequence (15 layers) with mixed nodata
        (
            [
                np.array([[1, 2, -9999, 4], [-9999, 6, 7, 8], [9, 10, -9999, 12]]),
                np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
                np.array([[3, 4, -9999, 6], [7, 8, 9, 10], [-9999, 12, 13, 14]]),
                np.array([[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
                np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
                np.array([[6, 7, -9999, 9], [10, 11, 12, 13], [-9999, 15, 16, 17]]),
                np.array([[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]]),
                np.array([[8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]]),
                np.array([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]),
                np.array([[10, 11, -9999, 13], [14, 15, 16, 17], [-9999, 19, 20, 21]]),
                np.array([[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]]),
                np.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]),
                np.array([[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]),
                np.array([[14, 15, 16, 17], [18, 19, 20, 21], [22, 23, 24, 25]]),
                np.array([[15, 16, 17, 18], [19, 20, 21, 22], [23, 24, 25, 26]]),
            ],
            np.array(
                [
                    [8.0, 9.0, 11.09090909, 11.0],
                    [12.5, 13.0, 14.0, 15.0],
                    [16.41666667, 17.0, 18.5, 19.0],
                ]
            ),
        ),
        # Case 1: Simple 2-layer sequence with some nodata and NaN values
        (
            [
                np.array([[1, -9999, 3, 4], [5, np.nan, 7, 8], [9, 10, -9999, 12]]),
                np.array([[2, 3, np.nan, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
            ],
            np.array([[1.5, 3, 3, 4.5], [5.5, 7, 7.5, 8.5], [9.5, 10.5, 12, 12.5]]),
        ),
        # Case 2: Sequence with 3 layers and varying nodata and NaN
        (
            [
                np.array([[1, 2, -9999, 4], [5, np.nan, 7, 8], [9, 10, 11, -9999]]),
                np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
                np.array([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]),
            ],
            np.array([[2, 3, 4.5, 5], [6, 7.5, 8, 9], [10, 11, 12, 13.5]]),
        ),
        # Case 3: Sequence with all NaN or nodata for some positions
        (
            [
                np.array([[1, -9999, 3, 4], [-9999, 6, np.nan, 8], [9, 10, 11, 12]]),
                np.array([[np.nan, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, np.nan]]),
                np.array([[2, 3, np.nan, 5], [6, np.nan, 9, 9], [10, 12, -9999, 13]]),
            ],
            np.array(
                [
                    [1.5, 3, 3.5, 4.6667],
                    [6, 6.5, 8.5, 8.667],
                    [9.667, 11, 11.5, 12.5],
                ]
            ),
        ),
        # Case 4: Longer sequence with 5 layers
        (
            [
                np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
                np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
                np.array([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]),
                np.array([[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
                np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
            ],
            np.array([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]),
        ),
        # Case 5: Sequence with 6 layers and sparse valid data
        (
            [
                np.array([[1, np.nan, 3, 4], [5, -9999, 7, 8], [9, 10, 11, 12]]),
                np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]),
                np.array([[-9999, 4, 5, 6], [7, 8, -9999, 10], [11, 12, 13, 14]]),
                np.array([[3, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
                np.array([[4, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
                np.array([[5, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17]]),
            ],
            np.array([[3, 5, 5.5, 6.5], [7.5, 9, 9.6, 10.5], [11.5, 12.5, 13.5, 14.5]]),
        ),
    ],
)
def test_mean_drawer_scenarios(
    draw_sequence: list[np.ndarray],
    exp_result: np.ndarray,
    nonzero_mean_drawer: MeanDrawer,
) -> None:
    for sequence in draw_sequence:
        nonzero_mean_drawer.draw(0, "b0", sequence, False)
    assert np.allclose(
        nonzero_mean_drawer.data["b0"][0], exp_result, equal_nan=True, atol=1e-3
    )
