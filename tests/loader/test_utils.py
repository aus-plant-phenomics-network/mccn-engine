import numpy as np
import pytest

from mccn.loader.utils import get_neighbor_mask, mask_aggregate


@pytest.mark.parametrize(
    "points, values, radius, op, exp_mask, exp_agg",
    [
        # Points fall exactly at grid point - max
        (
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([1, 2, 3]),
            1e-3,
            "max",
            np.array(
                [
                    [
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, np.nan, np.nan, np.nan],
                    [np.nan, 2, np.nan, np.nan],
                    [np.nan, np.nan, 3, np.nan],
                ],
                dtype=np.float32,
            ),
        ),
        # Points fall exactly at grid point - mean
        (
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([0, 1, 2]),
            1e-3,
            "mean",
            np.array(
                [
                    [
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [0, np.nan, np.nan, np.nan],
                    [np.nan, 1, np.nan, np.nan],
                    [np.nan, np.nan, 2, np.nan],
                ],
                dtype=np.float32,
            ),
        ),
        # Some points on, some points off grid - max
        (
            np.array([[0, 0], [0.5, 0.5], [1.5, 1.5], [1, 1]]),
            np.array([1, 2, 3, 4]),
            1e-3,
            "max",
            np.array(
                [
                    [
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, np.nan, np.nan, np.nan],
                    [np.nan, 4, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                ],
                dtype=np.float32,
            ),
        ),
        # Some points on, some points off grid - max
        (
            np.array([[0, 0], [0.5, 0.5], [1.5, 1.5], [1, 1]]),
            np.array([1, 2, 3, 4]),
            1e-3,
            "mean",
            np.array(
                [
                    [
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, np.nan, np.nan, np.nan],
                    [np.nan, 4, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                ],
                dtype=np.float32,
            ),
        ),
        # Points fall in the middle between grid points, radius is distance to the closest grid point - max
        (
            np.array([[0, 0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5]]),
            np.array([1, 2, 3, 4]),
            0.8,
            "max",
            np.array(
                [
                    [
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 1, 0, 0],
                        [0, 1, 1, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [2, 2, np.nan, np.nan],
                    [2, 4, 4, np.nan],
                    [np.nan, 4, 4, np.nan],
                ],
                dtype=np.float32,
            ),
        ),
        # Points fall in the middle between grid points, radius is distance to the closest grid point - mean
        (
            np.array([[0, 0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5]]),
            np.array([1, 2, 3, 4]),
            0.8,
            "mean",
            np.array(
                [
                    [
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 1, 0, 0],
                        [0, 1, 1, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1.5, 2, np.nan, np.nan],
                    [2, 3, 4, np.nan],
                    [np.nan, 4, 4, np.nan],
                ],
                dtype=np.float32,
            ),
        ),
        # Points fall in the middle between grid points, radius is distance to the closest grid point - max - nan values
        (
            np.array([[0, 0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5]]),
            np.array([1, np.nan, 3, 4]),
            0.8,
            "max",
            np.array(
                [
                    [
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 1, 0, 0],
                        [0, 1, 1, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, np.nan, np.nan, np.nan],
                    [np.nan, 4, 4, np.nan],
                    [np.nan, 4, 4, np.nan],
                ],
                dtype=np.float32,
            ),
        ),
        # Points fall in the middle between grid points, radius is distance to the closest grid point - mean - nan values
        (
            np.array([[0, 0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5]]),
            np.array([1, np.nan, 3, 4]),
            0.8,
            "mean",
            np.array(
                [
                    [
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 1, 0, 0],
                        [0, 1, 1, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, np.nan, np.nan, np.nan],
                    [np.nan, 3.5, 4, np.nan],
                    [np.nan, 4, 4, np.nan],
                ],
                dtype=np.float32,
            ),
        ),
        # No points
        (
            np.empty((0, 2)),  # Empty points array
            np.empty((0, 2)),
            1.0,
            "max",
            np.zeros((3, 4, 0), dtype=bool),  # Empty mask with correct shape
            np.full((3, 4), np.nan),
        ),
        # All points outside the radius
        (
            np.array([[10, 10], [-10, -10]]),  # Points far outside the grid
            np.array([1, 2]),
            1.0,  # Radius too small to include any points
            "max",
            np.zeros((3, 4, 2), dtype=bool),  # No points within radius
            np.full((3, 4), np.nan),
        ),
        # Large radius encompassing all points - max
        (
            np.array([[0, 0], [1, 1], [2, 2], [1, 3]]),  # Points on and off grid
            np.array([1, 2, 3, 4]),
            5.0,  # Large radius
            "max",
            np.ones((3, 4, 4), dtype=bool),  # All grid cells include all points
            np.array(
                [
                    [4, 4, 4, 4],
                    [4, 4, 4, 4],
                    [4, 4, 4, 4],
                ],
                dtype=np.float32,
            ),
        ),
        # Large radius encompassing all points - mean
        (
            np.array([[0, 0], [1, 1], [2, 2], [1, 3]]),  # Points on and off grid
            np.array([1, 2, 3, 4]),
            5.0,  # Large radius
            "mean",
            np.ones((3, 4, 4), dtype=bool),  # All grid cells include all points
            np.array(
                [
                    [2.5, 2.5, 2.5, 2.5],
                    [2.5, 2.5, 2.5, 2.5],
                    [2.5, 2.5, 2.5, 2.5],
                ],
                dtype=np.float32,
            ),
        ),
        # Large radius encompassing all points - max - with nan
        (
            np.array([[0, 0], [1, 1], [2, 2], [1, 3]]),  # Points on and off grid
            np.array([1, np.nan, 3, np.nan]),
            5.0,  # Large radius
            "max",
            np.ones((3, 4, 4), dtype=bool),  # All grid cells include all points
            np.array(
                [
                    [3, 3, 3, 3],
                    [3, 3, 3, 3],
                    [3, 3, 3, 3],
                ],
                dtype=np.float32,
            ),
        ),
        # Large radius encompassing all points - mean - with nan
        (
            np.array([[0, 0], [1, 1], [2, 2], [1, 3]]),  # Points on and off grid
            np.array([1, np.nan, 3, np.nan]),
            5.0,  # Large radius
            "mean",
            np.ones((3, 4, 4), dtype=bool),  # All grid cells include all points
            np.array(
                [
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_get_neighbor_mask(
    points: np.ndarray,
    values: np.ndarray,
    radius: float,
    op: str,
    exp_mask: np.ndarray,
    exp_agg: np.ndarray,
) -> None:
    mask = get_neighbor_mask(
        gx=np.arange(3), gy=np.arange(4), points=points, radius=radius
    )
    np.testing.assert_array_equal(mask, exp_mask)
    _op = np.nanmax if op == "max" else np.nanmean
    value = mask_aggregate(values, mask, _op)
    np.testing.assert_array_almost_equal(value, exp_agg, 6)
