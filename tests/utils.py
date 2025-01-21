import datetime
import math
from collections.abc import Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from odc.geo.geobox import GeoBox

from mccn.extent import GeoBoxBuilder


def get_geobox(
    frames: Sequence[gpd.GeoDataFrame],
    crs: Any = 4326,
    shape_x: int = 100,
    shape_y: int = 100,
) -> GeoBox:
    left, bottom, right, top = frames[0].total_bounds
    for frame in frames:
        left, bottom = (
            min([left, frame.total_bounds[0]]),
            min([bottom, frame.total_bounds[1]]),
        )
        right, top = (
            max([right, frame.total_bounds[2]]),
            max([top, frame.total_bounds[3]]),
        )
    return (
        GeoBoxBuilder(crs=crs, anchor="center")
        .set_bbox((left, bottom, right, top))
        .set_shape(shape_x, shape_y)
        .build()
    )


def generate_test_point_data(
    x: tuple[float, float] = (0, 10),
    y: tuple[float, float] = (0, 10),
    shape: int | tuple[int, int] = 10,
    z: tuple[float, float] | None = None,
    shape_z: int = 1,
    t: tuple[datetime.datetime, datetime.datetime] | tuple[str, str] | None = None,
    shape_t: int = 1,
    columns: Sequence[str] = ["radiation", "rainfall"],
    fill_pct: float = 0.2,
    crs: int = 4326,
) -> gpd.GeoDataFrame:
    if isinstance(shape, int):
        shape = (shape, shape)
    X = np.linspace(x[0], x[1], shape[0])
    Y = np.linspace(y[0], y[1], shape[1])
    Z = np.linspace(z[0], z[1], shape_z) if z is not None else None
    T = pd.date_range(t[0], t[1], shape_t) if t is not None else None
    N = math.ceil(shape[0] * shape[1] * fill_pct)
    N_rep = shape_t * shape_z
    data = {
        "X": np.repeat(np.random.choice(X, N), N_rep),
        "Y": np.repeat(np.random.choice(Y, N), N_rep),
    }
    if Z is not None:
        data["Z"] = np.tile(np.repeat(Z, shape_t), N)
    if T is not None:
        data["T"] = np.tile(T, shape_z * N)
    for col in columns:
        data[col] = np.random.rand(N * N_rep)
    return gpd.GeoDataFrame(
        pd.DataFrame(data), geometry=gpd.points_from_xy(data["X"], data["Y"], crs=crs)
    )
