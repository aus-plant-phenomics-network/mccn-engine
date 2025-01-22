from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.xr import xr_coords
from stac_generator.point.generator import read_csv

from mccn.loader.utils import (
    ASSET_KEY,
    BBOX_TOL,
    StacExtensionError,
    get_item_crs,
    get_item_href,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pystac
    from odc.geo.geobox import GeoBox

    from mccn._types import InterpMethods, MergeMethods


def get_required_columns(
    item: pystac.Item,
    fields: Sequence[str] | None = None,
) -> list[str]:
    """Get the requested columns from `fields` that are available in item.

    STAC Generator automatically generates `column_info` property (for csv point data)
    which contains a list of fields/attributes in the source asset. This function determines
    the subset of requested `fields` that is available in the source asset. This allows for
    efficient data loading as only relevant columns are read into memory instead of the whole
    data frame. This should work for both point and vector data type, as long as `column_info`
    is present.

    :param item: STAC item metadata
    :type item: pystac.Item
    :param fields: list of requested columns, some of which if not all may not be available
    in the source asset, defaults to None
    :type fields: Sequence[str] | None, optional
    :raises StacExtensionError: when column_info field is not found in item properties
    :return: subset of fields as list that can be read from source asset of item
    :rtype: list[str]
    """
    if "column_info" not in item.properties:
        raise StacExtensionError("No column info found in STAC metadata")
    if not fields:
        return [column["name"] for column in item.properties["column_info"]]
    return [
        column["name"]
        for column in item.properties["column_info"]
        if column["name"] in fields
    ]


def query_geobox(
    frame: gpd.GeoDataFrame,
    geobox: GeoBox,
    tol: float = BBOX_TOL,
) -> gpd.GeoDataFrame:
    """Constrain geodataframe to region specified by `geobox` and convert to `geobox` crs

    This method works on both point GeoDataFrame and vector GeoDataFrame.

    :param frame: input dataframe
    :type frame: gpd.GeoDataFrame
    :param geobox: input geobox
    :type geobox: GeoBox
    :param tol: tolerance for geobox bounding box query, defaults to BBOX_TOL
    :type tol: float, optional
    :return: constrainted and crs transformed geodataframe
    :rtype: gpd.GeoDataFrame
    """
    frame.to_crs(geobox.crs, inplace=True)
    left_right = slice(float(geobox.boundingbox[0]), float(geobox.boundingbox[2] + tol))
    bottom_top = slice(float(geobox.boundingbox[1]), float(geobox.boundingbox[3] + tol))
    return frame.cx[left_right, bottom_top]


def read_point_asset(
    item: pystac.Item,
    fields: Sequence[str] | None = None,
    asset_key: str | Mapping[str, str] = ASSET_KEY,
    t_col: str = "time",
) -> gpd.GeoDataFrame | None:
    """Read asset described in item STAC metadata.

    This function looks for the asset in item assets under `asset_key` and raise an error if it does not have a csv
    extension. The function then extract metadata required for processing point asset (the same metadata used in stac_generator).
    This metadata should be found in item properties, and if not found, an error will be raised. The function then loads
    the csv into a `GeoDataFrame` and rename the columns for longitude/X, latitude/Y, time/T (may not be present) with values
    from `x_col`, `y_col`, `t_col`. Note that if there is no T column in the original dataframe, a time column will be appended based
    on the `datetime` field of stac metadata. This ensures consistency with odc stac load for merging.

    :param item: item STAC metadata as pystac.Item object
    :type item: pystac.Item
    :param fields: columns in dataframe to read in. If fields is None, read all columns. If a list of fields are provided, will read
    the set of fields that are available in the dataframe. If no fields is available, return None. defaults to None
    :type fields: Sequence[str] | None, optional
    :param asset_key: key in assets of the source asset that the item describes, defaults to ASSET_KEY
    :type asset_key: str, optional
    :param t_col: renamed t column for merging consistency, defaults to "time"
    :type t_col: str, optional
    :raises StacExtensionError: if required metadata for processing asset csv is not provided in item properties
    :return: geopandas dataframe or None if the fields requested are not present in dataframe
    :rtype: gpd.GeoDataFrame | None
    """
    try:
        # Process metadata
        location = get_item_href(item, asset_key)
        columns = get_required_columns(item, fields)
        if not columns:
            return None
        epsg = get_item_crs(item)
        (
            X_name,
            Y_name,
            Z_name,
            T_name,
        ) = (
            item.properties["X"],
            item.properties["Y"],
            item.properties.get("Z", None),
            item.properties.get("T", None),
        )

        # Read csv
        gdf = read_csv(
            src_path=location,
            X_coord=X_name,
            Y_coord=Y_name,
            epsg=epsg,  # type: ignore[arg-type]
            T_coord=T_name,
            date_format=item.properties.get("date_format", "ISO8601"),
            Z_coord=Z_name,
            columns=columns,
        )
        # Rename T column for uniformity
        if T_name is None:
            T_name = t_col
            gdf[T_name] = item.datetime
        gdf.rename(columns={T_name: t_col}, inplace=True)
        # Drop X and Y columns since we will repopulate them after changing crs
        gdf.drop(columns=[X_name, Y_name], inplace=True)
    except KeyError as e:
        raise StacExtensionError("Missing field in stac config:") from e
    return gdf


def process_groupby(
    frame: gpd.GeoDataFrame,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
    merge_method: MergeMethods = "mean",
) -> gpd.GeoDataFrame:
    frame[x_col] = frame.geometry.x
    frame[y_col] = frame.geometry.y

    # Prepare aggregation method
    excluding_fields = set([x_col, y_col, t_col, "geometry"])
    fields = [name for name in frame.columns if name not in excluding_fields]
    # field_map determines replacement strategy for each field when there is a conflict
    field_map = (
        {field: merge_method[field] for field in fields if field in merge_method}
        if isinstance(merge_method, dict)
        else {field: merge_method for field in fields}
    )

    # Groupby + Aggregate
    return frame.groupby([t_col, y_col, x_col]).agg(field_map)


def point_data_to_xarray(
    frame: pd.DataFrame,
    geobox: GeoBox,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
    interp_method: InterpMethods | None = "nearest",
) -> xr.Dataset:
    ds: xr.Dataset = frame.to_xarray()
    # Sometime to_xarray bugs out with datetime index so need explicit conversion
    ds[t_col] = pd.DatetimeIndex(ds.coords[t_col].values)
    if interp_method is None:
        return ds
    coords_ = xr_coords(geobox, dims=(y_col, x_col))
    ds = ds.assign_coords(spatial_ref=coords_["spatial_ref"])
    coords = {y_col: coords_[y_col], x_col: coords_[x_col]}
    return ds.interp(coords=coords, method=interp_method)


def stac_load_point(
    items: Sequence[pystac.Item],
    geobox: GeoBox,
    asset_key: str | Mapping[str, str] = ASSET_KEY,
    fields: Sequence[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
    merge_method: MergeMethods = "mean",
    interp_method: InterpMethods | None = "nearest",
) -> xr.Dataset:
    frames = []
    for item in items:
        frame = read_point_asset(item, fields, asset_key, t_col)
        if frame is not None:  # Can be None if does not contain required column
            frame = frame.to_crs(geobox.crs)
            # Process groupby - i.e. average out over depth, duplicate entries, etc
            merged = process_groupby(frame, x_col, y_col, t_col, merge_method)
            frames.append(
                point_data_to_xarray(merged, geobox, x_col, y_col, t_col, interp_method)
            )
    return xr.merge(frames)
