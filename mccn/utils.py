from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Mapping, cast
from warnings import warn

import geopandas as gpd
import pandas as pd
import pystac
import xarray as xr
from odc.geo.xr import xr_coords
from pandas.api.types import is_datetime64_any_dtype
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from stac_generator.point.generator import read_csv

from mccn._types import BBox_T

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox

    from mccn._types import InterpMethods, MergeMethods
    from mccn.extent import TimeBox


class StacExtensionError(Exception): ...


ASSET_KEY = "data"
BBOX_TOL = 1e-10

# Data format Agnostic Utilities


def get_item_href(
    item: pystac.Item,
    asset_key: str | Mapping[str, str],
) -> str:
    """Get the href of the source asset in item.

    Source asset is the source file which the stac item primarily describes. This
    is to differentiate from other assets which might serve as summary for the source
    asset. The source asset should be indexed by asset key, which can either be a string,
    or a dictionary that maps item id to source asset key.

    :param item: stac item
    :type item: pystac.Item
    :param asset_key: if provided as a string, will be the key of the source asset in item.Assets.
    if provided as a dictionary, must contain an entry for the current item with item.id as key and
    asset key as value.
    :type asset_key: str | Mapping[str, str]
    :raises KeyError: if asset_key is provided as a dictionary but does not contain item.id
    :raises TypeError: if asset_key is neither a string or a dictionary
    :return: the source asset href
    :rtype: str
    """
    if isinstance(asset_key, str):
        return item.assets[asset_key].href
    elif isinstance(asset_key, dict):
        if item.id not in asset_key:
            raise KeyError(f"Asset key map does not have entry for item: {item.id}")
        return item.assets[asset_key[item.id]].href
    raise TypeError(
        f"Invalid type for asset key: {type(asset_key)}. Accepts either a string or a mapping"
    )


def get_item_crs(item: pystac.Item) -> CRS:
    """Extract CRS information from item properties.

    This will first look for CRS information encoded as proj extension, in the following order:
    `proj:code, proj:wkt2, proj:projjson, proj:epsg`

    If no proj extension fields is found, will attempt to look for the field `epsg` in properties.

    :param item: stac item metadata
    :type item: pystac.Item
    :raises StacExtensionError: if proj:projjson is provided but with an invalid format
    :raises StacExtensionError: if there is no crs field in the metadata
    :return: CRS information
    :rtype: CRS
    """
    if "proj:code" in item.properties:
        return CRS(item.properties.get("proj:code"))
    elif "proj:wkt2" in item.properties:
        return CRS(item.properties.get("proj:wkt2"))
    elif "proj:projjson" in item.properties:
        try:
            return CRS(json.loads(item.properties.get("proj:projjson")))  # type: ignore[arg-type]
        except json.JSONDecodeError as e:
            raise StacExtensionError("Invalid projjson encoding in STAC config") from e
    elif "proj:epsg" in item.properties:
        warn(
            "proj:epsg is deprecated in favor of proj:code. Please consider using proj:code, or if possible, the full wkt2 instead"
        )
        return CRS(int(item.properties.get("proj:epsg")))  # type: ignore[arg-type]
    elif "epsg" in item.properties:
        return CRS(int(item.properties.get("epsg")))  # type: ignore[arg-type]
    else:
        raise StacExtensionError("Missing CRS information in item properties")


def convert_bbox_to_target_crs(bbox: BBox_T, src: CRS, target: CRS) -> BBox_T:
    """Convert bbox from one crs to another

    :param bbox: bounding box
    :type bbox: tuple[float, float, float, float]
    :param src: crs of the bounding box
    :type src: CRS
    :param target: target crs
    :type target: CRS
    :return: the bounding box in target CRS
    :rtype: tuple[float, float, float, float]
    """
    if src == target:
        return bbox
    transformer = Transformer.from_crs(src, target, always_xy=True)
    left, bottom = transformer.transform(bbox[0], bbox[1])
    right, top = transformer.transform(bbox[2], bbox[3])
    return left, bottom, right, top


def item_in_geobox(item: pystac.Item, geobox: GeoBox) -> bool:
    """Check if item is contained in geobox

    :param item: pystac Item. Must have bbox field and crs information in properties either as proj:epsg or epsg
    :type item: pystac.Item
    :param geobox: geobox containing bounding box and crs information
    :type geobox: GeoBox
    :return: whether some areas of item overlap with the bbox of geobox
    :rtype: bool
    """
    item_bbox = convert_bbox_to_target_crs(
        cast(BBox_T, item.bbox), get_item_crs(item), cast(CRS, geobox.crs)
    )
    gbbox = geobox.boundingbox
    if (
        item_bbox[0] > gbbox[2]
        or gbbox[0] > item_bbox[2]
        or item_bbox[1] > gbbox[3]
        or gbbox[1] > item_bbox[3]
    ):
        return False
    return True


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


def query_timebox(
    frame: gpd.GeoDataFrame, timebox: TimeBox, t_col: str = "T"
) -> gpd.GeoDataFrame:
    """Constrain frame to time region specified by `timebox` and converts timezone to
    that of `timebox`.

    This method works on both point GeoDataFrame and vector GeoDataFrame

    :param frame: input dataframe. Dataframe must contain `t_col` of datetime type with valid timezone info.
    :type frame: gpd.GeoDataFrame
    :param timebox: time extent defines start date, end date, frequency and timezone information
    :type timebox: TimeBox
    :param t_col: name of time column in frame, defaults to "T"
    :type t_col: str, optional
    :raises KeyError: if `t_col` is not in frame
    :raises TypeError: if dtype of `t_col` is not any datetime type
    :raises ValueError: if `t_col` is of datatime type but no timezone info is present
    :return: frame confined to time extent defined by timebox
    :rtype: gpd.GeoDataFrame
    """
    if t_col not in frame.columns:
        raise KeyError(f"No time column {t_col} in dataframe")
    if not is_datetime64_any_dtype(frame[t_col]):
        raise TypeError(f"Column {t_col} is not of datetime type")
    if frame[t_col].dt.tz is None:
        raise ValueError(f"Date column {t_col} does not have timezone info.")

    # Convert to timezone specified in timebox
    # Note `tz_convert` converts one tz to another
    # `tz_localize` embeds tz information to a date column w/o tz
    frame[t_col] = frame[t_col].dt.tz_convert(timebox.tz)
    return frame[(frame[t_col] >= timebox.start) and (frame[t_col] <= timebox.end)]


# Point data Utilities


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
        X_name, Y_name, T_name = (
            item.properties["X"],
            item.properties["Y"],
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
    geobox: GeoBox,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
    merge_method: MergeMethods = "mean",
    tol: float = BBOX_TOL,
) -> gpd.GeoDataFrame:
    # Preprocess frames - query geobox
    frame = query_geobox(frame, geobox, tol)
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
    ds = frame.to_xarray()
    # Sometime to_xarray bugs out with datetime index so need explicit conversion
    ds[t_col] = pd.DatetimeIndex(ds.coords[t_col].values)
    if interp_method is None:
        return ds
    coords_ = xr_coords(geobox, dims=(y_col, x_col))
    ds = ds.assign_coords(spatial_ref=coords_["spatial_ref"])
    coords = {y_col: coords_[y_col], x_col: coords_[x_col]}
    return ds.interp(
        coords=coords,
        method=interp_method,
    )


# Vector data Utilities


def update_attr_legend(
    attr_dict: dict[str, Any], layer_name: str, field: str, frame: gpd.GeoDataFrame
) -> None:
    if pd.api.types.is_string_dtype(frame[field]):
        cat_map = {name: index for index, name in enumerate(frame[field].unique())}
        attr_dict[layer_name] = cat_map
        frame[field] = frame[field].map(cat_map)


def groupby_id(
    data: dict[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    fields: Sequence[str] | dict[str, Sequence[str]] | None = None,
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Load as pure mask
    if fields is None:
        return {
            key: (
                [y_col, x_col],
                rasterize(
                    value.geometry,
                    geobox.shape,
                    transform=geobox.transform,
                    masked=True,
                ),
            )
            for key, value in data.items()
        }, {}
    # Load as attribute per layer
    # Prepare field for each layer
    item_fields = {}
    if isinstance(fields, dict):
        if set(data.keys()).issubset(set(fields.keys())):
            raise ValueError(
                f"Vector Loader: when groupby id and field is provided as a dictionary, its key must be a superset of ids of all vector items in the collection. {set(data.keys()) - set(fields.keys())}"
            )
        item_fields = fields
    else:
        item_fields = {k: fields for k in data.keys()}

    ds_data = {}
    ds_attrs = {"legend": {}}
    # Field per layer
    for k, frame in data.items():
        for field in item_fields[k]:
            layer_name = f"{k}_{field}"
            update_attr_legend(ds_attrs["legend"], layer_name, field, frame)
            # Build legend mapping for categorical encoding of values
            ds_data[layer_name] = (
                [y_col, x_col],
                rasterize(
                    (
                        (geom, value)
                        for geom, value in zip(frame.geometry, frame[field])
                    ),
                    geobox.shape,
                    transform=geobox.transform,
                ),
            )
    return ds_data, ds_attrs


def groupby_field(
    data: dict[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    fields: Sequence[str],
    alias_renaming: dict[str, tuple[str, str]] | None = None,
    x_col: str = "x",
    y_col: str = "y",
) -> xr.Dataset:
    if fields is None:
        raise ValueError("When groupby field, fields parameter must not be None")
    # Rename columns based on alias map
    if alias_renaming:
        for field, (item_id, item_column) in alias_renaming:
            data[item_id].rename(columns={item_column: field}, inplace=True)
    if isinstance(fields, str):
        fields = [fields]
    gdf = pd.concat(data.values())
    ds_data = {}
    ds_attrs = {"legend": {}}
    for field in fields:
        update_attr_legend(ds_attrs["legend"], field, field, gdf)
        ds_data[field] = (
            [y_col, x_col],
            rasterize(
                ((geom, value) for geom, value in zip(gdf.geometry, gdf[field])),
                out_shape=geobox.shape,
                transform=geobox.transform,
            ),
        )
    return ds_data, ds_attrs
