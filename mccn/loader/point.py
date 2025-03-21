from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.xr import xr_coords
from stac_generator.core.point.generator import read_csv

from mccn.loader.utils import (
    ASSET_KEY,
    StacExtensionError,
    get_item_crs,
    get_item_href,
    get_required_columns,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pystac
    from odc.geo.geobox import GeoBox

    from mccn._types import InterpMethods, MergeMethods


def read_point_asset(
    item: pystac.Item,
    bands: Sequence[str] | None = None,
    asset_key: str | Mapping[str, str] = ASSET_KEY,
    t_col: str = "time",
    z_col: str = "z",
    band_preprocessing: Mapping[str, Callable[[Any], Any]] | None = None,
    band_renaming: Mapping[str, str] | None = None,
) -> gpd.GeoDataFrame | None:
    try:
        # Process metadata
        location = get_item_href(item, asset_key)
        columns = get_required_columns(item, bands, band_renaming)
        # Columns can be None from either
        # "column_info" not described in item's properties
        # no column in "column_info" is in band - this asset do not contained the desired bands
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
        rename_dict = {}
        if T_name is None:
            T_name = t_col
            gdf[T_name] = item.datetime
        else:
            rename_dict[T_name] = t_col
        if Z_name:
            rename_dict[Z_name] = z_col

        # Transform, Rename
        if band_preprocessing:
            for key, fn in band_preprocessing.items():
                if key in gdf.columns:
                    gdf[key] = gdf[key].apply(fn)
        if band_renaming:
            gdf.rename(columns=band_renaming, inplace=True)

        # Rename indices
        gdf.rename(columns=rename_dict, inplace=True)
        # Drop X and Y columns since we will repopulate them after changing crs
        gdf.drop(columns=[X_name, Y_name], inplace=True)
    except KeyError as e:
        raise StacExtensionError("Missing band in stac config:") from e
    return gdf


def process_groupby(
    frame: gpd.GeoDataFrame,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
    z_col: str = "z",
    use_z: bool = False,
    merge_method: MergeMethods = "mean",
) -> gpd.GeoDataFrame:
    frame[x_col] = frame.geometry.x
    frame[y_col] = frame.geometry.y

    # Prepare aggregation method
    excluding_bands = set([x_col, y_col, t_col, "geometry"])
    if use_z and z_col:
        excluding_bands.add(z_col)
    bands = [name for name in frame.columns if name not in excluding_bands]
    # band_map determines replacement strategy for each band when there is a conflict
    band_map = (
        {band: merge_method[band] for band in bands if band in merge_method}
        if isinstance(merge_method, dict)
        else {band: merge_method for band in bands}
    )

    # Groupby + Aggregate
    if use_z and z_col:
        return frame.groupby([t_col, y_col, x_col, z_col]).agg(band_map)
    return frame.groupby([t_col, y_col, x_col]).agg(band_map)


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
    bands: Sequence[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
    z_col: str = "z",
    use_z: bool = False,
    merge_method: MergeMethods = "mean",
    interp_method: InterpMethods | None = "nearest",
    band_preprocessing: Mapping[str, Callable[[Any], Any]] | None = None,
    band_renaming: Mapping[str, str] | None = None,
) -> xr.Dataset:
    frames = []
    for item in items:
        frame = read_point_asset(
            item=item,
            bands=bands,
            asset_key=asset_key,
            t_col=t_col,
            z_col=z_col,
            band_preprocessing=band_preprocessing,
            band_renaming=band_renaming,
        )
        if frame is not None:  # Can be None if does not contain required column
            frame = frame.to_crs(geobox.crs)
            # Process groupby - i.e. average out over depth, duplicate entries, etc
            merged = process_groupby(
                frame, x_col, y_col, t_col, z_col, use_z, merge_method
            )
            frames.append(
                point_data_to_xarray(merged, geobox, x_col, y_col, t_col, interp_method)
            )
    return xr.merge(frames)
