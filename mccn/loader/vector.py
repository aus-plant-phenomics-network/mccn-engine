from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Hashable, Literal, Mapping, Sequence, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.enums import MergeAlg
from rasterio.features import rasterize

from mccn._types import CubeConfig, FilterConfig, ParsedVector, ProcessConfig
from mccn.loader.base import Loader
from mccn.loader.utils import (
    bbox_from_geobox,
)

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox


@dataclass
class VectorLoadConfig:
    """Load config for vector type

    Aside from groupby, all config parameters come from rasterio.rasterize function.
    """

    fill: int = 0
    """rasterio.rasterize fill parameter"""
    all_touched: bool = False
    """rasterio.rasterize all_touched"""
    nodata: Any | None = None
    """rasterio.rasterize nodata"""
    merge_alg: Literal["REPLACE", "ADD"] | MergeAlg = MergeAlg.replace
    """rasterio.rasterize nodata"""
    dtype: Any | None = None
    """rasterio.rasterize dtype"""
    groupby: Literal["id", "field"] = "id"
    """Whether to load masks as separate layers (grouped by id) or in a combined MASKS layer (grouped by field). Attributes are loaded by field by default"""

    def __post_init__(self) -> None:
        # Instantiate merge_alg enum from string
        if isinstance(self.merge_alg, str):
            self.merge_alg = MergeAlg[self.merge_alg]


def update_attr_legend(
    attr_dict: dict[str, Any], field: str, frame: gpd.GeoDataFrame
) -> None:
    """Update attribute dict with legend for non numeric fields.

    If the field is non-numeric - i.e. string, values will be categoricalised
    i.e. 1, 2, 3, ...
    The mapping will be updated in attr_dict under field name

    Args:
        attr_dict (dict[str, Any]): attribute dict
        field (str): field name
        frame (gpd.GeoDataFrame): input data frame
    """
    if not pd.api.types.is_numeric_dtype(frame[field]):
        cat_map = {
            name: index for index, name in enumerate(frame[field].unique(), start=1)
        }
        attr_dict[field] = {v: k for k, v in cat_map.items()}
        frame[field] = frame[field].map(cat_map)


def _groupby_mask_field(
    data: Mapping[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    coords: Mapping[Hashable, xr.DataArray],
    fill: int = 0,
    all_touched: bool = False,
    nodata: Any | None = None,
    merge_alg: MergeAlg = MergeAlg.replace,
    dtype: Any | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
) -> xr.Dataset:
    attrs_dict = {}
    for idx, (k, v) in enumerate(data.items(), start=1):
        v["KEY"] = idx
        attrs_dict[idx] = k
    gdf = pd.concat(data.values())
    dates = pd.Series(sorted(gdf[t_col].unique()))
    raster = []
    for date in dates:
        raster.append(
            rasterize(
                (
                    (geom, value)
                    for geom, value in zip(
                        gdf[gdf[t_col] == date].geometry,
                        gdf[gdf[t_col] == date]["KEY"],
                    )
                ),
                out_shape=geobox.shape,
                transform=geobox.transform,
                fill=fill,
                all_touched=all_touched,
                nodata=nodata,
                merge_alg=merge_alg,
                dtype=dtype,
            ),
        )
    ds_data = np.stack(raster, axis=0)
    ds = xr.Dataset(
        {"MASKS": ([t_col, y_col, x_col], ds_data)},
        coords=coords,
        attrs={"MASKS": attrs_dict},
    )
    ds[t_col] = dates.values
    return ds


def _groupby_mask_id(
    data: Mapping[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    coords: Mapping[Hashable, xr.DataArray],
    fill: int = 0,
    all_touched: bool = False,
    nodata: Any | None = None,
    merge_alg: MergeAlg = MergeAlg.replace,
    dtype: Any | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
) -> xr.Dataset:
    dss = []
    for key, value in data.items():
        dates = pd.Series(sorted(value[t_col].unique()))
        raster = []
        for date in dates:
            raster.append(
                rasterize(
                    value[value[t_col] == date].geometry,
                    out_shape=(geobox.shape),
                    transform=geobox.transform,
                    fill=fill,
                    all_touched=all_touched,
                    nodata=nodata,
                    merge_alg=merge_alg,
                    dtype=dtype,
                )
            )
        ds_data = np.stack(raster, axis=0)
        ds = xr.Dataset({key: ([t_col, y_col, x_col], ds_data)}, coords=coords)
        ds[t_col] = dates.values
        dss.append(ds)
    return xr.merge(dss)


def groupby_mask(
    data: Mapping[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    coords: Mapping[Hashable, xr.DataArray],
    fill: int = 0,
    all_touched: bool = False,
    nodata: Any | None = None,
    merge_alg: MergeAlg = MergeAlg.replace,
    dtype: Any | None = None,
    groupby_mode: Literal["id", "field"] = "field",
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
) -> xr.Dataset:
    """Generate mask layer(s) for vector items that do not have column info described.

    Masks are loaded in two modes - groupby field and groupby id. If masks are grouped by
    field, all masks are loaded to a single MASKS layer with dimension (time, y, x).
    If masks are grouped by id, each item is loaded as an independent mask with layer name being
    the item's id.

    Args:
        data (Mapping[str, gpd.GeoDataFrame]): Input data - dictionary with key being item id and value being gdf with geometry
        geobox (GeoBox): target geobox
        coords (Mapping[Hashable, xr.DataArray]): dataset coordinates
        fill (int, optional): fill value from rasterio.rasterize. Defaults to 0.
        all_touched (bool, optional): all_touched value from rasterio.rasterize. Defaults to False.
        nodata (Any | None, optional): nodata value from rasterio.rasterize. Defaults to None.
        merge_alg (MergeAlg, optional): merg_alg value from rasterio.rasterize. Defaults to MergeAlg.replace.
        dtype (Any | None, optional): dtype value from rasterio.rasterize. Defaults to None.
        groupby_mode (Literal[&quot;id&quot;, &quot;field&quot;], optional): group by modes. Defaults to "field".
        x_col (str, optional): name of x dimension. Defaults to "x".
        y_col (str, optional): name of y dimension. Defaults to "y".
        t_col (str, optional): name of t dimension. Defaults to "time".

    Raises:
        ValueError: invalid groupby_mode value

    Returns:
        xr.Dataset: mask dataset
    """
    if not data:
        return xr.Dataset()
    if groupby_mode == "field":
        return _groupby_mask_field(
            data,
            geobox,
            coords,
            fill,
            all_touched,
            nodata,
            merge_alg,
            dtype,
            x_col,
            y_col,
            t_col,
        )
    if groupby_mode == "id":
        return _groupby_mask_id(
            data,
            geobox,
            coords,
            fill,
            all_touched,
            nodata,
            merge_alg,
            dtype,
            x_col,
            y_col,
            t_col,
        )
    raise ValueError(
        f"Expects groupby mask to be either 'id' or 'field'. Receive: {groupby_mode}"
    )


def groupby_field(
    data: Mapping[str, gpd.GeoDataFrame],
    geobox: GeoBox,
    fields: set[str],
    coords: Mapping[Hashable, xr.DataArray],
    fill: int = 0,
    all_touched: bool = False,
    nodata: Any | None = None,
    merge_alg: MergeAlg = MergeAlg.replace,
    dtype: Any | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
) -> xr.Dataset:
    """Generate a datacube from attributes of a combined dataframe.

    The target dataset has variables of dimension (time, y, x). Each variable
    is extracted from a field column in the dataframe

    Args:
        data (Mapping[str, gpd.GeoDataFrame]): Input data - dictionary with key being item id and value being dataframe
        geobox (GeoBox): target geobox
        fields (set[str]): datacube variables - columns from the combined dataframe
        coords (Mapping[Hashable, xr.DataArray]): _description_
        fill (int, optional): fill value from rasterio.rasterize. Defaults to 0.
        all_touched (bool, optional): all_touched value from rasterio.rasterize. Defaults to False.
        nodata (Any | None, optional): nodata value from rasterio.rasterize. Defaults to None.
        merge_alg (MergeAlg, optional): merg_alg value from rasterio.rasterize. Defaults to MergeAlg.replace.
        dtype (Any | None, optional): dtype value from rasterio.rasterize. Defaults to None.
        x_col (str, optional): name of x dimension. Defaults to "x".
        y_col (str, optional): name of y dimension. Defaults to "y".
        t_col (str, optional): name of t dimension. Defaults to "time".

    Returns:
        xr.Dataset: attribute dataset
    """
    if not data:
        return xr.Dataset()
    gdf = pd.concat(data.values())
    ds_data = {}
    ds_attrs: dict[str, Any] = {}
    dates = pd.Series(sorted(gdf[t_col].unique()))  # Date attributes
    for field in fields:
        update_attr_legend(ds_attrs, field, gdf)
        raster = []
        for date in dates:
            raster.append(
                rasterize(
                    (
                        (geom, value)
                        for geom, value in zip(
                            gdf[gdf[t_col] == date].geometry,
                            gdf[gdf[t_col] == date][field],
                        )
                    ),
                    out_shape=geobox.shape,
                    transform=geobox.transform,
                    fill=fill,
                    all_touched=all_touched,
                    nodata=nodata,
                    merge_alg=merge_alg,
                    dtype=dtype,
                ),
            )

        ds_data[field] = ([t_col, y_col, x_col], np.stack(raster, axis=0))
    ds = xr.Dataset(ds_data, attrs=ds_attrs, coords=coords)
    ds[t_col] = dates.values
    return ds


def read_vector_asset(
    item: ParsedVector, geobox: GeoBox, t_coord: str = "time"
) -> gpd.GeoDataFrame:
    """Load a single vector item

    Load vector asset. If a join asset is provided, will load the
    join asset and perform a join operation on common column (Inner Join)

    Args:
        item (ParsedVector): parsed vector item
        geobox (GeoBox): target geobox
        t_coord (str): name of the time dimension if valid

    Returns:
        gpd.GeoDataFrame: vector geodataframe
    """
    date_col = None
    # Prepare geobox for filtering
    bbox = bbox_from_geobox(geobox, item.crs)
    # Load main item
    gdf = gpd.read_file(
        item.location,
        bbox=bbox,
        columns=list(item.load_bands),
        layer=item.config.layer,
    )
    # Load aux df
    if item.load_aux_bands:
        if item.config.join_T_column and item.config.join_file:
            date_col = item.config.join_T_column
            aux_df = pd.read_csv(
                item.config.join_file,
                usecols=list(item.load_aux_bands),
                parse_dates=[item.config.join_T_column],
                date_format=cast(str, item.config.date_format),
            )
        else:
            aux_df = pd.read_csv(
                cast(str, item.config.join_file),
                usecols=list(item.load_aux_bands),
            )
        # Join dfs
        gdf = pd.merge(
            gdf,
            aux_df,
            left_on=item.config.join_attribute_vector,
            right_on=item.config.join_field,
        )
    # Convert CRS
    gdf.to_crs(geobox.crs, inplace=True)
    # Process date
    if not date_col:
        gdf[t_coord] = item.item.datetime
    else:
        gdf.rename(columns={date_col: t_coord}, inplace=True)
    return gdf


class VectorLoader(Loader[ParsedVector]):
    """
    Vector STAC loader

    Similar to other item loaders, each band is loaded with dimension (time, y, x)
    Time is derived from the asset (mainly the external asset that joins with the main vector file) if valid (join_T_column is present),
    or from item's datetime field otherwise.

    Vectors can be loaded as masks (if no column_info is described in STAC) or as attribute/band layer. If an external asset (join_file) is
    described in STAC, an inner join operation will join the vector file's join_vector_attribute with the external asset's join_field to produce
    a join frame whose attributes will be loaded as band/variable in the datacube.

    Masks can be loaded in two modes - groupby field and groupby id. If masks are grouped by
    field, all masks are loaded to a single MASKS layer with dimension (time, y, x).
    If masks are grouped by id, each item is loaded as an independent mask with layer name being
    the item's id. This parameter can be updated using load_config.

    Users can control the dimension of the cube by updating cube_config parameter, control the renaming and preprocessing of fields by updating
    process_config, and control the rasterize operation using load_config.

    """

    def __init__(
        self,
        items: Sequence[ParsedVector],
        filter_config: FilterConfig,
        cube_config: CubeConfig | None = None,
        process_config: ProcessConfig | None = None,
        load_config: VectorLoadConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.load_config = load_config if load_config else VectorLoadConfig()
        super().__init__(items, filter_config, cube_config, process_config, **kwargs)

    def _load(self) -> xr.Dataset:
        data = {}  # Mapping of item id to geodataframe
        bands = set()  # All bands available in vector collection
        mask_only = set()  # Set of items to be loaded as mask only

        # Prepare items
        for item in self.items:
            if not item.load_aux_bands and not item.load_bands:
                mask_only.add(item.item.id)
            bands.update(item.load_bands)
            bands.update(item.load_aux_bands)
            # Remove date column - not a variable
            if item.config.join_T_column:
                bands.remove(item.config.join_T_column)
            data[item.item.id] = self.apply_process(
                read_vector_asset(
                    item, self.filter_config.geobox, self.cube_config.t_coord
                ),
                self.process_config,
            )

        # Load attribute cube
        attr_data = groupby_field(
            data={k: v for k, v in data.items() if k not in mask_only},
            geobox=self.filter_config.geobox,
            fields=bands,
            x_col=self.cube_config.x_coord,
            y_col=self.cube_config.y_coord,
            t_col=self.cube_config.t_coord,
            coords=self.coords,
            fill=self.load_config.fill,
            all_touched=self.load_config.all_touched,
            nodata=self.load_config.nodata,
            merge_alg=self.load_config.merge_alg,
            dtype=self.load_config.dtype,
        )
        # Load mask cube
        mask_data = groupby_mask(
            data={k: v for k, v in data.items() if k in mask_only},
            geobox=self.filter_config.geobox,
            x_col=self.cube_config.x_coord,
            y_col=self.cube_config.y_coord,
            t_col=self.cube_config.t_coord,
            coords=self.coords,
            fill=self.load_config.fill,
            all_touched=self.load_config.all_touched,
            nodata=self.load_config.nodata,
            merge_alg=self.load_config.merge_alg,
            dtype=self.load_config.dtype,
            groupby_mode=self.load_config.groupby,
        )

        # Combine attribute + mask
        return xr.merge([attr_data, mask_data], combine_attrs="no_conflicts")
