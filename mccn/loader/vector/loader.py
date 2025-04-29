from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
from stac_generator.core.base.utils import (
    calculate_timezone,
    read_join_asset,
    read_vector_asset,
)

from mccn.loader.base import Loader
from mccn.loader.utils import (
    bbox_from_geobox,
)
from mccn.parser import ParsedVector

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox


def read_asset(
    id: str,
    item: ParsedVector,
    geobox: GeoBox,
    t_dim: str,
    period: str | None,
    mask_name: str = "",
) -> gpd.GeoDataFrame:
    """Load a single vector item

    Load vector asset. If a join asset is provided, will load the
    join asset and perform a join operation on common column (Inner Join)
    Convert all datetime to UTC but strip off timezone information. Convert
    date column to period specified in process config

    Args:
        item (ParsedVector): parsed vector item
        geobox (GeoBox): target geobox
        t_dim (str): name of the time dimension if valid
        period (str): time groupby period

    Returns:
        gpd.GeoDataFrame: vector geodataframe
    """
    # Prepare geobox for filtering
    bbox = bbox_from_geobox(geobox, item.crs)
    # Load main item
    gdf = read_vector_asset(
        src_path=item.location,
        bbox=bbox,
        columns=list(item.load_bands),
        layer=item.config.layer,
    )
    # Load aux df
    if item.load_aux_bands and item.config.join_config:
        join_config = item.config.join_config
        tzinfo = calculate_timezone(gdf.to_crs(4326).geometry)
        aux_df = read_join_asset(
            join_config.file,
            join_config.right_on,
            join_config.date_format,
            join_config.date_column,
            item.load_aux_bands,
            tzinfo,
        )
        # Join dfs
        gdf = pd.merge(
            gdf,
            aux_df,
            left_on=item.config.join_config.left_on,
            right_on=item.config.join_config.right_on,
        )
    # Convert CRS
    gdf.to_crs(geobox.crs, inplace=True)
    # Set mask column
    gdf[mask_name] = id
    # Set date value based on aux df if available or item.datetime property
    date_col = item.config.join_config.date_column if item.config.join_config else None
    if date_col and date_col in item.load_aux_bands:
        gdf.rename(columns={date_col: t_dim}, inplace=True)
    else:
        gdf[t_dim] = item.item.datetime
    # Convert to UTC and remove timezone info
    gdf[t_dim] = gdf[t_dim].dt.tz_convert("utc").dt.tz_localize(None)
    # Convert date periods for timegroupby option
    if period is not None:
        gdf[t_dim] = gdf[t_dim].dt.to_period(period).dt.start_time
    return gdf


class VectorLoader(Loader[ParsedVector]):
    def load(self) -> None:
        # Prepare items
        for item in self.items:
            id = item.item.id
            mask_name = (
                self.cube_config.mask_name if self.cube_config.combine_mask else id
            )
            bands = {mask_name} | item.load_bands | item.load_aux_bands
            if (
                item.config.join_config
                and item.config.join_config.date_column
                and item.config.join_config.date_column in bands
            ):
                bands.remove(item.config.join_config.date_column)
            gdf = self.apply_process(
                read_asset(
                    id,
                    item,
                    self.filter_config.geobox,
                    self.cube_config.t_dim,
                    self.process_config.period,
                    mask_name,
                ),
                bands,
            )

            self.rasteriser.rasterise_vector(
                data=gdf,
                bands=bands,
                geobox=self.filter_config.geobox,
            )
