from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import pandas as pd
import pystac
import pystac_client
import xarray as xr

from mccn.config import (
    CubeConfig,
    FilterConfig,
    ProcessConfig,
)
from mccn.extent import GeoBoxBuilder
from mccn.loader.point import PointLoadConfig, PointLoader
from mccn.loader.raster import RasterLoadConfig, RasterLoader
from mccn.loader.vector import VectorLoadConfig, VectorLoader
from mccn.parser import Parser

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox

    from mccn._types import (
        CRS_T,
        AnchorPos_T,
        BBox_T,
        Number_T,
        Resolution_T,
        Shape_T,
        TimeGroupby,
    )


class EndpointException(Exception): ...


class EndpointType(Exception): ...


class MCCN:
    def __init__(
        self,
        # Item discovery
        endpoint: str | Path | tuple[str, str] | None = None,
        collection: pystac.Collection | None = None,
        items: Sequence[pystac.Item] | None = None,
        # Geobox config
        shape: Shape_T | None = None,
        resolution: Resolution_T | None = None,
        bbox: BBox_T | None = None,
        anchor: AnchorPos_T = "default",
        crs: CRS_T = 4326,
        # Filter config
        geobox: GeoBox | None = None,
        start_ts: str | pd.Timestamp | datetime.datetime | None = None,
        end_ts: str | pd.Timestamp | datetime.datetime | None = None,
        bands: set[str] | None = None,
        mask_only: bool = False,
        use_all_vectors: bool = True,
        # Cube config
        x_coord: str = "x",
        y_coord: str = "y",
        t_coord: str = "time",
        mask_name: str = "__MASK__",
        # Process config
        rename_bands: Mapping[str, str] | None = None,
        process_bands: Mapping[str, Callable] | None = None,
        nodata: Number_T | Mapping[str, Number_T] = 0,
        nodata_fallback: Number_T = 0,
        categorical_encoding_start: int = 1,
        time_groupby: TimeGroupby = "time",
        # Additional configs
        point_load_config: PointLoadConfig | None = None,
        vector_load_config: VectorLoadConfig | None = None,
        raster_load_config: RasterLoadConfig | None = None,
    ) -> None:
        # Fetch Collection
        self.items = self.get_items(items, collection, endpoint)
        # Make geobox
        self.geobox = self.build_geobox(
            self.items, shape, resolution, bbox, anchor, crs, geobox
        )
        # Prepare configs
        self.filter_config = FilterConfig(
            geobox=self.geobox,
            start_ts=start_ts,
            end_ts=end_ts,
            bands=bands,
            mask_only=mask_only,
            use_all_vectors=use_all_vectors,
        )
        self.cube_config = CubeConfig(
            x_coord=x_coord,
            y_coord=y_coord,
            t_coord=t_coord,
            mask_name=mask_name,
        )
        self.process_config = ProcessConfig(
            rename_bands,
            process_bands,
            nodata,
            nodata_fallback,
            categorical_encoding_start,
            time_groupby,
        )
        self.point_load_config = point_load_config
        self.vector_load_config = vector_load_config
        self.raster_load_config = raster_load_config

        # Parse items
        self.parser = Parser(self.filter_config, self.items)
        self.parser()
        self.point_loader: PointLoader = PointLoader(
            self.parser.point,
            self.filter_config,
            self.cube_config,
            self.process_config,
            self.point_load_config,
        )
        self.vector_loader = VectorLoader(
            self.parser.vector,
            self.filter_config,
            self.cube_config,
            self.process_config,
            self.vector_load_config,
        )
        self.raster_loader: RasterLoader = RasterLoader(
            self.parser.raster,
            self.filter_config,
            self.cube_config,
            self.process_config,
            self.raster_load_config,
        )

    @staticmethod
    def build_geobox(
        items: list[pystac.Item],
        shape: Shape_T | None = None,
        resolution: Resolution_T | None = None,
        bbox: BBox_T | None = None,
        anchor: AnchorPos_T = "default",
        crs: CRS_T = 4326,
        # Filter config
        geobox: GeoBox | None = None,
    ) -> GeoBox:
        if geobox:
            return geobox
        try:
            builder = GeoBoxBuilder(crs, anchor=anchor)
            if bbox:
                builder = builder.set_bbox(bbox)
            if resolution is not None:
                if not isinstance(resolution, tuple):
                    resolution = (resolution, resolution)
                builder = builder.set_resolution(*resolution)
            if shape:
                if not isinstance(shape, tuple):
                    shape = (shape, shape)
                builder = builder.set_shape(*shape)
            return builder.build()
        except Exception:
            if not shape:
                raise ValueError(
                    "Unable to build geobox. For simplicity, user can pass a shape parameter, which will be used to build a geobox from collection."
                )
            return GeoBoxBuilder.from_items(items, shape, anchor)

    @staticmethod
    def get_geobox(
        collection: pystac.Collection,
        geobox: GeoBox | None = None,
        shape: int | tuple[int, int] | None = None,
    ) -> GeoBox:
        if geobox is not None:
            return geobox
        if shape is None:
            raise ValueError(
                "If geobox is not defined, shape must be provided to calculate geobox from collection"
            )
        return GeoBoxBuilder.from_collection(collection, shape)

    def load_point(self) -> xr.Dataset:
        return self.point_loader.load()

    def load_vector(self) -> xr.Dataset:
        return self.vector_loader.load()

    def load_raster(self) -> xr.Dataset:
        return self.raster_loader.load()

    def load(self) -> xr.Dataset:
        return xr.merge(
            [self.load_point(), self.load_vector(), self.load_raster()],
            combine_attrs="drop_conflicts",
        )

    @staticmethod
    def get_items(
        items: Sequence[pystac.Item] | None = None,
        collection: pystac.Collection | None = None,
        endpoint: str | tuple[str, str] | Path | None = None,
    ) -> list[pystac.Item]:
        if items:
            return list(items)
        collection = MCCN.get_collection(endpoint, collection)
        return list(collection.get_items(recursive=True))

    @staticmethod
    def get_collection(
        endpoint: str | tuple[str, str] | Path | None,
        collection: pystac.Collection | None = None,
    ) -> pystac.Collection:
        """Try to load collection from endpoint.

        Raises `EndpointType` if endpoint is not an acceptable type, or `EndpointException` if
        endpoint is not reachable
        """
        if collection:
            return collection
        if not endpoint:
            raise ValueError("Either a collection or an endpoint must be provided")
        try:
            if isinstance(endpoint, tuple):
                href, collection_id = endpoint
                return pystac_client.Client.open(href).get_collection(collection_id)
            if isinstance(endpoint, Path | str):
                return pystac.Collection.from_file(str(endpoint))
            raise EndpointType(
                f"Expects endpoint as a local file path or a (api_endpoint, collection_id) tuple. Receives: {endpoint}"
            )
        except EndpointType as e:
            raise e
        except Exception as exception:
            raise EndpointException from exception
