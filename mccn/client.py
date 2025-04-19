from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

import pystac
import pystac_client
import xarray as xr

from mccn._types import CRS_T, AnchorPos_T, BBox_T, Resolution_T, Shape_T
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


class EndpointException(Exception): ...


class EndpointType(Exception): ...


class MCCN:
    def __init__(
        self,
        endpoint: str | Path | tuple[str, str],
        # Geobox config
        shape: Shape_T | None = None,
        resolution: Resolution_T | None = None,
        bbox: BBox_T | None = None,
        anchor: AnchorPos_T = "default",
        crs: CRS_T = 4326,
        # Filter config
        geobox: GeoBox | None = None,
        start_ts: datetime.datetime | None = None,
        end_ts: datetime.datetime | None = None,
        bands: set[str] | None = None,
        # Cube config
        x_coord: str = "x",
        y_coord: str = "y",
        t_coord: str = "time",
        # Process config
        rename_bands: Mapping[str, str] | None = None,
        process_bands: Mapping[str, Callable] | None = None,
        # Additional configs
        point_load_config: PointLoadConfig | None = None,
        vector_load_config: VectorLoadConfig | None = None,
        raster_load_config: RasterLoadConfig | None = None,
    ) -> None:
        # Fetch Collection
        self.endpoint = endpoint
        self.collection = self.get_collection(endpoint)
        # Make geobox
        self.geobox = self.build_geobox(
            self.collection, shape, resolution, bbox, anchor, crs, geobox
        )
        # Prepare configs
        self.filter_config = FilterConfig(self.geobox, start_ts, end_ts, bands)
        self.cube_config = CubeConfig(x_coord, y_coord, t_coord)
        self.process_config = ProcessConfig(rename_bands, process_bands)
        self.point_load_config = point_load_config
        self.vector_load_config = vector_load_config
        self.raster_load_config = raster_load_config

        # Parse items
        self.parser = Parser(self.filter_config, self.collection)
        self.parser()
        self._point_loader: PointLoader | None = None
        self._vector_loader: VectorLoader | None = None
        self._raster_loader: RasterLoader | None = None

    @staticmethod
    def build_geobox(
        collection: pystac.Collection,
        shape: Shape_T | None = None,
        resolution: Resolution_T | None = None,
        bbox: BBox_T | None = None,
        anchor: AnchorPos_T = "default",
        crs: CRS_T = 4326,
        # Filter config
        geobox: GeoBox | None = None,
    ) -> GeoBox:
        try:
            if resolution and not isinstance(resolution, tuple):
                resolution = (resolution, resolution)
            if shape and not isinstance(shape, tuple):
                shape = (shape, shape)
            builder = (
                GeoBoxBuilder(crs, anchor)
                .set_bbox(bbox)
                .set_resolution(*resolution)
                .set_shape(*shape)
                .set_geobox(geobox)
            )
            return builder.build()
        except Exception:
            if not shape:
                raise ValueError(
                    "Unable to build geobox. For simplicity, user can pass a shape parameter, which will be used to build a geobox from collection."
                )
            return GeoBoxBuilder.from_collection(collection, shape, anchor)

    @property
    def point_loader(self) -> PointLoader:
        if not self._point_loader:
            self._point_loader = PointLoader(
                self.parser.point,
                self.filter_config,
                self.cube_config,
                self.process_config,
                self.point_load_config,
            )
        return self._point_loader

    @property
    def vector_loader(self) -> VectorLoader:
        if not self._vector_loader:
            self._vector_loader = VectorLoader(
                self.parser.vector,
                self.filter_config,
                self.cube_config,
                self.process_config,
                self.vector_load_config,
            )
        return self._vector_loader

    @property
    def raster_loader(self) -> RasterLoader:
        if not self._raster_loader:
            self._raster_loader = RasterLoader(
                self.parser.raster,
                self.filter_config,
                self.cube_config,
                self.process_config,
                self.raster_load_config,
            )
        return self._raster_loader

    def get_geobox(
        self,
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
    def get_collection(
        endpoint: str | tuple[str, str] | Path,
    ) -> pystac.Collection:
        """Try to load collection from endpoint.

        Raises `EndpointType` if endpoint is not an acceptable type, or `EndpointException` if
        endpoint is not reachable
        """
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
