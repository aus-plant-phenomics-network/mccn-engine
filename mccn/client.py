from __future__ import annotations

import datetime
import json
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
from mccn.drawer import Canvas, Rasteriser
from mccn.extent import GeoBoxBuilder
from mccn.loader.point import PointLoader
from mccn.loader.raster import RasterLoader
from mccn.loader.vector import VectorLoader
from mccn.parser import Parser

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox

    from mccn._types import (
        CRS_T,
        AnchorPos_T,
        BBox_T,
        DType_Map_T,
        Dtype_T,
        MergeMethod_Map_T,
        MergeMethod_T,
        Nodata_Map_T,
        Nodata_T,
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
        x_dim: str = "x",
        y_dim: str = "y",
        t_dim: str = "time",
        mask_name: str = "__MASK__",
        combine_mask: bool = False,
        # Process config
        rename_bands: Mapping[str, str] | None = None,
        process_bands: Mapping[str, Callable] | None = None,
        nodata: Nodata_Map_T = 0,
        nodata_fallback: Nodata_T = 0,
        time_groupby: TimeGroupby = "time",
        merge_method: MergeMethod_Map_T = None,
        merge_method_fallback: MergeMethod_T = "replace",
        dtype: DType_Map_T = None,
        dtype_fallback: Dtype_T = "float64",
        # Multi-processing
        num_workers: int = 4,
    ) -> None:
        """Constructor for the mccn engine

        Args:
            endpoint (str | Path | tuple[str, str] | None, optional): discover project by endpoint. Endpoint can be a tuple of string, which specifies the STAC API URL, and stac collection ID. Endpoint can also be a string or a Path object to a `collection.json` file on local file system. Defaults to None.
            collection (pystac.Collection | None, optional): discover project by a pystac Collection object. Defaults to None.
            items (Sequence[pystac.Item] | None, optional): discover project by a sequence of pystac Items. Defaults to None.
            shape (Shape_T | None, optional): define the shape of geobox. Defaults to None.
            resolution (Resolution_T | None, optional): define the resolution of the geobox. Defaults to None.
            bbox (BBox_T | None, optional): define the bounding box of the geobox. Defaults to None.
            anchor (AnchorPos_T, optional): additional geobox parameter. Defaults to "default".
            crs (CRS_T, optional): geobox's crs. Defaults to 4326.
            geobox (GeoBox | None, optional): geobox object. Defaults to None.
            start_ts (str | pd.Timestamp | datetime.datetime | None, optional): date filtering - start. Defaults to None.
            end_ts (str | pd.Timestamp | datetime.datetime | None, optional): date filtering - end. Defaults to None.
            bands (set[str] | None, optional): set of requested bands. Defaults to None.
            mask_only (bool, optional): whether to load only masks for vector assets. Defaults to False.
            use_all_vectors (bool, optional): when bands are requested, should all vectors be loaded or only vectors with matching bands be loaded. Non-matching vectors will have their geometry layers loaded to the datacube if True. Defaults to True.
            x_dim (str, optional): x dimension name of the datacube. Defaults to "x".
            y_dim (str, optional): y dimension name of the datacube. Defaults to "y".
            t_dim (str, optional): t dimension name of the datacube. Defaults to "time".
            mask_name (str, optional): name of the combined mask layer, if combine_mask is True. Defaults to "__MASK__".
            combine_mask (bool, optional): whether to combine all geometry layers of all vector assets into a single layer. By default, each geometry layer will be loaded as an independent geometry layer. Defaults to False.
            nodata (Nodata_Map_T, optional): fill value for nodata. If a single value is provided, the value will be used for all layers. If a dictionary is provided, each nodata value will apply for matching key layers. Defaults to 0.
            nodata_fallback (Nodata_T, optional): fill value fall back for nodata. If a dictionary is provided for nodata, the nodata_fallback value will be used for layers that are not in the nodata dict. Defaults to 0.
            time_groupby (TimeGroupby, optional): how datetimes are groupped. Acceptable values are year, month, day, hour, minute or time. If time is provided, no time round up is performed. If time is a value, will round up to the nearest matching date. Defaults to "time".
            merge_method (MergeMethod_Map_T, optional): how overlapping values are merged. Acceptable values are min, max, mean, sum, and replace and None. If None is provided, will use the replace strategy. Also accepts a dictionary if fine-grain control over a specific layer is required. Defaults to None.
            merge_method_fallback (MergeMethod_T, optional): merge value fallback, applies when a layer name is not in merge_method dictionary. Defaults to "replace".
            dtype (DType_Map_T, optional): set dtype for a layer. Also accepts a dictionary for fine-grained control. Defaults to None.
            dtype_fallback (Dtype_T, optional): dtype fallback, when a layer's name is not in dtype dictionary. Defaults to "float64".
        """
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
            x_dim=x_dim,
            y_dim=y_dim,
            t_dim=t_dim,
            mask_name=mask_name,
            combine_mask=combine_mask,
        )
        self.process_config = ProcessConfig(
            rename_bands,
            process_bands,
            nodata,
            nodata_fallback,
            time_groupby,
            merge_method,
            merge_method_fallback,
            dtype,
            dtype_fallback,
        )
        # Parse items
        self.parser = Parser(self.filter_config, self.items)
        self.parser()
        # Prepare canvas
        self.canvas = Canvas.from_geobox(
            self.cube_config.x_dim,
            self.cube_config.y_dim,
            self.cube_config.t_dim,
            self.cube_config.spatial_ref_dim,
            self.geobox,
            self.process_config.dtype,
            self.process_config.dtype_fallback,
            self.process_config.nodata,
            self.process_config.nodata_fallback,
            self.process_config.merge_method,
            self.process_config.merge_method_fallback,
        )
        self.rasteriser = Rasteriser(canvas=self.canvas)
        self.point_loader: PointLoader = PointLoader(
            self.parser.point,
            self.rasteriser,
            self.filter_config,
            self.cube_config,
            self.process_config,
        )
        self.vector_loader = VectorLoader(
            self.parser.vector,
            self.rasteriser,
            self.filter_config,
            self.cube_config,
            self.process_config,
        )
        self.raster_loader: RasterLoader = RasterLoader(
            self.parser.raster,
            self.rasteriser,
            self.filter_config,
            self.cube_config,
            self.process_config,
        )
        self.num_workers = num_workers

    def load(self) -> xr.Dataset:
        self.raster_loader.load()
        self.vector_loader.load()
        self.point_loader.load()
        return self.rasteriser.compile()

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

    @staticmethod
    def to_cdf(ds: xr.Dataset, path: str | Path) -> None:
        if ds.attrs:
            for k, v in ds.attrs.items():
                if isinstance(v, dict):
                    try:
                        ds.attrs[k] = json.dumps(v)
                    except Exception:
                        raise ValueError(
                            f"Unable to serialise cdf due to cube's attrs: {ds.attrs}"
                        )
        ds.to_netcdf(path)

    @staticmethod
    def from_cdf(path: str | Path) -> xr.Dataset:
        ds = xr.open_dataset(path)
        if ds.attrs:
            for k, v in ds.attrs.items():
                if isinstance(v, str):
                    try:
                        ds.attrs[k] = json.loads(v)
                    except Exception:
                        pass
        return ds
