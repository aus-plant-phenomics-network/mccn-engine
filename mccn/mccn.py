from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from io import BytesIO
from typing import Iterator, List, Optional, Dict

import numpy as np
from odc.geo.geobox import GeoBox
from odc.stac import stac_load
import pystac_client
from tqdm import tqdm
import rioxarray
import xarray

from mccn.wcs_importer import WcsImporterFactory


class Mccn:
    def __init__(self, stac_url):
        # This class needs to be responsible for gathering both public WCS and node generated STAC
        # described data into an x-array data structure.
        print(f"Connection to the STAC endpoint at {stac_url}.")
        self.stac_client = pystac_client.Client.open(stac_url)
        self.lat_count = None
        self.lon_count = None
        self.bbox = None

    def _query(self, col_id: str) -> Iterator:
        # TODO: Add options for querying.
        query = self.stac_client.search(collections=[col_id])
        return query.items()

    def load_stac(self, col_id: str, bands: Optional[List[str]] = None, groupby: str = "id",
                  crs: Optional[str] = None, geobox: Optional[GeoBox] = None,
                  lazy: bool = False) -> xarray.Dataset:
        # TODO: Expose other parameters to the stac_load function
        print(f"Loading data for {col_id}.")
        pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        # To lazy load we need to pass something to the chunks parameter of stac_load() function.
        if lazy:
            chunks = {'x': 2048, 'y': 2048}
        else:
            chunks = None
        xx = stac_load(
            self._query(col_id), bands=bands, groupby=groupby, chunks=chunks,  # type: ignore
            progress=tqdm, pool=pool, crs=crs, geobox=geobox
        )
        self.lat_count = xx.dims["latitude"]
        self.lon_count = xx.dims["longitude"]
        min_lon = xx.longitude.min().item()
        max_lon = xx.longitude.max().item()
        min_lat = xx.latitude.min().item()
        max_lat = xx.latitude.max().item()
        self.bbox = [min_lon, min_lat, max_lon, max_lat]
        return xx

    def load_public(self, source: str, bbox: List[float], layername=None):
        # Demo basic load function for WCS endpoint. Only DEM is currently supported.
        response = WcsImporterFactory().get_wcs_importer(source).get_data(bbox, layername)
        return rioxarray.open_rasterio(BytesIO(response.read()))

    @staticmethod
    def plot(xx: xarray.Dataset):
        # TODO: Only plots the time=0 index of the array. Options are to handle multiple time
        # TODO: indices in this function or stipulate one index as parameter.
        reduce_dim = "band"
        xx_0 = xx.isel(time=0).to_array(reduce_dim)
        return xx_0.plot.imshow(
            col=reduce_dim,
            size=5,
            vmin=int(xx_0.min()),
            vmax=int(xx_0.max()),
        )

    def load(self, col_id: str, bands: Optional[List[str]] = None, groupby: str = "id",
             crs: Optional[str] = None, geobox: Optional[GeoBox] = None, lazy: bool = False,
             source: Optional[Dict[str, str]] = None):
        xx = self.load_stac(col_id, bands=bands, groupby=groupby, crs=crs, geobox=geobox,
                            lazy=lazy)
        # min_lon = xx.longitude.min().item()
        # max_lon = xx.longitude.max().item()
        # min_lat = xx.latitude.min().item()
        # max_lat = xx.latitude.max().item()
        for source_name, layer_name in source.items():
            yy = self.load_public(source=source_name, bbox=self.bbox, layername=layer_name)
            # yy = yy.astype(np.int32)
            yy = yy.rename({"x": "longitude", "y": "latitude"})
            yy = yy.interp(
                # longitude=np.linspace(min_lon, max_lon, self.lon_count),
                # latitude=np.linspace(min_lat, max_lat, self.lat_count),
                longitude=list(xx.longitude.values),
                latitude=list(xx.latitude.values),
                method="linear"
            )
            # yy = yy.squeeze()
            yy = yy.to_dataset(name="elevation")
            yy["spatial_ref"] = xx.spatial_ref
            yy["time"] = xx.time
            yy = yy.squeeze(dim="band")
            xy = xarray.concat([xx, yy], dim="time")

        return xy
