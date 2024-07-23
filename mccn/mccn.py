from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from io import BytesIO
from typing import Iterator, List, Optional

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

    def _query(self, col_id: str) -> Iterator:
        # TODO: Add options for querying.
        query = self.stac_client.search(collections=[col_id])
        return query.items()

    def load(self, col_id: str, bands: Optional[List[str]] = None, groupby: str = "id",
             crs: Optional[str] = None, geobox: Optional[GeoBox] = None,
             lazy=False) -> xarray.Dataset:
        # TODO: Expose other parameters to the stac_load function
        print(f"Loading data for {col_id}.")
        pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        # To lazy load we need to pass something to the chunks parameter of stac_load() function.
        if lazy:
            chunks = {'x': 2048, 'y': 2048}
        else:
            chunks = None
        return stac_load(
            self._query(col_id), bands=bands, groupby=groupby, chunks=chunks,  # type: ignore
            progress=tqdm, pool=pool, crs=crs, geobox=geobox
        )

    @staticmethod
    def load_public(source: str, bbox: List[float]):
        # Demo basic load function for WCS endpoint. Only DEM is currently supported.
        response = WcsImporterFactory().get_wcs_importer(source).get_data(bbox)
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
