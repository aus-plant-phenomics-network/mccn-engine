from typing import Iterable

import affine
import geopandas as gpd
import pandas as pd
import pystac
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.io import MemoryFile
import xarray as xr

def timeseries_loader(fname: str):
    # Assume the data is in csv format.
    df = pd.read_csv(fname, header=0)
    df.set_index(["Longitude", "Latitude", "mid_depth"], inplace=True)
    ds = xr.Dataset.from_dataframe(df)
    return ds

def vector_loader(vector_filepath: str, shape: tuple=None, transform: affine.Affine=None,
                  crs: CRS=None):
    vector = gpd.read_file(vector_filepath)
    vector = vector.to_crs(str(crs))
    geom = [shapes for shapes in vector.geometry]
    rasterized = features.rasterize(geom,
                                     out_shape=shape,
                                     fill=0,
                                     out=None,
                                     transform=transform,
                                     all_touched=False,
                                     default_value=1,
                                     dtype=None)

    memfile = MemoryFile()
    with memfile.open(
            driver="GTiff",
            crs=crs,
            transform=transform,
            dtype=rasterio.uint16,
            count=1,
            width=shape[1],
            height=shape[0]) as dst:
        dst.write(rasterized, indexes=1)

    return memfile