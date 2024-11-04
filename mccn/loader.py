import geopandas as gpd
from odc.geo.geobox import GeoBox
import pystac
import rasterio
from rasterio import features
from rasterio.io import MemoryFile
import rioxarray
import xarray


def stac_load_vector(items: list[pystac.item.Item], gbox: GeoBox) -> xarray.Dataset:
    """
    Load several STAC :class: `~pystac.item.Item` objects (from the same or similar collections)
    as an :class: `xarray.Dataset`.

    This method takes STAC objects describing vector assets (shapefile, geojson) and rasterises
    them using the provided Geobox parameter.
    :param items: Iterable of STAC :class `~pystac.item.Item` to load.
    :param gbox: Allows to specify exact region/resolution/projection using
       :class:`~odc.geo.geobox.GeoBox` object
    :return: Xarray datacube with rasterised vector layers.
    """
    # TODO: Must check list of items for only vector data type
    # A temporary raster file is built in memory using attributes of the Geobox.
    with MemoryFile().open(
                        driver="GTiff",
                        crs=gbox.crs,
                        transform=gbox.transform,
                        dtype=rasterio.uint8,  # Results in uint8 dtype in xarray.DataArray
                        count=len(items),
                        width=gbox.width,
                        height=gbox.height) as memfile:
        for index, item in enumerate(items):
            vector_filepath = item.assets["data"].href
            vector = gpd.read_file(vector_filepath)
            # Reproject polygons into the target CRS
            vector = vector.to_crs(gbox.crs)
            geom = [shapes for shapes in vector.geometry]
            # Rasterise polygons with 1 if centre of pixel inside polygon, 0 otherwise
            rasterized = features.rasterize(geom,
                                            out_shape=gbox.shape,
                                            fill=0,
                                            out=None,
                                            transform=gbox.transform,
                                            all_touched=False,
                                            default_value=1,  # 1 for boolean mask
                                            dtype=None)
            # Raster bands are 1-indexed
            memfile.write(rasterized, index + 1)

    # The temporary raster file can then be read into xarray like a normal raster file.
    xx = rioxarray.open_rasterio(memfile.name)
    # TODO: Label the layers in the datacube
    return xx
