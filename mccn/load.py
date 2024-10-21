from collections.abc import Sequence

import pystac
import xarray as xr
from odc.geo import MaybeCRS
from odc.geo.geobox import GeoBox
from pyproj import Transformer
from stac_generator.csv.utils import read_csv


class StacExtensionError(Exception): ...


def csv_load(
    item: pystac.Item,
    column_info: Sequence[str] | str | None = None,
    crs: MaybeCRS = None,
    bbox: Sequence[float] | None = None,
    geobox: GeoBox | None = None,
) -> xr.Dataset:
    try:
        # Read in the raw csv file
        location = item.assets["source"].href
        X_name, Y_name = item.properties["X"], item.properties["Y"]
        prop_column_info = item.properties.get("column_info", None)
        if column_info:
            column_info = [column_info] if isinstance(column_info, str) else column_info
        columns = column_info if column_info else prop_column_info
        df = read_csv(
            location,
            X_name,
            Y_name,
            item.properties.get("T"),
            item.properties.get("date_format", "ISO8601"),
            columns,
            item.properties.get("groupby"),
        )
        X, Y = df.pop(X_name), df.pop(Y_name)
        # Change coord ref system if requested
        if crs:
            src_crs = item.properties.get("proj:epsg") if item.properties.get("proj:epsg") else 4326
            transformer = Transformer.from_crs(src_crs, crs, always_xy=True)
            X, Y = transformer.transform(X, Y)

        # Promote coord columns to multi index
        df.set_index([X, Y], inplace=True)

        # Slice geobox
        if geobox:
            bbox = geobox.boundingbox
            df = df.loc[(bbox[0], bbox[1]) : (bbox[2], bbox[3])]  # type: ignore[misc]
        return df.to_xarray()  # type: ignore[no-any-return]
    except KeyError as e:
        raise StacExtensionError("Key expected in stac item property not found") from e
