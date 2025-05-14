MCCN-Engine is a python library for loading and combining STAC described asset, generated using the [stac_generator](https://aus-plant-phenomics-network.github.io/stac-generator/), into an [xarray](https://docs.xarray.dev/en/stable/) datacube.

## Installation

Install from PyPi:

```bash
pip install mccn-engine
```

## Workflow

## Note

The current version of the mccn-engine only works with STAC items generated using the library [stac_generator](https://aus-plant-phenomics-network.github.io/stac-generator/).

### Item Discovery

The first phase is discovering all described STAC assets. The engine does this by checking one of the following parameters:

- `items`: a sequence of `pystac.Item` objects.
- `collection`: a `pystac.Collection` object that contains items to be loaded.
- `endpoint`:
    - a `str/Path` path to the file `collection.json` on local file system. This collection json will be parsed into a `pystac.Collection` object, from which all items will be collected.
    - a `tuple[str, str]`, containing a STAC API endpoint url, and the collection id. A request will be made to the endpoint, from which all items will be collected.

### Filtering Option

By default, the mccn engine will load all items found in the discovery phase. Users can filter items to be loaded into the data cube via the following options to be provided to the constructor:

### Filtering by geobox

For building datacubes, we utilise the [GeoBox](https://odc-geo.readthedocs.io/en/latest/intro-geobox.html) object, which is essentially a bounding box that knows its resolution and crs. The engine constructor accepts the following parameters:

- `geobox`: if the user provides a `GeoBox` object, no other geobox related parameters will be parsed. Only assets that overlapp with the geobox (either contains, is contained by, or intersects with) will be loaded.
- `shape`: cube shape - e.g `(100, 100)`
- `resolution`: cube spatial resolution - e.g 0.005.
- `bbox`: cube bounding box.
- `crs`: cube's CRS.

If `geobox` value is not provided, user MUST provide one of `shape` or `resolution`. If no spatial filtering is needed, the user can just leave `bbox` to None.

The easiest way to build a geobox is to just provide the `shape` value, from which a geobox of matching shape will be derived from the bounding box that contains all assets.


#### Filtering by dates:

User can provide a date string, a None value, or a python `datetime.datetime` or `pandas.Timestamp` object. If `start_ts` is None, there is no time lowerbound filtering applied and vice-versa. If timestamp is not utc, value will be converted to utc. If timestamp is not timezone aware, will be treated as utc time.

- `start_ts`: starting timestamp, inclusive.
- `end_ts`: ending timestamp, inclusive.

#### Filtering by bands:

User can provide a set of band names to be loaded as layers in the cube. Only assets containing the matched bands will be loaded. Note that by default, the engine load all geometry layers of vector assets, regardless of whether the vector assets contain the requesting bands. Users can change this behaviour by setting `use_all_vectors` to False.

### Rasterising Options

User can provide the following parameters to control how data get written to the datacube:

- `time_groupby`: group similar dates into a single value and aggregate data values within the period into the same layer. For instance, if `time_groupby` is `year`, a time series values over the same year (365 days) will be aggregated into a single layer with date being the 1st day of that year. Acceptable values for `time_groupby` are:
    - `year`: values within the same year are groupped together
    - `month`: values within the same month are groupped together
    - `day`: values within the same hour are groupped together
    - `minute`: values within the same minute are groupped together
    - `time`: does not perform grouping.
- `nodata`: fill value for nodata. If a single value is provided, the value will be used for all layers. If a dictionary is provided, each nodata value will apply for matching key layers. Defaults to 0.
- `nodata_fallback`: fill value fall back for nodata. If a dictionary is provided for nodata, the nodata_fallback value will be used for layers that are not in the nodata dict. Defaults to 0.
- `time_groupby`: how datetimes are groupped. Acceptable values are year, month, day, hour, minute or time. If time is provided, no time round up is performed. If time is a value, will round up to the nearest matching date. Defaults to "time".
- `merge_method`: how overlapping values are merged. Acceptable values are min, max, mean, sum, and replace and None. If None is provided, will use the replace strategy. Also accepts a dictionary if fine-grain control over a specific layer is required. Defaults to None.
- `merge_method_fallback`: merge value fallback, applies when a layer name is not in merge_method dictionary. Defaults to "replace".
- `dtype`: set dtype for a layer. Also accepts a dictionary for fine-grained control. Defaults to None.
- `dtype_fallback`: dtype fallback, when a layer's name is not in dtype dictionary. Defaults to "float64"
