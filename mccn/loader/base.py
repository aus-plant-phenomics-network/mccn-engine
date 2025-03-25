import abc
import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar

from odc.geo.geobox import GeoBox

from mccn.parser import ParsedItem

if TYPE_CHECKING:
    import xarray as xr

T = TypeVar("T", bound=ParsedItem)


@dataclass
class FilterConfig:
    geobox: GeoBox
    start_ts: datetime.datetime | None = None
    end_ts: datetime.datetime | None = None
    bands: set[str] | None = None


@dataclass
class CubeConfig:
    x_coord: str = "lon"
    """Name of the x coordinate in the datacube"""
    y_coord: str = "lat"
    """Name of the y coordinate in the datacube"""
    t_coord: str = "time"
    """Name of the time coordinate in the datacube"""
    z_coord: str = "alt"
    """Name of the altitude coordinate in the datacube"""
    use_z: bool = False
    """Whether to use the altitude coordinate as an axis"""


class Loader(abc.ABC, Generic[T]):
    def __init__(
        self,
        items: Sequence[T],
        filter_config: FilterConfig,
        cube_config: CubeConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.items = items
        self.filter_config = filter_config
        self.cube_config = cube_config if cube_config else CubeConfig()

    @abc.abstractmethod
    def load(self) -> xr.Dataset:
        raise NotADirectoryError
