from abc import ABC, abstractmethod
from typing import Any, List

from owslib.wcs import WebCoverageService


class WcsImporter(ABC):
    def __init__(self, url: str) -> None:
        self.url = url
        # TODO: It may be poor design to instantiate a WCS in the constructor and use it later.
        # TODO: Consider if preferable to create new instances each time a request is made.
        self.wcs = WebCoverageService(url, version="1.0.0", timeout=300)

    @abstractmethod
    def get_metadata(self) -> None:
        pass

    @abstractmethod
    def get_capabilities(self) -> tuple:
        pass

    @abstractmethod
    def get_data(self, bbox: List[float]) -> Any:
        pass


class DemWcsImporter(WcsImporter):
    def __init__(self) -> None:
        super().__init__(
            url="https://services.ga.gov.au/site_9/services/DEM_SRTM_1Second_Hydro_Enforced/"
                "MapServer/WCSServer?request=GetCapabilities&service=WCS")

    def get_metadata(self) -> None:
        raise NotImplementedError

    def get_capabilities(self) -> tuple:
        # Get coverages and content dict keys
        content = self.wcs.contents
        keys = content.keys()

        print("Following data layers are available:")
        title_list = []
        description_list = []
        bbox_list = []
        for key in keys:
            print(f"key: {key}")
            print(f"title: {self.wcs[key].title}")
            title_list.append(self.wcs[key].title)
            print(f"{self.wcs[key].abstract}")
            description_list.append(self.wcs[key].abstract)
            print(f"bounding box: {self.wcs[key].boundingBoxWGS84}")
            bbox_list.append(self.wcs[key].boundingBoxWGS84)
            print("")

        return keys, title_list, description_list, bbox_list

    def get_data(self, bbox=List[float]) -> Any:
        response = self.wcs.getCoverage(identifier="1", bbox=bbox, format="GeoTIFF",
                                        crs="EPSG:4326",
                                        resx=1 / 3600, resy=1 / 3600, Styles="tc")
        return response


class WcsImporterFactory:
    @staticmethod
    def get_wcs_importer(source: str) -> WcsImporter:
        if source == "dem":
            return DemWcsImporter()
        else:
            raise ValueError(f"Source: {source} is not supported.")
