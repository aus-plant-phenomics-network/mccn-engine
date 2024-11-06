from abc import ABC, abstractmethod
from typing import Any

from owslib.wcs import WebCoverageService


class WcsImporter(ABC):
    def __init__(self, url: str, version: str | None = None) -> None:
        self.url = url
        # TODO: It may be poor design to instantiate a WCS in the constructor and use it later.
        # TODO: Consider if preferable to create new instances each time a request is made.
        self.wcs = WebCoverageService(url, version=version, timeout=300)

    @abstractmethod
    def get_metadata(self) -> None:
        pass

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

    @abstractmethod
    def get_data(self, bbox: list[float], layername: str) -> Any:
        pass


class DeaWcsImporter(WcsImporter):
    def __init__(self) -> None:
        # DEA supports WCS versions 1.0.0, 2.0.0 and 2.1.0. OWS
        super().__init__(url="https://ows.dea.ga.gov.au", version="1.0.0")

    def get_metadata(self) -> None:
        raise NotImplementedError

    def get_data(
        self,
        bbox: list[float],
        layername: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Any:
        response = self.wcs.getCoverage(
            identifier=layername,
            bbox=bbox,
            format="GeoTIFF",
            crs="EPSG:4326",
            resx=1 / 7200,
            resy=1 / 7200,
            Styles="tc",
        )
        return response


class DemWcsImporter(WcsImporter):
    def __init__(self) -> None:
        super().__init__(
            url="https://services.ga.gov.au/site_9/services/DEM_SRTM_1Second_Hydro_Enforced/"
            "MapServer/WCSServer?service=WCS",
            version="1.0.0",
        )

    def get_metadata(self) -> None:
        raise NotImplementedError

    def get_data(self, bbox: list[float], layername: str | None = None) -> Any:
        response = self.wcs.getCoverage(
            identifier="1",
            bbox=bbox,
            format="GeoTIFF",
            crs="EPSG:4326",
            resx=1,
            resy=1,
            Styles="tc",
        )
        return response


# class SiloWcsImporter(WcsImporter):
#     def __init__(self) -> None:
#         # SILO is not served via WCS
#         super().__init__("pass")


class SlgaWcsImporter(WcsImporter):
    def __init__(self) -> None:
        # Have to use version="1.0.0" for SGLA.
        super().__init__(
            url="https://www.asris.csiro.au/ArcGIS/services/TERN/CLY_ACLEP_AU_NAT_C/"
            "MapServer/WCSServer",
            version="1.0.0",
        )
        # TODO: There are many WCS endpoints for SLGA that serve different types of data. We should
        # TODO: find a way to support all of the sources we are interested int.

    def get_metadata(self) -> None:
        raise NotImplementedError

    def get_data(self, bbox: list[float], layername: str | None = None) -> Any:
        response = self.wcs.getCoverage(
            identifier="1",
            bbox=bbox,
            format="GeoTIFF",
            crs="EPSG:4326",
            resx=1,
            resy=1,
            Styles="tc",
        )
        return response


class WcsImporterFactory:
    @staticmethod
    def get_wcs_importer(source: str) -> WcsImporter:
        if source == "dea":
            return DeaWcsImporter()
        if source == "dem":
            return DemWcsImporter()
        if source == "slga":
            return SlgaWcsImporter()
        raise ValueError(f"Source: {source} is not supported.")
