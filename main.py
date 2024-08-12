from io import BytesIO
import rioxarray
from mccn.wcs_importer import WcsImporterFactory
from mccn.mccn import Mccn

if __name__ == "__main__":
    # wcs_client = WcsImporterFactory().get_wcs_importer("dea")
    # wcs_client.get_capabilities()
    # bbox = [138.62892153159546, -34.96895662834354, 138.6293039082798, -34.968640581295116]
    # response = wcs_client.get_data(bbox=bbox, layername="s2_ls_combined")
    # dc = rioxarray.open_rasterio(BytesIO(response.read()))
    # print(dc.attrs)
    m = Mccn("placeholder")
    m.load("gryfn_test_waite", source={"dem": "1"})
