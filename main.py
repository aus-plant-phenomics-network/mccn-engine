from odc.geo.geobox import GeoBox

from mccn.mccn import Mccn


if __name__ == "__main__":
    m = Mccn("http://localhost:8082")
    gb = GeoBox.from_bbox((434067.17171849194, 6950889.892785188, 434208.66175697366, 6951062.348691959), "epsg:28356", shape=(100, 100))
    m.load("test_vector_loader", geobox=gb, lazy=False, mask=True)

    pass