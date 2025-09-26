
from forex_diffusion.patterns.triangles import make_triangle_detectors
from .conftest import make_trending_series

def test_triangle_runs():
    df = make_trending_series()
    dets = make_triangle_detectors()
    for d in dets:
        evs = d.detect(df)
        assert isinstance(evs, list)
