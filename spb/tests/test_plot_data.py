from sympy import symbols, cos, sin, log, Eq
from sympy.vector import CoordSys3D
from pytest import raises
from spb.plot_data import _build_series
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    ParametricSurfaceSeries, SurfaceOver2DRangeSeries, InteractiveSeries,
    ImplicitSeries, Vector2DSeries, Vector3DSeries
)

def test_build_series():
    x, y, u, v = symbols("x, y, u, v")

    # test automatic algoritm

    s = _build_series(cos(x), (x, -5, 5))
    assert isinstance(s, LineOver1DRangeSeries)

    s = _build_series((cos(x), sin(x)), (x, -5, 5))
    assert isinstance(s, Parametric2DLineSeries)

    s = _build_series(cos(x), sin(x), (x, -5, 5))
    assert isinstance(s, Parametric2DLineSeries)

    s = _build_series(cos(x), sin(x), x, (x, -5, 5))
    assert isinstance(s, Parametric3DLineSeries)

    s = _build_series(cos(x + y), (x, -5, 5))
    assert isinstance(s, SurfaceOver2DRangeSeries)

    s = _build_series(cos(x + y), sin(x + y), x, (x, -5, 5))
    assert isinstance(s, ParametricSurfaceSeries)

    s = _build_series((cos(x + y), sin(x + y), x), (x, -5, 5), (y, -2, 2))
    assert isinstance(s, ParametricSurfaceSeries)

    s = _build_series(u * cos(x), (x, -5, 5), params={u: 1})
    assert isinstance(s, InteractiveSeries)

    s = _build_series(Eq(x**2 + y**2, 5), (x, -5, 5), (y, -2, 2))
    assert isinstance(s, ImplicitSeries)

    s = _build_series(Eq(x**2 + y**2, 5) & (x > y), (x, -5, 5), (y, -2, 2))
    assert isinstance(s, ImplicitSeries)

    # test mapping
    s = _build_series(cos(x), (x, -5, 5), pt="p")
    assert isinstance(s, LineOver1DRangeSeries)

    s = _build_series(cos(x), sin(x), (x, -5, 5), pt="pp")
    assert isinstance(s, Parametric2DLineSeries)

    s = _build_series(cos(x), sin(x), x, (x, -5, 5), pt="p3dl")
    assert isinstance(s, Parametric3DLineSeries)

    s = _build_series(cos(x + y), (x, -5, 5), (y, -3, 3), pt="p3d")
    assert isinstance(s, SurfaceOver2DRangeSeries)

    # one missing rage
    s = _build_series(cos(x + y), (x, -5, 5), pt="p3d")
    assert isinstance(s, SurfaceOver2DRangeSeries)

    s = _build_series(cos(x + y), sin(x + y), x, (x, -5, 5), (y, -3, 3),
            pt="p3ds")
    assert isinstance(s, ParametricSurfaceSeries)

    # missing ranges
    s = _build_series(cos(x + y), sin(x + y), x, pt="p3ds")
    assert isinstance(s, ParametricSurfaceSeries)

    s = _build_series(u * cos(x), (x, -5, 5), params={u: 1}, pt="pinter")
    assert isinstance(s, InteractiveSeries)

def test_vectors():
    x, y, z = symbols("x:z")
    N = CoordSys3D("N")
    v1 = x * N.i + y * N.j
    v2 = z * N.i + x * N.j + y * N.k
    m1 = v1.to_matrix(N)
    m2 = v2.to_matrix(N)
    l1 = list(m1)
    # I need a 2D vector: delete the last component, which is zero
    l1 = l1[:-1]
    l2 = list(m2)

    # 2D vectors
    s = _build_series(v1, (x, -10, 10), (y, -5, 5))
    assert isinstance(s, Vector2DSeries)
    s = _build_series(m1, (x, -10, 10), (y, -5, 5))
    assert isinstance(s, Vector2DSeries)
    s = _build_series(l1, (x, -10, 10), (y, -5, 5))
    assert isinstance(s, Vector2DSeries)
    s = _build_series(v1, (x, -10, 10), (y, -5, 5), pt="v2d")
    assert isinstance(s, Vector2DSeries)
    s = _build_series(m1, (x, -10, 10), (y, -5, 5), pt="v2d")
    assert isinstance(s, Vector2DSeries)
    s = _build_series(l1, (x, -10, 10), (y, -5, 5), pt="v2d")
    assert isinstance(s, Vector2DSeries)

    s = _build_series(v2, (x, -10, 10), (y, -5, 5), (z, -8, 8))
    assert isinstance(s, Vector3DSeries)
    s = _build_series(m2, (x, -10, 10), (y, -5, 5), (z, -8, 8))
    assert isinstance(s, Vector3DSeries)
    s = _build_series(l2, (x, -10, 10), (y, -5, 5), (z, -8, 8))
    assert isinstance(s, Vector3DSeries)
    s = _build_series(v2, (x, -10, 10), (y, -5, 5), (z, -8, 8), pt="v3d")
    assert isinstance(s, Vector3DSeries)
    s = _build_series(m2, (x, -10, 10), (y, -5, 5), (z, -8, 8), pt="v3d")
    assert isinstance(s, Vector3DSeries)
    s = _build_series(l2, (x, -10, 10), (y, -5, 5), (z, -8, 8), pt="v3d")
    assert isinstance(s, Vector3DSeries)
