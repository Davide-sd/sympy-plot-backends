from sympy import (
    symbols, cos, sin, Eq, I, Abs, re, im, arg,
    exp, pi, gamma, sqrt
)
from sympy.geometry import (
    Plane, Polygon, Circle, Ellipse, Line, Segment,
    Ray, Line3D, Point2D, Point3D,
    Segment3D, Ray3D,
)
from sympy.vector import CoordSys3D
from pytest import raises
from spb.plot_data import _build_series, get_plot_data
from spb.series import (
    LineOver1DRangeSeries,
    AbsArgLineSeries,
    Parametric2DLineSeries,
    Parametric3DLineSeries,
    ParametricSurfaceSeries,
    SurfaceOver2DRangeSeries,
    ContourSeries,
    ImplicitSeries,
    Vector2DSeries,
    Vector3DSeries,
    InteractiveSeries,
    LineInteractiveSeries,
    AbsArgLineInteractiveSeries,
    Parametric2DLineInteractiveSeries,
    Parametric3DLineInteractiveSeries,
    ParametricSurfaceInteractiveSeries,
    SurfaceInteractiveSeries,
    ContourInteractiveSeries,
    GeometryInteractiveSeries,
    PlaneInteractiveSeries,
    Vector2DInteractiveSeries,
    Vector3DInteractiveSeries,
    Vector3DInteractiveSeries,
    SliceVector3DInteractiveSeries,
    ComplexSurfaceBaseSeries,
    ComplexSurfaceSeries,
    ComplexDomainColoringSeries,
    ComplexInteractiveBaseSeries,
    ComplexSurfaceInteractiveSeries,
    ComplexDomainColoringInteractiveSeries,
    SliceVector3DSeries,
    GeometrySeries,
    PlaneSeries,
)
from spb.ccomplex.wegert import domain_coloring, create_colorscale
from pytest import raises
import numpy as np


def test_auto_build_series():
    # verify that _build_series() is able to automatically detect the
    # correct data series to build

    x, y = symbols("x, y")

    s = _build_series(cos(x), (x, -5, 5))
    assert isinstance(s, LineOver1DRangeSeries)
    s = get_plot_data(cos(x), (x, -5, 5), get_series=True)
    assert isinstance(s, LineOver1DRangeSeries)
    s = get_plot_data(cos(x), (x, -5, 5), get_series=False)
    assert isinstance(s, (list, tuple))

    s = _build_series(sqrt(x), (x, -5, 5), absarg=False)
    assert isinstance(s, LineOver1DRangeSeries)
    s = _build_series(sqrt(x), (x, -5, 5), absarg=True)
    assert isinstance(s, AbsArgLineSeries)

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

    s = _build_series(Eq(x ** 2 + y ** 2, 5), (x, -5, 5), (y, -2, 2))
    assert isinstance(s, ImplicitSeries)

    s = _build_series(Eq(x ** 2 + y ** 2, 5) & (x > y), (x, -5, 5), (y, -2, 2))
    assert isinstance(s, ImplicitSeries)


def test_auto_build_series_interactive():
    # verify that _build_series() is able to automatically detect the
    # correct interactive data series to build

    x, y, u = symbols("x, y, u")

    s = _build_series(u * sqrt(x), (x, -5, 5), params={u: 1}, absarg=False)
    assert isinstance(s, LineInteractiveSeries)

    s = _build_series(u * sqrt(x), (x, -5, 5), params={u: 1}, absarg=True)
    assert isinstance(s, AbsArgLineInteractiveSeries)

    s = _build_series(u * cos(x), sin(x), (x, -5, 5), params={u: 1})
    assert isinstance(s, Parametric2DLineInteractiveSeries)

    s = _build_series(u * cos(x), sin(x), x, (x, -5, 5), params={u: 1})
    assert isinstance(s, Parametric3DLineInteractiveSeries)

    s = _build_series(u * cos(x + y), (x, -5, 5), (y, -5, 5),
        params={u: 1}, threed=True)
    assert isinstance(s, SurfaceInteractiveSeries)

    s = _build_series(u * cos(x + y), (x, -5, 5), (y, -5, 5),
        params={u: 1}, threed=False)
    assert isinstance(s, ContourInteractiveSeries)

    s = _build_series(u * cos(x + y), sin(x + y), x, (x, -5, 5), (y, -5, 5),
        params={u: 1})
    assert isinstance(s, ParametricSurfaceInteractiveSeries)


def test_auto_build_series_complex():
    # verify that _build_series() is able to automatically detect the
    # correct complex-related data series to build

    x = symbols("x")

    s = _build_series(sqrt(x), (x, -5, 5), absarg=True)
    assert isinstance(s, AbsArgLineSeries)

    s = _build_series(x, (x, -5, 5), real=True)
    assert isinstance(s, LineOver1DRangeSeries)
    assert s.expr == re(x)

    s = _build_series(x, (x, -5, 5), imag=True)
    assert isinstance(s, LineOver1DRangeSeries)
    assert s.expr == im(x)

    s = _build_series(x, (x, -5, 5), abs=True)
    assert isinstance(s, LineOver1DRangeSeries)
    assert s.expr == sqrt(re(x)**2 + im(x)**2)

    s = _build_series(x, (x, -5, 5), arg=True)
    assert isinstance(s, LineOver1DRangeSeries)
    assert s.expr == arg(x)

    s = _build_series(sqrt(x), (x, -5-5j, 5+5j),
        real=True, absarg=False, threed=True)
    assert isinstance(s, ComplexSurfaceSeries)
    assert s.is_3Dsurface and (not s.is_contour)

    s = _build_series(sqrt(x), (x, -5-5j, 5+5j),
        real=True, absarg=False, threed=False)
    assert isinstance(s, ComplexSurfaceSeries)
    assert (not s.is_3Dsurface) and s.is_contour

    s = _build_series(sqrt(x), (x, -5-5j, 5+5j), absarg=True, threed=False)
    assert isinstance(s, ComplexDomainColoringSeries)
    assert not s.is_3Dsurface

    s = _build_series(sqrt(x), (x, -5-5j, 5+5j), absarg=True, threed=True)
    assert isinstance(s, ComplexDomainColoringSeries)
    assert s.is_3Dsurface


def test_auto_build_series_complex_interactive():
    # verify that _build_series() is able to automatically detect the
    # correct interactive complex-related data series to build

    x, u = symbols("x, u")

    s = _build_series(u * sqrt(x), (x, -5, 5), params={u: 1}, absarg=True)
    assert isinstance(s, AbsArgLineInteractiveSeries)

    s = _build_series(u * x, (x, -5, 5), params={u: 1}, real=True)
    assert isinstance(s, LineInteractiveSeries)
    assert s.expr == re(u * x)

    s = _build_series(u * x, (x, -5, 5), params={u: 1}, imag=True)
    assert isinstance(s, LineInteractiveSeries)
    assert s.expr == im(u * x)

    s = _build_series(u * x, (x, -5, 5), params={u: 1}, abs=True)
    assert isinstance(s, LineInteractiveSeries)
    assert s.expr == sqrt(re(u * x)**2 + im(u * x)**2)

    s = _build_series(u * x, (x, -5, 5), params={u: 1}, arg=True)
    assert isinstance(s, LineInteractiveSeries)
    assert s.expr == arg(u * x)

    s = _build_series(u * sqrt(x), (x, -5-5j, 5+5j), params={u: 1},
        real=True, absarg=False, threed=True)
    assert isinstance(s, ComplexSurfaceInteractiveSeries)
    assert s.is_3Dsurface and (not s.is_contour)

    s = _build_series(u * sqrt(x), (x, -5-5j, 5+5j), params={u: 1},
        real=True, absarg=False, threed=False)
    assert isinstance(s, ComplexSurfaceInteractiveSeries)
    assert (not s.is_3Dsurface) and s.is_contour

    s = _build_series(u * sqrt(x), (x, -5-5j, 5+5j), params={u: 1},
        absarg=True, threed=False)
    assert isinstance(s, ComplexDomainColoringInteractiveSeries)
    assert not s.is_3Dsurface

    s = _build_series(u * sqrt(x), (x, -5-5j, 5+5j), params={u: 1},
        absarg=True, threed=True)
    assert isinstance(s, ComplexDomainColoringInteractiveSeries)
    assert s.is_3Dsurface

    

def test_mapping_build_series():
    # verify that the keyword `pt` produces the expected result

    x, y = symbols("x, y")

    s = _build_series(cos(x), (x, -5, 5), pt="p")
    assert isinstance(s, LineOver1DRangeSeries)
    s = get_plot_data(cos(x), (x, -5, 5), pt="p", get_series=True)
    assert isinstance(s, LineOver1DRangeSeries)
    s = get_plot_data(cos(x), (x, -5, 5), pt="p", get_series=False)
    assert isinstance(s, (list, tuple))

    s = _build_series(sqrt(x), (x, -5, 5), pt="p", absarg=False)
    assert isinstance(s, LineOver1DRangeSeries)
    s = _build_series(sqrt(x), (x, -5, 5), pt="p", absarg=True)
    assert isinstance(s, AbsArgLineSeries)

    s = _build_series(cos(x), sin(x), (x, -5, 5), pt="pp")
    assert isinstance(s, Parametric2DLineSeries)

    s = _build_series(cos(x), sin(x), x, (x, -5, 5), pt="p3dl")
    assert isinstance(s, Parametric3DLineSeries)

    s = _build_series(cos(x + y), (x, -5, 5), (y, -3, 3), pt="p3d")
    assert isinstance(s, SurfaceOver2DRangeSeries)

    # one missing rage
    s = _build_series(cos(x + y), (x, -5, 5), pt="p3d")
    assert isinstance(s, SurfaceOver2DRangeSeries)

    s = _build_series(cos(x + y), sin(x + y), x, (x, -5, 5), (y, -3, 3), pt="p3ds")
    assert isinstance(s, ParametricSurfaceSeries)

    # missing ranges
    s = _build_series(cos(x + y), sin(x + y), x, pt="p3ds")
    assert isinstance(s, ParametricSurfaceSeries)


def test_mapping_build_series_interactive():
    # verify that the keyword `pt` produces the expected interactive series

    x, y, u = symbols("x, y, u")

    # NOTE: with interactive series, all ranges must be provided

    s = _build_series(u * cos(x), (x, -5, 5), params={u: 1}, pt="pinter")
    assert isinstance(s, LineInteractiveSeries)

    s = _build_series(u * cos(x), (x, -5, 5), params={u: 1}, pt="pinter",
        absarg=True)
    assert isinstance(s, AbsArgLineInteractiveSeries)

    s = _build_series(u * cos(x), u * sin(x), (x, -5, 5),
        params={u: 1}, pt="pinter")
    assert isinstance(s, Parametric2DLineInteractiveSeries)

    s = _build_series(u * cos(x), u * sin(x), x, (x, -5, 5),
        params={u: 1}, pt="pinter")
    assert isinstance(s, Parametric3DLineInteractiveSeries)

    s = _build_series(u * cos(x + y), (x, -5, 5), (y, -3, 3),
        params={u: 1}, pt="pinter", threed=True)
    assert isinstance(s, SurfaceInteractiveSeries)

    s = _build_series(u * cos(x + y), (x, -5, 5), (y, -3, 3),
        params={u: 1}, pt="pinter", threed=False)
    assert isinstance(s, ContourInteractiveSeries)

    # one missing rage
    s = _build_series(u * cos(x + y), (x, -5, 5), (y, -3, 3),
        params={u: 1}, pt="pinter")
    assert isinstance(s, SurfaceInteractiveSeries)

    s = _build_series(u * cos(x + y), u * sin(x + y), x, (x, -5, 5), (y, -3, 3),
        params={u: 1}, pt="pinter")
    assert isinstance(s, ParametricSurfaceInteractiveSeries)


def test_auto_build_series_complex():
    # verify that the keyword `pt` produces the expected complex-related data
    # series

    x, u = symbols("x, u")

    s = _build_series(sqrt(x), (x, -5, 5), absarg=True, pt="p")
    assert isinstance(s, AbsArgLineSeries)

    s = _build_series(x, (x, -5, 5), real=True, pt="p")
    assert isinstance(s, LineOver1DRangeSeries)
    assert s.expr == re(x)

    s = _build_series(x, (x, -5, 5), imag=True, pt="p")
    assert isinstance(s, LineOver1DRangeSeries)
    assert s.expr == im(x)

    s = _build_series(x, (x, -5, 5), abs=True, pt="p")
    assert isinstance(s, LineOver1DRangeSeries)
    assert s.expr == sqrt(re(x)**2 + im(x)**2)

    s = _build_series(x, (x, -5, 5), arg=True, pt="p")
    assert isinstance(s, LineOver1DRangeSeries)
    assert s.expr == arg(x)

    s = _build_series(sqrt(x), (x, -5-5j, 5+5j),
        real=True, absarg=False, threed=True, pt="c")
    assert isinstance(s, ComplexSurfaceSeries)
    assert s.is_3Dsurface and (not s.is_contour)

    s = _build_series(sqrt(x), (x, -5-5j, 5+5j),
        real=True, absarg=False, threed=False, pt="c")
    assert isinstance(s, ComplexSurfaceSeries)
    assert (not s.is_3Dsurface) and s.is_contour

    s = _build_series(sqrt(x), (x, -5-5j, 5+5j),
        absarg=True, threed=False, pt="c")
    assert isinstance(s, ComplexDomainColoringSeries)
    assert not s.is_3Dsurface

    s = _build_series(sqrt(x), (x, -5-5j, 5+5j),
        absarg=True, threed=True, pt="c")
    assert isinstance(s, ComplexDomainColoringSeries)
    assert s.is_3Dsurface


def test_mapping_build_series_complex_interactive():
    # verify that the keyword `pt` produces the expected interactive 
    # complex-related data series

    x, u = symbols("x, u")

    s = _build_series(u * sqrt(x), (x, -5, 5), params={u: 1},
        pt="pinter", absarg=True)
    assert isinstance(s, AbsArgLineInteractiveSeries)

    s = _build_series(u * sqrt(x), (x, -5, 5), params={u: 1},
        pt="pinter", arg=True)
    assert isinstance(s, LineInteractiveSeries)
    assert s.expr == arg(u * sqrt(x))

    s = _build_series(u * sqrt(x), (x, -5, 5), params={u: 1},
        pt="pinter", abs=True)
    assert isinstance(s, LineInteractiveSeries)
    assert s.expr == sqrt(re(u * sqrt(x))**2 + im(u * sqrt(x))**2)

    s = _build_series(u * sqrt(x), (x, -5, 5), params={u: 1},
        pt="pinter", real=True)
    assert isinstance(s, LineInteractiveSeries)
    assert s.expr == re(u * sqrt(x))

    s = _build_series(u * sqrt(x), (x, -5, 5), params={u: 1},
        pt="pinter", imag=True)
    assert isinstance(s, LineInteractiveSeries)
    assert s.expr == im(u * sqrt(x))

    s = _build_series(u * sqrt(x), (x, -5-5j, 5+5j), params={u: 1},
        pt="pinter", real=True, threed=False)
    assert isinstance(s, ComplexSurfaceInteractiveSeries)
    assert (not s.is_3Dsurface) and s.is_contour

    s = _build_series(u * sqrt(x), (x, -5-5j, 5+5j), params={u: 1},
        pt="pinter", real=True, threed=True)
    assert isinstance(s, ComplexSurfaceInteractiveSeries)
    assert s.is_3Dsurface and (not s.is_contour)

    s = _build_series(u * sqrt(x), (x, -5-5j, 5+5j), params={u: 1},
        pt="pinter", absarg=True, threed=False)
    assert isinstance(s, ComplexDomainColoringInteractiveSeries)
    assert not s.is_3Dsurface

    s = _build_series(u * sqrt(x), (x, -5-5j, 5+5j), params={u: 1},
        pt="pinter", absarg=True, threed=True)
    assert isinstance(s, ComplexDomainColoringInteractiveSeries)
    assert s.is_3Dsurface


def test_geometry():
    def do_test(*g, s=GeometrySeries, **kwargs):
        s1 = _build_series(*g, pt="g", **kwargs)
        assert isinstance(s1, s)
        # since the range could be None, it is imperative to test that label
        # receive the correct value.
        assert s1.label == str(g[0])
        s2 = _build_series(*g, **kwargs)
        assert isinstance(s2, s)
        assert s2.label == str(g[0])
        assert np.array_equal(s1.get_data(), s2.get_data(), equal_nan=True)

    x, y, z = symbols("x, y, z")
    do_test(Point2D(1, 2))
    do_test(Point3D(1, 2, 3))
    do_test(Ray((1, 2), (3, 4)))
    do_test(Segment((1, 2), (3, 4)))
    do_test(Line((1, 2), (3, 4)), (x, -5, 5))
    do_test(Ray3D((1, 2, 3), (3, 4, 5)))
    do_test(Segment3D((1, 2, 3), (3, 4, 5)))
    do_test(Line3D((1, 2, 3), (3, 4, 5)))
    do_test(Polygon((1, 2), 3, n=10))
    do_test(Circle((1, 2), 3))
    do_test(Ellipse((1, 2), hradius=3, vradius=2))
    do_test(
        Plane((0, 0, 0), (1, 1, 1)), (x, -5, 5), (y, -4, 4), (z, -3, 3),
        s=PlaneSeries
    )

    # Interactive series. Note that GeometryInteractiveSeries is an instance of
    # GeometrySeries
    do_test(Point2D(x, y), params={x: 1, y: 2})
    do_test(
        Plane((x, y, z), (1, 1, 1)),
        (x, -5, 5),
        (y, -4, 4),
        (z, -3, 3),
        params={x: 1, y: 2, z: 3},
        s=PlaneInteractiveSeries,
    )

    # not enough ranges for PlaneSeries: this series require all three ranges
    # for reliably computing data
    raises(TypeError, lambda: get_plot_data(Plane((0, 0, 0), (1, 1, 1))))
    raises(TypeError, lambda: get_plot_data(
            Plane((0, 0, 0), (1, 1, 1)), (x, -5, 5)))
    raises(TypeError, lambda: get_plot_data(
            Plane((0, 0, 0), (1, 1, 1)), (x, -5, 5), (y, -5, 5)))
    raises(TypeError, lambda: get_plot_data(
            Plane((0, 0, 0), (1, 1, 1)), None, (y, -5, 5)))
    raises(TypeError, lambda: get_plot_data(
            Plane((0, 0, 0), (1, 1, 1)), (x, -5, 5), None, (y, -5, 5)))
    raises(TypeError, lambda: get_plot_data(
            Plane((0, 0, 0), (1, 1, 1)), None, (x, -5, 5), (y, -5, 5)))


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
    s = _build_series(
        l2, (x, -10, 10), (y, -5, 5), (z, -8, 8), slice=Plane((-2, 0, 0), (1, 0, 0))
    )
    assert isinstance(s, SliceVector3DSeries)
