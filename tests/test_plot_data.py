from sympy import (
    symbols, cos, sin, Eq, I, Abs,
    exp, pi, gamma, sqrt
)
from sympy.geometry import (
    Plane, Polygon, Circle, Ellipse, Line, Segment,
    Ray, Line3D, Point2D, Point3D,
    Segment3D, Ray3D,
)
from sympy.vector import CoordSys3D
from pytest import raises
from spb.plot_data import _build_series
from spb.series import (
    LineOver1DRangeSeries,
    Parametric2DLineSeries,
    Parametric3DLineSeries,
    ParametricSurfaceSeries,
    SurfaceOver2DRangeSeries,
    InteractiveSeries,
    ImplicitSeries,
    Vector2DSeries,
    Vector3DSeries,
    ComplexSeries,
    ComplexInteractiveSeries,
    SliceVector3DSeries,
    GeometrySeries,
    PlaneSeries,
    PlaneInteractiveSeries,
)
from pytest import raises
import numpy as np


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

    s = _build_series(Eq(x ** 2 + y ** 2, 5), (x, -5, 5), (y, -2, 2))
    assert isinstance(s, ImplicitSeries)

    s = _build_series(Eq(x ** 2 + y ** 2, 5) & (x > y), (x, -5, 5), (y, -2, 2))
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

    s = _build_series(cos(x + y), sin(x + y), x, (x, -5, 5), (y, -3, 3), pt="p3ds")
    assert isinstance(s, ParametricSurfaceSeries)

    # missing ranges
    s = _build_series(cos(x + y), sin(x + y), x, pt="p3ds")
    assert isinstance(s, ParametricSurfaceSeries)

    s = _build_series(u * cos(x), (x, -5, 5), params={u: 1}, pt="pinter")
    assert isinstance(s, InteractiveSeries)

    s = _build_series(
        u * sqrt(x), (x, -5, 5), params={u: 1}, pt="pinter", is_complex=True
    )
    assert isinstance(s, ComplexInteractiveSeries)


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
        Plane((0, 0, 0), (1, 1, 1)), (x, -5, 5), (y, -4, 4), (z, -3, 3), s=PlaneSeries
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


def test_complex():
    x, y, z = symbols("x:z")
    e1 = 1 + exp(-Abs(x)) * sin(I * sin(5 * x))

    def do_test_1(s, n, cls):
        assert isinstance(s, cls)
        data = s.get_data()
        assert len(data) == n
        return data

    def test_equal_results(data1, data2):
        for i, (d1, d2) in enumerate(zip(data1, data2)):
            # print("i = {}".format(i))
            assert np.allclose(d1, d2)

    ### Complex line plots: use adaptive=False in order to compare results.

    # NOTE: whenever we'd like to get a series related to complex numbers, it
    # doesn't matter the keyword argument pt. The internal algorithm is only
    # going to check that any of the following keyword algorithm are set:
    # real, imag, abs, arg, absarg.

    # raise ValueError when multiple flags are set to True
    raises(ValueError, lambda:  _build_series(e1, (x, -5, 5),
            adaptive=False, real=True, imag=True,
            absarg=False, n=10, pt="p"))

    # return x, mag(e1), arg(e1)
    s1 = _build_series(e1, (x, -5, 5), adaptive=False, real=False, imag=False,
            absarg=True, n=10, modules=None)
    data1 = do_test_1(s1, 3, LineOver1DRangeSeries)
    s2 = _build_series(e1, (x, -5, 5), adaptive=False, real=False, imag=False,
            absarg=True, n=10, pt="p", modules=None)
    data2 = do_test_1(s2, 3, LineOver1DRangeSeries)
    test_equal_results(data1, data2)

    # return x, real(e1)
    s1 = _build_series(e1, (x, -5, 5), adaptive=False, n=10,
            real=True, imag=False, modules=None)
    data1 = do_test_1(s1, 2, LineOver1DRangeSeries)
    s2 = _build_series(e1, (x, -5, 5), adaptive=False, n=10,
            real=True, imag=False, pt="p", modules=None)
    data2 = do_test_1(s2, 2, LineOver1DRangeSeries)
    test_equal_results(data1, data2)

    # return x, imag(e1)
    s1 = _build_series(e1, (x, -5, 5), adaptive=False, n=10,
            real=False, imag=True, modules=None)
    data1 = do_test_1(s1, 2, LineOver1DRangeSeries)
    s2 = _build_series(e1, (x, -5, 5), adaptive=False, n=10,
            real=False, imag=True, pt="p", modules=None)
    data2 = do_test_1(s2, 2, LineOver1DRangeSeries)
    test_equal_results(data1, data2)

    # return x, abs(e1)
    s1 = _build_series(e1, (x, -5, 5), adaptive=False, n=10,
            real=False, imag=False, abs=True, modules=None)
    data1 = do_test_1(s1, 2, LineOver1DRangeSeries)
    s2 = _build_series(e1, (x, -5, 5), adaptive=False, n=10,
            real=False, imag=False, abs=True, pt="p", modules=None)
    data2 = do_test_1(s2, 2, LineOver1DRangeSeries)
    test_equal_results(data1, data2)

    # return x, arg(e1)
    s1 = _build_series(e1, (x, -5, 5), adaptive=False, n=10,
            real=False, imag=False, arg=True, modules=None)
    data1 = do_test_1(s1, 2, LineOver1DRangeSeries)
    s2 = _build_series(e1, (x, -5, 5), adaptive=False, n=10,
            real=False, imag=False, arg=True, pt="p", modules=None)
    data2 = do_test_1(s2, 2, LineOver1DRangeSeries)
    test_equal_results(data1, data2)


    ### Lists of complex numbers: raise an error
    e2 = z * exp(2 * pi * I * z)
    l2 = [e2.subs(z, t / 20) for t in range(20)]
    raises(ValueError, lambda: _build_series(l2))

    ### Domain coloring: returns x, y, (mag, arg), ...
    s1 = _build_series(gamma(z), (z, -3 - 3 * I, 3 + 3 * I),
            n1=10, n2=10, modules=None)
    data1 = do_test_1(s1, 5, ComplexSeries)
    s2 = _build_series(gamma(z), (z, -3 - 3 * I, 3 + 3 * I),
            n1=10, n2=10, pt="c", modules=None)
    data2 = do_test_1(s2, 5, ComplexSeries)
    test_equal_results(data1, data2)
    _, _, mag_arg, _, _ = data1
    mag1, arg1 = mag_arg[:, :, 0], mag_arg[:, :, 1]

    ### 3D complex function: real part
    s1 = _build_series(gamma(z), (z, -3 - 3 * I, 3 + 3 * I), n1=10, n2=10,
            threed=True, real=True, imag=False, modules=None)
    data1 = do_test_1(s1, 3, ComplexSeries)
    s2 = _build_series(gamma(z), (z, -3 - 3 * I, 3 + 3 * I), n1=10, n2=10,
            threed=True, real=True, imag=False, pt="p3d", modules=None)
    data2 = do_test_1(s2, 3, ComplexSeries)
    test_equal_results(data1, data2)

    ### 3D complex function:  imaginary part
    s1 = _build_series(gamma(z), (z, -3 - 3 * I, 3 + 3 * I), n1=10, n2=10,
            threed=True, real=False, imag=True, modules=None)
    data1 = do_test_1(s1, 3, ComplexSeries)
    s2 = _build_series(gamma(z), (z, -3 - 3 * I, 3 + 3 * I), n1=10, n2=10,
            threed=True, real=False, imag=True, pt="p3d", modules=None)
    data2 = do_test_1(s2, 3, ComplexSeries)
    test_equal_results(data1, data2)

    ### 3D complex function:  absolute value
    s1 = _build_series(gamma(z), (z, -3 - 3 * I, 3 + 3 * I), n1=10, n2=10,
            threed=True, real=False, imag=False, abs=True, modules=None)
    data1 = do_test_1(s1, 3, ComplexSeries)
    s2 = _build_series(gamma(z), (z, -3 - 3 * I, 3 + 3 * I), n1=10, n2=10,
            threed=True, real=False, imag=False, abs=True, pt="p3d",
            modules=None)
    data2 = do_test_1(s2, 3, ComplexSeries)
    test_equal_results(data1, data2)
    _, _, mag2 = data1

    ### 3D complex function:  argument
    s1 = _build_series(gamma(z), (z, -3 - 3 * I, 3 + 3 * I), n1=10, n2=10,
            threed=True, real=False, imag=False, abs=False, arg=True,
            modules=None)
    data1 = do_test_1(s1, 3, ComplexSeries)
    s2 = _build_series(gamma(z), (z, -3 - 3 * I, 3 + 3 * I), n1=10, n2=10,
            threed=True, real=False, imag=False, abs=False, arg=True, pt="p3d",
            modules=None)
    data2 = do_test_1(s2, 3, ComplexSeries)
    test_equal_results(data1, data2)
    _, _, arg2 = data1

    test_equal_results([mag1, arg1], [mag2, arg2])

    ### 3D functions of 2 variables: real part
    s1 = _build_series(sqrt(x), (x, -5, 5), (y, -5, 5), n1=10, n2=10,
            threed=True, real=True, imag=False, abs=False, arg=False,
            modules=None)
    data1 = do_test_1(s1, 3, SurfaceOver2DRangeSeries)
    s2 = _build_series(sqrt(x), (x, -5, 5), (y, -5, 5), n1=10, n2=10,
            threed=True, real=True, imag=False, abs=False, arg=False, pt="p3d",
            modules=None)
    data2 = do_test_1(s2, 3, SurfaceOver2DRangeSeries)
    test_equal_results(data1, data2)

    ### 3D functions of 2 variables: imaginary part
    s1 = _build_series(sqrt(x), (x, -5, 5), (y, -5, 5), n1=10, n2=10,
            threed=True, real=False, imag=True, abs=False, arg=False,
            modules=None)
    data1 = do_test_1(s1, 3, SurfaceOver2DRangeSeries)
    s2 = _build_series(sqrt(x), (x, -5, 5), (y, -5, 5), n1=10, n2=10,
            threed=True, real=False, imag=True, abs=False, arg=False, pt="p3d",
            modules=None)
    data2 = do_test_1(s2, 3, SurfaceOver2DRangeSeries)
    test_equal_results(data1, data2)

    ### 3D functions of 2 variables: absolute value
    s1 = _build_series(sqrt(x), (x, -5, 5), (y, -5, 5), n1=10, n2=10,
            threed=True, real=False, imag=False, abs=True, arg=False,
            modules=None)
    data1 = do_test_1(s1, 3, SurfaceOver2DRangeSeries)
    s2 = _build_series(sqrt(x), (x, -5, 5), (y, -5, 5), n1=10, n2=10,
            threed=True, real=False, imag=False, abs=True, arg=False, pt="p3d",
            modules=None)
    data2 = do_test_1(s2, 3, SurfaceOver2DRangeSeries)
    test_equal_results(data1, data2)

    ### 3D functions of 2 variables: argument
    s1 = _build_series(sqrt(x), (x, -5, 5), (y, -5, 5), n1=10, n2=10,
            threed=True, real=False, imag=False, abs=False, arg=True,
            modules=None)
    data1 = do_test_1(s1, 3, SurfaceOver2DRangeSeries)
    s2 = _build_series(sqrt(x), (x, -5, 5), (y, -5, 5), n1=10, n2=10,
            threed=True, real=False, imag=False, abs=False, arg=True, pt="p3d",
            modules=None)
    data2 = do_test_1(s2, 3, SurfaceOver2DRangeSeries)
    test_equal_results(data1, data2)
