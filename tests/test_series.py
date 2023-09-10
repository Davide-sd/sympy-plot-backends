import pytest
from pytest import raises, warns
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    SurfaceOver2DRangeSeries, ContourSeries, ParametricSurfaceSeries,
    ImplicitSeries, Implicit3DSeries, RiemannSphereSeries,
    Vector2DSeries, Vector3DSeries, SliceVector3DSeries,
    ComplexSurfaceSeries, ComplexDomainColoringSeries,
    ComplexSurfaceBaseSeries, ComplexPointSeries, GeometrySeries,
    PlaneSeries, List2DSeries, List3DSeries, AbsArgLineSeries,
    _set_discretization_points, ColoredLineOver1DRangeSeries,
    HVLineSeries
)
from spb import plot3d_spherical
from sympy import (
    latex, exp, symbols, Tuple, I, pi, sin, cos, tan, log, sqrt,
    re, im, arg, frac, Plane, Circle, Point, Sum, S, Abs, lambdify,
    Function, dsolve, Eq, Ynm, floor
)
from sympy.vector import CoordSys3D, gradient
import numpy as np

# NOTE:
#
# These tests are meant to verify that the data series generates the expected
# numerical data.
#
# If your issue is related to the processing and generation of *Series
# objects, consider adding tests to test_functions.py.
# If your issue il related to the preprocessing and generation of a
# Vector series or a Complex Series, consider adding tests to
# test_build_series.
# If your issue is related to a particular keyword affecting a backend
# behaviour, consider adding tests to test_backends.py
#


def test_adaptive():
    # verify that adaptive-related keywords produces the expected results

    from adaptive.learner.learner1D import curvature_loss_function
    x, y = symbols("x, y")

    # use default adaptive options: adaptive_goal=0.01, loss_fn=None
    s1 = LineOver1DRangeSeries(sin(x), (x, -10, 10), "", adaptive=True)
    x1, _ = s1.get_data()
    # use a different goal: set a number
    s2 = LineOver1DRangeSeries(sin(x), (x, -10, 10), "", adaptive=True,
        adaptive_goal=0.001)
    x2, _ = s2.get_data()
    # use a different goal: set a function
    s3 = LineOver1DRangeSeries(sin(x), (x, -10, 10), "", adaptive=True,
        adaptive_goal=lambda l: l.npoints >= 100)
    x3, _ = s3.get_data()
    # use a different loss function
    s4 = LineOver1DRangeSeries(sin(x), (x, -10, 10), "", adaptive=True,
        adaptive_goal=0.01, loss_fn=curvature_loss_function())
    x4, _ = s4.get_data()
    assert len(x1) < len(x2)
    assert len(x3) >= 100
    # using the same adaptive_goal value, curvature_loss_function produces
    # less points than default_loss
    assert len(x1) > len(x4)

    s1 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=True)
    x1, _, _, = s1.get_data()
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=True, adaptive_goal=0.001)
    x2, _, _ = s2.get_data()
    s3 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=True, adaptive_goal=0.01, loss_fn=curvature_loss_function())
    x3, _, _ = s3.get_data()
    assert len(x1) < len(x2)
    assert len(x1) > len(x3)

    s1 = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2*pi),
        adaptive=True)
    x1, _, _, _ = s1.get_data()
    s2 = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2*pi),
        adaptive=True, adaptive_goal=0.001)
    x2, _, _, _ = s2.get_data()
    s3 = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2*pi),
        adaptive=True, adaptive_goal=0.01, loss_fn=curvature_loss_function())
    x3, _, _, _ = s3.get_data()
    assert len(x1) < len(x2)
    assert len(x1) > len(x3)

    # the more refined the goal, the greater the number of points
    s1 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        adaptive=True, adaptive_goal=0.5)
    x1, _, _ = s1.get_data()
    n1 = x1.shape[0] * x1.shape[1]
    s2 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        adaptive=True, adaptive_goal=0.1)
    x2, _, _ = s2.get_data()
    n2 = x2.shape[0] * x2.shape[1]
    assert n1 < n2


def test_adaptive_zerodivisionerror():
    x, y = symbols("x, y")

    # adaptive should be able to catch ZeroDivisionError
    s1 = LineOver1DRangeSeries(1 / x, (x, -10, 10), "", adaptive=True)
    x1, y1 = s1.get_data()

    s1 = Parametric2DLineSeries(cos(x), 1 / x, (x, -10, 10), "",
        adaptive=True)
    x1, y1, p1 = s1.get_data()

    s1 = Parametric3DLineSeries(cos(x), x, 1 / x, (x, -10, 10), "",
        adaptive=True)
    x1, y1, z1, p1 = s1.get_data()


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_detect_poles():
    x, u = symbols("x, u")

    s1 = LineOver1DRangeSeries(tan(x), (x, -pi, pi),
        adaptive=False, n=1000, detect_poles=False)
    xx1, yy1 = s1.get_data()
    s2 = LineOver1DRangeSeries(tan(x), (x, -pi, pi),
        adaptive=False, n=1000, detect_poles=True, eps=0.01)
    xx2, yy2 = s2.get_data()
    # eps is too small: doesn't detect any poles
    s3 = LineOver1DRangeSeries(tan(x), (x, -pi, pi),
        adaptive=False, n=1000, detect_poles=True, eps=1e-06)
    xx3, yy3 = s3.get_data()

    assert np.allclose(xx1, xx2) and np.allclose(xx1, xx3)
    assert not np.any(np.isnan(yy1))
    assert not np.any(np.isnan(yy3))
    assert np.any(np.isnan(yy2))

    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some of",
        ):
        s1 = LineOver1DRangeSeries(frac(x), (x, -10, 10),
            adaptive=False, n=1000, detect_poles=False)
        xx1, yy1 = s1.get_data()
        s2 = LineOver1DRangeSeries(frac(x), (x, -10, 10),
            adaptive=False, n=1000, detect_poles=True, eps=0.05)
        s3 = LineOver1DRangeSeries(frac(x), (x, -10, 10),
            adaptive=False, n=1000, detect_poles="symbolic")
        xx1, yy1 = s1.get_data()
        xx2, yy2 = s2.get_data()
        xx3, yy3 = s3.get_data()
        assert np.allclose(xx1, xx2) and np.allclose(xx1, xx3)
        assert not np.any(np.isnan(yy1))
        assert np.any(np.isnan(yy2)) and np.any(np.isnan(yy2))
        assert not np.allclose(yy1, yy2, equal_nan=True)
        assert len(s3.poles_locations) == 0

    s1 = LineOver1DRangeSeries(tan(u * x), (x, -pi, pi), params={u: 1},
        adaptive=False, n=1000, detect_poles=False)
    xx1, yy1 = s1.get_data()
    s2 = LineOver1DRangeSeries(tan(u * x), (x, -pi, pi), params={u: 1},
        adaptive=False, n=1000, detect_poles=True, eps=0.01)
    xx2, yy2 = s2.get_data()
    # eps is too small: doesn't detect any poles
    s3 = LineOver1DRangeSeries(tan(u * x), (x, -pi, pi), params={u: 1},
        adaptive=False, n=1000, detect_poles=True, eps=1e-06)
    xx3, yy3 = s3.get_data()

    assert np.allclose(xx1, xx2) and np.allclose(xx1, xx3)
    assert not np.any(np.isnan(yy1))
    assert not np.any(np.isnan(yy3))
    assert np.any(np.isnan(yy2))

    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some of",
        ):
        u, v = symbols("u, v", real=True)
        n = S(1) / 3
        f = (u + I * v)**n
        r, i = re(f), im(f)
        s1 = Parametric2DLineSeries(r.subs(u, -2), i.subs(u, -2), (v, -2, 2),
            adaptive=False, n=1000, detect_poles=False)
        s2 = Parametric2DLineSeries(r.subs(u, -2), i.subs(u, -2), (v, -2, 2),
            adaptive=False, n=1000, detect_poles=True)
        xx1, yy1, pp1 = s1.get_data()
        assert not np.isnan(yy1).any()
        xx2, yy2, pp2 = s2.get_data()
        assert np.isnan(yy2).any()

    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some of",
        ):
        f = (x * u + x * I * v)**n
        r, i = re(f), im(f)
        s1 = Parametric2DLineSeries(r.subs(u, -2), i.subs(u, -2),
            (v, -2, 2), params={x: 1},
            adaptive=False, n1=1000, detect_poles=False)
        xx1, yy1, pp1 = s1.get_data()
        assert not np.isnan(yy1).any()
        s2 = Parametric2DLineSeries(r.subs(u, -2), i.subs(u, -2),
            (v, -2, 2), params={x: 1},
            adaptive=False, n1=1000, detect_poles=True)
        xx2, yy2, pp2 = s2.get_data()
        assert np.isnan(yy2).any()


def test_number_discretization_points():
    # verify that the different ways to set the number of discretization
    # points are consistent with each other.
    x, y, z = symbols("x:z")

    for pt in [LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries]:
        kw1 = _set_discretization_points({"n": 10}, pt)
        kw2 = _set_discretization_points({"n": [10, 20, 30]}, pt)
        kw3 = _set_discretization_points({"n1": 10}, pt)
        assert all(("n1" in kw) and kw["n1"] == 10 for kw in [kw1, kw2, kw3])

    for pt in [SurfaceOver2DRangeSeries, ContourSeries, ParametricSurfaceSeries,
        ComplexSurfaceSeries, ComplexDomainColoringSeries, Vector2DSeries, ImplicitSeries]:
        kw1 = _set_discretization_points({"n": 10}, pt)
        kw2 = _set_discretization_points({"n": [10, 20, 30]}, pt)
        kw3 = _set_discretization_points({"n1": 10, "n2": 20}, pt)
        assert kw1["n1"] == kw1["n2"] == 10
        assert all((kw["n1"] == 10) and (kw["n2"] == 20) for kw in [kw2, kw3])

    for pt in [Vector3DSeries, SliceVector3DSeries, Implicit3DSeries]:
        kw1 = _set_discretization_points({"n": 10}, pt)
        kw2 = _set_discretization_points({"n": [10, 20, 30]}, pt)
        kw3 = _set_discretization_points({"n1": 10, "n2": 20, "n3": 30}, pt)
        assert kw1["n1"] == kw1["n2"] == kw1["n3"] == 10
        assert all(((kw["n1"] == 10) and (kw["n2"] == 20)
            and (kw["n3"] == 30)) for kw in [kw2, kw3])

    # verify that line-related series can deal with large float number of
    # discretization points
    LineOver1DRangeSeries(cos(x), (x, -5, 5), adaptive=False, n=1e04).get_data()
    AbsArgLineSeries(sqrt(x), (x, -5, 5), adaptive=False, n=1e04).get_data()


def test_list2dseries():
    x = symbols("x")
    xx = np.linspace(-3, 3, 10)
    yy1 = np.cos(xx)
    yy2 = np.linspace(-3, 3, 20)

    # same number of elements: everything is fine
    s = List2DSeries(xx, yy1)
    assert not s.is_parametric
    # different number of elements: error
    raises(ValueError, lambda: List2DSeries(xx, yy2))

    # no color func: returns only x, y components and s in not parametric
    s = List2DSeries(xx, yy1)
    xxs, yys = s.get_data()
    assert np.allclose(xx, xxs)
    assert np.allclose(yy1, yys)
    assert not s.is_parametric


def test_list3dseries():
    x = symbols("x")
    zz1 = np.linspace(-3, 3, 10)
    xx = np.cos(zz1)
    yy = np.sin(zz1)
    zz2 = np.linspace(-3, 3, 20)

    # same number of elements: everything is fine
    s = List3DSeries(xx, yy, zz1)
    assert not s.is_parametric
    # different number of elements: error
    raises(ValueError, lambda: List3DSeries(xx, yy, zz2))

    # no color func: returns only x, y components and s in not parametric
    s = List3DSeries(xx, yy, zz1)
    xxs, yys, zzs = s.get_data()
    assert np.allclose(xx, xxs)
    assert np.allclose(yy, yys)
    assert np.allclose(zz1, zzs)
    assert not s.is_parametric


def test_complexpointseries():
    # verify that ComplexPointSeries returns the correct data depending on
    # the provided keyword arguments

    x = symbols("x")

    s = ComplexPointSeries([1 + 2 * I, 3 + 4 * I])
    xx, yy = s.get_data()
    assert s.is_point and (not s.is_parametric) and (not s.color_func)
    assert np.allclose(xx, [1, 3])
    assert np.allclose(yy, [2, 4])

    s = ComplexPointSeries([1 + x * I, 1 + x + 4 * I], params={x: 2})
    xx, yy = s.get_data()
    assert s.is_point and (not s.is_parametric) and (not s.color_func)
    assert np.allclose(xx, [1, 3])
    assert np.allclose(yy, [2, 4])


def test_interactive_vs_noninteractive():
    # verify that if a *Series class receives a `params` dictionary, it sets
    # is_interactive=True
    x, y, z, u, v = symbols("x, y, z, u, v")

    s = LineOver1DRangeSeries(cos(x), (x, -5, 5))
    assert not s.is_interactive
    s = LineOver1DRangeSeries(u * cos(x), (x, -5, 5), params={u: 1})
    assert s.is_interactive

    s = AbsArgLineSeries(cos(x), (x, -5, 5))
    assert not s.is_interactive
    s = AbsArgLineSeries(u * cos(x), (x, -5, 5), params={u: 1})
    assert s.is_interactive

    s = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5))
    assert not s.is_interactive
    s = Parametric2DLineSeries(u * cos(x), u * sin(x), (x, -5, 5),
        params={u: 1})
    assert s.is_interactive

    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5))
    assert not s.is_interactive
    s = Parametric3DLineSeries(u * cos(x), u * sin(x), x, (x, -5, 5),
        params={u: 1})
    assert s.is_interactive

    s = SurfaceOver2DRangeSeries(cos(x * y), (x, -5, 5), (y, -5, 5))
    assert not s.is_interactive
    s = SurfaceOver2DRangeSeries(u * cos(x * y), (x, -5, 5), (y, -5, 5),
        params={u: 1})
    assert s.is_interactive

    s = ContourSeries(cos(x * y), (x, -5, 5), (y, -5, 5))
    assert not s.is_interactive
    s = ContourSeries(u * cos(x * y), (x, -5, 5), (y, -5, 5),
        params={u: 1})
    assert s.is_interactive

    s = ParametricSurfaceSeries(u * cos(v), v * sin(u), u + v,
        (u, -5, 5), (v, -5, 5))
    assert not s.is_interactive
    s = ParametricSurfaceSeries(u * cos(v * x), v * sin(u), u + v,
        (u, -5, 5), (v, -5, 5), params={x: 1})
    assert s.is_interactive

    s = Vector2DSeries(
        cos(y), sin(x), (x, -5, 5), (y, -5, 5))
    assert not s.is_interactive
    s = Vector2DSeries(
        u * cos(y), u * sin(x), (x, -5, 5), (y, -5, 5), params={u: 1})
    assert s.is_interactive

    s = Vector3DSeries(cos(y), sin(x), z,
        (x, -5, 5), (y, -5, 5), (z, -5, 5), slice=None)
    assert not s.is_interactive
    s = Vector3DSeries(u * cos(y), u * sin(x), z,
        (x, -5, 5), (y, -5, 5), (z, -5, 5), params={u: 1}, slice=None)
    assert s.is_interactive

    s = SliceVector3DSeries(Plane((0, 0, 0), (0, 1, 0)), cos(y), sin(x), z,
        (x, -5, 5), (y, -5, 5), (z, -5, 5))
    assert not s.is_interactive
    s = SliceVector3DSeries(Plane((0, 0, 0), (0, 1, 0)),
        u * cos(y), u * sin(x), z,
        (x, -5, 5), (y, -5, 5), (z, -5, 5), params={u: 1})
    assert s.is_interactive

    s = PlaneSeries(Plane((x, y, 0), (0, 1, 0)),
        (x, -5, 5), (y, -5, 5), (z, -5, 5))
    assert not s.is_interactive
    s = PlaneSeries(Plane((u * x, y, 0), (0, 1, 0)),
        (x, -5, 5), (y, -5, 5), (z, -5, 5), params={u: 1})
    assert s.is_interactive

    s = GeometrySeries(Circle(Point(0, 0), 5))
    assert not s.is_interactive
    s = GeometrySeries(Circle(Point(0, 0), u * 5), params={u: 1})
    assert s.is_interactive

    s = ComplexSurfaceSeries(sqrt(z), (z, -5 - 5j, 5 + 5j))
    assert not s.is_interactive
    s = ComplexSurfaceSeries(sqrt(u * z), (z, -5 - 5j, 5 + 5j), params={u: 1})
    assert s.is_interactive

    s = ComplexDomainColoringSeries(sqrt(z), (z, -5 - 5j, 5 + 5j))
    assert not s.is_interactive
    s = ComplexDomainColoringSeries(sqrt(u * z), (z, -5 - 5j, 5 + 5j),
        params={u: 1})
    assert s.is_interactive

    s = ComplexPointSeries([3 + 2 * I, 4 * I, 2])
    assert not s.is_interactive
    s = ComplexPointSeries([3 + u * 2 * I, 4 * I, u * 2], params={u: 1})
    assert s.is_interactive


def test_lin_log_scale():
    # Verify that data series create the correct spacing in the data.
    x, y, z = symbols("x, y, z")

    s = LineOver1DRangeSeries(x, (x, 1, 10), adaptive=False, n=50,
        xscale="linear")
    xx, _ = s.get_data()
    assert np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])

    s = LineOver1DRangeSeries(x, (x, 1, 10), adaptive=False, n=50,
        xscale="log")
    xx, _ = s.get_data()
    assert not np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])

    s = Parametric2DLineSeries(
        cos(x), sin(x), (x, pi / 2, 1.5 * pi), adaptive=False, n=50,
        xscale="linear")
    _, _, param = s.get_data()
    assert np.isclose(param[1] - param[0], param[-1] - param[-2])

    s = Parametric2DLineSeries(
        cos(x), sin(x), (x, pi / 2, 1.5 * pi), adaptive=False, n=50,
        xscale="log")
    _, _, param = s.get_data()
    assert not np.isclose(param[1] - param[0], param[-1] - param[-2])

    s = Parametric3DLineSeries(
        cos(x), sin(x), x, (x, pi / 2, 1.5 * pi), adaptive=False, n=50,
        xscale="linear")
    _, _, _, param = s.get_data()
    assert np.isclose(param[1] - param[0], param[-1] - param[-2])

    s = Parametric3DLineSeries(
        cos(x), sin(x), x, (x, pi / 2, 1.5 * pi), adaptive=False, n=50,
        xscale="log")
    _, _, _, param = s.get_data()
    assert not np.isclose(param[1] - param[0], param[-1] - param[-2])

    s = SurfaceOver2DRangeSeries(
        cos(x ** 2 + y ** 2), (x, 1, 5), (y, 1, 5), n=10,
        xscale="linear", yscale="linear")
    xx, yy, _ = s.get_data()
    assert np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])
    assert np.isclose(yy[1, 0] - yy[0, 0], yy[-1, 0] - yy[-2, 0])

    s = SurfaceOver2DRangeSeries(
        cos(x ** 2 + y ** 2), (x, 1, 5), (y, 1, 5), n=10,
        xscale="log", yscale="log")
    xx, yy, _ = s.get_data()
    assert not np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])
    assert not np.isclose(yy[1, 0] - yy[0, 0], yy[-1, 0] - yy[-2, 0])

    s = ImplicitSeries(
        cos(x ** 2 + y ** 2) > 0, (x, 1, 5), (y, 1, 5),
        n1=10, n2=10, xscale="linear", yscale="linear", adaptive=False)
    xx, yy, _, _ = s.get_data()
    assert np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])
    assert np.isclose(yy[1, 0] - yy[0, 0], yy[-1, 0] - yy[-2, 0])

    s = ImplicitSeries(
        cos(x ** 2 + y ** 2) > 0, (x, 1, 5), (y, 1, 5),
        n=10, xscale="log", yscale="log", adaptive=False)
    xx, yy, _, _ = s.get_data()
    assert not np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])
    assert not np.isclose(yy[1, 0] - yy[0, 0], yy[-1, 0] - yy[-2, 0])

    s = AbsArgLineSeries(cos(x), (x, 1e-05, 1e05), n=10,
        xscale="linear", adaptive=False)
    xx, yy, _ = s.get_data()
    assert np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])

    s = AbsArgLineSeries(cos(x), (x, 1e-05, 1e05), n=10,
        xscale="log", adaptive=False)
    xx, yy, _ = s.get_data()
    assert not np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])

    s = Vector3DSeries(
        x, y, z, (x, 1, 1e05), (y, 1, 1e05), (z, 1, 1e05),
        xscale="linear", yscale="linear", zscale="linear")
    xx, yy, zz, _, _, _ = s.get_data()
    assert np.isclose(
        xx[:, 0, 0][1] - xx[:, 0, 0][0], xx[:, 0, 0][-1] - xx[:, 0, 0][-2])
    assert np.isclose(
        yy[0, :, 0][1] - yy[0, :, 0][0], yy[0, :, 0][-1] - yy[0, :, 0][-2])
    assert np.isclose(
        zz[0, 0, :][1] - zz[0, 0, :][0], zz[0, 0, :][-1] - zz[0, 0, :][-2])

    s = Vector3DSeries(
        x, y, z, (x, 1, 1e05), (y, 1, 1e05), (z, 1, 1e05),
        xscale="log", yscale="log", zscale="log")
    xx, yy, zz, _, _, _ = s.get_data()
    assert not np.isclose(
        xx[:, 0, 0][1] - xx[:, 0, 0][0], xx[:, 0, 0][-1] - xx[:, 0, 0][-2])
    assert not np.isclose(
        yy[0, :, 0][1] - yy[0, :, 0][0], yy[0, :, 0][-1] - yy[0, :, 0][-2])
    assert not np.isclose(
        zz[0, 0, :][1] - zz[0, 0, :][0], zz[0, 0, :][-1] - zz[0, 0, :][-2])


def test_rendering_kw():
    # verify that each series exposes the `rendering_kw` attribute
    u, v, x, y, z = symbols("u, v, x:z")

    s = List2DSeries([1, 2, 3], [4, 5, 6])
    assert isinstance(s.rendering_kw, dict)

    s = List3DSeries([1, 2, 3], [4, 5, 6], [6, 7, 8])
    assert isinstance(s.rendering_kw, dict)

    s = LineOver1DRangeSeries(1, (x, -5, 5))
    assert isinstance(s.rendering_kw, dict)

    s = AbsArgLineSeries(1, (x, -5, 5))
    assert isinstance(s.rendering_kw, dict)

    s = Parametric2DLineSeries(sin(x), cos(x), (x, 0, pi))
    assert isinstance(s.rendering_kw, dict)

    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2 * pi))
    assert isinstance(s.rendering_kw, dict)

    s = SurfaceOver2DRangeSeries(x + y, (x, -2, 2), (y, -3, 3))
    assert isinstance(s.rendering_kw, dict)

    s = Implicit3DSeries(
        x**2 + y**3 - z**2, (x, -2, 2), (y, -3, 3), (z, -4, 4))
    assert isinstance(s.rendering_kw, dict)

    s = ContourSeries(x + y, (x, -2, 2), (y, -3, 3))
    assert isinstance(s.rendering_kw, dict)

    s = ParametricSurfaceSeries(1, x, y, (x, 0, 1), (y, 0, 1))
    assert isinstance(s.rendering_kw, dict)

    s = ComplexPointSeries(x + I * y)
    assert isinstance(s.rendering_kw, dict)

    s = ComplexSurfaceSeries(1, (x, -5 - 2 * I, 5 + 2 * I))
    assert isinstance(s.rendering_kw, dict)

    s = ComplexDomainColoringSeries(1, (x, -5 - 2 * I, 5 + 2 * I))
    assert isinstance(s.rendering_kw, dict)

    s = Vector2DSeries(-y, x, (x, -3, 3), (y, -4, 4))
    assert isinstance(s.rendering_kw, dict)

    s = Vector3DSeries(z, -y, x, (x, -3, 3), (y, -4, 4), (z, -5, 5))
    assert isinstance(s.rendering_kw, dict)

    s = SliceVector3DSeries(
        Plane((-1, 0, 0), (1, 0, 0)),
        z, -y, x,
        (x, -3, 3), (y, -2, 2), (z, -1, 1))
    assert isinstance(s.rendering_kw, dict)

    s = PlaneSeries(Plane((0, 0, 0), (1, 1, 1)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    assert isinstance(s.rendering_kw, dict)

    s = GeometrySeries(Circle(Point(0, 0), 5))
    assert isinstance(s.rendering_kw, dict)


def test_use_quiver_solid_color():
    # verify that the attribute `use_quiver_solid_color` is present

    u, x, y = symbols("u, x, y")

    s = Vector2DSeries(-y, x, (x, -3, 3), (y, -4, 4))
    assert isinstance(s.rendering_kw, dict)
    assert hasattr(s, "use_quiver_solid_color")
    assert s.use_quiver_solid_color


def test_data_shape():
    # Verify that the series produces the correct data shape when the input
    # expression is a number.
    u, x, y, z = symbols("u, x:z")

    # scalar expression: it should return a numpy ones array
    s = LineOver1DRangeSeries(1, (x, -5, 5))
    xx, yy = s.get_data()
    assert len(xx) == len(yy)
    assert np.all(yy == 1)

    s = LineOver1DRangeSeries(1, (x, -5, 5), adaptive=False, n=10)
    xx, yy = s.get_data()
    assert len(xx) == len(yy) == 10
    assert np.all(yy == 1)

    s = AbsArgLineSeries(1, (x, -5, 5))
    xx, _abs, _arg = s.get_data()
    assert len(xx) == len(_abs) == len(_arg)
    assert np.all(_abs == 1)

    s = AbsArgLineSeries(1, (x, -5, 5), adaptive=False, n=10)
    xx, _abs, _arg = s.get_data()
    assert len(xx) == len(_abs) == len(_arg) == 10
    assert np.all(_abs == 1)

    s = Parametric2DLineSeries(sin(x), 1, (x, 0, pi))
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(yy == 1)

    s = Parametric2DLineSeries(1, sin(x), (x, 0, pi))
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(xx == 1)

    s = Parametric2DLineSeries(sin(x), 1, (x, 0, pi), adaptive=False)
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(yy == 1)

    s = Parametric2DLineSeries(1, sin(x), (x, 0, pi), adaptive=False)
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(xx == 1)

    s = Parametric3DLineSeries(cos(x), sin(x), 1, (x, 0, 2 * pi))
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(zz)) and (len(xx) == len(param))
    assert np.all(zz == 1)

    s = Parametric3DLineSeries(cos(x), 1, x, (x, 0, 2 * pi))
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(zz)) and (len(xx) == len(param))
    assert np.all(yy == 1)

    s = Parametric3DLineSeries(1, sin(x), x, (x, 0, 2 * pi))
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(zz)) and (len(xx) == len(param))
    assert np.all(xx == 1)

    s = SurfaceOver2DRangeSeries(1, (x, -2, 2), (y, -3, 3))
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(zz == 1)

    s = Implicit3DSeries(x**2 + y**3 - z**2,
        (x, -2, 2), (y, -3, 3), (z, -4, 4),
        n1=5, n2=8, n3=10)
    xx, yy, zz, f = s.get_data()
    assert f.shape == xx.shape == yy.shape == zz.shape == (5, 8, 10)

    s = ParametricSurfaceSeries(1, x, y, (x, 0, 1), (y, 0, 1))
    xx, yy, zz, uu, vv = s.get_data()
    assert xx.shape == yy.shape == zz.shape == uu.shape == vv.shape
    assert np.all(xx == 1)

    s = ParametricSurfaceSeries(1, 1, y, (x, 0, 1), (y, 0, 1))
    xx, yy, zz, uu, vv = s.get_data()
    assert xx.shape == yy.shape == zz.shape == uu.shape == vv.shape
    assert np.all(yy == 1)

    s = ParametricSurfaceSeries(x, 1, 1, (x, 0, 1), (y, 0, 1))
    xx, yy, zz, uu, vv = s.get_data()
    assert xx.shape == yy.shape == zz.shape == uu.shape == vv.shape
    assert np.all(zz == 1)

    s = AbsArgLineSeries(1, (x, -5, 5), modules=None)
    xx, yy, aa = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == aa.shape)
    assert np.all(aa == 0)

    s = AbsArgLineSeries(1, (x, -5, 5), modules="mpmath")
    xx, yy, aa = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == aa.shape)
    assert np.all(aa == 0)

    s = ComplexSurfaceSeries(1, (x, -5 - 2 * I, 5 + 2 * I),
        n1=10, n2=10, modules=None)
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(zz == 1)

    s = ComplexSurfaceSeries(1, (x, -5 - 2 * I, 5 + 2 * I),
        n1=10, n2=10, modules="mpmath")
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(zz == 1)

    s = ComplexDomainColoringSeries(1, (x, -5 - 2 * I, 5 + 2 * I),
        n1=10, n2=10, modules=None)
    rr, ii, mag, arg, colors, _ = s.get_data()
    assert (rr.shape == ii.shape) and (rr.shape[:2] == colors.shape[:2])
    assert (rr.shape == mag.shape) and (rr.shape == arg.shape)

    s = ComplexDomainColoringSeries(1, (x, -5 - 2 * I, 5 + 2 * I),
        n1=10, n2=10, modules="mpmath")
    rr, ii, mag, arg, colors, _ = s.get_data()
    assert (rr.shape == ii.shape) and (rr.shape[:2] == colors.shape[:2])
    assert (rr.shape == mag.shape) and (rr.shape == arg.shape)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_only_integers():
    x, y, u, v = symbols("x, y, u, v")

    s = LineOver1DRangeSeries(sin(x), (x, -5.5, 4.5), "",
        adaptive=False, only_integers=True)
    xx, _ = s.get_data()
    assert len(xx) == 10
    assert xx[0] == -5 and xx[-1] == 4

    s = AbsArgLineSeries(sqrt(x), (x, -5.5, 4.5), "",
        adaptive=False, only_integers=True)
    xx, _, _ = s.get_data()
    assert len(xx) == 10
    assert xx[0] == -5 and xx[-1] == 4

    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2 * pi), "",
        adaptive=False, only_integers=True)
    _, _, p = s.get_data()
    assert len(p) == 7
    assert p[0] == 0 and p[-1] == 6

    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2 * pi), "",
        adaptive=False, only_integers=True)
    _, _, _, p = s.get_data()
    assert len(p) == 7
    assert p[0] == 0 and p[-1] == 6

    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -5.5, 5.5),
        (y, -3.5, 3.5), "",
        adaptive=False, only_integers=True)
    xx, yy, _ = s.get_data()
    assert xx.shape == yy.shape == (7, 11)
    assert np.allclose(xx[:, 0] - (-5) * np.ones(7), 0)
    assert np.allclose(xx[0, :] - np.linspace(-5, 5, 11), 0)
    assert np.allclose(yy[:, 0] - np.linspace(-3, 3, 7), 0)
    assert np.allclose(yy[0, :] - (-3) * np.ones(11), 0)

    r = 2 + sin(7 * u + 5 * v)
    expr = (
        r * cos(u) * sin(v),
        r * sin(u) * sin(v),
        r * cos(v)
    )
    s = ParametricSurfaceSeries(*expr, (u, 0, 2 * pi), (v, 0, pi), "",
        adaptive=False, only_integers=True)
    xx, yy, zz, uu, vv = s.get_data()
    assert xx.shape == yy.shape == zz.shape == uu.shape == vv.shape == (4, 7)

    s = ComplexSurfaceSeries(sqrt(x), (x, -3.5 - 2.5j, 3.5 + 2.5j), "",
        adaptive=False, only_integers=True)
    xx, yy, zz = s.get_data()
    assert xx.shape == yy.shape == zz.shape == (5, 7)
    assert xx[0, 0] == -3 and xx[-1, -1] == 3
    assert yy[0, 0] == -2 and yy[-1, -1] == 2

    s = ComplexDomainColoringSeries(sqrt(x), (x, -3.5 - 2.5j, 3.5 + 2.5j), "",
        adaptive=False, only_integers=True)
    xx, yy, zz, aa, _, _ = s.get_data()
    assert xx.shape == yy.shape == zz.shape == aa.shape == (5, 7)
    assert xx[0, 0] == -3 and xx[-1, -1] == 3
    assert yy[0, 0] == -2 and yy[-1, -1] == 2

    # only_integers also works with scalar expressions
    s = LineOver1DRangeSeries(1, (x, -5.5, 4.5), "",
        adaptive=False, only_integers=True)
    xx, _ = s.get_data()
    assert len(xx) == 10
    assert xx[0] == -5 and xx[-1] == 4

    s = Parametric2DLineSeries(cos(x), 1, (x, 0, 2 * pi), "",
        adaptive=False, only_integers=True)
    _, _, p = s.get_data()
    assert len(p) == 7
    assert p[0] == 0 and p[-1] == 6

    s = SurfaceOver2DRangeSeries(1, (x, -5.5, 5.5), (y, -3.5, 3.5), "",
        adaptive=False, only_integers=True)
    xx, yy, _ = s.get_data()
    assert xx.shape == yy.shape == (7, 11)
    assert np.allclose(xx[:, 0] - (-5) * np.ones(7), 0)
    assert np.allclose(xx[0, :] - np.linspace(-5, 5, 11), 0)
    assert np.allclose(yy[:, 0] - np.linspace(-3, 3, 7), 0)
    assert np.allclose(yy[0, :] - (-3) * np.ones(11), 0)

    r = 2 + sin(7 * u + 5 * v)
    expr = (
        r * cos(u) * sin(v),
        1,
        r * cos(v)
    )
    s = ParametricSurfaceSeries(*expr, (u, 0, 2 * pi), (v, 0, pi), "",
        adaptive=False, only_integers=True)
    xx, yy, zz, uu, vv = s.get_data()
    assert xx.shape == yy.shape == zz.shape == uu.shape == vv.shape == (4, 7)

    s = ComplexSurfaceSeries(1, (x, -3.5 - 2.5j, 3.5 + 2.5j), "",
        adaptive=False, only_integers=True)
    xx, yy, zz = s.get_data()
    assert xx.shape == yy.shape == zz.shape == (5, 7)
    assert xx[0, 0] == -3 and xx[-1, -1] == 3
    assert yy[0, 0] == -2 and yy[-1, -1] == 2

    s = ComplexDomainColoringSeries(1, (x, -3.5 - 2.5j, 3.5 + 2.5j), "",
        adaptive=False, only_integers=True)
    xx, yy, zz, aa, _, _ = s.get_data()
    assert xx.shape == yy.shape == zz.shape == aa.shape == (5, 7)
    assert xx[0, 0] == -3 and xx[-1, -1] == 3
    assert yy[0, 0] == -2 and yy[-1, -1] == 2

    s = Vector2DSeries(-y, x, (x, -3.5, 3.5), (y, -4.5, 4.5), "",
        only_integers=True)
    xx, yy, uu, vv = s.get_data()
    assert xx.shape == yy.shape == uu.shape == vv.shape == (9, 7)
    assert xx[0, 0] == -3 and xx[-1, -1] == 3
    assert yy[0, 0] == -4 and yy[-1, -1] == 4


def test_vector_data():
    # verify that vector data series generates data with the correct shape

    x, y, z = symbols("x:z")

    s = Vector2DSeries(x, y, (x, -5, 5), (y, -3, 3), "test", n1=10, n2=15)
    xx, yy, uu, vv = s.get_data()
    assert xx.shape == uu.shape == (15, 10)
    assert yy.shape == vv.shape == (15, 10)

    # at least one vector component is a scalar
    s = Vector2DSeries(1, y, (x, -5, 5), (y, -3, 3), "test", n1=10, n2=15)
    xx, yy, uu, vv = s.get_data()
    assert xx.shape == uu.shape == (15, 10)
    assert yy.shape == vv.shape == (15, 10)

    s = Vector3DSeries(
        x, y, z, (x, -5, 5), (y, -3, 3), (z, -2, 2), "test", n1=10, n2=15, n3=20
    )
    xx, yy, zz, uu, vv, ww = s.get_data()
    assert xx.shape == uu.shape == (10, 15, 20)
    assert yy.shape == vv.shape == (10, 15, 20)
    assert zz.shape == ww.shape == (10, 15, 20)

    # at least one vector component is a scalar
    s = Vector3DSeries(
        x, 1, z, (x, -5, 5), (y, -3, 3), (z, -2, 2), "test", n1=10, n2=15, n3=20
    )
    xx, yy, zz, uu, vv, ww = s.get_data()
    assert xx.shape == uu.shape == (10, 15, 20)
    assert yy.shape == vv.shape == (10, 15, 20)
    assert zz.shape == ww.shape == (10, 15, 20)


def test_is_point_is_filled():
    # verify that `is_point` and `is_filled` are attributes and that they
    # they receive the correct values

    x, u = symbols("x, u")

    s = LineOver1DRangeSeries(cos(x), (x, -5, 5), "",
        is_point=False, is_filled=True)
    assert (not s.is_point) and s.is_filled
    s = LineOver1DRangeSeries(cos(x), (x, -5, 5), "",
        is_point=True, is_filled=False)
    assert s.is_point and (not s.is_filled)

    s = AbsArgLineSeries(cos(x), (x, -5, 5), "",
        is_point=False, is_filled=True)
    assert (not s.is_point) and s.is_filled
    s = AbsArgLineSeries(cos(x), (x, -5, 5), "",
        is_point=True, is_filled=False)
    assert s.is_point and (not s.is_filled)

    s = List2DSeries([0, 1, 2], [3, 4, 5],
        is_point=False, is_filled=True)
    assert (not s.is_point) and s.is_filled
    s = List2DSeries([0, 1, 2], [3, 4, 5],
        is_point=True, is_filled=False)
    assert s.is_point and (not s.is_filled)

    s = List3DSeries([0, 1, 2], [3, 4, 5], [6, 7, 8],
        is_point=False, is_filled=True)
    assert (not s.is_point) and s.is_filled
    s = List3DSeries([0, 1, 2], [3, 4, 5], [6, 7, 8],
        is_point=True, is_filled=False)
    assert s.is_point and (not s.is_filled)

    s = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5),
        is_point=False, is_filled=True)
    assert (not s.is_point) and s.is_filled
    s = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5),
        is_point=True, is_filled=False)
    assert s.is_point and (not s.is_filled)

    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5),
        is_point=False, is_filled=True)
    assert (not s.is_point) and s.is_filled
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5),
        is_point=True, is_filled=False)
    assert s.is_point and (not s.is_filled)

    s = ComplexPointSeries([1 + 2 * I, 3 + 4 * I],
        is_point=False, is_filled=True)
    assert (not s.is_point) and s.is_filled
    s = ComplexPointSeries([1 + 2 * I, 3 + 4 * I],
        is_point=True, is_filled=False)
    assert s.is_point and (not s.is_filled)


def test_is_filled_2d():
    # verify that the is_filled attribute is exposed by the following series
    x, y = symbols("x, y")

    expr = cos(x**2 + y**2)
    ranges = (x, -2, 2), (y, -2, 2)

    s = ContourSeries(expr, *ranges)
    assert s.is_filled
    s = ContourSeries(expr, *ranges, is_filled=True)
    assert s.is_filled
    s = ContourSeries(expr, *ranges, is_filled=False)
    assert not s.is_filled

    s = GeometrySeries(Circle(Point(0, 0), 5))
    assert s.is_filled
    s = GeometrySeries(Circle(Point(0, 0), 5), is_filled=False)
    assert not s.is_filled
    s = GeometrySeries(Circle(Point(0, 0), 5), is_filled=True)
    assert s.is_filled

    # ComplexSurfaceSeries generates data for 3D surfaces or 2D contours
    s = ComplexSurfaceSeries(sqrt(x), (x, -2-2j, 2+2j))
    assert s.is_filled
    s = ComplexSurfaceSeries(sqrt(x), (x, -2-2j, 2+2j), is_filled=False)
    assert not s.is_filled
    s = ComplexSurfaceSeries(sqrt(x), (x, -2-2j, 2+2j), is_filled=True)
    assert s.is_filled


def test_steps():
    x, u = symbols("x, u")

    def do_test(s1, s2):
        if (not s1.is_parametric) and s1.is_2Dline:
            xx1, _ = s1.get_data()
            xx2, _ = s2.get_data()
        elif s1.is_parametric and s1.is_2Dline:
            xx1, _, _ = s1.get_data()
            xx2, _, _ = s2.get_data()
        elif (not s1.is_parametric) and s1.is_3Dline:
            xx1, _, _ = s1.get_data()
            xx2, _, _ = s2.get_data()
        else:
            xx1, _, _, _ = s1.get_data()
            xx2, _, _, _ = s2.get_data()
        assert len(xx1) != len(xx2)

    s1 = LineOver1DRangeSeries(cos(x), (x, -5, 5), "",
        adaptive=False, n=40, steps=False)
    s2 = LineOver1DRangeSeries(cos(x), (x, -5, 5), "",
        adaptive=False, n=40, steps=True)
    do_test(s1, s2)

    s1 = AbsArgLineSeries(cos(x), (x, -5, 5), "",
        adaptive=False, n=40, steps=False)
    s2 = AbsArgLineSeries(cos(x), (x, -5, 5), "",
        adaptive=False, n=40, steps=True)
    do_test(s1, s2)

    s1 = List2DSeries([0, 1, 2], [3, 4, 5],
        adaptive=False, n=40, steps=False)
    s2 = List2DSeries([0, 1, 2], [3, 4, 5],
        adaptive=False, n=40, steps=True)
    do_test(s1, s2)

    s1 = List3DSeries([0, 1, 2], [3, 4, 5], [6, 7, 8],
        adaptive=False, n=40, steps=False)
    s2 = List3DSeries([0, 1, 2], [3, 4, 5], [6, 7, 8],
        adaptive=False, n=40, steps=True)
    do_test(s1, s2)

    s1 = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5),
        adaptive=False, n=40, steps=False)
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5),
        adaptive=False, n=40, steps=True)
    do_test(s1, s2)

    s1 = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5),
        adaptive=False, n=40, steps=False)
    s2 = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5),
        adaptive=False, n=40, steps=True)
    do_test(s1, s2)

    s1 = ComplexPointSeries([1 + 2 * I, 3 + 4 * I],
        adaptive=False, n=40, steps=False)
    s2 = ComplexPointSeries([1 + 2 * I, 3 + 4 * I],
        adaptive=False, n=40, steps=True)
    do_test(s1, s2)


def test_interactive():
    u, x, y, z = symbols("u, x:z")

    # verify that InteractiveSeries produces the same numerical data as their
    # corresponding non-interactive series.
    def do_test(data1, data2):
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert np.allclose(d1, d2)

    s1 = LineOver1DRangeSeries(u * cos(x), (x, -5, 5), params={u: 1}, n=50)
    s2 = LineOver1DRangeSeries(cos(x), (x, -5, 5), adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())

    # complex function evaluated over a real line with numpy
    s1 = AbsArgLineSeries(u * cos(x), (x, -5, 5), params={u: 1}, n=50,
        modules=None)
    s2 = AbsArgLineSeries(cos(x), (x, -5, 5), adaptive=False, n=50,
        modules=None)
    do_test(s1.get_data(), s2.get_data())

    # complex function evaluated over a real line with mpmath
    s1 = AbsArgLineSeries(
        (z ** 2 + 1) / (z ** 2 - 1), (z, -3, 3), adaptive=False,
        n=11, modules="mpmath")
    s2 = AbsArgLineSeries(
        (z ** 2 + 1) / (z ** 2 - 1), (z, -3, 3), "", adaptive=False,
        n=11, modules="mpmath")
    do_test(s1.get_data(), s2.get_data())

    s1 = Parametric2DLineSeries(
        u * cos(x), u * sin(x), (x, -5, 5), params={u: 1}, n=50)
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5),
        adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())

    s1 = Parametric3DLineSeries(
        u * cos(x), u * sin(x), u * x, (x, -5, 5),
        params={u: 1}, n=50)
    s2 = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5),
        adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())

    s1 = SurfaceOver2DRangeSeries(
        u * cos(x ** 2 + y ** 2), (x, -3, 3), (y, -3, 3),
        params={u: 1}, n1=50, n2=50,)
    s2 = SurfaceOver2DRangeSeries(
        cos(x ** 2 + y ** 2), (x, -3, 3), (y, -3, 3),
        adaptive=False, n1=50, n2=50)
    do_test(s1.get_data(), s2.get_data())

    s1 = ParametricSurfaceSeries(
        u * cos(x + y), sin(x + y), x - y, (x, -3, 3), (y, -3, 3),
        params={u: 1}, n1=50, n2=50,)
    s2 = ParametricSurfaceSeries(
        cos(x + y), sin(x + y), x - y, (x, -3, 3), (y, -3, 3),
        adaptive=False, n1=50, n2=50,)
    do_test(s1.get_data(), s2.get_data())

    s1 = Vector2DSeries(
        -u * y, u * x, (x, -3, 3), (y, -2, 2),
        params={u: 1}, n1=15, n2=15)
    s2 = Vector2DSeries(-y, x, (x, -3, 3), (y, -2, 2), n1=15, n2=15)
    do_test(s1.get_data(), s2.get_data())

    s1 = Vector3DSeries(
        u * z, -u * y, u * x,
        (x, -3, 3), (y, -2, 2), (z, -1, 1),
        params={u: 1}, n1=15, n2=15, n3=15)
    s2 = Vector3DSeries(
        z, -y, x, (x, -3, 3), (y, -2, 2), (z, -1, 1), n1=15, n2=15, n3=15)
    do_test(s1.get_data(), s2.get_data())

    s1 = SliceVector3DSeries(
        Plane((-u, 0, 0), (1, 0, 0)), u * z, -u * y, u * x,
        (x, -3, 3), (y, -2, 2), (z, -1, 1),
        params={u: 1}, n1=15, n2=15, n3=15)
    s2 = SliceVector3DSeries(
        Plane((-1, 0, 0), (1, 0, 0)),
        z, -y, x,
        (x, -3, 3), (y, -2, 2), (z, -1, 1),
        n1=15, n2=15, n3=15)
    do_test(s1.get_data(), s2.get_data())

    # real part of a complex function evaluated over a real line with numpy
    expr = re((z ** 2 + 1) / (z ** 2 - 1))
    s1 = LineOver1DRangeSeries(u * expr, (z, -3, 3), adaptive=False, n=50,
        modules=None, params={u: 1})
    s2 = LineOver1DRangeSeries(expr, (z, -3, 3), adaptive=False, n=50,
        modules=None)
    do_test(s1.get_data(), s2.get_data())

    # real part of a complex function evaluated over a real line with mpmath
    expr = re((z ** 2 + 1) / (z ** 2 - 1))
    s1 = LineOver1DRangeSeries(u * expr, (z, -3, 3), n=50, modules="mpmath",
        params={u: 1})
    s2 = LineOver1DRangeSeries(expr, (z, -3, 3),
        adaptive=False, n=50, modules="mpmath")
    do_test(s1.get_data(), s2.get_data())

    # surface over a complex domain
    s1 = ComplexSurfaceSeries(
        u * (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I),
        n1=20, n2=20, params = {u: 1}, modules=None)
    s2 = ComplexSurfaceSeries(
        (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I),
        n1=20, n2=20, modules=None)
    do_test(s1.get_data(), s2.get_data())

    s1 = ComplexSurfaceSeries(
        u * (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I),
        n1=20, n2=20, params = {u: 1}, modules="mpmath")
    s2 = ComplexSurfaceSeries(
        (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I),
        n1=20, n2=20, modules="mpmath")
    do_test(s1.get_data(), s2.get_data())

    # domain coloring or 3D
    s1 = ComplexDomainColoringSeries(
        u * (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I),
        n1=20, n2=20, params = {u: 1}, modules=None)
    s2 = ComplexDomainColoringSeries(
        (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I),
        n1=20, n2=20, modules=None)
    do_test(s1.get_data(), s2.get_data())

    s1 = ComplexDomainColoringSeries(
        u * (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I),
        n1=20, n2=20, params = {u: 1}, modules="mpmath")
    s2 = ComplexDomainColoringSeries(
        (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I),
        n1=20, n2=20, modules="mpmath")
    do_test(s1.get_data(), s2.get_data())


def test_list2dseries_interactive():
    x, y, u = symbols("x, y, u")

    s = List2DSeries([1, 2, 3], [1, 2, 3])
    assert not s.is_interactive

    # symbolic expressions as coordinates, but no ``params``
    raises(ValueError, lambda: List2DSeries([cos(x)], [sin(x)]))

    # too few parameters
    raises(ValueError,
        lambda: List2DSeries([cos(x), y], [sin(x), 2], params={u: 1}))

    s = List2DSeries([cos(x)], [sin(x)], params={x: 1})
    assert s.is_interactive

    s = List2DSeries([x, 2, 3, 4], [4, 3, 2, x], params={x: 3})
    xx, yy = s.get_data()
    assert np.allclose(xx, [3, 2, 3, 4])
    assert np.allclose(yy, [4, 3, 2, 3])
    assert not s.is_parametric

    # numeric lists + params is present -> interactive series and
    # lists are converted to Tuple.
    s = List2DSeries([1, 2, 3], [1, 2, 3], params={x: 1})
    assert s.is_interactive
    assert isinstance(s.list_x, Tuple)
    assert isinstance(s.list_y, Tuple)


def test_list3dseries_interactive():
    x, y, u = symbols("x, y, u")

    s = List3DSeries([1, 2, 3], [1, 2, 3], [1, 2, 3])
    assert not s.is_interactive

    # symbolic expressions as coordinates, but no ``params``
    raises(ValueError, lambda: List3DSeries([cos(x)], [sin(x)], [x]))

    # too few parameters
    raises(ValueError,
        lambda: List3DSeries([cos(x), y], [sin(x), 2], [x, y], params={u: 1}))

    s = List3DSeries([cos(x)], [sin(x)], [x], params={x: 1})
    assert s.is_interactive

    s = List3DSeries([x, 2, 3, 4], [4, 3, 2, x], [1, 3, 4, x], params={x: 3})
    xx, yy, zz = s.get_data()
    assert s.is_interactive
    assert np.allclose(xx, [3, 2, 3, 4])
    assert np.allclose(yy, [4, 3, 2, 3])
    assert np.allclose(zz, [1, 3, 4, 3])
    assert not s.is_parametric


def test_mpmath():
    # test that the argument of complex functions evaluated with mpmath
    # might be different than the one computed with Numpy (different
    # behaviour at branch cuts)
    z, u = symbols("z, u")

    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some of",
        ):
        s1 = LineOver1DRangeSeries(im(sqrt(-z)), (z, 1e-03, 5),
            adaptive=True, modules=None)
        s2 = LineOver1DRangeSeries(im(sqrt(-z)), (z, 1e-03, 5),
            adaptive=True, modules="mpmath")
        xx1, yy1 = s1.get_data()
        xx2, yy2 = s2.get_data()
        assert np.all(yy1 < 0)
        assert np.all(yy2 > 0)

    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some of",
        ):
        s1 = LineOver1DRangeSeries(im(sqrt(-z)), (z, -5, 5),
            adaptive=False, n=20, modules=None)
        s2 = LineOver1DRangeSeries(im(sqrt(-z)), (z, -5, 5),
            adaptive=False, n=20, modules="mpmath")
        xx1, yy1 = s1.get_data()
        xx2, yy2 = s2.get_data()
        assert np.allclose(xx1, xx2)
        assert not np.allclose(yy1, yy2)

    # here, there will be different values at x+0j for positive x
    s1 = ComplexSurfaceSeries(arg(sqrt(-z)), (z, -3 - 3j, 3 + 3j),
        n1=21, n2=21, modules=None)
    s2 = ComplexSurfaceSeries(arg(sqrt(-z)), (z, -3 - 3j, 3 + 3j),
        n1=21, n2=21, modules="mpmath")
    xx1, yy1, zz1 = s1.get_data()
    xx2, yy2, zz2 = s2.get_data()
    assert np.allclose(xx1, xx2)
    assert np.allclose(yy1, yy2)
    assert not np.allclose(zz1, zz2)

    s1 = ComplexDomainColoringSeries(arg(sqrt(-z)), (z, -3 - 3j, 3 + 3j),
        n1=21, n2=21, coloring="a", modules=None)
    s2 = ComplexDomainColoringSeries(arg(sqrt(-z)), (z, -3 - 3j, 3 + 3j),
        n1=21, n2=21, coloring="a", modules="mpmath")
    xx1, yy1, _, aa1, ii1, _ = s1.get_data()
    xx2, yy2, _, aa2, ii2, _ = s2.get_data()
    assert np.allclose(xx1, xx2)
    assert np.allclose(yy1, yy2)
    assert not np.allclose(aa1, aa2)
    assert not np.allclose(ii1, ii2)


def test_str():
    u, x, y, z = symbols("u, x:z")

    s = LineOver1DRangeSeries(cos(x), (x, -4, 3))
    assert str(s) == "cartesian line: cos(x) for x over (-4.0, 3.0)"

    d = {"return": "real"}
    s = LineOver1DRangeSeries(cos(x), (x, -4, 3), **d)
    assert str(s) == "cartesian line: re(cos(x)) for x over (-4.0, 3.0)"

    d = {"return": "imag"}
    s = LineOver1DRangeSeries(cos(x), (x, -4, 3), **d)
    assert str(s) == "cartesian line: im(cos(x)) for x over (-4.0, 3.0)"

    d = {"return": "abs"}
    s = LineOver1DRangeSeries(cos(x), (x, -4, 3), **d)
    assert str(s) == "cartesian line: abs(cos(x)) for x over (-4.0, 3.0)"

    d = {"return": "arg"}
    s = LineOver1DRangeSeries(cos(x), (x, -4, 3), **d)
    assert str(s) == "cartesian line: arg(cos(x)) for x over (-4.0, 3.0)"

    s = LineOver1DRangeSeries(cos(u * x), (x, -4, 3), params={u: 1})
    assert str(s) == "interactive cartesian line: cos(u*x) for x over (-4.0, 3.0) and parameters (u,)"

    s = LineOver1DRangeSeries(cos(u * x), (x, -u, 3*y), params={u: 1, y: 1})
    assert str(s) == "interactive cartesian line: cos(u*x) for x over (-u, 3*y) and parameters (u, y)"

    s = AbsArgLineSeries(sqrt(x), (x, -5 + 2j, 5 + 2j))
    assert str(s) == "cartesian abs-arg line: sqrt(x) for x over ((-5+2j), (5+2j))"

    s = AbsArgLineSeries(cos(u * x), (x, -4, 3), params={u: 1})
    assert str(s) == "interactive cartesian abs-arg line: cos(u*x) for x over ((-4+0j), (3+0j)) and parameters (u,)"

    s = Parametric2DLineSeries(cos(x), sin(x), (x, -4, 3))
    assert str(s) == "parametric cartesian line: (cos(x), sin(x)) for x over (-4.0, 3.0)"

    s = Parametric2DLineSeries(cos(u * x), sin(x), (x, -4, 3), params={u: 1})
    assert str(s) == "interactive parametric cartesian line: (cos(u*x), sin(x)) for x over (-4.0, 3.0) and parameters (u,)"

    s = Parametric2DLineSeries(cos(u * x), sin(x), (x, -u, 3*y), params={u: 1, y:1})
    assert str(s) == "interactive parametric cartesian line: (cos(u*x), sin(x)) for x over (-u, 3*y) and parameters (u, y)"

    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -4, 3))
    assert str(s) == "3D parametric cartesian line: (cos(x), sin(x), x) for x over (-4.0, 3.0)"

    s = Parametric3DLineSeries(cos(u*x), sin(x), x, (x, -4, 3), params={u: 1})
    assert str(s) == "interactive 3D parametric cartesian line: (cos(u*x), sin(x), x) for x over (-4.0, 3.0) and parameters (u,)"

    s = Parametric3DLineSeries(cos(u*x), sin(x), x, (x, -u, 3*y), params={u: 1, y: 1})
    assert str(s) == "interactive 3D parametric cartesian line: (cos(u*x), sin(x), x) for x over (-u, 3*y) and parameters (u, y)"

    s = SurfaceOver2DRangeSeries(cos(x * y), (x, -4, 3), (y, -2, 5))
    assert str(s) == "cartesian surface: cos(x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0)"

    s = SurfaceOver2DRangeSeries(cos(u * x * y), (x, -4, 3), (y, -2, 5), params={u: 1})
    assert str(s) == "interactive cartesian surface: cos(u*x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0) and parameters (u,)"

    s = SurfaceOver2DRangeSeries(cos(u * x * y), (x, -4*u, 3), (y, -2, 5*u), params={u: 1})
    assert str(s) == "interactive cartesian surface: cos(u*x*y) for x over (-4*u, 3.0) and y over (-2.0, 5*u) and parameters (u,)"

    s = ContourSeries(cos(x * y), (x, -4, 3), (y, -2, 5))
    assert str(s) == "contour: cos(x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0)"

    s = ContourSeries(cos(u * x * y), (x, -4, 3), (y, -2, 5), params={u: 1})
    assert str(s) == "interactive contour: cos(u*x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0) and parameters (u,)"

    s = ParametricSurfaceSeries(cos(x * y), sin(x * y), x * y,
        (x, -4, 3), (y, -2, 5))
    assert str(s) == "parametric cartesian surface: (cos(x*y), sin(x*y), x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0)"

    s = ParametricSurfaceSeries(cos(u * x * y), sin(x * y), x * y,
        (x, -4, 3), (y, -2, 5), params={u: 1})
    assert str(s) == "interactive parametric cartesian surface: (cos(u*x*y), sin(x*y), x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0) and parameters (u,)"

    s = ImplicitSeries(x < y, (x, -5, 4), (y, -3, 2))
    assert str(s) == "Implicit expression: x < y for x over (-5.0, 4.0) and y over (-3.0, 2.0)"

    s = ComplexPointSeries(2 + 3 * I)
    assert str(s) == "complex points: (2 + 3*I,)"

    s = ComplexPointSeries([2 + 3 * I, 4 * I])
    assert str(s) == "complex points: (2 + 3*I, 4*I)"

    s = ComplexPointSeries(2 + 3 * I * y, params={y: 1})
    assert str(s) == "interactive complex points: (3*I*y + 2,) and parameters (y,)"

    s = ComplexPointSeries([2 + 3 * I, 4 * I * y], params={y: 1})
    assert str(s) == "interactive complex points: (2 + 3*I, 4*I*y) and parameters (y,)"

    d = {"threed": False, "return": "real"}
    s = ComplexSurfaceSeries(sqrt(z), (z, -2-3j, 4+5j), **d)
    assert str(s) == "complex contour: re(sqrt(z)) for re(z) over (-2.0, 4.0) and im(z) over (-3.0, 5.0)"

    d = {"threed": True, "return": "real"}
    s = ComplexSurfaceSeries(sqrt(z), (z, -2-3j, 4+5j), **d)
    assert str(s) == "complex cartesian surface: re(sqrt(z)) for re(z) over (-2.0, 4.0) and im(z) over (-3.0, 5.0)"

    d = {"threed": True, "return": "imag"}
    s = ComplexSurfaceSeries(sqrt(z), (z, -2-3j, 4+5j), **d)
    assert str(s) == "complex cartesian surface: im(sqrt(z)) for re(z) over (-2.0, 4.0) and im(z) over (-3.0, 5.0)"

    d = {"threed": True, "return": "abs"}
    s = ComplexSurfaceSeries(sqrt(z), (z, -2-3j, 4+5j), **d)
    assert str(s) == "complex cartesian surface: abs(sqrt(z)) for re(z) over (-2.0, 4.0) and im(z) over (-3.0, 5.0)"

    d = {"threed": True, "return": "arg"}
    s = ComplexSurfaceSeries(sqrt(z), (z, -2-3j, 4+5j), **d)
    assert str(s) == "complex cartesian surface: arg(sqrt(z)) for re(z) over (-2.0, 4.0) and im(z) over (-3.0, 5.0)"

    s = ComplexDomainColoringSeries(sqrt(z), (z, -2-3j, 4+5j),
        threed=False)
    assert str(s) == "complex domain coloring: sqrt(z) for re(z) over (-2.0, 4.0) and im(z) over (-3.0, 5.0)"

    s = ComplexDomainColoringSeries(sqrt(z), (z, -2-3j, 4+5j),
        threed=True)
    assert str(s) == "complex 3D domain coloring: sqrt(z) for re(z) over (-2.0, 4.0) and im(z) over (-3.0, 5.0)"

    d = {"return": "real"}
    s = ComplexSurfaceSeries(x * sqrt(z), (z, -2-3j, 4+5j),
        threed=False, params={x: 1}, **d)
    assert str(s) == "interactive complex contour: re(x*sqrt(z)) for re(z) over (-2.0, 4.0) and im(z) over (-3.0, 5.0) and parameters (x,)"

    s = ComplexSurfaceSeries(x * sqrt(z), (z, -2-3j, 4+5j),
        threed=True, params={x: 1}, **d)
    assert str(s) == "interactive complex cartesian surface: re(x*sqrt(z)) for re(z) over (-2.0, 4.0) and im(z) over (-3.0, 5.0) and parameters (x,)"

    a, b = symbols("a, b", real=True)
    s = ComplexSurfaceSeries(x * sqrt(z), (z, -2*a-3j, 4+5j*b),
        threed=True, params={x: 1, a: 1, b: 1}, **d)
    assert str(s) == "interactive complex cartesian surface: re(x*sqrt(z)) for re(z) over (-2*a, 4.0) and im(z) over (-3.0, 5.0*b) and parameters (x, a, b)"

    s = ComplexDomainColoringSeries(x * sqrt(z), (z, -2-3j, 4+5j),
        "test", params={x: 1}, threed=False)
    assert str(s) == "interactive complex domain coloring: x*sqrt(z) for re(z) over (-2.0, 4.0) and im(z) over (-3.0, 5.0) and parameters (x,)"

    s = ComplexDomainColoringSeries(x * sqrt(z), (z, -2-3j, 4+5j),
        "test", params={x: 1}, threed=True)
    assert str(s) == "interactive complex 3D domain coloring: x*sqrt(z) for re(z) over (-2.0, 4.0) and im(z) over (-3.0, 5.0) and parameters (x,)"

    a, b = symbols("a, b", real=True)
    s = ComplexDomainColoringSeries(x * sqrt(z), (z, -2*a-3j, 4+5j*b),
        threed=True, params={x: 1, a: 1, b: 1})
    assert str(s) == "interactive complex 3D domain coloring: x*sqrt(z) for re(z) over (-2*a, 4.0) and im(z) over (-3.0, 5.0*b) and parameters (x, a, b)"

    s = Vector2DSeries(-y, x, (x, -5, 4), (y, -3, 2))
    assert str(s) == "2D vector series: [-y, x] over (x, -5.0, 4.0), (y, -3.0, 2.0)"

    s = Vector3DSeries(z, y, x, (x, -5, 4), (y, -3, 2), (z, -6, 7))
    assert str(s) == "3D vector series: [z, y, x] over (x, -5.0, 4.0), (y, -3.0, 2.0), (z, -6.0, 7.0)"

    s = Vector2DSeries(-y, x * z, (x, -5, 4), (y, -3, 2),
        "test", params={z: 1})
    assert str(s) == "interactive 2D vector series: [-y, x*z] over (x, -5.0, 4.0), (y, -3.0, 2.0) and parameters (z,)"

    a, b = symbols("a, b")
    s = Vector2DSeries(-y, x * z, (x, -5 * a, 4 * b), (y, -3 * b, 2 * a),
        "test", params={z: 1, a: 1, b: 1})
    assert str(s) == "interactive 2D vector series: [-y, x*z] over (x, -5*a, 4*b), (y, -3*b, 2*a) and parameters (z, a, b)"

    s = Vector3DSeries(-y, x * z, x, (x, -5, 4), (y, -3, 2), (z, -6, 7),
        "test", params={z: 1})
    assert str(s) == "interactive 3D vector series: [-y, x*z, x] over (x, -5.0, 4.0), (y, -3.0, 2.0), (z, -6.0, 7.0) and parameters (z,)"

    s = SliceVector3DSeries(Plane((0, 0, 0), (1, 0, 0)), z, y, x,
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    assert str(s) == "sliced 3D vector series: [z, y, x] over (x, -5.0, 4.0), (y, -3.0, 2.0), (z, -6.0, 7.0) at plane series: Plane(Point3D(0, 0, 0), (1, 0, 0)) over (x, -5, 4), (y, -3, 2), (z, -6, 7)"

    s = SliceVector3DSeries(
        Plane((0, 0, 0), (1, 0, 0)), u * z, u * y, x,
        (x, -5, 4), (y, -3, 2), (z, -6, 7), params={u: 1})
    assert str(s) == "sliced interactive 3D vector series: [u*z, u*y, x] over (x, -5.0, 4.0), (y, -3.0, 2.0), (z, -6.0, 7.0) and parameters (u,) at plane series: Plane(Point3D(0, 0, 0), (1, 0, 0)) over (x, -5, 4), (y, -3, 2), (z, -6, 7)"

    s = PlaneSeries(Plane((0, 0, 0), (1, 1, 1)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    assert str(s) == "plane series: Plane(Point3D(0, 0, 0), (1, 1, 1)) over (x, -5, 4), (y, -3, 2), (z, -6, 7)"

    s = PlaneSeries(Plane((z, 0, 0), (1, 1, 1)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7), params={z: 1})
    assert str(s) == "interactive plane series: Plane(Point3D(z, 0, 0), (1, 1, 1)) over (x, -5, 4), (y, -3, 2), (z, -6, 7) and parameters (z,)"

    s = GeometrySeries(Circle(Point(0, 0), 5))
    assert str(s) == "geometry entity: Circle(Point2D(0, 0), 5)"

    s = GeometrySeries(Circle(Point(x, 0), 5), params={x: 1})
    assert str(s) == "interactive geometry entity: Circle(Point2D(x, 0), 5) and parameters (x,)"

    s = Implicit3DSeries(x**2 + y**3 - z**2, (x, -2, 2), (y, -3, 3), (z, -4, 4))
    assert str(s) == "implicit surface series: x**2 + y**3 - z**2 for x over (-2.0, 2.0) and y over (-3.0, 3.0) and z over (-4.0, 4.0)"


def test_use_cm():
    # verify that the `use_cm` attribute is implemented.

    u, x, y, z = symbols("u, x:z")

    s = List2DSeries([1, 2, 3, 4], [5, 6, 7, 8], use_cm=True)
    assert s.use_cm
    s = List2DSeries([1, 2, 3, 4], [5, 6, 7, 8], use_cm=False)
    assert not s.use_cm

    s = List3DSeries([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], use_cm=True)
    assert s.use_cm
    s = List3DSeries([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], use_cm=False)
    assert not s.use_cm

    s = AbsArgLineSeries(sqrt(x), (x, -5 + 2j, 5 + 2j), use_cm=True)
    assert s.use_cm
    s = AbsArgLineSeries(sqrt(x), (x, -5 + 2j, 5 + 2j), use_cm=False)
    assert not s.use_cm

    s = Parametric2DLineSeries(cos(x), sin(x), (x, -4, 3), use_cm=True)
    assert s.use_cm
    s = Parametric2DLineSeries(cos(x), sin(x), (x, -4, 3), use_cm=False)
    assert not s.use_cm

    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -4, 3),
        use_cm=True)
    assert s.use_cm
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -4, 3),
        use_cm=False)
    assert not s.use_cm

    s = SurfaceOver2DRangeSeries(cos(x * y), (x, -4, 3), (y, -2, 5),
        use_cm=True)
    assert s.use_cm
    s = SurfaceOver2DRangeSeries(cos(x * y), (x, -4, 3), (y, -2, 5),
        use_cm=False)
    assert not s.use_cm

    s = ParametricSurfaceSeries(cos(x * y), sin(x * y), x * y,
        (x, -4, 3), (y, -2, 5), use_cm=True)
    assert s.use_cm
    s = ParametricSurfaceSeries(cos(x * y), sin(x * y), x * y,
        (x, -4, 3), (y, -2, 5), use_cm=False)
    assert not s.use_cm

    s = ComplexSurfaceSeries(sqrt(z), (z, -2-3j, 4+5j), threed=False,
        use_cm=True)
    assert s.use_cm
    s = ComplexSurfaceSeries(sqrt(z), (z, -2-3j, 4+5j), threed=False,
        use_cm=False)
    assert not s.use_cm

    s = ComplexSurfaceSeries(sqrt(z), (z, -2-3j, 4+5j), threed=True,
        use_cm=True)
    assert s.use_cm
    s = ComplexSurfaceSeries(sqrt(z), (z, -2-3j, 4+5j), threed=True,
        use_cm=False)
    assert not s.use_cm

    s = ComplexDomainColoringSeries(sqrt(z), (z, -2-3j, 4+5j),
        threed=False, use_cm=True)
    assert s.use_cm
    s = ComplexDomainColoringSeries(sqrt(z), (z, -2-3j, 4+5j),
        threed=False, use_cm=False)
    assert not s.use_cm

    s = ComplexDomainColoringSeries(sqrt(z), (z, -2-3j, 4+5j),
        threed=True, use_cm=True)
    assert s.use_cm
    s = ComplexDomainColoringSeries(sqrt(z), (z, -2-3j, 4+5j),
        threed=True, use_cm=False)
    assert not s.use_cm

    s = Vector2DSeries(-y, x, (x, -5, 4), (y, -3, 2), use_cm=True)
    assert s.use_cm
    s = Vector2DSeries(-y, x, (x, -5, 4), (y, -3, 2), use_cm=False)
    assert not s.use_cm

    s = Vector3DSeries(z, y, x, (x, -5, 4), (y, -3, 2), (z, -6, 7),
        use_cm=True)
    assert s.use_cm
    s = Vector3DSeries(z, y, x, (x, -5, 4), (y, -3, 2), (z, -6, 7),
        use_cm=False)
    assert not s.use_cm

    s = SliceVector3DSeries(Plane((0, 0, 0), (1, 0, 0)), z, y, x,
        (x, -5, 4), (y, -3, 2), (z, -6, 7), use_cm=True)
    assert s.use_cm
    s = SliceVector3DSeries(Plane((0, 0, 0), (1, 0, 0)), z, y, x,
        (x, -5, 4), (y, -3, 2), (z, -6, 7), use_cm=False)
    assert not s.use_cm

    s = PlaneSeries(Plane((0, 0, 0), (1, 1, 1)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7), use_cm=True)
    assert s.use_cm
    s = PlaneSeries(Plane((0, 0, 0), (1, 1, 1)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7), use_cm=False)
    assert not s.use_cm

    s = GeometrySeries(Circle(Point(0, 0), 5), use_cm=True)
    assert s.use_cm
    s = GeometrySeries(Circle(Point(0, 0), 5), use_cm=False)
    assert not s.use_cm


def test_sums():
    # test that data series are able to deal with sums
    x, y, u = symbols("x, y, u")

    def do_test(data1, data2):
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert np.allclose(d1, d2)

    s = LineOver1DRangeSeries(Sum(1 / x ** y, (x, 1, 1000)), (y, 2, 10),
        adaptive=False, only_integers=True)
    xx, yy = s.get_data()

    s1 = LineOver1DRangeSeries(Sum(1 / x, (x, 1, y)), (y, 2, 10),
        adaptive=False, only_integers=True)
    xx1, yy1 = s1.get_data()

    s2 = LineOver1DRangeSeries(Sum(u / x, (x, 1, y)), (y, 2, 10),
        params={u: 1}, only_integers=True)
    xx2, yy2 = s2.get_data()
    xx1 = xx1.astype(float)
    xx2 = xx2.astype(float)
    do_test([xx1, yy1], [xx2, yy2])

    s = LineOver1DRangeSeries(Sum(1 / x, (x, 1, y)), (y, 2, 10),
        adaptive=True)
    with warns(
            UserWarning,
            match="The evaluation with NumPy/SciPy failed.",
        ):
        raises(TypeError, lambda: s.get_data())


def test_absargline():
    # verify that AbsArgLineSeries produces the correct results
    x, u = symbols("x, u")

    s1 = AbsArgLineSeries(sqrt(x), (x, -5, 5), adaptive=False, n=10)
    s2 = AbsArgLineSeries(sqrt(u * x), (x, -5, 5), adaptive=False, n=10,
        params={u: 1})
    data1 = s1.get_data()
    data2 = s2.get_data()
    assert len(data1) == len(data2)
    for d1, d2 in zip(data1, data2):
        assert np.allclose(d1, d2)
    # there shouldn't be nan values
    assert np.invert(np.isnan(data1[1])).all()
    assert np.invert(np.isnan(data1[2])).all()

    s3 = AbsArgLineSeries(sqrt(x), (x, -5, 5), adaptive=True)
    data3 = s3.get_data()
    assert np.invert(np.isnan(data3[1])).all()
    assert np.invert(np.isnan(data3[2])).all()


def test_apply_transforms():
    # verify that transformation functions get applied to the output
    # of data series
    x, y, z, u, v = symbols("x:z, u, v")

    s1 = LineOver1DRangeSeries(cos(x), (x, -2*pi, 2*pi), adaptive=False, n=10)
    s2 = LineOver1DRangeSeries(cos(x), (x, -2*pi, 2*pi), adaptive=False, n=10,
        tx=np.rad2deg)
    s3 = LineOver1DRangeSeries(cos(x), (x, -2*pi, 2*pi), adaptive=False, n=10,
        ty=np.rad2deg)
    s4 = LineOver1DRangeSeries(cos(x), (x, -2*pi, 2*pi), adaptive=False, n=10,
        tx=np.rad2deg, ty=np.rad2deg)

    x1, y1 = s1.get_data()
    x2, y2 = s2.get_data()
    x3, y3 = s3.get_data()
    x4, y4 = s4.get_data()
    assert np.isclose(x1[0], -2*np.pi) and np.isclose(x1[-1], 2*np.pi)
    assert (y1.min() < -0.9) and (y1.max() > 0.9)
    assert np.isclose(x2[0], -360) and np.isclose(x2[-1], 360)
    assert (y2.min() < -0.9) and (y2.max() > 0.9)
    assert np.isclose(x3[0], -2*np.pi) and np.isclose(x3[-1], 2*np.pi)
    assert (y3.min() < -52) and (y3.max() > 52)
    assert np.isclose(x4[0], -360) and np.isclose(x4[-1], 360)
    assert (y4.min() < -52) and (y4.max() > 52)

    xx = np.linspace(-2*np.pi, 2*np.pi, 10)
    yy = np.cos(xx)
    s1 = List2DSeries(xx, yy)
    s2 = List2DSeries(xx, yy, tx=np.rad2deg, ty=np.rad2deg)
    x1, y1 = s1.get_data()
    x2, y2 = s2.get_data()
    assert np.isclose(x1[0], -2*np.pi) and np.isclose(x1[-1], 2*np.pi)
    assert (y1.min() < -0.9) and (y1.max() > 0.9)
    assert np.isclose(x2[0], -360) and np.isclose(x2[-1], 360)
    assert (y2.min() < -52) and (y2.max() > 52)

    zz = np.linspace(-2*np.pi, 2*np.pi, 10)
    xx = np.cos(zz)
    yy = np.cos(zz)
    s1 = List3DSeries(xx, yy, zz)
    s2 = List3DSeries(xx, yy, zz, tx=lambda t: 2*t, ty=lambda t: 3*t, tz=lambda t: 4*t)
    x1, y1, z1 = s1.get_data()
    x2, y2, z2 = s2.get_data()
    assert np.allclose(xx, x1) and np.allclose(yy, y1) and np.allclose(zz, z1)
    assert np.allclose(xx, x2 / 2)
    assert np.allclose(yy, y2 / 3)
    assert np.allclose(zz, z2 / 4)

    s1 = AbsArgLineSeries(cos(x) + sin(I * x), (x, -2*pi, 2*pi),
        n=10, adaptive=False)
    s2 = AbsArgLineSeries(cos(x) + sin(I * x), (x, -2*pi, 2*pi),
        n=10, adaptive=False,
        tx=np.rad2deg, ty=lambda x: 2*x, tz=lambda x: 3*x)
    x1, y1, a1 = s1.get_data()
    x2, y2, a2 = s2.get_data()
    assert np.allclose(x1, np.deg2rad(x2))
    assert np.allclose(y1, y2 / 2)
    assert np.allclose(a1, a2 / 3)

    s1 = Parametric2DLineSeries(
        sin(x), cos(x), (x, -pi, pi), adaptive=False, n=10)
    s2 = Parametric2DLineSeries(
        sin(x), cos(x), (x, -pi, pi), adaptive=False, n=10,
        tx=np.rad2deg, ty=np.rad2deg, tp=np.rad2deg)
    x1, y1, a1 = s1.get_data()
    x2, y2, a2 = s2.get_data()
    assert np.allclose(x1, np.deg2rad(x2))
    assert np.allclose(y1, np.deg2rad(y2))
    assert np.allclose(a1, np.deg2rad(a2))

    s1 =  Parametric3DLineSeries(
        sin(x), cos(x), x, (x, -pi, pi), adaptive=False, n=10)
    s2 = Parametric3DLineSeries(
        sin(x), cos(x), x, (x, -pi, pi), adaptive=False, n=10, tp=np.rad2deg)
    x1, y1, z1, a1 = s1.get_data()
    x2, y2, z2, a2 = s2.get_data()
    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)
    assert np.allclose(z1, z2)
    assert np.allclose(a1, np.deg2rad(a2))

    s1 = SurfaceOver2DRangeSeries(
        cos(x**2 + y**2), (x, -2*pi, 2*pi), (y, -2*pi, 2*pi),
        adaptive=False, n1=10, n2=10)
    s2 = SurfaceOver2DRangeSeries(
        cos(x**2 + y**2), (x, -2*pi, 2*pi), (y, -2*pi, 2*pi),
        adaptive=False, n1=10, n2=10,
        tx=np.rad2deg, ty=lambda x: 2*x, tz=lambda x: 3*x)
    x1, y1, z1 = s1.get_data()
    x2, y2, z2 = s2.get_data()
    assert np.allclose(x1, np.deg2rad(x2))
    assert np.allclose(y1, y2 / 2)
    assert np.allclose(z1, z2 / 3)

    s1 = ParametricSurfaceSeries(
        u + v, u - v, u * v, (u, 0, 2*pi), (v, 0, pi),
        adaptive=False, n1=10, n2=10)
    s2 = ParametricSurfaceSeries(
        u + v, u - v, u * v, (u, 0, 2*pi), (v, 0, pi),
        adaptive=False, n1=10, n2=10,
        tx=np.rad2deg, ty=lambda x: 2*x, tz=lambda x: 3*x)
    x1, y1, z1, u1, v1 = s1.get_data()
    x2, y2, z2, u2, v2 = s2.get_data()
    assert np.allclose(x1, np.deg2rad(x2))
    assert np.allclose(y1, y2 / 2)
    assert np.allclose(z1, z2 / 3)
    assert np.allclose(u1, u2)
    assert np.allclose(v1, v2)

    s1 = Vector2DSeries(sin(y), cos(x), (x, -pi, pi), (y, -pi, pi), n1=5, n2=5)
    s2 = Vector2DSeries(sin(y), cos(x), (x, -pi, pi), (y, -pi, pi), n1=5, n2=5,
        tx=np.rad2deg, ty=lambda x: 2*x)
    x1, y1, u1, v1 = s1.get_data()
    x2, y2, u2, v2 = s2.get_data()
    assert np.allclose(x1, np.deg2rad(x2))
    assert np.allclose(y1, y2 / 2)
    assert np.allclose(u1, np.deg2rad(u2))
    assert np.allclose(v1, v2 / 2)

    s1 = Vector3DSeries(
        x, y, z, (x, -1, 1), (y, -1, 1), (z, -1, 1), n1=5, n2=5, n3=5)
    s2 = Vector3DSeries(
        x, y, z, (x, -1, 1), (y, -1, 1), (z, -1, 1), n1=5, n2=5, n3=5,
        tx=np.rad2deg, ty=lambda x: 2*x, tz=lambda x: 3*x)
    x1, y1, z1, u1, v1, w1 = s1.get_data()
    x2, y2, z2, u2, v2, w2 = s2.get_data()
    assert np.allclose(x1, np.deg2rad(x2))
    assert np.allclose(y1, y2 / 2)
    assert np.allclose(z1, z2 / 3)
    assert np.allclose(u1, np.deg2rad(u2))
    assert np.allclose(v1, v2 / 2)
    assert np.allclose(w1, w2 / 3)

    s1 = ComplexDomainColoringSeries(
        (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I),
        n1=10, n2=10, n3=10)
    s2 = ComplexDomainColoringSeries(
        (z ** 2 + 1) / (z ** 2 - 1), (z, -3 - 4 * I, 3 + 4 * I),
        n1=10, n2=10, n3=10, tz=lambda t: t/10)
    x1, y1, z1, a1, b1, c1 = s1.get_data()
    x2, y2, z2, a2, b2, c2 = s2.get_data()
    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)
    assert np.allclose(z1, z2 * 10)
    assert np.allclose(a1, a2)
    assert np.allclose(b1, b2)
    assert np.allclose(c1, c2)


def test_series_labels():
    # verify that series return the correct label, depending on the plot
    # type and input arguments. If the user set custom label on a data series,
    # it should returned un-modified.

    x, y, z, u, v = symbols("x, y, z, u, v")
    wrapper = "$%s$"

    expr = cos(x)
    s1 = LineOver1DRangeSeries(expr, (x, -2, 2), None)
    s2 = LineOver1DRangeSeries(expr, (x, -2, 2), "test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    expr = (cos(x), sin(x))
    s1 = Parametric2DLineSeries(*expr, (x, -2, 2), None, use_cm=True)
    s2 = Parametric2DLineSeries(*expr, (x, -2, 2), "test", use_cm=True)
    s3 = Parametric2DLineSeries(*expr, (x, -2, 2), None, use_cm=False)
    s4 = Parametric2DLineSeries(*expr, (x, -2, 2), "test", use_cm=False)
    assert s1.get_label(False) == "x"
    assert s1.get_label(True) == wrapper % "x"
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"
    assert s3.get_label(False) == str(expr)
    assert s3.get_label(True) == wrapper % latex(expr)
    assert s4.get_label(False) == "test"
    assert s4.get_label(True) == "test"

    expr = (cos(x), sin(x), x)
    s1 = Parametric3DLineSeries(*expr, (x, -2, 2), None, use_cm=True)
    s2 = Parametric3DLineSeries(*expr, (x, -2, 2), "test", use_cm=True)
    s3 = Parametric3DLineSeries(*expr, (x, -2, 2), None, use_cm=False)
    s4 = Parametric3DLineSeries(*expr, (x, -2, 2), "test", use_cm=False)
    assert s1.get_label(False) == "x"
    assert s1.get_label(True) == wrapper % "x"
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"
    assert s3.get_label(False) == str(expr)
    assert s3.get_label(True) == wrapper % latex(expr)
    assert s4.get_label(False) == "test"
    assert s4.get_label(True) == "test"

    expr = cos(x**2 + y**2)
    s1 = SurfaceOver2DRangeSeries(expr, (x, -2, 2), (y, -2, 2), None)
    s2 = SurfaceOver2DRangeSeries(expr, (x, -2, 2), (y, -2, 2), "test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    expr = (cos(x - y), sin(x + y), x - y)
    s1 = ParametricSurfaceSeries(*expr, (x, -2, 2), (y, -2, 2), None)
    s2 = ParametricSurfaceSeries(*expr, (x, -2, 2), (y, -2, 2), "test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    expr = Eq(cos(x - y), 0)
    s1 = ImplicitSeries(expr, (x, -10, 10), (y, -10, 10), None)
    s2 = ImplicitSeries(expr, (x, -10, 10), (y, -10, 10), "test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    expr = (-sin(y), cos(x))
    s1 = Vector2DSeries(*expr, (x, -2, 2), (y, -2, 2), None)
    s2 = Vector2DSeries(*expr, (x, -2, 2), (y, -2, 2), "test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    expr = (-sin(y), cos(x), cos(z))
    s1 = Vector3DSeries(*expr, (x, -2, 2), (y, -2, 2), (z, -2, 2), None)
    s2 = Vector3DSeries(*expr, (x, -2, 2), (y, -2, 2), (z, -2, 2), "test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    s1 = SliceVector3DSeries(Plane((-1, 0, 0), (1, 0, 0)), *expr,
        (x, -2, 2), (y, -2, 2), (z, -2, 2), None)
    s2 = SliceVector3DSeries(Plane((-1, 0, 0), (1, 0, 0)), *expr,
        (x, -2, 2), (y, -2, 2), (z, -2, 2), "test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    plane = Plane((-1, 0, 0), (1, 1, 0))
    s1 = PlaneSeries(plane, (x, -2, 2), (y, -2, 2), (z, -2, 2), None)
    s2 = PlaneSeries(plane, (x, -2, 2), (y, -2, 2), (z, -2, 2), "test")
    assert s1.get_label(False) == str(plane)
    assert s1.get_label(True) == wrapper % latex(plane)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    expr = Circle(Point(0, 0), 5)
    s1 = GeometrySeries(expr, label=None)
    s2 = GeometrySeries(expr, label="test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    s1 = List2DSeries([0, 1, 2, 3], [0, 1, 2, 3], "test")
    assert s1.get_label(False) == "test"
    assert s1.get_label(True) == "test"

    s1 = List3DSeries([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], "test")
    assert s1.get_label(False) == "test"
    assert s1.get_label(True) == "test"

    s1 = ComplexPointSeries([1 + 2 * I, 3 + 4 * I], "test")
    assert s1.get_label(False) == "test"
    assert s1.get_label(True) == "test"

    expr = cos(x)
    s1 = AbsArgLineSeries(expr, (x, 1e-05, 1e05), None)
    s2 = AbsArgLineSeries(expr, (x, 1e-05, 1e05), "test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    expr = sqrt(x)
    s1 = ComplexSurfaceSeries(expr, (x, -3.5 - 2.5j, 3.5 + 2.5j), None)
    s2 = ComplexSurfaceSeries(expr, (x, -3.5 - 2.5j, 3.5 + 2.5j), "test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    expr = sqrt(x)
    s1 = ComplexDomainColoringSeries(expr, (x, -3.5 - 2.5j, 3.5 + 2.5j),
        None)
    s2 = ComplexDomainColoringSeries(expr, (x, -3.5 - 2.5j, 3.5 + 2.5j),
        "test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"

    expr = x**2 + y**3 - z**2
    s1 = Implicit3DSeries(expr, (x, -2, 2), (y, -3, 3), (z, -4, 4),
        None)
    s2 = Implicit3DSeries(expr, (x, -2, 2), (y, -3, 3), (z, -4, 4),
        "test")
    assert s1.get_label(False) == str(expr)
    assert s1.get_label(True) == wrapper % latex(expr)
    assert s2.get_label(False) == "test"
    assert s2.get_label(True) == "test"


def test_surface_use_cm():
    # verify that SurfaceOver2DRangeSeries and ParametricSurfaceSeries get
    # the same value for use_cm

    x, y, u, v = symbols("x, y, u, v")

    # they read the same value from default settings
    s1 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2))
    s2 = ParametricSurfaceSeries(u * cos(v), u * sin(v), u,
        (u, 0, 1), (v, 0 , 2*pi))
    assert s1.use_cm == s2.use_cm

    # they get the same value
    s1 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        use_cm=False)
    s2 = ParametricSurfaceSeries(u * cos(v), u * sin(v), u,
        (u, 0, 1), (v, 0 , 2*pi), use_cm=False)
    assert s1.use_cm == s2.use_cm

    # they get the same value
    s1 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        use_cm=True)
    s2 = ParametricSurfaceSeries(u * cos(v), u * sin(v), u,
        (u, 0, 1), (v, 0 , 2*pi), use_cm=True)
    assert s1.use_cm == s2.use_cm


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_sliced_vector_interactive_series():
    # verify that interactive SliceVector3DSeries using an instance of
    # SurfaceBaseSeries as a slice, produced the correct results,
    # ie both the vector series and the slice series updates their parameters.
    x, y, z, u, v, t = symbols("x, y, z, u, v, t")

    N = CoordSys3D("N")
    i, j, k = N.base_vectors()
    xn, yn, zn = N.base_scalars()
    expr = -xn**2 * tan(t)**2 + yn**2 + zn**2
    g = gradient(expr)
    m = g.magnitude()

    slice_series = ParametricSurfaceSeries(
        u / tan(t), u * cos(v), u * sin(v),
        (u, 0.0, 1.0), (v, 0.0, 2 * pi), label="slice",
        params={t: 0.5},  n1=5, n2=5)
    slice_copy = ParametricSurfaceSeries(
        u / tan(t), u * cos(v), u * sin(v),
        (u, 0.0, 1.0), (v, 0.0, 2 * pi), label="slice_copy",
        params={t: 0.5},  n1=5, n2=5)
    s = SliceVector3DSeries(
        slice_series, *(g / m).to_matrix(N),
        (xn, -5, 5), (yn, -5, 5), (zn, -5, 5), label="vector",
        params={t: 0.5})

    x1, y1, z1, _, _ = slice_copy.get_data()
    x3, y3, z3, _, _ = slice_series.get_data()
    x2, y2, z2, _, _, _ = s.get_data()
    assert np.allclose(x1, x2) and np.allclose(y1, y2) and np.allclose(z1, z2)
    assert np.allclose(x1, x3) and np.allclose(y1, y3) and np.allclose(z1, z3)

    # now update the parameter: slice shoud produce the same data as slice_copy
    v = 0.25
    slice_copy.params = {t: v}
    s.params = {t: v}
    x1, y1, z1, _, _ = slice_copy.get_data()
    x3, y3, z3, _, _ = slice_series.get_data()
    x2, y2, z2, _, _, _ = s.get_data()
    assert np.allclose(x1, x2) and np.allclose(y1, y2) and np.allclose(z1, z2)
    assert np.allclose(x1, x3) and np.allclose(y1, y3) and np.allclose(z1, z3)


def test_sliced_vector_series_slice_exprs():
    # given a vector field discretized in some domain, for example x,y,z,
    # the slice expression can be f(x, y) or f(y, z) or f(x, z) and the series
    # would return correct data.

    x, y, z = symbols("x, y, z")

    _slice_xy = cos(sqrt(x**2 + y**2))
    s = SliceVector3DSeries(
        _slice_xy, z, y, x, (x, -10, 10), (y, -5, 5), (z, -3, 3),
        n1=4, n2=8, n3=12)
    data = s.get_data()
    assert all(d.shape == (8, 4) for d in data)
    assert np.allclose(data[0][0, :], np.linspace(-10, 10, 4))
    assert np.allclose(data[1][:, 0], np.linspace(-5, 5, 8))

    _slice_yz = cos(sqrt(y**2 + z**2))
    s = SliceVector3DSeries(
        _slice_yz, z, y, x, (x, -10, 10), (y, -5, 5), (z, -3, 3),
        n1=4, n2=8, n3=12)
    data = s.get_data()
    assert all(d.shape == (12, 8) for d in data)
    assert np.allclose(data[1][0, :], np.linspace(-5, 5, 8))
    assert np.allclose(data[2][:, 0], np.linspace(-3, 3, 12))

    _slice_xz = cos(sqrt(x**2 + z**2))
    s = SliceVector3DSeries(
        _slice_xz, z, y, x, (x, -10, 10), (y, -5, 5), (z, -3, 3),
        n1=4, n2=8, n3=12)
    data = s.get_data()
    assert all(d.shape == (12, 4) for d in data)
    assert np.allclose(data[0][0, :], np.linspace(-10, 10, 4))
    assert np.allclose(data[2][:, 0], np.linspace(-3, 3, 12))


def test_sliced_vector_series_slice_instantiation():
    # verify that the sliced surface is instantiated correctly.

    x, y, z, t = symbols("x, y, z, t")

    _slice_xy = cos(sqrt(x**2 + y**2))
    s = SliceVector3DSeries(
        _slice_xy, z, y, x, (x, -10, 10), (y, -5, 5), (z, -3, 3),
        n1=4, n2=8, n3=12)
    assert isinstance(s.slice_surf_series, SurfaceOver2DRangeSeries)
    assert not s.slice_surf_series.is_interactive

    _slice_xy = cos(sqrt(x**2 + y**2)) * t
    s = SliceVector3DSeries(
        _slice_xy, z, y, x, (x, -10, 10), (y, -5, 5), (z, -3, 3),
        n1=4, n2=8, n3=12, params={t: 1})
    assert isinstance(s.slice_surf_series, SurfaceOver2DRangeSeries)
    assert s.slice_surf_series.is_interactive


def test_is_polar_2d_parametric():
    # verify that Parametric2DLineSeries isable to apply polar discretization,
    # which is used when polar_plot is executed with polar_axis=True
    t, u = symbols("t u")

    # NOTE: a sufficiently big n must be provided, or else tests
    # are going to fail
    # No colormap
    f = sin(4 * t)
    s1 = Parametric2DLineSeries(f * cos(t), f * sin(t), (t, 0, 2*pi),
        adaptive=False, n=10, is_polar=False, use_cm=False)
    x1, y1, p1 = s1.get_data()
    s2 = Parametric2DLineSeries(f * cos(t), f * sin(t), (t, 0, 2*pi),
        adaptive=False, n=10, is_polar=True, use_cm=False)
    th, r, p2 = s2.get_data()
    assert (not np.allclose(x1, th)) and (not np.allclose(y1, r))
    assert np.allclose(p1, p2)

    # With colormap
    s3 = Parametric2DLineSeries(f * cos(t), f * sin(t), (t, 0, 2*pi),
        adaptive=False, n=10, is_polar=False, color_func=lambda t: 2*t)
    x3, y3, p3 = s3.get_data()
    s4 = Parametric2DLineSeries(f * cos(t), f * sin(t), (t, 0, 2*pi),
        adaptive=False, n=10, is_polar=True, color_func=lambda t: 2*t)
    th4, r4, p4 = s4.get_data()
    assert np.allclose(p3, p4) and (not np.allclose(p1, p3))
    assert np.allclose(x3, x1) and np.allclose(y3, y1)
    assert np.allclose(th4, th) and np.allclose(r4, r)


def test_is_polar_3d():
    # verify that SurfaceOver2DRangeSeries is able to apply
    # polar discretization

    x, y, t = symbols("x, y, t")
    expr = (x**2 - 1)**2
    s1 = SurfaceOver2DRangeSeries(expr, (x, 0, 1.5), (y, 0, 2 * pi),
        n=10, adaptive=False, is_polar=False)
    s2 = SurfaceOver2DRangeSeries(expr, (x, 0, 1.5), (y, 0, 2 * pi),
        n=10, adaptive=False, is_polar=True)
    x1, y1, z1 = s1.get_data()
    x2, y2, z2 = s2.get_data()
    x22, y22 = x1 * np.cos(y1), x1 * np.sin(y1)
    assert np.allclose(x2, x22)
    assert np.allclose(y2, y22)


def test_color_func():
    # verify that eval_color_func produces the expected results in order to
    # maintain back compatibility with the old sympy.plotting module

    x, y, z, u, v = symbols("x, y, z, u, v")

    # color func: returns x, y, color and s is parametric
    xx = np.linspace(-3, 3, 10)
    yy1 = np.cos(xx)
    s = List2DSeries(xx, yy1, color_func=lambda x, y: 2 * x, use_cm=True)
    xxs, yys, col = s.get_data()
    assert np.allclose(xx, xxs)
    assert np.allclose(yy1, yys)
    assert np.allclose(2 * xx, col)
    assert s.is_parametric

    s = List2DSeries(xx, yy1, color_func=lambda x, y: 2 * x, use_cm=False)
    assert len(s.get_data()) == 2
    assert not s.is_parametric

    zz = np.linspace(-3, 3, 10)
    xx = np.cos(zz)
    yy = np.sin(zz)
    s = List3DSeries(xx, yy, zz, color_func=lambda x, y, z: 2 * x, use_cm=True)
    xxs, yys, zzs, col = s.get_data()
    assert np.allclose(xx, xxs)
    assert np.allclose(yy, yys)
    assert np.allclose(zz, zzs)
    assert np.allclose(2 * xx, col)
    assert s.is_parametric

    s = List3DSeries(xx, yy, zz, color_func=lambda x, y, z: 2 * x, use_cm=False)
    assert len(s.get_data()) == 3
    assert not s.is_parametric

    s = ComplexPointSeries([1 + 2 * I, 3 + 4 * I],
        color_func=lambda x, y: x * y, use_cm=True)
    xx, yy, col = s.get_data()
    assert s.is_parametric
    assert np.allclose(xx, [1, 3])
    assert np.allclose(yy, [2, 4])
    assert np.allclose(col, [2, 12])

    s = ComplexPointSeries([1 + 2 * I, 3 + 4 * I],
        color_func=lambda x, y: x * y, use_cm=False)
    assert len(s.get_data()) == 2
    assert not s.is_parametric

    s = LineOver1DRangeSeries(sin(x), (x, -5, 5), adaptive=False, n=10,
        color_func=lambda x: x)
    xx, yy, col = s.get_data()
    assert np.allclose(col, xx)
    s = LineOver1DRangeSeries(sin(x), (x, -5, 5), adaptive=False, n=10,
        color_func=lambda x, y: y)
    xx, yy, col = s.get_data()
    assert np.allclose(col, yy)

    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda t: t)
    xx, yy, col = s.get_data()
    assert (not np.allclose(xx, col)) and (not np.allclose(yy, col))
    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda x, y: x * y)
    xx, yy, col = s.get_data()
    assert np.allclose(col, xx * yy)
    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda x, y, t: x * y * t)
    xx, yy, col = s.get_data()
    assert np.allclose(col, xx * yy * np.linspace(0, 2*np.pi, 10))

    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda t: t)
    xx, yy, zz, col = s.get_data()
    assert (not np.allclose(xx, col)) and (not np.allclose(yy, col))
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda x, y, z: x * y * z)
    xx, yy, zz, col = s.get_data()
    assert np.allclose(col, xx * yy * zz)
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda x, y, z, t: x * y * z * t)
    xx, yy, zz, col = s.get_data()
    assert np.allclose(col, xx * yy * zz * np.linspace(0, 2*np.pi, 10))

    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        adaptive=False, n1=10, n2=10, color_func=lambda x: x)
    xx, yy, zz = s.get_data()
    col = s.eval_color_func(xx, yy, zz)
    assert np.allclose(xx, col)
    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        adaptive=False, n1=10, n2=10, color_func=lambda x, y: x * y)
    xx, yy, zz = s.get_data()
    col = s.eval_color_func(xx, yy, zz)
    assert np.allclose(xx * yy, col)
    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        adaptive=False, n1=10, n2=10, color_func=lambda x, y, z: x * y * z)
    xx, yy, zz = s.get_data()
    col = s.eval_color_func(xx, yy, zz)
    assert np.allclose(xx * yy * zz, col)

    s = ParametricSurfaceSeries(1, x, y, (x, 0, 1), (y, 0, 1), adaptive=False,
        n1=10, n2=10, color_func=lambda u:u)
    xx, yy, zz, uu, vv = s.get_data()
    col = s.eval_color_func(xx, yy, zz, uu, vv)
    assert np.allclose(uu, col)
    s = ParametricSurfaceSeries(1, x, y, (x, 0, 1), (y, 0, 1), adaptive=False,
        n1=10, n2=10, color_func=lambda u, v: u * v)
    xx, yy, zz, uu, vv = s.get_data()
    col = s.eval_color_func(xx, yy, zz, uu, vv)
    assert np.allclose(uu * vv, col)
    s = ParametricSurfaceSeries(1, x, y, (x, 0, 1), (y, 0, 1), adaptive=False,
        n1=10, n2=10, color_func=lambda x, y, z: x * y * z)
    xx, yy, zz, uu, vv = s.get_data()
    col = s.eval_color_func(xx, yy, zz, uu, vv)
    assert np.allclose(xx * yy * zz, col)
    s = ParametricSurfaceSeries(1, x, y, (x, 0, 1), (y, 0, 1), adaptive=False,
        n1=10, n2=10, color_func=lambda x, y, z, u, v: x * y * z * u * v)
    xx, yy, zz, uu, vv = s.get_data()
    col = s.eval_color_func(xx, yy, zz, uu, vv)
    assert np.allclose(xx * yy * zz * uu * vv, col)

    # Interactive Series
    s = List2DSeries([0, 1, 2, x], [x, 2, 3, 4],
        color_func=lambda x, y: 2 * x, params={x: 1}, use_cm=True)
    xx, yy, col = s.get_data()
    assert np.allclose(xx, [0, 1, 2, 1])
    assert np.allclose(yy, [1, 2, 3, 4])
    assert np.allclose(2 * xx, col)
    assert s.is_parametric and s.use_cm

    s = List2DSeries([0, 1, 2, x], [x, 2, 3, 4],
        color_func=lambda x, y: 2 * x, params={x: 1}, use_cm=False)
    assert len(s.get_data()) == 2
    assert not s.is_parametric

    s = List3DSeries([0, 1, 2, x], [x, 2, 3, 4], [1, 3, 2, x],
        color_func=lambda x, y, z: 2 * x, params={x: 1}, use_cm=True)
    xx, yy, zz, col = s.get_data()
    assert np.allclose(xx, [0, 1, 2, 1])
    assert np.allclose(yy, [1, 2, 3, 4])
    assert np.allclose(zz, [1, 3, 2, 1])
    assert np.allclose(2 * xx, col)
    assert s.is_parametric and s.use_cm

    s = List3DSeries([0, 1, 2, x], [x, 2, 3, 4], [1, 3, 2, x],
        color_func=lambda x, y, z: 2 * x, params={x: 1}, use_cm=False)
    assert len(s.get_data()) == 3
    assert not s.is_parametric

    s = ComplexPointSeries([1 + x * I, 1 + x + 4 * I],
        color_func=lambda x, y: x * y, params={x: 2}, use_cm=True)
    xx, yy, col = s.get_data()
    assert s.is_parametric and s.use_cm
    assert np.allclose(xx, [1, 3])
    assert np.allclose(yy, [2, 4])
    assert np.allclose(col, [2, 12])

    s = ComplexPointSeries([1 + x * I, 1 + x + 4 * I],
        color_func=lambda x, y: x * y, params={x: 2}, use_cm=False)
    assert len(s.get_data()) == 2
    assert not s.is_parametric


def test_color_func_scalar_val():
    # verify that eval_color_func returns a numpy array even when color_func
    # evaluates to a scalar value

    x, y = symbols("x, y")

    s = LineOver1DRangeSeries(sin(x), (x, -5, 5), adaptive=False, n=10,
        color_func=lambda x: 1)
    xx, yy, col = s.get_data()
    assert np.allclose(col, np.ones(xx.shape))

    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda t: 1)
    xx, yy, col = s.get_data()
    assert np.allclose(col, np.ones(xx.shape))

    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda t: 1)
    xx, yy, zz, col = s.get_data()
    assert np.allclose(col, np.ones(xx.shape))

    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        adaptive=False, n1=10, n2=10, color_func=lambda x: 1)
    xx, yy, zz = s.get_data()
    assert np.allclose(s.eval_color_func(xx), np.ones(xx.shape))

    s = ParametricSurfaceSeries(1, x, y, (x, 0, 1), (y, 0, 1), adaptive=False,
        n1=10, n2=10, color_func=lambda u: 1)
    xx, yy, zz, uu, vv = s.get_data()
    col = s.eval_color_func(xx, yy, zz, uu, vv)
    assert np.allclose(col, np.ones(xx.shape))


def test_line_surface_color():
    # verify the back-compatibility with the old sympy.plotting module.
    # By setting line_color or surface_color to be a callable, it will set
    # the color_func attribute.

    x, y, z = symbols("x, y, z")

    s = LineOver1DRangeSeries(sin(x), (x, -5, 5), adaptive=False, n=10,
        line_color=lambda x: x)
    assert (s.line_color is None) and callable(s.color_func)

    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=False, n=10, line_color=lambda t: t)
    assert (s.line_color is None) and callable(s.color_func)

    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        n1=10, n2=10, surface_color=lambda x: x)
    assert (s.surface_color is None) and callable(s.color_func)


def test_complex_adaptive_false():
    # verify that series with adaptive=False is evaluated with discretized
    # ranges of type complex.

    x, y, u = symbols("x y u")

    def do_test(data1, data2):
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert np.allclose(d1, d2)

    expr1 = sqrt(x) * exp(-x**2)
    expr2 = sqrt(u * x) * exp(-x**2)
    s1 = LineOver1DRangeSeries(im(expr1), (x, -5, 5), adaptive=False, n=10)
    s2 = LineOver1DRangeSeries(im(expr2), (x, -5, 5),
        adaptive=False, n=10, params={u: 1})
    data1 = s1.get_data()
    data2 = s2.get_data()

    do_test(data1, data2)
    assert (not np.allclose(data1[1], 0)) and (not np.allclose(data2[1], 0))

    s1 = Parametric2DLineSeries(re(expr1), im(expr1), (x, -pi, pi),
        adaptive=False, n=10)
    s2 = Parametric2DLineSeries(re(expr2), im(expr2), (x, -pi, pi),
        adaptive=False, n=10, params={u: 1})
    data1 = s1.get_data()
    data2 = s2.get_data()
    do_test(data1, data2)
    assert (not np.allclose(data1[1], 0)) and (not np.allclose(data2[1], 0))

    s1 = SurfaceOver2DRangeSeries(im(expr1), (x, -5, 5), (y, -10, 10),
        adaptive=False, n1=30, n2=3)
    s2 = SurfaceOver2DRangeSeries(im(expr2), (x, -5, 5), (y, -10, 10),
        adaptive=False, n1=30, n2=3, params={u: 1})
    data1 = s1.get_data()
    data2 = s2.get_data()
    do_test(data1, data2)
    assert (not np.allclose(data1[1], 0)) and (not np.allclose(data2[1], 0))


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_complex_adaptive_true():
    # verify that series with adaptive=True is evaluated with discretized
    # ranges of type complex.

    x, y, u = symbols("x y u")

    expr1 = sqrt(x) * exp(-x**2)
    expr2 = sqrt(u * x) * exp(-x**2)
    s1 = LineOver1DRangeSeries(im(expr1), (x, -5, 5), "",
        adaptive=True, adaptive_goal=0.1)
    data1 = s1.get_data()
    assert all(not np.allclose(d, 0) for d in data1)

    s1 = Parametric2DLineSeries(re(expr1), im(expr1), (x, -pi, pi),
        adaptive=True, adaptive_goal=0.1)
    data1 = s1.get_data()
    assert all(not np.allclose(d, 0) for d in data1)

    s1 = SurfaceOver2DRangeSeries(im(expr1), (x, -5, 5), (y, -2, 2),
        adaptive=True, adaptive_goal=0.1)
    data1 = s1.get_data()
    assert all(not np.allclose(d, 0) for d in data1)

    s1 = SurfaceOver2DRangeSeries(sqrt(x * y), (x, -5, 5), (y, -5, 5),
        adaptive=True, adaptive_goal=0.1)
    data1 = s1.get_data()
    assert np.isnan(data1[-1]).any()


def test_expr_is_lambda_function():
    # verify that when a numpy function is provided, the series will be able
    # to evaluate it. Also, label should be empty in order to prevent some
    # backend from crashing.

    f = lambda x: np.cos(x)
    s1 = LineOver1DRangeSeries(f, ("x", -5, 5),
        adaptive=True, adaptive_goal=0.1)
    d1 = s1.get_data()
    s2 = LineOver1DRangeSeries(f, ("x", -5, 5),
        adaptive=False, n=10)
    d2 = s2.get_data()
    assert s1.label == s2.label == ""

    fx = lambda x: np.cos(x)
    fy = lambda x: np.sin(x)
    s1 = Parametric2DLineSeries(fx, fy, ("x", 0, 2*pi),
        adaptive=True, adaptive_goal=0.1)
    d1 = s1.get_data()
    s2 = Parametric2DLineSeries(fx, fy, ("x", 0, 2*pi),
        adaptive=False, n=10)
    d2 = s2.get_data()
    assert s1.label == s2.label == ""

    fz = lambda x: x
    s1 = Parametric3DLineSeries(fx, fy, fz, ("x", 0, 2*pi),
        adaptive=True, adaptive_goal=0.1)
    d1 = s1.get_data()
    s2 = Parametric3DLineSeries(fx, fy, fz, ("x", 0, 2*pi),
        adaptive=False, n=10)
    d2 = s2.get_data()
    assert s1.label == s2.label == ""

    f = lambda x, y: np.cos(x**2 + y**2)
    s1 = SurfaceOver2DRangeSeries(f, ("a", -2, 2), ("b", -3, 3),
        adaptive=False, n1=10, n2=10)
    d1 = s1.get_data()
    s2 = ContourSeries(f, ("a", -2, 2), ("b", -3, 3),
        adaptive=False, n1=10, n2=10)
    d2 = s2.get_data()
    assert s1.label == s2.label == ""

    fx = lambda u, v: np.cos(u + v)
    fy = lambda u, v: np.sin(u - v)
    fz = lambda u, v: u * v
    s1 = ParametricSurfaceSeries(fx, fy, fz, ("u", 0, pi), ("v", 0, 2*pi),
        adaptive=False, n1=10, n2=10)
    d1 = s1.get_data()
    assert s1.label == ""

    raises(TypeError, lambda : ImplicitSeries(lambda t: np.sin(t),
        ("x", -5, 5), ("y", -6, 6)))

    f = lambda x, y, z: x**2 + y**3 - z**2
    s1 = Implicit3DSeries(f, ("x", -2, 2), ("y", -2, 2), ("z", -2, 2))
    d1 = s1.get_data()
    assert s1.label == ""

    raises(TypeError, lambda: List2DSeries(lambda t: t, lambda t: t))
    raises(TypeError, lambda: List3DSeries(lambda t: t, lambda t: t))
    raises(TypeError, lambda: ComplexPointSeries(lambda t: t, lambda t: t))
    raises(TypeError, lambda: ComplexPointSeries(lambda t: t, lambda t: t))
    raises(TypeError, lambda: ComplexSurfaceSeries(
        lambda z: (z ** 2 + 1) / (z ** 2 - 1), ("z", -3 - 4 * I, 3 + 4 * I)))

    s1 = ComplexDomainColoringSeries(
        lambda z: (z ** 2 + 1) / (z ** 2 - 1), ("z", -3 - 4 * I, 3 + 4 * I))
    d1 = s1.get_data()
    assert s1.label == ""


def test_plane_series():
    # verify that PlaneSeries produces the expected results and that it passes
    # through the provided point

    def compute_distance(xx, yy, zz, point):
        p1 = np.array([xx[0, 0], yy[0, 0], zz[0, 0]])
        p2 = np.array([xx[-1, 0], yy[-1, 0], zz[-1, 0]])
        p3 = np.array([xx[0, -1], yy[0, -1], zz[0, -1]])
        v1 = p3 - p1
        v2 = p2 - p1
        n = np.cross(v1, v2)
        n = n / np.sqrt(np.dot(n, n))
        pv = np.array(point) - p1
        distance = np.dot(n, pv)
        return distance

    x, y, z = symbols("x:z")

    # plane parallel to YZ plane
    p = (0, 0, 0)
    s = PlaneSeries(Plane(p, (1, 0, 0)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    xx, yy, zz = s.get_data()
    assert np.allclose(xx, 0) and (not np.allclose(yy, 0)) and (not np.allclose(zz, 0))
    assert np.isclose(compute_distance(xx, yy, zz, p), 0)

    p = (-2, 4, 6)
    s = PlaneSeries(Plane(p, (1, 0, 0)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    xx, yy, zz = s.get_data()
    assert np.allclose(xx, -2) and (not np.allclose(yy, 0)) and (not np.allclose(zz, 0))
    assert np.isclose(compute_distance(xx, yy, zz, p), 0)

    # plane parallel to XZ plane
    p = (0, 0, 0)
    s = PlaneSeries(Plane(p, (0, 1, 0)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    xx, yy, zz = s.get_data()
    assert np.allclose(yy, 0) and (not np.allclose(xx, 0)) and (not np.allclose(zz, 0))
    assert np.isclose(compute_distance(xx, yy, zz, p), 0)

    p = (-2, 4, 6)
    s = PlaneSeries(Plane(p, (0, 1, 0)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    xx, yy, zz = s.get_data()
    assert np.allclose(yy, 4) and (not np.allclose(xx, 0)) and (not np.allclose(zz, 0))
    assert np.isclose(compute_distance(xx, yy, zz, p), 0)

    # plane parallel to XY plane
    p = (0, 0, 0)
    s = PlaneSeries(Plane(p, (0, 0, 1)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    xx, yy, zz = s.get_data()
    assert np.allclose(zz, 0) and (not np.allclose(yy, 0)) and (not np.allclose(xx, 0))
    assert np.isclose(compute_distance(xx, yy, zz, p), 0)

    p = (-2, 4, 6)
    s = PlaneSeries(Plane(p, (0, 0, 1)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    xx, yy, zz = s.get_data()
    assert np.allclose(zz, 6) and (not np.allclose(yy, 0)) and (not np.allclose(xx, 0))
    assert np.isclose(compute_distance(xx, yy, zz, p), 0)

    # generic vertical plane
    p = (0, 0, 0)
    s = PlaneSeries(Plane(p, (1, 1, 0)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    xx, yy, zz = s.get_data()
    assert all(not np.allclose(t, 0) for t in [xx, yy, zz])
    assert all(np.allclose(zz[i, :], zz[i, 0]) for i in range(zz.shape[0]))
    assert np.isclose(compute_distance(xx, yy, zz, p), 0)

    p = (-2, 4, 6)
    s = PlaneSeries(Plane(p, (1, 1, 0)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    xx, yy, zz = s.get_data()
    assert all(not np.allclose(t, 0) for t in [xx, yy, zz])
    assert all(np.allclose(zz[i, :], zz[i, 0]) for i in range(zz.shape[0]))
    assert np.isclose(compute_distance(xx, yy, zz, p), 0)

    # generic plane
    p = (0, 0, 0)
    s = PlaneSeries(Plane(p, (1, 1, 1)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    s._use_nan = False
    xx, yy, zz = s.get_data()
    assert all(not np.allclose(t, 0) for t in [xx, yy, zz])
    assert np.isclose(compute_distance(xx, yy, zz, p), 0)

    p = (-2, 4, 6)
    s = PlaneSeries(Plane(p, (-1, -1, 1)),
        (x, -5, 4), (y, -3, 2), (z, -6, 7))
    s._use_nan = False
    xx, yy, zz = s.get_data()
    assert all(not np.allclose(t, 0) for t in [xx, yy, zz])
    assert np.isclose(compute_distance(xx, yy, zz, p), 0)


def test_show_in_legend_lines():
    # verify that lines series correctly set the show_in_legend attribute
    x, u = symbols("x, u")

    s = LineOver1DRangeSeries(cos(x), (x, -2, 2), "test", show_in_legend=True)
    assert s.show_in_legend
    s = LineOver1DRangeSeries(cos(x), (x, -2, 2), "test", show_in_legend=False)
    assert not s.show_in_legend

    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 1), "test",
        show_in_legend=True)
    assert s.show_in_legend
    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 1), "test",
        show_in_legend=False)
    assert not s.show_in_legend

    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 1), "test",
        show_in_legend=True)
    assert s.show_in_legend
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 1), "test",
        show_in_legend=False)
    assert not s.show_in_legend


def test_particular_case_1():
    # Verify that symbolic expressions and numerical lambda functions are
    # evaluated with the same algorithm. In particular, uniform evaluation
    # is going to use np.vectorize, which correctly evaluates the following
    # mathematical function.
    def do_test(a, b):
        d1 = a.get_data()
        d2 = b.get_data()
        for t, v in zip(d1, d2):
            assert np.allclose(t, v)

    n = symbols("n")
    a = S(2) / 3
    epsilon = 0.01
    xn = (n**3 + n**2)**(S(1)/3) - (n**3 - n**2)**(S(1)/3)
    expr = Abs(xn - a) - epsilon
    math_func = lambdify([n], expr)
    s1 = LineOver1DRangeSeries(expr, (n, -10, 10), "",
        adaptive=True, adaptive_goal=0.2)
    s2 = LineOver1DRangeSeries(math_func, ("n", -10, 10), "",
        adaptive=True, adaptive_goal=0.2)
    do_test(s1, s2)

    s3 = LineOver1DRangeSeries(expr, (n, -10, 10), "",
        adaptive=False, n=10)
    s4 = LineOver1DRangeSeries(math_func, ("n", -10, 10), "",
        adaptive=False, n=10)
    do_test(s3, s4)


def test_vector_series_normalize():
    # verify that vector series exposes the normalize attribute

    x, y, z, u = symbols("x, y, z, u")

    # default behaviour
    s = Vector2DSeries(-sin(y), cos(x), (x, -2, 2), (y, -2, 2))
    assert not s.normalize
    s = Vector2DSeries(-sin(y), cos(x), (x, -2, 2), (y, -2, 2), normalize=False)
    assert not s.normalize
    s = Vector2DSeries(-sin(y), cos(x), (x, -2, 2), (y, -2, 2), normalize=True)
    assert s.normalize

    s = Vector3DSeries(-sin(y), cos(x), z, (x, -2, 2), (y, -2, 2), (z, -2, 2))
    assert not s.normalize
    s = Vector3DSeries(-sin(y), cos(x), z, (x, -2, 2), (y, -2, 2), (z, -2, 2),
        normalize=False)
    assert not s.normalize
    s = Vector3DSeries(-sin(y), cos(x), z, (x, -2, 2), (y, -2, 2), (z, -2, 2),
        normalize=True)
    assert s.normalize

    s = SliceVector3DSeries(Plane((-1, 0, 0), (1, 0, 0)),
        z, -y, x, (x, -3, 3), (y, -2, 2), (z, -1, 1))
    assert not s.normalize
    s = SliceVector3DSeries(Plane((-1, 0, 0), (1, 0, 0)),
        z, -y, x, (x, -3, 3), (y, -2, 2), (z, -1, 1), normalize=False)
    assert not s.normalize
    s = SliceVector3DSeries(Plane((-1, 0, 0), (1, 0, 0)),
        z, -y, x, (x, -3, 3), (y, -2, 2), (z, -1, 1), normalize=True)
    assert s.normalize


def test_complex_params_number_eval():
    # The main expression contains terms like sqrt(xi - 1), with
    # parameter (0 <= xi <= 1).
    # There shouldn't be any NaN values on the output.
    xi, wn, x0, v0, t = symbols("xi, omega_n, x0, v0, t")
    x = Function("x")(t)
    eq = x.diff(t, 2) + 2 * xi * wn * x.diff(t) + wn**2 * x
    sol = dsolve(eq, x, ics={x.subs(t, 0): x0, x.diff(t).subs(t, 0): v0})
    params = {
        wn: 0.5,
        xi: 0.25,
        x0: 0.45,
        v0: 0.0
    }
    s = LineOver1DRangeSeries(sol.rhs, (t, 0, 100), adaptive=False, n=5,
        params=params)
    x, y = s.get_data()
    assert not np.isnan(x).any()
    assert not np.isnan(y).any()


    # Fourier Series of a sawtooth wave
    # The main expression contains a Sum with a symbolic upper range.
    # The lambdified code looks like:
    #       sum(blablabla for for n in range(1, m+1))
    # But range requires integer numbers, whereas per above example, the series
    # casts parameters to complex. Verify that the series is able to detect
    # upper bounds in summations and cast it to int in order to get successfull
    # evaluation
    x, T, n, m = symbols("x, T, n, m")
    sawtooth = frac(x / T)
    fs = S(1) / 2 - (1 / pi) * Sum(sin(2 * n * pi * x / T) / n, (n, 1, m))
    params = {
        T: 4.5,
        m: 5
    }
    s = LineOver1DRangeSeries(fs, (x, 0, 10), adaptive=False, n=5,
        params=params)
    x, y = s.get_data()
    assert not np.isnan(x).any()
    assert not np.isnan(y).any()


def test_complex_range_line_plot_1():
    # verify that univariate functions are evaluated with a complex
    # data range (with zero imaginary part). There shouln't be any
    # NaN value in the output.

    x, u = symbols("x, u")
    expr1 = im(sqrt(x) * exp(-x**2))
    expr2 = im(sqrt(u * x) * exp(-x**2))
    s1 = LineOver1DRangeSeries(expr1, (x, -10, 10), adaptive=True,
        adaptive_goal=0.1)
    s2 = LineOver1DRangeSeries(expr1, (x, -10, 10), adaptive=False, n=30)
    s3 = LineOver1DRangeSeries(expr2, (x, -10, 10), adaptive=False, n=30,
        params={u: 1})
    data1 = s1.get_data()
    data2 = s2.get_data()
    data3 = s3.get_data()

    assert not np.isnan(data1[1]).any()
    assert not np.isnan(data2[1]).any()
    assert not np.isnan(data3[1]).any()
    assert np.allclose(data2[0], data3[0]) and np.allclose(data2[1], data3[1])


def test_complex_range_line_plot_2():
    # verify that univariate functions are evaluated with a complex
    # data range (with non-zero imaginary part). There shouln't be any
    # NaN value in the output.

    x, u = symbols("x, u")

    # adaptive and uniform meshing should produce the same data.
    # because of the adaptive nature, just compare the first and last points
    # of both series.
    s1 = LineOver1DRangeSeries(abs(sqrt(x)), (x, -5-2j, 5-2j), adaptive=True)
    s2 = LineOver1DRangeSeries(abs(sqrt(x)), (x, -5-2j, 5-2j), adaptive=False,
        n=10)
    d1 = s1.get_data()
    d2 = s2.get_data()
    xx1 = [d1[0][0], d1[0][-1]]
    xx2 = [d2[0][0], d2[0][-1]]
    yy1 = [d1[1][0], d1[1][-1]]
    yy2 = [d2[1][0], d2[1][-1]]
    assert np.allclose(xx1, xx2)
    assert np.allclose(yy1, yy2)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_force_real_eval():
    # verify that force_real_eval=True produces inconsistent results when
    # compared with evaluation of complex domain.
    x = symbols("x")

    expr = im(sqrt(x) * exp(-x**2))
    s1 = LineOver1DRangeSeries(expr, (x, -10, 10), adaptive=False, n=10,
        force_real_eval=False)
    s2 = LineOver1DRangeSeries(expr, (x, -10, 10), adaptive=False, n=10,
        force_real_eval=True)
    d1 = s1.get_data()
    d2 = s2.get_data()
    assert not np.allclose(d1[1], 0)
    assert np.allclose(d2[1], 0)


def test_contour_series_show_clabels():
    # verify that a contour series has the abiliy to set the visibility of
    # labels to contour lines

    x, y = symbols("x, y")
    s = ContourSeries(cos(x*y), (x, -2, 2), (y, -2, 2))
    assert s.show_clabels

    s = ContourSeries(cos(x*y), (x, -2, 2), (y, -2, 2), clabels=True)
    assert s.show_clabels

    s = ContourSeries(cos(x*y), (x, -2, 2), (y, -2, 2), clabels=False)
    assert not s.show_clabels


def test_LineOver1DRangeSeries_complex_range():
    # verify that LineOver1DRangeSeries can accept a complex range
    # if the imaginary part of the start and end values are the same

    x = symbols("x")

    s = LineOver1DRangeSeries(sqrt(x), (x, -10, 10))
    s = LineOver1DRangeSeries(sqrt(x), (x, -10-2j, 10-2j))
    raises(ValueError,
        lambda : LineOver1DRangeSeries(sqrt(x), (x, -10-2j, 10+2j)))


def test_symbolic_plotting_ranges():
    # verify that data series can use symbolic plotting ranges

    x, y, z, a, b = symbols("x, y, z, a, b")

    def do_test(s1, s2, new_params):
        d1 = s1.get_data()
        d2 = s2.get_data()
        for u, v in zip(d1, d2):
            assert np.allclose(u, v)
        s2.params = new_params
        d2 = s2.get_data()
        for u, v in zip(d1, d2):
            assert not np.allclose(u, v)

    s1 = LineOver1DRangeSeries(sin(x), (x, 0, 1), adaptive=False, n=10)
    s2 = LineOver1DRangeSeries(sin(x), (x, a, b), params={a: 0, b: 1},
        adaptive=False, n=10)
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # missing a parameter
    raises(ValueError,
        lambda : LineOver1DRangeSeries(sin(x), (x, a, b), params={a: 1}, n=10))

    s1 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 1), adaptive=False, n=10)
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, a, b), params={a: 0, b: 1},
        adaptive=False, n=10)
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # missing a parameter
    raises(ValueError,
        lambda : Parametric2DLineSeries(cos(x), sin(x), (x, a, b),
            params={a: 0}, adaptive=False, n=10))

    s1 = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 1),
        adaptive=False, n=10)
    s2 = Parametric3DLineSeries(cos(x), sin(x), x, (x, a, b),
        params={a: 0, b: 1}, adaptive=False, n=10)
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # missing a parameter
    raises(ValueError,
        lambda : Parametric3DLineSeries(cos(x), sin(x), x, (x, a, b),
            params={a: 0}, adaptive=False, n=10))

    s1 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        adaptive=False, n1=5, n2=5)
    s2 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -pi * a, pi * a),
        (y, -pi * b, pi * b), params={a: 1, b: 1},
        adaptive=False, n1=5, n2=5)
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # missing a parameter
    raises(ValueError,
        lambda : SurfaceOver2DRangeSeries(cos(x**2 + y**2),
        (x, -pi * a, pi * a), (y, -pi * b, pi * b), params={a: 1},
        adaptive=False, n1=5, n2=5))
    # one range symbol is included into another range's minimum or maximum val
    raises(ValueError,
        lambda : SurfaceOver2DRangeSeries(cos(x**2 + y**2),
        (x, -pi * a + y, pi * a), (y, -pi * b, pi * b), params={a: 1},
        adaptive=False, n1=5, n2=5))

    s1 = ParametricSurfaceSeries(
        cos(x - y), sin(x + y), x - y, (x, -2, 2), (y, -2, 2), n1=5, n2=5)
    s2 = ParametricSurfaceSeries(
        cos(x - y), sin(x + y), x - y, (x, -2 * a, 2), (y, -2, 2 * b),
        params={a: 1, b: 1}, n1=5, n2=5)
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # missing a parameter
    raises(ValueError,
        lambda : ParametricSurfaceSeries(
        cos(x - y), sin(x + y), x - y, (x, -2 * a, 2), (y, -2, 2 * b),
        params={a: 1}, n1=5, n2=5))

    s1 = ComplexSurfaceSeries(sin(x), (x, -2-2j, 2+2j), n1=5, n2=5)
    s2 = ComplexSurfaceSeries(sin(x), (x, -2*a-2j, 2+2j*b), n1=5, n2=5,
        params={a: 1, b: 1})
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # missing a parameter
    raises(ValueError,
        lambda : ComplexSurfaceSeries(sin(x), (x, -2*a-2j, 2+2j*b), n1=5, n2=5,
        params={a: 1}))

    s1 = ComplexDomainColoringSeries(sin(x), (x, -2-2j, 2+2j), n1=5, n2=5)
    s2 = ComplexDomainColoringSeries(sin(x), (x, -2*a-2j, 2+2j*b), n1=5, n2=5,
        params={a: 1, b: 1})
    d1 = s1.get_data()
    d2 = s2.get_data()
    for u, v in zip(d1, d2):
        assert np.allclose(u, v)
    s2.params = {a: 0.5, b: 1.5}
    d2 = s2.get_data()
    # NOTE: the last one is the colorbar, which is equal in both cases
    for u, v in zip(d1[:-1], d2[:-1]):
        assert not np.allclose(u, v)

    # missing a parameter
    raises(ValueError,
        lambda : ComplexDomainColoringSeries(sin(x), (x, -2*a-2j, 2+2j*b),
            n1=5, n2=5, params={a: 1}))

    s1 = Vector2DSeries(-cos(y), sin(x), (x, -5, 4), (y, -3, 2), n1=4, n2=4)
    s2 = Vector2DSeries(-cos(y), sin(x), (x, -5*a, 4*b), (y, -3*b, 2*a),
        n1=4, n2=4, params={a: 1, b: 1})
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # missing a parameter
    raises(ValueError,
        lambda : Vector2DSeries(-cos(y), sin(x), (x, -5*a, 4*b), (y, -3*b, 2*a),
        n1=4, n2=4, params={a: 1}))

    s1 = Vector3DSeries(-cos(y), sin(x), sin(z),
        (x, -5, 4), (y, -3, 2), (z, -6, 7), n1=4, n2=4)
    s2 = Vector3DSeries(-cos(y), sin(x), sin(z),
        (x, -5*a, 4*b), (y, -3*b, 2*a), (z, -6*a, 7*b),
        n1=4, n2=4, params={a: 1, b: 1})
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # missing a parameter
    raises(ValueError,
        lambda : Vector3DSeries(-cos(y), sin(x), sin(z),
        (x, -5*a, 4*b), (y, -3*b, 2*a), (z, -6*a, 7*b),
        n1=4, n2=4, params={a: 1}))


def test_color_func_expression():
    # verify that color_func is able to deal with instances of Expr: they will
    # be lambdified with the same signature used for the main expression.

    # NOTE: considering how complicated the following expression is,
    # in this case it is easier to use plot3d_spherical to generate the
    # data series.
    phi, theta = symbols("phi, theta", real=True)
    r = re(Ynm(3, 1, theta, phi).expand(func=True).rewrite(sin).expand())
    p = plot3d_spherical(
        abs(r), (theta, 0, pi), (phi, 0, 2*pi),
        color_func=r, use_cm=True, n=20,
        force_real_eval=True, grid=False, show=False)
    d = p[0].get_data()
    # the following statement should not raise errors
    colors = p[0].eval_color_func(*d)
    assert callable(p[0].color_func)

    x, y = symbols("x, y")
    s1 = LineOver1DRangeSeries(cos(x), (x, -10, 10), color_func=sin(x),
        adaptive=False, n=10)
    s2 = LineOver1DRangeSeries(cos(x), (x, -10, 10),
        color_func=lambda x: np.cos(x), adaptive=False, n=10)
    assert all(isinstance(t, ColoredLineOver1DRangeSeries) for t in [s1, s2])
    # the following statement should not raise errors
    d1 = s1.get_data()
    assert callable(s1.color_func)
    d2 = s2.get_data()
    assert not np.allclose(d1[-1], d2[-1])

    s1 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        color_func=sin(x), adaptive=False, n=10, use_cm=True)
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        color_func=lambda x: np.cos(x), adaptive=False, n=10, use_cm=True)
    # the following statement should not raise errors
    d1 = s1.get_data()
    assert callable(s1.color_func)
    d2 = s2.get_data()
    assert not np.allclose(d1[-1], d2[-1])

    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        color_func=sin(x**2 + y**2), adaptive=False, n1=5, n2=5)
    # the following statement should not raise errors
    d = s.get_data()
    assert callable(s.color_func)

    xx = [1, 2, 3, 4, 5]
    yy = [1, 2, 3, 4, 5]
    zz = [1, 2, 3, 4, 5]
    raises(TypeError,
        lambda : List2DSeries(xx, yy, use_cm=True, color_func=sin(x)))
    raises(TypeError,
        lambda : List3DSeries(xx, yy, zz, use_cm=True, color_func=sin(x)))


def test_2d_complex_domain_coloring_schemes():
    # verify that domain coloring schemes produce different data from one
    # another

    z = symbols("z")
    expr = (z - 1) / (z**2 + z + 1)
    n1 = n2 = 10
    colorings = "abcdefghijklmno"
    series = [
        ComplexDomainColoringSeries(
            expr, (z, -3 - 3 * I, 3 + 3 * I), "",
            coloring=c, n1=n1, n2=n2) for c in colorings]
    imgs = [s.get_data()[-2] for s in series]
    for i in range(len(series) - 1):
        for j in range(i + 1, len(series)):
            assert not np.allclose(imgs[i], imgs[j])


def test_2d_complex_domain_coloring_cmap_blevel():
    # verify that complex domain coloring is applying colormap and black level

    z = symbols("z")
    expr = (z - 1) / (z**2 + z + 1)
    n1 = n2 = 10
    s1 = ComplexDomainColoringSeries(
        expr, (z, -3 - 3 * I, 3 + 3 * I), "",
        coloring="b", cmap=None, blevel=0.75, n1=n1, n2=n2)
    s2 = ComplexDomainColoringSeries(
        expr, (z, -3 - 3 * I, 3 + 3 * I), "",
        coloring="b", cmap=None, blevel=0.85, n1=n1, n2=n2)
    s3 = ComplexDomainColoringSeries(
        expr, (z, -3 - 3 * I, 3 + 3 * I), "",
        coloring="b", cmap="twilight", blevel=0.75, n1=n1, n2=n2)
    s4 = ComplexDomainColoringSeries(
        expr, (z, -3 - 3 * I, 3 + 3 * I), "",
        coloring="b", cmap="twilight", blevel=0.85, n1=n1, n2=n2)
    d1, d2, d3, d4 = [t.get_data() for t in [s1, s2, s3, s4]]
    assert not np.allclose(d1[-2], d2[-2])
    assert not np.allclose(d1[-2], d3[-2])
    assert not np.allclose(d1[-2], d4[-2])
    assert not np.allclose(d2[-2], d3[-2])
    assert not np.allclose(d2[-2], d4[-2])
    assert not np.allclose(d3[-2], d4[-2])


def test_2d_complex_domain_coloring_zero_infinity():
    # verify that the domain coloring image is different for the same function
    # when evaluate around zero and around infinity, and that Riemann mask
    # works.

    z = symbols("z")
    expr = (z - 1) / (z**2 + z + 1)
    n1 = n2 = 10
    s1 = ComplexDomainColoringSeries(
        expr, (z, -1.25 - 1.25 * I, 1.25 + 1.25 * I), "",
        coloring="b", at_infinity=False, riemann_mask=False, n1=n1, n2=n2)
    s2 = ComplexDomainColoringSeries(
        expr, (z, -1.25 - 1.25 * I, 1.25 + 1.25 * I), "",
        coloring="b", at_infinity=False, riemann_mask=True, n1=n1, n2=n2)
    s3 = ComplexDomainColoringSeries(
        expr, (z, -1.25 - 1.25 * I, 1.25 + 1.25 * I), "",
        coloring="b", at_infinity=True, riemann_mask=False, n1=n1, n2=n2)
    s4 = ComplexDomainColoringSeries(
        expr, (z, -1.25 - 1.25 * I, 1.25 + 1.25 * I), "",
        coloring="b", at_infinity=True, riemann_mask=True, n1=n1, n2=n2)
    d1, d2, d3, d4 = [t.get_data() for t in [s1, s2, s3, s4]]
    assert not np.allclose(d1[-2], d2[-2])
    assert not np.allclose(d1[-2], d3[-2])
    assert not np.allclose(d1[-2], d4[-2])
    assert not np.allclose(d2[-2], d3[-2])
    assert not np.allclose(d2[-2], d4[-2])
    assert not np.allclose(d3[-2], d4[-2])


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_riemann_sphere_series():
    # verify that it correctly instatiates and doesn't raise error when
    # producing data.

    z = symbols("z")
    expr = (z - 1) / (z**2 + z + 1)
    t, p = symbols("theta phi")
    s = RiemannSphereSeries(expr, (t, 0, pi), (p, 0, 2*pi), n1=5, n2=15)
    d = s.get_data()
    assert d[0].shape == (5, 15)


def test_hvline_series():
    x, y = symbols("x y")

    s = HVLineSeries(4.5, True)
    assert s.is_horizontal
    assert not s.is_interactive
    loc = s.get_data()
    assert isinstance(loc, float)
    assert np.isclose(loc, 4.5)
    assert s.get_label(False) == ""

    s = HVLineSeries(4.5, False, "test")
    assert not s.is_horizontal
    assert not s.is_interactive
    loc = s.get_data()
    assert isinstance(loc, float)
    assert np.isclose(loc, 4.5)
    assert s.get_label(False) == "test"

    s = HVLineSeries(x + y, True, params={x: 2, y: 3.5})
    assert s.is_horizontal
    assert s.is_interactive
    loc = s.get_data()
    assert isinstance(loc, float)
    assert np.isclose(loc, 5.5)


def test_complex_surface_contour():
    # verify that ComplexSurfaceSeries exposes the proper attributes
    # to be rendered as a contour.

    z = symbols("z")
    s = ComplexSurfaceSeries(sqrt(z), (z, -5 - 5j, 5 + 5j))
    assert s.is_filled
    assert s.show_clabels

    s = ComplexSurfaceSeries(sqrt(z), (z, -5 - 5j, 5 + 5j), is_filled=False,
        clabels=False)
    assert not s.is_filled
    assert not s.show_clabels


def test_exclude_points():
    # verify that exclude works as expected

    x = symbols("x")

    expr = (floor(x) + S.Half) / (1 - (x - S.Half)**2)
    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some of",
        ):
        s = LineOver1DRangeSeries(expr, (x, -3.5, 3.5), adaptive=False, n=100,
        exclude=list(range(-3, 4)))
    xx, yy = s.get_data()
    assert not np.isnan(xx).any()
    assert np.count_nonzero(np.isnan(yy)) == 7
    assert len(xx) > 100

    e1 = log(floor(x)) * cos(x)
    e2 = log(floor(x)) * sin(x)
    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some of",
        ):
        s = Parametric2DLineSeries(e1, e2, (x, 1, 12), adaptive=False, n=100,
            exclude=list(range(1, 13)))
    xx, yy, pp = s.get_data()
    assert not np.isnan(pp).any()
    assert np.count_nonzero(np.isnan(xx)) == 11
    assert np.count_nonzero(np.isnan(yy)) == 11
    assert len(xx) > 100


def test_unwrap():
    # verify that unwrap works as expected

    x, y = symbols("x, y")
    expr = 1 / (x**3 + 2*x**2 + x)
    expr = arg(expr.subs(x, I*y*2*pi))
    s1 = LineOver1DRangeSeries(expr, (y, 1e-05, 1e05), xscale="log",
        adaptive=False, n=10, unwrap=False)
    s2 = LineOver1DRangeSeries(expr, (y, 1e-05, 1e05), xscale="log",
        adaptive=False, n=10, unwrap=True)
    s3 = LineOver1DRangeSeries(expr, (y, 1e-05, 1e05), xscale="log",
        adaptive=False, n=10, unwrap={"period": 4})
    x1, y1 = s1.get_data()
    x2, y2 = s2.get_data()
    x3, y3 = s3.get_data()
    assert np.allclose(x1, x2)
    # there must not be nan values in the results of these evaluations
    assert all(not np.isnan(t).any() for t in [y1, y2, y3])
    assert not np.allclose(y1, y2)
    assert not np.allclose(y1, y3)
    assert not np.allclose(y2, y3)
