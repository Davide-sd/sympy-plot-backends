import os
import pytest
from pytest import raises
from spb.defaults import set_defaults, cfg
from spb import *
from spb.interactive import InteractivePlot
from spb.series import (
    LineOver1DRangeSeries, List2DSeries, ContourSeries,
    Vector2DSeries, ParametricSurfaceSeries, Parametric3DLineSeries,
    ParametricSurfaceInteractiveSeries, Parametric3DLineInteractiveSeries,
    Parametric2DLineSeries
)
from spb.backends.matplotlib import MB, unset_show
from sympy import (
    symbols, Piecewise, sin, cos, tan, re, Abs, sqrt, real_root,
    Heaviside, exp, log, LambertW, exp_polar, meijerg,
    And, Or, Eq, Ne, Interval, Sum, oo, I, pi, S,
    sympify, Integral, Circle, gamma
)
from sympy.vector import CoordSys3D
from sympy.testing.pytest import skip, warns
from sympy.external import import_module
from sympy.testing.tmpfiles import TmpFileManager
from tempfile import TemporaryDirectory, NamedTemporaryFile, mkdtemp

# use MatplotlibBackend for the tests whenever the backend is not specified
cfg["backend_2D"] = "matplotlib"
cfg["backend_3D"] = "matplotlib"
set_defaults(cfg)

np = import_module('numpy', catch=(RuntimeError,))

unset_show()

# NOTE:
#
# These tests are meant to verify that the plotting functions generate the
# correct data series.
# Also, legacy tests from the old sympy.plotting module are included.
#
# If your issue is related to the generation of numerical data from a
# particular data series, consider adding tests to test_series.py.
# If your issue il related to the preprocessing and generation of a
# Vector series or a Complex Series, consider adding tests to
# test_build_series.
# If your issue is related to a particular keyword affecting a backend
# behaviour, consider adding tests to test_backends.py
#


def test_plot_parametric_region():
    # verify that plot_parametric_region creates the correct data series

    u, v, x, y = symbols("u, v, x, y")

    # single parametric region
    p = plot_parametric_region(u * cos(v), u * sin(v),
        (u, 1, 2), (v, 0, 2*pi/3), {"color": "k"}, show=False,
        n1=3, n2=5, n=10)
    assert len(p.series) == 8
    assert all(isinstance(s, Parametric2DLineSeries) for s in p.series)
    assert all((s.var, s.start, s.end) == (v, 0, float(2*pi/3)) for s in p.series[:3])
    assert all((s.var, s.start, s.end) == (u, 1, 2) for s in p.series[3:])
    assert all(s.n == 10 for s in p.series)
    assert all(s.rendering_kw == {"color": "k"} for s in p.series)

    p = plot_parametric_region(u * cos(v), u * sin(v),
        (u, 1, 2), (v, 0, 2*pi/3), {"color": "k"},
        rkw_u={"color": "r"}, rkw_v={"color": "b"},
        show=False, n1=3, n2=5, n=10)
    assert all(s.rendering_kw == {"color": "r"} for s in p.series[:3])
    assert all(s.rendering_kw == {"color": "b"} for s in p.series[3:])

    # multiple parametric regions with the same ranges
    p = plot_parametric_region(
        (u * cos(v), u * sin(v)),
        (2 * u * cos(v), 2 * u * sin(v)),
        (u, 1, 2), (v, 0, 2*pi/3), {"color": "k"}, show=False,
        n1=3, n2=5, n=10)
    assert len(p.series) == 16
    assert sum((s.var, s.start, s.end) == (u, 1, 2) for s in p.series) == 10
    assert sum((s.var, s.start, s.end) == (v, 0, float(2*pi/3)) for s in p.series) == 6

    # multiple parametric regions each one with its own ranges
    p = plot_parametric_region(
        (u * cos(v), u * sin(v), (u, 1, 2), (v, 0, 2*pi/3)),
        (2 * x * cos(y), 2 * x * sin(y), (x, 0, 1), (y, 0, pi)),
        show=False, n1=3, n2=5, n=10)
    assert len(p.series) == 16
    assert sum((s.var, s.start, s.end) == (u, 1, 2) for s in p.series) == 5
    assert sum((s.var, s.start, s.end) == (v, 0, float(2*pi/3)) for s in p.series) == 3
    assert sum((s.var, s.start, s.end) == (x, 0, 1) for s in p.series) == 5
    assert sum((s.var, s.start, s.end) == (y, 0, float(pi)) for s in p.series) == 3

    # parametric interactive plot
    p = lambda: plot_parametric_region(u * cos(x * v), u * sin(v),
        (u, 1, 2), (v, 0, 2*pi/3), show=False, n1=3, n2=5, n=10,
        params={x: (1, 0, 2)})
    raises(NotImplementedError, p)



def test_plot3d_revolution():
    # plot3d_revolution is going to call plot3d_parametric_surface and
    # plot3d_parametric_line: let's check that the data series are correct.

    t, phi = symbols("t, phi")

    # test that azimuthal angle is set correctly
    p = plot3d_revolution(cos(t), (t, 0, pi), show=False, n=5)
    assert len(p.series) == 1
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert (p[0].var_u, p[0].start_u, p[0].end_u) == (t, 0, float(pi))
    assert (p[0].var_v, p[0].start_v, p[0].end_v) == (phi, 0, float(2*pi))

    p = plot3d_revolution(cos(t), (t, 0, pi), (phi, 0, pi/2), show=False, n=5)
    assert len(p.series) == 1
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert (p[0].var_u, p[0].start_u, p[0].end_u) == (t, 0, float(pi))
    assert (p[0].var_v, p[0].start_v, p[0].end_v) == (phi, 0, float(pi / 2))

    # by setting parallel_axis it produces different expressions/data
    p1 = plot3d_revolution(cos(t), (t, 0, pi), show=False, n=5,
        show_curve=False, parallel_axis="z")
    assert len(p1.series) == 1
    assert isinstance(p1[0], ParametricSurfaceSeries)
    xx1, yy1, zz1, tt1, pp1 = p1[0].get_data()
    assert np.allclose([xx1.min(), xx1.max()], [-np.pi, np.pi])
    assert np.allclose([yy1.min(), yy1.max()], [-np.pi, np.pi])
    assert np.allclose([zz1.min(), zz1.max()], [-1, 1])
    assert np.allclose([tt1.min(), tt1.max()], [0, np.pi])
    assert np.allclose([pp1.min(), pp1.max()], [0, 2 * np.pi])

    p2 = plot3d_revolution(cos(t), (t, 0, pi), show=False, n=5,
        show_curve=False, parallel_axis="x")
    xx2, yy2, zz2, tt2, pp2 = p2[0].get_data()
    assert p2[0].expr != p1[0].expr

    p3 = plot3d_revolution(cos(t), (t, 0, pi), show=False, n=5,
        show_curve=False, parallel_axis="y")
    xx3, yy3, zz3, tt3, pp3 = p3[0].get_data()
    assert p3[0].expr != p1[0].expr
    assert p3[0].expr != p2[0].expr

    # by setting axis it produces different data
    p4 = plot3d_revolution(cos(t), (t, 0, pi), show=False, n=5,
        show_curve=False, parallel_axis="z", axis=(2, 1))
    xx4, yy4, zz4, tt4, pp4 = p4[0].get_data()
    assert not np.allclose(xx4, xx1)
    assert not np.allclose(yy4, yy1)
    assert np.allclose(zz4, zz1)
    assert np.allclose(tt4, tt1)
    assert np.allclose(pp4, pp1)

    p5 = plot3d_revolution(cos(t), (t, 0, pi), show=False, n=5,
        show_curve=False, parallel_axis="x", axis=(2, 1))
    xx5, yy5, zz5, tt5, pp5 = p5[0].get_data()
    assert np.allclose(xx5, xx2)
    assert not np.allclose(yy5, yy2)
    assert not np.allclose(zz5, zz2)
    assert np.allclose(tt5, tt2)
    assert np.allclose(pp5, pp2)

    p6 = plot3d_revolution(cos(t), (t, 0, pi), show=False, n=5,
        show_curve=False, parallel_axis="y", axis=(2, 1))
    xx6, yy6, zz6, tt6, pp6 = p6[0].get_data()
    assert not np.allclose(xx6, xx1)
    assert not np.allclose(yy6, yy1)
    assert not np.allclose(zz6, zz1)
    assert np.allclose(tt6, tt1)
    assert np.allclose(pp6, pp1)

    # wrong parallel_axis
    raises(ValueError, lambda : plot3d_revolution(
        cos(t), (t, 0, pi), show=False, n=5, parallel_axis="a"))

    # show_curve should add a Parametric3DLineSeries
    # keyword arguments should gets redirected to plot3d_parametric_surface
    # or plot3d_parametric_line
    p = plot3d_revolution(cos(t), (t, 0, pi), show=False, n=5,
        show_curve=True, rendering_kw={"color": "r"},
        curve_kw={"rendering_kw": {"color": "g"}})
    assert len(p.series) == 2
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert isinstance(p[1], Parametric3DLineSeries)
    assert p[1].expr == (t, 0, cos(t))
    assert p[0].rendering_kw == {"color": "r"}
    assert p[1].rendering_kw == {"color": "g"}

    # wireframe should add other series
    p = plot3d_revolution(cos(t), (t, 0, pi), show=False, n=5,
        show_curve=True, wireframe=True)
    assert len(p.series) > 2
    assert all(isinstance(t, Parametric3DLineSeries) for t in p.series[1:])

    # interactive widget plot
    k = symbols("k")
    p1 = plot3d_revolution(cos(k * t), (t, 0, pi), show=False, n=5,
        params={k: (1, 0, 2)}, parallel_axis="x", show_curve=False)
    assert isinstance(p1, InteractivePlot)
    assert len(p1.backend.series) == 1
    assert isinstance(p1.backend[0], ParametricSurfaceInteractiveSeries)

    p2 = plot3d_revolution(cos(k * t), (t, 0, pi), show=False, n=5,
        params={k: (1, 0, 2)}, parallel_axis="y", show_curve=False)
    assert isinstance(p2, InteractivePlot)
    assert p2.backend[0].expr != p1.backend[0].expr

    p3 = plot3d_revolution(cos(k * t), (t, 0, pi), show=False, n=5,
        params={k: (1, 0, 2)}, parallel_axis="z", show_curve=False)
    assert isinstance(p3, InteractivePlot)
    assert p3.backend[0].expr != p1.backend[0].expr
    assert p3.backend[0].expr != p2.backend[0].expr

    # show_curve should add a Parametric3DLineInteractiveSeries
    # keyword arguments should gets redirected to plot3d_parametric_surface
    # or plot3d_parametric_line
    p = plot3d_revolution(cos(k * t), (t, 0, pi), show=False, n=5,
        params={k: (1, 0, 2)}, show_curve=True, rendering_kw={"color": "r"},
        curve_kw={"rendering_kw": {"color": "g"}})
    assert isinstance(p, InteractivePlot)
    assert len(p.backend.series) == 2
    assert isinstance(p.backend[0], ParametricSurfaceInteractiveSeries)
    assert isinstance(p.backend[1], Parametric3DLineInteractiveSeries)
    assert p.backend[1].expr == (t, 0, cos(k * t))
    assert p.backend[0].rendering_kw == {"color": "r"}
    assert p.backend[1].rendering_kw == {"color": "g"}

    # wireframe should add other series
    p = plot3d_revolution(cos(k * t), (t, 0, pi), show=False, n=5,
        params={k: (1, 0, 2)}, show_curve=True, wireframe=True)
    assert isinstance(p, InteractivePlot)
    assert len(p.backend.series) > 2
    assert all(isinstance(t, Parametric3DLineInteractiveSeries) for t in p.backend.series[1:])


def test_plot3d_spherical():
    # plot3d_spherical is going to call plot3d_parametric_surface: let's
    # check that the data series are correct.

    phi, theta = symbols("phi, theta")
    p = plot3d_spherical(1, (theta, pi/4, 3*pi/4), (phi, pi/2, 3*pi/2),
        show=False)
    assert len(p.series) == 1
    assert isinstance(p[0], ParametricSurfaceSeries)
    # verify spherical to cartesian transformation
    assert p[0].expr_x == sin(theta) * cos(phi)
    assert p[0].expr_y == sin(theta) * sin(phi)
    assert p[0].expr_z == cos(theta)
    # verify ranges are set correctly
    assert (p[0].var_u == theta) and (p[0].var_v == phi)
    assert np.allclose([p[0].start_u, p[0].end_u], [float(pi/4), float(3*pi/4)])
    assert np.allclose([p[0].start_v, p[0].end_v], [float(pi/2), float(3*pi/2)])


def test_plot3d_wireframe_transform_function():
    # verify that when plot3d and wireframe=True are used together with
    # transformation functions (tx, ty, tz), those are also applied to
    # wireframe lines.

    x, y = symbols("x, y")

    fx = lambda t: t*2
    fy = lambda t: t*3
    fz = lambda t: t*4
    p = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2), n1=10, n2=10,
            wireframe=True, wf_n1=3, wf_n2=3, show=False,
            tx=fx, ty=fy, tz=fz)
    assert all((s._tx == fx) and (s._ty == fy) and (s._tz == fz) for s in p.series)


def test_plot3d_plot_contour_base_scalars():
    # verify that these functions are able to deal with base scalars

    C = CoordSys3D("")
    x, y, z = C.base_scalars()
    plot_contour(cos(x * y), (x, -2, 2), (y, -2, 2),
        use_latex=True, show=False)
    plot_contour(cos(x * y), (x, -2, 2), (y, -2, 2),
        use_latex=False, show=False)
    plot3d(cos(x * y), (x, -2, 2), (y, -2, 2),
        use_latex=True, show=False)
    plot3d(cos(x * y), (x, -2, 2), (y, -2, 2),
        use_latex=False, show=False)


def test_plot_list():
    # verify that plot_list creates the correct data series

    xx1 = np.linspace(-3, 3)
    yy1 = np.cos(xx1)
    xx2 = np.linspace(-5, 5)
    yy2 = np.sin(xx2)

    p = plot_list(xx1, yy1, backend=MB, show=False)
    assert len(p.series) == 1
    assert isinstance(p.series[0], List2DSeries)
    assert p.series[0].label == ""

    p = plot_list(xx1, yy1, "test", backend=MB, show=False)
    assert len(p.series) == 1
    assert isinstance(p.series[0], List2DSeries)
    assert p.series[0].label == "test"

    p = plot_list((xx1, yy1), (xx2, yy2), backend=MB, show=False)
    assert len(p.series) == 2
    assert all(isinstance(t, List2DSeries) for t in p.series)
    assert all(t.label == "" for t in p.series)

    p = plot_list((xx1, yy1, "cos"), (xx2, yy2, "sin"),
        backend=MB, show=False)
    assert len(p.series) == 2
    assert all(isinstance(t, List2DSeries) for t in p.series)
    assert p.series[0].label == "cos"
    assert p.series[1].label == "sin"


def test_process_sums():
    # verify that Sum containing infinity in its boundary, gets replaced with
    # a Sum with arbitrary big numbers instead.
    x, y = symbols("x, y")

    expr = Sum(1 / x ** y, (x, 1, oo))
    p1 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=1000, show=False)
    assert p1[0].expr.args[-1] == (x, 1, 1000)
    p2 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=500, show=False)
    assert p2[0].expr.args[-1] == (x, 1, 500)

    expr = Sum(1 / x ** y, (x, -oo, -1))
    p1 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=1000, show=False)
    assert p1[0].expr.args[-1] == (x, -1000, -1)
    p2 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=500, show=False)
    assert p2[0].expr.args[-1] == (x, -500, -1)

    expr = Sum(1 / x ** y, (x, -oo, oo))
    p1 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=1000, show=False)
    assert p1[0].expr.args[-1] == (x, -1000, 1000)
    p2 = plot(expr, (y, 2, 10), backend=MB, adaptive=False, n=20,
        sum_bound=500, show=False)
    assert p2[0].expr.args[-1] == (x, -500, 500)


def test_plot_piecewise():
    # Verify that univariate Piecewise objects are processed in such a way to
    # create multiple series, each one with the correct range.
    # Series representing filled dots should be last in order.

    x = symbols("x")
    f = Piecewise(
        (-1, x < -1),
        (x, And(-1 <= x, x < 0)),
        (x**2, And(0 <= x, x < 1)),
        (x**3, x >= 1)
    )
    p = plot_piecewise(f, (x, -5, 5), backend=MB, show=False)
    s = p.series
    assert not p.legend
    assert len(s) == 10
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 4
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 6
    assert (s[0].expr == -S(1)) and np.isclose(s[0].start.real, -5) and np.isclose(s[0].end.real, -1 - 1e-06)
    assert np.allclose(s[1].list_x, -1) and np.allclose(s[1].list_y, -1) and (not s[1].is_filled)
    assert (s[2].expr == x) and np.isclose(s[2].start.real, -1 - 1e-06) and np.isclose(s[2].end.real, -1e-06)
    assert np.allclose(s[3].list_x, -1e-06) and np.allclose(s[3].list_y, -1e-06) and (not s[3].is_filled)
    assert (s[4].expr == x**2) and np.isclose(s[4].start.real, 0) and np.isclose(s[4].end.real, 1 - 1e-06)
    assert np.allclose(s[5].list_x, 1 - 1e-06) and np.allclose(s[5].list_y, 1 - 2e-06) and (not s[5].is_filled)
    assert (s[6].expr == x**3) and np.isclose(s[6].start.real, 1) and np.isclose(s[6].end.real, 5)
    assert np.allclose(s[7].list_x, -1) and np.allclose(s[7].list_y, -1) and s[7].is_filled
    assert np.allclose(s[8].list_x, 0) and np.allclose(s[8].list_y, 0) and s[8].is_filled
    assert np.allclose(s[9].list_x, 1) and np.allclose(s[9].list_y, 1) and s[9].is_filled

    f = Heaviside(x, 0).rewrite(Piecewise)
    p = plot_piecewise(f, (x, -10, 10), backend=MB, show=False)
    s = p.series
    assert not p.legend
    assert len(s) == 4
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 2
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 2
    assert (s[0].expr == 0) and np.isclose(s[0].start.real, -10) and np.isclose(s[0].end.real, 0)
    assert (s[1].expr == 1) and np.isclose(s[1].start.real, 1e-06) and np.isclose(s[1].end.real, 10)
    assert np.allclose(s[2].list_x, 1e-06) and np.allclose(s[2].list_y, 1) and (not s[2].is_filled)
    assert np.allclose(s[3].list_x, 0) and np.allclose(s[3].list_y, 0) and s[3].is_filled

    f = Piecewise((x, Interval(0, 1).contains(x)), (0, True))
    p = plot_piecewise(f, (x, -10, 10), backend=MB, show=False)
    s = p.series
    assert not p.legend
    assert len(s) == 7
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 3
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 4
    assert len([t for t in s if isinstance(t, List2DSeries) and t.is_filled]) == 2

    f = Piecewise((x, x < 1), (x**2, (x >= -1) & (x <= 3)), (x, x > 3))
    p = plot_piecewise(f, (x, -10, 10), backend=MB, show=False)
    assert not p.legend
    s = p.series
    assert len(s) == 7
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 3
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 4
    assert len([t for t in s if isinstance(t, List2DSeries) and t.is_filled]) == 2

    # NotImplementedError: as_set is not implemented for relationals with
    # periodic solutions
    p1 = Piecewise((cos(x), x < 0), (0, True))
    f = Piecewise((0, Eq(p1, 0)), (p1 / Abs(p1), True))
    raises(NotImplementedError, lambda: plot_piecewise(f, (x, -10, 10),
        backend=MB, show=False))

    # The range is smaller than the function "domain"
    f = Piecewise(
        (1, x < -5),
        (x, Eq(x, 0)),
        (x**2, Eq(x, 2)),
        (x**3, (x > 0) & (x < 2)),
        (x**4, True)
    )
    p = plot_piecewise(f, (x, -3, 3), backend=MB, show=False)
    s = p.series
    assert not p.legend
    assert len(s) == 9
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 3
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 6
    assert len([t for t in s if isinstance(t, List2DSeries) and t.is_filled]) == 2


def test_plot_piecewise_dot_false():
    # Verify that plot_piecewise doesn't create endpoint dots when dots=False.

    x = symbols("x")
    f = Piecewise((x**2, x < 2), (5, Eq(x, 2)), (10 - x, True))
    p = plot_piecewise(f, (x, -2, 5), show=False)
    assert len(p.series) == 5
    assert len([t for t in p.series if isinstance(t, List2DSeries)]) == 3

    f = Piecewise((x**2, x < 2), (5, Eq(x, 2)), (10 - x, True))
    p = plot_piecewise(f, (x, -2, 5), show=False, dots=False)
    assert len(p.series) == 3
    assert len([t for t in p.series if isinstance(t, List2DSeries)]) == 1


###############################################################################
################### PLOT - PLOT_PARAMETRIC - PLOT3D-RELATED ###################
###############################################################################
##### These tests comes from the old sympy.plotting module
##### Some test have been rearranged to reflect the intentions of this new
##### plotting module

def test_plot_and_save_1():
    x, y, z = symbols("x, y, z")

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        ###
        # Examples from the 'introduction' notebook
        ###
        p = plot(x, legend=True)
        p = plot(x * sin(x), x * cos(x))
        p.extend(p)
        filename = "test_basic_options_and_colors.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p.extend(plot(x + 1))
        p.append(plot(x + 3, x ** 2)[1])
        filename = "test_plot_extend_append.png"
        p.save(os.path.join(tmpdir, filename))

        p[2] = plot(x ** 2, (x, -2, 3))
        filename = "test_plot_setitem.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(sin(x), (x, -2 * pi, 4 * pi))
        filename = "test_line_explicit.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(sin(x))
        filename = "test_line_default_range.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot((x ** 2, (x, -5, 5)), (x ** 3, (x, -3, 3)))
        filename = "test_line_multiple_range.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        raises(ValueError, lambda: plot(x, y))

        # Piecewise plots
        p = plot(Piecewise((1, x > 0), (0, True)), (x, -1, 1))
        filename = "test_plot_piecewise.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(Piecewise((x, x < 1), (x ** 2, True)), (x, -3, 3))
        filename = "test_plot_piecewise_2.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


def test_issue_7471():
    x = symbols("x")

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        p1 = plot(x)
        p2 = plot(3)
        p1.extend(p2)
        filename = "test_horizontal_line.png"
        p1.save(os.path.join(tmpdir, filename))
        p1.close()


def test_issue_10925():
    x = symbols("x")

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        f = Piecewise(
            (-1, x < -1),
            (x, And(-1 <= x, x < 0)),
            (x ** 2, And(0 <= x, x < 1)),
            (x ** 3, x >= 1),
        )
        p = plot(f, (x, -3, 3))
        filename = "test_plot_piecewise_3.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


def test_plot_and_save_2():
    x, y, z = symbols("x, y, z")

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        # parametric 2d plots.
        # Single plot with default range.
        p = plot_parametric(sin(x), cos(x))
        filename = "test_parametric.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Single plot with range.
        p = plot_parametric(sin(x), cos(x), (x, -5, 5), legend=True)
        filename = "test_parametric_range.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple plots with same range.
        p = plot_parametric((sin(x), cos(x)), (x, sin(x)))
        filename = "test_parametric_multiple.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple plots with different ranges.
        p = plot_parametric((sin(x), cos(x), (x, -3, 3)), (x, sin(x), (x, -5, 5)))
        filename = "test_parametric_multiple_ranges.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # depth of recursion specified.
        p = plot_parametric(x, sin(x), depth=13)
        filename = "test_recursion_depth.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # No adaptive sampling.
        p = plot_parametric(cos(x), sin(x), adaptive=False, n=500)
        filename = "test_adaptive.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # 3d parametric plots
        p = plot3d_parametric_line(sin(x), cos(x), x, legend=True)
        filename = "test_3d_line.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot3d_parametric_line(
            (sin(x), cos(x), x, (x, -5, 5)), (cos(x), sin(x), x, (x, -3, 3))
        )
        filename = "test_3d_line_multiple.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot3d_parametric_line(sin(x), cos(x), x, n=30)
        filename = "test_3d_line_points.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # 3d surface single plot.
        p = plot3d(x * y)
        filename = "test_surface.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple 3D plots with same range.
        p = plot3d(-x * y, x * y, (x, -5, 5))
        filename = "test_surface_multiple.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple 3D plots with different ranges.
        p = plot3d((x * y, (x, -3, 3), (y, -3, 3)), (-x * y, (x, -3, 3), (y, -3, 3)))
        filename = "test_surface_multiple_ranges.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Single Parametric 3D plot
        p = plot3d_parametric_surface(sin(x + y), cos(x - y), x - y)
        filename = "test_parametric_surface.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple Parametric 3D plots.
        p = plot3d_parametric_surface(
            (x * sin(z), x * cos(z), z, (x, -5, 5), (z, -5, 5)),
            (sin(x + y), cos(x - y), x - y, (x, -5, 5), (y, -5, 5)),
        )
        filename = "test_parametric_surface.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Single Contour plot.
        p = plot_contour(sin(x) * sin(y), (x, -5, 5), (y, -5, 5))
        filename = "test_contour_plot.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple Contour plots with same range.
        p = plot_contour(x ** 2 + y ** 2, x ** 3 + y ** 3, (x, -5, 5), (y, -5, 5))
        filename = "test_contour_plot.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple Contour plots with different range.
        p = plot_contour(
            (x ** 2 + y ** 2, (x, -5, 5), (y, -5, 5)),
            (x ** 3 + y ** 3, (x, -3, 3), (y, -3, 3)),
        )
        filename = "test_contour_plot.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


def test_plot_and_save_4():
    x, y = symbols("x, y")

    ###
    # Examples from the 'advanced' notebook
    ###

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        i = Integral(log((sin(x) ** 2 + 1) * sqrt(x ** 2 + 1)), (x, 0, y))
        p = plot(i, (y, 1, 5))
        filename = "test_advanced_integral.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


def test_plot_and_save_5():
    x, y = symbols("x, y")

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        s = Sum(1 / x ** y, (x, 1, oo))
        p = plot(s, (y, 2, 10), adaptive=False, only_integers=True)
        filename = "test_advanced_inf_sum.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(Sum(1 / x, (x, 1, y)), (y, 2, 10), adaptive=False,
            only_integers=True, steps=True)
        filename = "test_advanced_fin_sum.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


def test_plot_and_save_6():
    x = symbols("x")

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        filename = "test.png"
        ###
        # Test expressions that can not be translated to np and generate complex
        # results.
        ###
        p = plot(sin(x) + I * cos(x))
        p.save(os.path.join(tmpdir, filename))
        p = plot(sqrt(sqrt(-x)))
        p.save(os.path.join(tmpdir, filename))
        p = plot(LambertW(x))
        p.save(os.path.join(tmpdir, filename))
        p = plot(sqrt(LambertW(x)))
        p.save(os.path.join(tmpdir, filename))

        # Characteristic function of a StudentT distribution with nu=10
        x1 = 5 * x ** 2 * exp_polar(-I * pi) / 2
        m1 = meijerg(((1 / 2,), ()), ((5, 0, 1 / 2), ()), x1)
        x2 = 5 * x ** 2 * exp_polar(I * pi) / 2
        m2 = meijerg(((1 / 2,), ()), ((5, 0, 1 / 2), ()), x2)
        expr = (m1 + m2) / (48 * pi)
        p = plot(expr, (x, 1e-6, 1e-2), adaptive=False, n=20)
        p.save(os.path.join(tmpdir, filename))


def test_issue_11461():
    x = symbols("x")

    expr = real_root((log(x / (x - 2))), 3)
    p = plot(expr, backend=MB, show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) > 30

    # plot_piecewise is not able to deal with ConditionSet
    raises(TypeError, lambda: plot_piecewise(expr, backend=MB, show=False))


def test_issue_11865():
    k = symbols("k", integer=True)
    f = Piecewise(
        (-I * exp(I * pi * k) / k + I * exp(-I * pi * k) / k, Ne(k, 0)),
        (2 * pi, True)
    )
    p = plot(f, backend=MB, show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) > 30

    p = plot_piecewise(f, backend=MB, show=False)
    assert len(p[0].get_data()[0]) > 30


def test_issue_16572():
    x = symbols("x")
    p = plot(LambertW(x), backend=MB, show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30


def test_logplot_PR_16796():
    x = symbols("x")
    p = plot(x, (x, 0.001, 100), backend=MB, xscale="log", show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30
    assert p[0].end == 100.0
    assert p[0].start == 0.001


def test_issue_17405():
    x = symbols("x")
    f = x ** 0.3 - 10 * x ** 3 + x ** 2
    p = plot(f, (x, -10, 10), backend=MB, show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30


def test_issue_15265():
    x = symbols("x")
    eqn = sin(x)

    p = plot(eqn, xlim=(-S.Pi, S.Pi), ylim=(-1, 1))
    p.close()

    p = plot(eqn, xlim=(-1, 1), ylim=(-S.Pi, S.Pi))
    p.close()

    p = plot(eqn, xlim=(-1, 1), ylim=(sympify("-3.14"), sympify("3.14")))
    p.close()

    p = plot(eqn, xlim=(sympify("-3.14"), sympify("3.14")), ylim=(-1, 1))
    p.close()

    raises(ValueError, lambda: plot(eqn, xlim=(-S.ImaginaryUnit, 1), ylim=(-1, 1)))

    raises(ValueError, lambda: plot(eqn, xlim=(-1, 1), ylim=(-1, S.ImaginaryUnit)))

    raises(ValueError, lambda: plot(eqn, xlim=(S.NegativeInfinity, 1), ylim=(-1, 1)))

    raises(ValueError, lambda: plot(eqn, xlim=(-1, 1), ylim=(-1, S.Infinity)))


def test_append_issue_7140():
    x = symbols("x")
    p1 = plot(x, backend=MB, show=False)
    p2 = plot(x ** 2, backend=MB, show=False)

    # append a series
    p2.append(p1[0])
    assert len(p2._series) == 2

    with raises(TypeError):
        p1.append(p2)

    with raises(TypeError):
        p1.append(p2._series)


def test_plot_limits():
    x = symbols("x")
    p = plot(x, x ** 2, (x, -10, 10), backend=MB, show=False)

    xmin, xmax = p.fig.axes[0].get_xlim()
    assert abs(xmin + 10) < 2
    assert abs(xmax - 10) < 2
    ymin, ymax = p.fig.axes[0].get_ylim()
    assert abs(ymin + 10) < 10
    assert abs(ymax - 100) < 10


def test_plot3d_parametric_line_limits():
    x = symbols("x")

    v1 = (2 * cos(x), 2 * sin(x), 2 * x, (x, -5, 5))
    v2 = (sin(x), cos(x), x, (x, -5, 5))
    p = plot3d_parametric_line(v1, v2)

    xmin, xmax = p.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    ymin, ymax = p.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    zmin, zmax = p.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2

    p = plot3d_parametric_line(v2, v1)

    xmin, xmax = p.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    ymin, ymax = p.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    zmin, zmax = p.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2


###############################################################################
################################ PLOT IMPLICIT ################################
###############################################################################
##### These tests comes from the old sympy.plotting module

def tmp_file(dir=None, name=""):
    return NamedTemporaryFile(suffix=".png", dir=dir, delete=False).name


def test_plot_implicit_label_rendering_kw():
    # verify that label and rendering_kw keyword arguments works as expected

    x, y = symbols("x, y")
    p = plot_implicit(x + y, x - y, (x, -5, 5), (y, -5, 5),
        show=False, adaptive=False,
        label=["a", "b"], rendering_kw=[{"levels": 5}, {"alpha": 0.5}])
    assert p[0].get_label(True) == "a"
    assert p[1].get_label(True) == "b"
    assert p[0].rendering_kw == {"levels": 5}
    assert p[1].rendering_kw == {"alpha": 0.5}


@pytest.mark.xfail
def test_plot_implicit_adaptive_true():
    # verify that plot_implicit with `adaptive=True` produces correct results.

    # NOTE 1: how to test that the algorithm is producing correct results when
    # adaptive=True? Easiest approach is to compare the plots with vouched
    # ones. However, the following plots may slightly change
    # every time tests are run, hence we need a relatively large tolerance.

    # NOTE 2: these tests require Matplotlib v3.4.2. Using a different version
    # will likely make these tests fail.

    matplotlib = import_module(
        "matplotlib", min_module_version="1.1.0", catch=(RuntimeError,)
    )
    if not matplotlib:
        skip("Matplotlib not the default backend")

    from matplotlib.testing.compare import compare_images
    test_directory = os.path.dirname(os.path.abspath(__file__))

    temp_dir = mkdtemp()
    TmpFileManager.tmp_folder(temp_dir)

    def do_test(expr, range_x, range_y, filename, tol=1):
        test_filename = tmp_file(dir=temp_dir)
        cmp_filename = os.path.join(test_directory, "imgs", filename)
        p = plot_implicit(expr, range_x, range_y,
            size=(5, 4), adaptive=True, grid=False, show=False,
            use_latex=False)
        p.save(test_filename)
        p.close()
        assert compare_images(cmp_filename, test_filename, tol) is None

    x, y = symbols("x y")

    try:
        do_test(Eq(y, cos(x)), (x, -5, 5), (y, -5, 5), "pi_01.png", 1)
        do_test(Eq(y ** 2, x ** 3 - x), (x, -5, 5), (y, -5, 5), "pi_02.png", 1)
        do_test(y > 1 / x, (x, -5, 5), (y, -5, 5), "pi_03.png", 5)
        do_test(y < 1 / tan(x), (x, -5, 5), (y, -5, 5), "pi_04.png", 8)
        do_test(y >= 2 * sin(x) * cos(x), (x, -5, 5), (y, -5, 5), "pi_05.png", 1)
        do_test(y <= x ** 2, (x, -5, 5), (y, -5, 5), "pi_06.png", 1)
        do_test(y > x, (x, -5, 5), (y, -5, 5), "pi_07.png", 5)
        do_test(And(y > exp(x), y > x + 2), (x, -5, 5), (y, -5, 5), "pi_08.png", 1)
        do_test(Or(y > x, y > -x), (x, -5, 5), (y, -5, 5), "pi_09.png", 5)
        do_test(x ** 2 - 1, (x, -5, 5), (y, -5, 5), "pi_10.png", 1)
        do_test(y > cos(x), (x, -5, 5), (y, -5, 5), "pi_11.png", 1)
        do_test(y < cos(x), (x, -5, 5), (y, -5, 5), "pi_12.png", 1)
        do_test(And(y > cos(x), Or(y > x, Eq(y, x))), (x, -5, 5), (y, -5, 5), "pi_13.png", 1)
        do_test(y - cos(pi / x), (x, -5, 5), (y, -5, 5), "pi_14.png", 1)
        # NOTE: this should fallback to adaptive=False
        do_test(Eq(y, re(cos(x) + I * sin(x))), (x, -5, 5), (y, -5, 5), "pi_15.png", 1)
    finally:
        TmpFileManager.cleanup()


@pytest.mark.xfail
def test_plot_implicit_region_and():
    # NOTE: these tests require Matplotlib v3.4.2. Using a different version
    # will likely make these tests fail.

    matplotlib = import_module(
        "matplotlib", min_module_version="1.1.0", catch=(RuntimeError,)
    )
    if not matplotlib:
        skip("Matplotlib not the default backend")

    from matplotlib.testing.compare import compare_images
    test_directory = os.path.dirname(os.path.abspath(__file__))

    temp_dir = mkdtemp()
    TmpFileManager.tmp_folder(temp_dir)

    def do_test(expr, range_x, range_y, filename, tol=1):
        test_filename = tmp_file(dir=temp_dir)
        test_filename = filename
        cmp_filename = os.path.join(test_directory, "imgs", filename)
        p = plot_implicit(expr, range_x, range_y,
            size=(8, 6), adaptive=True, grid=False, show=False,
            use_latex=False)
        p.save(test_filename)
        p.close()
        assert compare_images(cmp_filename, test_filename, tol) is None

    x, y = symbols("x y")
    r1 = (x - 1) ** 2 + y ** 2 < 2
    r2 = (x + 1) ** 2 + y ** 2 < 2

    try:
        do_test(r1 & r2, (x, -5, 5), (y, -5, 5), "test_region_and.png", 0.005)
        do_test(r1 | r2, (x, -5, 5), (y, -5, 5), "test_region_or.png", 0.005)
        do_test(~r1, (x, -5, 5), (y, -5, 5), "test_region_not.png", 0.05)
        do_test(r1 ^ r2, (x, -5, 5), (y, -5, 5), "test_region_xor.png", 0.005)
    finally:
        TmpFileManager.cleanup()


def test_lambda_functions():
    # verify that plotting functions raises errors if they do not support
    # lambda functions.

    raises(TypeError, lambda : plot_piecewise(lambda t: t))


def test_functions_iplot_integration():
    # verify the integration between most important plot functions and iplot

    x, y, z, u, v = symbols("x, y, z, u, v")
    p = plot(cos(u * x), params={u: (1, 0, 2)}, show=False)
    assert isinstance(p, InteractivePlot)

    p = plot_parametric(cos(u * x), sin(x), params={u: (1, 0, 2)}, show=False)
    assert isinstance(p, InteractivePlot)

    p = plot3d_parametric_line(cos(u * x), sin(x), u * x,
        params={u: (1, 0, 2)}, show=False)
    assert isinstance(p, InteractivePlot)

    p = plot3d(cos(u * x**2 + y**2), params={u: (1, 0, 2)}, show=False)
    assert isinstance(p, InteractivePlot)

    p = plot_contour(cos(u * x**2 + y**2), params={u: (1, 0, 2)}, show=False)
    assert isinstance(p, InteractivePlot)

    r = 2 + sin(7 * u + 5 * v)
    expr = (
        r * cos(x * u) * sin(v),
        r * sin(x * u) * sin(v),
        r * cos(v)
        )
    p = plot3d_parametric_surface(*expr, (u, 0, 2 * pi), (v, 0, pi),
        params={x: (1, 0, 2)}, show=False)
    assert isinstance(p, InteractivePlot)

    p = lambda: plot3d_implicit(
        u * x**2 + y**3 - z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2),
        params={u: (1, 0, 2)}, show=False)
    raises(NotImplementedError, p)

    p = lambda: plot_implicit(Eq(u * x**2 + y**2, 3), (x, -3, 3), (y, -3, 3),
        params={u: (1, 0, 2)}, show=False)
    raises(NotImplementedError, p)

    p = plot_geometry(Circle((u, 0), 4),
        params={u: (1, 0, 2)}, show=False)
    assert isinstance(p, InteractivePlot)

    p = plot_list([1, 2, 3], [4, 5, 6],
        params={u: (1, 0, 2)}, show=False)
    assert isinstance(p, InteractivePlot)

    p = lambda: plot_piecewise(
        u * Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10),
        params={u: (1, 0, 2)}, show=False)
    raises(NotImplementedError, p)
