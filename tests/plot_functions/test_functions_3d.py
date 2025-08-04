from pytest import raises
import spb
from spb import prange
from spb.plot_functions.functions_3d import (
    plot3d, plot3d_parametric_line, plot3d_parametric_surface,
    plot3d_revolution, plot3d_spherical, plot3d_implicit
)
from spb.plot_functions.functions_2d import plot_contour
from spb.series import (
    ParametricSurfaceSeries, Parametric3DLineSeries,
    SurfaceOver2DRangeSeries
)
from spb.backends.matplotlib import MB
from sympy import symbols, sin, cos, pi, IndexedBase
from sympy.vector import CoordSys3D
from sympy.external import import_module
import pytest
import numpy as np

pn = import_module("panel")
adaptive_module = import_module("adaptive")


# NOTE:
#
# These tests are meant to verify that the plotting functions generate the
# correct data series.
# Also, legacy tests from the old sympy.plotting module are included.
#
# If your issue is related to the generation of numerical data from a
# particular data series, consider adding tests to tests/test_series.py.
# If your issue is related to a particular keyword affecting a backend
# behaviour, consider adding tests to tests/backends/test_*.py
#


@pytest.mark.filterwarnings("ignore:The following")
def test_plot3d_parametric_line(p_options):
    # Test arguments for plot3d_parametric_line()

    x, y = symbols("x, y")

    # single parametric expression
    p = plot3d_parametric_line(x + 1, x, sin(x), **p_options)
    assert isinstance(p[0], Parametric3DLineSeries)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}

    # single parametric expression with custom range, label and rendering kws
    p = plot3d_parametric_line(
        x + 1, x, sin(x), (x, -2, 2), "test", {"cmap": "Reds"}, **p_options
    )
    assert isinstance(p[0], Parametric3DLineSeries)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    p = plot3d_parametric_line(
        (x + 1, x, sin(x)), (x, -2, 2), "test", **p_options)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}

    # multiple parametric expression same symbol
    p = plot3d_parametric_line(
        (x + 1, x, sin(x)),
        (x**2, 1, cos(x), {"cmap": "Reds"}),
        **p_options
    )
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x**2, 1, cos(x))
    assert p[1].ranges == [(x, -10, 10)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # multiple parametric expression different symbols
    p = plot3d_parametric_line(
        (x + 1, x, sin(x)),
        (y**2, 1, cos(y)),
        **p_options)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (y**2, 1, cos(y))
    assert p[1].ranges == [(y, -10, 10)]
    assert p[1].get_label(False) == "y"
    assert p[1].rendering_kw == {}

    # multiple parametric expression, custom ranges and labels
    p = plot3d_parametric_line(
        (x + 1, x, sin(x)),
        (x**2, 1, cos(x), (x, -2, 2), "test", {"cmap": "Reds"}),
        **p_options
    )
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x**2, 1, cos(x))
    assert p[1].ranges == [(x, -2, 2)]
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # single argument: lambda function
    fx = lambda t: t
    fy = lambda t: 2 * t
    fz = lambda t: 3 * t
    p = plot3d_parametric_line(fx, fy, fz, **p_options)
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert "Dummy" in p[0].get_label(False)
    assert p[0].rendering_kw == {}

    # single argument: lambda function + custom range + label
    p = plot3d_parametric_line(fx, fy, fz, ("t", 0, 2), "test", **p_options)
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (0, 2)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}


@pytest.mark.filterwarnings("ignore:The following")
def test_plot3d_plot_contour(p_options):
    # Test arguments for plot3d() and plot_contour()

    x, y = symbols("x, y")

    # single expression
    p = plot3d(x + y, **p_options)
    assert isinstance(p[0], SurfaceOver2DRangeSeries)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}

    # single expression, custom range, label and rendering kws
    p = plot3d(x + y, (x, -2, 2), "test", {"cmap": "Reds"}, **p_options)
    assert isinstance(p[0], SurfaceOver2DRangeSeries)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -10, 10)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    p = plot3d(x + y, (x, -2, 2), (y, -4, 4), "test", **p_options)
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)

    # multiple expressions
    p = plot3d(x + y, x * y, **p_options)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[1].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[1].get_label(False) == "x*y"
    assert p[1].rendering_kw == {}

    # multiple expressions, same custom ranges
    p = plot3d(x + y, x * y, (x, -2, 2), (y, -4, 4), **p_options)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -2, 2)
    assert p[1].ranges[1] == (y, -4, 4)
    assert p[1].get_label(False) == "x*y"
    assert p[1].rendering_kw == {}

    # multiple expressions, custom ranges, labels and rendering kws
    p = plot3d(
        (x + y, (x, -2, 2), (y, -4, 4)),
        (x * y, (x, -3, 3), (y, -6, 6), "test", {"cmap": "Reds"}),
        **p_options
    )
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -3, 3)
    assert p[1].ranges[1] == (y, -6, 6)
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # single expression: lambda function
    f = lambda x, y: x + y
    p = plot3d(f, **p_options)
    assert callable(p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert p[0].ranges[1][1:] == (-10, 10)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}

    # single expression: lambda function + custom ranges + label
    p = plot3d(f, ("a", -5, 3), ("b", -2, 1), "test", **p_options)
    assert callable(p[0].expr)
    assert p[0].ranges[0][1:] == (-5, 3)
    assert p[0].ranges[1][1:] == (-2, 1)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}


@pytest.mark.filterwarnings("ignore:The following")
def test_plot3d_parametric_surface(p_options):
    # Test arguments for plot3d_parametric_surface()

    x, y = symbols("x, y")

    # single parametric expression
    p = plot3d_parametric_surface(x + y, cos(x + y), sin(x + y), **p_options)
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "(x + y, cos(x + y), sin(x + y))"
    assert p[0].rendering_kw == {}

    # single parametric expression, custom ranges, labels and rendering kws
    p = plot3d_parametric_surface(
        x + y,
        cos(x + y),
        sin(x + y),
        (x, -2, 2),
        (y, -4, 4),
        "test",
        {"cmap": "Reds"},
        **p_options
    )
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    # multiple parametric expressions
    p = plot3d_parametric_surface(
        (x + y, cos(x + y), sin(x + y)),
        (x - y, cos(x - y), sin(x - y), "test"),
        **p_options
    )
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "(x + y, cos(x + y), sin(x + y))"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x - y, cos(x - y), sin(x - y))
    assert p[1].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[1].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions, custom ranges and labels
    p = plot3d_parametric_surface(
        (x + y, cos(x + y), sin(x + y), (x, -2, 2), "test"),
        (
            x - y, cos(x - y), sin(x - y),
            (x, -3, 3), (y, -4, 4),
            "test2", {"cmap": "Reds"},
        ),
        **p_options
    )
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -10, 10)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x - y, cos(x - y), sin(x - y))
    assert p[1].ranges[0] == (x, -3, 3)
    assert p[1].ranges[1] == (y, -4, 4)
    assert p[1].get_label(False) == "test2"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # lambda functions instead of symbolic expressions for a single 3D
    # parametric surface
    p = plot3d_parametric_surface(
        lambda u, v: u,
        lambda u, v: v,
        lambda u, v: u + v,
        ("u", 0, 2),
        ("v", -3, 4),
        **p_options
    )
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (-0, 2)
    assert p[0].ranges[1][1:] == (-3, 4)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}

    # lambda functions instead of symbolic expressions for multiple 3D
    # parametric surfaces
    p = plot3d_parametric_surface(
        (
            lambda u, v: u, lambda u, v: v, lambda u, v: u + v,
            ("u", 0, 2), ("v", -3, 4)
        ),
        (
            lambda u, v: v, lambda u, v: u, lambda u, v: u - v,
            ("u", -2, 3), ("v", -4, 5), "test",
        ),
        **p_options
    )
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (0, 2)
    assert p[0].ranges[1][1:] == (-3, 4)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}
    assert all(callable(t) for t in p[1].expr)
    assert p[1].ranges[0][1:] == (-2, 3)
    assert p[1].ranges[1][1:] == (-4, 5)
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {}


@pytest.mark.filterwarnings("ignore:The following")
@pytest.mark.filterwarnings("ignore:NumPy is unable to evaluate")
def test_plot3d_revolution(paf_options, pi_options):
    # plot3d_revolution is going to call plot3d_parametric_surface and
    # plot3d_parametric_line: let's check that the data series are correct.

    t, phi = symbols("t, phi")

    options = paf_options.copy()
    options["n"] = 5

    # test that azimuthal angle is set correctly
    p = plot3d_revolution(cos(t), (t, 0, pi), **options)
    assert len(p.series) == 1
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert (p[0].var_u, p[0].start_u, p[0].end_u) == (t, 0, pi)
    assert (p[0].var_v, p[0].start_v, p[0].end_v) == (phi, 0, 2 * pi)

    p = plot3d_revolution(cos(t), (t, 0, pi), (phi, 0, pi / 2), **options)
    assert len(p.series) == 1
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert (p[0].var_u, p[0].start_u, p[0].end_u) == (t, 0, pi)
    assert (p[0].var_v, p[0].start_v, p[0].end_v) == (phi, 0, pi / 2)

    # by setting parallel_axis it produces different expressions/data
    p1 = plot3d_revolution(
        cos(t), (t, 0, pi), show_curve=False, parallel_axis="z", **options
    )
    assert len(p1.series) == 1
    assert isinstance(p1[0], ParametricSurfaceSeries)
    xx1, yy1, zz1, tt1, pp1 = p1[0].get_data()
    assert np.allclose([xx1.min(), xx1.max()], [-np.pi, np.pi])
    assert np.allclose([yy1.min(), yy1.max()], [-np.pi, np.pi])
    assert np.allclose([zz1.min(), zz1.max()], [-1, 1])
    assert np.allclose([tt1.min(), tt1.max()], [0, np.pi])
    assert np.allclose([pp1.min(), pp1.max()], [0, 2 * np.pi])

    p2 = plot3d_revolution(
        cos(t), (t, 0, pi), show_curve=False, parallel_axis="x", **options
    )
    xx2, yy2, zz2, tt2, pp2 = p2[0].get_data()
    assert p2[0].expr != p1[0].expr

    p3 = plot3d_revolution(
        cos(t), (t, 0, pi), show_curve=False, parallel_axis="y", **options
    )
    xx3, yy3, zz3, tt3, pp3 = p3[0].get_data()
    assert p3[0].expr != p1[0].expr
    assert p3[0].expr != p2[0].expr

    # by setting axis it produces different data
    p4 = plot3d_revolution(
        cos(t), (t, 0, pi),
        show_curve=False, parallel_axis="z", axis=(2, 1), **options
    )
    xx4, yy4, zz4, tt4, pp4 = p4[0].get_data()
    assert not np.allclose(xx4, xx1)
    assert not np.allclose(yy4, yy1)
    assert np.allclose(zz4, zz1)
    assert np.allclose(tt4, tt1)
    assert np.allclose(pp4, pp1)

    p5 = plot3d_revolution(
        cos(t), (t, 0, pi),
        show_curve=False, parallel_axis="x", axis=(2, 1), **options
    )
    xx5, yy5, zz5, tt5, pp5 = p5[0].get_data()
    assert np.allclose(xx5, xx2)
    assert not np.allclose(yy5, yy2)
    assert not np.allclose(zz5, zz2)
    assert np.allclose(tt5, tt2)
    assert np.allclose(pp5, pp2)

    p6 = plot3d_revolution(
        cos(t), (t, 0, pi),
        show_curve=False, parallel_axis="y", axis=(2, 1), **options
    )
    xx6, yy6, zz6, tt6, pp6 = p6[0].get_data()
    assert not np.allclose(xx6, xx1)
    assert not np.allclose(yy6, yy1)
    assert not np.allclose(zz6, zz1)
    assert np.allclose(tt6, tt1)
    assert np.allclose(pp6, pp1)

    # wrong parallel_axis
    raises(
        ValueError,
        lambda: plot3d_revolution(
            cos(t), (t, 0, pi),
            parallel_axis="a", **options),
    )

    # show_curve should add a Parametric3DLineSeries
    # keyword arguments should gets redirected to plot3d_parametric_surface
    # or plot3d_parametric_line
    p = plot3d_revolution(
        cos(t), (t, 0, pi),
        show_curve=True,
        rendering_kw={"color": "r"},
        curve_kw={"rendering_kw": {"color": "g"}},
        **options
    )
    assert len(p.series) == 2
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert isinstance(p[1], Parametric3DLineSeries)
    assert p[1].expr == (t, 0, cos(t))
    assert p[0].rendering_kw == {"color": "r"}
    assert p[1].rendering_kw == {"color": "g"}

    # wireframe should add other series
    p = plot3d_revolution(
        cos(t), (t, 0, pi), show_curve=True, wireframe=True, **options
    )
    assert len(p.series) > 2
    assert all(isinstance(t, Parametric3DLineSeries) for t in p.series[1:])


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings("ignore:The following")
@pytest.mark.filterwarnings("ignore:NumPy is unable to evaluate")
def test_plot3d_revolution_interactive(paf_options, pi_options):
    # plot3d_revolution is going to call plot3d_parametric_surface and
    # plot3d_parametric_line: let's check that the data series are correct.

    t, phi = symbols("t, phi")

    options = pi_options.copy()
    options["n"] = 5
    # interactive widget plot
    k = symbols("k")
    p1 = plot3d_revolution(
        cos(k * t),
        (t, 0, pi),
        params={k: (1, 0, 2)},
        parallel_axis="x",
        show_curve=False,
        **options
    )
    assert isinstance(p1, spb.interactive.panel.InteractivePlot)
    assert len(p1.backend.series) == 1
    s = p1.backend[0]
    assert isinstance(s, ParametricSurfaceSeries) and s.is_interactive

    p2 = plot3d_revolution(
        cos(k * t),
        (t, 0, pi),
        params={k: (1, 0, 2)},
        parallel_axis="y",
        show_curve=False,
        **options
    )
    assert isinstance(p2, spb.interactive.panel.InteractivePlot)
    assert p2.backend[0].expr != p1.backend[0].expr

    p3 = plot3d_revolution(
        cos(k * t),
        (t, 0, pi),
        params={k: (1, 0, 2)},
        parallel_axis="z",
        show_curve=False,
        **options
    )
    assert isinstance(p3, spb.interactive.panel.InteractivePlot)
    assert p3.backend[0].expr != p1.backend[0].expr
    assert p3.backend[0].expr != p2.backend[0].expr

    # show_curve should add a Parametric3DLineInteractiveSeries
    # keyword arguments should gets redirected to plot3d_parametric_surface
    # or plot3d_parametric_line
    p = plot3d_revolution(
        cos(k * t),
        (t, 0, pi),
        params={k: (1, 0, 2)},
        show_curve=True,
        rendering_kw={"color": "r"},
        curve_kw={"rendering_kw": {"color": "g"}},
        **options
    )
    assert isinstance(p, spb.interactive.panel.InteractivePlot)
    assert len(p.backend.series) == 2
    assert (
        isinstance(p.backend[0], ParametricSurfaceSeries)
        and p.backend[0].is_interactive
    )
    assert (
        isinstance(p.backend[1], Parametric3DLineSeries) and
        p.backend[1].is_interactive
    )
    assert p.backend[1].expr == (t, 0, cos(k * t))
    assert p.backend[0].rendering_kw == {"color": "r"}
    assert p.backend[1].rendering_kw == {"color": "g"}

    # wireframe should add other series
    p = plot3d_revolution(
        cos(k * t),
        (t, 0, pi),
        params={k: (1, 0, 2)},
        show_curve=True,
        wireframe=True,
        **options
    )
    assert isinstance(p, spb.interactive.panel.InteractivePlot)
    assert len(p.backend.series) > 2
    assert all(
        isinstance(t, Parametric3DLineSeries) and t.is_interactive
        for t in p.backend.series[1:]
    )


def test_plot3d_spherical(p_options):
    # plot3d_spherical is going to call plot3d_parametric_surface: let's
    # check that the data series are correct.

    phi, theta = symbols("phi, theta")
    p = plot3d_spherical(
        1, (theta, pi / 4, 3 * pi / 4), (phi, pi / 2, 3 * pi / 2), **p_options
    )
    assert len(p.series) == 1
    assert isinstance(p[0], ParametricSurfaceSeries)
    # verify spherical to cartesian transformation
    assert p[0].expr_x == sin(theta) * cos(phi)
    assert p[0].expr_y == sin(theta) * sin(phi)
    assert p[0].expr_z == cos(theta)
    # verify ranges are set correctly
    assert (p[0].var_u == theta) and (p[0].var_v == phi)
    assert np.allclose(
        [p[0].start_u, p[0].end_u],
        [float(pi / 4), float(3 * pi / 4)])
    assert np.allclose(
        [p[0].start_v, p[0].end_v],
        [float(pi / 2), float(3 * pi / 2)])


def test_plot3d_wireframe_transform_function():
    # verify that when plot3d and wireframe=True are used together with
    # transformation functions (tx, ty, tz), those are also applied to
    # wireframe lines.

    x, y = symbols("x, y")

    fx = lambda t: t * 2
    fy = lambda t: t * 3
    fz = lambda t: t * 4
    p = plot3d(
        cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        n1=10, n2=10,
        wireframe=True,
        wf_n1=3, wf_n2=3,
        show=False,
        tx=fx, ty=fy, tz=fz,
    )
    assert all(
        (s.tx == fx) and (s.ty == fy) and (s.tz == fz) for s in p.series)


def test_plot3d_plot_contour_base_scalars(paf_options):
    # verify that these functions are able to deal with base scalars

    options = paf_options.copy()
    options["n"] = 10

    C = CoordSys3D("")
    x, y, z = C.base_scalars()
    plot_contour(
        cos(x * y), (x, -2, 2), (y, -2, 2), use_latex=True, **options)
    plot_contour(
        cos(x * y), (x, -2, 2), (y, -2, 2), use_latex=False, **options)
    plot3d(cos(x * y), (x, -2, 2), (y, -2, 2), use_latex=True, **options)
    plot3d(cos(x * y), (x, -2, 2), (y, -2, 2), use_latex=False, **options)


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings("ignore:The following keyword arguments are unused.")
def test_functions_iplot_integration(pi_options):
    # verify the integration between most important plot functions and iplot

    u, v, x, y, z = symbols("u, v, x, y, z")

    p = plot3d_parametric_line(
        cos(u * x), sin(x), u * x, params={u: (1, 0, 2)}, **pi_options
    )
    assert isinstance(p, spb.interactive.panel.InteractivePlot)

    p = plot3d(cos(u * x**2 + y**2), params={u: (1, 0, 2)}, **pi_options)
    assert isinstance(p, spb.interactive.panel.InteractivePlot)

    p = plot_contour(cos(u * x**2 + y**2), params={u: (1, 0, 2)}, **pi_options)
    assert isinstance(p, spb.interactive.panel.InteractivePlot)

    r = 2 + sin(7 * u + 5 * v)
    expr = (r * cos(x * u) * sin(v), r * sin(x * u) * sin(v), r * cos(v))
    p = plot3d_parametric_surface(
        *expr, (u, 0, 2 * pi), (v, 0, pi), params={x: (1, 0, 2)}, **pi_options
    )
    assert isinstance(p, spb.interactive.panel.InteractivePlot)

    p = lambda: plot3d_implicit(
        u * x**2 + y**3 - z**2,
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        params={u: (1, 0, 2)},
        **pi_options
    )
    raises(NotImplementedError, p)


@pytest.mark.skipif(pn is None, reason="panel is not installed")
def test_plot3d_wireframe(pi_options):
    # verify that wireframe=True produces the expected data series
    x, y, u = symbols("x, y, u")
    pi_options["n"] = 10

    _plot3d = lambda wf: plot3d(
        cos(u * x**2 + y**2),
        (x, -2, 2), (y, -2, 2),
        params={u: (1, 0, 2)},
        wireframe=wf,
        **pi_options
    )
    t = _plot3d(False)
    assert isinstance(t, spb.interactive.panel.InteractivePlot)
    assert len(t.backend.series) == 1

    t = _plot3d(True)
    assert isinstance(t, spb.interactive.panel.InteractivePlot)
    assert len(t.backend.series) == 1 + 10 + 10
    assert isinstance(t.backend.series[0], SurfaceOver2DRangeSeries)
    assert all(
        isinstance(s, Parametric3DLineSeries) and s.is_interactive
        for s in t.backend.series[1:]
    )

    # wireframe lines works even when interactive ranges are used
    a, b = symbols("a, b")
    p = plot3d(
        cos(u * x**2 + y**2),
        prange(x, -2 * a, 2 * b),
        prange(y, -2 * b, 2 * a),
        params={
            u: (1, 0, 2),
            a: (1, 0, 2),
            b: (1, 0, 2),
        },
        wireframe=True,
        **pi_options
    )
    assert isinstance(p.backend[1], Parametric3DLineSeries)
    d1 = p.backend[1].get_data()
    p.backend.update_interactive({u: 1, a: 0.5, b: 1.5})
    d2 = p.backend[1].get_data()
    for s, t in zip(d1, d2):
        assert not np.allclose(s, t)


@pytest.mark.skipif(pn is None, reason="panel is not installed")
def test_plot3d_parametric_surface_wireframe(pi_options):
    # verify that wireframe=True produces the expected data series
    x, y, u = symbols("x, y, u")
    pi_options["n"] = 10
    pi_options.pop("adaptive")

    _plot3d_ps = lambda wf: plot3d_parametric_surface(
        u * x * cos(y), x * sin(y), x * cos(4 * y) / 2,
        (x, 0, pi), (y, 0, 2 * pi),
        params={u: (1, 0, 2)},
        wireframe=wf,
        **pi_options
    )
    t = _plot3d_ps(False)
    assert isinstance(t, spb.interactive.panel.InteractivePlot)
    assert len(t.backend.series) == 1

    t = _plot3d_ps(True)
    assert isinstance(t, spb.interactive.panel.InteractivePlot)
    assert len(t.backend.series) == 1 + 10 + 10
    assert isinstance(t.backend.series[0], ParametricSurfaceSeries)
    assert all(
        isinstance(s, Parametric3DLineSeries) and s.is_interactive
        for s in t.backend.series[1:]
    )

    # wireframe lines works even when interactive ranges are used
    a, b = symbols("a, b")
    p = plot3d_parametric_surface(
        u * x * cos(y), x * sin(y), x * cos(4 * y) / 2,
        prange(x, -2 * a, 2 * b),
        prange(y, -2 * b, 2 * a),
        params={
            u: (1, 0, 2),
            a: (1, 0, 2),
            b: (1, 0, 2),
        },
        wireframe=True,
        **pi_options
    )
    assert isinstance(p.backend[1], Parametric3DLineSeries)
    d1 = p.backend[1].get_data()
    p.backend.update_interactive({u: 1, a: 0.5, b: 1.5})
    d2 = p.backend[1].get_data()
    for s, t in zip(d1, d2):
        assert not np.allclose(s, t)


@pytest.mark.skipif(pn is None, reason="panel is not installed")
def test_plot3d_wireframe_and_labels(pi_options):
    # verify that `wireframe=True` produces the expected data series even when
    # `label` is set
    x, y, u = symbols("x, y, u")
    pi_options["n"] = 10

    t = plot3d(
        cos(u * x**2 + y**2),
        (x, -2, 2), (y, -2, 2),
        params={u: (1, 0, 2)},
        wireframe=True,
        label="test",
        **pi_options
    )
    assert isinstance(t, spb.interactive.panel.InteractivePlot)
    assert len(t.backend.series) == 1 + 10 + 10
    assert isinstance(t.backend.series[0], SurfaceOver2DRangeSeries)
    assert t.backend.series[0].get_label(False) == "test"
    assert all(
        isinstance(s, Parametric3DLineSeries) and s.is_interactive
        for s in t.backend.series[1:]
    )

    t = plot3d(
        (cos(u * x**2 + y**2), (x, -2, 0), (y, -2, 2)),
        (cos(u * x**2 + y**2), (x, 0, 2), (y, -2, 2)),
        params={u: (1, 0, 2)},
        wireframe=True,
        label=["a", "b"],
        **pi_options
    )
    assert isinstance(t, spb.interactive.panel.InteractivePlot)
    assert len(t.backend.series) == 2 + (10 + 10) * 2
    surfaces = [
        s for s in t.backend.series if isinstance(s, SurfaceOver2DRangeSeries)]
    assert len(surfaces) == 2
    assert [s.get_label(False) for s in surfaces] == ["a", "b"]
    wireframe_lines = [
        s for s in t.backend.series if isinstance(s, Parametric3DLineSeries)
    ]
    assert len(wireframe_lines) == (10 + 10) * 2


@pytest.mark.filterwarnings("ignore:The following keyword arguments are unused.")
def test_number_discretization_points():
    # verify the different ways of setting the numbers of discretization points
    x, y, z = symbols("x, y, z")
    options = dict(adaptive=False, show=False, backend=MB)

    p = plot3d_parametric_line(cos(x), sin(x), x, (x, 0, 2 * pi), **options)
    assert p[0].n[0] == 1000
    p = plot3d_parametric_line(
        cos(x), sin(x), x, (x, 0, 2 * pi), n=10, **options)
    assert p[0].n[0] == 10
    p = plot3d_parametric_line(
        cos(x), sin(x), x, (x, 0, 2 * pi), n1=10, **options)
    assert p[0].n[0] == 10
    p = plot3d_parametric_line(
        cos(x), sin(x), x, (x, 0, 2 * pi), nb_of_points=10, **options
    )
    assert p[0].n[0] == 10
    p = plot3d_parametric_line(
        (cos(x), sin(x), x),
        (sin(x), cos(x), x**2),
        (x, 0, 2 * pi),
        n=10, **options
    )
    assert (len(p.series) > 1) and all(t.n[0] == 10 for t in p.series)

    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), **options)
    assert p[0].n[:2] == [100, 100]
    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), n=400, **options)
    assert p[0].n[:2] == [400, 400]
    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), n1=50, **options)
    assert p[0].n[:2] == [50, 100]
    p = plot3d(
        cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), n1=50, n2=20, **options
    )
    assert p[0].n[:2] == [50, 20]
    p = plot3d(
        cos(x**2 + y**2),
        (x, -pi, pi), (y, -pi, pi),
        nb_of_points_x=50, nb_of_points_y=20,
        **options
    )
    assert p[0].n[:2] == [50, 20]
    p = plot3d(
        cos(x**2 + y**2),
        (x, -pi, pi), (y, -pi, pi),
        n1=50, n2=20,
        wireframe=True,
        wf_n1=5, wf_n2=5,
        **options
    )
    assert p[0].n[:2] == [50, 20]
    assert all(s.n[0] == 20 for s in p.series[1:6])
    assert all(s.n[0] == 50 for s in p.series[6:])
    assert p[0].n[:2] == [50, 20]
    p = plot3d(
        cos(x**2 + y**2), sin(x**2 + y**2),
        (x, -pi, pi), (y, -pi, pi),
        n=400,
        **options
    )
    assert (len(p.series) > 1) and all(t.n[:2] == [400, 400] for t in p.series)

    p = plot3d_parametric_surface(
        x + y, x - y, x, (x, -pi, pi), (y, -pi, pi), **options
    )
    assert p[0].n[:2] == [100, 100]
    p = plot3d_parametric_surface(
        x + y, x - y, x, (x, -pi, pi), (y, -pi, pi), n=400, **options
    )
    assert p[0].n[:2] == [400, 400]
    p = plot3d_parametric_surface(
        x + y, x - y, x, (x, -pi, pi), (y, -pi, pi), n1=50, **options
    )
    assert p[0].n[:2] == [50, 100]
    p = plot3d_parametric_surface(
        x + y, x - y, x, (x, -pi, pi), (y, -pi, pi), n1=50, n2=20, **options
    )
    assert p[0].n[:2] == [50, 20]
    p = plot3d_parametric_surface(
        x + y, x - y, x,
        (x, -pi, pi), (y, -pi, pi),
        nb_of_points_u=50,
        nb_of_points_v=20,
        **options
    )
    assert p[0].n[:2] == [50, 20]
    p = plot3d_parametric_surface(
        x + y, x - y, x,
        (x, -pi, pi), (y, -pi, pi),
        n1=50, n2=20,
        wireframe=True,
        wf_n1=5, wf_n2=5,
        **options
    )
    assert p[0].n[:2] == [50, 20]
    assert all(s.n[0] == 20 for s in p.series[1:6])
    assert all(s.n[0] == 50 for s in p.series[6:])

    p = plot3d_spherical(1, (x, 0, 0.7 * pi), (y, 0, 1.8 * pi), **options)
    assert p[0].n[:2] == [100, 100]
    p = plot3d_spherical(
        1, (x, 0, 0.7 * pi), (y, 0, 1.8 * pi), n=400, **options)
    assert p[0].n[:2] == [400, 400]
    p = plot3d_spherical(
        1, (x, 0, 0.7 * pi), (y, 0, 1.8 * pi), n1=50, **options)
    assert p[0].n[:2] == [50, 100]
    p = plot3d_spherical(
        1, (x, 0, 0.7 * pi), (y, 0, 1.8 * pi), n1=50, n2=20, **options)
    assert p[0].n[:2] == [50, 20]
    p = plot3d_spherical(
        1,
        (x, 0, 0.7 * pi), (y, 0, 1.8 * pi),
        nb_of_points_u=50, nb_of_points_v=20,
        **options
    )
    assert p[0].n[:2] == [50, 20]

    p = plot3d_implicit(
        x**2 + y**3 - z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2), **options
    )
    assert p[0].n == [60, 60, 60]
    p = plot3d_implicit(
        x**2 + y**3 - z**2,
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        n=400, **options
    )
    assert p[0].n == [400, 400, 400]
    p = plot3d_implicit(
        x**2 + y**3 - z**2,
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        n1=50, **options
    )
    assert p[0].n == [50, 60, 60]
    p = plot3d_implicit(
        x**2 + y**3 - z**2,
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        n1=50, n2=20, n3=10,
        **options
    )
    assert p[0].n == [50, 20, 10]


def test_indexed_objects():
    # verify that plot functions and series correctly process indexed objects
    x = IndexedBase("x")

    p = plot3d(
        cos(x[0] ** 2 + x[1] ** 2),
        (x[0], -pi, pi), (x[1], -pi, pi),
        adaptive=False,
        n=10,
        show=False,
        backend=MB,
        use_latex=False,
    )
    p[0].get_data()
    assert p.xlabel == "x[0]"
    assert p.ylabel == "x[1]"
    assert p.zlabel == "f(x[0], x[1])"


# -----------------------------------------------------------------------------
# ----------------- PLOT - PLOT_PARAMETRIC - PLOT3D-RELATED -------------------
# -----------------------------------------------------------------------------
# These tests comes from the old sympy.plotting module
# Some test have been rearranged to reflect the intentions of this new
# plotting module


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot3d_parametric_line_limits(paf_options, adaptive):
    if adaptive and (adaptive_module is None):
        return

    x = symbols("x")
    paf_options.update({"adaptive": adaptive, "n": 60})

    v1 = (2 * cos(x), 2 * sin(x), 2 * x, (x, -5, 5))
    v2 = (sin(x), cos(x), x, (x, -5, 5))
    p = plot3d_parametric_line(v1, v2, **paf_options)

    xmin, xmax = p.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    ymin, ymax = p.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    zmin, zmax = p.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2

    p = plot3d_parametric_line(v2, v1, **paf_options)

    xmin, xmax = p.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    ymin, ymax = p.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    zmin, zmax = p.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2
