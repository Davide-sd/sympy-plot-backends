import os
import pytest
from pytest import raises, warns
from spb.defaults import set_defaults, cfg
from spb import *
from spb.interactive.panel import InteractivePlot
from spb.series import (
    LineOver1DRangeSeries, List2DSeries, ContourSeries,
    Vector2DSeries, ParametricSurfaceSeries, Parametric3DLineSeries,
    Parametric2DLineSeries, SurfaceOver2DRangeSeries,
    ComplexSurfaceSeries, ComplexParametric3DLineSeries, ImplicitSeries
)
from spb.backends.matplotlib import MB, unset_show
from sympy import (
    symbols, Piecewise, sin, cos, tan, re, Abs, sqrt, real_root,
    Heaviside, exp, log, LambertW, exp_polar, meijerg,
    And, Or, Eq, Ne, Interval, Sum, oo, I, pi, S,
    sympify, Integral, Circle, gamma, Circle, Point, Ellipse, Rational,
    Polygon, Curve, Segment, Point2D, Point3D, Line3D, Plane, IndexedBase,
    Function
)
from sympy.vector import CoordSys3D
from sympy.testing.pytest import skip
from sympy.external import import_module
from sympy.testing.tmpfiles import TmpFileManager
from tempfile import TemporaryDirectory, NamedTemporaryFile, mkdtemp
import pytest
import numpy as np


# NOTE:
#
# These tests are meant to verify that the plotting functions generate the
# correct data series.
# Also, legacy tests from the old sympy.plotting module are included.
#
# If your issue is related to the generation of numerical data from a
# particular data series, consider adding tests to test_series.py.
# If your issue is related to a particular keyword affecting a backend
# behaviour, consider adding tests to test_backends.py
#


@pytest.fixture
def p_options():
    return dict(show=False, backend=MB)


@pytest.fixture
def paf_options(p_options):
    options = p_options.copy()
    options["adaptive"] = False
    options["n"] = 100
    return options


@pytest.fixture
def pi_options(paf_options, panel_options):
    options = paf_options.copy()
    options["imodule"] = panel_options["imodule"]
    return options


@pytest.fixture
def pat_options(p_options):
    options = p_options.copy()
    options["adaptive"] = True
    options["adaptive_goal"] = 0.05
    return options


@pytest.mark.filterwarnings("ignore:The following")
def test_plot_geometry(pi_options):
    # verify that plot_geometry create the correct plot and data series

    x, y, z = symbols('x, y, z')

    # geometric entities defined by numbers
    p = plot_geometry(
        Circle(Point(0, 0), 5),
        Ellipse(Point(-3, 2), hradius=3, eccentricity=Rational(4, 5)),
        Polygon((4, 0), 4, n=5),
        Curve((cos(x), sin(x)), (x, 0, 2 * pi)),
        Segment((-4, -6), (6, 6)),
        Point2D(0, 0), **pi_options)
    assert isinstance(p, MB)
    assert len(p.series) == 6
    assert all(not s.is_interactive for s in p.series)
    assert all(not s.is_3D for s in p.series)

    # symbolic geometric entities
    a, b, c, d = symbols("a, b, c, d")
    p = plot_geometry(
        (Polygon((a, b), c, n=d), "triangle"),
        (Polygon((a + 2, b + 3), c, n=d + 1), "square"),
        params = {a: 0, b: 1, c: 2, d: 3}, **pi_options)
    assert isinstance(p, MB)
    assert len(p.series) == 2
    assert all(not s.is_interactive for s in p.series)
    assert all(not s.is_3D for s in p.series)

    # 3d geometric entities
    p = plot_geometry(
        (Point3D(5, 5, 5), "center"),
        (Line3D(Point3D(-2, -3, -4), Point3D(2, 3, 4)), "line"),
        (Plane((0, 0, 0), (1, 1, 1)),
        (x, -5, 5), (y, -4, 4), (z, -10, 10)), **pi_options)
    assert isinstance(p, MB)
    assert len(p.series) == 3
    assert all(not s.is_interactive for s in p.series)
    assert all(s.is_3D for s in p.series)

    # interactive widget plot
    p = plot_geometry(
        (Polygon((a, b), c, n=d), "a"),
        (Polygon((a + 2, b + 3), c, n=d + 1), "b"),
        params = {
            a: (0, -1, 1),
            b: (1, -1, 1),
            c: (2, 1, 2),
            d: (3, 3, 8)
        },
        aspect="equal", is_filled=False, use_latex=False,
        xlim=(-2.5, 5.5), ylim=(-3, 6.5), **pi_options)
    assert isinstance(p, InteractivePlot)
    assert len(p.backend.series) == 2
    assert all(s.is_interactive for s in p.backend.series)
    assert all(not s.is_3D for s in p.backend.series)


@pytest.mark.filterwarnings("ignore:The following")
def test_plott(p_options):
    ### Test arguments for plot()

    x, y = symbols("x, y")

    # single expressions
    p = plot(x + 1, **p_options)
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x + 1"
    assert p[0].rendering_kw == {}

    # single expressions custom label
    p = plot(x + 1, "label", **p_options)
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "label"
    assert p[0].rendering_kw == {}

    # single expressions with range
    p = plot(x + 1, (x, -2, 2), **p_options)
    assert p[0].ranges == [(x, -2, 2)]

    # single expressions with range, label and rendering-kw dictionary
    p = plot(x + 1, (x, -2, 2), "test", {"color": "r"}, **p_options)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"color": "r"}

    # multiple expressions
    p = plot(x + 1, x**2, **p_options)
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x + 1"
    assert p[0].rendering_kw == {}
    assert isinstance(p[1], LineOver1DRangeSeries)
    assert p[1].expr == x**2
    assert p[1].ranges == [(x, -10, 10)]
    assert p[1].get_label(False) == "x**2"
    assert p[1].rendering_kw == {}

    # multiple expressions over the same range
    p = plot(x + 1, x**2, (x, 0, 5), **p_options)
    assert p[0].ranges == [(x, 0, 5)]
    assert p[1].ranges == [(x, 0, 5)]

    # multiple expressions over the same range with the same rendering kws
    p = plot(x + 1, x**2, (x, 0, 5), {"color": "r"}, **p_options)
    assert p[0].ranges == [(x, 0, 5)]
    assert p[1].ranges == [(x, 0, 5)]
    assert p[0].rendering_kw == {"color": "r"}
    assert p[1].rendering_kw == {"color": "r"}

    # multiple expressions with different ranges, labels and rendering kws
    p = plot(
        (x + 1, (x, 0, 5)),
        (x**2, (x, -2, 2), "test", {"color": "r"}), **p_options)
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, 0, 5)]
    assert p[0].get_label(False) == "x + 1"
    assert p[0].rendering_kw == {}
    assert isinstance(p[1], LineOver1DRangeSeries)
    assert p[1].expr == x**2
    assert p[1].ranges == [(x, -2, 2)]
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {"color": "r"}

    # single argument: lambda function
    f = lambda t: t
    p = plot(lambda t: t, **p_options)
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert callable(p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}

    # single argument: lambda function + custom range and label
    p = plot(f, ("t", -5, 6), "test", **p_options)
    assert p[0].ranges[0][1:] == (-5, 6)
    assert p[0].get_label(False) == "test"


@pytest.mark.filterwarnings("ignore:The following")
def test_plot_parametric(p_options):
    ### Test arguments for plot_parametric()

    x, y = symbols("x, y")

    # single parametric expression
    p = plot_parametric(x + 1, x, **p_options)
    assert isinstance(p[0], Parametric2DLineSeries)
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}

    # single parametric expression with custom range, label and rendering kws
    p = plot_parametric(x + 1, x, (x, -2, 2), "test",
        {"cmap": "Reds"}, **p_options)
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    p = plot_parametric((x + 1, x), (x, -2, 2), "test", **p_options)
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}

    # multiple parametric expressions same symbol
    p = plot_parametric((x + 1, x), (x ** 2, x + 1), **p_options)
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -10, 10)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions different symbols
    p = plot_parametric((x + 1, x), (y ** 2, y + 1, "test"), **p_options)
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (y ** 2, y + 1)
    assert p[1].ranges == [(y, -10, 10)]
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions same range
    p = plot_parametric((x + 1, x), (x ** 2, x + 1), (x, -2, 2), **p_options)
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -2, 2)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions, custom ranges and labels
    p = plot_parametric(
        (x + 1, x, (x, -2, 2), "test1"),
        (x ** 2, x + 1, (x, -3, 3), "test2", {"cmap": "Reds"}), **p_options)
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test1"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -3, 3)]
    assert p[1].get_label(False) == "test2"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # single argument: lambda function
    fx = lambda t: t
    fy = lambda t: 2 * t
    p = plot_parametric(fx, fy, **p_options)
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert "Dummy" in p[0].get_label(False)
    assert p[0].rendering_kw == {}

    # single argument: lambda function + custom range + label
    p = plot_parametric(fx, fy, ("t", 0, 2), "test", **p_options)
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (0, 2)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}


@pytest.mark.filterwarnings("ignore:The following")
def test_plot3d_parametric_line(p_options):
    ### Test arguments for plot3d_parametric_line()

    x, y = symbols("x, y")

    # single parametric expression
    p = plot3d_parametric_line(x + 1, x, sin(x), **p_options)
    assert isinstance(p[0], Parametric3DLineSeries)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}

    # single parametric expression with custom range, label and rendering kws
    p = plot3d_parametric_line(x + 1, x, sin(x), (x, -2, 2),
        "test", {"cmap": "Reds"}, **p_options)
    assert isinstance(p[0], Parametric3DLineSeries)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    p = plot3d_parametric_line((x + 1, x, sin(x)), (x, -2, 2), "test", **p_options)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}

    # multiple parametric expression same symbol
    p = plot3d_parametric_line(
        (x + 1, x, sin(x)), (x ** 2, 1, cos(x), {"cmap": "Reds"}), **p_options)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, 1, cos(x))
    assert p[1].ranges == [(x, -10, 10)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # multiple parametric expression different symbols
    p = plot3d_parametric_line((x + 1, x, sin(x)), (y ** 2, 1, cos(y)), **p_options)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (y ** 2, 1, cos(y))
    assert p[1].ranges == [(y, -10, 10)]
    assert p[1].get_label(False) == "y"
    assert p[1].rendering_kw == {}

    # multiple parametric expression, custom ranges and labels
    p = plot3d_parametric_line(
        (x + 1, x, sin(x)),
        (x ** 2, 1, cos(x), (x, -2, 2), "test", {"cmap": "Reds"}), **p_options)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, 1, cos(x))
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
    ### Test arguments for plot3d() and plot_contour()

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
        (x * y, (x, -3, 3), (y, -6, 6), "test", {"cmap": "Reds"}), **p_options)
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
    ### Test arguments for plot3d_parametric_surface()

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
    p = plot3d_parametric_surface(x + y, cos(x + y), sin(x + y),
        (x, -2, 2), (y, -4, 4), "test", {"cmap": "Reds"}, **p_options)
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    # multiple parametric expressions
    p = plot3d_parametric_surface(
        (x + y, cos(x + y), sin(x + y)),
        (x - y, cos(x - y), sin(x - y), "test"), **p_options)
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
        (x - y, cos(x - y), sin(x - y), (x, -3, 3), (y, -4, 4), "test2", {"cmap": "Reds"}),
        **p_options)
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
        lambda u, v: u, lambda u, v: v, lambda u, v: u + v,
        ("u", 0, 2), ("v", -3, 4), **p_options)
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (-0, 2)
    assert p[0].ranges[1][1:] == (-3, 4)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}

    # lambda functions instead of symbolic expressions for multiple 3D
    # parametric surfaces
    p = plot3d_parametric_surface(
        (lambda u, v: u, lambda u, v: v, lambda u, v: u + v,
        ("u", 0, 2), ("v", -3, 4)),
        (lambda u, v: v, lambda u, v: u, lambda u, v: u - v,
        ("u", -2, 3), ("v", -4, 5), "test"), **p_options)
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


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_implicit(p_options):
    ### Test arguments for plot_implicit

    x, y = symbols("x, y")

    # single expression with both ranges
    p = plot_implicit(x > 0, (x, -2, 2), (y, -3, 3), **p_options)
    assert isinstance(p[0], ImplicitSeries)
    assert p[0].ranges == [(x, -2, 2), (y, -3, 3)]
    assert p[0].get_label(False) == "x > 0"
    assert p[0].rendering_kw == {}
    assert p.xlim and p.ylim

    # single expression with one missing range
    p = plot_implicit(x > y, (x, -2, 2), "test", {"color": "k"}, **p_options)
    assert isinstance(p[0], ImplicitSeries)
    assert p[0].ranges == [(x, -2, 2), (y, -10, 10)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"color": "k"}
    assert p.xlim and p.ylim

    # multiple expressions
    with warns(
            UserWarning,
            match="The provided expression"
        ):
        p = plot_implicit(
            (x > 0, (x, -2, 2), (y, -3, 3)),
            ((x > 0) & (y < 0), (x, -10, 10), "test", {"color": "r"}), **p_options)
    assert isinstance(p[0], ImplicitSeries)
    assert p[0].ranges == [(x, -2, 2), (y, -3, 3)]
    assert p[0].get_label(False) == "x > 0"
    assert p[0].rendering_kw == {}
    assert isinstance(p[1], ImplicitSeries)
    assert p[1].ranges == [(x, -10, 10), (y, -10, 10)]
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {"color": "r"}
    assert p.xlim and p.ylim

    # incompatible free symbols between expression and ranges
    z = symbols("z")
    raises(ValueError,
        lambda: plot_implicit(x * y > 0, (x, -2, 2), (z, -3, 3), **p_options))


@pytest.mark.filterwarnings("ignore:The following")
def test_plot_parametric_region(p_options, pi_options):
    # verify that plot_parametric_region creates the correct data series

    u, v, x, y = symbols("u, v, x, y")

    # single parametric region
    p = plot_parametric_region(u * cos(v), u * sin(v),
        (u, 1, 2), (v, 0, 2*pi/3), {"color": "k"},
        n1=3, n2=5, n=10, **p_options)
    assert len(p.series) == 8
    assert all(isinstance(s, Parametric2DLineSeries) for s in p.series)
    assert all((s.var, s.start, s.end) == (v, 0, float(2*pi/3)) for s in p.series[:3])
    assert all((s.var, s.start, s.end) == (u, 1, 2) for s in p.series[3:])
    assert all(s.n[0] == 10 for s in p.series)
    assert all(s.rendering_kw == {"color": "k"} for s in p.series)

    p = plot_parametric_region(u * cos(v), u * sin(v),
        (u, 1, 2), (v, 0, 2*pi/3), {"color": "k"},
        rkw_u={"color": "r"}, rkw_v={"color": "b"},
        n1=3, n2=5, n=10, **p_options)
    assert all(s.rendering_kw == {"color": "r"} for s in p.series[:3])
    assert all(s.rendering_kw == {"color": "b"} for s in p.series[3:])

    # multiple parametric regions with the same ranges
    p = plot_parametric_region(
        (u * cos(v), u * sin(v)),
        (2 * u * cos(v), 2 * u * sin(v)),
        (u, 1, 2), (v, 0, 2*pi/3), {"color": "k"},
        n1=3, n2=5, n=10, **p_options)
    assert len(p.series) == 16
    assert sum((s.var, s.start, s.end) == (u, 1, 2) for s in p.series) == 10
    assert sum((s.var, s.start, s.end) == (v, 0, float(2*pi/3)) for s in p.series) == 6

    # multiple parametric regions each one with its own ranges
    p = plot_parametric_region(
        (u * cos(v), u * sin(v), (u, 1, 2), (v, 0, 2*pi/3)),
        (2 * x * cos(y), 2 * x * sin(y), (x, 0, 1), (y, 0, pi)),
        n1=3, n2=5, n=10, **p_options)
    assert len(p.series) == 16
    assert sum((s.var, s.start, s.end) == (u, 1, 2) for s in p.series) == 5
    assert sum((s.var, s.start, s.end) == (v, 0, float(2*pi/3)) for s in p.series) == 3
    assert sum((s.var, s.start, s.end) == (x, 0, 1) for s in p.series) == 5
    assert sum((s.var, s.start, s.end) == (y, 0, float(pi)) for s in p.series) == 3

    # parametric interactive plot
    p = lambda: plot_parametric_region(u * cos(x * v), u * sin(v),
        (u, 1, 2), (v, 0, 2*pi/3), n1=3, n2=5,
        params={x: (1, 0, 2)}, **pi_options)
    raises(NotImplementedError, p)



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
    assert (p[0].var_u, p[0].start_u, p[0].end_u) == (t, 0, float(pi))
    assert (p[0].var_v, p[0].start_v, p[0].end_v) == (phi, 0, float(2*pi))

    p = plot3d_revolution(cos(t), (t, 0, pi), (phi, 0, pi/2), **options)
    assert len(p.series) == 1
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert (p[0].var_u, p[0].start_u, p[0].end_u) == (t, 0, float(pi))
    assert (p[0].var_v, p[0].start_v, p[0].end_v) == (phi, 0, float(pi / 2))

    # by setting parallel_axis it produces different expressions/data
    p1 = plot3d_revolution(cos(t), (t, 0, pi),
        show_curve=False, parallel_axis="z", **options)
    assert len(p1.series) == 1
    assert isinstance(p1[0], ParametricSurfaceSeries)
    xx1, yy1, zz1, tt1, pp1 = p1[0].get_data()
    assert np.allclose([xx1.min(), xx1.max()], [-np.pi, np.pi])
    assert np.allclose([yy1.min(), yy1.max()], [-np.pi, np.pi])
    assert np.allclose([zz1.min(), zz1.max()], [-1, 1])
    assert np.allclose([tt1.min(), tt1.max()], [0, np.pi])
    assert np.allclose([pp1.min(), pp1.max()], [0, 2 * np.pi])

    p2 = plot3d_revolution(cos(t), (t, 0, pi),
        show_curve=False, parallel_axis="x", **options)
    xx2, yy2, zz2, tt2, pp2 = p2[0].get_data()
    assert p2[0].expr != p1[0].expr

    p3 = plot3d_revolution(cos(t), (t, 0, pi),
        show_curve=False, parallel_axis="y", **options)
    xx3, yy3, zz3, tt3, pp3 = p3[0].get_data()
    assert p3[0].expr != p1[0].expr
    assert p3[0].expr != p2[0].expr

    # by setting axis it produces different data
    p4 = plot3d_revolution(cos(t), (t, 0, pi),
        show_curve=False, parallel_axis="z", axis=(2, 1), **options)
    xx4, yy4, zz4, tt4, pp4 = p4[0].get_data()
    assert not np.allclose(xx4, xx1)
    assert not np.allclose(yy4, yy1)
    assert np.allclose(zz4, zz1)
    assert np.allclose(tt4, tt1)
    assert np.allclose(pp4, pp1)

    p5 = plot3d_revolution(cos(t), (t, 0, pi),
        show_curve=False, parallel_axis="x", axis=(2, 1), **options)
    xx5, yy5, zz5, tt5, pp5 = p5[0].get_data()
    assert np.allclose(xx5, xx2)
    assert not np.allclose(yy5, yy2)
    assert not np.allclose(zz5, zz2)
    assert np.allclose(tt5, tt2)
    assert np.allclose(pp5, pp2)

    p6 = plot3d_revolution(cos(t), (t, 0, pi),
        show_curve=False, parallel_axis="y", axis=(2, 1), **options)
    xx6, yy6, zz6, tt6, pp6 = p6[0].get_data()
    assert not np.allclose(xx6, xx1)
    assert not np.allclose(yy6, yy1)
    assert not np.allclose(zz6, zz1)
    assert np.allclose(tt6, tt1)
    assert np.allclose(pp6, pp1)

    # wrong parallel_axis
    raises(ValueError, lambda : plot3d_revolution(
        cos(t), (t, 0, pi), parallel_axis="a", **options))

    # show_curve should add a Parametric3DLineSeries
    # keyword arguments should gets redirected to plot3d_parametric_surface
    # or plot3d_parametric_line
    p = plot3d_revolution(cos(t), (t, 0, pi),
        show_curve=True, rendering_kw={"color": "r"},
        curve_kw={"rendering_kw": {"color": "g"}}, **options)
    assert len(p.series) == 2
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert isinstance(p[1], Parametric3DLineSeries)
    assert p[1].expr == (t, 0, cos(t))
    assert p[0].rendering_kw == {"color": "r"}
    assert p[1].rendering_kw == {"color": "g"}

    # wireframe should add other series
    p = plot3d_revolution(cos(t), (t, 0, pi),
        show_curve=True, wireframe=True, **options)
    assert len(p.series) > 2
    assert all(isinstance(t, Parametric3DLineSeries) for t in p.series[1:])

    options = pi_options.copy()
    options["n"] = 5
    # interactive widget plot
    k = symbols("k")
    p1 = plot3d_revolution(cos(k * t), (t, 0, pi),
        params={k: (1, 0, 2)}, parallel_axis="x", show_curve=False, **options)
    assert isinstance(p1, InteractivePlot)
    assert len(p1.backend.series) == 1
    s = p1.backend[0]
    assert isinstance(s, ParametricSurfaceSeries) and s.is_interactive

    p2 = plot3d_revolution(cos(k * t), (t, 0, pi),
        params={k: (1, 0, 2)}, parallel_axis="y", show_curve=False, **options)
    assert isinstance(p2, InteractivePlot)
    assert p2.backend[0].expr != p1.backend[0].expr

    p3 = plot3d_revolution(cos(k * t), (t, 0, pi),
        params={k: (1, 0, 2)}, parallel_axis="z", show_curve=False, **options)
    assert isinstance(p3, InteractivePlot)
    assert p3.backend[0].expr != p1.backend[0].expr
    assert p3.backend[0].expr != p2.backend[0].expr

    # show_curve should add a Parametric3DLineInteractiveSeries
    # keyword arguments should gets redirected to plot3d_parametric_surface
    # or plot3d_parametric_line
    p = plot3d_revolution(cos(k * t), (t, 0, pi),
        params={k: (1, 0, 2)}, show_curve=True, rendering_kw={"color": "r"},
        curve_kw={"rendering_kw": {"color": "g"}}, **options)
    assert isinstance(p, InteractivePlot)
    assert len(p.backend.series) == 2
    assert isinstance(p.backend[0], ParametricSurfaceSeries) and p.backend[0].is_interactive
    assert isinstance(p.backend[1], Parametric3DLineSeries) and p.backend[1].is_interactive
    assert p.backend[1].expr == (t, 0, cos(k * t))
    assert p.backend[0].rendering_kw == {"color": "r"}
    assert p.backend[1].rendering_kw == {"color": "g"}

    # wireframe should add other series
    p = plot3d_revolution(cos(k * t), (t, 0, pi),
        params={k: (1, 0, 2)}, show_curve=True, wireframe=True, **options)
    assert isinstance(p, InteractivePlot)
    assert len(p.backend.series) > 2
    assert all(isinstance(t, Parametric3DLineSeries) and t.is_interactive for t in p.backend.series[1:])


def test_plot3d_spherical(p_options):
    # plot3d_spherical is going to call plot3d_parametric_surface: let's
    # check that the data series are correct.

    phi, theta = symbols("phi, theta")
    p = plot3d_spherical(1, (theta, pi/4, 3*pi/4), (phi, pi/2, 3*pi/2),
        **p_options)
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


def test_plot3d_plot_contour_base_scalars(paf_options):
    # verify that these functions are able to deal with base scalars

    options = paf_options.copy()
    options["n"] = 10

    C = CoordSys3D("")
    x, y, z = C.base_scalars()
    plot_contour(cos(x * y), (x, -2, 2), (y, -2, 2),
        use_latex=True, **options)
    plot_contour(cos(x * y), (x, -2, 2), (y, -2, 2),
        use_latex=False, **options)
    plot3d(cos(x * y), (x, -2, 2), (y, -2, 2),
        use_latex=True, **options)
    plot3d(cos(x * y), (x, -2, 2), (y, -2, 2),
        use_latex=False, **options)


def test_plot_list(p_options):
    # verify that plot_list creates the correct data series

    xx1 = np.linspace(-3, 3)
    yy1 = np.cos(xx1)
    xx2 = np.linspace(-5, 5)
    yy2 = np.sin(xx2)

    p = plot_list(xx1, yy1, **p_options)
    assert len(p.series) == 1
    assert isinstance(p.series[0], List2DSeries)
    assert p.series[0].label == ""

    p = plot_list(xx1, yy1, "test", **p_options)
    assert len(p.series) == 1
    assert isinstance(p.series[0], List2DSeries)
    assert p.series[0].label == "test"

    p = plot_list((xx1, yy1), (xx2, yy2), **p_options)
    assert len(p.series) == 2
    assert all(isinstance(t, List2DSeries) for t in p.series)
    assert all(t.label == "" for t in p.series)

    p = plot_list((xx1, yy1, "cos"), (xx2, yy2, "sin"),
        **p_options)
    assert len(p.series) == 2
    assert all(isinstance(t, List2DSeries) for t in p.series)
    assert p.series[0].label == "cos"
    assert p.series[1].label == "sin"


def test_process_sums(paf_options):
    # verify that Sum containing infinity in its boundary, gets replaced with
    # a Sum with arbitrary big numbers instead.
    x, y = symbols("x, y")

    options = paf_options.copy()
    options["n"] = 20

    expr = Sum(1 / x ** y, (x, 1, oo))
    p1 = plot(expr, (y, 2, 10), sum_bound=1000, **options)
    assert p1[0].expr.args[-1] == (x, 1, 1000)
    p2 = plot(expr, (y, 2, 10), sum_bound=500, **options)
    assert p2[0].expr.args[-1] == (x, 1, 500)

    expr = Sum(1 / x ** y, (x, -oo, -1))
    p1 = plot(expr, (y, 2, 10), sum_bound=1000, **options)
    assert p1[0].expr.args[-1] == (x, -1000, -1)
    p2 = plot(expr, (y, 2, 10), sum_bound=500, **options)
    assert p2[0].expr.args[-1] == (x, -500, -1)

    expr = Sum(1 / x ** y, (x, -oo, oo))
    p1 = plot(expr, (y, 2, 10), sum_bound=1000, **options)
    assert p1[0].expr.args[-1] == (x, -1000, 1000)
    p2 = plot(expr, (y, 2, 10), sum_bound=500, **options)
    assert p2[0].expr.args[-1] == (x, -500, 500)


def test_plot_piecewise(p_options):
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
    p = plot_piecewise(f, (x, -5, 5), **p_options)
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
    p = plot_piecewise(f, (x, -10, 10), **p_options)
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
    p = plot_piecewise(f, (x, -10, 10), **p_options)
    s = p.series
    assert not p.legend
    assert len(s) == 7
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 3
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 4
    assert len([t for t in s if isinstance(t, List2DSeries) and t.is_filled]) == 2

    f = Piecewise((x, x < 1), (x**2, (x >= -1) & (x <= 3)), (x, x > 3))
    p = plot_piecewise(f, (x, -10, 10), **p_options)
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
        **p_options))

    # The range is smaller than the function "domain"
    f = Piecewise(
        (1, x < -5),
        (x, Eq(x, 0)),
        (x**2, Eq(x, 2)),
        (x**3, (x > 0) & (x < 2)),
        (x**4, True)
    )
    p = plot_piecewise(f, (x, -3, 3), **p_options)
    s = p.series
    assert not p.legend
    assert len(s) == 9
    assert len([t for t in s if isinstance(t, LineOver1DRangeSeries)]) == 3
    assert len([t for t in s if isinstance(t, List2DSeries)]) == 6
    assert len([t for t in s if isinstance(t, List2DSeries) and t.is_filled]) == 2


def test_plot_piecewise_dot_false(p_options):
    # Verify that plot_piecewise doesn't create endpoint dots when dots=False.

    x = symbols("x")
    f = Piecewise((x**2, x < 2), (5, Eq(x, 2)), (10 - x, True))
    p = plot_piecewise(f, (x, -2, 5), **p_options)
    assert len(p.series) == 5
    assert len([t for t in p.series if isinstance(t, List2DSeries)]) == 3

    f = Piecewise((x**2, x < 2), (5, Eq(x, 2)), (10 - x, True))
    p = plot_piecewise(f, (x, -2, 5), dots=False, **p_options)
    assert len(p.series) == 3
    assert len([t for t in p.series if isinstance(t, List2DSeries)]) == 1


def test_lambda_functions(p_options):
    # verify that plotting functions raises errors if they do not support
    # lambda functions.

    raises(TypeError, lambda : plot_piecewise(lambda t: t, **p_options))


@pytest.mark.filterwarnings("ignore:The following keyword arguments are unused.")
def test_functions_iplot_integration(pi_options):
    # verify the integration between most important plot functions and iplot

    x, y, z, u, v = symbols("x, y, z, u, v")
    p = plot(cos(u * x), params={u: (1, 0, 2)}, **pi_options)
    assert isinstance(p, InteractivePlot)

    p = plot_parametric(cos(u * x), sin(x), params={u: (1, 0, 2)}, **pi_options)
    assert isinstance(p, InteractivePlot)

    p = plot3d_parametric_line(cos(u * x), sin(x), u * x,
        params={u: (1, 0, 2)}, **pi_options)
    assert isinstance(p, InteractivePlot)

    p = plot3d(cos(u * x**2 + y**2), params={u: (1, 0, 2)}, **pi_options)
    assert isinstance(p, InteractivePlot)

    p = plot_contour(cos(u * x**2 + y**2), params={u: (1, 0, 2)}, **pi_options)
    assert isinstance(p, InteractivePlot)

    r = 2 + sin(7 * u + 5 * v)
    expr = (
        r * cos(x * u) * sin(v),
        r * sin(x * u) * sin(v),
        r * cos(v)
        )
    p = plot3d_parametric_surface(*expr, (u, 0, 2 * pi), (v, 0, pi),
        params={x: (1, 0, 2)}, **pi_options)
    assert isinstance(p, InteractivePlot)

    p = lambda: plot3d_implicit(
        u * x**2 + y**3 - z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2),
        params={u: (1, 0, 2)}, **pi_options)
    raises(NotImplementedError, p)

    # non-boolean expression -> works fine
    p = plot_implicit(Eq(u * x**2 + y**2, 3), (x, -3, 3), (y, -3, 3),
        params={u: (1, 0, 2)}, **pi_options)
    assert isinstance(p, InteractivePlot)
    p.backend.update_interactive({u: 2})

    # boolean expression -> raise error on update
    with warns(
            UserWarning,
            match="The provided expression contains Boolean functions."
        ):
        p = lambda : plot_implicit(
            And(Eq(u * x**2 + y**2, 3), x > y), (x, -3, 3), (y, -3, 3),
            params={u: (1, 0, 2)}, **pi_options)
        raises(NotImplementedError, p)

    p = plot_geometry(Circle((u, 0), 4),
        params={u: (1, 0, 2)}, **pi_options)
    assert isinstance(p, InteractivePlot)

    p = plot_list([1, 2, 3], [4, 5, 6],
        params={u: (1, 0, 2)}, **pi_options)
    assert isinstance(p, InteractivePlot)

    p = lambda: plot_piecewise(
        u * Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10),
        params={u: (1, 0, 2)}, **pi_options)
    raises(NotImplementedError, p)


def test_plot_label_rendering_kw(p_options, pi_options):
    # verify that label and rendering_kw keyword arguments gets applied
    # witht the plot function
    u, x, y = symbols("u, x, y")

    t = plot(
        sin(x), cos(x), (x, -5, 5),
        label=["a", "b"],
        rendering_kw=[{"color": "r"}, {"linestyle": ":"}],
        **p_options)
    assert isinstance(t, MB)
    assert len(t.series) == 2 and all(s.is_2Dline for s in t.series)
    assert [s.label for s in t.series] == ["a", "b"]
    assert t.series[0].rendering_kw == {"color": "r"}
    assert t.series[1].rendering_kw == {"linestyle": ":"}

    t = plot(
        sin(u * x), cos(u * x), (x, -5, 5),
        params={
            u: (2, 1, 3, 5),
        },
        label=["a", "b"],
        rendering_kw=[{"color": "r"}, {"linestyle": ":"}],
        **pi_options)
    assert isinstance(t, InteractivePlot)
    assert len(t.backend.series) == 2 and all(s.is_2Dline for s in t.backend.series)
    assert [s.label for s in t.backend.series] == ["a", "b"]
    assert t.backend.series[0].rendering_kw == {"color": "r"}
    assert t.backend.series[1].rendering_kw == {"linestyle": ":"}


def test_plot3d_wireframe(pi_options):
    # verify that wireframe=True produces the expected data series
    x, y, u = symbols("x, y, u")
    pi_options["n"] = 10

    _plot3d = lambda wf: plot3d(
        cos(u * x**2 + y**2), (x, -2, 2), (y, -2, 2),
        params = {
            u: (1, 0, 2)
        },
        wireframe=wf, **pi_options
    )
    t = _plot3d(False)
    assert isinstance(t, InteractivePlot)
    assert len(t.backend.series) == 1

    t = _plot3d(True)
    assert isinstance(t, InteractivePlot)
    assert len(t.backend.series) == 1 + 10 + 10
    assert isinstance(t.backend.series[0], SurfaceOver2DRangeSeries)
    assert all(isinstance(s, Parametric3DLineSeries) and s.is_interactive
        for s in t.backend.series[1:])

    # wireframe lines works even when interactive ranges are used
    a, b = symbols("a, b")
    p = plot3d(
        cos(u * x**2 + y**2), prange(x, -2*a, 2*b), prange(y, -2*b, 2*a),
        params = {
            u: (1, 0, 2),
            a: (1, 0, 2),
            b: (1, 0, 2),
        },
        wireframe=True, **pi_options
    )
    assert isinstance(p.backend[1], Parametric3DLineSeries)
    d1 = p.backend[1].get_data()
    p.backend.update_interactive({u: 1, a: 0.5, b: 1.5})
    d2 = p.backend[1].get_data()
    for s, t in zip(d1, d2):
        assert not np.allclose(s, t)


def test_plot3d_parametric_surface_wireframe(pi_options):
    # verify that wireframe=True produces the expected data series
    x, y, u = symbols("x, y, u")
    pi_options["n"] = 10

    _plot3d_ps = lambda wf: plot3d_parametric_surface(
        u * x * cos(y), x * sin(y), x * cos(4 * y) / 2,
        (x, 0, pi), (y, 0, 2*pi),
        params = {
            u: (1, 0, 2)
        },
        wireframe=wf, **pi_options
    )
    t = _plot3d_ps(False)
    assert isinstance(t, InteractivePlot)
    assert len(t.backend.series) == 1

    t = _plot3d_ps(True)
    assert isinstance(t, InteractivePlot)
    assert len(t.backend.series) == 1 + 10 + 10
    assert isinstance(t.backend.series[0], ParametricSurfaceSeries)
    assert all(isinstance(s, Parametric3DLineSeries) and s.is_interactive
        for s in t.backend.series[1:])

    # wireframe lines works even when interactive ranges are used
    a, b = symbols("a, b")
    p = plot3d_parametric_surface(
        u * x * cos(y), x * sin(y), x * cos(4 * y) / 2,
        prange(x, -2*a, 2*b), prange(y, -2*b, 2*a),
        params = {
            u: (1, 0, 2),
            a: (1, 0, 2),
            b: (1, 0, 2),
        },
        wireframe=True, **pi_options
    )
    assert isinstance(p.backend[1], Parametric3DLineSeries)
    d1 = p.backend[1].get_data()
    p.backend.update_interactive({u: 1, a: 0.5, b: 1.5})
    d2 = p.backend[1].get_data()
    for s, t in zip(d1, d2):
        assert not np.allclose(s, t)


def test_plot3d_wireframe_and_labels(pi_options):
    # verify that `wireframe=True` produces the expected data series even when
    # `label` is set
    x, y, u = symbols("x, y, u")
    pi_options["n"] = 10

    t = plot3d(
        cos(u * x**2 + y**2), (x, -2, 2), (y, -2, 2),
        params = {
            u: (1, 0, 2)
        },
        wireframe=True, label="test", **pi_options)
    assert isinstance(t, InteractivePlot)
    assert len(t.backend.series) == 1 + 10 + 10
    assert isinstance(t.backend.series[0], SurfaceOver2DRangeSeries)
    assert t.backend.series[0].get_label(False) == "test"
    assert all(isinstance(s, Parametric3DLineSeries) and s.is_interactive
        for s in t.backend.series[1:])

    t = plot3d(
        (cos(u * x**2 + y**2), (x, -2, 0), (y, -2, 2)),
        (cos(u * x**2 + y**2), (x, 0, 2), (y, -2, 2)),
        params = {
            u: (1, 0, 2)
        },
        wireframe=True, label=["a", "b"], **pi_options)
    assert isinstance(t, InteractivePlot)
    assert len(t.backend.series) == 2 + (10 + 10) * 2
    surfaces = [s for s in t.backend.series if
        isinstance(s, SurfaceOver2DRangeSeries)]
    assert len(surfaces) == 2
    assert [s.get_label(False) for s in surfaces] == ["a", "b"]
    wireframe_lines = [s for s in t.backend.series if
        isinstance(s, Parametric3DLineSeries)]
    assert len(wireframe_lines) == (10 + 10) * 2


def test_plot_real_imag_wireframe_true(pi_options):
    # verify that wireframe lines also work with plot_real_imag

    x, u = symbols("x, u")
    pi_options["n"] = 12

    t = plot_real_imag(
        sqrt(x) * exp(u * x), (x, -3-3j, 3+3j),
        wireframe=True, wf_n1=8, wf_n2=6,
        params={u: (0.25, 0, 1)},
        threed=True, use_latex=False, use_cm=True, **pi_options)
    assert isinstance(t, InteractivePlot)
    assert len(t.backend.series) == 2 + (8 + 6) * 2
    ss = [s for s in t.backend.series if isinstance(s, ComplexSurfaceSeries)]
    wfs = [s for s in t.backend.series if isinstance(s, ComplexParametric3DLineSeries)]
    assert len(ss) == 2
    assert len(wfs) == (8 + 6) * 2
    assert all(s.is_interactive for s in ss)
    assert all(s.is_interactive for s in wfs)

    # wireframe lines works even when interactive ranges are used
    a, b = symbols("a, b")
    p = plot_real_imag(sqrt(x), prange(x, -2*a-b*2j, 2*b+a*2j), imag=False,
        params={a: (1, 0, 2), b: (1, 0, 2)}, threed=True,
        wireframe=True, **pi_options)
    assert isinstance(p.backend[1], Parametric3DLineSeries)
    d1 = p.backend[1].get_data()
    p.backend.update_interactive({a: 0.5, b: 1.5})
    d2 = p.backend[1].get_data()
    for s, t in zip(d1, d2):
        assert not np.allclose(s, t)


###############################################################################
################### PLOT - PLOT_PARAMETRIC - PLOT3D-RELATED ###################
###############################################################################
##### These tests comes from the old sympy.plotting module
##### Some test have been rearranged to reflect the intentions of this new
##### plotting module

@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_1(paf_options, adaptive):
    x, y, z = symbols("x, y, z")
    pa_options = paf_options.copy()
    pa_options.update({"adaptive": adaptive, "n": 10})

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        ###
        # Examples from the 'introduction' notebook
        ###
        p = plot(x, legend=True, **pa_options)
        p = plot(x * sin(x), x * cos(x), **pa_options)
        p.extend(p)
        filename = "test_basic_options_and_colors.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p.extend(plot(x + 1, **pa_options))
        p.append(plot(x + 3, x ** 2, **pa_options)[1])
        filename = "test_plot_extend_append.png"
        p.save(os.path.join(tmpdir, filename))

        p[2] = plot(x ** 2, (x, -2, 3), **pa_options)
        filename = "test_plot_setitem.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(sin(x), (x, -2 * pi, 4 * pi), **pa_options)
        filename = "test_line_explicit.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(sin(x), **pa_options)
        filename = "test_line_default_range.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot((x ** 2, (x, -5, 5)), (x ** 3, (x, -3, 3)), **pa_options)
        filename = "test_line_multiple_range.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        raises(ValueError, lambda: plot(x, y))

        # Piecewise plots
        p = plot(Piecewise((1, x > 0), (0, True)), (x, -1, 1), **pa_options)
        filename = "test_plot_piecewise.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(Piecewise((x, x < 1), (x ** 2, True)), (x, -3, 3), **pa_options)
        filename = "test_plot_piecewise_2.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_7471(paf_options, adaptive):
    x = symbols("x")
    paf_options.update({"adaptive": adaptive, "n": 10})

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        p1 = plot(x, **paf_options)
        p2 = plot(3, **paf_options)
        p1.extend(p2)
        filename = "test_horizontal_line.png"
        p1.save(os.path.join(tmpdir, filename))
        p1.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_10925(paf_options, adaptive):
    x = symbols("x")
    paf_options.update({"adaptive": adaptive, "n": 10})

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        f = Piecewise(
            (-1, x < -1),
            (x, And(-1 <= x, x < 0)),
            (x ** 2, And(0 <= x, x < 1)),
            (x ** 3, x >= 1),
        )
        p = plot(f, (x, -3, 3), **paf_options)
        filename = "test_plot_piecewise_3.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


@pytest.mark.filterwarnings("ignore:.*does not support adaptive algorithm.")
@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_2(paf_options, adaptive):
    x, y, z = symbols("x, y, z")
    paf_options.update({"adaptive": adaptive, "n": 10})

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        # parametric 2d plots.
        # Single plot with default range.
        p = plot_parametric(sin(x), cos(x), **paf_options)
        filename = "test_parametric.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Single plot with range.
        p = plot_parametric(sin(x), cos(x), (x, -5, 5), legend=True,
            **paf_options)
        filename = "test_parametric_range.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple plots with same range.
        p = plot_parametric((sin(x), cos(x)), (x, sin(x)), **paf_options)
        filename = "test_parametric_multiple.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple plots with different ranges.
        p = plot_parametric(
            (sin(x), cos(x), (x, -3, 3)),
            (x, sin(x), (x, -5, 5)),
            **paf_options)
        filename = "test_parametric_multiple_ranges.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # No adaptive sampling.
        p = plot_parametric(cos(x), sin(x), **paf_options)
        filename = "test_adaptive_false.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # With adaptive sampling.
        p = plot_parametric(cos(x), sin(x), adaptive=True, adaptive_goal=0.1,
            show=False)
        filename = "test_adaptive_true.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # 3d parametric plots
        p = plot3d_parametric_line(sin(x), cos(x), x, legend=True,
            **paf_options)
        filename = "test_3d_line.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot3d_parametric_line(
            (sin(x), cos(x), x, (x, -5, 5)),
            (cos(x), sin(x), x, (x, -3, 3)),
            **paf_options)
        filename = "test_3d_line_multiple.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot3d_parametric_line(sin(x), cos(x), x, **paf_options)
        filename = "test_3d_line_points.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # 3d surface single plot.
        p = plot3d(x * y, **paf_options)
        filename = "test_surface.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple 3D plots with same range.
        p = plot3d(-x * y, x * y, (x, -5, 5), **paf_options)
        filename = "test_surface_multiple.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple 3D plots with different ranges.
        p = plot3d(
            (x * y, (x, -3, 3), (y, -3, 3)),
            (-x * y, (x, -3, 3), (y, -3, 3)),
            **paf_options)
        filename = "test_surface_multiple_ranges.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Single Parametric 3D plot
        p = plot3d_parametric_surface(sin(x + y), cos(x - y), x - y, **paf_options)
        filename = "test_parametric_surface.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple Parametric 3D plots.
        p = plot3d_parametric_surface(
            (x * sin(z), x * cos(z), z, (x, -5, 5), (z, -5, 5)),
            (sin(x + y), cos(x - y), x - y, (x, -5, 5), (y, -5, 5)), **paf_options)
        filename = "test_parametric_surface.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Single Contour plot.
        p = plot_contour(sin(x) * sin(y), (x, -5, 5), (y, -5, 5), **paf_options)
        filename = "test_contour_plot.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple Contour plots with same range.
        p = plot_contour(x ** 2 + y ** 2, x ** 3 + y ** 3,
            (x, -5, 5), (y, -5, 5), **paf_options)
        filename = "test_contour_plot.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # Multiple Contour plots with different range.
        p = plot_contour(
            (x ** 2 + y ** 2, (x, -5, 5), (y, -5, 5)),
            (x ** 3 + y ** 3, (x, -3, 3), (y, -3, 3)), **paf_options)
        filename = "test_contour_plot.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


@pytest.mark.filterwarnings("ignore:The evaluation with NumPy/SciPy failed.")
@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_4(paf_options, adaptive):
    x, y = symbols("x, y")
    paf_options.update({"adaptive": adaptive, "n": 10})

    ###
    # Examples from the 'advanced' notebook
    ###

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        i = Integral(log((sin(x) ** 2 + 1) * sqrt(x ** 2 + 1)), (x, 0, y))
        with warns(
                UserWarning,
                match="NumPy is unable to evaluate with complex numbers"
            ):
            p = plot(i, (y, 1, 5), **paf_options)
        filename = "test_advanced_integral.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_5a(paf_options, adaptive):
    x, y = symbols("x, y")
    paf_options.update({"adaptive": adaptive, "n": 10})

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        s = Sum(1 / x ** y, (x, 1, oo))
        p = plot(s, (y, 2, 10), only_integers=True, **paf_options)
        filename = "test_advanced_inf_sum.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


def test_plot_and_save_5b(paf_options):
    x, y = symbols("x, y")

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        p = plot(Sum(1 / x, (x, 1, y)), (y, 2, 10),
            only_integers=True, steps=True, **paf_options)
        filename = "test_advanced_fin_sum.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()


@pytest.mark.filterwarnings("ignore:The evaluation with NumPy/SciPy failed.")
@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_6(paf_options, adaptive):
    x = symbols("x")
    paf_options.update({"adaptive": adaptive, "n": 10})

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        filename = "test.png"
        ###
        # Test expressions that can not be translated to np and generate complex
        # results.
        ###
        p = plot(sin(x) + I * cos(x), **paf_options)
        p.save(os.path.join(tmpdir, filename))
        p = plot(sqrt(sqrt(-x)), **paf_options)
        p.save(os.path.join(tmpdir, filename))
        p = plot(LambertW(x), **paf_options)
        p.save(os.path.join(tmpdir, filename))
        p = plot(sqrt(LambertW(x)), **paf_options)
        p.save(os.path.join(tmpdir, filename))

        # Characteristic function of a StudentT distribution with nu=10
        x1 = 5 * x ** 2 * exp_polar(-I * pi) / 2
        m1 = meijerg(((1 / 2,), ()), ((5, 0, 1 / 2), ()), x1)
        x2 = 5 * x ** 2 * exp_polar(I * pi) / 2
        m2 = meijerg(((1 / 2,), ()), ((5, 0, 1 / 2), ()), x2)
        expr = (m1 + m2) / (48 * pi)
        p = plot(expr, (x, 1e-6, 1e-2), **paf_options)
        p.save(os.path.join(tmpdir, filename))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_issue_11461(paf_options, pat_options):
    x = symbols("x")

    expr = real_root((log(x / (x - 2))), 3)

    p = plot(expr, **paf_options)
    d = p[0].get_data()
    assert not np.isnan(d[1]).all()

    p = plot(expr, **pat_options)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    d = p[0].get_data()
    assert len(d[0]) >= 30
    assert not np.isnan(d[1]).all()

    # plot_piecewise is not able to deal with ConditionSet
    raises(TypeError, lambda: plot_piecewise(expr, backend=MB, show=False))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_issue_11865(paf_options, pat_options):
    k = symbols("k", integer=True)
    f = Piecewise(
        (-I * exp(I * pi * k) / k + I * exp(-I * pi * k) / k, Ne(k, 0)),
        (2 * pi, True)
    )
    p = plot(f, **pat_options)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    d = p[0].get_data()
    assert len(d[0]) >= 30
    assert not np.isnan(d[1]).all()

    p = plot_piecewise(f, **pat_options)
    d = p[0].get_data()
    assert not np.isnan(d[1]).all()


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_16572(paf_options, adaptive):
    x = symbols("x")
    paf_options.update({"adaptive": adaptive})
    p = plot(LambertW(x), **paf_options)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    d = p[0].get_data()
    assert len(d[0]) >= 30
    assert not np.isnan(d[1]).all()


@pytest.mark.parametrize("adaptive", [True, False])
def test_logplot_PR_16796(paf_options, adaptive):
    x = symbols("x")
    paf_options.update({"adaptive": adaptive})
    p = plot(x, (x, 0.001, 100), xscale="log", **paf_options)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    d = p[0].get_data()
    assert len(d[0]) >= 30
    assert not np.isnan(d[1]).all()
    assert p[0].end == 100.0
    assert p[0].start == 0.001


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_17405(paf_options, adaptive):
    x = symbols("x")
    paf_options.update({"adaptive": adaptive})
    f = x ** 0.3 - 10 * x ** 3 + x ** 2
    p = plot(f, (x, -10, 10), **paf_options)
    d = p[0].get_data()
    assert not np.isnan(d[1]).all()
    assert len(d[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_15265(paf_options, adaptive):
    x = symbols("x")
    paf_options.update({"adaptive": adaptive, "n": 10})
    eqn = sin(x)

    p = plot(eqn, xlim=(-S.Pi, S.Pi), ylim=(-1, 1), **paf_options)
    p.close()

    p = plot(eqn, xlim=(-1, 1), ylim=(-S.Pi, S.Pi), **paf_options)
    p.close()

    p = plot(eqn, xlim=(-1, 1), ylim=(sympify("-3.14"), sympify("3.14")),
        **paf_options)
    p.close()

    p = plot(eqn, xlim=(sympify("-3.14"), sympify("3.14")), ylim=(-1, 1),
        **paf_options)
    p.close()

    raises(ValueError,
        lambda: plot(eqn, xlim=(-S.ImaginaryUnit, 1), ylim=(-1, 1), **paf_options))

    raises(ValueError,
        lambda: plot(eqn, xlim=(-1, 1), ylim=(-1, S.ImaginaryUnit), **paf_options))

    raises(ValueError,
        lambda: plot(eqn, xlim=(S.NegativeInfinity, 1), ylim=(-1, 1), **paf_options))

    raises(ValueError,
        lambda: plot(eqn, xlim=(-1, 1), ylim=(-1, S.Infinity), **paf_options))


@pytest.mark.parametrize("adaptive", [True, False])
def test_append_issue_7140(paf_options, adaptive):
    x = symbols("x")
    paf_options.update({"adaptive": adaptive, "n": 10})
    p1 = plot(x, **paf_options)
    p2 = plot(x ** 2, **paf_options)

    # append a series
    p2.append(p1[0])
    assert len(p2._series) == 2

    with raises(TypeError):
        p1.append(p2)

    with raises(TypeError):
        p1.append(p2._series)


def test_plot_limits(paf_options):
    x = symbols("x")
    p = plot(x, x ** 2, (x, -10, 10), **paf_options)

    xmin, xmax = p.fig.axes[0].get_xlim()
    assert abs(xmin + 10) < 2
    assert abs(xmax - 10) < 2
    ymin, ymax = p.fig.axes[0].get_ylim()
    assert abs(ymin + 10) < 10
    assert abs(ymax - 100) < 10


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot3d_parametric_line_limits(paf_options, adaptive):
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


###############################################################################
################################ PLOT IMPLICIT ################################
###############################################################################
##### These tests comes from the old sympy.plotting module

def tmp_file(dir=None, name=""):
    return NamedTemporaryFile(suffix=".png", dir=dir, delete=False).name


def test_plot_implicit_label_rendering_kw(paf_options):
    # verify that label and rendering_kw keyword arguments works as expected

    x, y = symbols("x, y")
    p = plot_implicit(x + y, x - y, (x, -5, 5), (y, -5, 5),
        label=["a", "b"], rendering_kw=[{"levels": 5}, {"alpha": 0.5}],
        **paf_options)
    assert p[0].get_label(True) == "a"
    assert p[1].get_label(True) == "b"
    assert p[0].rendering_kw == {"levels": 5}
    assert p[1].rendering_kw == {"alpha": 0.5}
    assert p.xlim and p.ylim


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
            use_latex=False, backend=MB)
        assert p.xlim and p.ylim
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
        with warns(
                UserWarning,
                match="Adaptive meshing could not be applied"
            ):
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
            use_latex=False, backend=MB)
        assert p.xlim and p.ylim
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


def test_indexed_objects():
    # verify that plot functions and series correctly process indexed objects
    x = IndexedBase("x")

    kwargs = dict(adaptive=False, n=10, show=False, backend=MB,
        use_latex=False)
    p = plot(cos(x[0]), (x[0], -pi, pi), **kwargs)
    d = p[0].get_data()
    assert p.xlabel == "x[0]"

    p = plot3d(cos(x[0]**2 + x[1]**2), (x[0], -pi, pi), (x[1], -pi, pi), **kwargs)
    d = p[0].get_data()
    assert p.xlabel == "x[0]"
    assert p.ylabel == "x[1]"
    assert p.zlabel == "f(x[0], x[1])"


def test_appliedundef_objects():
    # verify that plot functions and series correctly process AppliedUndef
    # functions (especially useful for dealing with sympy.physics.mechanics)
    t = symbols("t")
    f1, f2 = symbols("f1, f2", cls=Function)
    f1 = f1(t)
    f2 = f2(t)

    kwargs = dict(adaptive=False, n=10, show=False, backend=MB,
        use_latex=False)
    p = plot(cos(f1), (f1, -pi, pi), **kwargs)
    d = p[0].get_data()
    assert p.xlabel == "f1"

    expr = 2*cos(f1) + 3*cos(f1 + f2) + 4 * cos(f1 - f2)
    p = plot_contour(expr, (f1, -pi, pi), (f2, -pi, pi), **kwargs)
    d = p[0].get_data()
    assert p.xlabel == "f1"
    assert p.ylabel == "f2"


@pytest.mark.filterwarnings("ignore:The following keyword arguments are unused.")
def test_number_discretization_points():
    # verify the different ways of setting the numbers of discretization points
    x, y, z = symbols("x, y, z")
    options = dict(adaptive=False, show=False, backend=MB)

    p = plot(cos(x), (x, -10, 10), **options)
    assert p[0].n[0] == 1000
    p = plot(cos(x), (x, -10, 10), n=10, **options)
    assert p[0].n[0] == 10
    p = plot(cos(x), (x, -10, 10), n1=10, **options)
    assert p[0].n[0] == 10
    p = plot(cos(x), (x, -10, 10), nb_of_points=10, **options)
    assert p[0].n[0] == 10
    p = plot(cos(x), sin(x), (x, -10, 10), n=10, **options)
    assert (len(p.series) > 1) and all(t.n[0] == 10 for t in p.series)

    p = plot_parametric(cos(x), sin(x), (x, 0, 2*pi), **options)
    assert p[0].n[0] == 1000
    p = plot_parametric(cos(x), sin(x), (x, 0, 2*pi), n=10, **options)
    assert p[0].n[0] == 10
    p = plot_parametric(cos(x), sin(x), (x, 0, 2*pi), n1=10, **options)
    assert p[0].n[0] == 10
    p = plot_parametric(cos(x), sin(x), (x, 0, 2*pi),
        nb_of_points=10, **options)
    assert p[0].n[0] == 10
    p = plot_parametric((cos(x), sin(x)), (cos(x), x), (x, 0, 2*pi),
        n=10, **options)
    assert (len(p.series) > 1) and all(t.n[0] == 10 for t in p.series)

    p = plot3d_parametric_line(cos(x), sin(x), x, (x, 0, 2*pi), **options)
    assert p[0].n[0] == 1000
    p = plot3d_parametric_line(cos(x), sin(x), x, (x, 0, 2*pi),
        n=10, **options)
    assert p[0].n[0] == 10
    p = plot3d_parametric_line(cos(x), sin(x), x, (x, 0, 2*pi),
        n1=10, **options)
    assert p[0].n[0] == 10
    p = plot3d_parametric_line(cos(x), sin(x), x, (x, 0, 2*pi),
        nb_of_points=10, **options)
    assert p[0].n[0] == 10
    p = plot3d_parametric_line(
        (cos(x), sin(x), x), (sin(x), cos(x), x**2), (x, 0, 2*pi),
        n=10, **options)
    assert (len(p.series) > 1) and all(t.n[0] == 10 for t in p.series)

    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), **options)
    assert p[0].n[:2] == [100, 100]
    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        n=400, **options)
    assert p[0].n[:2] == [400, 400]
    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        n1=50, **options)
    assert p[0].n[:2] == [50, 100]
    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        n1=50, n2=20, **options)
    assert p[0].n[:2] == [50, 20]
    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        nb_of_points_x=50, nb_of_points_y=20, **options)
    assert p[0].n[:2] == [50, 20]
    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        n1=50, n2=20, wireframe=True, wf_n1=5, wf_n2=5, **options)
    assert p[0].n[:2] == [50, 20]
    assert all(s.n[0] == 20 for s in p.series[1:6])
    assert all(s.n[0] == 50 for s in p.series[6:])
    assert p[0].n[:2] == [50, 20]
    p = plot3d(cos(x**2 + y**2), sin(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        n=400, **options)
    assert (len(p.series) > 1) and all(t.n[:2] == [400, 400] for t in p.series)

    p = plot_contour(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), **options)
    assert p[0].n[:2] == [100, 100]
    p = plot_contour(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        n=400, **options)
    assert p[0].n[:2] == [400, 400]
    p = plot_contour(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        n1=50, **options)
    assert p[0].n[:2] == [50, 100]
    p = plot_contour(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        n1=50, n2=20, **options)
    assert p[0].n[:2] == [50, 20]
    p = plot_contour(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        nb_of_points_x=50, nb_of_points_y=20, **options)
    assert p[0].n[:2] == [50, 20]
    p = plot_contour(
        cos(x**2 + y**2), sin(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        n=400, **options)
    assert (len(p.series) > 1) and all(t.n[:2] == [400, 400] for t in p.series)

    p = plot3d_parametric_surface(
        x + y, x - y, x, (x, -pi, pi), (y, -pi, pi), **options)
    assert p[0].n[:2] == [100, 100]
    p = plot3d_parametric_surface(
        x + y, x - y, x, (x, -pi, pi), (y, -pi, pi),
        n=400, **options)
    assert p[0].n[:2] == [400, 400]
    p = plot3d_parametric_surface(
        x + y, x - y, x, (x, -pi, pi), (y, -pi, pi),
        n1=50, **options)
    assert p[0].n[:2] == [50, 100]
    p = plot3d_parametric_surface(
        x + y, x - y, x, (x, -pi, pi), (y, -pi, pi),
        n1=50, n2=20, **options)
    assert p[0].n[:2] == [50, 20]
    p = plot3d_parametric_surface(
        x + y, x - y, x, (x, -pi, pi), (y, -pi, pi),
        nb_of_points_u=50, nb_of_points_v=20, **options)
    assert p[0].n[:2] == [50, 20]
    p = plot3d_parametric_surface(
        x + y, x - y, x, (x, -pi, pi), (y, -pi, pi),
        n1=50, n2=20, wireframe=True, wf_n1=5, wf_n2=5, **options)
    assert p[0].n[:2] == [50, 20]
    assert all(s.n[0] == 20 for s in p.series[1:6])
    assert all(s.n[0] == 50 for s in p.series[6:])

    p = plot3d_spherical(1, (x, 0, 0.7 * pi), (y, 0, 1.8 * pi), **options)
    assert p[0].n[:2] == [100, 100]
    p = plot3d_spherical(1, (x, 0, 0.7 * pi), (y, 0, 1.8 * pi),
        n=400, **options)
    assert p[0].n[:2] == [400, 400]
    p = plot3d_spherical(1, (x, 0, 0.7 * pi), (y, 0, 1.8 * pi),
        n1=50, **options)
    assert p[0].n[:2] == [50, 100]
    p = plot3d_spherical(1, (x, 0, 0.7 * pi), (y, 0, 1.8 * pi),
        n1=50, n2=20, **options)
    assert p[0].n[:2] == [50, 20]
    p = plot3d_spherical(1, (x, 0, 0.7 * pi), (y, 0, 1.8 * pi),
        nb_of_points_u=50, nb_of_points_v=20, **options)
    assert p[0].n[:2] == [50, 20]

    p = plot3d_implicit(
        x**2 + y**3 - z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2), **options)
    assert p[0].n == [60, 60, 60]
    p = plot3d_implicit(
        x**2 + y**3 - z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2),
        n=400, **options)
    assert p[0].n == [400, 400, 400]
    p = plot3d_implicit(
        x**2 + y**3 - z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2),
        n1=50, **options)
    assert p[0].n == [50, 60, 60]
    p = plot3d_implicit(
        x**2 + y**3 - z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2),
        n1=50, n2=20, n3=10, **options)
    assert p[0].n == [50, 20, 10]

    p = plot_implicit(y > x**2, (x, -5, 5), (y, -10, 10), **options)
    assert p[0].n[:2] == [100, 100]
    p = plot_implicit(y > x**2, (x, -5, 5), (y, -10, 10),
        n=400, **options)
    assert p[0].n[:2] == [400, 400]
    p = plot_implicit(y > x**2, (x, -5, 5), (y, -10, 10),
        n1=50, **options)
    assert p[0].n[:2] == [50, 100]
    p = plot_implicit(y > x**2, (x, -5, 5), (y, -10, 10),
        n1=50, n2=20, **options)
    assert p[0].n[:2] == [50, 20]
    p = plot_implicit(y > x**2, (x, -5, 5), (y, -10, 10),
        points=20, **options)
    assert p[0].n[:2] == [20, 20]
    p = plot_implicit(y > x**2, y < sin(x), (x, -5, 5), (y, -10, 10),
        n=400, **options)
    assert (len(p.series) > 1) and all(t.n[:2] == [400, 400] for t in p.series)
