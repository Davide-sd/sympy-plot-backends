import numpy as np
import pytest
from spb.graphics import (
    line, line_parametric_2d, contour, implicit_2d, line_polar, list_2d,
    geometry
)
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries,
    ContourSeries, ImplicitSeries, List2DSeries, Geometry2DSeries
)
from sympy import (
    symbols, cos, sin, pi, Rational,
    Circle, Ellipse, Polygon, Curve, Segment, Point2D, Point, Line2D
)


a, b, c, d, p1, p2 = symbols("a:d p1 p2")


@pytest.mark.parametrize("rang, label, rkw, n, params", [
    (None, None, None, None, None),
    ((-2, 3), "test", {"color": "r"}, None, None),
    ((-2, 3), "test", {"color": "r"}, 10, None),
    ((-2, 3), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}),
])
def test_line_1(default_range, rang, label, rkw, n, params):
    x = symbols("x")

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    kwargs = {}
    if params:
        kwargs["params"] = params
    if n:
        kwargs["n"] = n

    series = line(cos(x), range_x=r, label=label, rendering_kw=rkw, **kwargs)
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, LineOver1DRangeSeries)
    assert s.expr == cos(x)
    assert s.ranges[0] == (default_range(x) if not rang else r)
    assert s.get_label(False) == ("cos(x)" if not label else label)
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert s.n[0] == (1000 if not n else n)
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == ({} if not params else params)


@pytest.mark.parametrize("rang, label, rkw, n, params", [
    (None, None, None, None, None),
    ((-2, 3), "test", {"color": "r"}, None, None),
    ((-2, 3), "test", {"color": "r"}, 10, None),
    ((-2, 3), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}),
])
def test_line_parametric_2d(default_range, rang, label, rkw, n, params):
    x = symbols("x")

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    kwargs = {}
    if params:
        kwargs["params"] = params
    if n:
        kwargs["n"] = n

    series = line_parametric_2d(
        cos(x), sin(x), range_p=r, label=label,
        rendering_kw=rkw, **kwargs
    )
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, Parametric2DLineSeries)
    assert s.expr == (cos(x), sin(x))
    assert s.ranges[0] == (default_range(x) if not rang else r)
    assert s.get_label(False) == "x" if not label else label
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert s.n[0] == (1000 if not n else n)
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == ({} if not params else params)


@pytest.mark.parametrize("rang, label, rkw, n, params", [
    (None, None, None, None, None),
    ((-2, 3), "test", {"color": "r"}, None, None),
    ((-2, 3), "test", {"color": "r"}, 10, None),
    ((-2, 3), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}),
])
def test_line_polar(default_range, rang, label, rkw, n, params):
    x = symbols("x")

    r = (x, *rang) if isinstance(rang, (list, tuple)) else None
    kwargs = {}
    if params:
        kwargs["params"] = params
    if n:
        kwargs["n"] = n

    series = line_polar(
        3 * sin(2 * x), range_p=r, label=label,
        rendering_kw=rkw, **kwargs
    )
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, Parametric2DLineSeries)
    assert s.expr == (3*sin(2*x)*cos(x), 3*sin(x)*sin(2*x))
    assert s.ranges[0] == (default_range(x) if not rang else r)
    assert s.get_label(False) == "(3*sin(2*x)*cos(x), 3*sin(x)*sin(2*x))" if not label else label
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert s.n[0] == (1000 if not n else n)
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == ({} if not params else params)


@pytest.mark.filterwarnings("ignore:No ranges were provided.")
@pytest.mark.parametrize(
    "range1, range2, label, rkw, n, color, border_color, params", [
        (None, None, None, None, None, None, None, None),
        ((-2, 3), None, None, None, None, None, None, None),
        (None, (-2, 3), None, None, None, None, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, None, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, "gold", None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, None, "k", None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, "gold", "k", None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, None, None, {p1: (1, 0, 2), p2: (2, -1, 3)}),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, "gold", None, {p1: (1, 0, 2), p2: (2, -1, 3)}),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, None, "k", {p1: (1, 0, 2), p2: (2, -1, 3)}),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, "gold", "k", {p1: (1, 0, 2), p2: (2, -1, 3)}),
])
def test_implicit_2d(default_range, range1, range2, label, rkw, n,  color,
    border_color, params):
    x, y = symbols("x, y")

    r1 = (x, *range1) if isinstance(range1, (list, tuple)) else None
    r2 = (y, *range2) if isinstance(range2, (list, tuple)) else None
    kwargs = {}
    if params:
        kwargs["params"] = params
    if n:
        kwargs["n"] = n

    expr = (4 * (cos(x) - sin(y) / 5)**2 + 4 * (-cos(x) / 5 + sin(y))**2) <= pi
    expected_expr = -(4 * (cos(x) - sin(y) / 5)**2 + 4 * (-cos(x) / 5 + sin(y))**2) + pi
    series = implicit_2d(
        expr, range1=r1, range2=r2,
        label=label, rendering_kw=rkw, color=color,
        border_color=border_color, **kwargs
    )
    assert len(series) == 1 + (1 if border_color else 0)
    assert all(isinstance(t, ImplicitSeries) for t in series)
    assert all(t.expr == expected_expr for t in series)
    assert all(t.get_label(False) == str(expr) if not label else label for t in series)
    s = series[0]
    assert (s.ranges[0] == (default_range(x) if not range1 else r1)) or \
        (s.ranges[0] == (default_range(y) if not range1 else r1))
    assert (s.ranges[1] == (default_range(y) if not range2 else r2)) or \
        (s.ranges[1] == (default_range(x) if not range2 else r2))
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert s.color == color
    assert all(t == (100 if not n else n) for t in s.n[:-1])
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == ({} if not params else params)
    if border_color:
        s = series[1]
        assert (s.ranges[0] == (default_range(x) if not range1 else r1)) or \
            (s.ranges[0] == (default_range(y) if not range1 else r1))
        assert (s.ranges[1] == (default_range(y) if not range2 else r2)) or \
            (s.ranges[1] == (default_range(x) if not range2 else r2))
        assert s.rendering_kw == {}
        assert s.color == border_color
        assert all(t == (100 if not n else n) for t in s.n[:-1])
        assert s.is_interactive == (len(s.params) > 0)
        assert s.params == ({} if not params else params)


@pytest.mark.filterwarnings("ignore:No ranges were provided.")
@pytest.mark.parametrize(
    "range1, range2, label, rkw, n, params", [
        (None, None, None, None, None, None),
        ((-2, 3), None, None, None, None, None),
        (None, (-2, 3), None, None, None, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, None),
        ((-4, 6), (-2, 3), "test", {"color": "r"}, 10, {p1: (1, 0, 2), p2: (2, -1, 3)}),
])
def test_contour(default_range, range1, range2, label, rkw, n, params):
    x, y = symbols("x, y")

    r1 = (x, *range1) if isinstance(range1, (list, tuple)) else None
    r2 = (y, *range2) if isinstance(range2, (list, tuple)) else None
    kwargs = {}
    if params:
        kwargs["params"] = params
    if n:
        kwargs["n"] = n

    series = contour(
        cos(x*y), range1=r1, range2=r2, label=label,
        rendering_kw=rkw, **kwargs
    )
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, ContourSeries)
    assert s.expr == cos(x*y)
    assert (s.ranges[0] == (default_range(x) if not range1 else r1)) or \
        (s.ranges[0] == (default_range(y) if not range1 else r1))
    assert (s.ranges[1] == (default_range(y) if not range2 else r2)) or \
        (s.ranges[1] == (default_range(x) if not range2 else r2))
    assert s.get_label(False) == ("cos(x*y)" if not label else label)
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert all(t == (100 if not n else n) for t in s.n[:-1])
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == ({} if not params else params)


@pytest.mark.parametrize("label, rkw", [
    (None, None),
    ("test", {"color": "r"})
])
def test_list_2d(label, rkw):
    x = symbols("x")
    xx = [t / 100 * 6 - 3 for t in list(range(101))]
    yy = [cos(x).evalf(subs={x: t}) for t in xx]
    series = list_2d(xx, yy, label=label, rendering_kw=rkw)
    assert len(series) == 1
    s = series[0]
    assert isinstance(s, List2DSeries)
    assert s.get_label(False) == ("" if label is None else label)
    assert s.rendering_kw == ({} if not rkw else {"color": "r"})


@pytest.mark.parametrize("geom, label, rkw, fill, params", [
    (Circle(Point(0, 0), 5), None, None, False, None),
    (Ellipse(Point(-3, 2), hradius=3, eccentricity=Rational(4, 5)),
        "test", None, False, None),
    (Polygon((4, 0), 4, n=5), None, {"color": "r"}, False, None),
    (Curve((cos(a), sin(a)), (a, 0, 2 * pi)), "test", {"color": "r"}, False, None),
    (Segment((-4, -6), (6, 6)), None, None, False, None),
    (Point2D(0, 0), None, None, False, None),
    (Point2D(0, 0), "test", None, True, None),
    (Polygon((a, b), c, n=d), None, None, False, {a: (0, -1, 1), b: (1, -1, 1), c: (2, 1, 2), d: (3, 3, 8)}),
    (Polygon((a, b), c, n=d), "test", {"color": "r"}, True, {a: (0, -1, 1), b: (1, -1, 1), c: (2, 1, 2), d: (3, 3, 8)}),
])
def test_geometry(geom, label, rkw, fill, params):
    if params is None:
        params = {}
    series = geometry(geom, label=label, rendering_kw=rkw, fill=fill, params=params)
    assert len(series) == 1
    s = series[0]
    if isinstance(geom, Curve):
        assert isinstance(s, Parametric2DLineSeries)
        assert s.expr == geom.args[0]
        if not label:
            assert s.get_label(False) == str(geom)
        else:
            assert s.get_label(False) == label
    else:
        assert isinstance(s, Geometry2DSeries)
        assert s.expr == geom
        assert s.get_label(False) == (str(geom) if not label else label)
    assert s.is_filled == fill
    assert s.rendering_kw == ({} if not rkw else rkw)
    assert s.is_interactive == (len(s.params) > 0)
    assert s.params == ({} if not params else params)


def test_geometry_Line2D():
    # verify that range_x works as supposed to when a Line2D is passed in.
    l = Line2D((1, 2), (3, 4))
    s = geometry(l)[0]
    assert s.range_x == (0, 0)
    xx, yy = s.get_data()
    assert np.allclose(xx, [1, 3])
    assert np.allclose(yy, [2, 4])

    s = geometry(l, range_x=(-10, 5))[0]
    assert s.range_x == (-10, 5)
    xx, yy = s.get_data()
    assert np.allclose(xx, [-10, 5])
    assert np.allclose(yy, [-9, 6])
