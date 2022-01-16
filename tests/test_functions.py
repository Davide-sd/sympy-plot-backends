from sympy import (
    symbols, Piecewise, And, Eq, Interval, cos, sin, Abs,
    Sum, oo, S, Heaviside
)
import numpy as np
from spb.functions import (
    plot_piecewise, plot, plot_list
)
from spb.series import LineOver1DRangeSeries, List2DSeries
from spb.backends.matplotlib import MB
from pytest import raises

def test_plot_list():
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
    x = symbols("x")

    # Verify that univariate Piecewise objects are processed in such a way to
    # create multiple series, each one with the correct range.
    # Series representing filled dots should be last in order.

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
