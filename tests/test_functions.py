from sympy import symbols, Piecewise, And, Eq, Interval, cos, sin, Abs
import numpy as np
from spb.functions import (
    _process_piecewise, plot_list
)
from spb.series import LineOver1DRangeSeries, List2DSeries
from pytest import raises

def test_plot_list():
    xx1 = np.linspace(-3, 3)
    yy1 = np.cos(xx1)
    xx2 = np.linspace(-5, 5)
    yy2 = np.sin(xx2)

    p = plot_list(xx1, yy1, show=False)
    assert len(p.series) == 1
    assert isinstance(p.series[0], List2DSeries)
    assert p.series[0].label == ""

    p = plot_list(xx1, yy1, "test", show=False)
    assert len(p.series) == 1
    assert isinstance(p.series[0], List2DSeries)
    assert p.series[0].label == "test"

    p = plot_list((xx1, yy1), (xx2, yy2), show=False)
    assert len(p.series) == 2
    assert all(isinstance(t, List2DSeries) for t in p.series)
    assert all(t.label == "" for t in p.series)

    p = plot_list((xx1, yy1, "cos"), (xx2, yy2, "sin"), show=False)
    assert len(p.series) == 2
    assert all(isinstance(t, List2DSeries) for t in p.series)
    assert p.series[0].label == "cos"
    assert p.series[1].label == "sin"

def test_piecewise():
    x = symbols("x")

    # Test that univariate Piecewise objects are processed in such a way to
    # create multiple series, each one with the correct range

    f = Piecewise(
        (-1, x < -1),
        (x, And(-1 <= x, x < 0)),
        (x**2, And(0 <= x, x < 1)),
        (x**3, x >= 1)
    )
    s = _process_piecewise(f, (x, -5, 5), "A")
    assert len(s) == 4
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert (s[0].expr == -1) and (s[0].start == -5) and (s[0].end == -1)
    assert (s[1].expr == x) and (s[1].start == -1) and (s[1].end == 0)
    assert (s[2].expr == x**2) and (s[2].start == 0) and (s[2].end == 1)
    assert (s[3].expr == x**3) and (s[3].start == 1) and (s[3].end == 5)
    labels = ["A" + str(i + 1) for i in range(5)]
    assert all(t.label == l for t, l in zip(s, labels))

    f = Piecewise(
        (1, x < -5),
        (x, Eq(x, 0)),
        (x**2, Eq(x, 2)),
        (x**3, (x > 0) & (x < 2)),
        (x**4, True)
    )
    s = _process_piecewise(f, (x, -10, 10), "B")
    assert len(s) == 6
    assert all(isinstance(t, LineOver1DRangeSeries) for t in [s[0], s[3], s[4], s[5]])
    assert all(isinstance(t, List2DSeries) for t in [s[1], s[2]])
    assert (s[0].expr == 1) and (s[0].start == -10) and (s[0].end == -5)
    assert (np.allclose(s[1].list_x, np.array([0.])) and
        np.allclose(s[1].list_y, np.array([0.])))
    assert (np.allclose(s[2].list_x, np.array([2.])) and
        np.allclose(s[2].list_y, np.array([4.])))
    assert (s[3].expr == x**3) and (s[3].start == 0) and (s[3].end == 2)
    assert (s[4].expr == x**4) and (s[4].start == -5) and (s[4].end == 0)
    assert (s[5].expr == x**4) and (s[5].start == 2) and (s[5].end == 10)
    labels = ["B" + str(i + 1) for i in range(5)] + ["B5"]
    assert all(t.label == l for t, l in zip(s, labels))

    f = Piecewise((x, Interval(0, 1).contains(x)), (0, True))
    s = _process_piecewise(f, (x, -10, 10), "C")
    assert len(s) == 3
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert (s[0].expr == x) and (s[0].start == 0) and (s[0].end == 1)
    assert (s[1].expr == 0) and (s[1].start == -10) and (s[1].end == 0)
    assert (s[2].expr == 0) and (s[2].start == 1) and (s[2].end == 10)
    labels = ["C1", "C2", "C2"]
    assert all(t.label == l for t, l in zip(s, labels))

    f = Piecewise((x, Interval(0, 1, False, True).contains(x)), (0, True))
    s = _process_piecewise(f, (x, -10, 10), "D")
    assert len(s) == 3
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert (s[0].expr == x) and (s[0].start == 0) and (s[0].end == 1)
    assert (s[1].expr == 0) and (s[1].start == -10) and (s[1].end == 0)
    assert (s[2].expr == 0) and (s[2].start == 1) and (s[2].end == 10)
    labels = ["D1", "D2", "D2"]
    assert all(t.label == l for t, l in zip(s, labels))

    f = Piecewise((x, x < 1), (x**2, -1 <= x), (x, 3 < x))
    s = _process_piecewise(f, (x, -10, 10), "E")
    assert len(s) == 2
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert (s[0].expr == x) and (s[0].start == -10) and (s[0].end == 1)
    assert (s[1].expr == x**2) and (s[1].start == 1) and (s[1].end == 10)

    # NotImplementedError: as_set is not implemented for relationals with
    # periodic solutions
    p1 = Piecewise((cos(x), x < 0), (0, True))
    f = Piecewise((0, Eq(p1, 0)), (p1 / Abs(p1), True))
    raises(NotImplementedError, lambda: _process_piecewise(f, (x, -10, 10), "F"))

    f = Piecewise((1 - x, (x >= 0) & (x < 1)), (0, True))
    s = _process_piecewise(f, (x, -10, 10), "test")
    assert len(s) == 3
    assert all(isinstance(t, LineOver1DRangeSeries) for t in s)
    assert (s[0].expr == 1 - x) and (s[0].start == 0) and (s[0].end == 1)
    assert (s[1].expr == 0) and (s[1].start == -10) and (s[1].end == 0)
    assert (s[2].expr == 0) and (s[2].start == 1) and (s[2].end == 10)

    # The range is smaller than the function "domain"
    f = Piecewise(
        (1, x < -5),
        (x, Eq(x, 0)),
        (x**2, Eq(x, 2)),
        (x**3, (x > 0) & (x < 2)),
        (x**4, True)
    )
    s = _process_piecewise(f, (x, -3, 3), "A")
    labels = ["A2", "A3", "A4", "A5", "A5"]
    assert all(t.label == l for t, l in zip(s, labels))