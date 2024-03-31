import pytest
from pytest import raises
bokeh = pytest.importorskip("bokeh")
from bokeh.models.formatters import PrintfTickFormatter
import math
import numpy as np
from spb import MB, PB, BB, graphics, line, plot
from spb.interactive import _tuple_to_dict
from spb.interactive.ipywidgets import InteractivePlot as IPYInteractivePlot
from spb.interactive.panel import InteractivePlot as PANELInteractivePlot
from spb.series import (
    LineOver1DRangeSeries, HVLineSeries
)
from sympy import symbols, pi, Rational, Integer, cos, sin
from sympy.abc import a

tf = PrintfTickFormatter(format="%.3f")


@pytest.mark.parametrize(
    "tup, val, min, max, step, formatter, description, spacing",
    [
        ((1, 0, 5), 1, 0, 5, 0.125, None, "x", "linear"),
        ((5, 0, 1), 1, 0, 5, 0.125, None, "x", "linear"),
        ((0, 1, 5), 1, 0, 5, 0.125, None, "x", "linear"),
        ((0, 5, 1), 1, 0, 5, 0.125, None, "x", "linear"),
        ((0, 0, 1), 0, 0, 1, 0.025, None, "x", "linear"),
        ((1, 0, 1), 1, 0, 1, 0.025, None, "x", "linear"),
        ((pi, Rational(1, 2), Integer(15)), math.pi, 0.5, 15, 0.3625, None, "x", "linear"),
        ((1, 0, 5, 50), 1, 0, 5, 0.1, None, "x", "linear"),
        ((1, 0, 5, 50, ".5f"), 1, 0, 5, 0.1, ".5f", "x", "linear"),
        ((1, 0, 5, 50, ".5f", "test"), 1, 0, 5, 0.1, ".5f", "test", "linear"),
        ((1, 0, 5, 50, ".5f", "test", "log"), 1, 0, 5, 0.1, ".5f", "test", "log"),
        ((1, 0, 5, 50, "log"), 1, 0, 5, 0.1, None, "x", "log"),
        ((1, 0, 5, 50, "test"), 1, 0, 5, 0.1, None, "test", "linear"),
        ((1, 0, 5, "test"), 1, 0, 5, 0.125, None, "test", "linear"),
        ((1, 0, 5, "test", ".5f"), 1, 0, 5, 0.125, ".5f", "test", "linear"),
        ((1, 0, 5, tf), 1, 0, 5, 0.125, tf, "x", "linear"),
        ((1, 0, 5, tf, "test"), 1, 0, 5, 0.125, tf, "test", "linear"),
        ((1, 0, 5, tf, "log"), 1, 0, 5, 0.125, tf, "x", "log"),
        ((1, 0, 5, tf, "test", "log"), 1, 0, 5, 0.125, tf, "test", "log"),
    ]
)
def test_tuple_to_dict(
    tup, val, min, max, step, formatter, description, spacing
):
    x = symbols("x")
    t = _tuple_to_dict(x, tup, use_latex=False)
    assert math.isclose(t["value"], val)
    assert math.isclose(t["min"], min)
    assert math.isclose(t["max"], max)
    assert math.isclose(t["step"], step)
    assert t["formatter"] == formatter
    assert t["description"] == description
    assert t["type"] == spacing


@pytest.mark.parametrize(
    "tup, err, msg", [
        (1, TypeError, "Provide a tuple or list for the parameter"),
        ((1, 0), ValueError, "The parameter-tuple must have at least 3 elements"),
        ((a, 0, 2), TypeError, "The first three elements of the parameter-tuple"),
        ((0, a, 2), TypeError, "The first three elements of the parameter-tuple"),
        ((0, 0, a), TypeError, "The first three elements of the parameter-tuple"),
        ((0, 0, 0), ValueError, "The minimum value of the slider must be different"),
    ]
)
def test_tuple_to_dict_errors(tup, err, msg):
    x = symbols("x")
    with raises(err, match=msg):
        _tuple_to_dict(x, tup, use_latex=False)


@pytest.mark.parametrize(
    "use_latex, latex_wrapper, expected",
    [
        (False, "$%s$", "x_2"),
        (True, "$%s$", "$x_{2}$"),
        (True, "$$%s$$", "$$x_{2}$$"),
    ]
)
def test_tuple_to_dict_use_latex(use_latex, latex_wrapper, expected):
    x = symbols("x_2")
    t = _tuple_to_dict(x, (1, 0, 5),
        use_latex=use_latex, latex_wrapper=latex_wrapper)
    assert t["description"] == expected


@pytest.mark.parametrize(
    "backend, imodule, instance", [
        (MB, "ipywidgets", IPYInteractivePlot),
        (PB, "ipywidgets", IPYInteractivePlot),
        (BB, "ipywidgets", IPYInteractivePlot),
        (MB, "panel", PANELInteractivePlot),
        (PB, "panel", PANELInteractivePlot),
        (BB, "panel", PANELInteractivePlot)
    ]
)
def test_mix_interactive_non_interactive(backend, imodule, instance):
    # in this test, for some reasons the user wants to use a non-interactive
    # series with variable `a`, and an interactive series with parameter `a`.
    # the non-interactive series should not be updated by the params of
    # interactive series.
    a, b = symbols("a, b")
    params = {a: (1, 0, 5), b: (2, 0, 5)}
    p = graphics(
        line(cos(a), (a, 0, 10), n=10),
        HVLineSeries(a, horizontal=True, params=params),
        HVLineSeries(b, horizontal=False, params=params),
        backend=backend, show=False, imodule=imodule
    )
    assert isinstance(p, instance)
    assert isinstance(p._backend[0], LineOver1DRangeSeries)
    d1 = p._backend[0].get_data()
    d2 = p._backend[1].get_data()
    d3 = p._backend[2].get_data()
    p._backend.update_interactive({a: 2, b: 3})
    d4 = p._backend[0].get_data()
    d5 = p._backend[1].get_data()
    d6 = p._backend[2].get_data()
    assert np.allclose(d1, d4)
    assert not np.isclose(d2, d5)
    assert not np.isclose(d3, d6)


@pytest.mark.filterwarnings("ignore:The following keyword arguments are unused.")
@pytest.mark.parametrize(
    "imodule, iplot_type, backend", [
        ("panel", PANELInteractivePlot, MB),
        ("panel", PANELInteractivePlot, PB),
        ("panel", PANELInteractivePlot, BB),
        ("ipywidgets", IPYInteractivePlot, MB),
        ("ipywidgets", IPYInteractivePlot, PB),
        ("ipywidgets", IPYInteractivePlot, BB),
    ]
)
def test_iplot_sum_1(
    imodule, iplot_type, backend, ipywidgets_options, panel_options
):
    # verify that it is possible to add together different instances of
    # InteractivePlot (as well as Plot instances), provided that the same
    # parameters are used.

    if imodule == "panel":
        options = panel_options.copy()
        additional_options = {"pane_kw": {"width": 500}}
    else:
        options = ipywidgets_options.copy()
        additional_options = {}

    x, u = symbols("x, u")

    params = {u: (1, 0, 2)}
    p1 = plot(
        cos(u * x), (x, -5, 5),
        params=params,
        backend=backend,
        xlabel="x1",
        ylabel="y1",
        title="title 1",
        legend=True,
        **options, **additional_options
    )
    p2 = plot(
        sin(u * x), (x, -5, 5),
        params=params,
        backend=backend,
        xlabel="x2",
        ylabel="y2",
        title="title 2",
        **options
    )
    p3 = plot(
        sin(x) * cos(x), (x, -5, 5),
        backend=backend,
        adaptive=False,
        n=50,
        is_point=True,
        is_filled=True,
        line_kw=dict(marker="^"),
        **options
    )
    p = p1 + p2 + p3

    assert isinstance(p, iplot_type)
    assert isinstance(p.backend, backend)
    assert p.backend.title == "title 1"
    assert p.backend.xlabel == "x1"
    assert p.backend.ylabel == "y1"
    assert p.backend.legend
    assert len(p.backend.series) == 3
    assert len([s for s in p.backend.series if s.is_interactive]) == 2
    assert len([s for s in p.backend.series if not s.is_interactive]) == 1
    if imodule == "panel":
        assert p.pane_kw == {"width": 500}


@pytest.mark.parametrize(
    "imodule", ["panel", "ipywidgets"]
)
def test_iplot_sum_2(imodule, ipywidgets_options, panel_options):
    # verify that it is not possible to add together different instances of
    # InteractivePlot when they are using different parameters

    if imodule == "panel":
        options = panel_options.copy()
    else:
        options = ipywidgets_options.copy()

    x, u, v = symbols("x, u, v")

    p1 = plot(
        cos(u * x), (x, -5, 5),
        params={u: (1, 0, 1)},
        backend=MB,
        xlabel="x1",
        ylabel="y1",
        title="title 1",
        legend=True,
        **options
    )
    p2 = plot(
        sin(u * x) + v, (x, -5, 5),
        params={u: (1, 0, 1), v: (0, -2, 2)},
        backend=MB,
        xlabel="x2",
        ylabel="y2",
        title="title 2",
        **options
    )
    raises(ValueError, lambda: p1 + p2)
