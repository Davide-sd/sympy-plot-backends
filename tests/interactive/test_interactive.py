import pytest
bokeh = pytest.importorskip("bokeh")
from bokeh.models.formatters import PrintfTickFormatter
import math
from spb.interactive import _tuple_to_dict
from sympy import symbols, pi, Rational, Integer


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
