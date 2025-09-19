import pytest
from pytest import raises
ipywidgets = pytest.importorskip("ipywidgets")
from spb import (
    BB, PB, MB, plot, graphics, line, line_parametric_2d, domain_coloring
)
from spb.interactive.ipywidgets import (
    _build_widgets, InteractivePlot, _get_widget_from_param_module
)
from spb.utils import prange
from sympy import symbols, cos, sin, exp, pi, Float, Integer, Rational
import numpy as np
import param


def test_slider():
    # verify that by providing a tuple of elements to each parameters, we
    # end up with sliders. Simoultaneously, verify that we can use symbolic
    # numbers.

    x, phi, k, d = symbols("x, phi, k, d")
    params = {
        d: (Float(0.1), 0, 1),
        k: (2, Integer(1), 10, 20),
        phi: (0, 0, 2 * pi, 50, r"$\phi$ [rad]"),
        x: (1, -4, 5, Rational(100), "test", "log"),
    }
    widgets = _build_widgets(params, False)
    assert all(isinstance(w, ipywidgets.FloatSlider) for w in widgets[:-1])
    assert isinstance(widgets[-1], ipywidgets.FloatLogSlider)

    def get_values(i):
        return [
            widgets[i].value, widgets[i].min, widgets[i].max, widgets[i].step
        ]

    assert np.allclose(get_values(0), [0.1, 0, 1, (1 - 0) / 40])
    assert widgets[0].description == "d"
    assert np.allclose(get_values(1), [2, 1, 10, (10 - 1) / 20])
    assert widgets[1].description == "k"
    assert np.allclose(get_values(2), [0, 0, 2 * np.pi, (2 * np.pi - 0) / 50])
    assert widgets[2].description == r"$\phi$ [rad]"
    assert np.allclose([
        widgets[3].value, widgets[3].min,
        widgets[3].max],
        [1, -4, 5]
    )
    assert widgets[3].description == "test"


def test_widgets(ipywidgets_options):
    # verify that widgets can be used as parameters, instead of tuples

    x, phi, n, d = symbols("x, phi, n, d")
    params = {
        d: (0.1, 0, 1),
        n: ipywidgets.BoundedIntText(
            value=2, min=1, max=10, description="$n$"
        ),
        phi: (0, 0, 2 * pi, 50, r"$\phi$ [rad]"),
    }
    widgets = _build_widgets(params, False)
    assert isinstance(widgets[0], ipywidgets.FloatSlider)
    assert isinstance(widgets[1], ipywidgets.BoundedIntText)
    assert isinstance(widgets[2], ipywidgets.FloatSlider)
    plot(
        cos(x * n - phi) * exp(-abs(x) * d),
        (x, -5 * pi, 5 * pi),
        params=params,
        backend=PB,
        ylim=(-1.25, 1.25),
        n=10,
        **ipywidgets_options
    )


def test_plot_layout(ipywidgets_options):
    # verify that the plot uses the correct layout.

    x, t = symbols("x, t")
    options = ipywidgets_options.copy()
    options["show"] = True

    p1 = plot(
        cos(x) * exp(-x * t), (x, 0, 10),
        params={t: (0.1, 0, 2)},
        layout="tb",
        backend=PB,
        n=10,
        **ipywidgets_options
    )
    assert isinstance(p1, InteractivePlot)

    p1 = plot(
        cos(x) * exp(-x * t), (x, 0, 10),
        params={t: (0.1, 0, 2)},
        layout="tb",
        backend=PB,
        n=10,
        **options
    )
    assert isinstance(p1, ipywidgets.VBox)

    p2 = plot(
        cos(x) * exp(-x * t), (x, 0, 10),
        params={t: (0.1, 0, 2)},
        layout="bb",
        backend=PB,
        n=10,
        **options
    )
    assert isinstance(p2, ipywidgets.VBox)

    p3 = plot(
        cos(x) * exp(-x * t), (x, 0, 10),
        params={t: (0.1, 0, 2)},
        layout="sbl",
        backend=PB,
        n=10,
        **options
    )
    assert isinstance(p3, ipywidgets.HBox)

    p4 = plot(
        cos(x) * exp(-x * t), (x, 0, 10),
        params={t: (0.1, 0, 2)},
        layout="sbr",
        backend=PB,
        n=10,
        **options
    )
    assert isinstance(p4, ipywidgets.HBox)


@pytest.mark.skipif(ipywidgets is None, reason="ipywidgets is not installed")
def test_params_multi_value_widgets_1():
    a, b, c, x = symbols("a:c x")
    w1 = ipywidgets.FloatRangeSlider(value=(-2, 2), min=-5, max=5)
    w2 = ipywidgets.FloatSlider(value=1, min=0, max=5)
    p = plot(
        cos(c * x), prange(x, a, b), n=10,
        params={
            (a, b): w1,
            c: w2
        }, imodule="ipywidgets", show=False, backend=MB)
    fig = p.fig
    d1 = p.backend[0].get_data()
    # verify that no errors are raise when an update event is triggered
    w1.value = (-3, 3)
    w2.value = 2
    p._update(None)
    d2 = p.backend[0].get_data()
    assert not np.allclose(d1, d2)


@pytest.mark.skipif(ipywidgets is None, reason="ipywidgets is not installed")
def test_params_multi_value_widgets_2():
    a, b, c, d, x = symbols("a:d x")
    w1 = ipywidgets.FloatRangeSlider(value=(-2, 2), min=-5, max=5, step=0.1)
    w2 = ipywidgets.FloatSlider(value=1, min=0, max=5)
    w3 = ipywidgets.FloatSlider(value=2, min=0, max=6)
    p = graphics(
        line(cos(x), range_x=(x, -5, 5), n=10),
        line(cos(c * x), range_x=(x, a, b), n=10,
            params={
                (a, b): w1,
                c: w2
            }),
        line_parametric_2d(cos(d*x), sin(d*x), range_p=(x, -4, 4),
            n=10,
            params={d: w3}),
        imodule="ipywidgets", show=False, backend=MB)
    fig = p.fig
    d1 = p.backend[0].get_data()
    d2 = p.backend[1].get_data()
    d3 = p.backend[2].get_data()
    # verify that no errors are raise when an update event is triggered
    w1.value = (-3, 3)
    w2.value = 2
    w3.value = 3
    p._update(None)
    d4 = p.backend[0].get_data()
    d5 = p.backend[1].get_data()
    d6 = p.backend[2].get_data()
    assert np.allclose(d1, d4)
    assert not np.allclose(d2, d5)
    assert not np.allclose(d3, d6)


class ParamInteger(param.Parameterized):
    a = param.Integer(default=5)
    b = param.Integer(default=4, bounds=(-10, 10), step=2)
    c = param.Integer(default=5, bounds=(-10, 10), softbounds=(-70, 80))
    d = param.Integer(default=5, bounds=(-10, 10), softbounds=(-7, 8))
    e = param.Integer(default=5, bounds=(-10, 10), softbounds=(-7, None))
    f = param.Integer(default=5, bounds=(-10, 10), softbounds=(None, 8))
    g = param.Integer(default=5, inclusive_bounds=(True, True))
    h = param.Integer(default=5, bounds=(-10, 10), inclusive_bounds=(True, True))
    i = param.Integer(default=5, bounds=(-10, 10), inclusive_bounds=(False, True))
    j = param.Integer(default=5, bounds=(-10, 10), inclusive_bounds=(True, False))
    k = param.Integer(default=5, bounds=(-10, 10), inclusive_bounds=(False, False))
    l = param.Integer(default=5, bounds=(2, None))
    m = param.Integer(default=6, bounds=(None, 8), label="test", step=2)


class ParamNumber(param.Parameterized):
    a = param.Number(default=5)
    b = param.Number(default=4, bounds=(-10, 10), step=0.5)
    c = param.Number(default=5, bounds=(-10, 10), softbounds=(-70, 80))
    d = param.Number(default=5, bounds=(-10, 10), softbounds=(-7, 8))
    e = param.Number(default=5, bounds=(-10, 10), softbounds=(-7, None))
    f = param.Number(default=5, bounds=(-10, 10), softbounds=(None, 8))
    g = param.Number(default=5, inclusive_bounds=(True, True))
    h = param.Number(default=5, bounds=(-10, 10), inclusive_bounds=(True, True))
    i = param.Number(default=5, bounds=(-10, 10), inclusive_bounds=(False, True))
    j = param.Number(default=5, bounds=(-10, 10), inclusive_bounds=(True, False))
    k = param.Number(default=5, bounds=(-10, 10), inclusive_bounds=(False, False), step=0.1)
    l = param.Number(default=5, bounds=(2, None))
    m = param.Number(default=6, bounds=(None, 8), label="test", step=0.1)


class ParamSelector(param.Parameterized):
    a = param.Selector(default="a", objects=["a", "b", "c"])
    b = param.Selector(default="a", objects={"label 1": "a", "label 2": "b", "label 3": "c"})
    c = param.Selector(default="a", objects=["a", 2, True], label="test")


class ParamBoolean(param.Parameterized):
    a = param.Boolean(default=True)
    b = param.Boolean(default=False)


class ParamRange(param.Parameterized):
    a = param.Range(default=(1, 3))
    b = param.Range(default=(1, 3), bounds=(-5, 5))
    c = param.Range(default=(1, 3), bounds=(-5, 5), softbounds=(-3, 4))
    d = param.Range(default=(1, 3), bounds=(None, 5), softbounds=(-3, 4))
    e = param.Range(default=(1, 3), bounds=(-5, 5), softbounds=(None, 4))
    f = param.Range(default=(1, 3), bounds=(-5, None), softbounds=(-3, 4))
    g = param.Range(default=(1, 3), bounds=(-5, 5), softbounds=(-3, None))
    h = param.Range(default=(1, 3), bounds=(-5, None), softbounds=(-3, None))
    i = param.Range(default=(1, 3), inclusive_bounds=(True, True))
    j = param.Range(default=(1, 3), inclusive_bounds=(False, True))
    k = param.Range(default=(1, 3), inclusive_bounds=(True, False))
    l = param.Range(default=(1, 3), inclusive_bounds=(False, False))
    m = param.Range(default=(3, 3), label="test")
    n = param.Range(default=(3, 3), step=0.1)
    o = param.Range(default=(2.5, 3))


@pytest.mark.parametrize("p_name, label, val, min_, max_, step, w_type", [
    ("a", "A", 5, None, None, 1, ipywidgets.IntText),
    ("b", "B", 4, -10, 10, 2, ipywidgets.IntSlider),
    ("c", "C", 5, -10, 10, 1, ipywidgets.IntSlider),
    ("d", "D", 5, -7, 8, 1, ipywidgets.IntSlider),
    ("e", "E", 5, -7, 10, 1, ipywidgets.IntSlider),
    ("f", "F", 5, -10, 8, 1, ipywidgets.IntSlider),
    ("g", "G", 5, None, None, 1, ipywidgets.IntText),
    ("h", "H", 5, -10, 10, 1, ipywidgets.IntSlider),
    ("i", "I", 5, -9, 10, 1, ipywidgets.IntSlider),
    ("j", "J", 5, -10, 9, 1, ipywidgets.IntSlider),
    ("k", "K", 5, -9, 9, 1, ipywidgets.IntSlider),
    ("l", "L", 5, 2, 2000, 1, ipywidgets.BoundedIntText),
    ("m", "test", 6, -8000, 8, 2, ipywidgets.BoundedIntText),
])
def test_get_widget_from_param_module_integer(
    p_name, label, val, min_, max_, step, w_type
):
    obj = ParamInteger()
    widget = _get_widget_from_param_module(obj, p_name)
    assert isinstance(widget, w_type)
    assert widget.description == label
    assert widget.value == val
    assert widget.step == step
    if min_ is None:
        assert not hasattr(widget, "min")
    else:
        assert widget.min == min_
    if max_ is None:
        assert not hasattr(widget, "max")
    else:
        assert widget.max == max_


@pytest.mark.parametrize("p_name, label, val, min_, max_, step, w_type", [
    ("a", "A", 5, None, None, 1, ipywidgets.FloatText),
    ("b", "B", 4, -10, 10, 0.5, ipywidgets.FloatSlider),
    ("c", "C", 5, -10, 10, 1, ipywidgets.FloatSlider),
    ("d", "D", 5, -7, 8, 1, ipywidgets.FloatSlider),
    ("e", "E", 5, -7, 10, 1, ipywidgets.FloatSlider),
    ("f", "F", 5, -10, 8, 1, ipywidgets.FloatSlider),
    ("g", "G", 5, None, None, 1, ipywidgets.FloatText),
    ("h", "H", 5, -10, 10, 1, ipywidgets.FloatSlider),
    ("i", "I", 5, -9, 10, 1, ipywidgets.FloatSlider),
    ("j", "J", 5, -10, 9, 1, ipywidgets.FloatSlider),
    ("k", "K", 5, -9.9, 9.9, 0.1, ipywidgets.FloatSlider),
    ("l", "L", 5, 2, 2000, 1, ipywidgets.BoundedFloatText),
    ("m", "test", 6, -8000, 8, 0.1, ipywidgets.BoundedFloatText),
])
def test_get_widget_from_param_module_number(
    p_name, label, val, min_, max_, step, w_type
):
    obj = ParamNumber()
    widget = _get_widget_from_param_module(obj, p_name)
    assert isinstance(widget, w_type)
    assert widget.description == label
    assert widget.value == val
    assert widget.step == step
    if min_ is None:
        assert not hasattr(widget, "min")
    else:
        assert widget.min == min_
    if max_ is None:
        assert not hasattr(widget, "max")
    else:
        assert widget.max == max_


@pytest.mark.parametrize("p_name, label, val, options, w_type", [
    ("a", "A", "a", ("a", "b", "c"), ipywidgets.Dropdown),
    ("b", "B", "a", ("a", "b", "c"), ipywidgets.Dropdown),
    ("c", "test", "a", ("a", 2, True), ipywidgets.Dropdown),
])
def test_get_widget_from_param_module_selector(
    p_name, label, val, options, w_type
):
    obj = ParamSelector()
    widget = _get_widget_from_param_module(obj, p_name)
    assert isinstance(widget, w_type)
    assert widget.description == label
    assert widget.value == val
    assert widget.options == options


@pytest.mark.parametrize("p_name, label, val, w_type", [
    ("a", "A", True, ipywidgets.Checkbox),
    ("b", "B", False, ipywidgets.Checkbox),
])
def test_get_widget_from_param_module_boolean(p_name, label, val, w_type):
    obj = ParamBoolean()
    widget = _get_widget_from_param_module(obj, p_name)
    assert isinstance(widget, w_type)
    assert widget.description == label
    assert widget.value == val


@pytest.mark.parametrize("p_name, label, val, start, end, step, w_type", [
    ("a", "A", (1, 3), 1, 3, 1, ipywidgets.IntRangeSlider),
    ("b", "B", (1, 3), -5, 5, 1, ipywidgets.IntRangeSlider),
    ("c", "C", (1, 3), -3, 4, 1, ipywidgets.IntRangeSlider),
    ("d", "D", (1, 3), -3, 4, 1, ipywidgets.IntRangeSlider),
    ("e", "E", (1, 3), -5, 4, 1, ipywidgets.IntRangeSlider),
    ("f", "F", (1, 3), -3, 4, 1, ipywidgets.IntRangeSlider),
    ("g", "G", (1, 3), -3, 5, 1, ipywidgets.IntRangeSlider),
    ("h", "H", (1, 3), -3, 3, 1, ipywidgets.IntRangeSlider),
    ("i", "I", (1, 3), 1, 3, 1, ipywidgets.IntRangeSlider),
    ("j", "J", (2, 3), 2, 3, 1, ipywidgets.IntRangeSlider),
    ("k", "K", (1, 2), 1, 2, 1, ipywidgets.IntRangeSlider),
    ("l", "L", (2, 2), 2, 2, 1, ipywidgets.IntRangeSlider),
    ("m", "test", (3, 3), -7, 13, 1, ipywidgets.IntRangeSlider),
    ("n", "N", (3, 3), 2, 4, 0.1, ipywidgets.FloatRangeSlider),
    ("o", "O", (2.5, 3), 2.5, 3, 0.05, ipywidgets.FloatRangeSlider),
])
def test_get_widget_from_param_module_range(
    p_name, label, val, start, end, step, w_type
):
    obj = ParamRange()
    widget = _get_widget_from_param_module(obj, p_name)
    assert isinstance(widget, w_type)
    assert widget.description == label
    assert widget.value == val
    assert np.isclose(widget.step, step)
    assert np.isclose(widget.min, start)
    assert np.isclose(widget.max, end)


@pytest.mark.parametrize("backend", [MB, BB])
def test_domain_coloring_series_ui_controls(backend):
    # verify that UI controls related to ComplexDomainColoringSeries
    # are added to the interactive application

    x, u = symbols("x, u")
    p = graphics(
        domain_coloring(
            sin(u*x), (x, -2-2j, 2+2j), params={u: (1, 0, 2)}, n=10),
        backend=backend,
        grid=False,
        imodule="ipywidgets",
        layout="sbl",
        ncols=1,
        show=False
    )
    fig = p.fig
    s = p.backend[0]
    assert s.coloring == "a"
    _, _, _, _, img1, _ = s.get_data()

    # verify that no errors are raise when an update event is triggered
    widgets = list(p._additional_widgets.values())[0]
    widgets[2].value = "b"
    _, _, _, _, img2, _ = s.get_data()
    assert not np.allclose(img1, img2)

    widgets[0].value = 20
    _, _, _, _, img3, _ = s.get_data()
    assert img2.shape != img3.shape
