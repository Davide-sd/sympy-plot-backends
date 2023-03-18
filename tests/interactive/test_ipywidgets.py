from pytest import raises
from spb import BB, PB, MB, plot, plot3d
from spb.interactive.ipywidgets import _build_widgets
from sympy import symbols, cos, exp, pi, Float, Integer, Rational
import ipywidgets
import numpy as np


def test_slider():
    # verify that by providing a tuple of elements to each parameters, we
    # end up with sliders. Simoultaneously, verify that we can use symbolic
    # numbers.

    x, phi, k, d = symbols("x, phi, k, d")
    params = {
        d: (Float(0.1), 0, 1),
        k: (2, Integer(1), 10, 20),
        phi: (0, 0, 2*pi, 50, "$\phi$ [rad]"),
        x: (1, -4, 5, Rational(100), "test", "log")
    }
    widgets = _build_widgets(params, False)
    assert all(isinstance(w, ipywidgets.FloatSlider) for w in widgets[:-1])
    assert isinstance(widgets[-1], ipywidgets.FloatLogSlider)

    def get_values(i):
        return [widgets[i].value, widgets[i].min, widgets[i].max, widgets[i].step]

    assert np.allclose(get_values(0), [0.1, 0, 1, (1 - 0) / 40])
    assert widgets[0].description == "d"
    assert np.allclose(get_values(1), [2, 1, 10, (10 - 1) / 20])
    assert widgets[1].description == "k"
    assert np.allclose(get_values(2), [0, 0, 2*np.pi, (2*np.pi - 0) / 50])
    assert widgets[2].description == "$\phi$ [rad]"
    assert np.allclose(
        [widgets[3].value, widgets[3].min, widgets[3].max], [1, -4, 5])
    assert widgets[3].description == "test"


def test_widgets(ipywidgets_options):
    # verify that widgets can be used as parameters, instead of tuples

    x, phi, n, d = symbols("x, phi, n, d")
    params = {
        d: (0.1, 0, 1),
        n: ipywidgets.BoundedIntText(value=2, min=1, max=10, description="$n$"),
        phi: (0, 0, 2*pi, 50, "$\phi$ [rad]")
    }
    widgets = _build_widgets(params, False)
    assert isinstance(widgets[0], ipywidgets.FloatSlider)
    assert isinstance(widgets[1], ipywidgets.BoundedIntText)
    assert isinstance(widgets[2], ipywidgets.FloatSlider)
    plot(
        cos(x * n - phi) * exp(-abs(x) * d), (x, -5*pi, 5*pi),
        params=params,
        backend=PB, ylim=(-1.25, 1.25), n=10, **ipywidgets_options)


def test_plot_layout(ipywidgets_options):
    # verify that the plot uses the correct layout.

    x, t = symbols("x, t")

    p1 = plot(cos(x) * exp(-x * t), (x, 0, 10), params={t: (0.1, 0, 2)},
        layout="tb", backend=PB, n=10, **ipywidgets_options)
    assert isinstance(p1, ipywidgets.VBox)

    p2 = plot(cos(x) * exp(-x * t), (x, 0, 10), params={t: (0.1, 0, 2)},
        layout="bb", backend=PB, n=10, **ipywidgets_options)
    assert isinstance(p2, ipywidgets.VBox)

    p3 = plot(cos(x) * exp(-x * t), (x, 0, 10), params={t: (0.1, 0, 2)},
        layout="sbl", backend=PB, n=10, **ipywidgets_options)
    assert isinstance(p3, ipywidgets.HBox)

    p4 = plot(cos(x) * exp(-x * t), (x, 0, 10), params={t: (0.1, 0, 2)},
        layout="sbr", backend=PB, n=10, **ipywidgets_options)
    assert isinstance(p4, ipywidgets.HBox)
