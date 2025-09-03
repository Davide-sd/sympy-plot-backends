import pytest
panel = pytest.importorskip("panel")
from spb.backends.utils import (
    convert_colormap,
    tick_formatter_multiples_of,
    multiples_of_2_pi,
    multiples_of_pi,
    multiples_of_pi_over_2,
    multiples_of_pi_over_3,
    multiples_of_pi_over_4
)
from sympy.external import import_module
from sympy import pi, symbols, E, I
from sympy.abc import x


# NOTE:
#
# Let's verify that it is possible to use a color map from a specific
# plotting library with any other plotting libraries.
#

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorcet as cc
import k3d
import numpy as np
from bokeh.models import CustomJSTickFormatter, SingleIntervalTicker

_plotly_utils = import_module(
    "_plotly_utils",
    import_kwargs={"fromlist": ["basevalidators"]},
    catch=(RuntimeError,),
)

# load color maps
colorcet_cms = [
    getattr(cc, t)
    for t in dir(cc)
    if (t[0] != "_") and isinstance(getattr(cc, t), list)
]
get_cms = lambda module: [
    getattr(module, a)
    for a in dir(module)
    if (
        (a[0] != "_")
        and isinstance(getattr(module, a), list)
        and all([isinstance(e, (float, int)) for e in getattr(module, a)])
    )
]
k3d_cms = (
    get_cms(k3d.basic_color_maps)
    + get_cms(k3d.matplotlib_color_maps)
    + get_cms(k3d.paraview_color_maps)
)
c = _plotly_utils.basevalidators.ColorscaleValidator("colorscale", "")
plotly_colorscales = list(c.named_colorscales.keys())
matplotlib_cm = [cm.jet, cm.Blues]


def test_convert_to_k3d():
    # K3D color maps are list of floats:
    # colormap = [loc1, r1, g1, b1, loc2, r2, g2, b2, ...]
    # Therefore, len(colormap) % 4 = 0!

    def do_test(colormaps, same=False):
        if not same:
            for c in colormaps:
                colormap = convert_colormap(c, "k3d")
                assert all([isinstance(t, (float, int)) for t in colormap])
                # note that k3d built-in color maps may have locations < 0
                assert all([(t >= 0) and (t <= 1) for t in colormap])
                assert len(colormap) % 4 == 0
        else:
            for c in colormaps:
                colormap = convert_colormap(c, "k3d")
                assert c == colormap

    do_test(colorcet_cms)
    do_test(plotly_colorscales)
    do_test(matplotlib_cm)
    do_test(k3d_cms, True)


def test_convert_to_plotly():
    # Plotly color scales have the following form:
    # [[loc1, 'rgb1'], [loc2, 'rgb2'], ...]
    # Or:
    # [[loc1, '#hex1'], [loc2, '#hex2'], ...]

    def do_test(colormaps, same=False):
        if not same:
            for c in colormaps:
                r = convert_colormap(c, "plotly")
                assert isinstance(r, list)
                assert all([isinstance(t, list) for t in r])
                assert all([len(t) == 2 for t in r])
                assert all(
                    [isinstance(t[0], (int, float, np.int64)) for t in r]
                )
                assert all(
                    [
                        isinstance(t[1], str)
                        and ((t[1][:3] == "rgb") or (t[1][0] == "#"))
                        and ("int" not in t)
                        for t in r
                    ]
                )
        else:
            for c in colormaps:
                colormap = convert_colormap(c, "plotly")
                assert c == colormap

    do_test(colorcet_cms)
    do_test(plotly_colorscales, True)
    do_test(matplotlib_cm)
    do_test(k3d_cms)


def test_convert_to_matplotlib():
    # The returned result should be a 2D array with 4 columns, RGBA

    def do_test(colormaps, same=False):
        if not same:
            for c in colormaps:
                r = convert_colormap(c, "matplotlib")
                assert isinstance(r, np.ndarray)
                assert r.shape[1] == 4
                assert np.all(r >= 0)
                assert np.all(r <= 1)
        else:
            for c in colormaps:
                colormap = convert_colormap(c, "matplotlib")
                assert c == colormap

    do_test(colorcet_cms)
    do_test(plotly_colorscales)
    do_test(matplotlib_cm, True)
    do_test(k3d_cms)


@pytest.mark.parametrize("q, expected_q", [
    (np.pi, np.pi),
    (pi, np.pi),
    (np.e, np.e),
    (E, np.e),
])
def test_tick_formatter_multiples_of_cast_quantity_to_float(q, expected_q):
    t = tick_formatter_multiples_of(quantity=q)
    assert type(t.quantity) == type(expected_q)
    assert np.isclose(t.quantity, expected_q)


@pytest.mark.parametrize("q", ["a", x, x**2, 2+3*I])
def test_tick_formatter_wrong_quantity_type_1(q):
    pytest.raises(ValueError, lambda : tick_formatter_multiples_of(quantity=q))


@pytest.mark.parametrize("q", [2+3j])
def test_tick_formatter_wrong_quantity_type_1(q):
    # cannot convert complex to float
    pytest.raises(TypeError, lambda : tick_formatter_multiples_of(quantity=q))


def test_tick_formatter_MB_1():
    tf = tick_formatter_multiples_of(
        quantity=np.pi, label="\\pi", n=2)
    assert hasattr(tf, "MB_func_formatter")
    assert hasattr(tf, "MB_major_locator")
    assert hasattr(tf, "MB_minor_locator")
    f = tf.MB_func_formatter()
    assert callable(f)
    assert isinstance(tf.MB_major_locator(), plt.MultipleLocator)
    assert isinstance(tf.MB_minor_locator(), plt.MultipleLocator)


@pytest.mark.parametrize("q, l, n, t, out", [
    [np.pi, "\\pi", 0.5, 0, "$0$"],
    [np.pi, "\\pi", 0.5, 2*np.pi, "$2\\pi$"],
    [np.pi, "\\pi", 0.5, -2*np.pi, "$-2\\pi$"],
    [np.pi, "\\pi", 1, 0, "$0$"],
    [np.pi, "\\pi", 1, np.pi, "$\\pi$"],
    [np.pi, "\\pi", 1, -np.pi, "$-\\pi$"],
    [np.pi, "\\pi", 1, 2*np.pi, "$2\\pi$"],
    [np.pi, "\\pi", 1, -2*np.pi, "$-2\\pi$"],
    [np.pi, "\\pi", 2, 0, "$0$"],
    [np.pi, "\\pi", 2, np.pi, "$\\pi$"],
    [np.pi, "\\pi", 2, -np.pi, "$-\\pi$"],
    [np.pi, "\\pi", 2, np.pi/2, "$\\frac{\\pi}{2}$"],
    [np.pi, "\\pi", 2, -np.pi/2, "$-\\frac{\\pi}{2}$"],
    [np.pi, "\\pi", 2, 3*np.pi/2, "$\\frac{3\\pi}{2}$"],
    [np.pi, "\\pi", 2, -3*np.pi/2, "$-\\frac{3\\pi}{2}$"],
    [np.pi, "\\pi", 3, 0, "$0$"],
    [np.pi, "\\pi", 3, np.pi/3, "$\\frac{\\pi}{3}$"],
    [np.pi, "\\pi", 3, -np.pi/3, "$-\\frac{\\pi}{3}$"],
    [np.pi, "\\pi", 3, 2*np.pi/3, "$\\frac{2\\pi}{3}$"],
    [np.pi, "\\pi", 3, -2*np.pi/3, "$-\\frac{2\\pi}{3}$"],
    [np.pi, "\\pi", 3, np.pi, "$\\pi$"],
    [np.pi, "\\pi", 3, -np.pi, "$-\\pi$"],
    [np.pi, "\\pi", 3, 4*np.pi/3, "$\\frac{4\\pi}{3}$"],
    [np.pi, "\\pi", 3, -4*np.pi/3, "$-\\frac{4\\pi}{3}$"],
    [np.pi, "\\pi", 4, 0, "$0$"],
    [np.pi, "\\pi", 4, np.pi/4, "$\\frac{\\pi}{4}$"],
    [np.pi, "\\pi", 4, -np.pi/4, "$-\\frac{\\pi}{4}$"],
    [np.pi, "\\pi", 4, np.pi/2, "$\\frac{\\pi}{2}$"],
    [np.pi, "\\pi", 4, -np.pi/2, "$-\\frac{\\pi}{2}$"],
    [np.pi, "\\pi", 4, 3*np.pi/4, "$\\frac{3\\pi}{4}$"],
    [np.pi, "\\pi", 4, -3*np.pi/4, "$-\\frac{3\\pi}{4}$"],
    [np.pi, "\\pi", 4, np.pi, "$\\pi$"],
    [np.pi, "\\pi", 4, -np.pi, "$-\\pi$"],
    [np.pi, "\\pi", 4, 5*np.pi/4, "$\\frac{5\\pi}{4}$"],
    [np.pi, "\\pi", 4, -5*np.pi/4, "$-\\frac{5\\pi}{4}$"],
    [np.e, "e", 2, 0, "$0$"],
    [np.e, "e", 2, np.e, "$e$"],
    [np.e, "e", 2, -np.e, "$-e$"],
    [np.e, "e", 2, np.e/2, "$\\frac{e}{2}$"],
    [np.e, "e", 2, -np.e/2, "$-\\frac{e}{2}$"],
    [np.e, "e", 2, 3*np.e/2, "$\\frac{3e}{2}$"],
    [np.e, "e", 2, -3*np.e/2, "$-\\frac{3e}{2}$"],
])
def test_tick_formatter_MB_2(q, l, n, t, out):
    tf = tick_formatter_multiples_of(quantity=q, label=l, n=n)
    f = tf.MB_func_formatter()
    assert f(t, 0) == out


def test_tick_formatter_BB_1():
    tf = tick_formatter_multiples_of(
        quantity=np.pi, label="π", n=2)
    assert hasattr(tf, "BB_ticker")
    assert hasattr(tf, "BB_formatter")
    assert isinstance(tf.BB_formatter(), CustomJSTickFormatter)
    assert isinstance(tf.BB_ticker(), SingleIntervalTicker)


def test_tick_formatter_PB_1():
    tf = tick_formatter_multiples_of(quantity=np.pi, label="π", n=1)
    assert hasattr(tf, "PB_ticks")

    vals, texts = tf.PB_ticks(-np.pi, np.pi, latex=False)
    assert np.allclose(vals, [-np.pi, 0, np.pi])
    assert texts == ["-π", "0", "π"]
    vals, texts = tf.PB_ticks(-np.pi, np.pi, latex=True)
    assert np.allclose(vals, [-np.pi, 0, np.pi])
    assert texts == ["$-π$", "$0$", "$π$"]

    # ticks starts before the numerical range, and ends after the
    # numerical range
    vals, texts = tf.PB_ticks(-1.5*np.pi, 1.5*np.pi, latex=False)
    assert np.allclose(vals, [-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    assert texts == ["-2π", "-π", "0", "π", "2π"]
    vals, texts = tf.PB_ticks(-1.5*np.pi, 1.5*np.pi, latex=True)
    assert np.allclose(vals, [-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    assert texts == ["$-2π$", "$-π$", "$0$", "$π$", "$2π$"]

    tf = tick_formatter_multiples_of(quantity=np.pi, label="π", n=2)
    vals, texts = tf.PB_ticks(-np.pi, np.pi, latex=False)
    assert np.allclose(vals, [-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    assert texts == ["-π", "-π/2", "0", "π/2", "π"]
    vals, texts = tf.PB_ticks(-np.pi, np.pi, latex=True)
    assert np.allclose(vals, [-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    assert texts == ["$-π$", "$-\\frac{π}{2}$", "$0$", "$\\frac{π}{2}$", "$π$"]

    tf = tick_formatter_multiples_of(quantity=np.pi, label="π", n=3)
    vals, texts = tf.PB_ticks(-2*np.pi, 2*np.pi, latex=False)
    assert np.allclose(vals, [-2*np.pi, -5*np.pi/3, -4*np.pi/3, -np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
    assert texts == ["-2π", "-5π/3", "-4π/3", "-π", "-2π/3", "-π/3", "0", "π/3", "2π/3", "π", "4π/3", "5π/3", "2π"]
    vals, texts = tf.PB_ticks(-2*np.pi, 2*np.pi, latex=True)
    assert np.allclose(vals, [-2*np.pi, -5*np.pi/3, -4*np.pi/3, -np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
    assert texts == ["$-2π$", "$-\\frac{5π}{3}$", "$-\\frac{4π}{3}$", "$-π$", "$-\\frac{2π}{3}$", "$-\\frac{π}{3}$", "$0$", "$\\frac{π}{3}$", "$\\frac{2π}{3}$", "$π$", "$\\frac{4π}{3}$", "$\\frac{5π}{3}$", "$2π$"]

    tf = tick_formatter_multiples_of(quantity=np.e, label="e", n=2)
    vals, texts = tf.PB_ticks(-np.e, np.e, latex=False)
    assert np.allclose(vals, [-np.e, -np.e/2, 0, np.e/2, np.e])
    assert texts == ["-e", "-e/2", "0", "e/2", "e"]
    vals, texts = tf.PB_ticks(-np.e, np.e, latex=True)
    assert np.allclose(vals, [-np.e, -np.e/2, 0, np.e/2, np.e])
    assert texts == ["$-e$", "$-\\frac{e}{2}$", "$0$", "$\\frac{e}{2}$", "$e$"]


@pytest.mark.parametrize("formatter_func, n, n_minor", [
    (multiples_of_2_pi, 0.5, 3),
    (multiples_of_pi, 1, 3),
    (multiples_of_pi_over_2, 2, 3),
    (multiples_of_pi_over_3, 3, 3),
    (multiples_of_pi_over_4, 4, 3),
])
def test_multiples_of_2_pi(formatter_func, n, n_minor):
    tf = formatter_func()
    assert np.isclose(tf.quantity, np.pi)
    assert np.isclose(tf.n, n)
    assert np.isclose(tf.n_minor, n_minor)