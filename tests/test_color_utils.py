from spb.backends.utils import convert_colormap
from pytest import raises

import matplotlib.cm as cm
import colorcet as cc
import k3d
from _plotly_utils.basevalidators import ColorscaleValidator
import numpy as np

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
c = ColorscaleValidator("colorscale", "")
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
                assert all([isinstance(t[0], (int, float, np.int64)) for t in r])
                assert all(
                    [
                        isinstance(t[1], str)
                        and ((t[1][:3] == "rgb") or (t[1][0] == "#"))
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
