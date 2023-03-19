from spb.backends.base_backend import Plot
from spb.series import BaseSeries
from pytest import raises
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import k3d
import bokeh
from .make_tests import *


KB.skip_notebook_check = True


class UnsupportedSeries(BaseSeries):
    pass


def test_instance_plot():
    # verify that instantiating the Plot base class creates the correct
    # backend

    from bokeh.plotting.figure import Figure
    from k3d.plot import Plot as K3DPlot

    x, y, z = symbols("x, y, z")

    s = LineOver1DRangeSeries(cos(x), (x, -10, 10), "test")

    p = Plot(s, backend=MB, show=False)
    assert isinstance(p, MB)
    assert isinstance(p.fig, plt.Figure)

    p = Plot(s, backend=PB, show=False)
    assert isinstance(p, PB)
    assert isinstance(p.fig, go.Figure)

    p = Plot(s, backend=BB, show=False)
    assert isinstance(p, BB)
    assert isinstance(p.fig, Figure)

    s = SurfaceOver2DRangeSeries(cos(x * y), (x, -5, 5), (y, -5, 5), "test")
    p = Plot(s, backend=KB, show=False)
    assert isinstance(p, KB)
    assert isinstance(p.fig, K3DPlot)


def test_unsupported_series():
    # verify that an error is raised when an unsupported series is given in
    series = [UnsupportedSeries()]
    raises(
        NotImplementedError,
        lambda: Plot(*series, backend=MB).process_series())
    raises(
        NotImplementedError,
        lambda: Plot(*series, backend=PB).process_series())
    raises(
        NotImplementedError,
        lambda: Plot(*series, backend=BB).process_series())
    raises(
        NotImplementedError,
        lambda: Plot(*series, backend=KB).process_series())


def test_common_keywords():
    # TODO: here I didn't test axis_center, autoscale, margin
    kw = dict(
        title="a",
        xlabel="x",
        ylabel="y",
        zlabel="z",
        aspect="equal",
        grid=False,
        xscale="log",
        yscale="log",
        zscale="log",
        legend=True,
        xlim=(-1, 1),
        ylim=(-2, 2),
        zlim=(-3, 3),
        size=(5, 10),
        backend=BB,
    )
    p = Plot(**kw)
    assert isinstance(p, BB)
    assert p.title == "a"
    assert p.xlabel == "x"
    assert p.ylabel == "y"
    assert p.zlabel == "z"
    assert p.aspect == "equal"
    assert p.grid is False
    assert p.xscale == "log"
    assert p.yscale == "log"
    assert p.zscale == "log"
    assert p.legend is True
    assert p.xlim == (-1, 1)
    assert p.ylim == (-2, 2)
    assert p.zlim == (-3, 3)
    assert p.size == (5, 10)


def test_plot_sum():
    x, y = symbols("x, y")

    options = dict(adaptive=False, n=5, show=False)

    # the choice of the backend dictates the keyword arguments
    # inside rendering_kw
    p1 = plot(sin(x), backend=PB, rendering_kw=dict(line_color='black'),
        xlabel="x1", ylabel="y1", **options)
    p2 = plot(cos(x), backend=PB, rendering_kw=dict(line_dash='dash'),
        xlabel="x2", ylabel="y2", **options)
    p3 = plot(sin(x) * cos(x), backend=PB,
        rendering_kw=dict(line_dash='dot'), **options)
    p4 = p1 + p2 + p3
    assert isinstance(p4, PB)
    assert len(p4.series) == 3
    assert p4.series[0].expr == sin(x)
    assert p4.fig.data[0]["line"]["color"] == "black"
    colors = set()
    for l in p4.fig.data:
        colors.add(l["line"]["color"])
    assert len(colors) == 3
    assert p4.series[1].expr == cos(x)
    assert p4.fig.data[1]["line"]["dash"] == "dash"
    assert p4.series[2].expr == sin(x) * cos(x)
    assert p4.fig.data[2]["line"]["dash"] == "dot"
    # two or more series in the result: automatic legend turned on
    assert p4.legend is True
    # the resulting plot uses the attributes of the first plot in the sum
    assert p4.xlabel == "x1" and p4.ylabel == "y1"
    p4 = p2 + p1 + p3
    assert p4.xlabel == "x2" and p4.ylabel == "y2"

    # plots can be added together with sum()
    p5 = sum([p1, p2, p3])
    assert isinstance(p5, PB)
    assert len(p5.series) == 3

    # summing different types of plots: the result is consistent with the
    # original visualization. In particular, if no `rendering_kw` is given
    # to `p2` then the backend will use automatic coloring to differentiate
    # the series.
    hex2rgb = lambda h: tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    p1 = plot_vector([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
        backend=MB, scalar=True, **options)
    p2 = plot(sin(x), (x, -3, 3), backend=MB, **options)
    p3 = p1 + p2
    assert isinstance(p3, MB)
    assert len(p3.series) == 3
    assert len(p3.fig.axes[0].collections) > 1
    assert isinstance(p3.fig.axes[0].collections[-1], matplotlib.quiver.Quiver)
    quiver_col = p3.fig.axes[0].collections[-1].get_facecolors().flatten()[:-1]
    first_col = np.array(p3.colorloop[0])
    if quiver_col.dtype == first_col.dtype:
        assert np.allclose(quiver_col, first_col)
    else:
        assert np.allclose(quiver_col, hex2rgb(str(first_col)[1:]))
    line_col = np.array(p3.fig.axes[0].lines[0].get_color())
    second_col = np.array(p3.colorloop[1])
    if line_col.dtype == second_col.dtype == "<U7":
        assert str(line_col) == str(second_col)
    else:
        assert np.allclose(line_col, hex2rgb(str(second_col)[1:]))

    # summing plots with different backends: the first backend will be used in
    # the result
    p1 = plot(sin(x), backend=MB, **options)
    p2 = plot(cos(x), backend=PB, **options)
    p3 = p1 + p2
    assert isinstance(p3, MB)

    # summing plots with different backends: fail when backend-specific
    # keyword arguments are used.
    # NOTE: the output plot is of type MB
    p1 = plot(sin(x), backend=MB, rendering_kw=dict(linestyle=":"), **options)
    p2 = plot(cos(x), backend=PB, rendering_kw=dict(line_dash="dash"), **options)
    raises(AttributeError, lambda: (p1 + p2).process_series())

    # verify that summing up bokeh plots doesn't raise errors
    p1 = plot(sin(x), (x, -pi, pi), backend=BB, **options)
    p2 = plot(cos(x), (x, -pi, pi), backend=BB, **options)
    p3 = p1 + p2

    # verify that summing up K3D plots doesn't raise errors
    p1 = plot3d(sin(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), backend=KB,
        **options)
    p2 = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), backend=KB,
        **options)
    p3 = p1 + p2