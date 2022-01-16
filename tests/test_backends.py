from pytest import raises
from spb.backends.base_backend import Plot
from spb.backends.matplotlib import MB, unset_show
from spb.backends.bokeh import BB
from spb.backends.plotly import PB
from spb.backends.k3d import KB
from spb.series import BaseSeries, InteractiveSeries
from spb import (
    plot, plot3d, plot_contour, plot_implicit,
    plot_parametric, plot3d_parametric_line,
    plot_vector, plot_complex, plot_geometry, plot_real_imag,
    plot_list, plot_piecewise
)
from sympy import (
    symbols, cos, sin, Matrix, pi, sqrt, I, Heaviside, Piecewise, Eq
)
from sympy.geometry import Line, Circle, Polygon
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from tempfile import TemporaryDirectory
import os

unset_show()

x, y, z = symbols("x, y, z")

"""
NOTE
How to test plots? That's a great question. Here, I test that each backend:

1. receives the correct number of data series.
2. raises the necessary errors.
3. correctly use the common keyword arguments to customize the plot.
4. shows the expected labels.

This should be a decent starting point to provide a common user experience
between the different backends.
"""

# TODO:
#   1. test for empty plots
#   2. test for multiple colorscales when plotting multiple 3d expressions

# The following plot functions will be used by the different backends
p1 = lambda B: plot(
    sin(x), cos(x), sin(x / 2), cos(x / 2), 2 * sin(x), 2 * cos(x),
    backend=B, show=False,
)
p2 = lambda B, line_kw: plot(
    sin(x), cos(x), line_kw=line_kw, backend=B, show=False, legend=True
)
p3 = lambda B, line_kw: plot_parametric(
    cos(x), sin(x), (x, 0, 1.5 * pi), backend=B, show=False, line_kw=line_kw
)
p4 = lambda B, line_kw: plot3d_parametric_line(
    cos(x), sin(x), x, (x, -pi, pi), backend=B, show=False, line_kw=line_kw
)
p5 = lambda B, surface_kw: plot3d(
    cos(x ** 2 + y ** 2),
    (x, -3, 3),
    (y, -3, 3),
    n=20,
    use_cm=False,
    backend=B,
    show=False,
    surface_kw=surface_kw,
)
p6 = lambda B, contour_kw: plot_contour(
    cos(x ** 2 + y ** 2),
    (x, -3, 3),
    (y, -3, 3),
    n=20,
    backend=B,
    show=False,
    contour_kw=contour_kw,
)
p7a = lambda B, contour_kw, quiver_kw: plot_vector(
    Matrix([x, y]),
    (x, -5, 5),
    (y, -4, 4),
    backend=B,
    show=False,
    quiver_kw=quiver_kw,
    contour_kw=contour_kw,
)
p7b = lambda B, stream_kw, contour_kw: plot_vector(
    Matrix([x, y]),
    (x, -5, 5),
    (y, -4, 4),
    backend=B,
    scalar=(x + y),
    streamlines=True,
    show=False,
    stream_kw=stream_kw,
    contour_kw=contour_kw,
)
p7c = lambda B, stream_kw, contour_kw: plot_vector(
    Matrix([x, y]),
    (x, -5, 5),
    (y, -4, 4),
    backend=B,
    scalar=[(x + y), "test"],
    streamlines=True,
    show=False,
    stream_kw=stream_kw,
    contour_kw=contour_kw,
)
p9a = lambda B, quiver_kw, **kwargs: plot_vector(
    Matrix([z, y, x]),
    (x, -5, 5),
    (y, -4, 4),
    (z, -3, 3),
    backend=B,
    n=10,
    quiver_kw=quiver_kw,
    show=False,
    **kwargs
)
p9b = lambda B, stream_kw, kwargs=dict(): plot_vector(
    Matrix([z, y, x]),
    (x, -5, 5),
    (y, -4, 4),
    (z, -3, 3),
    backend=B,
    n=10,
    streamlines=True,
    show=False,
    stream_kw=stream_kw,
    **kwargs
)
p11a = lambda B, contour_kw: plot_implicit(
    x > y, (x, -5, 5), (y, -4, 4), n=20, backend=B, show=False,
    contour_kw=contour_kw
)
p11b = lambda B, contour_kw: plot_implicit(
    x > y,
    (x, -5, 5),
    (y, -4, 4),
    n=20,
    backend=B,
    adaptive=False,
    show=False,
    contour_kw=contour_kw,
)
p13 = lambda B: plot3d(
    (cos(x ** 2 + y ** 2), (x, -3, -2), (y, -3, 3)),
    (cos(x ** 2 + y ** 2), (x, -2, -1), (y, -3, 3)),
    (cos(x ** 2 + y ** 2), (x, -1, 0), (y, -3, 3)),
    (cos(x ** 2 + y ** 2), (x, 0, 1), (y, -3, 3)),
    (cos(x ** 2 + y ** 2), (x, 1, 2), (y, -3, 3)),
    (cos(x ** 2 + y ** 2), (x, 2, 3), (y, -3, 3)),
    n1 = 5, n2 = 5,
    backend=B,
    use_cm=False,
    show=False,
)
p14 = lambda B, line_kw: plot_real_imag(
    sqrt(x), (x, -5, 5), backend=B, line_kw=line_kw, show=False
)
p15a = lambda B, line_kw: plot_complex(
    sqrt(x), (x, -5, 5), backend=B, line_kw=line_kw, show=False
)
p15b = lambda B, image_kw: plot_complex(
    sqrt(x), (x, -5 - 5 * I, 5 + 5 * I), backend=B, coloring="a",
    image_kw=image_kw, show=False
)
p15c = lambda B, surface_kw: plot_complex(
    sqrt(x),
    (x, -5 - 5 * I, 5 + 5 * I),
    backend=B,
    threed=True,
    surface_kw=surface_kw,
    show=False,
    n=10
)
p16a = lambda B: plot_list([1, 2, 3], [1, 2, 3], 
    backend=B, is_point=True, is_filled=False, show=False)
p16b = lambda B: plot_list([1, 2, 3], [1, 2, 3], 
    backend=B, is_point=True, is_filled=True, show=False)
p17a = lambda B: plot_piecewise(
    Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10),
    backend=B, show=False)
p17b = lambda B: plot_piecewise(
    (Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10)),
    (Piecewise((sin(x), x < 0), (2, Eq(x, 0)), (cos(x), x > 0)), (x, -6, 4)),
    backend=B, show=False)
p18 = lambda B: plot_geometry(
    Line((1, 2), (5, 4)), Circle((0, 0), 4), Polygon((2, 2), 3, n=6),
    backend=B, show=False, fill=False
)


class UnsupportedSeries(BaseSeries):
    pass


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
    assert p.title == "a"
    assert p.xlabel == "x"
    assert p.ylabel == "y"
    assert p.zlabel == "z"
    assert p.aspect == "equal"
    assert p.grid == False
    assert p.xscale == "log"
    assert p.yscale == "log"
    assert p.zscale == "log"
    assert p.legend == True
    assert p.xlim == (-1, 1)
    assert p.ylim == (-2, 2)
    assert p.zlim == (-3, 3)
    assert p.size == (5, 10)


def test_plot_sum():
    # the choice of the backend dictates the keyword arguments inside line_kw
    p1 = plot(sin(x), backend=PB, line_kw=dict(line_color='black'),
        xlabel="x1", ylabel="y1", show=False)
    p2 = plot(cos(x), backend=PB, line_kw=dict(line_dash='dash'),
        xlabel="x2", ylabel="y2", show=False)
    p3 = plot(sin(x) * cos(x), backend=PB,
        line_kw=dict(line_dash='dot'), show=False)
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
    assert p4.legend == True
    # the resulting plot uses the attributes of the first plot in the sum
    assert p4.xlabel == "x1" and p4.ylabel == "y1"
    p4 = p2 + p1 + p3
    assert p4.xlabel == "x2" and p4.ylabel == "y2"

    # summing different types of plots: the result is consistent with the
    # original visualization. In particular, if no `line_kw` is given to `p2`
    # then the backend will use automatic coloring to differentiate the
    # series. 
    p1 = plot_vector([-sin(y), cos(x)], (x, -3, 3), (y, -3, 3),
        backend=MB, scalar=True, show=False)
    p2 = plot(sin(x), (x, -3, 3), backend=MB, show=False)
    p3 = p1 + p2
    assert isinstance(p3, MB)
    assert len(p3.series) == 3
    assert len(p3.fig.axes[0].collections) > 1
    assert type(p3.fig.axes[0].collections[-1]).__name__ == "Quiver"
    quiver_col = p3.fig.axes[0].collections[-1].get_facecolors().flatten()[:-1]
    first_col = np.array(p3.colorloop[0])
    assert np.allclose(quiver_col, first_col)
    line_col = np.array(p3.fig.axes[0].lines[0].get_color())
    second_col = np.array(p3.colorloop[1])
    assert np.allclose(line_col, second_col)

    # summing plots with different backends: the first backend will be used in
    # the result
    p1 = plot(sin(x), backend=MB, show=False)
    p2 = plot(cos(x), backend=PB, show=False)
    p3 = p1 + p2
    assert isinstance(p3, MB)

    # summing plots with different backends: fail when backend-specific
    # keyword arguments are used
    p1 = plot(sin(x), backend=MB, line_kw=dict(linestyle=":"), show=False)
    p2 = plot(cos(x), backend=PB, line_kw=dict(line_dash="dash"), show=False)
    raises(AttributeError, lambda: p1 + p2)


def test_MatplotlibBackend():
    assert hasattr(MB, "colorloop")
    assert isinstance(MB.colorloop, (ListedColormap, list, tuple))
    assert hasattr(MB, "colormaps")
    assert isinstance(MB.colormaps, (list, tuple))

    series = [UnsupportedSeries()]
    raises(NotImplementedError, lambda: Plot(*series, backend=MB))

    ### test for line_kw, surface_kw, quiver_kw, stream_kw: they should override
    ### defualt settings.

    p = p2(MB, line_kw=dict(color="red"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert isinstance(f, plt.Figure)
    assert isinstance(ax, Axes)
    assert len(ax.get_lines()) == 2
    assert ax.get_lines()[0].get_label() == "sin(x)"
    assert ax.get_lines()[0].get_color() == "red"
    assert ax.get_lines()[1].get_label() == "cos(x)"
    assert ax.get_lines()[1].get_color() == "red"
    p.close()

    p = p3(MB, line_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    # parametric plot with use_cm=True -> LineCollection
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], LineCollection)
    assert f.axes[1].get_ylabel() == "(cos(x), sin(x))"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = p4(MB, line_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], Line3DCollection)
    assert f.axes[1].get_ylabel() == "(cos(x), sin(x), x)"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    # use_cm=False will force to apply a default solid color to the mesh.
    # Here, I override that solid color with a custom color.
    p = p5(MB, surface_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], Poly3DCollection)
    # TODO: apparently, without showing the plot, the colors are not applied
    # to a Poly3DCollection...
    p.close()

    p = p6(MB, contour_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert f.axes[1].get_ylabel() == str(cos(x ** 2 + y ** 2))
    # TODO: how to retrieve the colormap from a contour series?????
    p.close()

    p = p7a(MB, quiver_kw=dict(color="red"), contour_kw=dict(cmap="jet"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert isinstance(ax.collections[-1], Quiver)
    assert f.axes[1].get_ylabel() == "Magnitude"
    # TODO: how to retrieve the colormap from a contour series?????
    p.close()

    p = p7b(MB, stream_kw=dict(color="red"), contour_kw=dict(cmap="jet"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert isinstance(ax.collections[-1], LineCollection)
    assert f.axes[1].get_ylabel() == "x + y"
    assert all(*(ax.collections[-1].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = p7c(MB, stream_kw=dict(color="red"), contour_kw=dict(cmap="jet"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert isinstance(ax.collections[-1], LineCollection)
    assert f.axes[1].get_ylabel() == "test"
    assert all(*(ax.collections[-1].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = p9a(MB, quiver_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], Line3DCollection)
    assert ax.collections[0].cmap.name == "jet"
    p.close()

    p = p9a(MB, quiver_kw=dict(cmap=None, color="red"), use_cm=False)
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], Line3DCollection)
    assert np.allclose(ax.collections[0].get_color(), np.array([[1., 0., 0., 1.]]))
    p.close()

    p = p9b(MB, stream_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], Line3DCollection)
    assert f.axes[1].get_ylabel() == "Matrix([[z], [y], [x]])"
    p.close()

    # test different combinations for streamlines: it should not raise errors
    p = p9b(MB, stream_kw=dict(starts=True))
    p = p9b(MB, stream_kw=dict(starts={
        "x": np.linspace(-5, 5, 10),
        "y": np.linspace(-4, 4, 10),
        "z": np.linspace(-3, 3, 10),
    }))
    p.close()

    # other keywords: it should not raise errors
    p = p9b(MB, stream_kw=dict(), kwargs=dict(use_cm=False))
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_color() == '#1f77b4'
    p.close()

    # NOTE: p11a is pretty much the same as p11b for MB
    p = p11b(MB, contour_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    # TODO: how to retrieve the colormap from a contour series?????
    p.close()

    p = p13(MB)
    assert len(p.series) == 6
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 6
    # TODO: apparently, without showing the plot, the colors are not applied
    # to a Poly3DCollection...
    p.close()

    p = p14(MB, line_kw=dict(color="red"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.get_lines()) == 2
    assert ax.get_lines()[0].get_label() == "Re(sqrt(x))"
    assert ax.get_lines()[0].get_color() == "red"
    assert ax.get_lines()[1].get_label() == "Im(sqrt(x))"
    assert ax.get_lines()[1].get_color() == "red"
    p.close()

    p = p15a(MB, line_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], LineCollection)
    assert f.axes[1].get_ylabel() == "Abs(sqrt(x))"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = p15b(MB, image_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.images) == 1
    assert f.axes[1].get_ylabel() == "Argument"
    assert ax.images[0].get_extent() == [-5.0, 5.0, -5.0, 5.0]
    p.close()

    p = p15b(MB, image_kw=dict(extent=[-6, 6, -7, 7]))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.images) == 1
    assert f.axes[1].get_ylabel() == "Argument"
    assert ax.images[0].get_extent() == [-6, 6, -7, 7]
    p.close()

    p = p15c(MB, surface_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], Poly3DCollection)
    assert f.axes[1].get_ylabel() == "Argument"
    # TODO: apparently, without showing the plot, the colors are not applied
    # to a Poly3DCollection...
    p.close()

    # verify markers
    p = p16a(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_markeredgecolor() != ax.lines[0].get_markerfacecolor()
    p.close()

    p = p16b(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_markeredgecolor() == ax.lines[0].get_markerfacecolor()
    p.close()

    # verify that plot_piecewise plotting N functions uses N colors
    p = p17a(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 4
    colors = set()
    for l in ax.lines:
        colors.add(l.get_color())
    assert len(colors) == 1

    p = p17b(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 9
    colors = set()
    for l in ax.lines:
        colors.add(l.get_color())
    assert len(colors) == 2

    # plot geometry
    p = p18(MB)
    assert len(p.series) == 3
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 3
    assert ax.get_lines()[0].get_label() == str(Line((1, 2), (5, 4)))
    assert ax.get_lines()[1].get_label() == str(Circle((0, 0), 4))
    assert ax.get_lines()[2].get_label() == str(Polygon((2, 2), 3, n=6))
    p.close()


class PBchild(PB):
    colorloop = ["red", "green", "blue"]


def test_PlotlyBackend():
    assert hasattr(PB, "colorloop")
    assert isinstance(PB.colorloop, (list, tuple))
    assert hasattr(PB, "colormaps")
    assert isinstance(PB.colormaps, (list, tuple))
    assert hasattr(PB, "quivers_colors")
    assert isinstance(PB.quivers_colors, (list, tuple))

    series = [UnsupportedSeries()]
    raises(NotImplementedError, lambda: Plot(*series, backend=PB))

    ### Setting custom color loop

    assert len(PBchild.colorloop) != len(PB.colorloop)
    _p1 = p1(PB)
    _p2 = p1(PBchild)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert all([isinstance(t, go.Scatter) for t in f1.data])
    assert all([isinstance(t, go.Scatter) for t in f2.data])
    # there are 6 unique colors in _p1 and 3 unique colors in _p2
    assert len(set([d["line"]["color"] for d in f1.data])) == 6
    assert len(set([d["line"]["color"] for d in f2.data])) == 3

    ### test for line_kw, surface_kw, quiver_kw, stream_kw: they should override
    ### defualt settings.

    p = p2(PB, line_kw=dict(line_color="red"))
    assert len(p.series) == 2
    f = p.fig
    assert isinstance(f, go.Figure)
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "sin(x)"
    assert f.data[0]["line"]["color"] == "red"
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[1]["name"] == "cos(x)"
    assert f.data[1]["line"]["color"] == "red"
    assert f.layout["showlegend"] == True

    p = p3(PB, line_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "(cos(x), sin(x))"
    assert f.data[0]["line"]["color"] == "red"
    assert f.data[0]["marker"]["colorbar"]["title"]["text"] == "(cos(x), sin(x))"

    p = p4(PB, line_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter3d)
    assert f.data[0]["line"]["color"] == "red"
    assert f.data[0]["name"] == "(cos(x), sin(x), x)"
    assert f.data[0]["line"]["colorbar"]["title"]["text"] == "(cos(x), sin(x), x)"

    # use_cm=False will force to apply a default solid color to the mesh.
    # Here, I override that solid color with a custom color.
    p = p5(PB, surface_kw=dict(colorscale=[[0, "cyan"], [1, "cyan"]]))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Surface)
    assert f.data[0]["name"] == "cos(x**2 + y**2)"
    assert f.data[0]["showscale"] == False
    assert f.data[0]["colorscale"] == ((0, "cyan"), (1, "cyan"))
    assert f.layout["showlegend"] == False
    assert f.data[0]["colorbar"]["title"]["text"] == "cos(x**2 + y**2)"

    p = p6(PB, contour_kw=dict(contours=dict(coloring="lines")))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Contour)
    assert f.data[0]["contours"]["coloring"] == "lines"
    assert f.data[0]["colorbar"]["title"]["text"] == str(cos(x ** 2 + y ** 2))

    p = p7a(
        PB,
        quiver_kw=dict(line_color="red"),
        contour_kw=dict(contours=dict(coloring="lines")),
    )
    assert len(p.series) == 2
    f = p.fig
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Contour)
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[0]["contours"]["coloring"] == "lines"
    assert f.data[0]["colorbar"]["title"]["text"] == "Magnitude"
    assert f.data[1]["line"]["color"] == "red"

    p = p7b(PB, stream_kw=dict(line_color="red"),
        contour_kw=dict(contours=dict(coloring="lines")),
    )
    assert len(p.series) == 2
    f = p.fig
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Contour)
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[0]["contours"]["coloring"] == "lines"
    assert f.data[0]["colorbar"]["title"]["text"] == "x + y"
    assert f.data[1]["line"]["color"] == "red"

    p = p7c(PB, stream_kw=dict(line_color="red"),
        contour_kw=dict(contours=dict(coloring="lines")))
    f = p.fig
    assert f.data[0]["colorbar"]["title"]["text"] == "test"

    p = p9a(PB, quiver_kw=dict(sizeref=5))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Cone)
    assert f.data[0]["sizeref"] == 5
    assert f.data[0]["colorbar"]["title"]["text"] == str(Matrix([z, y, x]))

    cs1 = f.data[0]["colorscale"]

    p = p9a(PB, quiver_kw=dict(colorscale="reds"))
    f = p.fig
    cs2 = f.data[0]["colorscale"]
    assert len(cs1) != len(cs2)

    p = p9b(PB, stream_kw=dict(colorscale=[[0, "red"], [1, "red"]]))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Streamtube)
    assert f.data[0]["colorscale"] == ((0, "red"), (1, "red"))
    assert f.data[0]["colorbar"]["title"]["text"] == str(Matrix([z, y, x]))

    # test different combinations for streamlines: it should not raise errors
    p = p9b(PB, stream_kw=dict(starts=True))
    p = p9b(PB, stream_kw=dict(starts={
        "x": np.linspace(-5, 5, 10),
        "y": np.linspace(-4, 4, 10),
        "z": np.linspace(-3, 3, 10),
    }))

    # other keywords: it should not raise errors
    p = p9b(MB, stream_kw=dict(), kwargs=dict(use_cm=False))

    p = p11a(PB, contour_kw=dict(colorscale=[[0, "rgba(0,0,0,0)"], [1, "red"]]))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Contour)
    assert f.data[0]["colorscale"] == ((0, "rgba(0,0,0,0)"), (1, "red"))

    p = p11b(PB, contour_kw=dict(fillcolor="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Contour)
    assert f.data[0]["fillcolor"] == "red"

    p = p14(PB, line_kw=dict(line_color="red"))
    assert len(p.series) == 2
    f = p.fig
    assert isinstance(f, go.Figure)
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "Re(sqrt(x))"
    assert f.data[0]["line"]["color"] == "red"
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[1]["name"] == "Im(sqrt(x))"
    assert f.data[1]["line"]["color"] == "red"
    assert f.layout["showlegend"] == True

    p = p15a(PB, line_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "Abs(sqrt(x))"
    assert f.data[0]["line"]["color"] == "red"
    assert p.fig.data[0]["marker"]["colorbar"]["title"]["text"] == "Abs(sqrt(x))"

    p = p15b(PB, image_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Image)
    assert f.data[0]["name"] == "sqrt(x)"
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[1]["marker"]["colorbar"]["title"]["text"] == "Argument"

    # TODO: set image_kw inside PB

    p = p15c(PB, surface_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Surface)
    assert f.data[0]["name"] == "sqrt(x)"
    assert f.data[0]["showscale"] == True
    assert f.data[0]["colorbar"]["title"]["text"] == "Argument"

    # test markers
    p = p16a(PB)
    assert len(p.series) == 1
    f = p.fig
    assert f.data[0]["marker"]["line"]["color"] is not None

    p = p16b(PB)
    assert len(p.series) == 1
    f = p.fig
    assert f.data[0]["marker"]["line"]["color"] is None

    # test plot_piecewise
    p = p17a(PB)
    assert len(p.series) == 4
    colors = set()
    for l in p.fig.data:
        colors.add(l["line"]["color"])
    assert len(colors) == 1

    p = p17b(PB)
    assert len(p.series) == 9
    colors = set()
    for l in p.fig.data:
        colors.add(l["line"]["color"])
    assert len(colors) == 2

    p = p18(PB)
    assert len(p.series) == 3
    f = p.fig
    assert len(f.data) == 3
    assert f.data[0]["name"] == str(Line((1, 2), (5, 4)))
    assert f.data[1]["name"] == str(Circle((0, 0), 4))
    assert f.data[2]["name"] == str(Polygon((2, 2), 3, n=6))


class BBchild(BB):
    colorloop = ["red", "green", "blue"]


def test_BokehBackend():
    from bokeh.models.glyphs import (Line, MultiLine, Image, Segment,
        ImageRGBA, Circle as BokehCircle)
    from bokeh.plotting.figure import Figure

    assert hasattr(BB, "colorloop")
    assert isinstance(BB.colorloop, (list, tuple))
    assert hasattr(BB, "colormaps")
    assert isinstance(BB.colormaps, (list, tuple))

    series = [UnsupportedSeries()]
    raises(NotImplementedError, lambda: Plot(*series, backend=PB))

    ### Test tools and tooltips on empty figure (populated figure might have
    # different tooltips, tested later on)
    from bokeh.models import (PanTool, WheelZoomTool, BoxZoomTool,
        ResetTool, HoverTool, SaveTool)
    f = plot(backend=BB, show=False).fig
    assert len(f.toolbar.tools) == 6
    assert isinstance(f.toolbar.tools[0], PanTool)
    assert isinstance(f.toolbar.tools[1], WheelZoomTool)
    assert isinstance(f.toolbar.tools[2], BoxZoomTool)
    assert isinstance(f.toolbar.tools[3], ResetTool)
    assert isinstance(f.toolbar.tools[4], HoverTool)
    assert isinstance(f.toolbar.tools[5], SaveTool)
    assert f.toolbar.tools[4].tooltips == [('x', '$x'), ('y', '$y')]

    ### Setting custom color loop

    assert len(BBchild.colorloop) != len(BB.colorloop)
    _p1 = p1(BB)
    _p2 = p1(BBchild)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert all([isinstance(t.glyph, Line) for t in f1.renderers])
    assert all([isinstance(t.glyph, Line) for t in f2.renderers])
    # there are 6 unique colors in _p1 and 3 unique colors in _p2
    assert len(set([r.glyph.line_color for r in f1.renderers])) == 6
    assert len(set([r.glyph.line_color for r in f2.renderers])) == 3

    ### test for line_kw, surface_kw, quiver_kw, stream_kw: they should override
    ### defualt settings.

    p = p2(BB, line_kw=dict(line_color="red"))
    assert len(p.series) == 2
    f = p.fig
    assert isinstance(f, Figure)
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, Line)
    assert f.legend[0].items[0].label["value"] == "sin(x)"
    assert f.renderers[0].glyph.line_color == "red"
    assert isinstance(f.renderers[1].glyph, Line)
    assert f.legend[0].items[1].label["value"] == "cos(x)"
    assert f.renderers[1].glyph.line_color == "red"
    assert f.legend[0].visible == True

    p = p3(BB, line_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, MultiLine)
    assert f.renderers[0].glyph.line_color == "red"
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "(cos(x), sin(x))"
    assert f.toolbar.tools[-2].tooltips == [('x', '$x'), ('y', '$y'), ("u", "@us")]

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError, lambda: p4(BB, line_kw=dict(line_color="red")))
    raises(
        NotImplementedError,
        lambda: p5(BB, surface_kw=dict(colorscale=[[0, "cyan"], [1, "cyan"]])),
    )

    # Bokeh doesn't use contour_kw dictionary. Nothing to customize yet.
    p = p6(BB, contour_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, Image)
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == str(cos(x ** 2 + y ** 2))

    p = p7a(BB, contour_kw=dict(), quiver_kw=dict(line_color="red"))
    assert len(p.series) == 2
    f = p.fig
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, Image)
    assert isinstance(f.renderers[1].glyph, Segment)
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "Magnitude"
    assert f.renderers[1].glyph.line_color == "red"

    p = p7b(BB, stream_kw=dict(line_color="red"), contour_kw=dict())
    assert len(p.series) == 2
    f = p.fig
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, Image)
    assert isinstance(f.renderers[1].glyph, MultiLine)
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "x + y"
    assert f.renderers[1].glyph.line_color == "red"

    p = p7c(BB, stream_kw=dict(line_color="red"), contour_kw=dict())
    f = p.fig
    assert f.right[0].title == "test"

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError, lambda: p9a(BB, quiver_kw=dict(sizeref=5)))
    raises(
        NotImplementedError,
        lambda: p9b(BB, stream_kw=dict(colorscale=[[0, "red"], [1, "red"]])),
    )

    p = p11a(BB, contour_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, Image)

    p = p11b(BB, contour_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, Image)

    p = p14(BB, line_kw=dict(line_color="red"))
    assert len(p.series) == 2
    f = p.fig
    assert isinstance(f, Figure)
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, Line)
    assert f.legend[0].items[0].label["value"] == "Re(sqrt(x))"
    assert f.renderers[0].glyph.line_color == "red"
    assert isinstance(f.renderers[1].glyph, Line)
    assert f.legend[0].items[1].label["value"] == "Im(sqrt(x))"
    assert f.renderers[1].glyph.line_color == "red"
    assert f.legend[0].visible == True

    p = p15a(BB, line_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, MultiLine)
    assert f.renderers[0].glyph.line_color == "red"
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "Abs(sqrt(x))"

    p = p15b(BB, image_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, ImageRGBA)
    assert f.right[0].title == "Argument"
    assert (f.toolbar.tools[-2].tooltips == [('x', '$x'), ('y', '$y'),
        ("Abs", "@abs"), ("Arg", "@arg")])

    # test markers
    p = p16a(BB)
    assert len(p.series) == 1
    f = p.fig
    assert isinstance(f.renderers[0].glyph, BokehCircle)
    assert f.renderers[0].glyph.line_color != f.renderers[0].glyph.fill_color

    p = p16b(BB)
    assert len(p.series) == 1
    f = p.fig
    assert isinstance(f.renderers[0].glyph, BokehCircle)
    assert f.renderers[0].glyph.line_color == f.renderers[0].glyph.fill_color

    # test plot_piecewise
    p = p17a(BB)
    assert len(p.series) == 4
    colors = set()
    for l in p.fig.renderers:
        colors.add(l.glyph.line_color)
    assert len(colors) == 1

    p = p17b(BB)
    assert len(p.series) == 9
    colors = set()
    for l in p.fig.renderers:
        colors.add(l.glyph.line_color)
    assert len(colors) == 2


    from sympy.geometry import Line as SymPyLine
    p = p18(BB)
    assert len(p.series) == 3
    f = p.fig
    assert len(f.renderers) == 3
    assert all(isinstance(r.glyph, Line) for r in f.renderers)
    assert f.legend[0].items[0].label["value"] == str(SymPyLine((1, 2), (5, 4)))
    assert f.legend[0].items[1].label["value"] == str(Circle((0, 0), 4))
    assert f.legend[0].items[2].label["value"] == str(Polygon((2, 2), 3, n=6))


class KBchild1(KB):
    def _get_mode(self):
        # tells the backend it is running into Jupyter, even if it is not.
        # this is necessary to run these tests.
        return 0


class KBchild2(KBchild1):
    colorloop = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def test_K3DBackend():
    from k3d.objects import Mesh, Line, Vectors
    from matplotlib.colors import ListedColormap

    assert hasattr(KB, "colorloop")
    assert isinstance(KB.colorloop, (list, tuple, ListedColormap))
    assert hasattr(KB, "colormaps")
    assert isinstance(KB.colormaps, (list, tuple))

    series = [UnsupportedSeries()]
    raises(NotImplementedError, lambda: Plot(*series, backend=KBchild2))

    ### Setting custom color loop

    assert len(KBchild1.colorloop) != len(KBchild2.colorloop)
    _p1 = p13(KBchild1)
    _p2 = p13(KBchild2)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert all([isinstance(t, Mesh) for t in f1.objects])
    assert all([isinstance(t, Mesh) for t in f2.objects])
    # there are 6 unique colors in _p1 and 3 unique colors in _p2
    assert len(set([o.color for o in f1.objects])) == 6
    assert len(set([o.color for o in f2.objects])) == 3

    # ### test for line_kw, surface_kw, quiver_kw, stream_kw: they should override
    # ### defualt settings.

    # K3D doesn't support 2D plots
    raises(NotImplementedError, lambda: p2(KBchild1, line_kw=dict(line_color="red")))
    raises(NotImplementedError, lambda: p3(KBchild1, line_kw=dict(line_color="red")))

    p = p4(KBchild1, line_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], Line)
    assert f.objects[0].color == 16711680
    assert f.objects[0].name is None

    # use_cm=False will force to apply a default solid color to the mesh.
    # Here, I override that solid color with a custom color.
    p = p5(KBchild1, surface_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], Mesh)
    assert f.objects[0].color == 16711680
    assert f.objects[0].name is None

    # K3D doesn't support 2D plots
    raises(NotImplementedError, lambda: p6(KBchild1, contour_kw=dict()))
    raises(
        NotImplementedError, lambda: p7a(KBchild1, quiver_kw=dict(), contour_kw=dict())
    )
    raises(
        NotImplementedError, lambda: p7b(KBchild1, stream_kw=dict(), contour_kw=dict())
    )

    p = p9a(KBchild1, quiver_kw=dict(scale=0.5, color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], Vectors)
    assert all([c == 16711680 for c in f.objects[0].colors])

    p = p9b(KBchild1, stream_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], Line)
    assert f.objects[0].color == 16711680

    # test different combinations for streamlines: it should not raise errors
    p = p9b(KBchild1, stream_kw=dict(starts=True))
    p = p9b(KBchild1, stream_kw=dict(starts={
        "x": np.linspace(-5, 5, 10),
        "y": np.linspace(-4, 4, 10),
        "z": np.linspace(-3, 3, 10),
    }))

    # other keywords: it should not raise errors
    p = p9b(MB, stream_kw=dict(), kwargs=dict(use_cm=False))

    # K3D doesn't support 2D plots
    raises(NotImplementedError, lambda: p11a(KBchild1, contour_kw=dict()))
    raises(NotImplementedError, lambda: p11b(KBchild1, contour_kw=dict()))
    raises(NotImplementedError, lambda: p14(KBchild1, line_kw=dict()))
    raises(NotImplementedError, lambda: p15a(KBchild1, line_kw=dict()))
    raises(NotImplementedError, lambda: p15b(KBchild1, image_kw=dict()))

    p = p15c(KBchild1, surface_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], Mesh)
    assert f.objects[0].name is None

def test_save():
    x, y, z = symbols("x:z")

    # There are already plenty of tests for saving Matplotlib plots in
    # test_plot.py. Here, I'm going to test that:
    # 1. the save method accepts keyword arguments.
    # 2. Bokeh and Plotly should not be able to save static pictures because
    #    by default they need additional libraries. See the documentation of
    #    the save methods for each backends to see what is required.
    #    Hence, if in our system those libraries are installed, tests will
    #    fail!

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        p = plot(sin(x), cos(x), backend=MB)
        filename = "test_mpl_save_1.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(sin(x), cos(x), backend=MB)
        filename = "test_mpl_save_2.pdf"
        p.save(os.path.join(tmpdir, filename), dpi=150)
        p.close()

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=MB)
        filename = "test_mpl_save_3.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=MB)
        filename = "test_mpl_save_4.pdf"
        p.save(os.path.join(tmpdir, filename), dpi=150)
        p.close()

        # Bokeh requires additional libraries to save static pictures.
        # Raise an error because their are not installed.
        p = plot(sin(x), cos(x), backend=BB, show=False)
        filename = "test_bokeh_save_1.png"
        raises(RuntimeError, lambda: p.save(os.path.join(tmpdir, filename)))

        p = plot(sin(x), cos(x), backend=BB, show=False)
        filename = "test_bokeh_save_2.svg"
        raises(RuntimeError, lambda: p.save(os.path.join(tmpdir, filename)))

        p = plot(sin(x), cos(x), backend=BB, show=False)
        filename = "test_bokeh_save_3.html"
        p.save(os.path.join(tmpdir, filename))

        from bokeh.resources import INLINE
        p = plot(sin(x), cos(x), backend=BB, show=False)
        filename = "test_bokeh_save_4.html"
        p.save(os.path.join(tmpdir, filename), resources=INLINE)

        # Plotly requires additional libraries to save static pictures.
        # Raise an error because their are not installed.
        p = plot(sin(x), cos(x), backend=PB, show=False)
        filename = "test_plotly_save_1.png"
        raises(ValueError, lambda: p.save(os.path.join(tmpdir, filename)))

        p = plot(sin(x), cos(x), backend=PB, show=False)
        filename = "test_plotly_save_2.svg"
        raises(ValueError, lambda: p.save(os.path.join(tmpdir, filename)))

        p = plot(sin(x), cos(x), backend=PB, show=False)
        filename = "test_plotly_save_3.html"
        p.save(os.path.join(tmpdir, filename))

        p = plot(sin(x), cos(x), backend=PB, show=False)
        filename = "test_plotly_save_4.html"
        p.save(os.path.join(tmpdir, filename), include_plotlyjs="cdn")

        # K3D-Jupyter: use KBchild1 in order to run tests.
        # NOTE: K3D is designed in such a way that the plots need to be shown on
        # the screen before saving them. It is not possible to show the on the
        # screen during tests, hence we are only going to test that it proceeds
        # smoothtly or it raises errors when wrong options are given
        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1)
        filename = "test_k3d_save_1.png"
        p.save(os.path.join(tmpdir, filename))

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1)
        filename = "test_k3d_save_2.jpg"
        raises(ValueError, lambda: p.save(os.path.join(tmpdir, filename)))

        # unexpected keyword argument
        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1)
        filename = "test_k3d_save_3.jpg"
        raises(ValueError, lambda: p.save(os.path.join(tmpdir, filename),
            parameter=True))

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1)
        filename = "test_k3d_save_4.html"
        p.save(os.path.join(tmpdir, filename))

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1)
        filename = "test_k3d_save_4.html"
        p.save(os.path.join(tmpdir, filename), include_js=True)

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1)
        filename = "test_k3d_save_4.html"
        raises(TypeError, lambda: p.save(os.path.join(tmpdir, filename),
            include_js=True, parameter=True))


def test_vectors_update_interactive():
    a, b, c, x, y, z = symbols("a:c, x:z")

    # Some backends do not support streamlines with iplot. Test that the
    # backends raise error.

    def func(B):
        params = { a: 1, b: 2, c: 3 }
        s = InteractiveSeries(
            [a * z, b * y, c * x],
            [(x, -5, 5), (y, -5, 5), (z, -5, 5)],
            "test",
            params = params,
            streamlines = True,
            n1 = 10, n2 = 10, n3 = 10
        )
        p = B(s)
        raises(NotImplementedError, lambda : p._update_interactive(params))

    func(KBchild1)
    func(PB)
    func(MB)
