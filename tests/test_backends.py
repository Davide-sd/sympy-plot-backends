from pytest import raises
from spb.backends.base_backend import Plot
from spb.backends.matplotlib import MB, unset_show
from spb.backends.bokeh import BB
from spb.backends.plotly import PB
from spb.backends.k3d import KB
from spb.series import (
    BaseSeries, InteractiveSeries, LineOver1DRangeSeries,
    SurfaceOver2DRangeSeries
)
from spb import (
    plot, plot3d, plot_contour, plot_implicit,
    plot_parametric, plot3d_parametric_line,
    plot_vector, plot_complex, plot_geometry, plot_real_imag,
    plot_list, plot_piecewise
)
from sympy import (
    symbols, cos, sin, Matrix, pi, sqrt, I, Heaviside, Piecewise, Eq, log
)
from sympy.geometry import Line, Circle, Polygon
from sympy.external import import_module
from tempfile import TemporaryDirectory
import os

np = import_module('numpy', catch=(RuntimeError,))
matplotlib = import_module(
    'matplotlib',
    import_kwargs={'fromlist':['pyplot', 'axes', 'cm', 'collections', 'colors', 'quiver']},
    min_module_version='1.1.0',
    catch=(RuntimeError,))
plt = matplotlib.pyplot
mpl_toolkits = import_module(
    'mpl_toolkits', # noqa
    import_kwargs={'fromlist': ['mplot3d']},
    catch=(RuntimeError,))
plotly = import_module(
    'plotly',
    import_kwargs={'fromlist': ['graph_objects', 'figure_factory']},
    min_module_version='5.0.0',
    catch=(RuntimeError,))
go = plotly.graph_objects
k3d = import_module(
    'k3d',
    import_kwargs={'fromlist': ['plot', 'objects']},
    min_module_version='2.9.7',
    catch=(RuntimeError,))
bokeh = import_module(
    'bokeh',
    import_kwargs={'fromlist': ['models', 'resources', 'plotting']},
    min_module_version='2.3.0',
    catch=(RuntimeError,))
unset_show()

# NOTE
# Here, let's test that each backend:
#
# 1. receives the correct number of data series.
# 2. raises the necessary errors.
# 3. correctly use the common keyword arguments to customize the plot.
# 4. shows the expected labels.
#
# This should be a good starting point to provide a common user experience
# between different backends.
#
# If your issue is related to the generation of numerical data from a
# particular data series, consider adding tests to test_series.py.
# If your issue is related to the processing and generation of *Series
# objects, consider adding tests to test_functions.py.
# If your issue il related to the preprocessing and generation of a
# Vector series or a Complex Series, consider adding tests to
# test_build_series.
#

# NOTE
# While BB, PB, KB creates the figure at instantiation, MB creates the figure
# once the `show()` method is called. All backends do not populate the figure
# at instantiation. Numerical data is added only when `show()` or `fig` is
# called.
# In the following tests, we will use `show=False`, hence the `show()` method
# won't be executed. To add numerical data to the plots we either call `fig`
# or `process_series()`.


class UnsupportedSeries(BaseSeries):
    pass

class MBchild(MB):
    colorloop = ["red", "green", "blue"]

class PBchild(PB):
    colorloop = ["red", "green", "blue"]

class BBchild(BB):
    colorloop = ["red", "green", "blue"]

class KBchild1(KB):
    def _get_mode(self):
        # tells the backend it is running into Jupyter, even if it is not.
        # this is necessary to run these tests.
        return 0

class KBchild2(KBchild1):
    colorloop = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


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
    p = Plot(s, backend=KBchild1, show=False)
    assert isinstance(p, KBchild1)
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
        lambda: Plot(*series, backend=KBchild2).process_series())


def test_colorloop_colormaps():
    # verify that backends exposes important class attributes enabling
    # automatic coloring

    assert hasattr(MB, "colorloop")
    assert isinstance(MB.colorloop, (list, tuple))
    assert hasattr(MB, "colormaps")
    assert isinstance(MB.colormaps, (list, tuple))

    assert hasattr(PB, "colorloop")
    assert isinstance(PB.colorloop, (list, tuple))
    assert hasattr(PB, "colormaps")
    assert isinstance(PB.colormaps, (list, tuple))
    assert hasattr(PB, "quivers_colors")
    assert isinstance(PB.quivers_colors, (list, tuple))

    assert hasattr(BB, "colorloop")
    assert isinstance(BB.colorloop, (list, tuple))
    assert hasattr(BB, "colormaps")
    assert isinstance(BB.colormaps, (list, tuple))

    assert hasattr(KB, "colorloop")
    assert isinstance(KB.colorloop, (list, tuple))
    assert hasattr(KB, "colormaps")
    assert isinstance(KB.colormaps, (list, tuple))


def test_custom_colorloop():
    # verify that it is possible to modify the backend's class attributes
    # in order to change custom coloring

    x, y = symbols("x, y")

    _plot = lambda B: plot(
        sin(x), cos(x), sin(x / 2), cos(x / 2), 2 * sin(x), 2 * cos(x),
        backend=B, show=False)

    assert len(MBchild.colorloop) != len(MB.colorloop)
    _p1 = _plot(MB)
    _p2 = _plot(MBchild)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert len(f1.axes[0].lines) == 6
    assert len(f2.axes[0].lines) == 6
    # there are 6 unique colors in _p1 and 3 unique colors in _p2
    assert len(set([l.get_color() for l in f1.axes[0].lines])) == 6
    assert len(set([l.get_color() for l in f2.axes[0].lines])) == 3

    assert len(PBchild.colorloop) != len(PB.colorloop)
    _p1 = _plot(PB)
    _p2 = _plot(PBchild)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert all([isinstance(t, go.Scatter) for t in f1.data])
    assert all([isinstance(t, go.Scatter) for t in f2.data])
    # there are 6 unique colors in _p1 and 3 unique colors in _p2
    assert len(set([d["line"]["color"] for d in f1.data])) == 6
    assert len(set([d["line"]["color"] for d in f2.data])) == 3

    assert len(BBchild.colorloop) != len(BB.colorloop)
    _p1 = _plot(BB)
    _p2 = _plot(BBchild)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert all([isinstance(t.glyph, bokeh.models.glyphs.Line) for t in f1.renderers])
    assert all([isinstance(t.glyph, bokeh.models.glyphs.Line) for t in f2.renderers])
    # there are 6 unique colors in _p1 and 3 unique colors in _p2
    assert len(set([r.glyph.line_color for r in f1.renderers])) == 6
    assert len(set([r.glyph.line_color for r in f2.renderers])) == 3

    _plot3d = lambda B: plot3d(
        (cos(x ** 2 + y ** 2), (x, -3, -2), (y, -3, 3)),
        (cos(x ** 2 + y ** 2), (x, -2, -1), (y, -3, 3)),
        (cos(x ** 2 + y ** 2), (x, -1, 0), (y, -3, 3)),
        (cos(x ** 2 + y ** 2), (x, 0, 1), (y, -3, 3)),
        (cos(x ** 2 + y ** 2), (x, 1, 2), (y, -3, 3)),
        (cos(x ** 2 + y ** 2), (x, 2, 3), (y, -3, 3)),
        n1 = 5, n2 = 5,
        backend=B,
        use_cm=False,
        show=False)

    assert len(KBchild1.colorloop) != len(KBchild2.colorloop)
    _p1 = _plot3d(KBchild1)
    _p2 = _plot3d(KBchild2)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert all([isinstance(t, k3d.objects.Mesh) for t in f1.objects])
    assert all([isinstance(t, k3d.objects.Mesh) for t in f2.objects])
    # there are 6 unique colors in _p1 and 3 unique colors in _p2
    assert len(set([o.color for o in f1.objects])) == 6
    assert len(set([o.color for o in f2.objects])) == 3


def test_bokeh_tools():
    # verify tools and tooltips on empty Bokeh figure (populated figure
    # might have different tooltips, tested later on)

    f = plot(backend=BB, show=False).fig
    assert len(f.toolbar.tools) == 6
    assert isinstance(f.toolbar.tools[0], bokeh.models.PanTool)
    assert isinstance(f.toolbar.tools[1], bokeh.models.WheelZoomTool)
    assert isinstance(f.toolbar.tools[2], bokeh.models.BoxZoomTool)
    assert isinstance(f.toolbar.tools[3], bokeh.models.ResetTool)
    assert isinstance(f.toolbar.tools[4], bokeh.models.HoverTool)
    assert isinstance(f.toolbar.tools[5], bokeh.models.SaveTool)
    assert f.toolbar.tools[4].tooltips == [('x', '$x'), ('y', '$y')]


def test_MatplotlibBackend():
    # verify a few important things to assure the correct behaviour

    # `_handle` is needed in order to correctly update the data with iplot
    x, y = symbols("x, y")
    p = plot3d(cos(x**2 + y**2), backend=MB, show=False, n1=10, n2=10)
    p.process_series()
    assert hasattr(p, "_handles") and isinstance(p._handles, dict)
    assert len(p._handles) == 1
    assert isinstance(p._handles[0], (tuple, list))
    assert "cmap" in p._handles[0][1].keys()


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
    assert p4.legend is True
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
    assert isinstance(p3.fig.axes[0].collections[-1], matplotlib.quiver.Quiver)
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
    # keyword arguments are used.
    # NOTE: the output plot is of type MB
    p1 = plot(sin(x), backend=MB, line_kw=dict(linestyle=":"), show=False)
    p2 = plot(cos(x), backend=PB, line_kw=dict(line_dash="dash"), show=False)
    raises(AttributeError, lambda: (p1 + p2).process_series())


def test_plot():
    # verify that the backends produce the expected results when `plot()`
    # is called and `line_kw` overrides the default line settings

    x = symbols("x")

    _plot = lambda B, line_kw: plot(
        sin(x), cos(x), line_kw=line_kw, backend=B, show=False, legend=True)

    p = _plot(MB, line_kw=dict(color="red"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.get_lines()) == 2
    assert ax.get_lines()[0].get_label() == "sin(x)"
    assert ax.get_lines()[0].get_color() == "red"
    assert ax.get_lines()[1].get_label() == "cos(x)"
    assert ax.get_lines()[1].get_color() == "red"
    p.close()

    p = _plot(PB, line_kw=dict(line_color="red"))
    assert len(p.series) == 2
    f = p.fig
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "sin(x)"
    assert f.data[0]["line"]["color"] == "red"
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[1]["name"] == "cos(x)"
    assert f.data[1]["line"]["color"] == "red"
    assert f.layout["showlegend"] is True

    p = _plot(BB, line_kw=dict(line_color="red"))
    assert len(p.series) == 2
    f = p.fig
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Line)
    assert f.legend[0].items[0].label["value"] == "sin(x)"
    assert f.renderers[0].glyph.line_color == "red"
    assert isinstance(f.renderers[1].glyph, bokeh.models.glyphs.Line)
    assert f.legend[0].items[1].label["value"] == "cos(x)"
    assert f.renderers[1].glyph.line_color == "red"
    assert f.legend[0].visible is True

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot(KBchild1,
            line_kw=dict(line_color="red")).process_series())


def test_plot_parametric():
    # verify that the backends produce the expected results when
    # `plot_parametric()` is called and `line_kw` overrides the default
    # line settings

    x = symbols("x")

    _plot_parametric = lambda B, line_kw: plot_parametric(
        cos(x), sin(x), (x, 0, 1.5 * pi), backend=B,
        show=False, line_kw=line_kw
    )

    p = _plot_parametric(MB, line_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    # parametric plot with use_cm=True -> LineCollection
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == "(cos(x), sin(x))"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = _plot_parametric(PB, line_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "(cos(x), sin(x))"
    assert f.data[0]["line"]["color"] == "red"
    assert f.data[0]["marker"]["colorbar"]["title"]["text"] == "(cos(x), sin(x))"

    p = _plot_parametric(BB, line_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.MultiLine)
    assert f.renderers[0].glyph.line_color == "red"
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "(cos(x), sin(x))"
    assert f.toolbar.tools[-2].tooltips == [('x', '$x'), ('y', '$y'), ("u", "@us")]

    raises(
        NotImplementedError,
        lambda: _plot_parametric(KBchild1,
            line_kw=dict(line_color="red")).process_series())


def test_plot3d_parametric_line():
    # verify that the backends produce the expected results when
    # `plot3d_parametric_line()` is called and `line_kw` overrides the
    # default line settings

    x = symbols("x")

    _plot3d_parametric_line = lambda B, line_kw: plot3d_parametric_line(
        cos(x), sin(x), x, (x, -pi, pi), backend=B,
        show=False, line_kw=line_kw
    )

    p = _plot3d_parametric_line(MB, line_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert f.axes[1].get_ylabel() == "(cos(x), sin(x), x)"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = _plot3d_parametric_line(PB, line_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter3d)
    assert f.data[0]["line"]["color"] == "red"
    assert f.data[0]["name"] == "(cos(x), sin(x), x)"
    assert f.data[0]["line"]["colorbar"]["title"]["text"] == "(cos(x), sin(x), x)"

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError, lambda: _plot3d_parametric_line(BB,
        line_kw=dict(line_color="red")).process_series())

    p = _plot3d_parametric_line(KBchild1, line_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Line)
    assert f.objects[0].color == 16711680
    assert f.objects[0].name is None


def test_plot3d():
    # verify that the backends produce the expected results when
    # `plot3d()` is called and `surface_kw` overrides the default surface
    # settings

    x, y = symbols("x, y")

    _plot3d = lambda B, surface_kw: plot3d(
        cos(x ** 2 + y ** 2),
        (x, -3, 3),
        (y, -3, 3),
        n=20,
        use_cm=False,
        backend=B,
        show=False,
        surface_kw=surface_kw,
    )

    # use_cm=False will force to apply a default solid color to the mesh.
    # Here, I override that solid color with a custom color.
    p = _plot3d(MB, surface_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Poly3DCollection)
    # TODO: apparently, without showing the plot, the colors are not applied
    # to a Poly3DCollection...
    p.close()

    p = _plot3d(PB, surface_kw=dict(colorscale=[[0, "cyan"], [1, "cyan"]]))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Surface)
    assert f.data[0]["name"] == "cos(x**2 + y**2)"
    assert not f.data[0]["showscale"]
    assert f.data[0]["colorscale"] == ((0, "cyan"), (1, "cyan"))
    assert not f.layout["showlegend"]
    assert f.data[0]["colorbar"]["title"]["text"] == "cos(x**2 + y**2)"

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: _plot3d(BB, surface_kw=dict(
            colorscale=[[0, "cyan"], [1, "cyan"]])).process_series())

    p = _plot3d(KBchild1, surface_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Mesh)
    assert f.objects[0].color == 16711680
    assert f.objects[0].name is None


def test_plot_contour():
    # verify that the backends produce the expected results when
    # `plot_contour()` is called and `contour_kw` overrides the default
    # surface settings

    x, y = symbols("x, y")

    _plot_contour = lambda B, contour_kw: plot_contour(
        cos(x ** 2 + y ** 2),
        (x, -3, 3),
        (y, -3, 3),
        n=20,
        backend=B,
        show=False,
        contour_kw=contour_kw,
    )

    p = _plot_contour(MB, contour_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert f.axes[1].get_ylabel() == str(cos(x ** 2 + y ** 2))
    # TODO: how to retrieve the colormap from a contour series?????
    p.close()

    p = _plot_contour(PB, contour_kw=dict(contours=dict(coloring="lines")))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Contour)
    assert f.data[0]["contours"]["coloring"] == "lines"
    assert f.data[0]["colorbar"]["title"]["text"] == str(cos(x ** 2 + y ** 2))

    # Bokeh doesn't use contour_kw dictionary. Nothing to customize yet.
    p = _plot_contour(BB, contour_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Image)
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == str(cos(x ** 2 + y ** 2))

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_contour(KBchild1,
            contour_kw=dict()).process_series())


def test_plot_vector_2d_quivers():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`quiver_kw` overrides the
    # default settings

    x, y = symbols("x, y")

    _plot_vector = lambda B, contour_kw, quiver_kw: plot_vector(
        Matrix([x, y]),
        (x, -5, 5),
        (y, -4, 4),
        backend=B,
        show=False,
        quiver_kw=quiver_kw,
        contour_kw=contour_kw,
    )

    p = _plot_vector(MB, quiver_kw=dict(color="red"),
        contour_kw=dict(cmap="jet"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert isinstance(ax.collections[-1], matplotlib.quiver.Quiver)
    assert f.axes[1].get_ylabel() == "Magnitude"
    # TODO: how to retrieve the colormap from a contour series?????
    p.close()

    p = _plot_vector(
        PB,
        quiver_kw=dict(line_color="red"),
        contour_kw=dict(contours=dict(coloring="lines")))
    assert len(p.series) == 2
    f = p.fig
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Contour)
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[0]["contours"]["coloring"] == "lines"
    assert f.data[0]["colorbar"]["title"]["text"] == "Magnitude"
    assert f.data[1]["line"]["color"] == "red"

    p = _plot_vector(BB, contour_kw=dict(), quiver_kw=dict(line_color="red"))
    assert len(p.series) == 2
    f = p.fig
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Image)
    assert isinstance(f.renderers[1].glyph, bokeh.models.glyphs.Segment)
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "Magnitude"
    assert f.renderers[1].glyph.line_color == "red"

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_vector(KBchild1, quiver_kw=dict(),
            contour_kw=dict()).process_series())


def test_plot_vector_2d_streamlines_custom_scalar_field():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`stream_kw` overrides the
    # default settings

    x, y = symbols("x, y")

    _plot_vector = lambda B, stream_kw, contour_kw: plot_vector(
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

    p = _plot_vector(MB, stream_kw=dict(color="red"),
        contour_kw=dict(cmap="jet"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert isinstance(ax.collections[-1], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == "x + y"
    assert all(*(ax.collections[-1].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = _plot_vector(PB, stream_kw=dict(line_color="red"),
        contour_kw=dict(contours=dict(coloring="lines")))
    assert len(p.series) == 2
    f = p.fig
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Contour)
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[0]["contours"]["coloring"] == "lines"
    assert f.data[0]["colorbar"]["title"]["text"] == "x + y"
    assert f.data[1]["line"]["color"] == "red"

    p = _plot_vector(BB, stream_kw=dict(line_color="red"), contour_kw=dict())
    assert len(p.series) == 2
    f = p.fig
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Image)
    assert isinstance(f.renderers[1].glyph, bokeh.models.glyphs.MultiLine)
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "x + y"
    assert f.renderers[1].glyph.line_color == "red"

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_vector(KBchild1, stream_kw=dict(),
            contour_kw=dict()).process_series())


def test_plot_vector_2d_streamlines_custom_scalar_field_custom_label():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`stream_kw` overrides the
    # default settings

    x, y = symbols("x, y")

    _plot_vector = lambda B, stream_kw, contour_kw: plot_vector(
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

    p = _plot_vector(MB, stream_kw=dict(color="red"),
        contour_kw=dict(cmap="jet"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert isinstance(ax.collections[-1], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == "test"
    assert all(*(ax.collections[-1].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = _plot_vector(PB, stream_kw=dict(line_color="red"),
        contour_kw=dict(contours=dict(coloring="lines")))
    f = p.fig
    assert f.data[0]["colorbar"]["title"]["text"] == "test"

    p = _plot_vector(BB, stream_kw=dict(line_color="red"), contour_kw=dict())
    f = p.fig
    assert f.right[0].title == "test"

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_vector(KBchild1, stream_kw=dict(),
            contour_kw=dict()).process_series())


def test_plot_vector_2d_matplotlib():
    # verify that when scalar=False, quivers/streamlines comes together with
    # a colorbar

    x, y = symbols("x, y")
    _plot_vector_1 = lambda scalar, streamlines, use_cm=True: plot_vector(
        Matrix([x, y]),
        (x, -5, 5),
        (y, -4, 4),
        backend=MB,
        scalar=scalar,
        streamlines=streamlines,
        use_cm=use_cm,
        show=False)

    # contours + quivers: 1 colorbar for the contours
    p = _plot_vector_1(True, False)
    assert len(p.series) == 2
    assert len(p.fig.axes) == 2
    assert len(p.fig.axes[0].collections) > 1

    # contours + streamlines: 1 colorbar for the contours
    p = _plot_vector_1(True, True)
    assert len(p.series) == 2
    assert len(p.fig.axes) == 2
    assert len(p.fig.axes[0].collections) > 1

    # only quivers: 1 colorbar for the quivers
    p = _plot_vector_1(False, False)
    assert len(p.series) == 1
    assert len(p.fig.axes) == 2
    assert len(p.fig.axes[0].collections) == 1

    # only streamlines: 1 colorbar for the streamlines
    p = _plot_vector_1(False, False)
    assert len(p.series) == 1
    assert len(p.fig.axes) == 2
    assert len(p.fig.axes[0].collections) == 1

    # only quivers with solid color
    p = _plot_vector_1(False, False, False)
    assert len(p.series) == 1
    assert len(p.fig.axes) == 1
    assert len(p.fig.axes[0].collections) == 1

    # only streamlines with solid color
    p = _plot_vector_1(False, False, False)
    assert len(p.series) == 1
    assert len(p.fig.axes) == 1
    assert len(p.fig.axes[0].collections) == 1


def test_plot_vector_3d_quivers():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `quiver_kw` overrides the
    # default settings

    x, y, z = symbols("x, y, z")

    _plot_vector = lambda B, quiver_kw, **kwargs: plot_vector(
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

    p = _plot_vector(MB, quiver_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert ax.collections[0].cmap.name == "jet"
    p.close()

    p = _plot_vector(MB, quiver_kw=dict(cmap=None, color="red"), use_cm=False)
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert np.allclose(ax.collections[0].get_color(), np.array([[1., 0., 0., 1.]]))
    p.close()

    p = _plot_vector(PB, quiver_kw=dict(sizeref=5))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Cone)
    assert f.data[0]["sizeref"] == 5
    assert f.data[0]["colorbar"]["title"]["text"] == str(Matrix([z, y, x]))

    cs1 = f.data[0]["colorscale"]

    p = _plot_vector(PB, quiver_kw=dict(colorscale="reds"))
    f = p.fig
    cs2 = f.data[0]["colorscale"]
    assert len(cs1) != len(cs2)

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError,
        lambda: _plot_vector(BB, quiver_kw=dict(sizeref=5)).process_series())

    p = _plot_vector(KBchild1, quiver_kw=dict(scale=0.5, color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Vectors)
    assert all([c == 16711680 for c in f.objects[0].colors])


def test_plot_vector_3d_streamlines():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `stream_kw` overrides the
    # default settings

    x, y, z = symbols("x, y, z")

    _plot_vector = lambda B, stream_kw, kwargs=dict(): plot_vector(
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

    p = _plot_vector(MB, stream_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert f.axes[1].get_ylabel() == "Matrix([[z], [y], [x]])"
    p.close()

    # test different combinations for streamlines: it should not raise errors
    p = _plot_vector(MB, stream_kw=dict(starts=True))
    p = _plot_vector(MB, stream_kw=dict(starts={
        "x": np.linspace(-5, 5, 10),
        "y": np.linspace(-4, 4, 10),
        "z": np.linspace(-3, 3, 10),
    }))
    p.close()

    # other keywords: it should not raise errors
    p = _plot_vector(MB, stream_kw=dict(), kwargs=dict(use_cm=False))
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_color() == '#1f77b4'
    p.close()

    p = _plot_vector(PB, stream_kw=dict(colorscale=[[0, "red"], [1, "red"]]))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Streamtube)
    assert f.data[0]["colorscale"] == ((0, "red"), (1, "red"))
    assert f.data[0]["colorbar"]["title"]["text"] == str(Matrix([z, y, x]))

    # test different combinations for streamlines: it should not raise errors
    p = _plot_vector(PB, stream_kw=dict(starts=True))
    p = _plot_vector(PB, stream_kw=dict(starts={
        "x": np.linspace(-5, 5, 10),
        "y": np.linspace(-4, 4, 10),
        "z": np.linspace(-3, 3, 10),
    }))

    # other keywords: it should not raise errors
    p = _plot_vector(MB, stream_kw=dict(), kwargs=dict(use_cm=False))

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: _plot_vector(BB, stream_kw=dict(
            colorscale=[[0, "red"], [1, "red"]])).process_series())

    p = _plot_vector(KBchild1, stream_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Line)
    assert f.objects[0].color == 16711680

    # test different combinations for streamlines: it should not raise errors
    p = _plot_vector(KBchild1, stream_kw=dict(starts=True))
    p = _plot_vector(KBchild1, stream_kw=dict(starts={
        "x": np.linspace(-5, 5, 10),
        "y": np.linspace(-4, 4, 10),
        "z": np.linspace(-3, 3, 10),
    }))

    # other keywords: it should not raise errors
    p = _plot_vector(MB, stream_kw=dict(), kwargs=dict(use_cm=False))


def test_plot_implicit_adaptive_true():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True`

    x, y = symbols("x, y")

    _plot_implicit = lambda B, contour_kw: plot_implicit(
        x > y, (x, -5, 5), (y, -4, 4), backend=B, show=False,
        adaptive=True, contour_kw=contour_kw
    )

    p = _plot_implicit(MB, contour_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 0
    assert len(ax.patches) == 1
    p.close()

    # PlotlyBackend doesn't support 2D plots
    raises(NotImplementedError,
        lambda: _plot_implicit(PB, contour_kw=dict()).process_series())

    # BokehBackend doesn't support 2D plots
    raises(NotImplementedError,
        lambda: _plot_implicit(BB, contour_kw=dict()).process_series())

    # K3D doesn't support 2D plots
    raises(NotImplementedError,
        lambda: _plot_implicit(KBchild1, contour_kw=dict()).process_series())


def test_plot_implicit_adaptive_false():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True` and `contour_kw`
    # overrides the default settings

    x, y = symbols("x, y")

    _plot_implicit = lambda B, contour_kw: plot_implicit(
        x > y,
        (x, -5, 5),
        (y, -4, 4),
        n=20,
        backend=B,
        adaptive=False,
        show=False,
        contour_kw=contour_kw,
    )

    p = _plot_implicit(MB, contour_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    # TODO: how to retrieve the colormap from a contour series?????
    p.close()

    # PlotlyBackend doesn't support 2D plots
    raises(NotImplementedError,
        lambda: _plot_implicit(PB, contour_kw=dict()).process_series())

    # BokehBackend doesn't support 2D plots
    raises(NotImplementedError,
        lambda: _plot_implicit(BB, contour_kw=dict()).process_series())

    # K3D doesn't support 2D plots
    raises(NotImplementedError,
        lambda: _plot_implicit(KBchild1, contour_kw=dict()).process_series())


def test_plot_real_imag():
    # verify that the backends produce the expected results when
    # `plot_real_imag()` is called and `line_kw` overrides the default
    # settings

    x = symbols("x")

    _plot_real_imag = lambda B, line_kw: plot_real_imag(
        sqrt(x), (x, -5, 5), backend=B, line_kw=line_kw, show=False
    )

    p = _plot_real_imag(MB, line_kw=dict(color="red"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.get_lines()) == 2
    assert ax.get_lines()[0].get_label() == "Re(sqrt(x))"
    assert ax.get_lines()[0].get_color() == "red"
    assert ax.get_lines()[1].get_label() == "Im(sqrt(x))"
    assert ax.get_lines()[1].get_color() == "red"
    p.close()

    p = _plot_real_imag(PB, line_kw=dict(line_color="red"))
    assert len(p.series) == 2
    f = p.fig
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "Re(sqrt(x))"
    assert f.data[0]["line"]["color"] == "red"
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[1]["name"] == "Im(sqrt(x))"
    assert f.data[1]["line"]["color"] == "red"
    assert f.layout["showlegend"] is True

    p = _plot_real_imag(BB, line_kw=dict(line_color="red"))
    assert len(p.series) == 2
    f = p.fig
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Line)
    assert f.legend[0].items[0].label["value"] == "Re(sqrt(x))"
    assert f.renderers[0].glyph.line_color == "red"
    assert isinstance(f.renderers[1].glyph, bokeh.models.glyphs.Line)
    assert f.legend[0].items[1].label["value"] == "Im(sqrt(x))"
    assert f.renderers[1].glyph.line_color == "red"
    assert f.legend[0].visible is True

    # K3D doesn't support 2D plots
    raises(NotImplementedError,
        lambda: _plot_real_imag(KBchild1, line_kw=dict()).process_series())


def test_plot_complex_1d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `line_kw` overrides the default
    # settings

    x = symbols("x")

    _plot_complex = lambda B, line_kw: plot_complex(
        sqrt(x), (x, -5, 5), backend=B, line_kw=line_kw, show=False
    )

    p = _plot_complex(MB, line_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == "Arg(sqrt(x))"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = _plot_complex(PB, line_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "Arg(sqrt(x))"
    assert f.data[0]["line"]["color"] == "red"
    assert p.fig.data[0]["marker"]["colorbar"]["title"]["text"] == "Arg(sqrt(x))"

    p = _plot_complex(BB, line_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.MultiLine)
    assert f.renderers[0].glyph.line_color == "red"
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "Arg(sqrt(x))"

    # K3D doesn't support 2D plots
    raises(NotImplementedError,
        lambda: _plot_complex(KBchild1, line_kw=dict()).process_series())


def test_plot_complex_2d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `image_kw` overrides the default
    # settings

    x = symbols("x")

    _plot_complex = lambda B, image_kw: plot_complex(
        sqrt(x), (x, -5 - 5 * I, 5 + 5 * I), backend=B, coloring="a",
        image_kw=image_kw, show=False
    )

    p = _plot_complex(MB, image_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.images) == 1
    assert f.axes[1].get_ylabel() == "Argument"
    assert ax.images[0].get_extent() == [-5.0, 5.0, -5.0, 5.0]
    p.close()

    p = _plot_complex(MB, image_kw=dict(extent=[-6, 6, -7, 7]))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.images) == 1
    assert f.axes[1].get_ylabel() == "Argument"
    assert ax.images[0].get_extent() == [-6, 6, -7, 7]
    p.close()

    p = _plot_complex(PB, image_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Image)
    assert f.data[0]["name"] == "sqrt(x)"
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[1]["marker"]["colorbar"]["title"]["text"] == "Argument"

    p = _plot_complex(BB, image_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.ImageRGBA)
    assert f.right[0].title == "Argument"
    assert (f.toolbar.tools[-2].tooltips == [('x', '$x'), ('y', '$y'),
        ("Abs", "@abs"), ("Arg", "@arg")])

    # K3D doesn't support 2D plots
    raises(NotImplementedError,
        lambda: _plot_complex(KBchild1, image_kw=dict()).process_series())


def test_plot_complex_3d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `surface_kw` overrides the default
    # settings

    x = symbols("x")

    _plot_complex = lambda B, surface_kw: plot_complex(
        sqrt(x),
        (x, -5 - 5 * I, 5 + 5 * I),
        backend=B,
        threed=True,
        surface_kw=surface_kw,
        show=False,
        n=10
    )

    p = _plot_complex(MB, surface_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Poly3DCollection)
    assert f.axes[1].get_ylabel() == "Argument"
    # TODO: apparently, without showing the plot, the colors are not applied
    # to a Poly3DCollection...
    p.close()

    p = _plot_complex(PB, surface_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Surface)
    assert f.data[0]["name"] == "sqrt(x)"
    assert f.data[0]["showscale"] is True
    assert f.data[0]["colorbar"]["title"]["text"] == "Argument"

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: _plot_complex(BB, surface_kw=dict()).process_series())

    p = _plot_complex(KBchild1, surface_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Mesh)
    assert f.objects[0].name is None


def test_plot_list_is_filled_false():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=False`

    _plot_list = lambda B: plot_list([1, 2, 3], [1, 2, 3],
        backend=B, is_point=True, is_filled=False, show=False)

    p = _plot_list(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_markeredgecolor() != ax.lines[0].get_markerfacecolor()
    p.close()

    p = _plot_list(PB)
    assert len(p.series) == 1
    f = p.fig
    assert f.data[0]["marker"]["line"]["color"] is not None

    p = _plot_list(BB)
    assert len(p.series) == 1
    f = p.fig
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Circle)
    assert f.renderers[0].glyph.line_color != f.renderers[0].glyph.fill_color

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_list(KBchild1).process_series())


def test_plot_list_is_filled_true():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=True`

    _plot_list = lambda B: plot_list([1, 2, 3], [1, 2, 3],
        backend=B, is_point=True, is_filled=True, show=False)

    p = _plot_list(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_markeredgecolor() == ax.lines[0].get_markerfacecolor()
    p.close()

    p = _plot_list(PB)
    assert len(p.series) == 1
    f = p.fig
    assert f.data[0]["marker"]["line"]["color"] is None

    p = _plot_list(BB)
    assert len(p.series) == 1
    f = p.fig
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Circle)
    assert f.renderers[0].glyph.line_color == f.renderers[0].glyph.fill_color

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_list(KBchild1).process_series())


def test_plot_piecewise_single_series():
    # verify that plot_piecewise plotting 1 piecewise composed of N
    # sub-expressions uses only 1 color

    x = symbols("x")

    _plot_piecewise = lambda B: plot_piecewise(
        Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10),
        backend=B, show=False)

    p = _plot_piecewise(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 4
    colors = set()
    for l in ax.lines:
        colors.add(l.get_color())
    assert len(colors) == 1
    assert not p.legend

    p = _plot_piecewise(PB)
    assert len(p.series) == 4
    colors = set()
    for l in p.fig.data:
        colors.add(l["line"]["color"])
    assert len(colors) == 1
    assert not p.legend

    p = _plot_piecewise(BB)
    assert len(p.series) == 4
    colors = set()
    for l in p.fig.renderers:
        colors.add(l.glyph.line_color)
    assert len(colors) == 1
    assert not p.legend

    # K3D doesn't support 2D plots
    raises(NotImplementedError, lambda: _plot_piecewise(KBchild1))


def test_plot_piecewise_multiple_series():
    # verify that plot_piecewise plotting N piecewise expressions uses
    # only N different colors

    x = symbols("x")

    _plot_piecewise = lambda B: plot_piecewise(
        (Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10)),
        (Piecewise((sin(x), x < 0), (2, Eq(x, 0)), (cos(x), x > 0)), (x, -6, 4)),
        backend=B, show=False)

    p = _plot_piecewise(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 9
    colors = set()
    for l in ax.lines:
        colors.add(l.get_color())
    assert len(colors) == 2

    p = _plot_piecewise(PB)
    assert len(p.series) == 9
    colors = set()
    for l in p.fig.data:
        colors.add(l["line"]["color"])
    assert len(colors) == 2

    p = _plot_piecewise(BB)
    assert len(p.series) == 9
    colors = set()
    for l in p.fig.renderers:
        colors.add(l.glyph.line_color)
    assert len(colors) == 2

    # K3D doesn't support 2D plots
    raises(NotImplementedError, lambda: _plot_piecewise(KBchild1))


def test_plot_geometry_1():
    # verify that the backends produce the expected results when
    # `plot_geometry()` is called
    from sympy.geometry import Line as SymPyLine

    _plot_geometry = lambda B: plot_geometry(
        SymPyLine((1, 2), (5, 4)), Circle((0, 0), 4), Polygon((2, 2), 3, n=6),
        backend=B, show=False, is_filled=False)

    p = _plot_geometry(MB)
    assert len(p.series) == 3
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 3
    assert ax.get_lines()[0].get_label() == str(SymPyLine((1, 2), (5, 4)))
    assert ax.get_lines()[1].get_label() == str(Circle((0, 0), 4))
    assert ax.get_lines()[2].get_label() == str(Polygon((2, 2), 3, n=6))
    p.close()

    p = _plot_geometry(PB)
    assert len(p.series) == 3
    f = p.fig
    assert len(f.data) == 3
    assert f.data[0]["name"] == str(SymPyLine((1, 2), (5, 4)))
    assert f.data[1]["name"] == str(Circle((0, 0), 4))
    assert f.data[2]["name"] == str(Polygon((2, 2), 3, n=6))

    p = _plot_geometry(BB)
    assert len(p.series) == 3
    f = p.fig
    assert len(f.renderers) == 3
    assert all(isinstance(r.glyph, bokeh.models.glyphs.Line) for r in f.renderers)
    assert f.legend[0].items[0].label["value"] == str(SymPyLine((1, 2), (5, 4)))
    assert f.legend[0].items[1].label["value"] == str(Circle((0, 0), 4))
    assert f.legend[0].items[2].label["value"] == str(Polygon((2, 2), 3, n=6))

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_geometry(KBchild1).process_series())

def test_plot_geometry_2():
    # verify that is_filled works correctly
    from sympy.geometry import (
        Line as SymPyLine, Ellipse, Curve, Point2D, Segment, Polygon
    )
    from sympy import Rational

    x = symbols("x")

    _plot_geometry = lambda B, is_filled: plot_geometry(
        Circle(Point2D(0, 0), 5),
        Ellipse(Point2D(-3, 2), hradius=3, eccentricity=Rational(4, 5)),
        Polygon((4, 0), 4, n=5),
        Curve((cos(x), sin(x)), (x, 0, 2 * pi)),
        Segment((-4, -6), (6, 6)),
        Point2D(0, 0), is_filled=is_filled, backend=B, show=False)

    p = _plot_geometry(MB, False)
    assert len(p.fig.axes[0].lines) == 5
    assert len(p.fig.axes[0].collections) == 1
    assert len(p.fig.axes[0].patches) == 0
    p = _plot_geometry(MB, True)
    assert len(p.fig.axes[0].lines) == 2
    assert len(p.fig.axes[0].collections) == 1
    assert len(p.fig.axes[0].patches) == 3

    p = _plot_geometry(PB, False)
    assert len([t["fill"] for t in p.fig.data if t["fill"] is not None]) == 0
    p = _plot_geometry(PB, True)
    assert len([t["fill"] for t in p.fig.data if t["fill"] is not None]) == 3

    p = _plot_geometry(BB, False)
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Line)]) == 4
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.MultiLine)]) == 1
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Circle)]) == 1
    p = _plot_geometry(BB, True)
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Line)]) == 1
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Patch)]) == 3
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.MultiLine)]) == 1
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Circle)]) == 1


def test_save():
    # Verify that:
    # 1. the save method accepts keyword arguments.
    # 2. Bokeh and Plotly should not be able to save static pictures because
    #    by default they need additional libraries. See the documentation of
    #    the save methods for each backends to see what is required.
    #    Hence, if in our system those libraries are installed, tests will
    #    fail!
    x, y, z = symbols("x:z")

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

        p = plot(sin(x), cos(x), backend=BB, show=False)
        filename = "test_bokeh_save_4.html"
        p.save(os.path.join(tmpdir, filename), resources=bokeh.resources.INLINE)

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
        # NOTE: K3D is designed in such a way that the plots need to be shown
        # on the screen before saving them. Since it is not possible to show
        # them on the screen during tests, we are only going to test that it
        # proceeds smoothtly or it raises errors when wrong options are given
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
        params = {a: 1, b: 2, c: 3}
        s = InteractiveSeries(
            [a * z, b * y, c * x],
            [(x, -5, 5), (y, -5, 5), (z, -5, 5)],
            "test",
            params = params,
            streamlines = True,
            n1 = 10, n2 = 10, n3 = 10
        )
        p = B(s)
        raises(NotImplementedError, lambda: p._update_interactive(params))

    func(KBchild1)
    func(PB)
    func(MB)


def test_aspect_ratio_2d_issue_11764():
    # verify that the backends apply the provided aspect ratio.
    # NOTE: read the backend docs to understand which options are available.
    x = symbols("x")

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        show=False, backend=MB)
    assert p.aspect == "auto"
    assert p.fig.axes[0].get_aspect() == "auto"
    p.close()

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        aspect=(1, 1), show=False, backend=MB)
    assert p.aspect == (1, 1)
    assert p.fig.axes[0].get_aspect() == 1
    p.close()

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        aspect="equal", show=False, backend=MB)
    assert p.aspect == "equal"
    assert p.fig.axes[0].get_aspect() == 1
    p.close()

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        show=False, backend=PB)
    assert p.aspect == "auto"
    assert p.fig.layout.yaxis.scaleanchor is None

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        aspect="equal", show=False, backend=PB)
    assert p.aspect == "equal"
    assert p.fig.layout.yaxis.scaleanchor == "x"

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        show=False, backend=BB)
    assert p.aspect == "auto"
    assert not p.fig.match_aspect

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        aspect="equal", show=False, backend=BB)
    assert p.aspect == "equal"
    assert p.fig.match_aspect


def test_aspect_ratio_3d():
    # verify that the backends apply the provided aspect ratio.
    # NOTE:
    # 1. read the backend docs to understand which options are available.
    # 2. K3D doesn't use the `aspect` keyword argument.
    x, y = symbols("x, y")

    p = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        n1=20, n2=20, backend=MB, show=False)
    assert p.aspect == "auto"

    # matplotlib's Axes3D currently only supports the aspect argument 'auto'
    raises(NotImplementedError,
        lambda: plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
            n1=20, n2=20, backend=MB, show=False, aspect=(1, 1)).process_series())

    p = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        n1=20, n2=20, backend=PB, show=False)
    assert p.aspect == "auto"
    assert p.fig.layout.scene.aspectmode == "auto"

    p = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        n1=20, n2=20, backend=PB, show=False, aspect="cube")
    assert p.aspect == "cube"
    assert p.fig.layout.scene.aspectmode == "cube"


def test_plot_size():
    # verify that the keyword `size` is doing it's job
    # NOTE: K3DBackend doesn't support custom size

    x = symbols("x")

    p = plot(sin(x), backend=MB, size=(8, 4), show=False)
    s = p.fig.get_size_inches()
    assert (s[0] == 8) and (s[1] == 4)
    p.close()

    p = plot(sin(x), backend=MB, size=(10, 5), show=False)
    s = p.fig.get_size_inches()
    assert (s[0] == 10) and (s[1] == 5)
    p.close()

    p = plot(sin(x), backend=PB, show=False)
    assert p.fig.layout.width is None
    assert p.fig.layout.height is None

    p = plot(sin(x), backend=PB, size=(800, 400), show=False)
    assert p.fig.layout.width == 800
    assert p.fig.layout.height == 400

    p = plot(sin(x), backend=BB, show=False)
    assert p.fig.sizing_mode == "stretch_width"

    p = plot(sin(x), backend=BB, size=(400, 200), show=False)
    assert p.fig.sizing_mode == "fixed"
    assert (p.fig.width == 400) and (p.fig.height == 200)


def test_plot_scale_lin_log():
    # verify that backends are applying the correct scale to the axes
    # NOTE: none of the 3D libraries currently support log scale.

    x, y = symbols("x, y")

    p = plot(log(x), backend=MB, xscale="linear", yscale="linear", show=False)
    assert p.fig.axes[0].get_xscale() == "linear"
    assert p.fig.axes[0].get_yscale() == "linear"
    p.close()

    p = plot(log(x), backend=MB, xscale="log", yscale="linear", show=False)
    assert p.fig.axes[0].get_xscale() == "log"
    assert p.fig.axes[0].get_yscale() == "linear"
    p.close()

    p = plot(log(x), backend=MB, xscale="linear", yscale="log", show=False)
    assert p.fig.axes[0].get_xscale() == "linear"
    assert p.fig.axes[0].get_yscale() == "log"
    p.close()

    p = plot(log(x), backend=PB, xscale="linear", yscale="linear", show=False)
    assert p.fig.layout["xaxis"]["type"] == "linear"
    assert p.fig.layout["yaxis"]["type"] == "linear"

    p = plot(log(x), backend=PB, xscale="log", yscale="linear", show=False)
    assert p.fig.layout["xaxis"]["type"] == "log"
    assert p.fig.layout["yaxis"]["type"] == "linear"

    p = plot(log(x), backend=PB, xscale="linear", yscale="log", show=False)
    assert p.fig.layout["xaxis"]["type"] == "linear"
    assert p.fig.layout["yaxis"]["type"] == "log"

    p = plot(log(x), backend=BB, xscale="linear", yscale="linear", show=False)
    assert isinstance(p.fig.x_scale, bokeh.models.scales.LinearScale)
    assert isinstance(p.fig.y_scale, bokeh.models.scales.LinearScale)

    p = plot(log(x), backend=BB, xscale="log", yscale="linear", show=False)
    assert isinstance(p.fig.x_scale, bokeh.models.scales.LogScale)
    assert isinstance(p.fig.y_scale, bokeh.models.scales.LinearScale)

    p = plot(log(x), backend=BB, xscale="linear", yscale="log", show=False)
    assert isinstance(p.fig.x_scale, bokeh.models.scales.LinearScale)
    assert isinstance(p.fig.y_scale, bokeh.models.scales.LogScale)
