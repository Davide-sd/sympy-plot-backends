import os
from PIL import Image
from pytest import raises
from spb import *
from spb.backends.base_backend import Plot
from spb.backends.matplotlib import unset_show
from spb.series import (
    BaseSeries, InteractiveSeries, LineOver1DRangeSeries,
    SurfaceOver2DRangeSeries, Parametric3DLineSeries, ParametricSurfaceSeries
)
from sympy import (
    latex, gamma, exp, symbols, Eq, Matrix, pi, I, sin, cos,
    sqrt, log, Heaviside, Piecewise, Line, Circle, Polygon
)
from sympy.external import import_module
from tempfile import TemporaryDirectory


np = import_module('numpy', catch=(RuntimeError,))
matplotlib = import_module(
    'matplotlib',
    import_kwargs={'fromlist':['pyplot', 'axes', 'cm', 'collections', 'colors', 'quiver', 'projections']},
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

try:
    # NOTE: mayavi is extremely difficult to install on any setup
    # (OS + Python version + conda/pip, ...). There was a moment in time
    # where it was easy to install: as a matter of fact, v1.4.0 of this module
    # succedeed Github actions with mayavi installed. However, something
    # changed, and it became extremely difficult to install it. Hence, from
    # v1.5.0 of this module, MayaviBackend will only be tested locally on an
    # Ubuntu system with Python 3.10.
    # Consider removing MayaviBackend when merging this module into SymPy.
    import mayavi
    from mayavi import mlab
    # mayavi = import_module(
    #     'mayavi',
    #     import_kwargs={'fromlist':['mlab', 'core']},
    #     min_module_version='4.8.0',
    #     catch=(RuntimeError,))
    # prevent Mayavi window from opening
    mlab.options.offscreen = True
    is_mayavi_available = True
except:
    mayavi = None

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


class MABchild(MAB):
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

    if mayavi:
        p = Plot(s, backend=MAB, show=False)
        assert isinstance(p, MAB)
        assert isinstance(p.fig, mayavi.core.scene.Scene)


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
    if mayavi:
        raises(
            NotImplementedError,
            lambda: Plot(*series, backend=MAB).process_series())


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

    assert hasattr(MAB, "colorloop")
    assert isinstance(MAB.colorloop, (list, tuple))
    assert hasattr(MAB, "colormaps")
    assert isinstance(MAB.colormaps, (list, tuple))


def test_custom_colorloop():
    # verify that it is possible to modify the backend's class attributes
    # in order to change custom coloring

    x, y = symbols("x, y")

    _plot = lambda B: plot(
        sin(x), cos(x), sin(x / 2), cos(x / 2), 2 * sin(x), 2 * cos(x),
        backend=B, adaptive=False, n=5, show=False)

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

    _plot3d = lambda B, show=False: plot3d(
        (cos(x ** 2 + y ** 2), (x, -3, -2), (y, -3, 3)),
        (cos(x ** 2 + y ** 2), (x, -2, -1), (y, -3, 3)),
        (cos(x ** 2 + y ** 2), (x, -1, 0), (y, -3, 3)),
        (cos(x ** 2 + y ** 2), (x, 0, 1), (y, -3, 3)),
        (cos(x ** 2 + y ** 2), (x, 1, 2), (y, -3, 3)),
        (cos(x ** 2 + y ** 2), (x, 2, 3), (y, -3, 3)),
        n1 = 5, n2 = 5,
        backend=B,
        use_cm=False,
        show=show)

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

    if mayavi:
        # NOTE: Mayavi applies colors only when the plot is shown. Luckily, we
        # turned off the window from opening at the beginning of this module.
        assert len(MAB.colorloop) != len(MABchild.colorloop)
        _p1 = _plot3d(MAB, show=True)
        _p2 = _plot3d(MABchild, show=True)
        assert len(_p1.series) == len(_p2.series)
        f1 = _p1.fig
        f2 = _p2.fig
        # there are 6 unique colors in _p1 and 3 unique colors in _p2
        assert len(set([_p1._handles[k].actor.property.color for k in _p1._handles.keys()])) == 6
        assert len(set([_p2._handles[k].actor.property.color for k in _p2._handles.keys()])) == 3


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
    p = plot3d(cos(x**2 + y**2), backend=MB, show=False, n1=5, n2=5, use_cm=True)
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

    # the choice of the backend dictates the keyword arguments
    # inside rendering_kw
    p1 = plot(sin(x), backend=PB, rendering_kw=dict(line_color='black'),
        xlabel="x1", ylabel="y1", adaptive=False, n=5, show=False)
    p2 = plot(cos(x), backend=PB, rendering_kw=dict(line_dash='dash'),
        xlabel="x2", ylabel="y2", adaptive=False, n=5, show=False)
    p3 = plot(sin(x) * cos(x), backend=PB,
        rendering_kw=dict(line_dash='dot'), adaptive=False, n=5, show=False)
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
        backend=MB, scalar=True, adaptive=False, n1=5, n2=5, show=False)
    p2 = plot(sin(x), (x, -3, 3), backend=MB, adaptive=False, n=5, show=False)
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
    p1 = plot(sin(x), backend=MB, adaptive=False, n=5, show=False)
    p2 = plot(cos(x), backend=PB, adaptive=False, n=5, show=False)
    p3 = p1 + p2
    assert isinstance(p3, MB)

    # summing plots with different backends: fail when backend-specific
    # keyword arguments are used.
    # NOTE: the output plot is of type MB
    p1 = plot(sin(x), backend=MB, rendering_kw=dict(linestyle=":"), show=False)
    p2 = plot(cos(x), backend=PB, rendering_kw=dict(line_dash="dash"), show=False)
    raises(AttributeError, lambda: (p1 + p2).process_series())

    # verify that summing up bokeh plots doesn't raise errors
    p1 = plot(sin(x), (x, -pi, pi), backend=BB, show=False, adaptive=False, n=5)
    p2 = plot(cos(x), (x, -pi, pi), backend=BB, show=False, adaptive=False, n=5)
    p3 = p1 + p2

    # verify that summing up K3D plots doesn't raise errors
    p1 = plot3d(sin(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), backend=KBchild1,
        show=False, adaptive=False, n=5)
    p2 = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), backend=KBchild1,
        show=False, adaptive=False, n=5)
    p3 = p1 + p2


def test_plot():
    # verify that the backends produce the expected results when `plot()`
    # is called and `rendering_kw` overrides the default line settings

    x = symbols("x")

    _plot = lambda B, rendering_kw, use_latex=False: plot(
        sin(x), cos(x), rendering_kw=rendering_kw, backend=B,
        show=False, legend=True, use_latex=use_latex, adaptive=False, n=5)

    p = _plot(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.get_lines()) == 2
    assert ax.get_lines()[0].get_label() == "sin(x)"
    assert ax.get_lines()[0].get_color() == "red"
    assert ax.get_lines()[1].get_label() == "cos(x)"
    assert ax.get_lines()[1].get_color() == "red"
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "f(x)"
    p.close()

    p = _plot(MB, rendering_kw=dict(color="red"), use_latex=True)
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_label() == "$\\sin{\\left(x \\right)}$"
    assert ax.get_xlabel() == "$x$"
    assert ax.get_ylabel() == "$f\\left(x\\right)$"

    p = _plot(PB, rendering_kw=dict(line_color="red"))
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
    # PB separates the data generation from the layout creation. Make sure
    # the layout has been processed
    assert f.layout["xaxis"]["title"]["text"] == "x"
    assert f.layout["yaxis"]["title"]["text"] == "f(x)"

    p = _plot(PB, rendering_kw=dict(line_color="red"), use_latex=True)
    f = p.fig
    assert f.data[0]["name"] == "$\\sin{\\left(x \\right)}$"
    assert f.layout["xaxis"]["title"]["text"] == "$x$"
    assert f.layout["yaxis"]["title"]["text"] == "$f\\left(x\\right)$"

    p = _plot(BB, rendering_kw=dict(line_color="red"))
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

    p = _plot(BB, rendering_kw=dict(line_color="red"), use_latex=True)
    f = p.fig
    assert f.legend[0].items[0].label["value"] == "$\\sin{\\left(x \\right)}$"

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot(KBchild1,
            rendering_kw=dict(line_color="red")).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot(MAB,
                rendering_kw=dict(color=(1, 0, 0))).process_series())


def test_plot_parametric():
    # verify that the backends produce the expected results when
    # `plot_parametric()` is called and `rendering_kw` overrides the default
    # line settings

    x = symbols("x")

    _plot_parametric = lambda B, rendering_kw: plot_parametric(
        cos(x), sin(x), (x, 0, 1.5 * pi), backend=B,
        show=False, rendering_kw=rendering_kw, use_latex=False,
        adaptive=False, n=5
    )

    p = _plot_parametric(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    # parametric plot with use_cm=True -> LineCollection
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == "x"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = _plot_parametric(PB, rendering_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "x"
    assert f.data[0]["line"]["color"] == "red"
    assert f.data[0]["marker"]["colorbar"]["title"]["text"] == "x"

    p = _plot_parametric(BB, rendering_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.MultiLine)
    assert f.renderers[0].glyph.line_color == "red"
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "x"
    assert f.toolbar.tools[-2].tooltips == [('x', '$x'), ('y', '$y'), ("u", "@us")]

    raises(
        NotImplementedError,
        lambda: _plot_parametric(KBchild1,
            rendering_kw=dict(line_color="red")).process_series())

    if mayavi:
        raises(
            NotImplementedError,
            lambda: _plot_parametric(MAB,
                rendering_kw=dict(color=(1, 0, 0))).process_series())


def test_plot3d_parametric_line():
    # verify that the backends produce the expected results when
    # `plot3d_parametric_line()` is called and `rendering_kw` overrides the
    # default line settings

    x = symbols("x")

    def _plot3d_parametric_line(B, rendering_kw, show=False):
        return plot3d_parametric_line(
            cos(x), sin(x), x, (x, -pi, pi), backend=B,
            show=show, rendering_kw=rendering_kw, use_latex=False,
            adaptive=False, n=5
        )

    p = _plot3d_parametric_line(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert f.axes[1].get_ylabel() == "x"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = _plot3d_parametric_line(PB, rendering_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter3d)
    assert f.data[0]["line"]["color"] == "red"
    assert f.data[0]["name"] == "x"
    assert f.data[0]["line"]["colorbar"]["title"]["text"] == "x"

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError, lambda: _plot3d_parametric_line(BB,
        rendering_kw=dict(line_color="red")).process_series())

    p = _plot3d_parametric_line(KBchild1, rendering_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Line)
    assert f.objects[0].color == 16711680
    assert f.objects[0].name is None

    if mayavi:
        p = _plot3d_parametric_line(MAB, rendering_kw=dict(color=(1, 0, 0)),
            show=True)
        assert len(p.series) == 1
        f = p.fig
        assert p._handles[0].actor.property.color == (1, 0, 0)


def test_plot3d():
    # verify that the backends produce the expected results when
    # `plot3d()` is called and `rendering_kw` overrides the default surface
    # settings

    x, y = symbols("x, y")

    def _plot3d(B, rendering_kw, use_latex=False, show=False):
        return plot3d(
            cos(x ** 2 + y ** 2),
            (x, -3, 3),
            (y, -3, 3),
            n=5,
            use_cm=False,
            backend=B,
            show=show,
            rendering_kw=rendering_kw,
            use_latex=use_latex
        )

    # use_cm=False will force to apply a default solid color to the mesh.
    # Here, I override that solid color with a custom color.
    p = _plot3d(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Poly3DCollection)
    # TODO: apparently, without showing the plot, the colors are not applied
    # to a Poly3DCollection...
    p.close()

    p = _plot3d(PB, rendering_kw=dict(colorscale=[[0, "cyan"], [1, "cyan"]]))
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
        lambda: _plot3d(BB, rendering_kw=dict(
            colorscale=[[0, "cyan"], [1, "cyan"]])).process_series())

    p = _plot3d(KBchild1, rendering_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Mesh)
    assert f.objects[0].color == 16711680
    assert f.objects[0].name is None

    if mayavi:
        p = _plot3d(MAB, rendering_kw=dict(color=(1, 0, 0)), show=True)
        assert len(p.series) == 1
        f = p.fig
        assert p._handles[0].actor.property.color == (1, 0, 0)


def test_plot3d_2():
    # verify that the backends uses string labels when `plot3d()` is called
    # with `use_latex=False` and `use_cm=True`

    x, y = symbols("x, y")

    _plot3d = lambda B, show=False: plot3d(
        cos(x ** 2 + y ** 2),
        sin(x ** 2 + y ** 2),
        (x, -3, 3),
        (y, -3, 3),
        n=5,
        use_cm=True,
        backend=B,
        show=show,
        use_latex=False
    )

    p = _plot3d(MB)
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 2
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_zlabel() == "f(x, y)"
    assert len(f.axes) == 3
    assert f.axes[1].get_ylabel() == str(cos(x ** 2 + y ** 2))
    assert f.axes[2].get_ylabel() == str(sin(x ** 2 + y ** 2))
    p.close()

    p = _plot3d(PB)
    assert len(p.series) == 2
    f = p.fig
    assert len(f.data) == 2
    assert p.fig.layout.scene.xaxis.title.text == "x"
    assert p.fig.layout.scene.yaxis.title.text == "y"
    assert p.fig.layout.scene.zaxis.title.text == "f(x, y)"
    assert f.data[0].colorbar.title.text == str(cos(x ** 2 + y ** 2))
    assert f.data[1].colorbar.title.text == str(sin(x ** 2 + y ** 2))
    assert f.data[0].name == str(cos(x ** 2 + y ** 2))
    assert f.data[1].name == str(sin(x ** 2 + y ** 2))
    assert f.data[0]["showscale"]
    assert f.layout["showlegend"]

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: _plot3d(BB).process_series())

    p = _plot3d(KBchild1)
    assert len(p.series) == 2
    f = p.fig
    assert len(f.objects) == 2
    assert p.fig.axes == ["x", "y", "f(x, y)"]

    if mayavi:
        p = _plot3d(MAB, show=True)
        assert len(p.series) == 2
        assert len(p._handles) == 2
        # orientation axis
        o = p.fig.children[1].children[0].children[0].children[3]
        xlabel = o.axes.x_axis_label_text
        ylabel = o.axes.y_axis_label_text
        zlabel = o.axes.z_axis_label_text
        assert [xlabel, ylabel, zlabel] == ["x", "y", "f(x, y)"]


def test_plot3d_wireframe():
    # verify that wireframe=True is going to add the expected number of line
    # data series and that appropriate keyword arguments work as expected

    x, y = symbols("x, y")
    _plot3d1 = lambda B, wf=True: plot3d(
        cos(x**2 + y**2), (x, -2, 2), (y, -3, 3), n1=5, n2=8,
        use_cm=True, backend=B, wireframe=wf, show=False)

    p0 = _plot3d1(PB, False)
    assert len(p0.series) == 1

    p1 = _plot3d1(PB)
    assert len(p1.series) == 21
    assert isinstance(p1[0], SurfaceOver2DRangeSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p1.series[1:])
    assert all((not s.adaptive) and (s.n == p1[0].n2) for s in p1.series[1:11])
    assert all((not s.adaptive) and (s.n == p1[0].n1) for s in p1.series[11:])
    assert all(p1.fig.data[1]["line"]["color"] == "#000000" for s in p1.series[1:])
    assert np.allclose(
        [t.x[0] for t in p1.fig.data[1:11]], np.linspace(-2, 2, 10))
    assert np.allclose(
        [t.y[0] for t in p1.fig.data[11:]], np.linspace(-3, 3, 10))

    p2 = _plot3d1(KBchild1)
    assert all(p2.fig.objects[1].color == 0 for s in p2.series[1:])

    if mayavi:
        p2b = _plot3d1(MAB)
        assert all(h[0].actor.property.color == (1, 0, 0) for h in p2b._handles)

    p2c = _plot3d1(MB)

    _plot3d2 = lambda B, rk: plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        use_cm=True, backend=B, n1=5, n2=8,
        wireframe=True, wf_n1=20, wf_n2=30,
        wf_rendering_kw=rk, wf_npoints=12, show=False)
    p3 = _plot3d2(PB, {"line_color": "#ff0000"})
    assert len(p3.series) == 1 + 20 + 30
    assert all(s.n == 12 for s in p3.series[1:])
    assert all(t["line"]["color"] == "#ff0000" for t in p3.fig.data[1:])

    p3 = _plot3d2(KBchild1, {"color": 0xff0000})
    assert all(t.color == 0xff0000 for t in p3.fig.objects[1:])

    r, theta = symbols("r, theta")
    _plot3d3 = lambda B, wf: plot3d(
        (cos(r**2) * exp(-r / 3), (r, 0, 3.25), (theta, 0, 2 * pi), "r"), backend=B, is_polar=True, use_cm=True, legend=True, n1=5, n2=8,
        color_func=lambda x, y, z: (x**2 + y**2)**0.5,
        wireframe=True, wf_n1=20, wf_n2=40, wf_rendering_kw=wf, show=False)
    p4 = _plot3d3(PB, {"line_color": "red"})
    assert len(p4.series) == 1 + 20 + 40
    assert isinstance(p4[0], SurfaceOver2DRangeSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p4.series[1:])
    assert all((not s.adaptive) and (s.n == p4[0].n2) for s in p4.series[1:21])
    assert all((not s.adaptive) and (s.n == p4[0].n1) for s in p4.series[21:])
    assert all(t["line"]["color"] == "red" for t in p4.fig.data[1:])
    assert np.allclose(
        [t.x[0] for t in p4.fig.data[1:21]], np.linspace(0, 3.25, 20))
    param = p4.series[1].get_data()[-1]
    assert np.isclose(param.min(), 0) and np.isclose(param.max(), 2*np.pi)
    param = p4.series[21].get_data()[-1]
    assert np.isclose(param.min(), 0) and np.isclose(param.max(), 3.25)


def test_plot3d_wireframe_lambda_function():
    # verify that wireframe=True correctly works also when the expression is
    # a lambda function

    _plot3d1 = lambda B, wf=True: plot3d(
        lambda x, y: np.cos(x**2 + y**2), ("x", -2, 2), ("y", -3, 3),
        n1=5, n2=8,
        use_cm=True, backend=B, wireframe=wf, show=False)

    p0 = _plot3d1(PB, False)
    assert len(p0.series) == 1

    p1 = _plot3d1(PB)
    assert len(p1.series) == 21
    assert isinstance(p1[0], SurfaceOver2DRangeSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p1.series[1:])
    assert all((not s.adaptive) and (s.n == p1[0].n2) for s in p1.series[1:11])
    assert all((not s.adaptive) and (s.n == p1[0].n1) for s in p1.series[11:])
    assert all(p1.fig.data[1]["line"]["color"] == "#000000" for s in p1.series[1:])
    assert np.allclose(
        [t.x[0] for t in p1.fig.data[1:11]], np.linspace(-2, 2, 10))
    assert np.allclose(
        [t.y[0] for t in p1.fig.data[11:]], np.linspace(-3, 3, 10))

    _plot3d3 = lambda B, wf: plot3d(
        lambda r, theta: np.cos(r**2) * np.exp(-r / 3),
        ("r", 0, 3.25), ("theta", 0, 2 * np.pi), "r",
        backend=B, is_polar=True, use_cm=True, legend=True, n1=5, n2=8,
        color_func=lambda x, y, z: (x**2 + y**2)**0.5,
        wireframe=True, wf_n1=20, wf_n2=40, wf_rendering_kw=wf, show=False)
    p4 = _plot3d3(PB, {"line_color": "red"})
    assert len(p4.series) == 1 + 20 + 40
    assert isinstance(p4[0], SurfaceOver2DRangeSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p4.series[1:])
    assert all((not s.adaptive) and (s.n == p4[0].n2) for s in p4.series[1:21])
    assert all((not s.adaptive) and (s.n == p4[0].n1) for s in p4.series[21:])
    assert all(t["line"]["color"] == "red" for t in p4.fig.data[1:])
    assert np.allclose(
        [t.x[0] for t in p4.fig.data[1:21]], np.linspace(0, 3.25, 20))
    param = p4.series[1].get_data()[-1]
    assert np.isclose(param.min(), 0) and np.isclose(param.max(), 2*np.pi)
    param = p4.series[21].get_data()[-1]
    assert np.isclose(param.min(), 0) and np.isclose(param.max(), 3.25)


def test_plot3d_parametric_surface_wireframe():
    # verify that wireframe=True is going to add the expected number of line
    # data series and that appropriate keyword arguments work as expected

    u, v = symbols("u, v")
    x = (1 + v / 2 * cos(u / 2)) * cos(u)
    y = (1 + v / 2 * cos(u / 2)) * sin(u)
    z = v / 2 * sin(u / 2)
    p = plot3d_parametric_surface(x, y, z, (u, 0, 2*pi), (v, -1, 1),
        backend=PB, use_cm=True, n1=5, n2=8,
        wireframe=True, wf_n1=5, wf_n2=6,
        wf_rendering_kw={"line_color": "red"}, show=False)
    assert len(p.series) == 1 + 5 + 6
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p.series[1:])
    assert all((not s.adaptive) and (s.n == p[0].n2) for s in p.series[1:6])
    assert all((not s.adaptive) and (s.n == p[0].n1) for s in p.series[6:])
    assert all(t["line"]["color"] == "red" for t in p.fig.data[1:])
    assert all([np.isclose(k[0], -1) and np.isclose(k[-1], 1)
        for k in [t.get_data()[-1] for t in p.series[1:6]]])
    assert all([np.isclose(k[0], 0) and np.isclose(k[-1], 2*np.pi)
        for k in [t.get_data()[-1] for t in p.series[6:]]])


def test_plot3d_parametric_surface_wireframe_lambda_function():
    # verify that wireframe=True correctly works also when the expression is
    # a lambda function
    x = lambda u, v: v * np.cos(u)
    y = lambda u, v: v * np.sin(u)
    z = lambda u, v: np.sin(4 * u)
    _plot = lambda wf: plot3d_parametric_surface(
        x, y, z, ("u", 0, 2*np.pi), ("v", -1, 0), n1=5, n2=8,
        backend=PB, use_cm=True,
        wireframe=wf, wf_n1=5, wf_n2=6,
        show=False)

    p0 = _plot(False)
    assert len(p0.series) == 1

    p1 = _plot(True)
    assert len(p1.series) == 1 + 5 + 6
    assert isinstance(p1[0], ParametricSurfaceSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p1.series[1:])
    assert all((not s.adaptive) and (s.n == p1[0].n2) for s in p1.series[1:6])
    assert all((not s.adaptive) and (s.n == p1[0].n1) for s in p1.series[6:])
    assert all(p1.fig.data[1]["line"]["color"] == "#000000" for s in p1.series[1:])
    assert all([np.isclose(k[0], -1) and np.isclose(k[-1], 0)
        for k in [t.get_data()[-1] for t in p1.series[1:6]]])
    assert all([np.isclose(k[0], 0) and np.isclose(k[-1], 2*np.pi)
        for k in [t.get_data()[-1] for t in p1.series[6:]]])


def test_plot_contour():
    # verify that the backends produce the expected results when
    # `plot_contour()` is called and `rendering_kw` overrides the default
    # surface settings

    x, y = symbols("x, y")

    _plot_contour = lambda B, rendering_kw: plot_contour(
        cos(x ** 2 + y ** 2),
        (x, -3, 3),
        (y, -3, 3),
        n=5,
        backend=B,
        show=False,
        rendering_kw=rendering_kw,
        use_latex=False
    )

    p = _plot_contour(MB, rendering_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert f.axes[1].get_ylabel() == str(cos(x ** 2 + y ** 2))
    # TODO: how to retrieve the colormap from a contour series?????
    p.close()

    p = _plot_contour(PB, rendering_kw=dict(contours=dict(coloring="lines")))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Contour)
    assert f.data[0]["contours"]["coloring"] == "lines"
    assert f.data[0]["colorbar"]["title"]["text"] == str(cos(x ** 2 + y ** 2))

    # Bokeh doesn't use rendering_kw dictionary. Nothing to customize yet.
    p = _plot_contour(BB, rendering_kw=dict())
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
            rendering_kw=dict()).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot_contour(MAB,
                rendering_kw=dict()).process_series())


def test_plot_contour_is_filled():
    # verify that is_filled=True produces different results than
    # is_filled=False
    x, y = symbols("x, y")

    _plot_contour = lambda B, is_filled: plot_contour(
        cos(x ** 2 + y ** 2),
        (x, -3, 3),
        (y, -3, 3),
        n=5,
        backend=B,
        show=False,
        use_latex=False,
        is_filled=is_filled
    )

    p1 = _plot_contour(MB, True)
    p1.process_series()
    p2 = _plot_contour(MB, False)
    p2.process_series()
    assert p1._handles[0][-1] is None
    assert hasattr(p2._handles[0][-1], "__iter__")
    assert len(p2._handles[0][-1]) > 0

    p1 = _plot_contour(PB, True)
    p2 = _plot_contour(PB, False)
    assert p1.fig.data[0].showscale
    assert p1.fig.data[0].contours.coloring is None
    assert p1.fig.data[0].contours.showlabels is False
    assert not p2.fig.data[0].showscale
    assert p2.fig.data[0].contours.coloring == "lines"
    assert p2.fig.data[0].contours.showlabels


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
        use_latex=False,
        n1=5, n2=8
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

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot_vector(MAB, quiver_kw=dict(),
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
        use_latex=False,
        n1=5, n2=5
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

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot_vector(MAB, stream_kw=dict(),
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
        use_latex=False,
        n1=5, n2=5
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

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot_vector(MAB, stream_kw=dict(),
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
        show=False,
        use_latex=False,
        n1=5, n2=8)

    # contours + quivers: 1 colorbar for the contours
    p = _plot_vector_1(True, False)
    assert len(p.series) == 2
    assert len(p.fig.axes) == 2
    assert len(p.fig.axes[0].collections) > 1
    assert p.fig.axes[1].get_ylabel() == "Magnitude"

    # contours + streamlines: 1 colorbar for the contours
    p = _plot_vector_1(True, True)
    assert len(p.series) == 2
    assert len(p.fig.axes) == 2
    assert len(p.fig.axes[0].collections) > 1
    assert p.fig.axes[1].get_ylabel() == "Magnitude"

    # only quivers: 1 colorbar for the quivers
    p = _plot_vector_1(False, False)
    assert len(p.series) == 1
    assert len(p.fig.axes) == 2
    assert len(p.fig.axes[0].collections) == 1
    assert p.fig.axes[1].get_ylabel() == "(x, y)"

    # only streamlines: 1 colorbar for the streamlines
    p = _plot_vector_1(False, False)
    assert len(p.series) == 1
    assert len(p.fig.axes) == 2
    assert len(p.fig.axes[0].collections) == 1
    assert p.fig.axes[1].get_ylabel() == "(x, y)"

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

    _plot_vector = lambda B, quiver_kw, show=False, **kwargs: plot_vector(
    Matrix([z, y, x]),
        (x, -5, 5),
        (y, -4, 4),
        (z, -3, 3),
        backend=B,
        n=5,
        quiver_kw=quiver_kw,
        show=show,
        use_latex=False,
        **kwargs
    )

    p = _plot_vector(MB, quiver_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert ax.collections[0].cmap.name == "jet"
    assert f.axes[1].get_ylabel() == str((z, y, x))
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
    assert f.data[0]["colorbar"]["title"]["text"] == str((z, y, x))

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

    if mayavi:
        p = _plot_vector(MAB, quiver_kw=dict(color=(1, 0, 0)), show=True)
        assert len(p.series) == 1
        assert len(p.fig.children) == 1
        assert p._handles[0].actor.property.color == (1, 0, 0)


def test_plot_vector_3d_streamlines():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `stream_kw` overrides the
    # default settings

    x, y, z = symbols("x, y, z")

    _plot_vector = lambda B, stream_kw, show=False, kwargs=dict(): plot_vector(
        Matrix([z, y, x]),
        (x, -5, 5),
        (y, -4, 4),
        (z, -3, 3),
        backend=B,
        n=5,
        streamlines=True,
        show=show,
        stream_kw=stream_kw,
        use_latex=False,
        **kwargs
    )

    p = _plot_vector(MB, stream_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert f.axes[1].get_ylabel() == str((z, y, x))
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
    assert f.data[0]["colorbar"]["title"]["text"] == str((z, y, x))

    # test different combinations for streamlines: it should not raise errors
    p = _plot_vector(PB, stream_kw=dict(starts=True))
    p = _plot_vector(PB, stream_kw=dict(starts={
        "x": np.linspace(-5, 5, 10),
        "y": np.linspace(-4, 4, 10),
        "z": np.linspace(-3, 3, 10),
    }))

    # other keywords: it should not raise errors
    p = _plot_vector(PB, stream_kw=dict(), kwargs=dict(use_cm=False))

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
    p = _plot_vector(KBchild1, stream_kw=dict(), kwargs=dict(use_cm=False))

    if mayavi:
        p = _plot_vector(MAB, stream_kw=dict(color=(1, 0, 0)), show=True)
        assert len(p.series) == 1
        assert len(p.fig.children) == 1
        assert p._handles[0].actor.property.color == (1, 0, 0)


def test_plot_vector_2d_normalize():
    # verify that backends are capable of normalizing a vector field before
    # plotting it. Since all backend are different from each other, let's test
    # that data in the figures is different in the two cases normalize=True
    # and normalize=False

    x, y, u, v = symbols("x, y, u, v")

    _pv = lambda B, norm=False: plot_vector(
        [-sin(y), cos(x)], (x, -2, 2), (y, -2, 2),
        backend=B, normalize=norm, n=5, scalar=False, use_cm=False, show=False)

    p1 = _pv(MB, False)
    p2 = _pv(MB, True)
    uu1 = p1.fig.axes[0].collections[0].U
    vv1 = p1.fig.axes[0].collections[0].V
    uu2 = p2.fig.axes[0].collections[0].U
    vv2 = p2.fig.axes[0].collections[0].V
    assert not np.allclose(uu1, uu2)
    assert not np.allclose(vv1, vv2)
    assert not np.allclose(np.sqrt(uu1**2 + vv1**2), 1)
    assert np.allclose(np.sqrt(uu2**2 + vv2**2), 1)

    p1 = _pv(PB, False)
    p2 = _pv(PB, True)
    d1x = np.array(p1.fig.data[0].x).astype(float)
    d1y = np.array(p1.fig.data[0].y).astype(float)
    d2x = np.array(p2.fig.data[0].x).astype(float)
    d2y = np.array(p2.fig.data[0].y).astype(float)
    assert not np.allclose(d1x, d2x, equal_nan=True)
    assert not np.allclose(d1y, d2y, equal_nan=True)

    p1 = _pv(BB, False)
    p2 = _pv(BB, True)
    x01 = p1.fig.renderers[0].data_source.data["x0"]
    x11 = p1.fig.renderers[0].data_source.data["x1"]
    y01 = p1.fig.renderers[0].data_source.data["y0"]
    y11 = p1.fig.renderers[0].data_source.data["y1"]
    m1 = p1.fig.renderers[0].data_source.data["magnitude"]
    x02 = p2.fig.renderers[0].data_source.data["x0"]
    x12 = p2.fig.renderers[0].data_source.data["x1"]
    y02 = p2.fig.renderers[0].data_source.data["y0"]
    y12 = p2.fig.renderers[0].data_source.data["y1"]
    m2 = p2.fig.renderers[0].data_source.data["magnitude"]
    assert not np.allclose(x01, x02)
    assert not np.allclose(x11, x12)
    assert not np.allclose(y01, y02)
    assert not np.allclose(y11, y12)
    assert np.allclose(m1, m2)

    # interactive plots
    _pv2 = lambda B, norm=False: plot_vector(
        [-u * sin(y), cos(x)], (x, -2, 2), (y, -2, 2),
        backend=B, normalize=norm, n=5, scalar=False, use_cm=False, show=False,
        params={u: (1, 0, 2)})

    p1 = _pv2(MB, False)
    p1.backend._update_interactive({u: 1.5})
    p2 = _pv2(MB, True)
    p2.backend._update_interactive({u: 1.5})
    uu1 = p1.backend.fig.axes[0].collections[0].U
    vv1 = p1.backend.fig.axes[0].collections[0].V
    uu2 = p2.backend.fig.axes[0].collections[0].U
    vv2 = p2.backend.fig.axes[0].collections[0].V
    assert not np.allclose(uu1, uu2)
    assert not np.allclose(vv1, vv2)
    assert not np.allclose(np.sqrt(uu1**2 + vv1**2), 1)
    assert np.allclose(np.sqrt(uu2**2 + vv2**2), 1)

    p1 = _pv2(PB, False)
    p1.backend._update_interactive({u: 1.5})
    p2 = _pv2(PB, True)
    p2.backend._update_interactive({u: 1.5})
    d1x = np.array(p1.fig.data[0].x).astype(float)
    d1y = np.array(p1.fig.data[0].y).astype(float)
    d2x = np.array(p2.fig.data[0].x).astype(float)
    d2y = np.array(p2.fig.data[0].y).astype(float)
    assert not np.allclose(d1x, d2x, equal_nan=True)
    assert not np.allclose(d1y, d2y, equal_nan=True)

    p1 = _pv2(BB, False)
    p1.backend._update_interactive({u: 1.5})
    p2 = _pv2(BB, True)
    p2.backend._update_interactive({u: 1.5})
    x01 = p1.fig.renderers[0].data_source.data["x0"]
    x11 = p1.fig.renderers[0].data_source.data["x1"]
    y01 = p1.fig.renderers[0].data_source.data["y0"]
    y11 = p1.fig.renderers[0].data_source.data["y1"]
    m1 = p1.fig.renderers[0].data_source.data["magnitude"]
    x02 = p2.fig.renderers[0].data_source.data["x0"]
    x12 = p2.fig.renderers[0].data_source.data["x1"]
    y02 = p2.fig.renderers[0].data_source.data["y0"]
    y12 = p2.fig.renderers[0].data_source.data["y1"]
    m2 = p2.fig.renderers[0].data_source.data["magnitude"]
    assert not np.allclose(x01, x02)
    assert not np.allclose(x11, x12)
    assert not np.allclose(y01, y02)
    assert not np.allclose(y11, y12)
    assert np.allclose(m1, m2)


def test_plot_vector_3d_normalize():
    # verify that backends are capable of normalizing a vector field before
    # plotting it. Since all backend are different from each other, let's test
    # that data in the figures is different in the two cases normalize=True
    # and normalize=False

    x, y, z, u, v = symbols("x, y, z, u, v")

    _pv = lambda B, norm=False: plot_vector(
        [z, -x, y], (x, -2, 2), (y, -2, 2), (z, -2, 2),
        backend=B, normalize=norm, n=3, use_cm=False, show=False)

    p1 = _pv(MB, False)
    p2 = _pv(MB, True)
    seg1 = np.array(p1.fig.axes[0].collections[0].get_segments())
    seg2 = np.array(p2.fig.axes[0].collections[0].get_segments())
    # TODO: how can I test that these two quivers are different?
    # assert not np.allclose(seg1, seg2)

    p1 = _pv(PB, False)
    p2 = _pv(PB, True)
    assert not np.allclose(p1.fig.data[0]["u"], p2.fig.data[0]["u"])
    assert not np.allclose(p1.fig.data[0]["v"], p2.fig.data[0]["v"])
    assert not np.allclose(p1.fig.data[0]["w"], p2.fig.data[0]["w"])

    p1 = _pv(KBchild1, False)
    p2 = _pv(KBchild1, True)
    assert not np.allclose(p1.fig.objects[0].vectors, p2.fig.objects[0].vectors)

    # interactive plots
    _pv2 = lambda B, norm=False: plot_vector(
        [u * z, -x, y], (x, -2, 2), (y, -2, 2), (z, -2, 2),
        backend=B, normalize=norm, n=3, use_cm=False, show=False,
        params={u: (1, 0, 2)})

    p1 = _pv2(MB, False)
    p1.backend._update_interactive({u: 1.5})
    p2 = _pv2(MB, True)
    p2.backend._update_interactive({u: 1.5})
    seg1 = np.array(p1.fig.axes[0].collections[0].get_segments())
    seg2 = np.array(p2.fig.axes[0].collections[0].get_segments())
    # TODO: how can I test that these two quivers are different?
    # assert not np.allclose(seg1, seg2)

    p1 = _pv2(PB, False)
    p1.backend._update_interactive({u: 1.5})
    p2 = _pv2(PB, True)
    p2.backend._update_interactive({u: 1.5})
    assert not np.allclose(p1.fig.data[0]["u"], p2.fig.data[0]["u"])
    assert not np.allclose(p1.fig.data[0]["v"], p2.fig.data[0]["v"])
    assert not np.allclose(p1.fig.data[0]["w"], p2.fig.data[0]["w"])

    p1 = _pv2(KBchild1, False)
    p1.backend._update_interactive({u: 1.5})
    p2 = _pv2(KBchild1, True)
    p2.backend._update_interactive({u: 1.5})
    assert not np.allclose(p1.fig.objects[0].vectors, p2.fig.objects[0].vectors)


def test_plot_implicit_adaptive_true():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True`

    x, y = symbols("x, y")

    _plot_implicit = lambda B, contour_kw: plot_implicit(
        x > y, (x, -5, 5), (y, -4, 4), backend=B, show=False,
        adaptive=True, contour_kw=contour_kw, use_latex=False
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

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(NotImplementedError,
            lambda: _plot_implicit(MAB, contour_kw=dict()).process_series())


def test_plot_implicit_adaptive_false():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True` and `contour_kw`
    # overrides the default settings

    x, y = symbols("x, y")

    _plot_implicit = lambda B, contour_kw: plot_implicit(
        x > y,
        (x, -5, 5),
        (y, -4, 4),
        n=5,
        backend=B,
        adaptive=False,
        show=False,
        contour_kw=contour_kw,
        use_latex=False,
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

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(NotImplementedError,
            lambda: _plot_implicit(MAB, contour_kw=dict()).process_series())


def test_plot_real_imag():
    # verify that the backends produce the expected results when
    # `plot_real_imag()` is called and `rendering_kw` overrides the default
    # settings

    x = symbols("x")

    _plot_real_imag = lambda B, rendering_kw: plot_real_imag(
        sqrt(x), (x, -5, 5), backend=B, rendering_kw=rendering_kw, show=False,
        use_latex=False, adaptive=False, n=5
    )

    p = _plot_real_imag(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.get_lines()) == 2
    assert ax.get_lines()[0].get_label() == "Re(sqrt(x))"
    assert ax.get_lines()[0].get_color() == "red"
    assert ax.get_lines()[1].get_label() == "Im(sqrt(x))"
    assert ax.get_lines()[1].get_color() == "red"
    p.close()

    p = _plot_real_imag(PB, rendering_kw=dict(line_color="red"))
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

    p = _plot_real_imag(BB, rendering_kw=dict(line_color="red"))
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
        lambda: _plot_real_imag(KBchild1, rendering_kw=dict()).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(NotImplementedError,
            lambda: _plot_real_imag(MAB, rendering_kw=dict()).process_series())


def test_plot_complex_1d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    x = symbols("x")

    _plot_complex = lambda B, rendering_kw: plot_complex(
        sqrt(x), (x, -5, 5), backend=B, rendering_kw=rendering_kw, show=False,
        adaptive=False, n=5
    )

    p = _plot_complex(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == "Arg(sqrt(x))"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()

    p = _plot_complex(PB, rendering_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "Arg(sqrt(x))"
    assert f.data[0]["line"]["color"] == "red"
    assert p.fig.data[0]["marker"]["colorbar"]["title"]["text"] == "Arg(sqrt(x))"

    p = _plot_complex(BB, rendering_kw=dict(line_color="red"))
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
        lambda: _plot_complex(KBchild1, rendering_kw=dict()).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(NotImplementedError,
            lambda: _plot_complex(MAB, rendering_kw=dict()).process_series())


def test_plot_complex_2d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    x = symbols("x")

    _plot_complex = lambda B, rendering_kw: plot_complex(
        sqrt(x), (x, -5 - 5 * I, 5 + 5 * I), backend=B, coloring="a",
        rendering_kw=rendering_kw, show=False, adaptive=False, n=5
    )

    p = _plot_complex(MB, rendering_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.images) == 1
    assert f.axes[1].get_ylabel() == "Argument"
    assert ax.images[0].get_extent() == [-5.0, 5.0, -5.0, 5.0]
    p.close()

    p = _plot_complex(MB, rendering_kw=dict(extent=[-6, 6, -7, 7]))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.images) == 1
    assert f.axes[1].get_ylabel() == "Argument"
    assert ax.images[0].get_extent() == [-6, 6, -7, 7]
    p.close()

    p = _plot_complex(PB, rendering_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Image)
    assert f.data[0]["name"] == "sqrt(x)"
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[1]["marker"]["colorbar"]["title"]["text"] == "Argument"

    p = _plot_complex(BB, rendering_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.ImageRGBA)
    assert f.right[0].title == "Argument"
    assert (f.toolbar.tools[-2].tooltips == [('x', '$x'), ('y', '$y'),
        ("Abs", "@abs"), ("Arg", "@arg")])

    # K3D doesn't support 2D plots
    raises(NotImplementedError,
        lambda: _plot_complex(KBchild1, rendering_kw=dict()).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(NotImplementedError,
            lambda: _plot_complex(MAB, rendering_kw=dict()).process_series())


def test_plot_complex_3d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    x = symbols("x")

    _plot_complex = lambda B, rendering_kw: plot_complex(
        sqrt(x),
        (x, -5 - 5 * I, 5 + 5 * I),
        backend=B,
        threed=True,
        rendering_kw=rendering_kw,
        show=False,
        use_cm=False,
        n=5
    )

    p = _plot_complex(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Poly3DCollection)
    # TODO: apparently, without showing the plot, the colors are not applied
    # to a Poly3DCollection...
    p.close()

    p = _plot_complex(PB, rendering_kw=dict())
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
        lambda: _plot_complex(BB, rendering_kw=dict()).process_series())

    p = _plot_complex(KBchild1, rendering_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Mesh)
    assert f.objects[0].name is None


def test_plot_complex_list():
    # verify that no errors are raise when plotting lists of complex points
    p = plot_complex_list(3 + 2 * I, 4 * I, 2, backend=MB, show=False)
    f = p.fig


def test_plot_list_is_filled_false():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=False`

    _plot_list = lambda B: plot_list([1, 2, 3], [1, 2, 3],
        backend=B, is_point=True, is_filled=False, show=False, use_latex=False)

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
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Scatter)
    assert f.renderers[0].glyph.line_color != f.renderers[0].glyph.fill_color

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_list(KBchild1).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot_list(MAB).process_series())


def test_plot_list_is_filled_true():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=True`

    _plot_list = lambda B: plot_list([1, 2, 3], [1, 2, 3],
        backend=B, is_point=True, is_filled=True, show=False, use_latex=False)

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
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Scatter)
    assert f.renderers[0].glyph.line_color == f.renderers[0].glyph.fill_color

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_list(KBchild1).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot_list(MAB).process_series())


def test_plot_list_color_func():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `color_func`

    _plot_list = lambda B: plot_list([1, 2, 3], [1, 2, 3],
        backend=B, color_func=lambda x, y: np.arange(len(x)), show=False, use_latex=False, is_point=True)

    p = _plot_list(MB)
    f = p.fig
    ax = f.axes[0]
    # TODO:  matplotlib applie colormap color only after being shown :|
    # assert p.ax.collections[0].get_facecolors().shape == (3, 4)
    p.close()

    p = _plot_list(PB)
    f = p.fig
    assert f.data[0]["mode"] == "markers"
    assert np.allclose(f.data[0]["marker"]["color"], [0, 1, 2])


    p = _plot_list(BB)
    assert len(p.series) == 1
    f = p.fig
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Scatter)
    assert np.allclose(f.renderers[0].data_source.data["us"], [0, 1, 2])

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_list(KBchild1).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot_list(MAB).process_series())


def test_plot_piecewise_single_series():
    # verify that plot_piecewise plotting 1 piecewise composed of N
    # sub-expressions uses only 1 color

    x = symbols("x")

    _plot_piecewise = lambda B: plot_piecewise(
        Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10),
        backend=B, show=False, use_latex=False,
        adaptive=False, n=5)

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

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(NotImplementedError, lambda: _plot_piecewise(MAB))


def test_plot_piecewise_multiple_series():
    # verify that plot_piecewise plotting N piecewise expressions uses
    # only N different colors

    x = symbols("x")

    _plot_piecewise = lambda B: plot_piecewise(
        (Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10)),
        (Piecewise((sin(x), x < 0), (2, Eq(x, 0)), (cos(x), x > 0)), (x, -6, 4)),
        backend=B, show=False, use_latex=False,
        adaptive=False, n=5)

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

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(NotImplementedError, lambda: _plot_piecewise(MAB))


def test_plot_geometry_1():
    # verify that the backends produce the expected results when
    # `plot_geometry()` is called
    from sympy.geometry import Line as SymPyLine

    _plot_geometry = lambda B: plot_geometry(
        SymPyLine((1, 2), (5, 4)), Circle((0, 0), 4), Polygon((2, 2), 3, n=6),
        backend=B, show=False, is_filled=False, use_latex=False)

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

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(NotImplementedError, lambda: _plot_geometry(MAB).process_series())

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
        Point2D(0, 0), is_filled=is_filled, backend=B, show=False, use_latex=False)

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
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Scatter)]) == 1
    p = _plot_geometry(BB, True)
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Line)]) == 1
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Patch)]) == 3
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.MultiLine)]) == 1
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Scatter)]) == 1


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
        p = plot(sin(x), cos(x), backend=MB, show=False, adaptive=False, n=5)
        filename = "test_mpl_save_1.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(sin(x), cos(x), backend=MB, show=False, adaptive=False, n=5)
        filename = "test_mpl_save_2.pdf"
        p.save(os.path.join(tmpdir, filename), dpi=150)
        p.close()

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=MB,
            show=False, adaptive=False, n1=5, n2=5)
        filename = "test_mpl_save_3.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=MB,
            show=False, adaptive=False, n1=5, n2=5)
        filename = "test_mpl_save_4.pdf"
        p.save(os.path.join(tmpdir, filename), dpi=150)
        p.close()

        # Bokeh requires additional libraries to save static pictures.
        # Raise an error because their are not installed.
        p = plot(sin(x), cos(x), backend=BB, show=False, adaptive=False, n=5)
        filename = "test_bokeh_save_1.png"
        raises(RuntimeError, lambda: p.save(os.path.join(tmpdir, filename)))

        p = plot(sin(x), cos(x), backend=BB, show=False, adaptive=False, n=5)
        filename = "test_bokeh_save_2.svg"
        raises(RuntimeError, lambda: p.save(os.path.join(tmpdir, filename)))

        p = plot(sin(x), cos(x), backend=BB, show=False, adaptive=False, n=5)
        filename = "test_bokeh_save_3.html"
        p.save(os.path.join(tmpdir, filename))

        p = plot(sin(x), cos(x), backend=BB, show=False, adaptive=False, n=5)
        filename = "test_bokeh_save_4.html"
        p.save(os.path.join(tmpdir, filename), resources=bokeh.resources.INLINE)

        # Plotly requires additional libraries to save static pictures.
        # Raise an error because their are not installed.
        p = plot(sin(x), cos(x), backend=PB, show=False, adaptive=False, n=5)
        filename = "test_plotly_save_1.png"
        raises(ValueError, lambda: p.save(os.path.join(tmpdir, filename)))

        p = plot(sin(x), cos(x), backend=PB, show=False, adaptive=False, n=5)
        filename = "test_plotly_save_3.html"
        p.save(os.path.join(tmpdir, filename))

        p = plot(sin(x), cos(x), backend=PB, show=False, adaptive=False, n=5)
        filename = "test_plotly_save_4.html"
        p.save(os.path.join(tmpdir, filename), include_plotlyjs="cdn")

        # K3D-Jupyter: use KBchild1 in order to run tests.
        # NOTE: K3D is designed in such a way that the plots need to be shown
        # on the screen before saving them. Since it is not possible to show
        # them on the screen during tests, we are only going to test that it
        # proceeds smoothtly or it raises errors when wrong options are given
        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1,
            adaptive=False, n1=5, n2=5)
        filename = "test_k3d_save_1.png"
        p.save(os.path.join(tmpdir, filename))

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1,
            adaptive=False, n1=5, n2=5)
        filename = "test_k3d_save_2.jpg"
        raises(ValueError, lambda: p.save(os.path.join(tmpdir, filename)))

        # unexpected keyword argument
        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1,
            adaptive=False, n1=5, n2=5)
        filename = "test_k3d_save_3.jpg"
        raises(ValueError, lambda: p.save(os.path.join(tmpdir, filename),
            parameter=True))

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1,
            adaptive=False, n1=5, n2=5)
        filename = "test_k3d_save_4.html"
        p.save(os.path.join(tmpdir, filename))

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1,
            adaptive=False, n1=5, n2=5)
        filename = "test_k3d_save_4.html"
        p.save(os.path.join(tmpdir, filename), include_js=True)

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=KBchild1,
            adaptive=False, n1=5, n2=5)
        filename = "test_k3d_save_4.html"
        raises(TypeError, lambda: p.save(os.path.join(tmpdir, filename),
            include_js=True, parameter=True))

        if mayavi:
            p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=MAB,
                adaptive=False, n1=5, n2=5)
            filename = "test_mab_save_1.png"
            p.save(os.path.join(tmpdir, filename))


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
            n1 = 5, n2 = 5, n3 = 5
        )
        p = B(s)
        raises(NotImplementedError, lambda: p._update_interactive(params))

    func(KBchild1)
    if mayavi:
        func(MAB) # Mayavi doesn't implement _update_interactive yet
    func(PB)
    func(MB)


def test_aspect_ratio_2d_issue_11764():
    # verify that the backends apply the provided aspect ratio.
    # NOTE: read the backend docs to understand which options are available.
    x = symbols("x")

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        show=False, backend=MB, adaptive=False, n=5)
    assert p.aspect == "auto"
    assert p.fig.axes[0].get_aspect() == "auto"
    p.close()

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        aspect=(1, 1), show=False, backend=MB, adaptive=False, n=5)
    assert p.aspect == (1, 1)
    assert p.fig.axes[0].get_aspect() == 1
    p.close()

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        aspect="equal", show=False, backend=MB, adaptive=False, n=5)
    assert p.aspect == "equal"
    assert p.fig.axes[0].get_aspect() == 1
    p.close()

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        show=False, backend=PB, adaptive=False, n=5)
    assert p.aspect == "auto"
    assert p.fig.layout.yaxis.scaleanchor is None

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        aspect="equal", show=False, backend=PB, adaptive=False, n=5)
    assert p.aspect == "equal"
    assert p.fig.layout.yaxis.scaleanchor == "x"

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        show=False, backend=BB, adaptive=False, n=5)
    assert p.aspect == "auto"
    assert not p.fig.match_aspect

    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        aspect="equal", show=False, backend=BB, adaptive=False, n=5)
    assert p.aspect == "equal"
    assert p.fig.match_aspect


def test_aspect_ratio_3d():
    # verify that the backends apply the provided aspect ratio.
    # NOTE:
    # 1. read the backend docs to understand which options are available.
    # 2. K3D doesn't use the `aspect` keyword argument.
    x, y = symbols("x, y")

    p = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        n1=5, n2=5, backend=MB, show=False)
    assert p.aspect == "auto"

    # matplotlib's Axes3D currently only supports the aspect argument 'auto'
    raises(NotImplementedError,
        lambda: plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
            n1=5, n2=5, backend=MB, show=False, aspect=(1, 1)).process_series())

    p = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        n1=5, n2=5, backend=PB, show=False)
    assert p.aspect == "auto"
    assert p.fig.layout.scene.aspectmode == "auto"

    p = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        n1=5, n2=5, backend=PB, show=False, aspect="cube")
    assert p.aspect == "cube"
    assert p.fig.layout.scene.aspectmode == "cube"

    d = dict(x=1, y=1, z=1)
    p = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        n1=5, n2=5, backend=PB, show=False, aspect=d)
    assert p.aspect == d
    assert p.fig.layout.scene.aspectmode == "manual"
    assert all(p.fig.layout.scene.aspectratio[k] == d[k] for k in ["x", "y", "z"])

    if mayavi:
        p = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
            n1=15, n2=15, backend=MAB, show=True, aspect="equal")
        assert p.aspect == "equal"
        assert np.allclose(
            p.fig.children[0].children[0].children[0].children[1].axes.bounds,
            [-2, 2, -2, 2, -1, 1], rtol=1e-02)

        p = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
            n1=5, n2=5, backend=MAB, show=True, aspect="auto")
        assert p.aspect == "auto"
        assert np.allclose(
            p.fig.children[0].children[0].children[0].children[1].axes.bounds,
            [0, 1, 0, 1, 0, 1], rtol=1e-02)


def test_plot_size():
    # verify that the keyword `size` is doing it's job
    # NOTE: K3DBackend doesn't support custom size

    x, y = symbols("x, y")

    p = plot(sin(x), backend=MB, size=(8, 4), show=False, adaptive=False, n=5)
    s = p.fig.get_size_inches()
    assert (s[0] == 8) and (s[1] == 4)
    p.close()

    p = plot(sin(x), backend=MB, size=(10, 5), show=False, adaptive=False, n=5)
    s = p.fig.get_size_inches()
    assert (s[0] == 10) and (s[1] == 5)
    p.close()

    p = plot(sin(x), backend=PB, show=False, adaptive=False, n=5)
    assert p.fig.layout.width is None
    assert p.fig.layout.height is None

    p = plot(sin(x), backend=PB, size=(800, 400), show=False,
        adaptive=False, n=5)
    assert p.fig.layout.width == 800
    assert p.fig.layout.height == 400

    p = plot(sin(x), backend=BB, show=False, adaptive=False, n=5)
    assert p.fig.sizing_mode == "stretch_width"

    p = plot(sin(x), backend=BB, size=(400, 200), show=False,
        adaptive=False, n=5)
    assert p.fig.sizing_mode == "fixed"
    assert (p.fig.width == 400) and (p.fig.height == 200)

    if mayavi:
        # NOTE: have no idea how to retrieve the size from a Mayavi scene.
        # Let's measure the image dimensions
        with TemporaryDirectory(prefix="sympy_") as tmpdir:
            p = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
                backend=MAB, size=(400, 200), show=True, adaptive=False, n1=5, n2=5)
            filename = "test_mab_save_1.png"
            p.save(os.path.join(tmpdir, filename))
            png_pil_img = Image.open(os.path.join(tmpdir, filename))
            assert png_pil_img.size == p.size == (400, 200)


def test_plot_scale_lin_log():
    # verify that backends are applying the correct scale to the axes
    # NOTE: none of the 3D libraries currently support log scale.

    x, y = symbols("x, y")

    p = plot(log(x), backend=MB, xscale="linear", yscale="linear", show=False,
        adaptive=False, n=5)
    assert p.fig.axes[0].get_xscale() == "linear"
    assert p.fig.axes[0].get_yscale() == "linear"
    p.close()

    p = plot(log(x), backend=MB, xscale="log", yscale="linear", show=False,
        adaptive=False, n=5)
    assert p.fig.axes[0].get_xscale() == "log"
    assert p.fig.axes[0].get_yscale() == "linear"
    p.close()

    p = plot(log(x), backend=MB, xscale="linear", yscale="log", show=False,
        adaptive=False, n=5)
    assert p.fig.axes[0].get_xscale() == "linear"
    assert p.fig.axes[0].get_yscale() == "log"
    p.close()

    p = plot(log(x), backend=PB, xscale="linear", yscale="linear", show=False,
        adaptive=False, n=5)
    assert p.fig.layout["xaxis"]["type"] == "linear"
    assert p.fig.layout["yaxis"]["type"] == "linear"

    p = plot(log(x), backend=PB, xscale="log", yscale="linear", show=False,
        adaptive=False, n=5)
    assert p.fig.layout["xaxis"]["type"] == "log"
    assert p.fig.layout["yaxis"]["type"] == "linear"

    p = plot(log(x), backend=PB, xscale="linear", yscale="log", show=False,
        adaptive=False, n=5)
    assert p.fig.layout["xaxis"]["type"] == "linear"
    assert p.fig.layout["yaxis"]["type"] == "log"

    p = plot(log(x), backend=BB, xscale="linear", yscale="linear", show=False,
        adaptive=False, n=5)
    assert isinstance(p.fig.x_scale, bokeh.models.scales.LinearScale)
    assert isinstance(p.fig.y_scale, bokeh.models.scales.LinearScale)

    p = plot(log(x), backend=BB, xscale="log", yscale="linear", show=False,
        adaptive=False, n=5)
    assert isinstance(p.fig.x_scale, bokeh.models.scales.LogScale)
    assert isinstance(p.fig.y_scale, bokeh.models.scales.LinearScale)

    p = plot(log(x), backend=BB, xscale="linear", yscale="log", show=False,
        adaptive=False, n=5)
    assert isinstance(p.fig.x_scale, bokeh.models.scales.LinearScale)
    assert isinstance(p.fig.y_scale, bokeh.models.scales.LogScale)


##############################################################################
########################### BACKEND CAN USE LATEX ############################
##############################################################################


def test_backend_latex_labels():
    # verify that backends are going to set axis latex-labels in the
    # 2D and 3D case
    x1, x2 = symbols("x_1^2, x_2")
    p = lambda B, use_latex: plot(cos(x1), (x1, -1, 1), backend=B,
        show=False, use_latex=use_latex, adaptive=False, n=5)

    p1 = p(MB, True)
    p2 = p(MB, False)
    assert p1.xlabel == p1.fig.axes[0].get_xlabel() == '$x^{2}_{1}$'
    assert p2.xlabel == p2.fig.axes[0].get_xlabel() == 'x_1^2'
    assert p1.ylabel == p1.fig.axes[0].get_ylabel() == '$f\\left(x^{2}_{1}\\right)$'
    assert p2.ylabel == p2.fig.axes[0].get_ylabel() == 'f(x_1^2)'

    p1 = p(PB, True)
    p2 = p(PB, False)
    assert p1.xlabel == p1.fig.layout.xaxis.title.text == '$x^{2}_{1}$'
    assert p2.xlabel == p2.fig.layout.xaxis.title.text == 'x_1^2'
    assert p1.ylabel == p1.fig.layout.yaxis.title.text == '$f\\left(x^{2}_{1}\\right)$'
    assert p2.ylabel == p2.fig.layout.yaxis.title.text == 'f(x_1^2)'

    p1 = p(BB, True)
    p2 = p(BB, False)
    assert p1.xlabel == p1.fig.xaxis.axis_label == '$x^{2}_{1}$'
    assert p2.xlabel == p2.fig.xaxis.axis_label == 'x_1^2'
    assert p1.ylabel == p1.fig.yaxis.axis_label == '$f\\left(x^{2}_{1}\\right)$'
    assert p2.ylabel == p2.fig.yaxis.axis_label == 'f(x_1^2)'

    p = lambda B, use_latex, show=False: plot3d(
        cos(x1**2 + x2**2), (x1, -1, 1), (x2, -1, 1), backend=B,
        show=show, use_latex=use_latex, adaptive=False, n=5)

    p1 = p(MB, True)
    p2 = p(MB, False)
    assert p1.xlabel == p1.fig.axes[0].get_xlabel() == '$x^{2}_{1}$'
    assert p1.ylabel == p1.fig.axes[0].get_ylabel() == '$x_{2}$'
    assert p1.zlabel == p1.fig.axes[0].get_zlabel() == '$f\\left(x^{2}_{1}, x_{2}\\right)$'
    assert p2.xlabel == p2.fig.axes[0].get_xlabel() == 'x_1^2'
    assert p2.ylabel == p2.fig.axes[0].get_ylabel() == 'x_2'
    assert p2.zlabel == p2.fig.axes[0].get_zlabel() == 'f(x_1^2, x_2)'

    # Plotly currently doesn't support latex on 3D plots, hence it will fall
    # back to string representation.
    p1 = p(PB, True)
    p2 = p(PB, False)
    assert p1.xlabel == p1.fig.layout.scene.xaxis.title.text == '$x^{2}_{1}$'
    assert p1.ylabel == p1.fig.layout.scene.yaxis.title.text == '$x_{2}$'
    assert p1.zlabel == p1.fig.layout.scene.zaxis.title.text == '$f\\left(x^{2}_{1}, x_{2}\\right)$'
    assert p2.xlabel == p2.fig.layout.scene.xaxis.title.text == 'x_1^2'
    assert p2.ylabel == p2.fig.layout.scene.yaxis.title.text == 'x_2'
    assert p2.zlabel == p2.fig.layout.scene.zaxis.title.text == 'f(x_1^2, x_2)'

    p1 = p(KBchild1, True)
    p2 = p(KBchild1, False)
    assert p1.xlabel == p1.fig.axes[0] == 'x^{2}_{1}'
    assert p1.ylabel == p1.fig.axes[1] == 'x_{2}'
    assert p1.zlabel == p1.fig.axes[2] == 'f\\left(x^{2}_{1}, x_{2}\\right)'
    assert p2.xlabel == p2.fig.axes[0] == 'x_1^2'
    assert p2.ylabel == p2.fig.axes[1] == 'x_2'
    assert p2.zlabel == p2.fig.axes[2] == 'f(x_1^2, x_2)'

    if mayavi:
        p1 = p(MAB, True, show=True)
        p2 = p(MAB, False, show=True)
        o1 = p1.fig.children[0].children[0].children[0].children[3]
        xlabel1 = o1.axes.x_axis_label_text
        ylabel1 = o1.axes.y_axis_label_text
        zlabel1 = o1.axes.z_axis_label_text
        assert p1.xlabel == xlabel1 == '$x^{2}_{1}$'
        assert p1.ylabel == ylabel1 == '$x_{2}$'
        assert p1.zlabel == zlabel1 == '$f\\left(x^{2}_{1}, x_{2}\\right)$'
        o2 = p2.fig.children[0].children[0].children[0].children[3]
        xlabel2 = o2.axes.x_axis_label_text
        ylabel2 = o2.axes.y_axis_label_text
        zlabel2 = o2.axes.z_axis_label_text
        assert p2.xlabel == xlabel2 == 'x_1^2'
        assert p2.ylabel == ylabel2 == 'x_2'
        assert p2.zlabel == zlabel2 == 'f(x_1^2, x_2)'


def test_plot_use_latex():
    # verify that the backends produce the expected results when `plot()`
    # is called and `rendering_kw` overrides the default line settings

    x = symbols("x")

    _plot = lambda B: plot(
        sin(x), cos(x), backend=B, show=False, legend=True,
        use_latex=True, adaptive=False, n=5)

    p = _plot(MB)
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_label() == "$\\sin{\\left(x \\right)}$"
    assert ax.get_lines()[1].get_label() == "$\\cos{\\left(x \\right)}$"
    p.close()

    p = _plot(PB)
    f = p.fig
    assert f.data[0]["name"] == "$\\sin{\\left(x \\right)}$"
    assert f.data[1]["name"] == "$\\cos{\\left(x \\right)}$"
    assert f.layout["showlegend"] is True

    p = _plot(BB)
    f = p.fig
    assert f.legend[0].items[0].label["value"] == "$\\sin{\\left(x \\right)}$"
    assert f.legend[0].items[1].label["value"] == "$\\cos{\\left(x \\right)}$"
    assert f.legend[0].visible is True

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot(KBchild1).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot(MAB).process_series())


def test_plot_parametric_use_latex():
    # verify that the colorbar uses latex label

    x = symbols("x")

    _plot_parametric = lambda B: plot_parametric(
        cos(x), sin(x), (x, 0, 1.5 * pi), backend=B,
        show=False, use_latex=True, adaptive=False, n=5
    )

    p = _plot_parametric(MB)
    assert len(p.series) == 1
    f = p.fig
    assert f.axes[1].get_ylabel() == "$x$"
    p.close()

    p = _plot_parametric(PB)
    f = p.fig
    assert f.data[0]["name"] == "$x$"
    assert f.data[0]["marker"]["colorbar"]["title"]["text"] == "$x$"

    p = _plot_parametric(BB)
    f = p.fig
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "$x$"

    raises(
        NotImplementedError,
        lambda: _plot_parametric(KBchild1).process_series())

    if mayavi:
        raises(
            NotImplementedError,
            lambda: _plot_parametric(MAB).process_series())


def test_plot_contour_use_latex():
    # verify that the colorbar uses latex label
    x, y = symbols("x, y")

    _plot_contour = lambda B: plot_contour(
        cos(x ** 2 + y ** 2),
        (x, -3, 3),
        (y, -3, 3),
        n=5,
        backend=B,
        show=False,
        use_latex=True
    )

    p = _plot_contour(MB)
    assert len(p.series) == 1
    f = p.fig
    assert f.axes[1].get_ylabel() == "$%s$" % latex(cos(x ** 2 + y ** 2))

    p = _plot_contour(PB)
    f = p.fig
    assert f.data[0]["colorbar"]["title"]["text"] == "$%s$" % latex(cos(x ** 2 + y ** 2))

    p = _plot_contour(BB)
    f = p.fig
    assert f.right[0].title == "$%s$" % latex(cos(x ** 2 + y ** 2))


def test_plot3d_parametric_line_use_latex():
    # verify that the colorbar uses latex label

    x = symbols("x")

    _plot3d_parametric_line = lambda B, show=False: plot3d_parametric_line(
        cos(x), sin(x), x, (x, -pi, pi), backend=B,
        show=show, use_latex=True, adaptive=False, n=5
    )

    p = _plot3d_parametric_line(MB)
    assert len(p.series) == 1
    f = p.fig
    assert f.axes[1].get_ylabel() == "$x$"
    p.close()

    p = _plot3d_parametric_line(PB)
    f = p.fig
    assert f.data[0]["name"] == "$x$"
    assert f.data[0]["line"]["colorbar"]["title"]["text"] == "$x$"

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError, lambda: _plot3d_parametric_line(BB).process_series())

    # NOTE: K3D doesn't show a label to colorbar
    p = _plot3d_parametric_line(KBchild1)

    if mayavi:
        p = _plot3d_parametric_line(MAB, True)
        assert p.fig.children[0].children[0].children[0].children[0].scalar_lut_manager.scalar_bar.title == "$x$"


def test_plot3d_use_latex():
    # verify that the colorbar uses latex label

    x, y = symbols("x, y")

    _plot3d = lambda B, show=False: plot3d(
        cos(x ** 2 + y ** 2),
        sin(x ** 2 + y ** 2),
        (x, -3, 3),
        (y, -3, 3),
        n=5,
        use_cm=True,
        backend=B,
        show=show,
        use_latex=True
    )

    p = _plot3d(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(f.axes) == 3
    assert f.axes[1].get_ylabel() == "$%s$" % latex(cos(x ** 2 + y ** 2))
    assert f.axes[2].get_ylabel() == "$%s$" % latex(sin(x ** 2 + y ** 2))
    p.close()

    p = _plot3d(PB)
    f = p.fig
    assert f.data[0].colorbar.title.text == "$%s$" % latex(cos(x ** 2 + y ** 2))
    assert f.data[1].colorbar.title.text == "$%s$" % latex(sin(x ** 2 + y ** 2))
    assert f.data[0].name == "$%s$" % latex(cos(x ** 2 + y ** 2))
    assert f.data[1].name == "$%s$" % latex(sin(x ** 2 + y ** 2))
    assert f.data[0]["showscale"]
    assert f.layout["showlegend"]

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: _plot3d(BB).process_series())

    p = _plot3d(KBchild1)
    f = p.fig
    assert p.fig.axes == ["x", "y", "f\\left(x, y\\right)"]

    if mayavi:
        p = _plot3d(MAB, True)
        assert p.fig.children[0].children[0].children[0].scalar_lut_manager.scalar_bar.title == "$%s$" % latex(cos(x ** 2 + y ** 2))
        assert p.fig.children[1].children[0].children[0].scalar_lut_manager.scalar_bar.title == "$%s$" % latex(sin(x ** 2 + y ** 2))


def test_plot_vector_2d_quivers_use_latex():
    # verify that the colorbar uses latex label

    x, y = symbols("x, y")

    _plot_vector = lambda B: plot_vector(
        Matrix([x, y]),
        (x, -5, 5),
        (y, -4, 4),
        backend=B,
        show=False,
        n=5
    )

    p = _plot_vector(MB)
    f = p.fig
    assert f.axes[1].get_ylabel() == "Magnitude"
    p.close()

    p = _plot_vector(PB)
    f = p.fig
    assert f.data[0]["colorbar"]["title"]["text"] == "Magnitude"

    p = _plot_vector(BB)
    f = p.fig
    assert f.right[0].title == "Magnitude"

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_vector(KBchild1).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot_vector(MAB).process_series())


def test_plot_vector_2d_streamlines_custom_scalar_field_use_latex():
    # verify that the colorbar uses latex label

    x, y = symbols("x, y")

    _plot_vector = lambda B: plot_vector(
        Matrix([x, y]),
        (x, -5, 5),
        (y, -4, 4),
        backend=B,
        scalar=(x + y),
        streamlines=True,
        show=False,
        use_latex=True,
        n=5
    )

    p = _plot_vector(MB)
    f = p.fig
    assert f.axes[1].get_ylabel() == "$x + y$"
    p.close()

    p = _plot_vector(PB)
    f = p.fig
    assert f.data[0]["colorbar"]["title"]["text"] == "$x + y$"

    p = _plot_vector(BB)
    f = p.fig
    assert f.right[0].title == "$x + y$"

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_vector(KBchild1).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot_vector(MAB).process_series())


def test_plot_vector_2d_streamlines_custom_scalar_field_custom_label_use_latex():
    # verify that the colorbar uses latex label

    x, y = symbols("x, y")

    _plot_vector = lambda B: plot_vector(
        Matrix([x, y]),
        (x, -5, 5),
        (y, -4, 4),
        backend=B,
        scalar=[(x + y), "test"],
        streamlines=True,
        show=False,
        use_latex=True,
        n=5
    )

    p = _plot_vector(MB)
    f = p.fig
    assert f.axes[1].get_ylabel() == "test"

    p = _plot_vector(PB)
    f = p.fig
    assert f.data[0]["colorbar"]["title"]["text"] == "test"

    p = _plot_vector(BB)
    f = p.fig
    assert f.right[0].title == "test"

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: _plot_vector(KBchild1).process_series())

    if mayavi:
        # Mayavi doesn't support 2D plots
        raises(
            NotImplementedError,
            lambda: _plot_vector(MAB).process_series())


def test_plot_vector_2d_matplotlib_use_latex():
    # verify that the colorbar uses latex label

    x, y = symbols("x, y")
    _plot_vector_1 = lambda B, scalar, streamlines: plot_vector(
        Matrix([x, y]),
        (x, -5, 5),
        (y, -4, 4),
        backend=B,
        scalar=scalar,
        streamlines=streamlines,
        use_cm=True,
        show=False,
        use_latex=True,
        n=5)

    # contours + quivers: 1 colorbar for the contours
    p = _plot_vector_1(MB, True, False)
    assert p.fig.axes[1].get_ylabel() == "Magnitude"
    p.close()

    p = _plot_vector_1(PB, True, False)
    assert p.fig.data[0]["colorbar"]["title"]["text"] == "Magnitude"

    p = _plot_vector_1(BB, True, False)
    assert p.fig.right[0].title == "Magnitude"

    # contours + streamlines: 1 colorbar for the contours
    p = _plot_vector_1(MB, True, True)
    assert p.fig.axes[1].get_ylabel() == "Magnitude"
    p.close()

    p = _plot_vector_1(PB, True, True)
    assert p.fig.data[0]["colorbar"]["title"]["text"] == "Magnitude"

    p = _plot_vector_1(BB, True, True)
    assert p.fig.right[0].title == "Magnitude"

    # only quivers: 1 colorbar for the quivers
    p = _plot_vector_1(MB, False, False)
    assert p.fig.axes[1].get_ylabel() == "$\\left( x, \\  y\\right)$"
    p.close()

    p = _plot_vector_1(PB, False, False)
    assert p.fig.data[0]["name"] == "$\\left( x, \\  y\\right)$"

    p = _plot_vector_1(BB, False, False)
    assert p.fig.right[0].title == "$\\left( x, \\  y\\right)$"

    # only streamlines: 1 colorbar for the streamlines
    p = _plot_vector_1(MB, False, True)
    assert p.fig.axes[1].get_ylabel() == "$\\left( x, \\  y\\right)$"
    p.close()

    p = _plot_vector_1(PB, False, True)
    assert p.fig.data[0]["name"] == "$\\left( x, \\  y\\right)$"

    # Bokeh doesn't support gradient streamlines, hence no colorbar
    p = _plot_vector_1(BB, False, True)


def test_plot_vector_3d_quivers_use_latex():
    # verify that the colorbar uses latex label

    x, y, z = symbols("x, y, z")

    _plot_vector = lambda B, show=False: plot_vector(
    Matrix([z, y, x]),
        (x, -5, 5),
        (y, -4, 4),
        (z, -3, 3),
        backend=B,
        show=show,
        use_cm=True,
        use_latex=True,
        n=5
    )

    p = _plot_vector(MB)
    assert len(p.fig.axes) == 2
    assert p.fig.axes[1].get_ylabel() == '$\\left( z, \\  y, \\  x\\right)$'
    p.close()

    p = _plot_vector(PB)
    assert p.fig.data[0]["colorbar"]["title"]["text"] == '$\\left( z, \\  y, \\  x\\right)$'

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError,
        lambda: _plot_vector(BB).process_series())

    # K3D doesn't show label on colorbar
    p = _plot_vector(KBchild1)
    assert len(p.series) == 1

    if mayavi:
        p = _plot_vector(MAB, True)
        assert p.fig.children[0].children[0].vector_lut_manager.scalar_bar.title == '$\\left( z, \\  y, \\  x\\right)$'


def test_plot_vector_3d_streamlines_use_latex():
    # verify that the colorbar uses latex label

    x, y, z = symbols("x, y, z")

    _plot_vector = lambda B, show=False: plot_vector(
        Matrix([z, y, x]),
        (x, -5, 5),
        (y, -4, 4),
        (z, -3, 3),
        backend=B,
        streamlines=True,
        show=show,
        use_latex=True,
        n=5
    )

    p = _plot_vector(MB)
    assert p.fig.axes[1].get_ylabel() == '$\\left( z, \\  y, \\  x\\right)$'
    p.close()

    p = _plot_vector(PB)
    assert p.fig.data[0]["colorbar"]["title"]["text"] == '$\\left( z, \\  y, \\  x\\right)$'

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: _plot_vector(BB).process_series())

    # K3D doesn't show labels on colorbar
    p = _plot_vector(KBchild1)
    assert len(p.series) == 1

    if mayavi:
        p = _plot_vector(MAB, True)
        assert p.fig.children[0].children[0].children[0].scalar_lut_manager.scalar_bar.title == '$\\left( z, \\  y, \\  x\\right)$'


def test_plot_complex_use_latex():
    # complex plot function should return the same result (for axis labels)
    # wheter use_latex is True or False

    x, y, z = symbols("x, y, z")

    _plot_complex = lambda B: plot_complex(
        cos(x) + sin(I * x), (x, -2, 2), show=False, adaptive=False, n=5,
        use_latex=True, backend=B)

    p = _plot_complex(MB)
    assert p.fig.axes[0].get_xlabel() == "Real"
    assert p.fig.axes[0].get_ylabel() == "Abs"
    assert p.fig.axes[1].get_ylabel() == 'Arg(cos(x) + I*sinh(x))'
    p.close()

    p = _plot_complex(PB)
    assert p.fig.layout.xaxis.title.text == 'Real'
    assert p.fig.layout.yaxis.title.text == 'Abs'
    assert p.fig.data[0].name == "Arg(cos(x) + I*sinh(x))"
    assert p.fig.data[0]["marker"]["colorbar"]["title"]["text"] == 'Arg(cos(x) + I*sinh(x))'

    p = _plot_complex(BB)
    assert p.fig.right[0].title == 'Arg(cos(x) + I*sinh(x))'
    assert p.fig.xaxis.axis_label == "Real"
    assert p.fig.yaxis.axis_label == "Abs"

    raises(
        NotImplementedError,
        lambda: _plot_complex(KBchild1).process_series())

    if mayavi:
        raises(
            NotImplementedError,
            lambda: _plot_complex(MAB).process_series())

    _plot_complex_2 = lambda B: plot_complex(
        gamma(z), (z, -3 - 3*I, 3 + 3*I), show=False, adaptive=False, n=5,
        use_latex=True, backend=B)

    p = _plot_complex_2(MB)
    assert p.fig.axes[0].get_xlabel() == "Re"
    assert p.fig.axes[0].get_ylabel() == "Im"
    assert p.fig.axes[1].get_ylabel() == 'Argument'
    p.close()

    p = _plot_complex_2(PB)
    assert p.fig.layout.xaxis.title.text == 'Re'
    assert p.fig.layout.yaxis.title.text == 'Im'
    assert p.fig.data[0].name == "$gamma(z)$"
    assert p.fig.data[1]["marker"]["colorbar"]["title"]["text"] == "Argument"

    p = _plot_complex_2(BB)
    assert p.fig.right[0].title == 'Argument'
    assert p.fig.xaxis.axis_label == "Re"
    assert p.fig.yaxis.axis_label == "Im"

    raises(
        NotImplementedError,
        lambda: _plot_complex_2(KBchild1).process_series())

    if mayavi:
        raises(
            NotImplementedError,
            lambda: _plot_complex_2(MAB).process_series())


def test_plot_real_imag_use_latex():
    # real/imag plot function should return the same result (for axis labels)
    # wheter use_latex is True or False

    x, y, z = symbols("x, y, z")

    _plot_real_imag = lambda B: plot_real_imag(sqrt(x), (x, -3, 3),
        backend=B, use_latex=True, show=False, adaptive=False, n=5)

    p = _plot_real_imag(MB)
    assert p.fig.axes[0].get_xlabel() == "$x$"
    assert p.fig.axes[0].get_ylabel() == r"$f\left(x\right)$"
    assert p.fig.axes[0].lines[0].get_label() == 'Re(sqrt(x))'
    assert p.fig.axes[0].lines[1].get_label() == 'Im(sqrt(x))'
    p.close()

    p = _plot_real_imag(PB)
    assert p.fig.layout.xaxis.title.text == "$x$"
    assert p.fig.layout.yaxis.title.text == r"$f\left(x\right)$"
    assert p.fig.data[0]["name"] == 'Re(sqrt(x))'
    assert p.fig.data[1]["name"] == 'Im(sqrt(x))'

    p = _plot_real_imag(BB)
    assert p.fig.xaxis.axis_label == "$x$"
    assert p.fig.yaxis.axis_label == r"$f\left(x\right)$"
    assert p.fig.legend[0].items[0].label["value"] == 'Re(sqrt(x))'
    assert p.fig.legend[0].items[1].label["value"] == 'Im(sqrt(x))'

    raises(
        NotImplementedError,
        lambda: _plot_real_imag(KBchild1).process_series())

    if mayavi:
        raises(
            NotImplementedError,
            lambda: _plot_real_imag(MAB).process_series())


##############################################################################
##############################################################################
##############################################################################


def test_plot3d_use_cm():
    # verify that use_cm produces the expected results on plot3d

    x, y = symbols("x, y")
    p1 = plot3d(cos(x**2 + y**2), (x, -1, 1), (y, -1, 1), backend=MB, show=False, use_cm=True, n=5)
    p2 = plot3d(cos(x**2 + y**2), (x, -1, 1), (y, -1, 1), backend=MB, show=False, use_cm=False, n=5)
    p1.process_series()
    p2.process_series()
    assert "cmap" in p1._handles[0][1].keys()
    assert "cmap" not in p2._handles[0][1].keys()

    p1 = plot3d(cos(x**2 + y**2), (x, -1, 1), (y, -1, 1), backend=PB, show=False, use_cm=True, n=5)
    p2 = plot3d(cos(x**2 + y**2), (x, -1, 1), (y, -1, 1), backend=PB, show=False, use_cm=False, n=5)
    assert len(p1.fig.data[0]["colorscale"]) > 2
    assert len(p2.fig.data[0]["colorscale"]) == 2

    p1 = plot3d(cos(x**2 + y**2), (x, -1, 1), (y, -1, 1), backend=KBchild1, show=False, use_cm=True, n=5)
    p2 = plot3d(cos(x**2 + y**2), (x, -1, 1), (y, -1, 1), backend=KBchild1, show=False, use_cm=False, n=5)
    n1 = len(p1.fig.objects[0].color_map)
    n2 = len(p2.fig.objects[0].color_map)
    if n1 == n2:
        assert not np.allclose(p1.fig.objects[0].color_map, p2.fig.objects[0].color_map)

    if mayavi:
        p1 = plot3d(cos(x**2 + y**2), (x, -1, 1), (y, -1, 1), backend=MAB, show=True, use_cm=True, n=5)
        p2 = plot3d(cos(x**2 + y**2), (x, -1, 1), (y, -1, 1), backend=MAB, show=True, use_cm=False, n=5)
        assert np.allclose(
            p1.fig.children[0].children[0].children[0].children[0].actor.property.color,
            (1, 1, 1)
        )
        assert np.allclose(
            p2.fig.children[0].children[0].children[0].children[0].actor.property.color,
            (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
        )


def test_plot3d_update_interactive():
    # verify that MB._update_interactive applies the original color/colormap
    # each time it gets called
    # Since matplotlib doesn't apply color/colormaps until the figure is shown,
    # verify that the correct keyword arguments are into _handles.

    x, y, u = symbols("x, y, u")

    s = InteractiveSeries(
        [u * cos(x**2 + y**2)],
        [(x, -5, 5), (y, -5, 5)],
        "test",
        threed = True,
        use_cm = False,
        params = {u: 1},
        n1=3, n2=3, n3=3
    )
    p = MB(s, show=False)
    p.process_series()
    kw, _, _ = p._handles[0][1:]
    c1 = kw["color"]
    p._update_interactive({u: 2})
    kw, _, _ = p._handles[0][1:]
    c2 = kw["color"]
    assert c1 == c2

    s = InteractiveSeries(
        [u * cos(x**2 + y**2)],
        [(x, -5, 5), (y, -5, 5)],
        "test",
        threed = True,
        use_cm = True,
        params = {u: 1},
        n1=3, n2=3, n3=3
    )
    p = MB(s, show=False)
    p.process_series()
    kw, _, _ = p._handles[0][1:]
    c1 = kw["cmap"]
    p._update_interactive({u: 2})
    kw, _, _ = p._handles[0][1:]
    c2 = kw["cmap"]
    assert c1 == c2


def test_k3d_vector_pivot():
    # verify that K3DBackend accepts quiver_kw={"pivot": "something"} and
    # produces different results
    x, y, z = symbols("x, y, z")

    _plot_vector = lambda pivot: plot_vector(
        Matrix([z, y, x]),
        (x, -5, 5),
        (y, -4, 4),
        (z, -3, 3),
        n1=3, n2=3, n3=3,
        quiver_kw={"pivot": pivot},
        backend=KBchild1,
        show=False
    )

    p1 = _plot_vector("tail")
    p2 = _plot_vector("mid")
    p3 = _plot_vector("tip")
    assert not np.allclose(p1.fig.objects[0].origins, p2.fig.objects[0].origins, equal_nan=True)
    assert not np.allclose(p1.fig.objects[0].origins, p3.fig.objects[0].origins, equal_nan=True)
    assert not np.allclose(p2.fig.objects[0].origins, p3.fig.objects[0].origins, equal_nan=True)


def test_plot_polar():
    # verify that 2D polar plot uses polar projection
    x = symbols("x")
    _plot_polar = lambda B: plot_polar(1 + sin(10 * x) / 10, (x, 0, 2 * pi),
        backend=B, aspect="equal", show=False, adaptive=False, n=5)

    p = _plot_polar(MB)
    fig = p.fig
    assert isinstance(fig.axes[0], matplotlib.projections.polar.PolarAxes)

    p1 = _plot_polar(PB)
    assert isinstance(p1.fig.data[0], go.Scatterpolar)

    # Bokeh doesn't have polar projection. Here we check that the backend
    # transforms the data.
    p2 = _plot_polar(BB)
    plotly_data = p1[0].get_data()
    bokeh_data = p2.fig.renderers[0].data_source.data
    assert not np.allclose(plotly_data[0], bokeh_data["xs"])
    assert not np.allclose(plotly_data[1], bokeh_data["ys"])


def test_plot3d_implicit():
    x, y, z = symbols("x:z")

    _plot3d_implicit = lambda B, show=False: plot3d_implicit(
        x**2 + y**3 - z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2),
        backend=B, n1=5, n2=5, n3=5, show=show)

    raises(NotImplementedError, lambda : _plot3d_implicit(MB).process_series())

    raises(NotImplementedError, lambda : _plot3d_implicit(BB).process_series())

    p = _plot3d_implicit(PB)
    assert isinstance(p.fig.data[0], go.Isosurface)

    p = _plot3d_implicit(KBchild1)
    assert isinstance(p.fig.objects[0], k3d.objects.MarchingCubes)

    if mayavi:
        p = _plot3d_implicit(MAB, True)
        assert len(p.fig.children) > 0


def test_surface_color_func():
    # After the addition of `color_func`, `SurfaceOver2DRangeSeries` and
    # `ParametricSurfaceSeries` returns different elements.
    # Verify that backends do not raise errors when plotting surfaces and that
    # the color function is applied.

    x, y, z, u, v = symbols("x:z, u, v")
    p3d = lambda B, col, show=False: plot3d(
        cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        backend=B, n=5, color_func=col, show=show, use_cm=True)

    p1 = p3d(MB, lambda x, y, z: z).process_series()
    p2 = p3d(MB, lambda x, y, z: np.sqrt(x**2 + y**2)).process_series()

    p1 = p3d(PB, lambda x, y, z: z)
    p2 = p3d(PB, lambda x, y, z: np.sqrt(x**2 + y**2))
    assert not np.allclose(p1.fig.data[0]["surfacecolor"], p2.fig.data[0]["surfacecolor"])

    p1 = p3d(KBchild1, lambda x, y, z: z)
    p2 = p3d(KBchild1, lambda x, y, z: np.sqrt(x**2 + y**2))
    assert not np.allclose(p1.fig.objects[0].attribute, p2.fig.objects[0].attribute)

    if mayavi:
        p1 = p3d(MAB, lambda x, y, z: z, show=True)
        p2 = p3d(MAB, lambda x, y, z: np.sqrt(x**2 + y**2), show=True)
        # NOTE: no idea where Mayavi stores the entire scalar field, just the min
        # and max
        assert np.allclose(
            p1.fig.children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
            [-1,  1],
            rtol=1e-01
        )
        assert np.allclose(
            p2.fig.children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
            [0, 4.24264069],
            rtol=1e-01
        )

    r = 2 + sin(7 * u + 5 * v)
    expr = (r * cos(u) * sin(v), r * sin(u) * sin(v), r * cos(v))
    p3dps = lambda B, col, show=False: plot3d_parametric_surface(
        *expr, (u, 0, 2 * pi), (v, 0, pi), show=show, use_cm=True, n=5,
        backend=B, color_func=col)

    p1 = p3dps(MB, lambda x, y, z, u, v: z).process_series()
    p2 = p3dps(MB, lambda x, y, z, u, v: np.sqrt(x**2 + y**2)).process_series()

    p1 = p3dps(PB, lambda x, y, z, u, v: z)
    p2 = p3dps(PB, lambda x, y, z, u, v: np.sqrt(x**2 + y**2))
    assert not np.allclose(p1.fig.data[0]["surfacecolor"], p2.fig.data[0]["surfacecolor"])

    p1 = p3dps(KBchild1, lambda x, y, z, u, v: z)
    p2 = p3dps(KBchild1, lambda x, y, z, u, v: np.sqrt(x**2 + y**2))
    assert not np.allclose(p1.fig.objects[0].attribute, p2.fig.objects[0].attribute)

    if mayavi:
        p1 = p3dps(MAB, lambda x, y, z, u, v: z, show=True)
        p2 = p3dps(MAB, lambda x, y, z, u, v: np.sqrt(x**2 + y**2), show=True)
        # NOTE: no idea where Mayavi stores the entire scalar field, just the min
        # and max
        assert np.allclose(
            p1.fig.children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
            [-3,  3],
            rtol=1e-01
        )
        assert np.allclose(
            p2.fig.children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
            [0, 3.],
            rtol=1e-01
        )


def test_surface_interactive_color_func():
    # After the addition of `color_func`, `SurfaceInteractiveSeries` and
    # `ParametricSurfaceInteractiveSeries` returns different elements.
    # Verify that backends do not raise errors when updating surfaces and a
    # color function is applied.

    x, y, z, t, u, v = symbols("x:z, u, v, t")

    expr1 = t * cos(x**2 + y**2)
    r = 2 + sin(7 * u + 5 * v)
    expr2 = (t * r * cos(u) * sin(v), r * sin(u) * sin(v), r * cos(v))

    s1 = InteractiveSeries([expr1], [(x, -5, 5), (y, -5, 5)],
        n1=5, n2=5, params={t: 1}, use_cm=True,
        color_func=lambda x, y, z: z, threed=True)
    s2 = InteractiveSeries([expr1], [(x, -5, 5), (y, -5, 5)],
        n1=5, n2=5, params={t: 1}, use_cm=True,
        color_func=lambda x, y, z: np.sqrt(x**2 + y**2), threed=True)
    s3 = InteractiveSeries([*expr2], [(u, -5, 5), (v, -5, 5)],
        n1=5, n2=5, params={t: 1}, use_cm=True,
        color_func=lambda x, y, z, u, v: z)
    s4 = InteractiveSeries([*expr2], [(u, -5, 5), (v, -5, 5)],
        n1=5, n2=5, params={t: 1}, use_cm=True,
        color_func=lambda x, y, z, u, v: np.sqrt(x**2 + y**2))

    p = MB(s1, s2, s3, s4)
    p.process_series()
    p._update_interactive({t: 2})

    p = PB(s1, s2, s3, s4)
    p._update_interactive({t: 2})
    assert not np.allclose(p.fig.data[0]["surfacecolor"], p.fig.data[1]["surfacecolor"])
    assert not np.allclose(p.fig.data[2]["surfacecolor"], p.fig.data[3]["surfacecolor"])

    p = KBchild1(s1, s2, s3, s4)
    p._update_interactive({t: 2})
    assert not np.allclose(p.fig.objects[0].attribute, p.fig.objects[1].attribute)
    assert not np.allclose(p.fig.objects[2].attribute, p.fig.objects[3].attribute)


def test_line_color_func():
    # Verify that backends do not raise errors when plotting lines and that
    # the color function is applied.

    x, u = symbols("x, u")
    pl = lambda B, col: plot(cos(x), (x, -3, 3),
        backend=B, adaptive=False, n=5, color_func=col, show=False, legend=True)

    p1 = pl(MB, None)
    p1.process_series()
    p2 = pl(MB, lambda x, y: np.cos(x))
    p2.process_series()
    assert len(p1.fig.axes[0].lines) == 1
    assert isinstance(p2.fig.axes[0].collections[0], matplotlib.collections.LineCollection)
    assert np.allclose(p2.fig.axes[0].collections[0].get_array(), np.cos(np.linspace(-3, 3, 5)))


    p1 = pl(PB, None)
    p2 = pl(PB, lambda x, y: np.cos(x))
    assert p1.fig.data[0].marker.color is None
    assert np.allclose(p2.fig.data[0].marker.color, np.cos(np.linspace(-3, 3, 5)))

    p1 = pl(BB, None)
    p2 = pl(BB, lambda x, y: np.cos(x))
    assert isinstance(p1.fig.renderers[0].glyph, bokeh.models.glyphs.Line)
    assert isinstance(p2.fig.renderers[0].glyph, bokeh.models.glyphs.MultiLine)


def test_line_interactive_color_func():
    # Verify that backends do not raise errors when updating lines and a
    # color function is applied.

    x, t = symbols("x, t")

    expr = t * cos(x * t)
    s1 = InteractiveSeries([expr], [(x, -3, 3)],
        n1=5, params={t: 1}, color_func=None)
    s2 = InteractiveSeries([expr], [(x, -3, 3)],
        n1=5, params={t: 1}, color_func=lambda x, y: np.cos(x))

    p = MB(s1, s2)
    p.process_series()
    p._update_interactive({t: 2})
    assert len(p.fig.axes[0].lines) == 1
    assert isinstance(p.fig.axes[0].collections[0], matplotlib.collections.LineCollection)
    assert np.allclose(p.fig.axes[0].collections[0].get_array(), np.cos(np.linspace(-3, 3, 5)))

    p = PB(s1, s2)
    p._update_interactive({t: 2})
    assert p.fig.data[0].marker.color is None
    assert np.allclose(p.fig.data[1].marker.color, np.cos(np.linspace(-3, 3, 5)))

    p = BB(s1, s2)
    p._update_interactive({t: 2})
    assert isinstance(p.fig.renderers[0].glyph, bokeh.models.glyphs.Line)
    assert isinstance(p.fig.renderers[1].glyph, bokeh.models.glyphs.MultiLine)


def test_line_color_plot():
    # verify back-compatibility with old sympy.plotting module when using
    # line_color

    x, y = symbols("x, y")

    _plot = lambda B, lc: plot(sin(x), adaptive=False, n=5,
        line_color=lc, backend=B, show=False)

    p = _plot(MB, "red")
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_color() == "red"
    p = _plot(MB, lambda x: -x)
    f = p.fig
    assert len(p.fig.axes) == 2 # there is a colorbar

    p = _plot(PB, "red")
    assert p.fig.data[0]["line"]["color"] == "red"
    p = _plot(PB, lambda x: -x)
    assert p.fig.data[0].marker.showscale

    p = _plot(BB, "red")
    assert p.fig.renderers[0].glyph.line_color == "red"
    p = _plot(BB, lambda x: -x)
    assert len(p.fig.right) == 1
    assert p.fig.right[0].title == "sin(x)"


def test_line_color_plot3d_parametric_line():
    # verify back-compatibility with old sympy.plotting module when using
    # line_color

    x, y = symbols("x, y")

    _plot = lambda B, lc, use_cm, show=False: plot3d_parametric_line(
        cos(x), sin(x), x, (x, 0, 2*pi), adaptive=False, n=5,
        line_color=lc, backend=B, show=show, use_cm=use_cm)

    p = _plot(MB, "red", False)
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_color() == "red"
    p = _plot(MB, lambda x: -x, True)
    f = p.fig
    assert len(p.fig.axes) == 2 # there is a colorbar

    p = _plot(PB, "red", False)
    assert p.fig.data[0].line.color == "red"
    p = _plot(PB, lambda x: -x, True)
    assert p.fig.data[0].line.showscale

    p = _plot(KBchild1, 0xff0000, False)
    assert p.fig.objects[0].color == 16711680
    p = _plot(KBchild1, lambda x: -x, True)
    assert len(p.fig.objects[0].attribute) > 0

    if mayavi:
        p = _plot(MAB, (1, 0, 0), False, True)
        p._handles[0].actor.property.color == (1, 0, 0)
        p = _plot(MAB, lambda x: -x, True, True)
        assert np.allclose(
            p.fig.children[0].children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
            [-2*np.pi, 0],
            rtol=1e-02
        )


def test_surface_color_plot3d():
    # verify back-compatibility with old sympy.plotting module when using
    # surface_color

    x, y = symbols("x, y")
    _plot = lambda B, sc, use_cm, show=False: plot3d(
        cos(x**2 + y**2), (x, 0, 2), (y, 0, 2), adaptive=False, n=5,
        surface_color=sc, backend=B, show=show, use_cm=use_cm, legend=True)

    p = _plot(MB, "red", False)
    assert len(p.fig.axes) == 1
    p = _plot(MB, lambda x, y, z: -x, True)
    assert len(p.fig.axes) == 2

    p = _plot(PB, "red", False)
    assert p.fig.data[0].colorscale == ((0, 'red'), (1, 'red'))
    p = _plot(PB, lambda x, y, z: -x, True)
    assert len(p.fig.data[0].colorscale) > 2

    p = _plot(KBchild1, 0xff0000, False)
    assert p.fig.objects[0].color == 16711680
    p = _plot(KBchild1, lambda x, y, z: -x, True)
    assert len(p.fig.objects[0].attribute) > 0

    if mayavi:
        p = _plot(MAB, (1, 0, 0), False, True)
        p._handles[0].actor.property.color == (1, 0, 0)
        p = _plot(MAB, lambda x: -x, True, True)
        assert np.allclose(
            p.fig.children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
            [-1, 1],
            rtol=1e-02
        )


def test_label_after_plot_instantiation():
    # verify that it is possible to set a label after a plot has been created
    x = symbols("x")

    p = plot(sin(x), cos(x), show=False, backend=MB, adaptive=False, n=5)
    p[0].label = "a"
    p[1].label = "$b^{2}$"
    f = p.fig
    assert f.axes[0].lines[0].get_label() == "a"
    assert f.axes[0].lines[1].get_label() == "$b^{2}$"


def test_plotly_3d_many_line_series():
    # verify that plotly is capable of dealing with many 3D lines.

    t = symbols("t")

    # in this example there are 31 lines. 30 of them use solid colors, the
    # last one uses a colormap. No errors should be raised.
    p = plot3d_revolution(cos(t), (t, 0, pi), backend=PB, n=5, show=False,
        wireframe=True, wf_n1=15, wf_n2=15,
        show_curve=True, curve_kw={"use_cm": True})
    f = p.fig


def test_k3d_high_aspect_ratio_meshes():
    # K3D is not great at dealing with high aspect ratio meshes. So, users
    # should set zlim and the backend should add clipping planes and modify
    # the camera position.

    z = symbols("z")
    p1 = plot_complex(1 / sin(pi + z**3), (z, -2-2j, 2+2j),
        grid=False, threed=True, use_cm=True, backend=KBchild1, coloring="a",
        n=5, show=False)
    p1.process_series()
    p2 = plot_complex(1 / sin(pi + z**3), (z, -2-2j, 2+2j),
        grid=False, threed=True, use_cm=True, backend=KBchild1, coloring="a",
        n=5, zlim=(0, 6), show=False)
    p2.process_series()

    assert p1._bounds != p2._bounds
    assert p1.fig.camera != p2.fig.camera
    assert len(p1.fig.clipping_planes) == 0
    assert p1.fig.clipping_planes != p2.fig.clipping_planes


def test_k3d_update_interactive():
    # quick round down of test to verify that _update_interactive doesn't
    # raise errors

    u, v, x, y, z = symbols("u, v, x:z")

    # points
    p = plot3d_parametric_line(
        cos(u * x), sin(x), x, (x, -pi, pi), backend=KBchild1, is_point=True,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    # line
    p = plot3d_parametric_line(
        cos(u * x), sin(x), x, (x, -pi, pi), backend=KBchild1, is_point=False,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot3d(cos(u * x**2 + y**2), (x, -2, 2), (y, -2, 2), backend=KBchild1,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    u, v = symbols("u, v")
    fx = (1 + v / 2 * cos(u / 2)) * cos(x * u)
    fy = (1 + v / 2 * cos(u / 2)) * sin(x * u)
    fz = v / 2 * sin(u / 2)
    p = plot3d_parametric_surface(fx, fy, fz, (u, 0, 2*pi), (v, -1, 1),
        backend=KBchild1, use_cm=True, n1=5, n2=5, show=False,
        params={x: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({x: 2})

    p = plot_vector(Matrix([u * z, y, x]), (x, -5, 5), (y, -4, 4), (z, -3, 3),
        backend=KBchild1, n=4, show=False, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_complex(sqrt(u * x), (x, -5 - 5 * I, 5 + 5 * I), show=False,
        backend=KBchild1, threed=True, use_cm=True, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})


def test_plotly_update_interactive():
    # quick round down of test to verify that _update_interactive doesn't
    # raise errors

    u, v, x, y, z = symbols("u, v, x:z")

    p = plot(sin(u * x), (x, -pi, pi), adaptive=False, n=5,
        backend=PB, show=False, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_polar(1 + sin(10 * u * x) / 10, (x, 0, 2 * pi),
        adaptive=False, n=5, backend=PB, show=False, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_parametric(cos(u * x), sin(u * x), (x, 0, 2*pi), adaptive=False,
        n=5, backend=PB, show=False, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    # points
    p = plot3d_parametric_line(
        cos(u * x), sin(x), x, (x, -pi, pi), backend=PB, is_point=True,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    # line
    p = plot3d_parametric_line(
        cos(u * x), sin(x), x, (x, -pi, pi), backend=PB, is_point=False,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)}, use_cm=False)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot3d_parametric_line(
        cos(u * x), sin(x), x, (x, -pi, pi), backend=PB, is_point=False,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)}, use_cm=True)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot3d(cos(u * x**2 + y**2), (x, -2, 2), (y, -2, 2), backend=PB,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_contour(cos(u * x**2 + y**2), (x, -2, 2), (y, -2, 2), backend=PB,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    u, v = symbols("u, v")
    fx = (1 + v / 2 * cos(u / 2)) * cos(x * u)
    fy = (1 + v / 2 * cos(u / 2)) * sin(x * u)
    fz = v / 2 * sin(u / 2)
    p = plot3d_parametric_surface(fx, fy, fz, (u, 0, 2*pi), (v, -1, 1),
        backend=PB, use_cm=True, n1=5, n2=5, show=False,
        params={x: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({x: 2})

    p = plot_vector(Matrix([-u * y, x]), (x, -5, 5), (y, -4, 4),
        backend=PB, n=4, show=False, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_vector(Matrix([u * z, y, x]), (x, -5, 5), (y, -4, 4), (z, -3, 3),
        backend=PB, n=4, show=False, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_complex(sqrt(u * x), (x, -5 - 5 * I, 5 + 5 * I), show=False,
        backend=PB, threed=True, use_cm=True, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    from sympy.geometry import Line as SymPyLine
    p = plot_geometry(
        SymPyLine((u, 2), (5, 4)), Circle((0, 0), u), Polygon((2, u), 3, n=6),
        backend=PB, show=False, is_filled=False, use_latex=False,
        params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_geometry(
        SymPyLine((u, 2), (5, 4)), Circle((0, 0), u), Polygon((2, u), 3, n=6),
        backend=PB, show=False, is_filled=True, use_latex=False,
        params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})


def test_matpotlib_update_interactive():
    # quick round down of test to verify that _update_interactive doesn't
    # raise errors

    u, v, x, y, z = symbols("u, v, x:z")

    p = plot(sin(u * x), (x, -pi, pi), adaptive=False, n=5,
        backend=MB, show=False, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_parametric(cos(u * x), sin(u * x), (x, 0, 2*pi), adaptive=False,
        n=5, backend=MB, show=False, params={u: (1, 0, 2)},
        use_cm=True, is_point=False)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_parametric(cos(u * x), sin(u * x), (x, 0, 2*pi), adaptive=False,
        n=5, backend=MB, show=False, params={u: (1, 0, 2)},
        use_cm=True, is_point=True)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_parametric(cos(u * x), sin(u * x), (x, 0, 2*pi), adaptive=False,
        n=5, backend=MB, show=False, params={u: (1, 0, 2)}, use_cm=False)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_implicit(x**2 + y**2 - 4, (x, -5, 5), (y, -5, 5), adaptive=False,
        n=5, show=False, backend=MB)

    # points
    p = plot3d_parametric_line(
        cos(u * x), sin(x), x, (x, -pi, pi), backend=MB, is_point=True,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    # line with colormap
    p = plot3d_parametric_line(
        cos(u * x), sin(x), x, (x, -pi, pi), backend=MB, is_point=False,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)}, use_cm=True)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    # line with solid color
    p = plot3d_parametric_line(
        cos(u * x), sin(x), x, (x, -pi, pi), backend=MB, is_point=False,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)}, use_cm=False)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot3d_parametric_line(
        cos(u * x), sin(x), x, (x, -pi, pi), backend=MB, is_point=False,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)}, use_cm=True)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot3d(cos(u * x**2 + y**2), (x, -2, 2), (y, -2, 2), backend=MB,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_contour(cos(u * x**2 + y**2), (x, -2, 2), (y, -2, 2), backend=MB,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)}, is_filled=False)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_contour(cos(u * x**2 + y**2), (x, -2, 2), (y, -2, 2), backend=MB,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)}, is_filled=True)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    u, v = symbols("u, v")
    fx = (1 + v / 2 * cos(u / 2)) * cos(x * u)
    fy = (1 + v / 2 * cos(u / 2)) * sin(x * u)
    fz = v / 2 * sin(u / 2)
    p = plot3d_parametric_surface(fx, fy, fz, (u, 0, 2*pi), (v, -1, 1),
        backend=MB, use_cm=True, n1=5, n2=5, show=False,
        params={x: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({x: 2})

    p = plot_vector(Matrix([-u * y, x]), (x, -5, 5), (y, -4, 4),
        backend=MB, n=4, show=False, params={u: (1, 0, 2)}, scalar=True)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_vector(Matrix([-u * y, x]), (x, -5, 5), (y, -4, 4),
        backend=MB, n=4, show=False, params={u: (1, 0, 2)}, scalar=False)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_vector(Matrix([u * z, y, x]), (x, -5, 5), (y, -4, 4), (z, -3, 3),
        backend=MB, n=4, show=False, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_complex(sqrt(u * x), (x, -5 - 5 * I, 5 + 5 * I), show=False,
        backend=MB, threed=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_complex(sqrt(u * x), (x, -5 - 5 * I, 5 + 5 * I), show=False,
        backend=MB, threed=True, use_cm=True, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    from sympy.geometry import Line as SymPyLine
    p = plot_geometry(
        SymPyLine((u, 2), (5, 4)), Circle((0, 0), u), Polygon((2, u), 3, n=6),
        backend=MB, show=False, is_filled=False, use_latex=False,
        params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_geometry(
        SymPyLine((u, 2), (5, 4)), Circle((0, 0), u), Polygon((2, u), 3, n=6),
        backend=MB, show=False, is_filled=True, use_latex=False,
        params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})


def test_bokeh_update_interactive():
    # quick round down of test to verify that _update_interactive doesn't
    # raise errors

    u, v, x, y, z = symbols("u, v, x:z")

    p = plot(sin(u * x), (x, -pi, pi), adaptive=False, n=5,
        backend=BB, show=False, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_parametric(cos(u * x), sin(u * x), (x, 0, 2*pi), adaptive=False,
        n=5, backend=BB, show=False, params={u: (1, 0, 2)},
        use_cm=True, is_point=False)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_parametric(cos(u * x), sin(u * x), (x, 0, 2*pi), adaptive=False,
        n=5, backend=BB, show=False, params={u: (1, 0, 2)},
        use_cm=True, is_point=True)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_parametric(cos(u * x), sin(u * x), (x, 0, 2*pi), adaptive=False,
        n=5, backend=BB, show=False, params={u: (1, 0, 2)}, use_cm=False)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_contour(cos(u * x**2 + y**2), (x, -2, 2), (y, -2, 2), backend=BB,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_vector(Matrix([-u * y, x]), (x, -5, 5), (y, -4, 4),
        backend=BB, n=4, show=False, params={u: (1, 0, 2)}, streamlines=True)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_vector(Matrix([-u * y, x]), (x, -5, 5), (y, -4, 4),
        backend=BB, n=4, show=False, params={u: (1, 0, 2)}, streamlines=False,
        scalar=True)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_vector(Matrix([-u * y, x]), (x, -5, 5), (y, -4, 4),
        backend=BB, n=4, show=False, params={u: (1, 0, 2)}, streamlines=False,
        scalar=False)
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    p = plot_complex(sqrt(u * x), (x, -5 - 5 * I, 5 + 5 * I), show=False,
        backend=BB, threed=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})

    from sympy.geometry import Line as SymPyLine
    p = plot_geometry(
        Polygon((2, u), 3, n=6),
        backend=BB, show=False, is_filled=True, use_latex=False,
        params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend._update_interactive({u: 2})


def test_generic_data_series():
    # verify that backends do not raise errors when generic data series
    # are used

    x = symbols("x")
    p = plot(x, backend=MB, show=False, adaptive=False, n=5,
        markers=[{"args":[[0, 1], [0, 1]], "marker": "*", "linestyle": "none"}],
        annotations=[{"text": "test", "xy": (0, 0)}],
        fill=[{"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3]}],
        rectangles=[{"xy": (0, 0), "width": 5, "height": 1}])
    p.process_series()

    from bokeh.models import ColumnDataSource
    source = ColumnDataSource(data=dict(x=[0], y=[0], text=["test"]))
    p = plot(x, backend=BB, show=False, adaptive=False, n=5,
        markers=[{"x": [0, 1], "y": [0, 1], "marker": "square"}],
        annotations=[{"x": "x", "y": "y", "source": source}],
        fill=[{"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3], "y2": [0, 0, 0, 0]}],
        rectangles=[{"x": 0, "y": -3, "width": 5, "height": 2}])
    p.process_series()

    p = plot(x, backend=PB, show=False, adaptive=False, n=5,
        markers=[{"x": [0, 1], "y": [0, 1], "mode": "markers"}],
        annotations=[{"x": [0, 1], "y": [0, 1], "text": ["a", "b"]}],
        fill=[{"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "fill": "tozeroy"}],
        rectangles=[{"type": "rect", "x0": 1, "y0": 1, "x1": 2, "y1": 3}])
    p.process_series()


def test_matplotlib_axis_center():
    # verify that axis_center doesn't raise any errors
    x = symbols("x")

    _plot = lambda ac: plot(sin(x), adaptive=False, n=5, backend=MB,
        show=False, axis_center=ac)

    _plot("center").process_series()
    _plot("auto").process_series()
    _plot((0, 0)).process_series()
