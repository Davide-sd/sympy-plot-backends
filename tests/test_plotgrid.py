import pytest
from pytest import raises
from spb import (
    MB, PB, BB, KB,
    plotgrid, PlotGrid, plot, plot3d, plot_contour, plot_vector, plot_polar,
    plot_complex, plot_parametric, plot3d_parametric_line,
)
from spb.interactive import IPlot
from spb.plotgrid import _nrows_ncols
from sympy import symbols, sin, cos, tan, exp, pi, Piecewise
from sympy.external import import_module
bokeh = import_module("bokeh")
ipywidgets = import_module("ipywidgets")
pn = import_module("panel")
plotly = import_module("plotly")
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mpl_toolkits
import numpy as np
import os
from tempfile import TemporaryDirectory


class KBchild1(KB):
    def _get_mode(self):
        # tells the backend it is running into Jupyter, even if it is not.
        # this is necessary to run these tests.
        return 0


def test_nrows_ncols():
    # default to 1 column
    nr, nc = _nrows_ncols(-1, 0, 5)
    assert nr == 5 and nc == 1
    nr, nc = _nrows_ncols(0, 0, 5)
    assert nr == 5 and nc == 1
    nr, nc = _nrows_ncols(0, -1, 5)
    assert nr == 5 and nc == 1

    # one row
    nr, nc = _nrows_ncols(1, -1, 5)
    assert nr == 1 and nc == 5
    nr, nc = _nrows_ncols(1, 0, 5)
    assert nr == 1 and nc == 5
    nr, nc = _nrows_ncols(1, 1, 5)
    assert nr == 1 and nc == 5
    nr, nc = _nrows_ncols(1, 2, 5)
    assert nr == 1 and nc == 5

    # not enough grid-elements to plot all plots: keep adding rows
    nr, nc = _nrows_ncols(2, 2, 5)
    assert nr == 3 and nc == 2

    # enough grid-elements: do not modify nr, nc
    nr, nc = _nrows_ncols(2, 2, 4)
    assert nr == 2 and nc == 2
    nr, nc = _nrows_ncols(3, 2, 5)
    assert nr == 3 and nc == 2


def test_empty_plotgrid():
    p = plotgrid(show=False)
    assert isinstance(p, PlotGrid)
    assert isinstance(p.fig, plt.Figure)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plotgrid_mode_1_matplotlib():
    x, y, z = symbols("x, y, z")
    options = dict(n=10, backend=MB, show=False)

    # all plots with MatplotlibBackend: combine them into a matplotlib figure
    p1 = plot(cos(x), (x, -5, 5), ylabel="a", use_latex=True, **options)
    p2 = plot(sin(x), (x, -7, 7), ylabel="b", use_latex=False, **options)
    p3 = plot(tan(x), (x, -10, 10), ylabel="c", use_latex=False, **options)
    p = plotgrid(p1, p2, p3, show=False)
    assert isinstance(p, PlotGrid)
    assert isinstance(p.fig, plt.Figure)
    assert len(p.fig.axes) == 3
    assert p.fig.axes[0].get_xlabel() == "$x$"
    assert p.fig.axes[0].get_ylabel() == "a"
    assert len(p.fig.axes[0].get_lines()) == 1
    assert p.fig.axes[0].get_lines()[0].get_label() == "$\\cos{\\left(x \\right)}$"
    assert p.fig.axes[1].get_xlabel() == "x"
    assert p.fig.axes[1].get_ylabel() == "b"
    assert len(p.fig.axes[1].get_lines()) == 1
    assert p.fig.axes[1].get_lines()[0].get_label() == "sin(x)"
    assert p.fig.axes[2].get_xlabel() == "x"
    assert p.fig.axes[2].get_ylabel() == "c"
    assert len(p.fig.axes[2].get_lines()) == 1
    assert p.fig.axes[2].get_lines()[0].get_label() == "tan(x)"

    # no errors are raised when the number of plots is less than the number
    # of grid-cells
    p = plotgrid(p1, p2, p3, nr=2, nc=2, show=False)

    # everything works fine when including 3d plots
    p1 = plot3d(
        cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        backend=MB,
        n1=20, n2=20,
        show=False,
        use_latex=False,
    )
    p2 = plot(sin(x), cos(x), (x, -7, 7), use_latex=False, **options)
    p = plotgrid(p1, p2, nc=2, show=False)
    assert isinstance(p, PlotGrid)
    assert isinstance(p.fig, plt.Figure)
    assert len(p.fig.axes) == 2
    assert isinstance(p.fig.axes[0], mpl_toolkits.mplot3d.Axes3D)
    assert not isinstance(p.fig.axes[1], mpl_toolkits.mplot3d.Axes3D)
    assert p.fig.axes[0].get_xlabel() == "x"
    assert p.fig.axes[0].get_ylabel() == "y"
    assert p.fig.axes[0].get_zlabel() == "f(x, y)"
    assert len(p.fig.axes[0].collections) == 1
    assert p.fig.axes[1].get_xlabel() == "x"
    assert p.fig.axes[1].get_ylabel() == "f(x)"
    assert len(p.fig.axes[1].get_lines()) == 2


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plotgrid_mode_1_different_backends():
    x, y, z = symbols("x, y, z")
    options = dict(n=10, backend=MB, show=False, imodule="panel")

    p1 = plot(cos(x), (x, -3, 3), **options)
    p2 = plot_contour(
        cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        backend=PB,
        n1=20, n2=20,
        show=False,
        imodule="panel"
    )
    p3 = plot3d(
        cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        backend=KBchild1,
        n1=20, n2=20,
        show=False,
        imodule="panel"
    )
    p4 = plot_vector(
        [-y, x], (x, -5, 5), (y, -5, 5),
        backend=BB,
        show=False,
        imodule="panel"
    )

    p = plotgrid(p1, p2, p3, p4, nr=2, nc=2, show=False, imodule="panel")
    assert isinstance(p.fig, pn.GridSpec)
    assert p.ncolumns == 2 and p.nrows == 2
    assert isinstance(p.fig.objects[(0, 0, 1, 1)], pn.pane.plot.Matplotlib)
    assert isinstance(p.fig.objects[(0, 1, 1, 2)], pn.pane.plotly.Plotly)
    assert isinstance(p.fig.objects[(1, 0, 2, 1)], pn.pane.ipywidget.IPyWidget)
    assert isinstance(p.fig.objects[(1, 1, 2, 2)], pn.pane.plot.Bokeh)


@pytest.mark.skipif(ipywidgets is None, reason="ipywidgets is not installed")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plotgrid_mode_1_interactive_ipywidgets():
    # verify the correct behavior when providing interactive widget plots.

    def build_plotgrid(backend):
        x, y, z = symbols("x, y, z")
        options = dict(
            n=10,
            backend=backend,
            imodule="ipywidgets",
            show=False,
            params={
                y: (1, 0, 2),
                z: (5, 0, 10),
            },
        )

        # all plots with PlotlyBackend
        p1 = plot(sin(x * y) * exp(-abs(x) / z), **options)
        p2 = plot(cos(x * y) * exp(-abs(x) / z), **options)
        p3 = plot(cos(x * y) * sin(x * y) * exp(-abs(x) / z), **options)
        return plotgrid(p1, p2, p3, show=False, imodule="ipywidgets")

    # NOTE: I'm going to test with Plotly and not Matplotlib because I have
    # no idea how to setup test for Matplotlib, which requires interactivity...
    p = build_plotgrid(PB)
    assert isinstance(p, IPlot)
    res = p.show()
    assert isinstance(res, ipywidgets.VBox)
    assert len(res.children) == 2
    assert all(isinstance(t, ipywidgets.GridspecLayout) for t in res.children)
    # widgets grid
    assert res.children[0].n_rows == 1
    assert res.children[0].n_columns == 2
    assert isinstance(res.children[0][0, 0], ipywidgets.FloatSlider)
    assert isinstance(res.children[0][0, 1], ipywidgets.FloatSlider)
    # plots grid
    assert res.children[1].n_rows == 3
    assert res.children[1].n_columns == 1
    go = plotly.graph_objects
    assert all(
        isinstance(res.children[1][i, 0].children[0], go.FigureWidget)
        for i in range(3)
    )

    # quick run-down to verify that no errors are raised when changing the
    # value of a widget
    res.children[0][0, 0].value = 2

    # NOTE: this opens pictures in the browser. Why?
    p = build_plotgrid(BB)
    res = p.show()
    res.children[0][0, 0].value = 2

    # TODO: it would be nice to test matplotlib too, but how?
    p = build_plotgrid(MB)
    # res = p.show()
    # res.children[0][0, 0].value = 2


@pytest.mark.skipif(pn is None, reason="panel is not installed")
def test_plotgrid_mode_1_interactive_panel():
    # verify the correct behavior when providing interactive widget plots.

    def build_plotgrid(backend):
        x, y, z = symbols("x, y, z")
        options = dict(
            n=10,
            backend=backend,
            imodule="panel",
            show=False,
            params={
                y: (1, 0, 2),
                z: (5, 0, 10),
            },
        )

        p1 = plot(sin(x * y) * exp(-abs(x) / z), **options)
        p2 = plot(cos(x * y) * exp(-abs(x) / z), **options)
        p3 = plot(cos(x * y) * sin(x * y) * exp(-abs(x) / z), **options)
        return plotgrid(p1, p2, p3, show=False, imodule="panel")

    # NOTE: I'm going to test with Plotly and not Matplotlib because I have
    # no idea how to setup test for Matplotlib, which requires interactivity...
    p = build_plotgrid(PB)
    assert isinstance(p, IPlot)
    res = p.show()
    assert isinstance(res, pn.Column)
    assert len(res.objects) == 2
    assert isinstance(res.objects[0], pn.GridBox)
    assert isinstance(res.objects[1], pn.GridSpec)
    # widgets grid
    grid = res.objects[0]
    assert grid.ncols == 2
    assert grid.nrows is None
    sliders = grid.objects
    assert len(sliders) == 2
    assert all(
        isinstance(o, pn.widgets.FloatSlider) for o in sliders
    )
    # plots grid
    plots = res.objects[1]
    assert plots.ncols == 1
    assert plots.nrows == 3
    assert len(plots.objects) == 3
    assert isinstance(plots.objects[(0, 0, 1, 1)], pn.pane.Plotly)
    assert isinstance(plots.objects[(1, 0, 2, 1)], pn.pane.Plotly)
    assert isinstance(plots.objects[(2, 0, 3, 1)], pn.pane.Plotly)

    # quick run-down to verify that no errors are raised when changing the
    # value of a widget
    sliders[0].value = 2

    p = build_plotgrid(BB)
    res = p.show()
    grid = res.objects[0]
    sliders = grid.objects
    sliders[0].value = 2

    p = build_plotgrid(MB)
    res = p.show()
    grid = res.objects[0]
    sliders = grid.objects
    sliders[0].value = 2


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plotgrid_mode_2_matplotlib():
    from matplotlib.gridspec import GridSpec

    x, y, z = symbols("x, y, z")

    def test_line_axes(ax):
        assert ax.get_xlabel() == "x" and ax.get_ylabel() == "f(x)"
        assert len(ax.get_lines()) == 1

    def test_3d_axes(ax):
        assert (
            ax.get_xlabel() == "x"
            and ax.get_ylabel() == "y"
            and ax.get_zlabel() == "f(x, y)"
        )
        assert len(ax.collections) == 1

    # all plots are instances of MatplotlibBackend
    options = dict(
        n=100, backend=MB, show=False, use_latex=False
    )
    p1 = plot(exp(x), **options)
    p2 = plot(sin(x), **options)
    p3 = plot(tan(x), detect_poles=True, eps=0.1, ylim=(-5, 5), **options)
    p4 = plot(cos(x), **options)
    options["n"] = 20
    p5 = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2), **options)

    # gs is not a dictionary
    raises(TypeError, lambda: plotgrid(p1, p2, p3, p4, p5, gs=1, show=False))
    # wrong type of the keys
    gs = {1: p1, 2: p2}
    raises(
        ValueError,
        lambda: plotgrid(p1, p2, p3, p4, p5, gs=gs, show=False)
    )

    gs = GridSpec(3, 3)
    mapping = {
        gs[0, :1]: p1,
        gs[1, :1]: p2,
        gs[2:, :1]: p3,
        gs[2:, 1:]: p4,
        gs[0:2, 1:]: p5,
    }
    p = plotgrid(gs=mapping, show=False)

    assert isinstance(p, PlotGrid)
    assert isinstance(p.fig, plt.Figure)
    assert len(p.fig.axes) == 5
    assert isinstance(p.fig.axes[-1], mpl_toolkits.mplot3d.Axes3D)
    test_line_axes(p.fig.axes[0])
    test_line_axes(p.fig.axes[1])
    test_line_axes(p.fig.axes[2])
    test_line_axes(p.fig.axes[3])
    test_3d_axes(p.fig.axes[4])


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plotgrid_mode_2_different_backends():
    from matplotlib.gridspec import GridSpec

    x, y, z = symbols("x, y, z")
    # Mixture of different backends
    options = dict(
        n=100, show=False, use_latex=False, imodule="panel"
    )
    p1 = plot(exp(x), backend=MB, **options)
    p2 = plot(sin(x), backend=PB, **options)
    p3 = plot(
        tan(x),
        backend=MB,
        detect_poles=True, eps=0.1,
        ylim=(-5, 5),
        **options
    )
    p4 = plot(cos(x), backend=BB, **options)
    p5 = plot3d(
        cos(x**2 + y**2),
        (x, -2, 2),
        (y, -2, 2),
        backend=KBchild1,
        n=20,
        show=False,
        use_latex=False,
        imodule="panel"
    )

    gs = GridSpec(3, 3)
    mapping = {
        gs[0, :1]: p1,
        gs[1, :1]: p2,
        gs[2:, :1]: p3,
        gs[2:, 1:]: p4,
        gs[0:2, 1:]: p5,
    }
    p = plotgrid(gs=mapping, show=False, imodule="panel")
    assert isinstance(p, PlotGrid)
    assert isinstance(p.fig, pn.GridSpec)
    assert p.fig.nrows == p.fig.ncols == 3
    assert len(p.fig.objects) == 5
    assert isinstance(p.fig.objects[(0, 0, 1, 1)], pn.pane.plot.Matplotlib)
    assert isinstance(p.fig.objects[(1, 0, 2, 1)], pn.pane.plotly.Plotly)
    assert isinstance(p.fig.objects[(2, 0, 3, 1)], pn.pane.plot.Matplotlib)
    assert isinstance(p.fig.objects[(2, 1, 3, 3)], pn.pane.plot.Bokeh)
    assert isinstance(p.fig.objects[(0, 1, 2, 3)], pn.pane.ipywidget.IPyWidget)


@pytest.mark.skipif(ipywidgets is None, reason="ipywidgets is not installed")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plotgrid_mode_2_interactive_ipywidgets():
    # verify the correct behavior when providing interactive widget plots.
    # NOTE: I'm going to test with Plotly and not Matplotlib because I have
    # no idea how to setup test for Matplotlib, which requires interactivity...

    x, y, z = symbols("x, y, z")
    # all plots are instances of PlotlyBackend
    options = dict(
        n=100,
        backend=PB,
        show=False,
        use_latex=False,
        params={
            y: (1, 0, 2),
            z: (5, 0, 10),
        },
    )

    p1 = plot(exp(x * y) * exp(-abs(x) / z), **options)
    p2 = plot(sin(x * y) * exp(-abs(x) / z), **options)
    p3 = plot(
        tan(x * y) * exp(-abs(x) / z),
        detect_poles=True, eps=0.1,
        ylim=(-5, 5),
        **options
    )
    p4 = plot(cos(x * y) * exp(-abs(x) / z), **options)
    options["n"] = 20
    p5 = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2), **options)

    # gs is not a dictionary
    raises(
        TypeError,
        lambda: plotgrid(
            p1, p2, p3, p4, p5, gs=1, show=False, imodule="ipywidgets"),
    )
    # wrong type of the keys
    gs = {1: p1, 2: p2}
    raises(
        ValueError,
        lambda: plotgrid(
            p1, p2, p3, p4, p5, gs=gs, show=False, imodule="ipywidgets"),
    )

    gs = GridSpec(3, 3)
    mapping = {
        gs[0, :1]: p1,
        gs[1, :1]: p2,
        gs[2:, :1]: p3,
        gs[2:, 1:]: p4,
        gs[0:2, 1:]: p5,
    }
    p = plotgrid(gs=mapping, show=False, imodule="ipywidgets")

    assert isinstance(p, IPlot)
    res = p.show()
    assert isinstance(res, ipywidgets.VBox)
    assert len(res.children) == 2
    assert all(isinstance(t, ipywidgets.GridspecLayout) for t in res.children)
    # widgets grid
    assert res.children[0].n_rows == 1
    assert res.children[0].n_columns == 2
    assert isinstance(res.children[0][0, 0], ipywidgets.FloatSlider)
    assert isinstance(res.children[0][0, 1], ipywidgets.FloatSlider)
    # plots grid
    assert res.children[1].n_rows == 3
    assert res.children[1].n_columns == 3
    go = plotly.graph_objects
    assert isinstance(res.children[1][0, 0].children[0], go.FigureWidget)
    assert isinstance(res.children[1][1, 0].children[0], go.FigureWidget)
    assert isinstance(res.children[1][2, 0].children[0], go.FigureWidget)
    assert isinstance(res.children[1][2, 1:].children[0], go.FigureWidget)
    assert isinstance(res.children[1][0:2, 1:].children[0], go.FigureWidget)


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plotgrid_mode_2_interactive_panel():
    # verify the correct behavior when providing interactive widget plots.

    x, y, z = symbols("x, y, z")
    # all plots are instances of PlotlyBackend
    options = dict(
        n=100,
        backend=PB,
        imodule="panel",
        show=False,
        use_latex=False,
        params={
            y: (1, 0, 2),
            z: (5, 0, 10),
        },
    )

    p1 = plot(exp(x * y) * exp(-abs(x) / z), **options)
    p2 = plot(sin(x * y) * exp(-abs(x) / z), **options)
    p3 = plot(
        tan(x * y) * exp(-abs(x) / z),
        detect_poles=True,
        eps=0.1,
        ylim=(-5, 5),
        **options
    )
    p4 = plot(cos(x * y) * exp(-abs(x) / z), **options)
    options["n"] = 20
    p5 = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2), **options)

    # gs is not a dictionary
    raises(
        TypeError,
        lambda: plotgrid(
            p1, p2, p3, p4, p5, gs=1, show=False, imodule="panel"),
    )
    # wrong type of the keys
    gs = {1: p1, 2: p2}
    raises(
        ValueError,
        lambda: plotgrid(
            p1, p2, p3, p4, p5, gs=gs, show=False, imodule="panel"),
    )

    gs = GridSpec(3, 3)
    mapping = {
        gs[0, :1]: p1,
        gs[1, :1]: p2,
        gs[2:, :1]: p3,
        gs[2:, 1:]: p4,
        gs[0:2, 1:]: p5,
    }
    p = plotgrid(gs=mapping, show=False, imodule="panel")

    assert isinstance(p, IPlot)
    res = p.show()
    assert isinstance(res, pn.Column)
    assert len(res.objects) == 2
    assert isinstance(res.objects[0], pn.GridBox)
    assert isinstance(res.objects[1], pn.GridSpec)
    # widgets grid
    grid = res.objects[0]
    assert grid.ncols == 2
    assert grid.nrows is None
    sliders = grid.objects
    assert all(
        isinstance(s, pn.widgets.FloatSlider) for s in sliders
    )
    # plots grid
    plots = res.objects[1]
    assert plots.ncols == 3
    assert plots.nrows == 3
    assert len(plots.objects) == 5
    assert isinstance(plots.objects[(0, 0, 1, 1)], pn.pane.Plotly)
    assert isinstance(plots.objects[(1, 0, 2, 1)], pn.pane.Plotly)
    assert isinstance(plots.objects[(2, 0, 3, 1)], pn.pane.Plotly)
    assert isinstance(plots.objects[(2, 1, 3, 3)], pn.pane.Plotly)
    assert isinstance(plots.objects[(0, 1, 2, 3)], pn.pane.Plotly)


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_panel_kw_imodule_panel():
    x = symbols("x")
    options = dict(n=100, show=False, imodule="panel")
    p1 = plot(sin(x), backend=MB, **options)
    p2 = plot(tan(x), backend=PB, **options)
    p3 = plot(exp(-x), backend=BB, **options)
    pg1 = plotgrid(p1, p2, p3, nr=1, nc=3, show=False, imodule="panel")
    pg2 = plotgrid(
        p1, p2, p3,
        nr=1, nc=3,
        show=False,
        imodule="panel",
        panel_kw=dict(sizing_mode="stretch_width", height=250),
    )
    assert (pg1.fig.height != pg2.fig.height) and (pg2.fig.height == 250)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plotgrid_mode_1_polar_axis():
    # verify that when polar_axis is used on a plot, the resulting subplot
    # will use a polar projection

    theta = symbols("theta")
    r = 3 * sin(4 * theta)

    p = lambda pa=False, yl=None: plot_polar(
        r, (theta, 0, 2 * pi),
        show=False, backend=MB, polar_axis=pa, ylim=yl
    )
    pg = plotgrid(p(True, (0, 3)), p(), nr=1, nc=-1, show=False)
    assert isinstance(pg.fig.axes[0], matplotlib.projections.polar.PolarAxes)
    assert not isinstance(
        pg.fig.axes[1], matplotlib.projections.polar.PolarAxes
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plot_size_matplotlib():
    # verify that the `size` keyword argument works fine

    x = symbols("x")
    options = dict(n=100, backend=MB, show=False)
    p1 = plot(sin(x), **options)
    p2 = plot(cos(x), **options)
    p = plotgrid(p1, p2, size=(5, 7.5), show=False)
    assert np.allclose(p.fig.get_size_inches(), [5, 7.5])


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plot_size_matplotlib():
    # verify that the `size` keyword argument works fine

    x = symbols("x")
    options = dict(n=100, backend=PB, show=False)
    p1 = plot(sin(x), **options)
    p2 = plot(cos(x), **options)
    p = plotgrid(p1, p2, size=(400, 600), show=False, imodule="ipywidgets")
    res = p.show()
    assert res.width == "400px"
    # assert res.height == "600px"

    options = dict(
        n=100, backend=PB, show=False, imodule="panel"
    )
    p1 = plot(sin(x), **options)
    p2 = plot(cos(x), **options)
    p = plotgrid(p1, p2, size=(400, 600), show=False, imodule="panel")
    assert p.fig.width == 400
    assert p.fig.height == 600


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_imagegrid():
    # verify that matplotlib's ImageGrid axis works fine

    x = symbols("x")
    options = dict(backend=MB, show=False, n=10)
    p1 = plot_complex(sin(x), (x, -2 - 2j, 2 + 2j), **options)
    p2 = plot_complex(cos(x), (x, -2 - 2j, 2 + 2j), **options)
    p = plotgrid(p1, p2, nr=1, imagegrid=True, show=False)
    assert len(p.fig.axes) == 4
    assert p.fig.axes[0].get_xlabel() == "Re"
    assert p.fig.axes[1].get_xlabel() == "Re"
    assert p.fig.axes[2].get_ylabel() == "Argument"

    p = plotgrid(p1, p2, nr=1, imagegrid=False, show=False)
    assert len(p.fig.axes) == 4
    assert p.fig.axes[0].get_xlabel() == "Re"
    assert p.fig.axes[1].get_ylabel() == "Argument"
    assert p.fig.axes[2].get_xlabel() == "Re"
    assert p.fig.axes[3].get_ylabel() == "Argument"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_save():
    x, y = symbols("x, y")

    def _create_plots(B):
        options = dict(n=10, show=False, backend=B)
        p1 = plot(x, **options)
        p2 = plot_parametric(
            (sin(x), cos(x)), (x, sin(x)), **options)
        p3 = plot_parametric(
            cos(x), sin(x), **options)
        p4 = plot3d_parametric_line(
            sin(x), cos(x), x, **options)
        p5 = plot(cos(x), (x, -pi, pi), **options)
        p5[0].color_func = lambda a: a
        p6 = plot(
            Piecewise((1, x > 0), (0, True)), (x, -1, 1),
            **options)
        p7 = plot_contour(
            (x**2 + y**2, (x, -5, 5), (y, -5, 5)),
            (x**3 + y**3, (x, -3, 3), (y, -3, 3)),
            **options
        )
        return p1, p2, p3, p4, p5, p6, p7

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        # matplotlib figures
        p1, p2, p3, p4, p5, p6, p7 = _create_plots(MB)

        # symmetric grid
        p = PlotGrid(2, 2, p1, p2, p3, p4, show=False)
        filename = "test_grid1.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # grid size greater than the number of subplots
        p = PlotGrid(3, 4, p1, p2, p3, p4, show=False)
        filename = "test_grid2.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        # unsymmetric grid (subplots in one line)
        p = PlotGrid(1, 3, p5, p6, p7, show=False)
        filename = "test_grid3.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        if plotly:
            #
            # holoviz's panel objects
            p1, p2, p3, p4, p5, p6, p7 = _create_plots(PB)

            # symmetric grid
            p = PlotGrid(2, 2, p1, p2, p3, p4, show=False, imodule="panel")
            p.save("test_1.html")

            # grid size greater than the number of subplots
            p = PlotGrid(3, 4, p1, p2, p3, p4, show=False, imodule="panel")
            p.save("test_2.html")

            # unsymmetric grid (subplots in one line)
            p = PlotGrid(1, 3, p5, p6, p7, show=False, imodule="panel")
            p.save("test_3.html")


@pytest.mark.skipif(pn is None, reason="panel is not installed")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_plotgrid_interactive_mixed_modules():
    def build_plots(imodule1, imodule2):
        x, y, z = symbols("x, y, z")
        # all plots are instances of PlotlyBackend
        options = dict(
            n=100,
            backend=PB,  # imodule="panel",
            show=False,
            use_latex=False,
            params={
                y: (1, 0, 2),
                z: (5, 0, 10),
            },
        )
        p1 = plot(exp(x * y) * exp(-abs(x) / z), imodule=imodule1, **options)
        p2 = plot(sin(x * y) * exp(-abs(x) / z), imodule=imodule1, **options)
        p3 = plot(
            tan(x * y) * exp(-abs(x) / z),
            imodule=imodule2,
            detect_poles=True, eps=0.1,
            ylim=(-5, 5),
            **options
        )
        return p1, p2, p3

    # plots without imodule are built with default module, ipywidgets.
    # no error because imodule corresponds between plots and plotgrid
    p1, p2, p3 = build_plots(None, None)
    plotgrid(p1, p2, p3, imodule="ipywidgets", show=False)

    # error because imodule is different between plots and plotgrid
    p1, p2, p3 = build_plots(None, None)
    raises(
        ValueError,
        lambda: plotgrid(p1, p2, p3, imodule="panel", show=False)
    )

    # error because mix of different interactive modules on plots
    p1, p2, p3 = build_plots("panel", "ipywidgets")
    raises(
        ValueError,
        lambda: plotgrid(p1, p2, p3, imodule="ipywidgets", show=False)
    )

    # error because mix of different interactive modules on plots
    p1, p2, p3 = build_plots("panel", None)
    raises(
        ValueError,
        lambda: plotgrid(p1, p2, p3, imodule="ipywidgets", show=False)
    )

    # error because mix of different interactive modules on plots
    p1, p2, p3 = build_plots(None, "panel")
    raises(
        ValueError,
        lambda: plotgrid(p1, p2, p3, imodule="ipywidgets", show=False)
    )

    # error because mix of different interactive modules on plots
    p1, p2, p3 = build_plots("ipywidgets", "panel")
    raises(
        ValueError,
        lambda: plotgrid(p1, p2, p3, imodule="ipywidgets", show=False)
    )

    # same module on plots and plotgrid
    p1, p2, p3 = build_plots("ipywidgets", "ipywidgets")
    plotgrid(p1, p2, p3, imodule="ipywidgets", show=False)

    p1, p2, p3 = build_plots("panel", "panel")
    plotgrid(p1, p2, p3, imodule="panel", show=False)

    # plots and plotgrid uses different interactive modules
    p1, p2, p3 = build_plots("ipywidgets", "ipywidgets")
    raises(
        ValueError,
        lambda: plotgrid(p1, p2, p3, imodule="panel", show=False)
    )

    p1, p2, p3 = build_plots("panel", "panel")
    raises(
        ValueError,
        lambda: plotgrid(p1, p2, p3, imodule="ipywidgets", show=False)
    )
