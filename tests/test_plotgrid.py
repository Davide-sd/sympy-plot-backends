from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import sin, cos, tan
from sympy.functions.elementary.exponential import exp
from sympy.external import import_module
from spb.backends.matplotlib import MB
from spb.backends.plotly import PB
from spb.backends.bokeh import BB
from spb.backends.k3d import KB
from spb.plotgrid import plotgrid, _nrows_ncols
from spb.functions import plot, plot3d, plot_contour
from spb.vectors import plot_vector
from pytest import raises

matplotlib = import_module(
    'matplotlib',
    import_kwargs={'fromlist': ['pyplot', 'axes', 'cm',
        'collections', 'colors', 'quiver']},
    min_module_version='1.1.0',
    catch=(RuntimeError,))
plt = matplotlib.pyplot
mpl_toolkits = import_module(
    'mpl_toolkits', # noqa
    import_kwargs={'fromlist': ['mplot3d']},
    catch=(RuntimeError,))
pn = import_module(
    'panel',
    min_module_version='0.12.0',
    catch=(RuntimeError,))


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
    nr, nc = _nrows_ncols(1, 1, 5)
    assert nr == 5 and nc == 1

    # one row
    nr, nc = _nrows_ncols(1, -1, 5)
    assert nr == 1 and nc == 5
    nr, nc = _nrows_ncols(1, 0, 5)
    assert nr == 1 and nc == 5

    # not enough grid-elements to plot all plots: keep adding rows
    nr, nc = _nrows_ncols(1, 2, 5)
    assert nr == 3 and nc == 2
    nr, nc = _nrows_ncols(2, 2, 5)
    assert nr == 3 and nc == 2

    # enough grid-elements: do not modify nr, nc
    nr, nc = _nrows_ncols(2, 2, 4)
    assert nr == 2 and nc == 2

    nr, nc = _nrows_ncols(3, 2, 5)
    assert nr == 3 and nc == 2


def test_empty_plotgrid():
    p = plotgrid(show=False)
    assert isinstance(p, plt.Figure)


def test_plotgrid_mode_1():
    x, y, z = symbols("x, y, z")

    # all plots with MatplotlibBackend: combine them into a matplotlib figure
    p1 = plot(cos(x), (x, -5, 5), adaptive=False, n=100,
        backend=MB, show=False, ylabel="a")
    p2 = plot(sin(x), (x, -7, 7), adaptive=False, n=100,
        backend=MB, show=False, ylabel="b")
    p3 = plot(tan(x), (x, -10, 10), adaptive=False, n=100,
        backend=MB, show=False, ylabel="c")
    p = plotgrid(p1, p2, p3, show=False)
    assert isinstance(p, plt.Figure)
    assert len(p.axes) == 3
    assert p.axes[0].get_xlabel() == "x" and p.axes[0].get_ylabel() == "a"
    assert len(p.axes[0].get_lines()) == 1
    assert p.axes[0].get_lines()[0].get_label() == "cos(x)"
    assert p.axes[1].get_xlabel() == "x" and p.axes[1].get_ylabel() == "b"
    assert len(p.axes[1].get_lines()) == 1
    assert p.axes[1].get_lines()[0].get_label() == "sin(x)"
    assert p.axes[2].get_xlabel() == "x" and p.axes[2].get_ylabel() == "c"
    assert len(p.axes[2].get_lines()) == 1
    assert p.axes[2].get_lines()[0].get_label() == "tan(x)"

    # no errors are raised when the number of plots is less than the number
    # of grid-cells
    p = plotgrid(p1, p2, p3, nr=2, nc=2, show=False)

    # everything works fine when including 3d plots
    p1 = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        backend=MB, n1=20, n2=20, show=False)
    p2 = plot(sin(x), cos(x), (x, -7, 7), adaptive=False, n=100,
        backend=MB, show=False)
    p = plotgrid(p1, p2, nc=2, show=False)
    assert isinstance(p, plt.Figure)
    assert len(p.axes) == 2
    assert isinstance(p.axes[0], mpl_toolkits.mplot3d.Axes3D)
    assert not isinstance(p.axes[1], mpl_toolkits.mplot3d.Axes3D)
    assert p.axes[0].get_xlabel() == "x" and p.axes[0].get_ylabel() == "y" and p.axes[0].get_zlabel() == "f(x, y)"
    assert len(p.axes[0].collections) == 1
    assert p.axes[1].get_xlabel() == "x" and p.axes[1].get_ylabel() == "f(x)"
    assert len(p.axes[1].get_lines()) == 2

    # mix different backends
    p1 = plot(cos(x), (x, -3, 3), adaptive=False, n=100,
        backend=MB, show=False)
    p2 = plot_contour(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        backend=PB, n1=20, n2=20, show=False)
    p3 = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        backend=KBchild1, n1=20, n2=20, show=False)
    p4 = plot_vector([-y, x], (x, -5, 5), (y, -5, 5), backend=BB, show=False)

    p = plotgrid(p1, p2, p3, p4, nr=2, nc=2)
    assert isinstance(p, pn.GridSpec)
    assert p.ncols == 2 and p.nrows == 2
    assert isinstance(p.objects[(0, 0, 1, 1)], pn.pane.plot.Matplotlib)
    assert isinstance(p.objects[(0, 1, 1, 2)], pn.pane.plotly.Plotly)
    assert isinstance(p.objects[(1, 0, 2, 1)], pn.pane.ipywidget.IPyWidget)
    assert isinstance(p.objects[(1, 1, 2, 2)], pn.pane.plot.Bokeh)


def test_plotgrid_mode_2():
    from matplotlib.gridspec import GridSpec

    x, y, z = symbols("x, y, z")

    def test_line_axes(ax):
        assert ax.get_xlabel() == "x" and ax.get_ylabel() == "f(x)"
        assert len(ax.get_lines()) == 1

    def test_3d_axes(ax):
        assert ax.get_xlabel() == "x" and ax.get_ylabel() == "y" and ax.get_zlabel() == "f(x, y)"
        assert len(ax.collections) == 1

    # all plots are instances of MatplotlibBackend
    p1 = plot(exp(x), adaptive=False, n=100, backend=MB, show=False)
    p2 = plot(sin(x), adaptive=False, n=100, backend=MB, show=False)
    p3 = plot(tan(x), backend=MB, show=False, adaptive=False,
        detect_poles=True, eps=0.1, ylim=(-5, 5))
    p4 = plot(cos(x), adaptive=False, n=100, backend=MB, show=False)
    p5 = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        backend=MB, n1=20, n2=20, show=False)

    # gs is not a dictionary
    raises(TypeError, lambda: plotgrid(p1, p2, p3, p4, p5, gs=1, show=False))
    # wrong type of the keys
    gs = {1: p1, 2: p2}
    raises(ValueError, lambda: plotgrid(p1, p2, p3, p4, p5, gs=gs, show=False))

    gs = GridSpec(3, 3)
    mapping = {
        gs[0, :1]: p1,
        gs[1, :1]: p2,
        gs[2:, :1]: p3,
        gs[2:, 1:]: p4,
        gs[0:2, 1:]: p5,
    }
    p = plotgrid(gs=mapping, show=False)

    assert isinstance(p, plt.Figure)
    assert len(p.axes) == 5
    assert isinstance(p.axes[-1], mpl_toolkits.mplot3d.Axes3D)
    test_line_axes(p.axes[0])
    test_line_axes(p.axes[1])
    test_line_axes(p.axes[2])
    test_line_axes(p.axes[3])
    test_3d_axes(p.axes[4])

    # Mixture of different backends
    p1 = plot(exp(x), adaptive=False, n=100, backend=MB, show=False)
    p2 = plot(sin(x), adaptive=False, n=100, backend=PB, show=False)
    p3 = plot(tan(x), backend=MB, show=False, adaptive=False,
        detect_poles=True, eps=0.1, ylim=(-5, 5))
    p4 = plot(cos(x), adaptive=False, n=100, backend=BB, show=False)
    p5 = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        backend=KBchild1, n1=20, n2=20, show=False)

    gs = GridSpec(3, 3)
    mapping = {
        gs[0, :1]: p1,
        gs[1, :1]: p2,
        gs[2:, :1]: p3,
        gs[2:, 1:]: p4,
        gs[0:2, 1:]: p5,
    }
    p = plotgrid(gs=mapping)
    assert isinstance(p, pn.GridSpec)
    assert p.nrows == p.ncols == 3
    assert len(p.objects) == 5
    assert isinstance(p.objects[(0, 0, 1, 1)], pn.pane.plot.Matplotlib)
    assert isinstance(p.objects[(1, 0, 2, 1)], pn.pane.plotly.Plotly)
    assert isinstance(p.objects[(2, 0, 3, 1)], pn.pane.plot.Matplotlib)
    assert isinstance(p.objects[(2, 1, 3, 3)], pn.pane.plot.Bokeh)
    assert isinstance(p.objects[(0, 1, 2, 3)], pn.pane.ipywidget.IPyWidget)


def test_panel_kw():
    x = symbols("x")
    p1 = plot(sin(x), adaptive=False, n=100, backend=MB, show=False)
    p2 = plot(tan(x), adaptive=False, n=100, backend=PB, show=False)
    p3 = plot(exp(-x), adaptive=False, n=100, backend=BB, show=False)
    pg1 = plotgrid(p1, p2, p3, nr=1, nc=3, show=False)
    pg2 = plotgrid(p1, p2, p3, nr=1, nc=3, show=False,
        panel_kw=dict(sizing_mode="stretch_width", height=250))
    assert (pg1.height != pg2.height) and (pg2.height == 250)
