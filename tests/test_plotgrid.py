from sympy import symbols, sin, cos, pi, tan, I, exp
from spb.backends.matplotlib import MB
from spb.backends.plotly import PB
from spb.backends.bokeh import BB
from spb.backends.k3d import KB
from spb.backends.plotgrid import plotgrid, _nrows_ncols
from spb.functions import plot, plot3d, plot_contour
from spb.ccomplex.complex import plot_complex
from spb.vectors import plot_vector
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import panel as pn
from pytest import raises

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
    p = plotgrid()
    assert isinstance(p, plt.Figure)

def test_plotgrid_mode_1():
    x, y, z = symbols("x, y, z")

    # all plots with MatplotlibBackend: combine them into a matplotlib figure
    p1 = plot(cos(x), (x, -5, 5), backend=MB, show=False, ylabel="a")
    p2 = plot(sin(x), (x, -7, 7), backend=MB, show=False, ylabel="b")
    p3 = plot(tan(x), (x, -10, 10), backend=MB, show=False, ylabel="c")
    p = plotgrid(p1, p2, p3)
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

    p1 = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        backend=MB, n1=20, n2=20, show=False)
    p2 = plot(sin(x), cos(x), (x, -7, 7), backend=MB, show=False)
    p = plotgrid(p1, p2, nc=2)
    assert isinstance(p, plt.Figure)
    assert len(p.axes) == 2
    assert isinstance(p.axes[0], Axes3D)
    assert not isinstance(p.axes[1], Axes3D)
    assert p.axes[0].get_xlabel() == "x" and p.axes[0].get_ylabel() == "y" and p.axes[0].get_zlabel() == "f(x, y)"
    assert len(p.axes[0].collections) == 1
    assert p.axes[1].get_xlabel() == "x" and p.axes[1].get_ylabel() == "f(x)"
    assert len(p.axes[1].get_lines()) == 2 

    # mix different backends
    p1 = plot(cos(x), (x, -3, 3), backend=MB, show=False)
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
    p1 = plot(exp(x), backend=MB, show=False)
    p2 = plot(sin(x), backend=MB, show=False)
    p3 = plot(tan(x), backend=MB, show=False, adaptive=False,
        detect_poles=True, eps=0.1, ylim=(-5, 5))
    p4 = plot(cos(x), backend=MB, show=False)
    p5 = plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        backend=MB, n1=20, n2=20, show=False)
    
    # gs is not a dictionary
    raises(TypeError, lambda: plotgrid(p1, p2, p3, p4, p5, gs=1))
    # wrong type of the keys
    gs = {1: p1, 2: p2}
    raises(ValueError, lambda: plotgrid(p1, p2, p3, p4, p5, gs=gs))

    gs = GridSpec(3, 3)
    mapping = {
        gs[0, :1]: p1,
        gs[1, :1]: p2,
        gs[2:, :1]: p3,
        gs[2:, 1:]: p4,
        gs[0:2, 1:]: p5,
    }
    p = plotgrid(gs=mapping)

    assert isinstance(p, plt.Figure)
    assert len(p.axes) == 5
    assert isinstance(p.axes[-1], Axes3D)
    test_line_axes(p.axes[0])
    test_line_axes(p.axes[1])
    test_line_axes(p.axes[2])
    test_line_axes(p.axes[3])
    test_3d_axes(p.axes[4])

    # Mixture of different backends
    p1 = plot(exp(x), backend=MB, show=False)
    p2 = plot(sin(x), backend=PB, show=False)
    p3 = plot(tan(x), backend=MB, show=False, adaptive=False,
        detect_poles=True, eps=0.1, ylim=(-5, 5))
    p4 = plot(cos(x), backend=BB, show=False)
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
    