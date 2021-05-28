from pytest import raises
from spb.backends.base_backend import Plot
from spb.backends.bokeh import BB
from spb.backends.plotly import PB
from spb.backends.k3d import KB
from spb.series import BaseSeries
from spb import (
    plot, plot3d, vector_plot, plot_contour, plot_implicit, plot_parametric,
    plot3d_parametric_line
)
from sympy import symbols, cos, sin, Matrix, pi
import plotly.graph_objects as go

x, y, z = symbols("x, y, z")

"""
How to test plots? That's a great question. Here, I'm only going to test that
each backend:
1. produces the necessary numerical data.
2. raises the necessary errors
3. correctly use the common keyword arguments to customize the plot.
"""

# TODO:
#   1. test for empty plots
#   2. test for multiple colorscales when plotting multiple 3d expressions

# The following plot functions will be used by the different backends
p1 = lambda B: plot(sin(x), cos(x), sin(x / 2),
        cos(x / 2), 2 * sin(x), 2 * cos(x),
        backend=B, show=False)
p2 = lambda B, line_kw: plot(sin(x), cos(x), line_kw=line_kw,
        backend=B, show=False, legend=True)
p3 = lambda B, line_kw: plot_parametric(cos(x), sin(x), (x, 0, 1.5 * pi),
        backend=B, show=False, line_kw=line_kw)
p4 = lambda B, line_kw: plot3d_parametric_line(cos(x), sin(x), x, (x, -pi, pi),
        backend=B, show=False, line_kw=line_kw)
p5 = lambda B, surface_kw: plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        n=20, use_cm=False, backend=B, show=False, surface_kw=surface_kw)
p6 = lambda B, contour_kw: plot_contour(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        n=20, backend=B, show=False, contour_kw=contour_kw)
p7 = lambda B, contour_kw, quiver_kw: vector_plot(
        Matrix([x, y]), (x, -5, 5), (y, -4, 4),
        backend=B, show=False, quiver_kw=quiver_kw, contour_kw=contour_kw)
p8 = lambda B, stream_kw, contour_kw: vector_plot(
        Matrix([x, y]), (x, -5, 5), (y, -4, 4),
        backend=B, scalar=(x + y), streamlines=True, show=False,
        stream_kw=stream_kw, contour_kw=contour_kw)
p9 = lambda B, quiver_kw: vector_plot(
        Matrix([z, y, x]), (x, -5, 5), (y, -4, 4), (z, -3, 3),
        backend=B, n=10, quiver_kw=quiver_kw, show=False)
p10 = lambda B, stream_kw: vector_plot(
        Matrix([z, y, x]), (x, -5, 5), (y, -4, 4), (z, -3, 3),
        backend=B, n=10, streamlines=True, show=False, stream_kw=stream_kw)
p11 = lambda B, contour_kw: plot_implicit(x > y, (x, -5, 5), (y, -4, 4),
        n=20, backend=B, show=False, contour_kw=contour_kw)
p12 = lambda B, contour_kw: plot_implicit(x > y, (x, -5, 5), (y, -4, 4),
        n=20, backend=B, adaptive=False, show=False, contour_kw=contour_kw)
p13 = lambda B: plot3d(
    (cos(x**2 + y**2), (x, -3, -2), (y, -3, 3)),
    (cos(x**2 + y**2), (x, -2, -1), (y, -3, 3)),
    (cos(x**2 + y**2), (x, -1, 0), (y, -3, 3)),
    (cos(x**2 + y**2), (x, 0, 1), (y, -3, 3)),
    (cos(x**2 + y**2), (x, 1, 2), (y, -3, 3)),
    (cos(x**2 + y**2), (x, 2, 3), (y, -3, 3)),
    backend=B, use_cm=False, show=False
)

class UnsupportedSeries(BaseSeries):
    pass

def test_common_keywords():
    # TODO: here I didn't test axis_center, autoscale, margin
    kw = dict(title="a", xlabel="x", ylabel="y", zlabel="z", aspect="equal",
        grid=False, xscale="log", yscale="log", zscale="log", legend=True,
        xlim=(-1, 1), ylim=(-2, 2), zlim=(-3, 3), size=(5, 10), backend=BB)
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
    assert p._kwargs == kw

class PBchild(PB):
    colorloop = ["red", "green", "blue"]

def test_PlotlyBackend():
    assert hasattr(PB, "colorloop")
    assert isinstance(PB.colorloop, (list, tuple))
    assert hasattr(PB, "colormaps")
    assert isinstance(PB.colormaps, (list, tuple))
    assert hasattr(PB, "wireframe_colors")
    assert isinstance(PB.wireframe_colors, (list, tuple))
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
    assert f.data[0]["colorscale"] == ((0, 'cyan'), (1, 'cyan'))
    assert f.layout["showlegend"] == False

    p = p6(PB, contour_kw=dict(contours=dict(coloring="lines")))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Contour)
    assert f.data[0]["contours"]["coloring"] == "lines"
    assert f.data[0]["colorbar"]["title"]["text"] == str(cos(x**2 + y**2))

    p = p7(PB,
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

    p = p8(PB,
            stream_kw=dict(line_color="red"),
            contour_kw=dict(contours=dict(coloring="lines")))
    assert len(p.series) == 2
    f = p.fig
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Contour)
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[0]["contours"]["coloring"] == "lines"
    assert f.data[0]["colorbar"]["title"]["text"] == "x + y"
    assert f.data[1]["line"]["color"] == "red"

    p = p9(PB, quiver_kw=dict(sizeref=5))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Cone)
    assert f.data[0]["sizeref"] == 5
    assert f.data[0]["colorbar"]["title"]["text"] == str(Matrix([z, y, x]))

    p = p10(PB, stream_kw=dict(colorscale=[[0, "red"], [1, "red"]]))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Streamtube)
    assert f.data[0]["colorscale"] == ((0, "red"), (1, "red"))
    assert f.data[0]["colorbar"]["title"]["text"] == str(Matrix([z, y, x]))

    p = p11(PB, contour_kw=dict(colorscale = [[0, 'rgba(0,0,0,0)'], [1, 'red']]))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Heatmap)
    assert f.data[0]["colorscale"] == ((0, 'rgba(0,0,0,0)'), (1, 'red'))

    p = p12(PB, contour_kw=dict(fillcolor="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Contour)
    assert f.data[0]["fillcolor"] == "red"

class BBchild(BB):
    colorloop = ["red", "green", "blue"]

def test_BokehBackend():
    from bokeh.models.glyphs import Line, MultiLine, Image, Segment
    from bokeh.plotting.figure import Figure

    assert hasattr(BB, "colorloop")
    assert isinstance(BB.colorloop, (list, tuple))
    assert hasattr(BB, "colormaps")
    assert isinstance(BB.colormaps, (list, tuple))
    assert hasattr(BB, "contour_colormaps")
    assert isinstance(BB.contour_colormaps, (list, tuple))
    assert hasattr(BB, "quivers_colormaps")
    assert isinstance(BB.quivers_colormaps, (list, tuple))

    series = [UnsupportedSeries()]
    raises(NotImplementedError, lambda: Plot(*series, backend=PB))

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

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError, lambda: p4(BB, line_kw=dict(line_color="red")))
    raises(NotImplementedError, 
        lambda: p5(BB, surface_kw=dict(colorscale=[[0, "cyan"], [1, "cyan"]])))

    # Bokeh doesn't use contour_kw dictionary. Nothing to customize yet.
    p = p6(BB, contour_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, Image)
    assert f.right[0].title == str(cos(x**2 + y**2))

    p = p7(BB,
            contour_kw=dict(),
            quiver_kw=dict(line_color="red"))
    assert len(p.series) == 2
    f = p.fig
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, Image)
    assert isinstance(f.renderers[1].glyph, Segment)
    assert f.right[0].title == "Magnitude"
    assert f.renderers[1].glyph.line_color == "red"

    p = p8(BB,
            stream_kw=dict(line_color="red"),
            contour_kw=dict(contours=dict(coloring="lines")))
    assert len(p.series) == 2
    f = p.fig
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, Image)
    assert isinstance(f.renderers[1].glyph, MultiLine)
    assert f.right[0].title == "x + y"
    assert f.renderers[1].glyph.line_color == "red"

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError, lambda: p9(BB, quiver_kw=dict(sizeref=5)))
    raises(NotImplementedError, lambda: p10(BB,
        stream_kw=dict(colorscale=[[0, "red"], [1, "red"]])))
    
    p = p11(BB, contour_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, Image)

    p = p12(BB, contour_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, Image)

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
    assert hasattr(KB, "quivers_colormaps")
    assert isinstance(KB.quivers_colormaps, (list, tuple))

    series = [UnsupportedSeries()]
    raises(NotImplementedError, lambda: Plot(*series, backend=KBchild2))
    
    ### Setting custom color loop

    assert len(KBchild1.colorloop.colors) != len(KBchild2.colorloop)
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
    raises(NotImplementedError, lambda: p7(KBchild1,
            quiver_kw=dict(), contour_kw=dict()))
    raises(NotImplementedError, lambda: p8(KBchild1,
            stream_kw=dict(), contour_kw=dict()))


    p = p9(KBchild1, quiver_kw=dict(scale=0.5, color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], Vectors)
    assert all([c == 16711680 for c in f.objects[0].colors])

    p = p10(KBchild1, stream_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], Line)
    assert f.objects[0].color == 16711680

    # K3D doesn't support 2D plots
    raises(NotImplementedError, lambda: p11(KBchild1, contour_kw=dict()))
    raises(NotImplementedError, lambda: p12(KBchild1, contour_kw=dict()))
