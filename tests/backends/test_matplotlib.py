import matplotlib
import mpl_toolkits
import numpy as np
from pytest import raises
import os
from tempfile import TemporaryDirectory
from .make_tests import *


# NOTE
# While BB, PB, KB creates the figure at instantiation, MB creates the figure
# once the `show()` method is called. All backends do not populate the figure
# at instantiation. Numerical data is added only when `show()` or `fig` is
# called.
# In the following tests, we will use `show=False`, hence the `show()` method
# won't be executed. To add numerical data to the plots we either call `fig`
# or `process_series()`.


class MBchild(MB):
    colorloop = ["red", "green", "blue"]


def test_colorloop_colormaps():
    # verify that backends exposes important class attributes enabling
    # automatic coloring

    assert hasattr(MB, "colorloop")
    assert isinstance(MB.colorloop, (list, tuple))
    assert hasattr(MB, "colormaps")
    assert isinstance(MB.colormaps, (list, tuple))


def test_MatplotlibBackend():
    # verify that MB keeps track of the handles and a few other important
    # keyword arguments

    # `_handle` is needed in order to correctly update the data with iplot
    x, y = symbols("x, y")
    p = plot3d(cos(x**2 + y**2), backend=MB, show=False, n1=5, n2=5, use_cm=True)
    p.process_series()
    assert hasattr(p, "_handles") and isinstance(p._handles, dict)
    assert len(p._handles) == 1
    assert isinstance(p._handles[0], (tuple, list))
    assert "cmap" in p._handles[0][1].keys()


def test_custom_colorloop():
    # verify that it is possible to modify the backend's class attributes
    # in order to change custom coloring

    assert len(MBchild.colorloop) != len(MB.colorloop)
    _p1 = custom_colorloop_1(MB)
    _p2 = custom_colorloop_1(MBchild)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert len(f1.axes[0].lines) == 6
    assert len(f2.axes[0].lines) == 6
    # there are 6 unique colors in _p1 and 3 unique colors in _p2
    assert len(set([l.get_color() for l in f1.axes[0].lines])) == 6
    assert len(set([l.get_color() for l in f2.axes[0].lines])) == 3


def test_plot():
    # verify that the backends produce the expected results when `plot()`
    # is called and `rendering_kw` overrides the default line settings

    p = make_plot_1(MB, rendering_kw=dict(color="red"))
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

    p = make_plot_1(MB, rendering_kw=dict(color="red"), use_latex=True)
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_label() == "$\\sin{\\left(x \\right)}$"
    assert ax.get_xlabel() == "$x$"
    assert ax.get_ylabel() == "$f\\left(x\\right)$"


def test_plot_parametric():
    # verify that the backends produce the expected results when
    # `plot_parametric()` is called and `rendering_kw` overrides the default
    # line settings

    p = make_plot_parametric_1(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    # parametric plot with use_cm=True -> LineCollection
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == "x"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()


def test_plot3d_parametric_line():
    # verify that the backends produce the expected results when
    # `plot3d_parametric_line()` is called and `rendering_kw` overrides the
    # default line settings

    p = make_plot3d_parametric_line_1(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert f.axes[1].get_ylabel() == "x"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()


def test_plot3d():
    # verify that the backends produce the expected results when
    # `plot3d()` is called and `rendering_kw` overrides the default surface
    # settings

    # use_cm=False will force to apply a default solid color to the mesh.
    # Here, I override that solid color with a custom color.
    p = make_plot3d_1(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Poly3DCollection)
    # TODO: apparently, without showing the plot, the colors are not applied
    # to a Poly3DCollection...
    p.close()


def test_plot3d_2():
    # verify that the backends uses string labels when `plot3d()` is called
    # with `use_latex=False` and `use_cm=True`

    p = make_plot3d_2(MB)
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


def test_plot3d_wireframe():
    # verify that wireframe=True is going to add the expected number of line
    # data series and that appropriate keyword arguments work as expected

    p0 = make_plot3d_wireframe_1(MB, False)
    assert len(p0.series) == 1

    p1 = make_plot3d_wireframe_1(MB)
    assert len(p1.series) == 21

    p3 = make_plot3d_wireframe_2(MB, {"lw": "0.5"})
    assert len(p3.series) == 1 + 20 + 30
    assert all(s.rendering_kw == {"lw": "0.5"} for s in p3.series[1:])
    assert all(s.n[0] == 12 for s in p3.series[1:])

    p4 = make_plot3d_wireframe_3(MB, {"lw": "0.5"})
    assert len(p4.series) == 1 + 20 + 40
    assert all(s.rendering_kw == {"lw": "0.5"} for s in p3.series[1:])


def test_plot_contour():
    # verify that the backends produce the expected results when
    # `plot_contour()` is called and `rendering_kw` overrides the default
    # surface settings

    p = make_plot_contour_1(MB, rendering_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert f.axes[1].get_ylabel() == str(cos(x ** 2 + y ** 2))
    # TODO: how to retrieve the colormap from a contour series?????
    p.close()


def test_plot_contour_is_filled():
    # verify that is_filled=True produces different results than
    # is_filled=False
    x, y = symbols("x, y")

    p1 = make_plot_contour_is_filled(MB, True)
    p1.process_series()
    p2 = make_plot_contour_is_filled(MB, False)
    p2.process_series()
    assert p1._handles[0][-1] is None
    assert hasattr(p2._handles[0][-1], "__iter__")
    assert len(p2._handles[0][-1]) > 0


def test_plot_vector_2d_quivers():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`quiver_kw` overrides the
    # default settings

    p = make_plot_vector_2d_quiver(MB, quiver_kw=dict(color="red"),
        contour_kw=dict(cmap="jet"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert isinstance(ax.collections[-1], matplotlib.quiver.Quiver)
    assert f.axes[1].get_ylabel() == "Magnitude"
    # TODO: how to retrieve the colormap from a contour series?????
    p.close()


def test_plot_vector_2d_streamlines_custom_scalar_field():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`stream_kw` overrides the
    # default settings

    p = make_plot_vector_2d_streamlines_1(MB, stream_kw=dict(color="red"),
        contour_kw=dict(cmap="jet"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert isinstance(ax.collections[-1], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == "x + y"
    assert all(*(ax.collections[-1].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()


def test_plot_vector_2d_streamlines_custom_scalar_field_custom_label():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`stream_kw` overrides the
    # default settings

    p = make_plot_vector_2d_streamlines_2(MB, stream_kw=dict(color="red"),
        contour_kw=dict(cmap="jet"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert isinstance(ax.collections[-1], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == "test"
    assert all(*(ax.collections[-1].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()


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

    p = make_plot_vector_3d_quiver(MB, quiver_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert ax.collections[0].cmap.name == "jet"
    assert f.axes[1].get_ylabel() == str((z, y, x))
    p.close()

    p = make_plot_vector_3d_quiver(
        MB, quiver_kw=dict(cmap=None, color="red"), use_cm=False)
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert np.allclose(ax.collections[0].get_color(), np.array([[1., 0., 0., 1.]]))
    p.close()


def test_plot_vector_3d_streamlines():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `stream_kw` overrides the
    # default settings

    p = make_plot_vector_3d_streamlines_1(MB, stream_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert f.axes[1].get_ylabel() == str((z, y, x))
    p.close()

    # test different combinations for streamlines: it should not raise errors
    p = make_plot_vector_3d_streamlines_1(MB, stream_kw=dict(starts=True))
    p = make_plot_vector_3d_streamlines_1(MB, stream_kw=dict(starts={
        "x": np.linspace(-5, 5, 10),
        "y": np.linspace(-4, 4, 10),
        "z": np.linspace(-3, 3, 10),
    }))
    p.close()

    # other keywords: it should not raise errors
    p = make_plot_vector_3d_streamlines_1(
        MB, stream_kw=dict(), kwargs=dict(use_cm=False))
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_color() == '#1f77b4'
    p.close()


def test_plot_vector_2d_normalize():
    # verify that backends are capable of normalizing a vector field before
    # plotting it. Since all backend are different from each other, let's test
    # that data in the figures is different in the two cases normalize=True
    # and normalize=False

    p1 = make_plot_vector_2d_normalize_1(MB, False)
    p2 = make_plot_vector_2d_normalize_1(MB, True)
    uu1 = p1.fig.axes[0].collections[0].U
    vv1 = p1.fig.axes[0].collections[0].V
    uu2 = p2.fig.axes[0].collections[0].U
    vv2 = p2.fig.axes[0].collections[0].V
    assert not np.allclose(uu1, uu2)
    assert not np.allclose(vv1, vv2)
    assert not np.allclose(np.sqrt(uu1**2 + vv1**2), 1)
    assert np.allclose(np.sqrt(uu2**2 + vv2**2), 1)

    # interactive plots
    p1 = make_plot_vector_2d_normalize_2(MB, False)
    p1.backend._update_interactive({u: 1.5})
    p2 = make_plot_vector_2d_normalize_2(MB, True)
    p2.backend._update_interactive({u: 1.5})
    uu1 = p1.backend.fig.axes[0].collections[0].U
    vv1 = p1.backend.fig.axes[0].collections[0].V
    uu2 = p2.backend.fig.axes[0].collections[0].U
    vv2 = p2.backend.fig.axes[0].collections[0].V
    assert not np.allclose(uu1, uu2)
    assert not np.allclose(vv1, vv2)
    assert not np.allclose(np.sqrt(uu1**2 + vv1**2), 1)
    assert np.allclose(np.sqrt(uu2**2 + vv2**2), 1)


def test_plot_vector_3d_normalize():
    # verify that backends are capable of normalizing a vector field before
    # plotting it. Since all backend are different from each other, let's test
    # that data in the figures is different in the two cases normalize=True
    # and normalize=False

    p1 = make_plot_vector_3d_normalize_1(MB, False)
    p2 = make_plot_vector_3d_normalize_1(MB, True)
    seg1 = np.array(p1.fig.axes[0].collections[0].get_segments())
    seg2 = np.array(p2.fig.axes[0].collections[0].get_segments())
    # TODO: how can I test that these two quivers are different?
    # assert not np.allclose(seg1, seg2)

    # interactive plots
    p1 = make_plot_vector_3d_normalize_2(MB, False)
    p1.backend._update_interactive({u: 1.5})
    p2 = make_plot_vector_3d_normalize_2(MB, True)
    p2.backend._update_interactive({u: 1.5})
    seg1 = np.array(p1.fig.axes[0].collections[0].get_segments())
    seg2 = np.array(p2.fig.axes[0].collections[0].get_segments())
    # TODO: how can I test that these two quivers are different?
    # assert not np.allclose(seg1, seg2)


def test_plot_vector_2d_quiver_color_func():
    # verify that color_func gets applied to 2D quivers

    p1 = make_plot_vector_2d_quiver_color_func_1(MB, None)
    p2 = make_plot_vector_2d_quiver_color_func_1(MB, lambda x, y, u, v: x)
    a1 = p1.fig.axes[0].collections[0].get_array()
    a2 = p2.fig.axes[0].collections[0].get_array()
    assert not np.allclose(a1, a2)

    x, y, a = symbols("x y a")
    _pv2 = lambda B, cf: plot_vector((-a * y, x), (x, -2, 2), (y, -2, 2),
        scalar=False, use_cm=True, color_func=cf, show=False, backend=B, n=3,
        params={a: (1, 0, 2)})

    p1 = _pv2(MB, None)
    p2 = _pv2(MB, lambda x, y, u, v: u)
    p3 = _pv2(MB, lambda x, y, u, v: u)
    p3.backend._update_interactive({a: 1.5})
    a1 = p1.fig.axes[0].collections[0].get_array()
    a2 = p2.fig.axes[0].collections[0].get_array()
    a3 = p3.fig.axes[0].collections[0].get_array()
    assert (not np.allclose(a1, a2)) and (not np.allclose(a2, a3))


def test_plot_vector_2d_streamline_color_func():
    # verify that color_func gets applied to 2D streamlines

    x, y, a = symbols("x, y, a")

    _pv = lambda cf: plot_vector((-y, x), (x, -2, 2), (y, -2, 2),
        scalar=False, streamlines=True, use_cm=True, color_func=cf,
        show=False, backend=MB, n=3)

    # TODO: seems like streamline colors get applied only after the plot is
    # show... How do I perform this test?
    p1 = _pv(None)
    p2 = _pv(lambda x, y, u, v: x)
    c1 = p1.fig.axes[0].collections[0].get_colors()
    c2 = p2.fig.axes[0].collections[0].get_colors()
    # assert not np.allclose(c1, c2)


def test_plot_vector_3d_quivers_color_func():
    # verify that color_func gets applied to 3D quivers


    # TODO: is it possible to check matplotlib colors without showing the plot?
    p1 = make_plot_vector_3d_quiver_color_func_1(MB, None)
    p2 = make_plot_vector_3d_quiver_color_func_1(MB, lambda x, y, z, u, v, w: x)
    p1.process_series()
    p2.process_series()

    p1 = make_plot_vector_3d_quiver_color_func_2(MB, None)
    p2 = make_plot_vector_3d_quiver_color_func_2(MB, lambda x, y, z, u, v, w: u)
    p3 = make_plot_vector_3d_quiver_color_func_2(MB, lambda x, y, z, u, v, w: u)
    p1.backend._update_interactive({a: 0})
    p2.backend._update_interactive({a: 0})
    p3.backend._update_interactive({a: 2})


def test_plot_vector_3d_streamlines_color_func():
    # verify that color_func gets applied to 3D quivers

    # TODO: is it possible to check matplotlib colors without showing the plot?
    p1 = make_plot_vector_3d_streamlines_color_func(MB, None)
    p2 = make_plot_vector_3d_streamlines_color_func(MB, lambda x, y, z, u, v, w: x)
    p1.process_series()
    p2.process_series()


def test_plot_implicit_adaptive_true():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True`

    p = make_test_plot_implicit_adaptive_true(MB, contour_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 0
    assert len(ax.patches) == 1
    p.close()


def test_plot_implicit_adaptive_false():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True` and `contour_kw`
    # overrides the default settings

    p = make_test_plot_implicit_adaptive_false(MB, contour_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    # TODO: how to retrieve the colormap from a contour series?????
    p.close()


def test_plot_implicit_multiple_expressions():
    # verify that legend show multiple entries when multiple expressions are
    # plotted on the same plot.

    x, y = symbols("x, y")
    options = dict(adaptive=False, n=5, show=False)
    p1 = plot_implicit(x + y, x, y, **options)
    p2 = plot_implicit(x - y, x, y, **options)
    p3 = p1 + p2
    p3.process_series()
    legend = [t for t in p3.ax.get_children() if isinstance(t, matplotlib.legend.Legend)][0]
    assert len(legend.get_lines()) > 0


def test_plot_real_imag():
    # verify that the backends produce the expected results when
    # `plot_real_imag()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_real_imag(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.get_lines()) == 2
    assert ax.get_lines()[0].get_label() == "Re(sqrt(x))"
    assert ax.get_lines()[0].get_color() == "red"
    assert ax.get_lines()[1].get_label() == "Im(sqrt(x))"
    assert ax.get_lines()[1].get_color() == "red"
    p.close()


def test_plot_complex_1d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_plot_complex_1d(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == "Arg(sqrt(x))"
    assert all(*(ax.collections[0].get_color() - np.array([1.0, 0.0, 0.0, 1.0])) == 0)
    p.close()


def test_plot_complex_2d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_plot_complex_2d(MB, rendering_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.images) == 1
    assert f.axes[1].get_ylabel() == "Argument"
    assert ax.images[0].get_extent() == [-5.0, 5.0, -5.0, 5.0]
    p.close()

    p = make_test_plot_complex_2d(MB, rendering_kw=dict(extent=[-6, 6, -7, 7]))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.images) == 1
    assert f.axes[1].get_ylabel() == "Argument"
    assert ax.images[0].get_extent() == [-6, 6, -7, 7]
    p.close()


def test_plot_complex_3d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_plot_complex_3d(MB, rendering_kw=dict(color="red"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Poly3DCollection)
    # TODO: apparently, without showing the plot, the colors are not applied
    # to a Poly3DCollection...
    p.close()


def test_plot_complex_list():
    # verify that no errors are raise when plotting lists of complex points
    p = plot_complex_list(3 + 2 * I, 4 * I, 2, backend=MB, show=False)
    f = p.fig


def test_plot_list_is_filled_false():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=False`

    p = make_test_plot_list_is_filled_false(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_markeredgecolor() != ax.lines[0].get_markerfacecolor()
    p.close()


def test_plot_list_is_filled_true():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=True`

    p = make_test_plot_list_is_filled_true(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_markeredgecolor() == ax.lines[0].get_markerfacecolor()
    p.close()


def test_plot_list_color_func():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `color_func`

    p = make_test_plot_list_color_func(MB)
    f = p.fig
    ax = f.axes[0]
    # TODO:  matplotlib applie colormap color only after being shown :|
    # assert p.ax.collections[0].get_facecolors().shape == (3, 4)
    p.close()


def test_plot_piecewise_single_series():
    # verify that plot_piecewise plotting 1 piecewise composed of N
    # sub-expressions uses only 1 color

    p = make_test_plot_piecewise_single_series(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 4
    colors = set()
    for l in ax.lines:
        colors.add(l.get_color())
    assert len(colors) == 1
    assert not p.legend


def test_plot_piecewise_multiple_series():
    # verify that plot_piecewise plotting N piecewise expressions uses
    # only N different colors

    p = make_test_plot_piecewise_multiple_series(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 9
    colors = set()
    for l in ax.lines:
        colors.add(l.get_color())
    assert len(colors) == 2


def test_plot_geometry_1():
    # verify that the backends produce the expected results when
    # `plot_geometry()` is called

    p = make_test_plot_geometry_1(MB)
    assert len(p.series) == 3
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 3
    assert ax.get_lines()[0].get_label() == str(Line((1, 2), (5, 4)))
    assert ax.get_lines()[1].get_label() == str(Circle((0, 0), 4))
    assert ax.get_lines()[2].get_label() == str(Polygon((2, 2), 3, n=6))
    p.close()


def test_plot_geometry_2():
    # verify that is_filled works correctly

    p = make_test_plot_geometry_2(MB, False)
    assert len(p.fig.axes[0].lines) == 5
    assert len(p.fig.axes[0].collections) == 1
    assert len(p.fig.axes[0].patches) == 0
    p = make_test_plot_geometry_2(MB, True)
    assert len(p.fig.axes[0].lines) == 2
    assert len(p.fig.axes[0].collections) == 1
    assert len(p.fig.axes[0].patches) == 3


def test_plot_geometry_3d():
    # verify that no errors are raised when 3d geometric entities are plotted

    p = make_test_plot_geometry_3d(MB)
    p.process_series()


def test_save():
    # Verify that the save method accepts keyword arguments.

    x, y, z = symbols("x:z")
    options = dict(backend=MB, show=False, adaptive=False, n=5)

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        p = plot(sin(x), cos(x), **options)
        filename = "test_mpl_save_1.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot(sin(x), cos(x), **options)
        filename = "test_mpl_save_2.pdf"
        p.save(os.path.join(tmpdir, filename), dpi=150)
        p.close()

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), **options)
        filename = "test_mpl_save_3.png"
        p.save(os.path.join(tmpdir, filename))
        p.close()

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), **options)
        filename = "test_mpl_save_4.pdf"
        p.save(os.path.join(tmpdir, filename), dpi=150)
        p.close()

def test_vectors_3d_update_interactive():
    # Some backends do not support streamlines with iplot. Test that the
    # backends raise error.

    p = make_test_vectors_3d_update_interactive(MB)
    raises(NotImplementedError,
        lambda:p.backend._update_interactive({a: 2, b: 2, c: 2}))


def test_aspect_ratio_2d_issue_11764():
    # verify that the backends apply the provided aspect ratio.
    # NOTE: read the backend docs to understand which options are available.

    p = make_test_aspect_ratio_2d_issue_11764(MB)
    assert p.aspect == "auto"
    assert p.fig.axes[0].get_aspect() == "auto"
    p.close()

    p = make_test_aspect_ratio_2d_issue_11764(MB, (1, 1))
    assert p.aspect == (1, 1)
    assert p.fig.axes[0].get_aspect() == 1
    p.close()

    p = make_test_aspect_ratio_2d_issue_11764(MB, "equal")
    assert p.aspect == "equal"
    assert p.fig.axes[0].get_aspect() == 1
    p.close()


def test_aspect_ratio_3d():
    # verify that the backends apply the provided aspect ratio.
    # NOTE:
    # 1. read the backend docs to understand which options are available.
    # 2. K3D doesn't use the `aspect` keyword argument.
    x, y = symbols("x, y")

    p = make_test_aspect_ratio_3d(MB)
    assert p.aspect == "auto"

    # Matplotlib 3D axis requires a string-valued aspect ratio
    # depending on the version, it raises one of the following errors
    raises((NotImplementedError, ValueError),
        lambda: make_test_aspect_ratio_3d(MB, (1, 1)).process_series())


def test_plot_size():
    # verify that the keyword `size` is doing it's job
    # NOTE: K3DBackend doesn't support custom size

    x, y = symbols("x, y")

    p = make_test_plot_size(MB, (8, 4))
    s = p.fig.get_size_inches()
    assert (s[0] == 8) and (s[1] == 4)
    p.close()

    p = make_test_plot_size(MB, (10, 5))
    s = p.fig.get_size_inches()
    assert (s[0] == 10) and (s[1] == 5)
    p.close()


def test_plot_scale_lin_log():
    # verify that backends are applying the correct scale to the axes
    # NOTE: none of the 3D libraries currently support log scale.

    x, y = symbols("x, y")

    p = make_test_plot_scale_lin_log(MB, "linear", "linear")
    assert p.fig.axes[0].get_xscale() == "linear"
    assert p.fig.axes[0].get_yscale() == "linear"
    p.close()

    p = make_test_plot_scale_lin_log(MB, "log", "linear")
    assert p.fig.axes[0].get_xscale() == "log"
    assert p.fig.axes[0].get_yscale() == "linear"
    p.close()

    p = make_test_plot_scale_lin_log(MB, "linear", "log")
    assert p.fig.axes[0].get_xscale() == "linear"
    assert p.fig.axes[0].get_yscale() == "log"
    p.close()


def test_backend_latex_labels():
    # verify that backends are going to set axis latex-labels in the
    # 2D and 3D case

    p1 = make_test_backend_latex_labels_1(MB, True)
    p2 = make_test_backend_latex_labels_1(MB, False)
    assert p1.xlabel == p1.fig.axes[0].get_xlabel() == '$x^{2}_{1}$'
    assert p2.xlabel == p2.fig.axes[0].get_xlabel() == 'x_1^2'
    assert p1.ylabel == p1.fig.axes[0].get_ylabel() == '$f\\left(x^{2}_{1}\\right)$'
    assert p2.ylabel == p2.fig.axes[0].get_ylabel() == 'f(x_1^2)'

    p1 = make_test_backend_latex_labels_2(MB, True)
    p2 = make_test_backend_latex_labels_2(MB, False)
    assert p1.xlabel == p1.fig.axes[0].get_xlabel() == '$x^{2}_{1}$'
    assert p1.ylabel == p1.fig.axes[0].get_ylabel() == '$x_{2}$'
    assert p1.zlabel == p1.fig.axes[0].get_zlabel() == '$f\\left(x^{2}_{1}, x_{2}\\right)$'
    assert p2.xlabel == p2.fig.axes[0].get_xlabel() == 'x_1^2'
    assert p2.ylabel == p2.fig.axes[0].get_ylabel() == 'x_2'
    assert p2.zlabel == p2.fig.axes[0].get_zlabel() == 'f(x_1^2, x_2)'


def test_plot_use_latex():
    # verify that the backends produce the expected results when `plot()`
    # is called and `rendering_kw` overrides the default line settings

    p = make_test_plot_use_latex(MB)
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_label() == "$\\sin{\\left(x \\right)}$"
    assert ax.get_lines()[1].get_label() == "$\\cos{\\left(x \\right)}$"
    p.close()


def test_plot_parametric_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_parametric_use_latex(MB)
    assert len(p.series) == 1
    f = p.fig
    assert f.axes[1].get_ylabel() == "$x$"
    p.close()


def test_plot_contour_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_contour_use_latex(MB)
    assert len(p.series) == 1
    f = p.fig
    assert f.axes[1].get_ylabel() == "$%s$" % latex(cos(x ** 2 + y ** 2))


def test_plot3d_parametric_line_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot3d_parametric_line_use_latex(MB)
    assert len(p.series) == 1
    f = p.fig
    assert f.axes[1].get_ylabel() == "$x$"
    p.close()


def test_plot3d_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot3d_use_latex(MB)
    f = p.fig
    ax = f.axes[0]
    assert len(f.axes) == 3
    assert f.axes[1].get_ylabel() == "$%s$" % latex(cos(x ** 2 + y ** 2))
    assert f.axes[2].get_ylabel() == "$%s$" % latex(sin(x ** 2 + y ** 2))
    p.close()


def test_plot_vector_2d_quivers_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_2d_quivers_use_latex(MB)
    f = p.fig
    assert f.axes[1].get_ylabel() == "Magnitude"
    p.close()


def test_plot_vector_2d_streamlines_custom_scalar_field_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_2d_streamlines_custom_scalar_field_use_latex(MB)
    f = p.fig
    assert f.axes[1].get_ylabel() == "$x + y$"
    p.close()


def test_plot_vector_2d_streamlines_custom_scalar_field_custom_label_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_2d_streamlines_custom_scalar_field_custom_label_use_latex(MB)
    f = p.fig
    assert f.axes[1].get_ylabel() == "test"


def test_plot_vector_2d_use_latex_colorbar():
    # verify that the colorbar uses latex label

    # contours + quivers: 1 colorbar for the contours
    p = make_test_plot_vector_2d_use_latex_colorbar(MB, True, False)
    assert p.fig.axes[1].get_ylabel() == "Magnitude"
    p.close()

    # contours + streamlines: 1 colorbar for the contours
    p = make_test_plot_vector_2d_use_latex_colorbar(MB, True, True)
    assert p.fig.axes[1].get_ylabel() == "Magnitude"
    p.close()

    # only quivers: 1 colorbar for the quivers
    p = make_test_plot_vector_2d_use_latex_colorbar(MB, False, False)
    assert p.fig.axes[1].get_ylabel() == "$\\left( x, \\  y\\right)$"
    p.close()

    # only streamlines: 1 colorbar for the streamlines
    p = make_test_plot_vector_2d_use_latex_colorbar(MB, False, True)
    assert p.fig.axes[1].get_ylabel() == "$\\left( x, \\  y\\right)$"
    p.close()


def test_plot_vector_3d_quivers_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_3d_quivers_use_latex(MB)
    assert len(p.fig.axes) == 2
    assert p.fig.axes[1].get_ylabel() == '$\\left( z, \\  y, \\  x\\right)$'
    p.close()


def test_plot_vector_3d_streamlines_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_3d_streamlines_use_latex(MB)
    assert p.fig.axes[1].get_ylabel() == '$\\left( z, \\  y, \\  x\\right)$'
    p.close()


def test_plot_complex_use_latex():
    # complex plot function should return the same result (for axis labels)
    # wheter use_latex is True or False

    p = make_test_plot_complex_use_latex_1(MB)
    assert p.fig.axes[0].get_xlabel() == "Real"
    assert p.fig.axes[0].get_ylabel() == "Abs"
    assert p.fig.axes[1].get_ylabel() == 'Arg(cos(x) + I*sinh(x))'
    p.close()

    p = make_test_plot_complex_use_latex_2(MB)
    assert p.fig.axes[0].get_xlabel() == "Re"
    assert p.fig.axes[0].get_ylabel() == "Im"
    assert p.fig.axes[1].get_ylabel() == 'Argument'
    p.close()


def test_plot_real_imag_use_latex():
    # real/imag plot function should return the same result (for axis labels)
    # wheter use_latex is True or False

    p = make_test_plot_real_imag_use_latex(MB)
    assert p.fig.axes[0].get_xlabel() == "$x$"
    assert p.fig.axes[0].get_ylabel() == r"$f\left(x\right)$"
    assert p.fig.axes[0].lines[0].get_label() == 'Re(sqrt(x))'
    assert p.fig.axes[0].lines[1].get_label() == 'Im(sqrt(x))'
    p.close()


def test_plot3d_use_cm():
    # verify that use_cm produces the expected results on plot3d

    x, y = symbols("x, y")
    p1 = make_test_plot3d_use_cm(MB, True)
    p2 = make_test_plot3d_use_cm(MB, False)
    p1.process_series()
    p2.process_series()
    assert "cmap" in p1._handles[0][1].keys()
    assert "cmap" not in p2._handles[0][1].keys()


def test_plot3d_update_interactive():
    # verify that MB._update_interactive applies the original color/colormap
    # each time it gets called
    # Since matplotlib doesn't apply color/colormaps until the figure is shown,
    # verify that the correct keyword arguments are into _handles.

    x, y, u = symbols("x, y, u")

    s = SurfaceOver2DRangeSeries(
        u * cos(x**2 + y**2),
        (x, -5, 5), (y, -5, 5),
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

    s = SurfaceOver2DRangeSeries(
        u * cos(x**2 + y**2),
        (x, -5, 5), (y, -5, 5),
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


def test_plot_polar():
    # verify that 2D polar plot can create plots with cartesian axis and
    #  polar axis

    # test for cartesian axis
    p1 = make_test_plot_polar(MB, False)
    assert not isinstance(p1.fig.axes[0], matplotlib.projections.polar.PolarAxes)

    # polar axis
    p1 = make_test_plot_polar(MB, True)
    assert isinstance(p1.fig.axes[0], matplotlib.projections.polar.PolarAxes)


def test_plot_polar_use_cm():
    # verify the correct behavior of plot_polar when color_func
    # or use_cm are applied

    # cartesian axis, no colormap
    p = make_test_plot_polar_use_cm(MB, False, False)
    assert len(p.ax.lines) > 0
    assert len(p.ax.collections) == 0

    # cartesian axis, with colormap
    p = make_test_plot_polar_use_cm(MB, False, True)
    assert len(p.ax.lines) == 0
    assert len(p.ax.collections) > 0

    # polar axis, no colormap
    p = make_test_plot_polar_use_cm(MB, True, False)
    assert len(p.ax.lines) > 0
    assert len(p.ax.collections) == 0

    # polar axis, with colormap
    p = make_test_plot_polar_use_cm(MB, True, True, lambda t: t)
    assert len(p.ax.lines) == 0
    assert len(p.ax.collections) > 0


def test_plot3d_implicit():
    # verify that plot3d_implicit don't raise errors

    raises(NotImplementedError,
        lambda : make_test_plot3d_implicit(MB).process_series())


def test_surface_color_func():
    # After the addition of `color_func`, `SurfaceOver2DRangeSeries` and
    # `ParametricSurfaceSeries` returns different elements.
    # Verify that backends do not raise errors when plotting surfaces and that
    # the color function is applied.

    p1 = make_test_surface_color_func_1(MB, lambda x, y, z: z)
    p1.process_series()
    p2 = make_test_surface_color_func_1(MB, lambda x, y, z: np.sqrt(x**2 + y**2))
    p2.process_series()

    p1 = make_test_surface_color_func_2(MB, lambda x, y, z, u, v: z)
    p1.process_series()
    p2 = make_test_surface_color_func_2(MB, lambda x, y, z, u, v: np.sqrt(x**2 + y**2))
    p2.process_series()


def test_surface_interactive_color_func():
    # After the addition of `color_func`, `SurfaceInteractiveSeries` and
    # `ParametricSurfaceInteractiveSeries` returns different elements.
    # Verify that backends do not raise errors when updating surfaces and a
    # color function is applied.

    p = make_test_surface_interactive_color_func(MB)
    p.process_series()
    p._update_interactive({t: 2})


def test_line_color_func():
    # Verify that backends do not raise errors when plotting lines and that
    # the color function is applied.

    p1 = make_test_line_color_func(MB, None)
    p1.process_series()
    p2 = make_test_line_color_func(MB, lambda x, y: np.cos(x))
    p2.process_series()
    assert len(p1.fig.axes[0].lines) == 1
    assert isinstance(p2.fig.axes[0].collections[0], matplotlib.collections.LineCollection)
    assert np.allclose(p2.fig.axes[0].collections[0].get_array(), np.cos(np.linspace(-3, 3, 5)))


def test_line_interactive_color_func():
    # Verify that backends do not raise errors when updating lines and a
    # color function is applied.

    p = make_test_line_interactive_color_func(MB)
    p.process_series()
    p._update_interactive({t: 2})
    assert len(p.fig.axes[0].lines) == 1
    assert isinstance(p.fig.axes[0].collections[0], matplotlib.collections.LineCollection)
    assert np.allclose(p.fig.axes[0].collections[0].get_array(), np.cos(np.linspace(-3, 3, 5)))


def test_line_color_plot():
    # verify back-compatibility with old sympy.plotting module when using
    # line_color

    p = make_test_line_color_plot(MB, "red")
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_color() == "red"
    p = make_test_line_color_plot(MB, lambda x: -x)
    f = p.fig
    assert len(p.fig.axes) == 2 # there is a colorbar


def test_line_color_plot3d_parametric_line():
    # verify back-compatibility with old sympy.plotting module when using
    # line_color

    p = make_test_line_color_plot3d_parametric_line(MB, "red", False)
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_color() == "red"
    p = make_test_line_color_plot3d_parametric_line(MB, lambda x: -x, True)
    f = p.fig
    assert len(p.fig.axes) == 2 # there is a colorbar


def test_surface_color_plot3d():
    # verify back-compatibility with old sympy.plotting module when using
    # surface_color

    p = make_test_surface_color_plot3d(MB, "red", False)
    assert len(p.fig.axes) == 1
    p = make_test_surface_color_plot3d(MB, lambda x, y, z: -x, True)
    assert len(p.fig.axes) == 2


def test_label_after_plot_instantiation():
    # verify that it is possible to set a label after a plot has been created
    x = symbols("x")

    p = plot(sin(x), cos(x), show=False, backend=MB, adaptive=False, n=5)
    p[0].label = "a"
    p[1].label = "$b^{2}$"
    f = p.fig
    assert f.axes[0].lines[0].get_label() == "a"
    assert f.axes[0].lines[1].get_label() == "$b^{2}$"


def test_update_interactive():
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


def test_axis_center():
    # verify that axis_center doesn't raise any errors

    x = symbols("x")
    _plot = lambda ac: plot(sin(x), adaptive=False, n=5, backend=MB,
        show=False, axis_center=ac)

    _plot("center").process_series()
    _plot("auto").process_series()
    _plot((0, 0)).process_series()


def test_plot3d_list_use_cm_False():
    # verify that plot3d_list produces the expected results when no color map
    # is required

    # solid color line
    p = make_test_plot3d_list_use_cm_False(MB, False, False)
    p.process_series()
    assert len(p.series) == 1
    assert len(p.ax.lines) == 1
    assert p.ax.lines[0].get_color() == '#1f77b4'

    # solid color markers with empty faces
    p = make_test_plot3d_list_use_cm_False(MB, True, False)
    p.process_series()
    assert len(p.ax.collections) == 1
    assert p.ax.collections[0].get_facecolors().size == 0

    # solid color markers with filled faces
    p = make_test_plot3d_list_use_cm_False(MB, True, True)
    p.process_series()
    assert len(p.ax.collections) == 1
    assert p.ax.collections[0].get_facecolors().size > 0


def test_plot3d_list_use_cm_color_func():
    # verify that use_cm=True and color_func do their job

    # line with colormap
    # if color_func is not provided, the same parameter will be used
    # for all points
    p1 = make_test_plot3d_list_use_cm_color_func(MB, False, False, None)
    p1.process_series()
    c1 = p1.ax.collections[0].get_array()
    p2 = make_test_plot3d_list_use_cm_color_func(MB, False, False, lambda x, y, z: x)
    p2.process_series()
    c2 = p2.ax.collections[0].get_array()
    assert not np.allclose(c1, c2)

    # markers with empty faces
    p1 = make_test_plot3d_list_use_cm_color_func(MB, True, False, None)
    p1.process_series()
    c1 = p1.ax.collections[0].get_array()
    p2 = make_test_plot3d_list_use_cm_color_func(MB, False, False, lambda x, y, z: x)
    p2.process_series()
    c2 = p2.ax.collections[0].get_array()
    assert not np.allclose(c1, c2)


def test_plot3d_list_interactive():
    # verify that no errors are raises while updating a plot3d_list

    p = make_test_plot3d_list_interactive(MB)
    p.backend._update_interactive({t: 1})


def test_contour_and_3d():
    # verify that it's possible to combine contour and 3d plots, but that
    # combining a 2d line plot with contour and 3d plot raises an error.

    x, y = symbols("x, y")
    expr = cos(x * y) * exp(-0.05 * (x**2 + y**2))
    ranges = (x, -5, 5), (y, -5, 5)

    p1 = plot3d(expr, *ranges,
        show=False, legend=True, zlim=(-2, 1), n=4)
    p2 = plot_contour(
        expr, *ranges,
        {"zdir": "z", "offset": -2, "levels": 5},
        show=False, is_filled=False, legend=True, n=4)
    p3 = plot(cos(x), (x, 0, 2*pi), adaptive=False, n=5, show=False)

    p = p1 + p2
    p.process_series()
    p = p2 + p1
    p.process_series()
    p = p2 + p3
    p.process_series()
    p = p1 + p3
    raises(ValueError, lambda : p.process_series())
    p = p1 + p2 + p3
    raises(ValueError, lambda : p.process_series())
    p = p2 + p1 + p3
    raises(ValueError, lambda : p.process_series())
