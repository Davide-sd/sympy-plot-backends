import matplotlib
import mpl_toolkits
import numpy as np
import pytest
from pytest import raises, warns
from matplotlib.axes import Axes
from sympy import Symbol, symbols
import os
from tempfile import TemporaryDirectory
from spb import (
    MB, plot, plot_riemann_sphere, plot_real_imag, plot_complex,
    plot_vector, plot3d_revolution, plot3d_spherical,
    plot3d_parametric_surface, plot_contour, plot3d, plot3d_parametric_line,
    plot_parametric, plot_implicit, plot_list, plot_geometry,
    plot_complex_list, graphics, vector_field_2d, plot_nyquist, plot_nichols,
    plot_step_response, multiples_of_pi_over_3, multiples_of_pi_over_4,
    tick_formatter_multiples_of, line
)
from spb.series import (
    RootLocusSeries, SGridLineSeries, ZGridLineSeries, ContourSeries,
    Vector2DSeries
)
from spb.series import SurfaceOver2DRangeSeries
from sympy import (
    sin, cos, I, pi, Eq, exp, Circle, Polygon, sqrt, Matrix, Line, Segment,
    latex, log
)
from sympy.abc import x, y, z, u, t, a, b, c
from sympy.external import import_module
from .make_tests import (
    custom_colorloop_1,
    make_test_plot,
    make_test_plot_parametric,
    make_test_plot3d_parametric_line,
    make_test_plot3d,
    make_plot3d_wireframe_1,
    make_plot3d_wireframe_2,
    make_plot3d_wireframe_3,
    make_test_plot_contour,
    make_plot_contour_is_filled,
    make_test_plot_vector_2d_quiver,
    make_test_plot_vector_2d_streamlines,
    make_test_plot_vector_3d_quiver_streamlines,
    make_test_plot_vector_2d_normalize,
    make_test_plot_vector_3d_normalize,
    make_test_plot_vector_2d_color_func,
    make_test_plot_vector_3d_quiver_color_func,
    make_test_plot_vector_3d_streamlines_color_func,
    make_test_plot_implicit_adaptive_true,
    make_test_plot_implicit_adaptive_false,
    make_test_plot_complex_1d,
    make_test_plot_complex_2d,
    make_test_plot_complex_3d,
    make_test_plot_list_is_filled,
    make_test_plot_piecewise_single_series,
    make_test_plot_piecewise_multiple_series,
    make_test_plot_geometry_1,
    make_test_plot_geometry_2,
    make_test_plot_geometry_3d,
    make_test_aspect_ratio_2d_issue_11764,
    make_test_aspect_ratio_3d,
    make_test_plot_size,
    make_test_plot_scale_lin_log,
    make_test_backend_latex_labels_1,
    make_test_backend_latex_labels_2,
    make_test_plot_polar,
    make_test_plot_polar_use_cm,
    make_test_plot3d_implicit,
    make_test_surface_color_func,
    make_test_line_color_plot,
    make_test_line_color_plot3d_parametric_line,
    make_test_surface_color_plot3d,
    make_test_plot3d_list,
    make_test_contour_show_clabels,
    make_test_color_func_expr_1,
    make_test_color_func_expr_2,
    make_test_legend_plot_sum_1,
    make_test_legend_plot_sum_2,
    make_test_domain_coloring_2d,
    make_test_show_in_legend_2d,
    make_test_show_in_legend_3d,
    make_test_analytic_landscape,
    make_test_detect_poles,
    make_test_detect_poles_interactive,
    make_test_plot_riemann_sphere,
    make_test_parametric_texts_2d,
    make_test_parametric_texts_3d,
    make_test_line_color_func,
    make_test_plot_list_color_func,
    make_test_real_imag,
    make_test_arrow_2d,
    make_test_arrow_3d,
    make_test_root_locus_1,
    make_test_root_locus_2,
    make_test_root_locus_3,
    make_test_root_locus_4,
    make_test_plot_pole_zero,
    make_test_poles_zeros_sgrid,
    make_test_ngrid,
    make_test_sgrid,
    make_test_zgrid,
    make_test_mcircles,
    make_test_hvlines,
    make_test_tick_formatters_2d,
    make_test_tick_formatters_3d,
    make_test_tick_formatter_polar_axis,
    make_test_hooks_2d,
    make_test_hline_vline_label
)

ct = import_module("control")
ipy = import_module("ipywidgets")
scipy = import_module("scipy")
vtk = import_module("vtk", catch=(RuntimeError,))


# NOTE
# While BB, PB, KB creates the figure at instantiation, MB creates the figure
# once the `show()` method is called. All backends do not populate the figure
# at instantiation. Numerical data is added only when `show()` or `fig` is
# called.
# In the following tests, we will use `show=False`, hence the `show()` method
# won't be executed. To add numerical data to the plots we either call `fig`
# or `draw()`.


class MBchild(MB):
    colorloop = ["red", "green", "blue"]


def test_MatplotlibBackend():
    # verify that MB keeps track of the handles and a few other important
    # keyword arguments

    x, y = symbols("x, y")
    p = plot3d(
        cos(x**2 + y**2),
        backend=MB, show=False, n1=5, n2=5, use_cm=True
    )
    p.draw()
    assert len(p.renderers) == 1
    assert isinstance(p.renderers[0].handles[0], (tuple, list))
    assert "cmap" in p.renderers[0].handles[0][1].keys()


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


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
@pytest.mark.parametrize(
    "use_latex, xlabel, ylabel", [
        (False, "x", "f(x)"),
        (True, "$x$", "$f\\left(x\\right)$")
    ]
)
def test_plot_1(use_latex, xlabel, ylabel, label_func):
    # verify that the backends produce the expected results when `plot()`
    # is called and `rendering_kw` overrides the default line settings

    p = make_test_plot(MB, rendering_kw=dict(color="red"), use_latex=use_latex)
    assert len(p.backend.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.get_lines()) == 2
    assert ax.get_lines()[0].get_label() == label_func(use_latex, sin(a * x))
    assert ax.get_lines()[0].get_color() == "red"
    assert ax.get_lines()[1].get_label() == label_func(use_latex, cos(b * x))
    assert ax.get_lines()[1].get_color() == "red"
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == ylabel
    p.backend.update_interactive({a: 2, b: 2})
    p.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_plot_parametric():
    # verify that the backends produce the expected results when
    # `plot_parametric()` is called and `rendering_kw` overrides the default
    # line settings

    p = make_test_plot_parametric(MB, rendering_kw=dict(color="red"),
        use_cm=False)
    assert len(p.backend.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_label() == "(cos(a*x), sin(b*x))"
    assert ax.get_lines()[0].get_color() == "red"
    p.backend.update_interactive({a: 2, b: 2})
    p.backend.close()

    # parametric plot with use_cm=True -> LineCollection
    p1 = make_test_plot_parametric(MB, rendering_kw={},
        use_cm=True)
    p2 = make_test_plot_parametric(MB, rendering_kw=dict(cmap="autumn"),
        use_cm=True)
    f1, f2 = p1.fig, p2.fig
    ax1, ax2 = f1.axes[0], f2.axes[0]
    assert len(ax1.collections) == 1
    assert isinstance(ax1.collections[0], matplotlib.collections.LineCollection)
    assert f1.axes[1].get_ylabel() == "x"
    # TODO: how to test for different colormaps?
    # assert not np.allclose(
    #     ax1.collections[0].get_colors(),
    #     ax2.collections[0].get_colors()
    # )
    p1.backend.update_interactive({a: 2, b: 2})
    p2.backend.update_interactive({a: 2, b: 2})
    p1.backend.close()
    p2.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
@pytest.mark.parametrize(
    "use_latex", [False, True]
)
def test_plot3d_parametric_line(use_latex, label_func):
    # verify that the backends produce the expected results when
    # `plot3d_parametric_line()` is called and `rendering_kw` overrides the
    # default line settings

    p = make_test_plot3d_parametric_line(
        MB, rendering_kw=dict(color="red"), use_latex=use_latex, use_cm=False)
    assert len(p.backend.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_label() == label_func(
        use_latex, (cos(a * x), sin(b * x), x))
    assert ax.get_lines()[0].get_color() == "red"
    p.backend.update_interactive({a: 2, b: 2})
    p.backend.close()

    p2 = make_test_plot3d_parametric_line(
        MB, rendering_kw={}, use_latex=use_latex, use_cm=True)
    p1 = make_test_plot3d_parametric_line(
        MB, rendering_kw=dict(cmap="autumn"), use_latex=use_latex, use_cm=True)
    f1, f2 = p1.fig, p2.fig
    ax1, ax2 = f1.axes[0], f2.axes[0]
    assert len(ax1.collections) == 1
    assert isinstance(
        ax1.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert f1.axes[1].get_ylabel() == f2.axes[1].get_ylabel() == label_func(
        use_latex, x
    )
    # TODO: how to test for different colormaps?
    p1.backend.update_interactive({a: 2, b: 2})
    p2.backend.update_interactive({a: 2, b: 2})
    p1.backend.close()
    p2.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
@pytest.mark.parametrize(
    "use_latex, xl, yl, zl", [
        (False, "x", "y", "f(x, y)"),
        (True, "$x$", "$y$", r"$f\left(x, y\right)$")
    ]
)
def test_plot3d_1(use_latex, xl, yl, zl, label_func):
    # verify that the backends produce the expected results when
    # `plot3d()` is called and `rendering_kw` overrides the default surface
    # settings

    # use_cm=False will force to apply a default solid color to the mesh.
    # Here, I override that solid color with a custom color.
    p = make_test_plot3d(MB, rendering_kw=dict(color="red"), use_cm=False,
        use_latex=use_latex)
    assert len(p.backend.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 2
    assert all(isinstance(
        c, mpl_toolkits.mplot3d.art3d.Poly3DCollection) for c in ax.collections
    )
    assert ax.get_xlabel() == xl
    assert ax.get_ylabel() == yl
    assert ax.get_zlabel() == zl
    assert (
        ax.get_legend().legend_handles[0].get_label()
        == label_func(use_latex, cos(a*x**2 + y**2))
    )
    assert (
        ax.get_legend().legend_handles[1].get_label()
        == label_func(use_latex, sin(b*x**2 + y**2))
    )
    assert "cmap" not in p.backend.renderers[0].handles[0][1].keys()
    # TODO: how to test for different colormaps?
    p.backend.update_interactive({a: 2, b: 2})
    p.backend.close()


    p = make_test_plot3d(MB, rendering_kw=dict(cmap="autumn"), use_cm=True,
        use_latex=use_latex)
    f = p.fig
    assert f.axes[1].get_ylabel() == label_func(use_latex, cos(a*x**2 + y**2))
    assert f.axes[2].get_ylabel() == label_func(use_latex, sin(b*x**2 + y**2))
    # TODO: how to test for different colormaps?
    assert "cmap" in p.backend.renderers[0].handles[0][1].keys()
    p.backend.update_interactive({a: 2, b: 2})
    p.backend.close()


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


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
@pytest.mark.parametrize(
    "use_latex, xl, yl", [
        (False, "x", "y"),
        (True, "$x$", "$y$")
    ]
)
def test_plot_contour(use_latex, xl, yl, label_func):
    # verify that the backends produce the expected results when
    # `plot_contour()` is called and `rendering_kw` overrides the default
    # surface settings

    p = make_test_plot_contour(MB, rendering_kw=dict(cmap="jet"),
        use_latex=use_latex)
    assert len(p.backend.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert ax.get_xlabel() == xl
    assert ax.get_ylabel() == yl
    assert "cmap" in p.backend.renderers[0].handles[0][1].keys()
    assert f.axes[1].get_ylabel() == label_func(use_latex, cos(a*x**2 + y**2))
    # TODO: how to test for different colormaps?
    p.backend.update_interactive({a: 2})
    p.backend.close()


def test_plot_contour_is_filled():
    # verify that is_filled=True produces different results than
    # is_filled=False
    x, y = symbols("x, y")

    p1 = make_plot_contour_is_filled(MB, True)
    p1.draw()
    p2 = make_plot_contour_is_filled(MB, False)
    p2.draw()
    assert p1.renderers[0].handles[0][-1] is None
    assert hasattr(p2.renderers[0].handles[0][-1], "__iter__")
    assert len(p2.renderers[0].handles[0][-1]) > 0


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_plot_vector_2d_quivers():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`quiver_kw` overrides the
    # default settings

    p = make_test_plot_vector_2d_quiver(
        MB, quiver_kw=dict(color="red"), contour_kw=dict(cmap="jet")
    )
    assert len(p.backend.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert isinstance(ax.collections[-1], matplotlib.quiver.Quiver)
    assert f.axes[1].get_ylabel() == "Magnitude"
    # TODO: how to test for different colormaps?
    p.backend.update_interactive({a: 2})
    p.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
@pytest.mark.parametrize(
    "scalar, use_latex, expected_label", [
        (True, False, "Magnitude"),
        (True, True, "Magnitude"),
        (x + y, False, "x + y"),
        (x + y, True, "$x + y$"),
        ([(x + y), "test"], False, "test"),
        ([(x + y), "test"], True, "test")
    ]
)
def test_plot_vector_2d_streamlines_custom_scalar_field(
    scalar, use_latex, expected_label
):
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`stream_kw` overrides the
    # default settings

    p = make_test_plot_vector_2d_streamlines(
        MB, stream_kw=dict(color="red"), contour_kw=dict(cmap="jet"),
        scalar=scalar, use_latex=use_latex
    )
    assert len(p.backend.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    assert isinstance(ax.collections[-1], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == expected_label
    # TODO: how to test for different colormaps?
    raises(NotImplementedError, lambda :p.backend.update_interactive({a: 2}))
    p.backend.close()


@pytest.mark.parametrize(
    "scalar, streamlines, use_cm, n_series, n_axes, n_collections, "
    "use_latex, label, greater", [
        # contours + quivers: 1 colorbar for the contours
        (True, False, None, 2, 2, 1, False, "Magnitude", True),
        (True, False, None, 2, 2, 1, True, "Magnitude", True),
        # contours + streamlines: 1 colorbar for the contours
        (True, True, None, 2, 2, 1, False, "Magnitude", True),
        (True, True, None, 2, 2, 1, True, "Magnitude", True),
        # only quivers: 1 colorbar for the quivers
        (False, False, None, 1, 2, 1, False, "(x, y)", False),
        (False, False, None, 1, 2, 1, True, r"$\left( x, \  y\right)$", False),
        # only streamlines: 1 colorbar for the streamlines
        (False, True, None, 1, 2, 1, False, "(x, y)", False),
        (False, True, None, 1, 2, 1, True, r"$\left( x, \  y\right)$", False),
        # only quivers with solid color
        (False, False, False, 1, 1, 1, False, "", False),
        (False, False, False, 1, 1, 1, True, "", False),
        # only streamlines with solid color
        (False, True, False, 1, 1, 1, False, "", False),
        (False, True, False, 1, 1, 1, True, "", False),
    ]
)
def test_plot_vector_2d_matplotlib(
    scalar, streamlines, use_cm, n_series, n_axes, n_collections,
    use_latex, label, greater
):
    # verify that when scalar=False, quivers/streamlines comes together with
    # a colorbar

    x, y = symbols("x, y")
    kwargs = {"scalar": scalar, "streamlines": streamlines}
    if use_cm is not None:
        kwargs["use_cm"] = use_cm
    p = plot_vector(
        Matrix([x, y]), (x, -5, 5), (y, -4, 4),
        backend=MB, show=False, use_latex=use_latex, n1=5, n2=8,
        **kwargs
    )

    # contours + quivers: 1 colorbar for the contours
    assert len(p.series) == n_series
    assert len(p.fig.axes) == n_axes
    if greater:
        assert len(p.fig.axes[0].collections) > n_collections
    else:
        assert len(p.fig.axes[0].collections) == n_collections
    idx = 1 if use_cm is None else 0
    assert p.fig.axes[idx].get_ylabel() == label


def test_vector_2d_multiple_series():
    # In the following example there is one contour series and 2 vector series
    # using solid colors. There should be two entries on the legend.

    x, y = symbols("x, y")
    scalar_expr = sqrt((-sin(y))**2 + cos(x)**2)

    g = graphics(
        vector_field_2d(-sin(y), cos(x), (x, -5, 5), (y, -3, 3), n=10, nc=10,
            scalar=[scalar_expr, "$%s$" % latex(scalar_expr)],
            contour_kw={"cmap": "summer"},
            quiver_kw={"color": "k"}),
        vector_field_2d(2 * y, x, (x, -5, 5), (y, -3, 3), n=10,
            scalar=False, quiver_kw={"color": "r"}, use_cm=False),
        aspect="equal", grid=False, xlabel="x", ylabel="y", show=False)

    assert len(g.ax.get_legend().legend_handles) == 2


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
@pytest.mark.parametrize(
    "use_latex", [True, False]
)
def test_plot_vector_3d_quivers(use_latex, label_func):
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `quiver_kw` overrides the
    # default settings

    p = make_test_plot_vector_3d_quiver_streamlines(
        MB, False, quiver_kw=dict(cmap="jet"), use_latex=use_latex)
    assert len(p.backend.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(
        ax.collections[0],
        mpl_toolkits.mplot3d.art3d.Line3DCollection
    )
    assert ax.collections[0].cmap.name == "jet"
    assert f.axes[1].get_ylabel() == label_func(use_latex, (a * z, y, x))
    p.backend.update_interactive({a: 2})
    p.backend.close()

    p = make_test_plot_vector_3d_quiver_streamlines(
        MB, False, quiver_kw=dict(cmap=None, color="red"), use_cm=False
    )
    assert len(p.backend.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(
        ax.collections[0],
        mpl_toolkits.mplot3d.art3d.Line3DCollection
    )
    assert np.allclose(
        ax.collections[0].get_color(),
        np.array([[1.0, 0.0, 0.0, 1.0]])
        )
    p.backend.update_interactive({a: 2})
    p.backend.close()


@pytest.mark.skipif(vtk is None, reason="vtk is not installed")
@pytest.mark.parametrize(
    "use_latex", [True, False]
)
def test_plot_vector_3d_streamlines(use_latex, label_func):
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `stream_kw` overrides the
    # default settings

    p = make_test_plot_vector_3d_quiver_streamlines(
        MB, True, stream_kw=dict(), use_latex=use_latex)
    assert len(p.backend.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], mpl_toolkits.mplot3d.art3d.Line3DCollection)
    assert f.axes[1].get_ylabel() == label_func(use_latex, (a*z, y, x))
    raises(
        NotImplementedError,
        lambda: p.backend.update_interactive({a: 2})
    )
    p.backend.close()

    # test different combinations for streamlines: it should not raise errors
    p = make_test_plot_vector_3d_quiver_streamlines(
        MB, True, stream_kw=dict(starts=True))
    p.backend.close()
    p = make_test_plot_vector_3d_quiver_streamlines(
        MB, True,
        stream_kw=dict(
            starts={
                "x": np.linspace(-5, 5, 10),
                "y": np.linspace(-4, 4, 10),
                "z": np.linspace(-3, 3, 10),
            }
        ),
    )
    p.backend.close()

    # other keywords: it should not raise errors
    p = make_test_plot_vector_3d_quiver_streamlines(
        MB, True, stream_kw=dict(), use_cm=False
    )
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_color() == "#1f77b4"
    p.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_plot_vector_2d_normalize():
    # verify that backends are capable of normalizing a vector field before
    # plotting it. Since all backend are different from each other, let's test
    # that data in the figures is different in the two cases normalize=True
    # and normalize=False
    p1 = make_test_plot_vector_2d_normalize(MB, False)
    p1.backend.update_interactive({u: 1.5})
    p2 = make_test_plot_vector_2d_normalize(MB, True)
    p2.backend.update_interactive({u: 1.5})
    uu1 = p1.backend.fig.axes[0].collections[0].U
    vv1 = p1.backend.fig.axes[0].collections[0].V
    uu2 = p2.backend.fig.axes[0].collections[0].U
    vv2 = p2.backend.fig.axes[0].collections[0].V
    assert not np.allclose(uu1, uu2)
    assert not np.allclose(vv1, vv2)
    assert not np.allclose(np.sqrt(uu1**2 + vv1**2), 1)
    assert np.allclose(np.sqrt(uu2**2 + vv2**2), 1)
    p1.backend.close()
    p2.backend.close()


@pytest.mark.skipif(ct is None, reason="control is not installed")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_plot_vector_3d_normalize():
    # verify that backends are capable of normalizing a vector field before
    # plotting it. Since all backend are different from each other, let's test
    # that data in the figures is different in the two cases normalize=True
    # and normalize=False

    p1 = make_test_plot_vector_3d_normalize(MB, False)
    p1.backend.update_interactive({u: 1.5})
    p2 = make_test_plot_vector_3d_normalize(MB, True)
    p2.backend.update_interactive({u: 1.5})
    seg1 = np.array(p1.fig.axes[0].collections[0].get_segments())
    seg2 = np.array(p2.fig.axes[0].collections[0].get_segments())
    # TODO: how can I test that these two quivers are different?
    # assert not np.allclose(seg1, seg2)
    p1.backend.close()
    p2.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_plot_vector_2d_quiver_color_func():
    # verify that color_func gets applied to 2D quivers

    p1 = make_test_plot_vector_2d_color_func(MB, False, None)
    p2 = make_test_plot_vector_2d_color_func(MB, False, lambda x, y, u, v: u)
    p3 = make_test_plot_vector_2d_color_func(MB, False, lambda x, y, u, v: u)
    p3.backend.update_interactive({a: 1.5})
    a1 = p1.fig.axes[0].collections[0].get_array()
    a2 = p2.fig.axes[0].collections[0].get_array()
    a3 = p3.fig.axes[0].collections[0].get_array()
    assert (not np.allclose(a1, a2)) and (not np.allclose(a2, a3))
    p1.backend.close()
    p2.backend.close()
    p3.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_plot_vector_2d_streamline_color_func():
    # verify that color_func gets applied to 2D streamlines

    # TODO: seems like streamline colors get applied only after the plot is
    # show... How do I perform this test?
    p1 = make_test_plot_vector_2d_color_func(MB, True, None)
    p2 = make_test_plot_vector_2d_color_func(MB, True, lambda x, y, u, v: x)
    c1 = p1.fig.axes[0].collections[0].get_colors()
    c2 = p2.fig.axes[0].collections[0].get_colors()
    # assert not np.allclose(c1, c2)
    p1.backend.close()
    p2.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_plot_vector_3d_quivers_color_func_interactive():
    # verify that color_func gets applied to 3D quivers

    p1 = make_test_plot_vector_3d_quiver_color_func(MB, None)
    p2 = make_test_plot_vector_3d_quiver_color_func(
        MB, lambda x, y, z, u, v, w: np.cos(u))
    # TODO: is it possible to check matplotlib colors without showing the plot?
    p1.backend.update_interactive({a: 2})
    p2.backend.update_interactive({a: 2})
    p1.backend.close()
    p2.backend.close()


@pytest.mark.skipif(vtk is None, reason="vtk is not installed")
def test_plot_vector_3d_streamlines_color_func():
    # verify that color_func gets applied to 3D quivers

    # TODO: is it possible to check matplotlib colors without showing the plot?
    p1 = make_test_plot_vector_3d_streamlines_color_func(MB, None)
    p2 = make_test_plot_vector_3d_streamlines_color_func(
        MB, lambda x, y, z, u, v, w: x)
    p1.fig
    p2.fig
    raises(NotImplementedError, lambda: p1.backend.update_interactive({a: 2}))
    raises(NotImplementedError, lambda: p2.backend.update_interactive({a: 2}))


def test_plot_implicit_adaptive_true():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True`

    p = make_test_plot_implicit_adaptive_true(MB, rendering_kw=dict(color="r"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 0
    assert len(ax.patches) == 1
    assert ax.patches[0].get_facecolor() == (1, 0, 0, 1)
    p.close()


def test_plot_implicit_adaptive_false():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True` and `contour_kw`
    # overrides the default settings

    p = make_test_plot_implicit_adaptive_false(
        MB, rendering_kw=dict(cmap="jet"))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) > 0
    # TODO: how to test for different colormaps?
    p.close()


def test_plot_implicit_multiple_expressions():
    # verify that legend show multiple entries when multiple expressions are
    # plotted on the same plot.

    x, y = symbols("x, y")
    options = dict(adaptive=False, n=5, show=False)
    p1 = plot_implicit(x + y, x, y, **options)
    p2 = plot_implicit(x - y, x, y, **options)
    p3 = p1 + p2
    p3.draw()
    legend = [
        t for t in p3.ax.get_children()
        if isinstance(t, matplotlib.legend.Legend)
    ][0]
    assert len(legend.get_lines()) > 0


@pytest.mark.parametrize(
    "use_latex", [True, False]
)
def test_plot_real_imag(use_latex, label_func):
    # verify that the backends produce the expected results when
    # `plot_real_imag()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_real_imag(MB, rendering_kw=dict(color="red"),
        use_latex=use_latex)
    assert len(p.series) == 2
    f = p.fig
    ax = f.axes[0]
    assert len(ax.get_lines()) == 2
    assert ax.get_lines()[0].get_label() == "Re(sqrt(x))"
    assert ax.get_lines()[0].get_color() == "red"
    assert ax.get_lines()[1].get_label() == "Im(sqrt(x))"
    assert ax.get_lines()[1].get_color() == "red"
    assert ax.get_xlabel() == label_func(use_latex, x)
    assert ax.get_ylabel() == (r"$f\left(x\right)$" if use_latex else "f(x)")

    p.close()


@pytest.mark.parametrize(
    "use_latex", [True, False]
)
def test_plot_complex_1d(use_latex):
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_plot_complex_1d(
        MB, rendering_kw=dict(cmap="autumn"), use_latex=use_latex)
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], matplotlib.collections.LineCollection)
    assert f.axes[1].get_ylabel() == "Arg(sqrt(x))"
    assert f.axes[0].get_xlabel() == "Real"
    assert f.axes[0].get_ylabel() == "Abs"
    # TODO: how to test for different colormaps?
    p.close()


@pytest.mark.parametrize(
    "use_latex", [True, False]
)
def test_plot_complex_2d(use_latex):
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_plot_complex_2d(MB, rendering_kw=dict(), use_latex=use_latex)
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.images) == 1
    assert f.axes[0].get_xlabel() == "Re"
    assert f.axes[0].get_ylabel() == "Im"
    assert f.axes[1].get_ylabel() == "Argument"
    assert ax.images[0].get_extent() == [-5.0, 5.0, -5.0, 5.0]
    p.close()

    p = make_test_plot_complex_2d(MB, rendering_kw=dict(extent=[-6, 6, -7, 7]))
    assert len(p.series) == 1
    f = p.fig
    ax = f.axes[0]
    assert len(ax.images) == 1
    assert f.axes[0].get_xlabel() == "Re"
    assert f.axes[0].get_ylabel() == "Im"
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
    assert isinstance(
        ax.collections[0], mpl_toolkits.mplot3d.art3d.Poly3DCollection)
    # TODO: apparently, without showing the plot, the colors are not applied
    # to a Poly3DCollection...
    p.close()


def test_plot_complex_list():
    # verify that no errors are raise when plotting lists of complex points
    p = plot_complex_list(3 + 2 * I, 4 * I, 2, backend=MB, show=False)
    p.fig


@pytest.mark.parametrize(
    "is_filled", [True, False]
)
def test_plot_list_is_filled(is_filled):
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=False`

    p = make_test_plot_list_is_filled(MB, is_filled)
    f = p.fig
    ax = f.axes[0]
    assert len(ax.lines) == 1
    test = ax.lines[0].get_markeredgecolor() == ax.lines[0].get_markerfacecolor()
    assert test is is_filled
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
    p.close()


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
    p.close()


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
    assert len(p.ax.get_legend().legend_handles) == 3
    p.close()


@pytest.mark.parametrize(
    "is_filled, n_lines, n_coll, n_patches, n_legend", [
        (False, 5, 1, 0, 5),
        (True, 2, 1, 3, 5),
    ]
)
def test_plot_geometry_2(is_filled, n_lines, n_coll, n_patches, n_legend):
    # verify that is_filled works correctly

    p = make_test_plot_geometry_2(MB, is_filled)
    assert len(p.fig.axes[0].lines) == n_lines
    assert len(p.fig.axes[0].collections) == n_coll
    assert len(p.fig.axes[0].patches) == n_patches
    assert len(p.ax.get_legend().legend_handles) == n_legend
    p.close()


def test_plot_geometry_3d():
    # verify that no errors are raised when 3d geometric entities are plotted

    p = make_test_plot_geometry_3d(MB)
    p.draw()
    p.close()


def test_plot_geometry_rendering_kw():
    # verify that rendering_kw works fine
    p = plot_geometry(
        Segment((0, 0), (1, 0)), "r", {"color": "red"},
        show=False
    )
    assert p[0].rendering_kw == {"color": "red"}
    p.draw()
    assert p.ax.lines[0].get_color() == "red"
    p.close()


def test_save():
    # Verify that the save method accepts keyword arguments.

    x, y, z = symbols("x:z")
    options = dict(backend=MB, show=False, n=5)

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


@pytest.mark.parametrize(
    "aspect, expected", [
        ("auto", "auto"),
        ((1, 1), 1),
        ("equal", 1),
    ]
)
def test_aspect_ratio_2d_issue_11764(aspect, expected):
    # verify that the backends apply the provided aspect ratio.
    # NOTE: read the backend docs to understand which options are available.

    p = make_test_aspect_ratio_2d_issue_11764(MB, aspect)
    assert p.aspect == aspect
    assert p.fig.axes[0].get_aspect() == expected
    p.close()


def test_aspect_ratio_3d():
    # verify that the backends apply the provided aspect ratio.
    # NOTE:
    # 1. read the backend docs to understand which options are available.
    # 2. K3D doesn't use the `aspect` keyword argument.
    x, y = symbols("x, y")

    p = make_test_aspect_ratio_3d(MB)
    assert p.aspect == "auto"
    p.close()

    # Matplotlib 3D axis requires a string-valued aspect ratio
    # depending on the version, it raises one of the following errors
    raises(
        (NotImplementedError, ValueError),
        lambda: make_test_aspect_ratio_3d(MB, (1, 1)).draw(),
    )


def test_plot_size():
    # verify that the keyword `size` is doing it's job

    x, y = symbols("x, y")

    p = make_test_plot_size(MB, (8, 4))
    s = p.fig.get_size_inches()
    assert (s[0] == 8) and (s[1] == 4)
    p.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "xscale, yscale", [
        ("linear", "linear"),
        ("log", "linear"),
        ("linear", "log"),
        ("log", "log"),
    ]
)
def test_plot_scale_lin_log(xscale, yscale):
    # verify that backends are applying the correct scale to the axes
    # NOTE: none of the 3D libraries currently support log scale.

    x, y = symbols("x, y")

    p = make_test_plot_scale_lin_log(MB, xscale, yscale)
    assert p.fig.axes[0].get_xscale() == xscale
    assert p.fig.axes[0].get_yscale() == yscale
    p.close()


def test_backend_latex_labels():
    # verify that backends are going to set axis latex-labels in the
    # 2D and 3D case

    p1 = make_test_backend_latex_labels_1(MB, True)
    p2 = make_test_backend_latex_labels_1(MB, False)
    assert p1.xlabel == p1.fig.axes[0].get_xlabel() == "$x^{2}_{1}$"
    assert p2.xlabel == p2.fig.axes[0].get_xlabel() == "x_1^2"
    assert p1.ylabel == p1.fig.axes[0].get_ylabel() == "$f\\left(x^{2}_{1}\\right)$"
    assert p2.ylabel == p2.fig.axes[0].get_ylabel() == "f(x_1^2)"
    p1.close()
    p2.close()

    p1 = make_test_backend_latex_labels_2(MB, True)
    p2 = make_test_backend_latex_labels_2(MB, False)
    assert p1.xlabel == p1.fig.axes[0].get_xlabel() == "$x^{2}_{1}$"
    assert p1.ylabel == p1.fig.axes[0].get_ylabel() == "$x_{2}$"
    assert (
        p1.zlabel == p1.fig.axes[0].get_zlabel() == "$f\\left(x^{2}_{1}, x_{2}\\right)$"
    )
    assert p2.xlabel == p2.fig.axes[0].get_xlabel() == "x_1^2"
    assert p2.ylabel == p2.fig.axes[0].get_ylabel() == "x_2"
    assert p2.zlabel == p2.fig.axes[0].get_zlabel() == "f(x_1^2, x_2)"
    p1.close()
    p2.close()


def test_plot3d_update_interactive():
    # verify that MB._update_interactive applies the original color/colormap
    # each time it gets called
    # Since matplotlib doesn't apply color/colormaps until the figure is shown,
    # verify that the correct keyword arguments are into _handles.

    x, y, u = symbols("x, y, u")

    s = SurfaceOver2DRangeSeries(
        u * cos(x**2 + y**2),
        (x, -5, 5),
        (y, -5, 5),
        "test",
        use_cm=False,
        params={u: 1},
        n1=3,
        n2=3,
    )
    p = MB(s, show=False)
    p.draw()
    kw, _, _ = p.renderers[0].handles[0][1:]
    c1 = kw["color"]
    p.update_interactive({u: 2})
    kw, _, _ = p.renderers[0].handles[0][1:]
    c2 = kw["color"]
    assert c1 == c2
    p.close()

    s = SurfaceOver2DRangeSeries(
        u * cos(x**2 + y**2),
        (x, -5, 5),
        (y, -5, 5),
        "test",
        use_cm=True,
        params={u: 1},
        n1=3,
        n2=3,
    )
    p = MB(s, show=False)
    p.draw()
    kw, _, _ = p.renderers[0].handles[0][1:]
    c1 = kw["cmap"]
    p.update_interactive({u: 2})
    kw, _, _ = p.renderers[0].handles[0][1:]
    c2 = kw["cmap"]
    assert c1 == c2
    p.close()


def test_plot_polar():
    # verify that 2D polar plot can create plots with cartesian axis and
    #  polar axis

    # test for cartesian axis
    p1 = make_test_plot_polar(MB, False)
    assert not isinstance(
        p1.fig.axes[0], matplotlib.projections.polar.PolarAxes)
    p1.close()

    # polar axis
    p1 = make_test_plot_polar(MB, True)
    assert isinstance(p1.fig.axes[0], matplotlib.projections.polar.PolarAxes)
    p1.close()


def test_plot_polar_use_cm():
    # verify the correct behavior of plot_polar when color_func
    # or use_cm are applied

    # cartesian axis, no colormap
    p = make_test_plot_polar_use_cm(MB, False, False)
    assert len(p.ax.lines) > 0
    assert len(p.ax.collections) == 0
    p.close()

    # cartesian axis, with colormap
    p = make_test_plot_polar_use_cm(MB, False, True)
    assert len(p.ax.lines) == 0
    assert len(p.ax.collections) > 0
    p.close()

    # polar axis, no colormap
    p = make_test_plot_polar_use_cm(MB, True, False)
    assert len(p.ax.lines) > 0
    assert len(p.ax.collections) == 0
    p.close()

    # polar axis, with colormap
    p = make_test_plot_polar_use_cm(MB, True, True, lambda t: t)
    assert len(p.ax.lines) == 0
    assert len(p.ax.collections) > 0
    p.close()


def test_plot3d_implicit():
    # verify that plot3d_implicit don't raise errors

    raises(NotImplementedError, lambda: make_test_plot3d_implicit(MB).draw())


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_surface_color_func():
    # After the addition of `color_func`, `SurfaceOver2DRangeSeries` and
    # `ParametricSurfaceSeries` returns different elements.
    # Verify that backends do not raise errors when plotting surfaces and that
    # the color function is applied.

    p = make_test_surface_color_func(MB)
    fig = p.fig
    p.backend.update_interactive({t: 2})
    p.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_line_color_func():
    # Verify that backends do not raise errors when updating lines and a
    # color function is applied.

    p = make_test_line_color_func(MB)
    p.backend.draw()
    p.backend.update_interactive({t: 2})
    assert len(p.fig.axes[0].lines) == 1
    assert isinstance(
        p.fig.axes[0].collections[0], matplotlib.collections.LineCollection
    )
    assert np.allclose(
        p.fig.axes[0].collections[0].get_array(),
        np.cos(np.linspace(-3, 3, 5))
    )
    p.backend.close()


def test_line_color_plot():
    # verify back-compatibility with old sympy.plotting module when using
    # line_color

    p = make_test_line_color_plot(MB, "red")
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_color() == "red"
    p.close()
    p = make_test_line_color_plot(MB, lambda x: -x)
    f = p.fig
    assert len(p.fig.axes) == 2  # there is a colorbar
    p.close()


def test_line_color_plot3d_parametric_line():
    # verify back-compatibility with old sympy.plotting module when using
    # line_color

    p = make_test_line_color_plot3d_parametric_line(MB, "red", False)
    f = p.fig
    ax = f.axes[0]
    assert ax.get_lines()[0].get_color() == "red"
    p = make_test_line_color_plot3d_parametric_line(MB, lambda x: -x, True)
    f = p.fig
    assert len(p.fig.axes) == 2  # there is a colorbar


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

    p = plot(sin(x), cos(x), show=False, backend=MB, n=5)
    p[0].label = "a"
    p[1].label = "$b^{2}$"
    f = p.fig
    assert f.axes[0].lines[0].get_label() == "a"
    assert f.axes[0].lines[1].get_label() == "$b^{2}$"


def test_min_install():
    # quick round down of test to verify that ordinay plots don't
    # raise errors. Useful to test minimum installation of the module

    x, y, z = symbols("x:z")
    options = dict(n=5, backend=MB, show=False)

    p = plot(sin(x), (x, -pi, pi), **options)
    p.draw()
    p.close()

    p = plot_parametric(
        cos(x), sin(x), (x, 0, 2 * pi),
        use_cm=True, is_scatter=False, **options
    )
    p.draw()
    p.close()

    p = plot_parametric(
        cos(x), sin(x), (x, 0, 2 * pi),
        use_cm=True, is_scatter=True, **options
    )
    p.draw()
    p.close()

    p = plot_parametric(
        cos(x), sin(x), (x, 0, 2 * pi),
        use_cm=False, **options
    )
    p.draw()
    p.close()

    p = plot_implicit(
        x**2 + y**2 - 4, (x, -5, 5), (y, -5, 5),
        adaptive=False, **options
    )
    p.draw()
    p.close()

    # points
    p = plot3d_parametric_line(
        cos(x), sin(x), x, (x, -pi, pi),
        is_scatter=True, **options
    )
    p.draw()
    p.close()

    # line with colormap
    p = plot3d_parametric_line(
        cos(x), sin(x), x, (x, -pi, pi),
        is_scatter=False, use_cm=True, **options
    )
    p.draw()
    p.close()

    # line with solid color
    p = plot3d_parametric_line(
        cos(x), sin(x), x, (x, -pi, pi),
        is_scatter=False, use_cm=False, **options
    )
    p.draw()
    p.close()

    p = plot3d_parametric_line(
        cos(x), sin(x), x, (x, -pi, pi),
        is_scatter=False, use_cm=True, **options
    )
    p.draw()
    p.close()

    p = plot3d(
        cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        **options
    )
    p.draw()
    p.close()

    p = plot_contour(
        cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        is_filled=False, **options
    )
    p.draw()
    p.close()

    p = plot_contour(
        cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        is_filled=True, **options
    )
    p.draw()
    p.close()

    u, v = symbols("u, v")
    fx = (1 + v / 2 * cos(u / 2)) * cos(u)
    fy = (1 + v / 2 * cos(u / 2)) * sin(u)
    fz = v / 2 * sin(u / 2)
    p = plot3d_parametric_surface(
        fx, fy, fz, (u, 0, 2 * pi), (v, -1, 1),
        use_cm=True, **options
    )
    p.draw()
    p.close()

    p = plot_vector(
        Matrix([-y, x]), (x, -5, 5), (y, -4, 4),
        scalar=True, **options
    )
    p.draw()
    p.close()

    p = plot_vector(
        Matrix([-y, x]), (x, -5, 5), (y, -4, 4),
        scalar=False, **options
    )
    p.draw()
    p.close()

    p = plot_vector(
        Matrix([z, y, x]), (x, -5, 5), (y, -4, 4), (z, -3, 3),
        **options
    )
    p.draw()
    p.close()

    p = plot_complex(
        sqrt(x), (x, -5 - 5 * I, 5 + 5 * I),
        threed=False, **options
    )
    p.draw()
    p.close()

    p = plot_complex(
        sqrt(x), (x, -5 - 5 * I, 5 + 5 * I),
        threed=True, use_cm=True, **options
    )
    p.draw()
    p.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_update_interactive():
    # quick round down of test to verify that update_interactive doesn't
    # raise errors

    u, v, x, y, z = symbols("u, v, x:z")

    u, v = symbols("u, v")
    fx = (1 + v / 2 * cos(u / 2)) * cos(x * u)
    fy = (1 + v / 2 * cos(u / 2)) * sin(x * u)
    fz = v / 2 * sin(u / 2)
    p = plot3d_parametric_surface(
        fx, fy, fz, (u, 0, 2 * pi), (v, -1, 1),
        backend=MB, use_cm=True, n1=5, n2=5, show=False,
        params={x: (1, 0, 2)},
    )
    p.backend.draw()
    p.backend.update_interactive({x: 2})
    p.backend.close()

    p = plot_complex(
        sqrt(u * x), (x, -5 - 5 * I, 5 + 5 * I),
        show=False, backend=MB, threed=False, n=5,
        params={u: (1, 0, 2)},
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})
    p.backend.close()

    p = plot_complex(
        sqrt(u * x), (x, -5 - 5 * I, 5 + 5 * I),
        show=False, backend=MB, threed=True, use_cm=True, n=5,
        params={u: (1, 0, 2)},
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})
    p.backend.close()

    from sympy.geometry import Line as SymPyLine

    p = plot_geometry(
        SymPyLine((u, 2), (5, 4)),
        Circle((0, 0), u),
        Polygon((2, u), 3, n=6),
        backend=MB,
        show=False,
        is_filled=False,
        use_latex=False,
        params={u: (1, 0, 2)},
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})
    p.backend.update_interactive({u: 3})
    p.backend.close()

    p = plot_geometry(
        SymPyLine((u, 2), (5, 4)),
        Circle((0, 0), u),
        Polygon((2, u), 3, n=6),
        backend=MB,
        show=False,
        is_filled=True,
        use_latex=False,
        params={u: (1, 0, 2)},
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})
    p.backend.update_interactive({u: 3})
    p.backend.close()


def test_generic_data_series():
    # verify that backends do not raise errors when generic data series
    # are used

    x = symbols("x")
    p = plot(
        x,
        backend=MB,
        show=False,
        n=5,
        markers=[{"args": [[0, 1], [0, 1]], "marker": "*", "linestyle": "none"}],
        annotations=[{"text": "test", "xy": (0, 0)}],
        fill=[{"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3]}],
        rectangles=[{"xy": (0, 0), "width": 5, "height": 1}],
    )
    p.draw()


@pytest.mark.parametrize("ac", ["center", "auto", (0, 0)])
def test_axis_center(ac):
    # verify that axis_center doesn't raise any errors

    x = symbols("x")
    p = plot(
        sin(x),
        n=5,
        backend=MB, show=False, axis_center=ac
    )
    p.draw()
    p.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_plot3d_list():
    # verify that no errors are raises while updating a plot3d_list

    p = make_test_plot3d_list(MB, False, None)
    ax = p.backend.ax
    d1 = p.backend[0].get_data()
    assert len(ax.lines) == 1
    assert len(ax.collections) == 2
    c1, c2 = ax.collections
    arr1 = c1.get_array()
    # TODO: when use_cm=True, is_filled=False, this shouldn't happen. Bug.
    assert len(c1.get_edgecolor()) == len(c1.get_facecolor())
    assert len(c2.get_edgecolor()) != len(c2.get_facecolor())
    p.backend.update_interactive({t: 1})
    p.backend.close()

    p = make_test_plot3d_list(MB, True, None)
    ax = p.backend.ax
    d2 = p.backend[0].get_data()
    assert len(ax.lines) == 1
    assert len(ax.collections) == 2
    c1, c2 = ax.collections
    arr2 = c1.get_array()
    assert len(c1.get_edgecolor()) == len(c1.get_facecolor())
    assert len(c2.get_edgecolor()) == len(c2.get_facecolor())
    p.backend.update_interactive({t: 1})
    p.backend.close()

    p = make_test_plot3d_list(MB, False, lambda x, y, z: x)
    ax = p.backend.ax
    d3 = p.backend[0].get_data()
    assert len(ax.lines) == 1
    assert len(ax.collections) == 2
    c1, c2 = ax.collections
    arr3 = c1.get_array()
    # TODO: when use_cm=True, is_filled=False, this shouldn't happen. Bug.
    assert len(c1.get_edgecolor()) == len(c1.get_facecolor())
    assert len(c2.get_edgecolor()) != len(c2.get_facecolor())
    p.backend.update_interactive({t: 1})
    p.backend.close()

    assert len(d1) != len(d3)
    assert not np.allclose(arr1, arr3)

    p = make_test_plot3d_list(MB, True, lambda x, y, z: x)
    ax = p.backend.ax
    d4 = p.backend[0].get_data()
    assert len(ax.lines) == 1
    assert len(ax.collections) == 2
    c1, c2 = ax.collections
    arr4 = c1.get_array()
    # TODO: when use_cm=True, is_filled=False, this shouldn't happen. Bug.
    assert len(c1.get_edgecolor()) == len(c1.get_facecolor())
    assert len(c2.get_edgecolor()) == len(c2.get_facecolor())
    p.backend.update_interactive({t: 1})
    p.backend.close()

    assert len(d2) != len(d4)
    assert not np.allclose(arr2, arr4)


def test_contour_and_3d():
    # verify that it's possible to combine contour and 3d plots, but that
    # combining a 2d line plot with contour and 3d plot raises an error.

    x, y = symbols("x, y")
    expr = cos(x * y) * exp(-0.05 * (x**2 + y**2))
    ranges = (x, -5, 5), (y, -5, 5)

    p1 = plot3d(expr, *ranges, show=False, legend=True, zlim=(-2, 1), n=4)
    p2 = plot_contour(
        expr,
        *ranges,
        {"zdir": "z", "offset": -2, "levels": 5},
        show=False,
        is_filled=False,
        legend=True,
        n=4
    )
    p3 = plot(cos(x), (x, 0, 2 * pi), n=5, show=False)

    p = p1 + p2
    p.draw()
    p = p2 + p1
    p.draw()
    p = p2 + p3
    with warns(UserWarning, match="The following kwargs were not used by contour"):
        p.draw()
    p.close()
    p = p1 + p3
    raises(ValueError, lambda: p.draw())
    p.close()
    p = p1 + p2 + p3
    raises(ValueError, lambda: p.draw())
    p.close()
    p = p2 + p1 + p3
    raises(ValueError, lambda: p.draw())
    p.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_contour_show_clabels():
    p = make_test_contour_show_clabels(MB, False)
    p.backend.update_interactive({a: 2})
    assert len(p.backend.ax.texts) == 0
    p.backend.close()

    p = make_test_contour_show_clabels(MB, True)
    p.backend.update_interactive({a: 2})
    assert len(p.backend.ax.texts) > 0
    p.backend.close()


@pytest.mark.filterwarnings("ignore:The provided expression contains Boolean functions")
def test_plot_implicit_legend_artists():
    # verify that plot_implicit sets appropriate plot artists

    # 2 expressions plotted with contour lines -> 2 lines in legend
    V, t, b, L = symbols("V, t, b, L")
    L_array = [5, 10]
    b_val = 0.0032
    expr = b * V * 0.277 * t - b * L - log(1 + b * V * 0.277 * t)
    expr_list = [expr.subs({b: b_val, L: L_val}) for L_val in L_array]
    labels = ["L = %s" % L_val for L_val in L_array]
    p = plot_implicit(
        *expr_list, (t, 0, 3), (V, 0, 1000),
        n=50, label=labels, show=False, backend=MB
    )
    assert len(p.ax.get_legend().get_lines()) == 2
    assert len(p.ax.get_legend().get_patches()) == 0
    p.close()

    # 2 expressions plotted with contourf -> 2 rectangles in legend
    p = plot_implicit(
        y > x**2,
        y < -(x**2) + 1,
        (x, -5, 5),
        grid=False,
        backend=MB,
        n=20,
        show=False,
    )
    assert len(p.ax.get_legend().get_lines()) == 0
    assert len(p.ax.get_legend().get_patches()) == 2
    p.close()

    # two expressions plotted with fill -> 2 rectangles in legend
    p = plot_implicit(
        Eq(y, sin(x)) & (y > 0),
        Eq(y, sin(x)) & (y < 0),
        (x, -2 * pi, 2 * pi),
        (y, -4, 4),
        backend=MB,
        show=False,
    )
    assert len(p.ax.get_legend().get_lines()) == 0
    assert len(p.ax.get_legend().get_patches()) == 2
    p.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_color_func_expr():
    # verify that passing an expression to color_func is supported

    p1 = make_test_color_func_expr_1(MB, False)
    p2 = make_test_color_func_expr_1(MB, True)
    p3 = make_test_color_func_expr_2(MB)

    # compute the original figure: no errors should be raised
    p1.fig
    p2.fig
    p3.fig

    # update the figure with new parameters: no errors should be raised
    p1.backend.update_interactive({u: 0.5})
    p1.backend.close()
    # interactive plots with streamlines are not implemented
    raises(
        NotImplementedError,
        lambda: p2.backend.update_interactive({u: 0.5})
    )
    p2.backend.close()
    p3.backend.update_interactive({u: 0.5})
    p3.backend.close()


def test_legend_plot_sum():
    # when summing up plots together, the first plot dictates if legend
    # is visible or not

    # first case: legend is specified on the first plot
    # if legend is not specified, the resulting plot will show the legend
    p = make_test_legend_plot_sum_1(MB, None)
    assert len(p.ax.get_legend().legend_handles) == 3
    p.close()
    p = make_test_legend_plot_sum_1(MB, True)
    assert len(p.ax.get_legend().legend_handles) == 3
    p.close()
    # first plot has legend=False: output plot won't show the legend
    p = make_test_legend_plot_sum_1(MB, False)
    assert p.ax.get_legend() is None
    p.close()

    # second case: legend is specified on the second plot
    # the resulting plot will always show the legend
    p = make_test_legend_plot_sum_2(MB, None)
    assert len(p.ax.get_legend().legend_handles) == 3
    p.close()
    p = make_test_legend_plot_sum_2(MB, True)
    assert len(p.ax.get_legend().legend_handles) == 3
    p.close()
    p = make_test_legend_plot_sum_2(MB, False)
    assert len(p.ax.get_legend().legend_handles) == 3
    p.close()

    # because plot_implicit creates custom proxy artists to show on the legend,
    # need to make sure that every legend artists is shown when combining
    # plot_implicit with some other plot type.

    x, y = symbols("x, y")
    p1 = plot_implicit(
        x**2 + y**2 - 1,
        "plot_implicit",
        (x, -1.2, 1.2),
        (y, -1.2, 1.5),
        legend=True,
        aspect="equal",
        show=False,
    )
    p2 = plot_list(
        [0],
        [1],
        "point",
        legend=True,
        is_scatter=True,
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.5),
        aspect="equal",
        show=False,
    )
    p3 = p1 + p2
    handles = p3.ax.get_legend().legend_handles
    assert len(handles) == 2
    p3.close()


def test_domain_coloring_2d():
    # verify that at_infinity=True flips the image

    p1 = make_test_domain_coloring_2d(MB, False)
    _, _, _, _, img1a, _ = p1[0].get_data()
    img1b = p1.ax.images[0].get_array()
    assert np.allclose(img1a, img1b)
    p1.close()

    p2 = make_test_domain_coloring_2d(MB, True)
    _, _, _, _, img2a, _ = p2[0].get_data()
    img2b = p2.ax.images[0].get_array()
    assert np.allclose(img2b, np.flip(np.flip(img2a, axis=0), axis=1))
    p2.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore:The following keyword arguments are unused.")
@pytest.mark.filterwarnings("ignore:NumPy is unable to evaluate with complex numbers")
def test_show_hide_colorbar():
    x, y, z = symbols("x, y, z")
    options = dict(use_cm=True, n=5, backend=MB, show=False)

    p = lambda c: plot_parametric(
        cos(x), sin(x), (x, 0, 2 * pi),
        colorbar=c, **options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1
    p = lambda c: plot_parametric(
        (cos(x), sin(x)),
        (cos(x) / 2, sin(x) / 2),
        (x, 0, 2 * pi),
        colorbar=c,
        **options
    )
    assert len(p(True).fig.axes) == 3
    assert len(p(False).fig.axes) == 1

    p = lambda c: plot(cos(x), color_func=lambda t: t, colorbar=c, **options)
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1

    p = lambda c: plot3d_parametric_line(
        cos(x), sin(x), x, (x, 0, 2 * pi), colorbar=c, **options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1
    p = lambda c: plot3d_parametric_line(
        (cos(x), sin(x), x),
        (cos(x) / 2, sin(x) / 2, x),
        (x, 0, 2 * pi),
        colorbar=c,
        **options
    )
    assert len(p(True).fig.axes) == 3
    assert len(p(False).fig.axes) == 1

    p = lambda c: plot3d(
        cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), colorbar=c, **options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1
    p = lambda c: plot3d(
        cos(x**2 + y**2), x * y, (x, -pi, pi), (y, -pi, pi),
        colorbar=c, **options
    )
    assert len(p(True).fig.axes) == 3
    assert len(p(False).fig.axes) == 1

    p = lambda c: plot_contour(
        cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), colorbar=c, **options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1

    p = lambda c: plot3d_parametric_surface(
        x * cos(y),
        x * sin(y),
        x * cos(4 * y) / 2,
        (x, 0, pi),
        (y, 0, 2 * pi),
        colorbar=c,
        **options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1

    p = lambda c: plot3d_spherical(
        1, (x, 0, 0.7 * pi), (y, 0, 1.8 * pi), colorbar=c, **options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1

    p = lambda c: plot3d_revolution(cos(t), (t, 0, pi), colorbar=c, **options)
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1

    # in plot_vector, use_cm is not set by default.
    mod_options = options.copy()
    mod_options.pop("use_cm")
    p = lambda c: plot_vector(
        [sin(x - y), cos(x + y)], (x, -3, 3), (y, -3, 3),
        colorbar=c, **mod_options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1
    p = lambda c: plot_vector(
        [sin(x - y), cos(x + y)],
        (x, -3, 3),
        (y, -3, 3),
        scalar=False,
        colorbar=c,
        **mod_options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1
    p = lambda c: plot_vector(
        [sin(x - y), cos(x + y)],
        (x, -3, 3),
        (y, -3, 3),
        scalar=False,
        streamlines=True,
        colorbar=c,
        **options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1

    p = lambda c: plot_vector(
        [z, y, x], (x, -10, 10), (y, -10, 10), (z, -10, 10),
        colorbar=c, **mod_options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1

    p = lambda c: plot_complex(
        cos(x) + sin(I * x), "f", (x, -2, 2), colorbar=c, **options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1
    p = lambda c: plot_complex(
        sin(z), (z, -3 - 3j, 3 + 3j), colorbar=c, **options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1
    p = lambda c: plot_complex(
        sin(z), (z, -3 - 3j, 3 + 3j), colorbar=c, threed=True, **options
    )
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1

    p = lambda c: plot_real_imag(
        sqrt(x), (x, -3 - 3j, 3 + 3j), threed=True, colorbar=c, **options
    )
    assert len(p(True).fig.axes) == 3
    assert len(p(False).fig.axes) == 1

    expr = (z - 1) / (z**2 + z + 1)
    p = lambda c: plot_riemann_sphere(expr, threed=True, colorbar=c, **options)
    assert len(p(True).fig.axes) == 2
    assert len(p(False).fig.axes) == 1


def test_show_in_legend():
    # verify that ability of hiding traces from the legend

    p1, p2 = make_test_show_in_legend_2d(MB)
    p3, p4 = make_test_show_in_legend_3d(MB)

    assert len(p1.ax.get_legend().legend_handles) == 2
    assert len(p2.ax.get_legend().legend_handles) == 2
    assert len(p3.ax.get_legend().legend_handles) == 2
    assert len(p4.ax.get_legend().legend_handles) == 2
    p1.close()
    p2.close()
    p3.close()
    p4.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_make_analytic_landscape_black_and_white():
    # verify that the backend doesn't raise an error when grayscale coloring
    # schemes are required

    p = make_test_analytic_landscape(MB)
    p.fig
    p.close()


def test_axis_limits():
    # when lines using colormaps or surface (both with colormaps or
    # solid colors), axis limits must be set in order to get the correct
    # visualization. Axis limits can't be NaN or Inf.
    # The following examples shouldn't raise any error.

    x = symbols("x")
    expr = 1 / cos(10 * x) + 5 * sin(x)
    p = plot(
        expr,
        (x, -5, 5),
        ylim=(-10, 10),
        detect_poles=True,
        n=1000,
        eps=1e-04,
        color_func=lambda x, y: x,
        show=False,
    )
    p.draw()
    p.close()


def test_xaxis_inverted():
    # verify that for a plot containing a LineOver1DRangeSeries,
    # if range is given as (symb, max, min) then x-axis is inverted.

    x = symbols("x")
    p = plot(sin(x), (x, 0, 3), backend=MB, show=False, n=10)
    assert not p.ax.xaxis.get_inverted()
    p.close()

    p = plot(sin(x), (x, 3, 0), backend=MB, show=False, n=10)
    assert p.ax.xaxis.get_inverted()
    p.close()


def test_detect_poles():
    # no detection: only one line is visible
    p = make_test_detect_poles(MB, False)
    p.draw()
    assert len(p.ax.lines) == 1
    p.close()

    # detection is done only with numerical data
    # only one line is visible
    p = make_test_detect_poles(MB, True)
    p.draw()
    assert len(p.ax.lines) == 1
    p.close()

    # detection is done only both with numerical data
    # and symbolic analysis. Multiple lines are visible
    p = make_test_detect_poles(MB, "symbolic")
    p.draw()
    assert len(p.ax.lines) > 1
    assert all(l.get_color() == "k" for l in p.ax.lines[1:])
    p.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_detect_poles_interactive():
    # no detection: only one line is visible
    ip = make_test_detect_poles_interactive(MB, False)
    p = ip.backend
    p.draw()
    assert len(p.ax.lines) == 1

    # detection is done only with numerical data
    # only one line is visible
    ip = make_test_detect_poles_interactive(MB, True)
    p = ip.backend
    p.draw()
    assert len(p.ax.lines) == 1

    # no errors are raised
    p.update_interactive({y: 1})
    p.update_interactive({y: -1})

    # detection is done only both with numerical data
    # and symbolic analysis. Multiple lines are visible
    ip = make_test_detect_poles_interactive(MB, "symbolic")
    p = ip.backend
    p.draw()
    assert len(p.ax.lines) == 7
    assert all(l.get_color() == "k" for l in p.ax.lines[1:])

    # one more discontinuity is getting into the visible range
    p.update_interactive({y: 1})
    assert len(p.ax.lines) == 8

    p.update_interactive({y: -1})
    assert len(p.ax.lines) == 8


@pytest.mark.parametrize(
    "annotate, n_imgs, n_lines, n_texts", [
        (True, 1, 3, 4),
        (False, 1, 1, 0)
    ]
)
def test_plot_riemann_sphere(annotate, n_imgs, n_lines, n_texts):
    p = make_test_plot_riemann_sphere(MB, annotate)
    fig = p.fig
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]
    assert len(ax1.images) == len(ax2.images) == n_imgs
    assert len(ax1.lines) == len(ax2.lines) == n_lines
    assert len(ax1.texts) == len(ax2.texts) == n_texts
    p.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_parametric_texts():
    # verify that xlabel, ylabel, zlabel, title accepts parametric texts
    x, y, p = make_test_parametric_texts_2d(MB)
    assert p.backend.ax.get_title() == "y=1.0, z=0.000"
    assert p.backend.ax.get_xlabel() == "test y+z=1.00"
    assert p.backend.ax.get_ylabel() == "test z=0.00"
    p.backend.update_interactive({y: 1.5, z: 2})
    assert p.backend.ax.get_title() == "y=1.5, z=2.000"
    assert p.backend.ax.get_xlabel() == "test y+z=3.50"
    assert p.backend.ax.get_ylabel() == "test z=2.00"
    p.backend.close()

    a, b, p = make_test_parametric_texts_3d(MB)
    assert p.backend.ax.get_title() == "a=1.0, a+b=1.000"
    assert p.backend.ax.get_xlabel() == "test a=1.00"
    assert p.backend.ax.get_ylabel() == "test b=0.00"
    assert p.backend.ax.get_zlabel() == "test a=1.00, b=0.00"
    p.backend.update_interactive({a: 1.5, b: 2})
    assert p.backend.ax.get_title() == "a=1.5, a+b=3.500"
    assert p.backend.ax.get_xlabel() == "test a=1.50"
    assert p.backend.ax.get_ylabel() == "test b=2.00"
    assert p.backend.ax.get_zlabel() == "test a=1.50, b=2.00"
    p.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_arrow_2d():
    a, b = symbols("a, b")
    p = make_test_arrow_2d(MB, "test", {"color": "r"}, True)
    ax = p.backend.ax
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 1
    assert len(ax.get_legend().legend_handles) == 1
    assert ax.get_legend().legend_handles[0].get_label() == "test"
    assert ax.get_legend().legend_handles[0].get_color() == "r"
    p.backend.update_interactive({a: 4, b: 5})
    p.backend.close()

    p = make_test_arrow_2d(MB, "test", {"color": "r"}, False)
    ax = p.backend.ax
    assert len(ax.patches) == 1
    assert ax.get_legend() is None
    p.backend.close()


def test_existing_figure_lines():
    # verify that user can provide an existing figure containing lines
    # and plot over it

    fig, ax = matplotlib.pyplot.subplots()
    xx = np.linspace(-np.pi, np.pi, 10)
    yy = np.cos(xx)
    ax.plot(xx, yy, label="l1")
    assert len(ax.lines) == 1

    t = symbols("t")
    p = plot(sin(t), (t, -pi, pi), "l2", n=10,
        backend=MB, show=False, ax=ax)
    assert p.ax is ax
    assert len(ax.lines) == 2
    assert ax.lines[0].get_label() == "l1"
    assert ax.lines[0].get_color() == '#1f77b4'
    assert ax.lines[1].get_label() == "l2"
    assert ax.lines[1].get_color() == '#ff7f0e'
    p.close()


def test_existing_figure_surfaces():
    # verify that user can provide an existing figure containing surfaces
    # and plot over it

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(projection="3d")
    xx, yy = np.mgrid[-3:3:10j, -3:3:10j]
    ax.plot_surface(xx, yy, xx*yy)
    assert len(ax.collections) == 1

    x, y = symbols("x, y")
    p = plot3d(x*y, (x, -3, 3), (y, -3, 3), n=10, backend=MB,
        ax=ax, use_cm=False, show=False)
    assert p.ax is ax
    assert len(ax.collections) == 2
    # the two surfaces are identical. Here, I'm just interested to see
    # different colors
    assert not np.allclose(
        ax.collections[0].get_facecolors()[0],
        ax.collections[1].get_facecolors()[0]
    )
    p.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_arrow_3d():
    a, b, c = symbols("a, b, c")
    p = make_test_arrow_3d(MB, "test", {"color": "r"}, True)
    ax = p.backend.ax
    assert isinstance(ax, mpl_toolkits.mplot3d.axes3d.Axes3D)
    assert len(ax.patches) == 1
    assert len(ax.get_legend().legend_handles) == 1
    assert ax.get_legend().legend_handles[0].get_label() == "test"
    assert ax.get_legend().legend_handles[0].get_color() == "r"
    # only way to test if it renders what it's supposed to
    assert np.allclose(ax.patches[0]._xyz, [1, 2, 3])
    assert np.allclose(ax.patches[0]._dxdydz, [4, 5, 6])
    p.backend.update_interactive({a: 4, b: 5, c: 6})
    p.backend.close()

    p = make_test_arrow_3d(MB, "test", {"color": "r"}, False)
    ax = p.backend.ax
    assert len(ax.patches) == 1
    assert ax.get_legend() is None
    # only way to test if it renders what it's supposed to
    assert np.allclose(ax.patches[0]._xyz, [1, 2, 3])
    assert np.allclose(ax.patches[0]._dxdydz, [4, 5, 6])
    p.backend.close()


@pytest.mark.skipif(ct is None, reason="control is not installed")
@pytest.mark.parametrize(
    "sgrid, zgrid, n_lines, n_texts, instance", [
        # the grid is drawn by some kind of matplotlib's grid locator
        (True, False, 1, 0, SGridLineSeries),
        # the grid is drawn manually, line by line
        (False, True, 33, 20, ZGridLineSeries),
    ]
)
def test_plot_root_locus_1(sgrid, zgrid, n_lines, n_texts, instance):
    a = symbols("a")
    p = make_test_root_locus_1(MB, sgrid, zgrid)
    assert isinstance(p.backend, MB)
    assert len(p.backend.series) == 2
    # NOTE: the backend is going to reorder data series such that grid
    # series are placed at the end.
    assert isinstance(p.backend[0], RootLocusSeries)
    assert isinstance(p.backend[1], instance)
    ax = p.backend.ax
    assert len(ax.lines) == n_lines
    assert ax.get_legend() is None
    assert len(ax.texts) == n_texts # number of sgrid labels on the plot
    line_colors = {'#1f77b4', '0.75'}
    assert all(l.get_color() in line_colors for l in ax.lines)
    p.backend.update_interactive({a: 2})
    p.backend.close()


@pytest.mark.skipif(ct is None, reason="control is not installed")
@pytest.mark.parametrize(
    "sgrid, zgrid, n_lines, n_texts, instance", [
        (True, False, 35, 21, SGridLineSeries),
        (False, True, 33, 20, ZGridLineSeries),
    ]
)
def test_plot_root_locus_3(sgrid, zgrid, n_lines, n_texts, instance):
    a = symbols("a")
    p = make_test_root_locus_3(MB, sgrid, zgrid)
    assert isinstance(p.backend, MB)
    assert len(p.backend.series) == 2
    # NOTE: the backend is going to reorder data series such that grid
    # series are placed at the end.
    assert isinstance(p.backend[0], RootLocusSeries)
    assert isinstance(p.backend[1], instance)
    ax = p.backend.ax
    assert len(ax.lines) == n_lines
    assert ax.get_legend() is None
    assert len(ax.texts) == n_texts # number of sgrid labels on the plot
    line_colors = {'#1f77b4', '0.75'}
    assert all(l.get_color() in line_colors for l in ax.lines)
    p.backend.update_interactive({a: 2})
    p.backend.close()


@pytest.mark.skipif(ct is None, reason="control is not installed")
def test_plot_root_locus_2():
    p = make_test_root_locus_2(MB)
    assert isinstance(p, MB)
    assert len(p.series) == 3
    assert isinstance(p[0], RootLocusSeries)
    assert isinstance(p[1], RootLocusSeries)
    assert isinstance(p[2], SGridLineSeries)
    ax = p.ax
    assert len(ax.lines) == 2
    assert len(ax.texts) == 0
    assert len(ax.get_legend().texts) == 2
    assert p.ax.get_legend().texts[0].get_text() == "a"
    assert p.ax.get_legend().texts[1].get_text() == "b"
    line_colors = {'#1f77b4', '#ff7f0e'}
    assert all(l.get_color() in line_colors for l in ax.lines)
    p.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
@pytest.mark.parametrize(
    "sgrid, zgrid, T, is_filled", [
        (True, False, None, True),
        (False, True, None, True),
        (True, False, 0.05, True),
        (False, True, 0.05, True),
        (False, False, None, False),
    ]
)
def test_plot_pole_zero(sgrid, zgrid, T, is_filled):
    a = symbols("a")
    p = make_test_plot_pole_zero(MB, sgrid=sgrid, zgrid=zgrid, T=T,
        is_filled=is_filled)
    fig = p.fig
    p.backend.update_interactive({a: 2})
    p.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_poles_zeros_sgrid():
    # verify that SGridLineSeries is rendered with "proper" axis limits

    a = symbols("a")
    p = make_test_poles_zeros_sgrid(MB)
    ax = p.backend.ax
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert (xlim is not None) and (ylim is not None)
    # these are eyeball numbers, it should allows a little bit of tweeking at
    # the code for better positioning the grid...
    assert xlim[0] > -5 and xlim[1] < 2
    assert ylim[0] > -5 and ylim[1] < 5
    p.backend.update_interactive({a: 2})
    p.backend.close()


@pytest.mark.skipif(ct is None, reason="control is not installed")
@pytest.mark.parametrize(
    "sgrid, zgrid", [
        (True, False),
        (False, True),
        (False, False)
    ]
)
def test_plot_root_locus_axis_limits(sgrid, zgrid):
    # verify that root locus is renderered with appropriate axis limits

    p = make_test_root_locus_4(MB, sgrid, zgrid)
    ax = p.backend.ax
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert (xlim is not None) and (ylim is not None)
    # these are eyeball numbers, it should allows a little bit of tweeking at
    # the code for better positioning the grid...
    assert xlim[0] > -5 and xlim[1] < 2
    assert ylim[0] > -5 and ylim[1] < 5
    p.backend.close()


@pytest.mark.parametrize(
    "cl_mags, cl_phases, label_cl_phases, n_lines, n_texts",
    [
        (None, None, False, 27, 17),
        (None, None, True, 27, 26),
        (-30, False, False, 2, 1),
        (False, -200, False, 2, 0),
    ]
)
def test_ngrid(cl_mags, cl_phases, label_cl_phases, n_lines, n_texts):
    p = make_test_ngrid(MB, cl_mags, cl_phases, label_cl_phases)
    ax = p.ax
    assert len(ax.lines) == n_lines
    assert len(ax.texts) == n_texts
    p.close()


@pytest.mark.parametrize(
    "xi, wn, tp, ts, auto, show_control_axis, params, n_lines, n_texts",
    [
        (None, None, None, None, False, True, None, 34, 21),
        (None, None, None, None, False, False, None, 35, 21),
        (None, None, None, None, True, True, None, 0, 0),
        (None, None, None, None, True, False, None, 0, 0),
        (0.5, False, None, None, False, False, None, 2, 1),
        ([0.5, 0.75], False, None, None, False, False, None, 4, 2),
        (False, 2, None, None, False, False, None, 1, 1),
        (False, [2, 3], None, None, False, False, None, 2, 2),
        (False, False, 2, None, False, False, None, 1, 0),
        (False, False, None, 3, False, False, None, 1, 0),
        (False, False, 2, 3, False, False, None, 2, 0),
        (False, False, [2, 3], 3, False, False, None, 3, 0),
        (False, False, [2, 3], [3, 4], False, False, None, 4, 0),
        (x, y, z, x+y, False, False,
            {x:(0.5, 0, 1), y:(2, 0, 4), z: (3, 0, 5)},
            5, 2)
    ]
)
def test_sgrid(xi, wn, tp, ts, auto, show_control_axis, params, n_lines, n_texts):
    kw = {}
    if params:
        if ipy is None:
            return
        kw["params"] = params

    p = make_test_sgrid(MB, xi, wn, tp, ts, auto, show_control_axis, **kw)
    ax = p.backend.ax if params else p.ax
    assert len(ax.lines) == n_lines
    assert len(ax.texts) == n_texts
    if params:
        p.backend.update_interactive({x: 0.75, y: 0.8, z: 0.85})
        p.backend.close()
    else:
        p.close()


@pytest.mark.parametrize(
    "xi, wn, tp, ts, show_control_axis, params, n_lines, n_texts",
    [
        (None, None, None, None, True, None, 32, 20),
        (None, None, None, None, False, None, 30, 20),
        (0.5, False, None, None, False, None, 2, 1),
        ([0.5, 0.75], False, None, None, False, None, 4, 2),
        (False, 2/3, None, None, False, None, 1, 1),
        (False, [2/3, 3/4], None, None, False, None, 2, 2),
        (False, False, 2, None, False, None, 1, 1),
        (False, False, None, 3, False, None, 1, 1),
        (False, False, 2, 3, False, None, 2, 2),
        (False, False, [2, 3], 3, False, None, 3, 3),
        (False, False, [2, 3], [3, 4], False, None, 4, 4),
        (x, y, z, x+y, False,
            {x:(0.5, 0, 1), y:(0.75, 0, 4), z: (0.8, 0, 5)},
            5, 4)
    ]
)
def test_zgrid(xi, wn, tp, ts, show_control_axis, params, n_lines, n_texts):
    kw = {}
    if params:
        if ipy is None:
            return
        kw["params"] = params

    p = make_test_zgrid(MB, xi, wn, tp, ts, show_control_axis, **kw)
    ax = p.backend.ax if params else p.ax
    assert len(ax.lines) == n_lines
    assert len(ax.texts) == n_texts
    if params:
        p.backend.update_interactive({x: 0.75, y: 0.8, z: 0.85})
        p.backend.close()
    else:
        p.close()


# On Github, it fails on the minimum installation version,
# with Matplotlib 3.8.3. On local machine it works fine. Why???
@pytest.mark.xfail
@pytest.mark.parametrize("update_event, num_callbacks", [
    (False, 2),
    (True, 3)
])
def test_matplotlib_update_ranges(update_event, num_callbacks):
    # verify that `update_event` doesn't raise errors

    x, y = symbols("x, y")
    p = plot(cos(x), (x, -pi, pi), n=10, backend=MB,
        show=False, update_event=update_event)
    assert len(p.fig._canvas_callbacks.callbacks["button_release_event"]) == num_callbacks

    if update_event:
        p._update_axis_limits("button_release_event")
    p.close()

    p = plot_contour(cos(x**2+y**2), (x, -pi, pi), (y, -pi, pi),
        n=10, backend=MB, show=False, update_event=update_event)
    assert len(p.fig._canvas_callbacks.callbacks["button_release_event"]) == num_callbacks

    if update_event:
        p._update_axis_limits("button_release_event")
    p.close()


@pytest.mark.parametrize(
    "mag, n_lines, n_labels",
    [
        (None, 12, 11),
        (-3, 2, 1),
        (0, 2, 1),
    ]
)
def test_mcircles(mag, n_lines, n_labels):
    p = make_test_mcircles(MB, mag)
    ax = p.ax
    assert len(ax.lines) == n_lines
    assert len(ax.texts) == n_labels
    p.close()


@pytest.mark.skipif(ct is None, reason="control is not installed")
@pytest.mark.parametrize(
    "m_circles, start_marker, mirror_style, arrows, n_lines, n_patches, n_texts",
    [
        (False, "+", None, None, 5, 0, 0),  # no m-circles, no arrows
        (False, None, None, None, 4, 0, 0), # no m-circles, no arrows, no start marker
        (False, "+", None, 2, 5, 4, 0),     # no m-circles
        (False, "+", False, 2, 3, 2, 0),    # no m-circles, no mirror image
        (True, "+", None, 3, 17, 6, 11),    # m-circles, mirror image, arrows, start marker
    ]
)
def test_plot_nyquist_matplotlib(
    m_circles, start_marker, mirror_style, arrows, n_lines, n_patches, n_texts
):
    # verify that plot_nyquist adds the necessary objects to the plot

    s = symbols("s")
    tf1 = 1 / (s**2 + 0.5*s + 2)

    p = plot_nyquist(tf1, show=False, n=10, m_circles=m_circles, arrows=arrows,
        mirror_style=mirror_style, start_marker=start_marker)
    ax = p.ax
    assert len(ax.lines) == n_lines
    assert len(ax.patches) == n_patches
    assert len(ax.texts) == n_texts
    p.close()


@pytest.mark.skipif(ct is None, reason="control is not installed")
@pytest.mark.parametrize(
    "primary_style, mirror_style",
    [
        ("-", ":"),
        (["-", "-."], ["--", ":"]),
        ({"linestyle": "-"}, {"linestyle": ":"}),
        ([{"linestyle": "-"}, {"linestyle": ":"}], [{"linestyle": "--"}, {"linestyle": "-."}]),
        (2, 2),
    ]
)
def test_plot_nyquist_matplotlib_linestyles(primary_style, mirror_style):
    s = symbols("s")
    tf1 = 1 / (s**2 + 0.5*s + 2)

    p = plot_nyquist(tf1, show=False, n=10,
        primary_style=primary_style, mirror_style=mirror_style)
    if not isinstance(primary_style, int):
        ax = p.ax
    else:
        raises(ValueError, lambda: p.ax)
    p.close()


@pytest.mark.skipif(ct is None, reason="control is not installed")
def test_plot_nyquist_matplotlib_interactive():
    # verify that interactive update doesn't raise errors

    a, s = symbols("a, s")
    tf = 1 / (s + a)
    pl = plot_nyquist(
        tf, xlim=(-2, 1), ylim=(-1, 1),
        aspect="equal", m_circles=True,
        params={a: (1, 0, 2)},
        arrows=4, n=10, show=False
    )
    ax = pl.backend.ax # force first draw
    pl.backend.update_interactive({a: 2}) # update with new value
    pl.backend.close()


def test_plot_nichols():
    s = symbols("s")
    tf = (5 * (s - 1)) / (s**2 * (s**2 + s + 4))

    # with nichols grid lines
    p = plot_nichols(tf, ngrid=True, show=False, n=10)
    ax = p.ax
    assert len(ax.lines) > 2
    assert len(ax.texts) > 0
    p.close()

    # no nichols grid lines
    p = plot_nichols(tf, ngrid=False, show=False, n=10)
    ax = p.ax
    assert len(ax.lines) == 1
    assert len(ax.texts) == 0
    p.close()


@pytest.mark.parametrize(
    "arrows, n_arrows",
    [
        (True, 3),
        (False, 0),
        (None, 0),
        (4, 4),
        ([0.2, 0.5, 0.8], 3)
    ]
)
def test_plot_nichols_arrows(arrows, n_arrows):
    s = symbols("s")
    tf = (5 * (s - 1)) / (s**2 * (s**2 + s + 4))
    p = plot_nichols(tf, ngrid=False, show=False, n=10, arrows=arrows)
    ax = p.ax
    assert len(ax.patches) == n_arrows
    p.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
@pytest.mark.parametrize(
    "scatter, use_cm, n_lines, n_collections", [
        (False, False, 1, 0),
        (False, True, 0, 1),
        (True, False, 1, 0),
        (True, True, 0, 1),
    ]
)
def test_plot_nichols_lines_scatter(scatter, use_cm, n_lines, n_collections):
    # no errors are raised with different types of line
    a, s = symbols("a, s")
    tf = (a * (s - 1)) / (s**2 * (s**2 + s + 4))

    # with nichols grid lines
    p = plot_nichols(tf, ngrid=False, show=False, n=10, backend=MB,
        scatter=scatter, use_cm=use_cm, params={a: (5, 0, 10)})
    ax = p.backend.ax
    assert len(ax.lines) == n_lines
    assert len(ax.collections) == n_collections
    p.backend.update_interactive({a: 6})
    p.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_step_response():
    # this should not raise any errors during updates

    a, b, c, d, e, f, g, s = symbols("a, b, c, d, e, f, g, s")
    tf1 = (8*s**2 + 18*s + 32) / (s**3 + 6*s**2 + 14*s + 24)
    tf2 = (s**2 + a*s + b) / (s**3 + c*s**2 + d*s + e)
    p = plot_step_response(
        (tf1, "A"), (tf2, "B"), lower_limit=f, upper_limit=g,
        control=True,
        params={
            a: (3.7, 0, 5),
            b: (10, 0, 20),
            c: (7, 0, 8),
            d: (6, 0, 25),
            e: (16, 0, 25),
            f: (0, 0, 10, 50, "lower limit"),
            g: (10, 0, 25, 50, "upper limit"),
        },
        backend=MB, n=10, show=False
    )
    fig = p.fig
    p.backend.update_interactive({
        a: 4, b: 11, c:6, d: 8, e: 18, f: 5, g: 20
    })
    p.backend.close()


@pytest.mark.skipif(ipy is None, reason="ipywidgets is not installed")
def test_hvlines():
    a, b = symbols("a, b")
    p = make_test_hvlines(MB)
    ax = p.backend.ax
    assert len(ax.lines) == 2
    assert not np.allclose(
        ax.lines[0].get_data(), ax.lines[1].get_data()
    )
    p.backend.update_interactive({a: 3, b: 4})
    p.backend.close()


def test_plot_vector_2d_legend_1():
    x, y = symbols("x, y")
    p = plot_vector(
        [-sin(y), cos(x)], (x, -pi, pi), (y, -pi, pi),
        backend=MB, scalar=True, n=10, show=False
    )
    assert isinstance(p[0], ContourSeries)
    assert isinstance(p[1], Vector2DSeries)
    assert not p[1].use_cm
    # this is because there is only one vector series with use_cm=False
    assert p.legend is None

@pytest.mark.parametrize(
    "streamlines, use_cm, expected", [
        (False, False, True),
        (False, True, None),
        (True, False, True),
        (True, True, None)
    ]
)
def test_plot_vector_2d_legend_2(streamlines, use_cm, expected):
    x, y = symbols("x, y")
    p = plot_vector(
        [-sin(y), cos(x)], [cos(x), -sin(y)], (x, -pi, pi), (y, -pi, pi),
        backend=MB, n=10, show=False, streamlines=streamlines, use_cm=use_cm
    )
    assert p.legend is expected


def test_tick_formatter_multiples_of_2d():
    # NOTE: this character `−` is different from the keyboard `-`
    expected_x1 = ["−4", "−3", "−2", "−1", "0", "1", "2", "3", "4"]
    expected_x2 = ["$-\\frac{3\\pi}{2}$", "$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"]
    expected_y1 = ["−8", "−6", "−4", "−2", "0", "2", "4", "6", "8"]
    expected_y2 = ["$-3\\pi$", "$-2\\pi$", "$-\\pi$", "$0$", "$\\pi$", "$2\\pi$", "$3\\pi$"]

    tf_x = tick_formatter_multiples_of(quantity=np.pi, label="\\pi", n=2)
    tf_y = tick_formatter_multiples_of(quantity=np.pi, label="\\pi", n=1)

    p = make_test_tick_formatters_2d(MB, None, None)
    x_ticks = p.ax.get_xticklabels()
    y_ticks = p.ax.get_yticklabels()
    assert len(x_ticks) == 9
    assert all(isinstance(t, matplotlib.text.Text) for t in x_ticks)
    assert [t.get_text() for t in x_ticks] == expected_x1
    assert len(y_ticks) == 9
    assert all(isinstance(t, matplotlib.text.Text) for t in y_ticks)
    assert [t.get_text() for t in y_ticks] ==expected_y1

    p = make_test_tick_formatters_2d(MB, tf_x, None)
    x_ticks = p.ax.get_xticklabels()
    y_ticks = p.ax.get_yticklabels()
    assert len(x_ticks) == 7
    assert all(isinstance(t, matplotlib.text.Text) for t in x_ticks)
    assert [t.get_text() for t in x_ticks] == expected_x2
    assert len(y_ticks) == 9
    assert all(isinstance(t, matplotlib.text.Text) for t in y_ticks)
    assert [t.get_text() for t in y_ticks] ==expected_y1

    p = make_test_tick_formatters_2d(MB, None, tf_y)
    x_ticks = p.ax.get_xticklabels()
    y_ticks = p.ax.get_yticklabels()
    assert len(x_ticks) == 9
    assert all(isinstance(t, matplotlib.text.Text) for t in x_ticks)
    assert [t.get_text() for t in x_ticks] == expected_x1
    assert len(y_ticks) == 7
    assert all(isinstance(t, matplotlib.text.Text) for t in y_ticks)
    assert [t.get_text() for t in y_ticks] ==expected_y2

    p = make_test_tick_formatters_2d(MB, tf_x, tf_y)
    x_ticks = p.ax.get_xticklabels()
    y_ticks = p.ax.get_yticklabels()
    assert len(x_ticks) == 7
    assert all(isinstance(t, matplotlib.text.Text) for t in x_ticks)
    assert [t.get_text() for t in x_ticks] == expected_x2
    assert len(y_ticks) == 7
    assert all(isinstance(t, matplotlib.text.Text) for t in y_ticks)
    assert [t.get_text() for t in y_ticks] ==expected_y2


def test_tick_formatter_multiples_of_3d():
    # NOTE: this character `−` is different from the keyboard `-`
    expected_x1 = ["−4", "−3", "−2", "−1", "0", "1", "2", "3", "4"]
    expected_x2 = ["$-\\frac{3\\pi}{2}$", "$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$"]
    expected_y1 = ["−8", "−6", "−4", "−2", "0", "2", "4", "6", "8"]
    expected_y2 = ["$-3\\pi$", "$-2\\pi$", "$-\\pi$", "$0$", "$\\pi$", "$2\\pi$", "$3\\pi$"]

    tf_x = tick_formatter_multiples_of(quantity=np.pi, label="\\pi", n=2)
    tf_y = tick_formatter_multiples_of(quantity=np.pi, label="\\pi", n=1)
    p = make_test_tick_formatters_3d(MB, tf_x, tf_y)
    x_ticks = p.ax.get_xticklabels()
    y_ticks = p.ax.get_yticklabels()
    assert len(x_ticks) == 7
    assert all(isinstance(t, matplotlib.text.Text) for t in x_ticks)
    assert [t.get_text() for t in x_ticks] == expected_x2
    assert len(y_ticks) == 7
    assert all(isinstance(t, matplotlib.text.Text) for t in y_ticks)
    assert [t.get_text() for t in y_ticks] ==expected_y2


@pytest.mark.parametrize("x_ticks_formatter, expected_positions, expected_labels", [
    (
        None,
        np.linspace(0, 2*np.pi, 9)[:-1],
        ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"]
    ),
    (
        multiples_of_pi_over_3(),
        np.linspace(0, 2*np.pi, 7)[:-1],
        [
            '$0$', '$\\frac{\\pi}{3}$', '$\\frac{2\\pi}{3}$', '$\\pi$',
            '$\\frac{4\\pi}{3}$', '$\\frac{5\\pi}{3}$'
        ]
    )
])
def test_tick_formatter_multiples_of_polar_plot(
    x_ticks_formatter, expected_positions, expected_labels
):
    p = make_test_tick_formatter_polar_axis(MB, x_ticks_formatter)

    ticks = p.ax.get_xticklabels()
    positions = [t.get_position()[0] for t in ticks]
    labels = [t.get_text() for t in ticks]
    assert np.allclose(positions, expected_positions)
    assert labels == expected_labels


@pytest.mark.parametrize("case, expected_positions, expected_labels", [
    (
        0,
        [0, 1, 2, 3, 4, 5],
        ['0', '1', '2', '3', '4', '5']
    ),
    (
        1,
        [
            -0.7853981633974483, 0.0, 0.7853981633974483, 1.5707963267948966,
            2.356194490192345, 3.141592653589793, 3.9269908169872414,
            4.71238898038469, 5.497787143782138
        ],
        [
            '$-\\frac{\\pi}{4}$', '$0$', '$\\frac{\\pi}{4}$',
            '$\\frac{\\pi}{2}$', '$\\frac{3\\pi}{4}$', '$\\pi$',
            '$\\frac{5\\pi}{4}$', '$\\frac{3\\pi}{2}$', '$\\frac{7\\pi}{4}$'
        ]
    )
])
def test_hooks(case, expected_positions, expected_labels):
    def colorbar_ticks_formatter(plot_object):
        fig = plot_object.fig
        cax = fig.axes[1]
        formatter = multiples_of_pi_over_4()
        cax.yaxis.set_major_locator(formatter.MB_major_locator())
        cax.yaxis.set_major_formatter(formatter.MB_func_formatter())

    p = make_test_hooks_2d(
        MB,
        [colorbar_ticks_formatter] if case else []
    )
    cax = p.fig.axes[1]
    ticks = cax.yaxis.get_ticklabels()
    positions = [t.get_position()[1] for t in ticks]
    labels = [t.get_text() for t in ticks]
    assert np.allclose(positions, expected_positions)
    assert labels == expected_labels


def test_hline_vline_label():
    p = make_test_hline_vline_label(MB)
    ax = p.ax
    assert len(ax.get_legend().legend_handles) == 2
    assert ax.get_legend().legend_handles[0].get_label() == "line"
    assert ax.get_legend().legend_handles[1].get_label() == "hline"


def test_issue_57():
    from matplotlib.ticker import MultipleLocator

    fig, ax = matplotlib.pyplot.subplots()
    ax.yaxis.set_major_locator(MultipleLocator(1))

    assert not isinstance(ax.xaxis.get_major_locator(), MultipleLocator)
    assert isinstance(ax.yaxis.get_major_locator(), MultipleLocator)

    x = symbols("x")
    p = graphics(
        line(sin(x), (x, 0, 2), n=10),
        line(cos(x), (x, 0, 2)), n=10,
        backend=MB, show=False, ax=ax
    )
    assert not isinstance(p.ax.xaxis.get_major_locator(), MultipleLocator)
    assert isinstance(p.ax.yaxis.get_major_locator(), MultipleLocator)

