import pytest
from pytest import raises, warns
go = pytest.importorskip("plotly").graph_objects
from spb.series import Parametric3DLineSeries
import numpy as np
import os
from sympy import Symbol
from tempfile import TemporaryDirectory
from spb import (
    PB, plot, plot_riemann_sphere, plot_real_imag, plot_complex,
    plot_vector, plot3d_revolution, plot3d_spherical,
    plot3d_parametric_surface, plot_contour, plot3d, plot3d_parametric_line,
    plot_parametric, plot_geometry,
    plot_polar
)
from spb.series import SurfaceOver2DRangeSeries, ParametricSurfaceSeries
from sympy import (
    sin, cos, I, pi, Circle, Polygon, sqrt, Matrix, Line, latex, symbols
)
from sympy.abc import x, y, z, u, t, a, b, c
from .make_tests import (
    custom_colorloop_1,
    make_plot_1,
    make_plot_parametric_1,
    make_plot3d_parametric_line_1,
    make_plot3d_1,
    make_plot3d_2,
    make_plot3d_wireframe_1,
    make_plot3d_wireframe_2,
    make_plot3d_wireframe_3,
    make_plot3d_wireframe_4,
    make_plot3d_wireframe_5,
    make_plot3d_parametric_surface_wireframe_1,
    make_plot3d_parametric_surface_wireframe_2,
    make_plot_contour_1,
    make_plot_contour_is_filled,
    make_plot_vector_2d_quiver,
    make_plot_vector_2d_streamlines_1,
    make_plot_vector_2d_streamlines_2,
    make_plot_vector_3d_quiver,
    make_plot_vector_3d_streamlines_1,
    make_plot_vector_2d_normalize_1,
    make_plot_vector_2d_normalize_2,
    make_plot_vector_3d_normalize_1,
    make_plot_vector_3d_normalize_2,
    make_test_plot_implicit_adaptive_true,
    make_test_plot_implicit_adaptive_false,
    make_test_plot_complex_1d,
    make_test_plot_complex_2d,
    make_test_plot_complex_3d,
    make_test_plot_list_is_filled_false,
    make_test_plot_list_is_filled_true,
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
    make_test_plot_use_latex,
    make_test_plot_parametric_use_latex,
    make_test_plot_contour_use_latex,
    make_test_plot_vector_2d_quivers_use_latex,
    make_test_plot_vector_2d_streamlines_custom_scalar_field_custom_label_use_latex,
    make_test_plot_vector_2d_streamlines_custom_scalar_field_use_latex,
    make_test_plot_vector_2d_use_latex_colorbar,
    make_test_plot_vector_3d_quivers_use_latex,
    make_test_plot_vector_3d_streamlines_use_latex,
    make_test_plot_complex_use_latex_1,
    make_test_plot_complex_use_latex_2,
    make_test_plot_real_imag_use_latex,
    make_test_plot3d_use_cm,
    make_test_plot_polar,
    make_test_plot_polar_use_cm,
    make_test_plot3d_implicit,
    make_test_surface_color_func_1,
    make_test_surface_color_func_2,
    make_test_surface_interactive_color_func,
    make_test_line_interactive_color_func,
    make_test_line_color_plot,
    make_test_line_color_plot3d_parametric_line,
    make_test_surface_color_plot3d,
    make_test_plot3d_list_use_cm_False,
    make_test_plot3d_list_use_cm_color_func,
    make_test_plot3d_list_interactive,
    make_test_contour_show_clabels_1,
    make_test_contour_show_clabels_2,
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
    make_test_plot3d_parametric_line_use_latex,
    make_test_plot3d_use_latex,
    make_test_vectors_3d_update_interactive,
    make_test_plot_list_color_func,
    make_test_real_imag,
    make_test_arrow_2d,
    make_test_hvlines
)


# NOTE
# While BB, PB, KB creates the figure at instantiation, MB creates the figure
# once the `show()` method is called. All backends do not populate the figure
# at instantiation. Numerical data is added only when `show()` or `fig` is
# called.
# In the following tests, we will use `show=False`, hence the `show()` method
# won't be executed. To add numerical data to the plots we either call `fig`
# or `draw()`.


class PBchild(PB):
    colorloop = ["red", "green", "blue"]


def test_colorloop_colormaps():
    # verify that backends exposes important class attributes enabling
    # automatic coloring

    assert hasattr(PB, "colorloop")
    assert isinstance(PB.colorloop, (list, tuple))
    assert hasattr(PB, "colormaps")
    assert isinstance(PB.colormaps, (list, tuple))
    assert hasattr(PB, "quivers_colors")
    assert isinstance(PB.quivers_colors, (list, tuple))


def test_custom_colorloop():
    # verify that it is possible to modify the backend's class attributes
    # in order to change custom coloring

    assert len(PBchild.colorloop) != len(PB.colorloop)
    _p1 = custom_colorloop_1(PB)
    _p2 = custom_colorloop_1(PBchild)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert all([isinstance(t, go.Scatter) for t in f1.data])
    assert all([isinstance(t, go.Scatter) for t in f2.data])
    # there are 6 unique colors in _p1 and 3 unique colors in _p2
    assert len(set([d["line"]["color"] for d in f1.data])) == 6
    assert len(set([d["line"]["color"] for d in f2.data])) == 3


def test_plot():
    # verify that the backends produce the expected results when `plot()`
    # is called and `rendering_kw` overrides the default line settings

    p = make_plot_1(PB, rendering_kw=dict(line_color="red"))
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

    p = make_plot_1(PB, rendering_kw=dict(line_color="red"), use_latex=True)
    f = p.fig
    assert f.data[0]["name"] == "$\\sin{\\left(x \\right)}$"
    assert f.layout["xaxis"]["title"]["text"] == "$x$"
    assert f.layout["yaxis"]["title"]["text"] == "$f\\left(x\\right)$"


def test_plot_parametric():
    # verify that the backends produce the expected results when
    # `plot_parametric()` is called and `rendering_kw` overrides the default
    # line settings

    p = make_plot_parametric_1(PB, rendering_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "x"
    assert f.data[0]["line"]["color"] == "red"
    assert f.data[0]["marker"]["colorbar"]["title"]["text"] == "x"


def test_plot3d_parametric_line():
    # verify that the backends produce the expected results when
    # `plot3d_parametric_line()` is called and `rendering_kw` overrides the
    # default line settings

    p = make_plot3d_parametric_line_1(PB, rendering_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter3d)
    assert f.data[0]["line"]["color"] == "red"
    assert f.data[0]["name"] == "x"
    assert f.data[0]["line"]["colorbar"]["title"]["text"] == "x"


def test_plot3d():
    # verify that the backends produce the expected results when
    # `plot3d()` is called and `rendering_kw` overrides the default surface
    # settings

    p = make_plot3d_1(
        PB, rendering_kw=dict(colorscale=[[0, "cyan"], [1, "cyan"]]))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Surface)
    assert f.data[0]["name"] == "cos(x**2 + y**2)"
    assert not f.data[0]["showscale"]
    assert f.data[0]["colorscale"] == ((0, "cyan"), (1, "cyan"))
    assert not f.layout["showlegend"]
    assert f.data[0]["colorbar"]["title"]["text"] == "cos(x**2 + y**2)"


def test_plot3d_2():
    # verify that the backends uses string labels when `plot3d()` is called
    # with `use_latex=False` and `use_cm=True`

    p = make_plot3d_2(PB)
    assert len(p.series) == 2
    f = p.fig
    assert len(f.data) == 2
    assert p.fig.layout.scene.xaxis.title.text == "x"
    assert p.fig.layout.scene.yaxis.title.text == "y"
    assert p.fig.layout.scene.zaxis.title.text == "f(x, y)"
    assert f.data[0].colorbar.title.text == str(cos(x**2 + y**2))
    assert f.data[1].colorbar.title.text == str(sin(x**2 + y**2))
    assert f.data[0].name == str(cos(x**2 + y**2))
    assert f.data[1].name == str(sin(x**2 + y**2))
    assert f.data[0]["showscale"]
    assert f.layout["showlegend"] is False


def test_plot3d_wireframe():
    # verify that wireframe=True is going to add the expected number of line
    # data series and that appropriate keyword arguments work as expected

    p0 = make_plot3d_wireframe_1(PB, False)
    assert len(p0.series) == 1

    p1 = make_plot3d_wireframe_1(PB)
    assert len(p1.series) == 21
    assert isinstance(p1[0], SurfaceOver2DRangeSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p1.series[1:])
    assert all(
        (not s.adaptive) and (s.n[0] == p1[0].n[1]) for s in p1.series[1:11])
    assert all(
        (not s.adaptive) and (s.n[0] == p1[0].n[0]) for s in p1.series[11:])
    assert all(
        p1.fig.data[1]["line"]["color"] == "#000000" for s in p1.series[1:])
    assert np.allclose(
        [t.x[0] for t in p1.fig.data[1:11]], np.linspace(-2, 2, 10))
    assert np.allclose(
        [t.y[0] for t in p1.fig.data[11:]], np.linspace(-3, 3, 10))

    p3 = make_plot3d_wireframe_2(PB, {"line_color": "#ff0000"})
    assert len(p3.series) == 1 + 20 + 30
    assert all(s.n[0] == 12 for s in p3.series[1:])
    assert all(t["line"]["color"] == "#ff0000" for t in p3.fig.data[1:])

    p4 = make_plot3d_wireframe_3(PB, {"line_color": "red"})
    assert len(p4.series) == 1 + 20 + 40
    assert isinstance(p4[0], SurfaceOver2DRangeSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p4.series[1:])
    assert all(
        (not s.adaptive) and (s.n[0] == p4[0].n[1]) for s in p4.series[1:21])
    assert all(
        (not s.adaptive) and (s.n[0] == p4[0].n[0]) for s in p4.series[21:])
    assert all(t["line"]["color"] == "red" for t in p4.fig.data[1:])
    assert np.allclose(
        [t.x[0] for t in p4.fig.data[1:21]], np.linspace(0, 3.25, 20))
    param = p4.series[1].get_data()[-1]
    assert np.isclose(param.min(), 0) and np.isclose(param.max(), 2 * np.pi)
    param = p4.series[21].get_data()[-1]


def test_plot3d_wireframe_lambda_function():
    # verify that wireframe=True correctly works also when the expression is
    # a lambda function

    p0 = make_plot3d_wireframe_4(PB, False)
    assert len(p0.series) == 1

    p1 = make_plot3d_wireframe_4(PB)
    assert len(p1.series) == 21
    assert isinstance(p1[0], SurfaceOver2DRangeSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p1.series[1:])
    assert all(
        (not s.adaptive) and (s.n[0] == p1[0].n[1]) for s in p1.series[1:11])
    assert all(
        (not s.adaptive) and (s.n[0] == p1[0].n[0]) for s in p1.series[11:])
    assert all(
        p1.fig.data[1]["line"]["color"] == "#000000" for s in p1.series[1:])
    assert np.allclose(
        [t.x[0] for t in p1.fig.data[1:11]], np.linspace(-2, 2, 10))
    assert np.allclose(
        [t.y[0] for t in p1.fig.data[11:]], np.linspace(-3, 3, 10))

    p4 = make_plot3d_wireframe_5(PB, {"line_color": "red"})
    assert len(p4.series) == 1 + 20 + 40
    assert isinstance(p4[0], SurfaceOver2DRangeSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p4.series[1:])
    assert all(
        (not s.adaptive) and (s.n[0] == p4[0].n[1]) for s in p4.series[1:21]
    )
    assert all(
        (not s.adaptive) and (s.n[0] == p4[0].n[0]) for s in p4.series[21:]
    )
    assert all(t["line"]["color"] == "red" for t in p4.fig.data[1:])
    assert np.allclose(
        [t.x[0] for t in p4.fig.data[1:21]], np.linspace(0, 3.25, 20)
    )
    param = p4.series[1].get_data()[-1]
    assert np.isclose(param.min(), 0) and np.isclose(param.max(), 2 * np.pi)
    param = p4.series[21].get_data()[-1]
    assert np.isclose(param.min(), 0) and np.isclose(param.max(), 3.25)


def test_plot3d_parametric_surface_wireframe():
    # verify that wireframe=True is going to add the expected number of line
    # data series and that appropriate keyword arguments work as expected

    p = make_plot3d_parametric_surface_wireframe_1(PB, {"line_color": "red"})
    assert len(p.series) == 1 + 5 + 6
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p.series[1:])
    assert all(
        (not s.adaptive) and (s.n[0] == p[0].n[1]) for s in p.series[1:6])
    assert all(
        (not s.adaptive) and (s.n[0] == p[0].n[0]) for s in p.series[6:])
    assert all(t["line"]["color"] == "red" for t in p.fig.data[1:])
    assert all(
        [
            np.isclose(k[0], -1) and np.isclose(k[-1], 1)
            for k in [t.get_data()[-1] for t in p.series[1:6]]
        ]
    )
    assert all(
        [
            np.isclose(k[0], 0) and np.isclose(k[-1], 2 * np.pi)
            for k in [t.get_data()[-1] for t in p.series[6:]]
        ]
    )


def test_plot3d_parametric_surface_wireframe_lambda_function():
    # verify that wireframe=True correctly works also when the expression is
    # a lambda function

    p0 = make_plot3d_parametric_surface_wireframe_2(PB, False)
    assert len(p0.series) == 1

    p1 = make_plot3d_parametric_surface_wireframe_2(PB, True)
    assert len(p1.series) == 1 + 5 + 6
    assert isinstance(p1[0], ParametricSurfaceSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p1.series[1:])
    assert all(
        (not s.adaptive) and (s.n[0] == p1[0].n[1]) for s in p1.series[1:6])
    assert all(
        (not s.adaptive) and (s.n[0] == p1[0].n[0]) for s in p1.series[6:])
    assert all(
        p1.fig.data[1]["line"]["color"] == "#000000" for s in p1.series[1:])
    assert all(
        [
            np.isclose(k[0], -1) and np.isclose(k[-1], 0)
            for k in [t.get_data()[-1] for t in p1.series[1:6]]
        ]
    )
    assert all(
        [
            np.isclose(k[0], 0) and np.isclose(k[-1], 2 * np.pi)
            for k in [t.get_data()[-1] for t in p1.series[6:]]
        ]
    )


def test_plot_contour():
    # verify that the backends produce the expected results when
    # `plot_contour()` is called and `rendering_kw` overrides the default
    # surface settings

    p = make_plot_contour_1(
        PB, rendering_kw=dict(contours=dict(coloring="lines")))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Contour)
    assert f.data[0]["contours"]["coloring"] == "lines"
    assert f.data[0]["colorbar"]["title"]["text"] == str(cos(x**2 + y**2))


def test_plot_contour_is_filled():
    # verify that is_filled=True produces different results than
    # is_filled=False

    p1 = make_plot_contour_is_filled(PB, True)
    p2 = make_plot_contour_is_filled(PB, False)
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

    p = make_plot_vector_2d_quiver(
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


def test_plot_vector_2d_streamlines_custom_scalar_field():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`stream_kw` overrides the
    # default settings

    p = make_plot_vector_2d_streamlines_1(
        PB,
        stream_kw=dict(line_color="red"),
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


def test_plot_vector_2d_streamlines_custom_scalar_field_custom_label():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`stream_kw` overrides the
    # default settings

    p = make_plot_vector_2d_streamlines_2(
        PB,
        stream_kw=dict(line_color="red"),
        contour_kw=dict(contours=dict(coloring="lines")),
    )
    f = p.fig
    assert f.data[0]["colorbar"]["title"]["text"] == "test"


def test_plot_vector_3d_quivers():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `quiver_kw` overrides the
    # default settings

    p = make_plot_vector_3d_quiver(PB, quiver_kw=dict(sizeref=5))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Cone)
    assert f.data[0]["sizeref"] == 5
    assert f.data[0]["colorbar"]["title"]["text"] == str((z, y, x))

    cs1 = f.data[0]["colorscale"]

    p = make_plot_vector_3d_quiver(PB, quiver_kw=dict(colorscale="reds"))
    f = p.fig
    cs2 = f.data[0]["colorscale"]
    assert len(cs1) != len(cs2)


def test_plot_vector_3d_streamlines():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `stream_kw` overrides the
    # default settings

    p = make_plot_vector_3d_streamlines_1(
        PB, stream_kw=dict(colorscale=[[0, "red"], [1, "red"]])
    )
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Streamtube)
    assert f.data[0]["colorscale"] == ((0, "red"), (1, "red"))
    assert f.data[0]["colorbar"]["title"]["text"] == str((z, y, x))

    # test different combinations for streamlines: it should not raise errors
    p = make_plot_vector_3d_streamlines_1(PB, stream_kw=dict(starts=True))
    p = make_plot_vector_3d_streamlines_1(
        PB,
        stream_kw=dict(
            starts={
                "x": np.linspace(-5, 5, 10),
                "y": np.linspace(-4, 4, 10),
                "z": np.linspace(-3, 3, 10),
            }
        ),
    )

    # other keywords: it should not raise errors
    p = make_plot_vector_3d_streamlines_1(
        PB, stream_kw=dict(), kwargs=dict(use_cm=False)
    )


def test_plot_vector_2d_normalize():
    # verify that backends are capable of normalizing a vector field before
    # plotting it. Since all backend are different from each other, let's test
    # that data in the figures is different in the two cases normalize=True
    # and normalize=False

    p1 = make_plot_vector_2d_normalize_1(PB, False)
    p2 = make_plot_vector_2d_normalize_1(PB, True)
    d1x = np.array(p1.fig.data[0].x).astype(float)
    d1y = np.array(p1.fig.data[0].y).astype(float)
    d2x = np.array(p2.fig.data[0].x).astype(float)
    d2y = np.array(p2.fig.data[0].y).astype(float)
    assert not np.allclose(d1x, d2x, equal_nan=True)
    assert not np.allclose(d1y, d2y, equal_nan=True)

    p1 = make_plot_vector_2d_normalize_2(PB, False)
    p1.backend.update_interactive({u: 1.5})
    p2 = make_plot_vector_2d_normalize_2(PB, True)
    p2.backend.update_interactive({u: 1.5})
    d1x = np.array(p1.fig.data[0].x).astype(float)
    d1y = np.array(p1.fig.data[0].y).astype(float)
    d2x = np.array(p2.fig.data[0].x).astype(float)
    d2y = np.array(p2.fig.data[0].y).astype(float)
    assert not np.allclose(d1x, d2x, equal_nan=True)
    assert not np.allclose(d1y, d2y, equal_nan=True)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_plot_vector_3d_normalize():
    # verify that backends are capable of normalizing a vector field before
    # plotting it. Since all backend are different from each other, let's test
    # that data in the figures is different in the two cases normalize=True
    # and normalize=False

    p1 = make_plot_vector_3d_normalize_1(PB, False)
    p2 = make_plot_vector_3d_normalize_1(PB, True)
    assert not np.allclose(p1.fig.data[0]["u"], p2.fig.data[0]["u"])
    assert not np.allclose(p1.fig.data[0]["v"], p2.fig.data[0]["v"])
    assert not np.allclose(p1.fig.data[0]["w"], p2.fig.data[0]["w"])

    p1 = make_plot_vector_3d_normalize_2(PB, False)
    p1.backend.update_interactive({u: 1.5})
    p2 = make_plot_vector_3d_normalize_2(PB, True)
    p2.backend.update_interactive({u: 1.5})
    assert not np.allclose(p1.fig.data[0]["u"], p2.fig.data[0]["u"])
    assert not np.allclose(p1.fig.data[0]["v"], p2.fig.data[0]["v"])
    assert not np.allclose(p1.fig.data[0]["w"], p2.fig.data[0]["w"])


def test_plot_implicit_adaptive_true():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True`

    # PlotlyBackend doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_test_plot_implicit_adaptive_true(
            PB, rendering_kw=dict()).draw(),
    )


def test_plot_implicit_adaptive_false():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True` and `contour_kw`
    # overrides the default settings

    # PlotlyBackend doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_test_plot_implicit_adaptive_false(
            PB, rendering_kw=dict()).draw(),
    )


def test_plot_real_imag():
    # verify that the backends produce the expected results when
    # `plot_real_imag()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_real_imag(PB, rendering_kw=dict(line_color="red"))
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


def test_plot_complex_1d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_plot_complex_1d(PB, rendering_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Scatter)
    assert f.data[0]["name"] == "Arg(sqrt(x))"
    assert f.data[0]["line"]["color"] == "red"
    assert p.fig.data[0]["marker"]["colorbar"]["title"]["text"] == "Arg(sqrt(x))"


def test_plot_complex_2d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_plot_complex_2d(PB, rendering_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 2
    assert isinstance(f.data[0], go.Image)
    assert f.data[0]["name"] == "sqrt(x)"
    assert isinstance(f.data[1], go.Scatter)
    assert f.data[1]["marker"]["colorbar"]["title"]["text"] == "Argument"


def test_plot_complex_3d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_plot_complex_3d(PB, rendering_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.data) == 1
    assert isinstance(f.data[0], go.Surface)
    assert f.data[0]["name"] == "sqrt(x)"
    assert f.data[0]["showscale"] is True
    assert f.data[0]["colorbar"]["title"]["text"] == "Argument"


def test_plot_list_is_filled_false():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=False`

    p = make_test_plot_list_is_filled_false(PB)
    assert len(p.series) == 1
    f = p.fig
    assert f.data[0]["marker"]["line"]["color"] is not None


def test_plot_list_is_filled_true():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=True`

    p = make_test_plot_list_is_filled_true(PB)
    assert len(p.series) == 1
    f = p.fig
    assert f.data[0]["marker"]["line"]["color"] is None


def test_plot_list_color_func():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `color_func`

    p = make_test_plot_list_color_func(PB)
    f = p.fig
    assert f.data[0]["mode"] == "markers"
    assert np.allclose(f.data[0]["marker"]["color"], [0, 1, 2])


def test_plot_piecewise_single_series():
    # verify that plot_piecewise plotting 1 piecewise composed of N
    # sub-expressions uses only 1 color

    p = make_test_plot_piecewise_single_series(PB)
    assert len(p.series) == 4
    colors = set()
    for l in p.fig.data:
        colors.add(l["line"]["color"])
    assert len(colors) == 1
    assert not p.legend


def test_plot_piecewise_multiple_series():
    # verify that plot_piecewise plotting N piecewise expressions uses
    # only N different colors

    p = make_test_plot_piecewise_multiple_series(PB)
    assert len(p.series) == 9
    colors = set()
    for l in p.fig.data:
        colors.add(l["line"]["color"])
    assert len(colors) == 2


def test_plot_geometry_1():
    # verify that the backends produce the expected results when
    # `plot_geometry()` is called

    p = make_test_plot_geometry_1(PB)
    assert len(p.series) == 3
    f = p.fig
    assert len(f.data) == 3
    assert f.data[0]["name"] == str(Line((1, 2), (5, 4)))
    assert f.data[1]["name"] == str(Circle((0, 0), 4))
    assert f.data[2]["name"] == str(Polygon((2, 2), 3, n=6))


def test_plot_geometry_2():
    # verify that is_filled works correctly

    p = make_test_plot_geometry_2(PB, False)
    assert len([t["fill"] for t in p.fig.data if t["fill"] is not None]) == 0
    p = make_test_plot_geometry_2(PB, True)
    assert len([t["fill"] for t in p.fig.data if t["fill"] is not None]) == 3


def test_plot_geometry_3d():
    # verify that no errors are raised when 3d geometric entities are plotted

    p = make_test_plot_geometry_3d(PB)
    p.draw()


@pytest.mark.xfail
def test_save():
    # NOTE: xfail because locally I need to have kaleido installed.

    x, y, z = symbols("x:z")
    options = dict(backend=PB, show=False, adaptive=False, n=5)

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        # Plotly requires additional libraries to save static pictures.
        # Raise an error because their are not installed.
        p = plot(sin(x), cos(x), **options)
        filename = "test_plotly_save_1.png"
        raises(ValueError, lambda: p.save(os.path.join(tmpdir, filename)))

        p = plot(sin(x), cos(x), **options)
        filename = "test_plotly_save_3.html"
        p.save(os.path.join(tmpdir, filename))

        p = plot(sin(x), cos(x), **options)
        filename = "test_plotly_save_4.html"
        p.save(os.path.join(tmpdir, filename), include_plotlyjs="cdn")


def test_vectors_3d_update_interactive():
    # Some backends do not support streamlines with iplot. Test that the
    # backends raise error.

    p = make_test_vectors_3d_update_interactive(PB)
    raises(
        NotImplementedError,
        lambda: p.backend.update_interactive({a: 2, b: 2, c: 2})
    )


def test_aspect_ratio_2d_issue_11764():
    # verify that the backends apply the provided aspect ratio.
    # NOTE: read the backend docs to understand which options are available.

    p = make_test_aspect_ratio_2d_issue_11764(PB)
    assert p.aspect == "auto"
    assert p.fig.layout.yaxis.scaleanchor is None

    p = make_test_aspect_ratio_2d_issue_11764(PB, "equal")
    assert p.aspect == "equal"
    assert p.fig.layout.yaxis.scaleanchor == "x"


def test_aspect_ratio_3d():
    # verify that the backends apply the provided aspect ratio.

    p = make_test_aspect_ratio_3d(PB)
    assert p.aspect == "auto"
    assert p.fig.layout.scene.aspectmode == "auto"

    p = make_test_aspect_ratio_3d(PB, "cube")
    assert p.aspect == "cube"
    assert p.fig.layout.scene.aspectmode == "cube"


def test_plot_size():
    # verify that the keyword `size` is doing it's job
    # NOTE: K3DBackend doesn't support custom size

    p = make_test_plot_size(PB)
    assert p.fig.layout.width is None
    assert p.fig.layout.height is None

    p = make_test_plot_size(PB, (800, 400))
    assert p.fig.layout.width == 800
    assert p.fig.layout.height == 400


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_plot_scale_lin_log():
    # verify that backends are applying the correct scale to the axes
    # NOTE: none of the 3D libraries currently support log scale.

    p = make_test_plot_scale_lin_log(PB, "linear", "linear")
    assert p.fig.layout["xaxis"]["type"] == "linear"
    assert p.fig.layout["yaxis"]["type"] == "linear"

    p = make_test_plot_scale_lin_log(PB, "log", "linear")
    assert p.fig.layout["xaxis"]["type"] == "log"
    assert p.fig.layout["yaxis"]["type"] == "linear"

    p = make_test_plot_scale_lin_log(PB, "linear", "log")
    assert p.fig.layout["xaxis"]["type"] == "linear"
    assert p.fig.layout["yaxis"]["type"] == "log"


def test_backend_latex_labels():
    # verify that backends are going to set axis latex-labels in the
    # 2D and 3D case

    p1 = make_test_backend_latex_labels_1(PB, True)
    p2 = make_test_backend_latex_labels_1(PB, False)
    assert p1.xlabel == p1.fig.layout.xaxis.title.text == "$x^{2}_{1}$"
    assert p2.xlabel == p2.fig.layout.xaxis.title.text == "x_1^2"
    assert p1.ylabel == p1.fig.layout.yaxis.title.text == "$f\\left(x^{2}_{1}\\right)$"
    assert p2.ylabel == p2.fig.layout.yaxis.title.text == "f(x_1^2)"

    # Plotly currently doesn't support latex on 3D plots, hence it will fall
    # back to string representation.
    p1 = make_test_backend_latex_labels_2(PB, True)
    p2 = make_test_backend_latex_labels_2(PB, False)
    assert p1.xlabel == p1.fig.layout.scene.xaxis.title.text == "$x^{2}_{1}$"
    assert p1.ylabel == p1.fig.layout.scene.yaxis.title.text == "$x_{2}$"
    assert (
        p1.zlabel
        == p1.fig.layout.scene.zaxis.title.text
        == "$f\\left(x^{2}_{1}, x_{2}\\right)$"
    )
    assert p2.xlabel == p2.fig.layout.scene.xaxis.title.text == "x_1^2"
    assert p2.ylabel == p2.fig.layout.scene.yaxis.title.text == "x_2"
    assert p2.zlabel == p2.fig.layout.scene.zaxis.title.text == "f(x_1^2, x_2)"


def test_plot_use_latex():
    # verify that the backends produce the expected results when `plot()`
    # is called and `rendering_kw` overrides the default line settings

    p = make_test_plot_use_latex(PB)
    f = p.fig
    assert f.data[0]["name"] == "$\\sin{\\left(x \\right)}$"
    assert f.data[1]["name"] == "$\\cos{\\left(x \\right)}$"
    assert f.layout["showlegend"] is True


def test_plot_parametric_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_parametric_use_latex(PB)
    f = p.fig
    assert f.data[0]["name"] == "$x$"
    assert f.data[0]["marker"]["colorbar"]["title"]["text"] == "$x$"


def test_plot_contour_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_contour_use_latex(PB)
    f = p.fig
    assert f.data[0]["colorbar"]["title"]["text"] == "$%s$" % latex(
        cos(x**2 + y**2)
    )


def test_plot3d_parametric_line_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot3d_parametric_line_use_latex(PB)
    f = p.fig
    assert f.data[0]["name"] == "$x$"
    assert f.data[0]["line"]["colorbar"]["title"]["text"] == "$x$"


def test_plot3d_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot3d_use_latex(PB)
    f = p.fig
    assert f.data[0].colorbar.title.text == "$%s$" % latex(cos(x**2 + y**2))
    assert f.data[1].colorbar.title.text == "$%s$" % latex(sin(x**2 + y**2))
    assert f.data[0].name == "$%s$" % latex(cos(x**2 + y**2))
    assert f.data[1].name == "$%s$" % latex(sin(x**2 + y**2))
    assert f.data[0]["showscale"]
    assert f.layout["showlegend"] is False


def test_plot_vector_2d_quivers_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_2d_quivers_use_latex(PB)
    f = p.fig
    assert f.data[0]["colorbar"]["title"]["text"] == "Magnitude"


def test_plot_vector_2d_streamlines_custom_scalar_field_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_2d_streamlines_custom_scalar_field_use_latex(PB)
    f = p.fig
    assert f.data[0]["colorbar"]["title"]["text"] == "$x + y$"


def test_plot_vector_2d_streamlines_custom_scalar_field_custom_label_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_2d_streamlines_custom_scalar_field_custom_label_use_latex(
        PB
    )
    f = p.fig
    assert f.data[0]["colorbar"]["title"]["text"] == "test"


def test_plot_vector_2d_use_latex_colorbar():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_2d_use_latex_colorbar(PB, True, False)
    assert p.fig.data[0]["colorbar"]["title"]["text"] == "Magnitude"

    p = make_test_plot_vector_2d_use_latex_colorbar(PB, True, True)
    assert p.fig.data[0]["colorbar"]["title"]["text"] == "Magnitude"

    p = make_test_plot_vector_2d_use_latex_colorbar(PB, False, False)
    assert p.fig.data[0]["name"] == "$\\left( x, \\  y\\right)$"

    p = make_test_plot_vector_2d_use_latex_colorbar(PB, False, True)
    assert p.fig.data[0]["name"] == "$\\left( x, \\  y\\right)$"


def test_plot_vector_3d_quivers_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_3d_quivers_use_latex(PB)
    assert (
        p.fig.data[0]["colorbar"]["title"]["text"]
        == "$\\left( z, \\  y, \\  x\\right)$"
    )


def test_plot_vector_3d_streamlines_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_3d_streamlines_use_latex(PB)
    assert (
        p.fig.data[0]["colorbar"]["title"]["text"]
        == "$\\left( z, \\  y, \\  x\\right)$"
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_plot_complex_use_latex():
    # complex plot function should return the same result (for axis labels)
    # wheter use_latex is True or False

    p = make_test_plot_complex_use_latex_1(PB)
    assert p.fig.layout.xaxis.title.text == "Real"
    assert p.fig.layout.yaxis.title.text == "Abs"
    assert p.fig.data[0].name == "Arg(cos(x) + I*sinh(x))"
    assert (
        p.fig.data[0]["marker"]["colorbar"]["title"]["text"]
        == "Arg(cos(x) + I*sinh(x))"
    )

    p = make_test_plot_complex_use_latex_2(PB)
    assert p.fig.layout.xaxis.title.text == "Re"
    assert p.fig.layout.yaxis.title.text == "Im"
    assert p.fig.data[0].name == "$gamma(z)$"
    assert p.fig.data[1]["marker"]["colorbar"]["title"]["text"] == "Argument"


def test_plot_real_imag_use_latex():
    # real/imag plot function should return the same result (for axis labels)
    # wheter use_latex is True or False

    p = make_test_plot_real_imag_use_latex(PB)
    assert p.fig.layout.xaxis.title.text == "$x$"
    assert p.fig.layout.yaxis.title.text == r"$f\left(x\right)$"
    assert p.fig.data[0]["name"] == "Re(sqrt(x))"
    assert p.fig.data[1]["name"] == "Im(sqrt(x))"


def test_plot3d_use_cm():
    # verify that use_cm produces the expected results on plot3d

    p1 = make_test_plot3d_use_cm(PB, True)
    p2 = make_test_plot3d_use_cm(PB, False)
    assert len(p1.fig.data[0]["colorscale"]) > 2
    assert len(p2.fig.data[0]["colorscale"]) == 2


def test_plot_polar():
    # verify that 2D polar plot can create plots with cartesian axis and
    #  polar axis

    p2 = make_test_plot_polar(PB, False)
    assert not isinstance(p2.fig.data[0], go.Scatterpolar)

    p2 = make_test_plot_polar(PB, True)
    assert isinstance(p2.fig.data[0], go.Scatterpolar)


def test_plot_polar_use_cm():
    # verify the correct behavior of plot_polar when color_func
    # or use_cm are applied

    p = make_test_plot_polar_use_cm(PB, False, False)
    assert not p.fig.data[0].marker.showscale

    p = make_test_plot_polar_use_cm(PB, False, True)
    assert p.fig.data[0].marker.showscale

    p = make_test_plot_polar_use_cm(PB, True, False)
    assert p.fig.data[0].marker.showscale is None

    p = make_test_plot_polar_use_cm(PB, True, True, lambda t: t)
    assert p.fig.data[0].marker.showscale


def test_plot3d_implicit():
    # verify that plot3d_implicit don't raise errors

    p = make_test_plot3d_implicit(PB)
    assert isinstance(p.fig.data[0], go.Isosurface)


def test_surface_color_func():
    # After the addition of `color_func`, `SurfaceOver2DRangeSeries` and
    # `ParametricSurfaceSeries` returns different elements.
    # Verify that backends do not raise errors when plotting surfaces and that
    # the color function is applied.

    p1 = make_test_surface_color_func_1(PB, lambda x, y, z: z)
    p2 = make_test_surface_color_func_1(
        PB, lambda x, y, z: np.sqrt(x**2 + y**2))
    assert not np.allclose(
        p1.fig.data[0]["surfacecolor"], p2.fig.data[0]["surfacecolor"]
    )

    p1 = make_test_surface_color_func_2(PB, lambda x, y, z, u, v: z)
    p2 = make_test_surface_color_func_2(
        PB, lambda x, y, z, u, v: np.sqrt(x**2 + y**2)
    )
    assert not np.allclose(
        p1.fig.data[0]["surfacecolor"], p2.fig.data[0]["surfacecolor"]
    )


def test_surface_interactive_color_func():
    # After the addition of `color_func`, `SurfaceInteractiveSeries` and
    # `ParametricSurfaceInteractiveSeries` returns different elements.
    # Verify that backends do not raise errors when updating surfaces and a
    # color function is applied.

    p = make_test_surface_interactive_color_func(PB)
    p.update_interactive({t: 2})
    assert not np.allclose(
        p.fig.data[0]["surfacecolor"],
        p.fig.data[1]["surfacecolor"]
    )
    assert not np.allclose(
        p.fig.data[2]["surfacecolor"],
        p.fig.data[3]["surfacecolor"]
    )


def test_line_color_func():
    # Verify that backends do not raise errors when plotting lines and that
    # the color function is applied.

    p1 = make_test_line_color_func(PB, None)
    p2 = make_test_line_color_func(PB, lambda x, y: np.cos(x))
    assert p1.fig.data[0].marker.color is None
    assert np.allclose(
        p2.fig.data[0].marker.color,
        np.cos(np.linspace(-3, 3, 5))
    )


def test_line_interactive_color_func():
    # Verify that backends do not raise errors when updating lines and a
    # color function is applied.

    p = make_test_line_interactive_color_func(PB)
    p.update_interactive({t: 2})
    assert p.fig.data[0].marker.color is None
    assert np.allclose(
        p.fig.data[1].marker.color,
        np.cos(np.linspace(-3, 3, 5))
    )


def test_line_color_plot():
    # verify back-compatibility with old sympy.plotting module when using
    # line_color

    p = make_test_line_color_plot(PB, "red")
    assert p.fig.data[0]["line"]["color"] == "red"
    p = make_test_line_color_plot(PB, lambda x: -x)
    assert p.fig.data[0].marker.showscale


def test_line_color_plot3d_parametric_line():
    # verify back-compatibility with old sympy.plotting module when using
    # line_color

    p = make_test_line_color_plot3d_parametric_line(PB, "red", False)
    assert p.fig.data[0].line.color == "red"
    p = make_test_line_color_plot3d_parametric_line(PB, lambda x: -x, True)
    assert p.fig.data[0].line.showscale


def test_surface_color_plot3d():
    # verify back-compatibility with old sympy.plotting module when using
    # surface_color

    p = make_test_surface_color_plot3d(PB, "red", False)
    assert p.fig.data[0].colorscale == ((0, "red"), (1, "red"))
    p = make_test_surface_color_plot3d(PB, lambda x, y, z: -x, True)
    assert len(p.fig.data[0].colorscale) > 2


def test_plotly_3d_many_line_series():
    # verify that plotly is capable of dealing with many 3D lines.

    t = symbols("t")

    # in this example there are 31 lines. 30 of them use solid colors, the
    # last one uses a colormap. No errors should be raised.
    with warns(
        UserWarning,
        match="NumPy is unable to evaluate with complex numbers"
    ):
        p = plot3d_revolution(
            cos(t),
            (t, 0, pi),
            backend=PB,
            n=5,
            show=False,
            wireframe=True,
            wf_n1=15,
            wf_n2=15,
            show_curve=True,
            curve_kw={"use_cm": True},
        )
    p.fig


def test_update_interactive():
    # quick round down of tests to verify that _update_interactive doesn't
    # raise errors

    u, v, x, y, z = symbols("u, v, x:z")

    def do_test(p, params, prop):
        p.backend.draw()
        d1 = p.backend[0].get_data()
        v1 = np.array(p.fig.data[0][prop])
        # quivers contain None, which is not comparable with np.allclose
        v1[v1 == None] = np.nan
        p.backend.update_interactive(params)
        d2 = p.backend[0].get_data()
        v2 = np.array(p.fig.data[0][prop])
        v2[v2 == None] = np.nan
        c1 = not all(np.allclose(s, t) for s, t in zip(d1, d2))
        c2 = not np.allclose(
            v1.astype(float), v2.astype(float), equal_nan=True
        )
        return c1 and c2

    p = plot(
        sin(u * x),
        (x, -pi, pi),
        adaptive=False,
        n=5,
        backend=PB,
        show=False,
        params={u: (1, 0, 2)},
    )
    assert do_test(p, {u: 2}, "y")

    p = plot_polar(
        1 + sin(10 * u * x) / 10,
        (x, 0, 2 * pi),
        adaptive=False,
        n=5,
        backend=PB,
        show=False,
        params={u: (1, 0, 2)},
    )
    assert do_test(p, {u: 1.5}, "y")

    p = plot_parametric(
        cos(u * x),
        sin(u * x),
        (x, 0, 2 * pi),
        adaptive=False,
        n=5,
        backend=PB,
        show=False,
        params={u: (1, 0, 2)},
    )
    assert do_test(p, {u: 2}, "y")

    # points
    p = plot3d_parametric_line(
        cos(u * x),
        sin(x),
        x,
        (x, -pi, pi),
        backend=PB,
        is_point=True,
        show=False,
        adaptive=False,
        n=5,
        params={u: (1, 0, 2)},
    )
    assert do_test(p, {u: 2}, "x")

    # line
    p = plot3d_parametric_line(
        cos(u * x),
        sin(x),
        x,
        (x, -pi, pi),
        backend=PB,
        is_point=False,
        show=False,
        adaptive=False,
        n=5,
        params={u: (1, 0, 2)},
        use_cm=False,
    )
    assert do_test(p, {u: 2}, "x")

    p = plot3d_parametric_line(
        cos(u * x),
        sin(x),
        x,
        (x, -pi, pi),
        backend=PB,
        is_point=False,
        show=False,
        adaptive=False,
        n=5,
        params={u: (1, 0, 2)},
        use_cm=True,
    )
    assert do_test(p, {u: 2}, "x")

    p = plot3d(
        cos(u * x**2 + y**2),
        (x, -2, 2),
        (y, -2, 2),
        backend=PB,
        show=False,
        adaptive=False,
        n=5,
        params={u: (1, 0, 2)},
    )
    assert do_test(p, {u: 2}, "z")

    p = plot_contour(
        cos(u * x**2 + y**2),
        (x, -2, 2),
        (y, -2, 2),
        backend=PB,
        show=False,
        adaptive=False,
        n=5,
        params={u: (1, 0, 2)},
    )
    assert do_test(p, {u: 2}, "z")

    u, v = symbols("u, v")
    fx = (1 + v / 2 * cos(u / 2)) * cos(x * u)
    fy = (1 + v / 2 * cos(u / 2)) * sin(x * u)
    fz = v / 2 * sin(u / 2)
    p = plot3d_parametric_surface(
        fx,
        fy,
        fz,
        (u, 0, 2 * pi),
        (v, -1, 1),
        backend=PB,
        use_cm=True,
        n1=5,
        n2=5,
        show=False,
        params={x: (1, 0, 2)},
    )
    assert do_test(p, {x: 2}, "x")

    p = plot_vector(
        Matrix([-u * y, x]),
        (x, -5, 5),
        (y, -4, 4),
        backend=PB,
        n=4,
        show=False,
        params={u: (1, 0, 2)},
        scalar=False,
    )
    assert do_test(p, {u: 2}, "x")

    p = plot_vector(
        Matrix([u * z, y, x]),
        (x, -5, 5),
        (y, -4, 4),
        (z, -3, 3),
        backend=PB,
        n=4,
        show=False,
        params={u: (1, 0, 2)},
    )
    assert do_test(p, {u: 2}, "u")

    p = plot_complex(
        sqrt(u * x),
        (x, -5 - 5 * I, 5 + 5 * I),
        show=False,
        backend=PB,
        threed=True,
        use_cm=True,
        n=5,
        params={u: (1, 0, 2)},
    )
    assert do_test(p, {u: 2}, "z")

    from sympy.geometry import Line as SymPyLine

    p = plot_geometry(
        SymPyLine((u, 2), (5, 4)),
        Circle((0, 0), u),
        Polygon((2, u), 3, n=6),
        backend=PB,
        show=False,
        is_filled=False,
        use_latex=False,
        params={u: (1, 0, 2)},
    )
    assert do_test(p, {u: 2}, "x")

    p = plot_geometry(
        SymPyLine((u, 2), (5, 4)),
        Circle((0, 0), u),
        Polygon((2, u), 3, n=6),
        backend=PB,
        show=False,
        is_filled=True,
        use_latex=False,
        params={u: (1, 0, 2)},
    )
    assert do_test(p, {u: 2}, "x")


def test_generic_data_series():
    # verify that backends do not raise errors when generic data series
    # are used

    x = symbols("x")
    p = plot(
        x,
        backend=PB,
        show=False,
        adaptive=False,
        n=5,
        markers=[{"x": [0, 1], "y": [0, 1], "mode": "markers"}],
        annotations=[{"x": [0, 1], "y": [0, 1], "text": ["a", "b"]}],
        fill=[{"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "fill": "tozeroy"}],
        rectangles=[{"type": "rect", "x0": 1, "y0": 1, "x1": 2, "y1": 3}],
    )
    p.draw()


def test_scatter_gl():
    # verify that if a line contains enough points, PlotlyBackend will use
    # Scattergl instead of Scatter.

    n = PB.scattergl_threshold
    x = symbols("x")
    p1 = plot(cos(x), adaptive=False, n=n - 100, backend=PB, show=False)
    p2 = plot(cos(x), adaptive=False, n=n + 100, backend=PB, show=False)
    p3 = plot_polar(
        1 + sin(10 * x) / 10,
        (x, 0, 2 * pi),
        adaptive=False,
        n=n - 100,
        backend=PB,
        show=False,
        polar_axis=True,
    )
    p4 = plot_polar(
        1 + sin(10 * x) / 10,
        (x, 0, 2 * pi),
        adaptive=False,
        n=n + 100,
        backend=PB,
        show=False,
        polar_axis=True,
    )
    p5 = plot_parametric(
        cos(x),
        sin(x),
        (x, 0, 2 * pi),
        backend=PB,
        show=False,
        adaptive=False,
        n=n - 100,
    )
    p6 = plot_parametric(
        cos(x),
        sin(x),
        (x, 0, 2 * pi),
        backend=PB,
        show=False,
        adaptive=False,
        n=n + 100,
    )
    f1, f2, f3, f4, f5, f6 = [t.fig for t in [p1, p2, p3, p4, p5, p6]]
    assert all(isinstance(t.data[0], go.Scatter) for t in [f1, f5])
    assert all(isinstance(t.data[0], go.Scattergl) for t in [f2, f6])
    assert isinstance(f3.data[0], go.Scatterpolar)
    assert isinstance(f4.data[0], go.Scatterpolargl)


def test_plot3d_list_use_cm_False():
    # verify that plot3d_list produces the expected results when no color map
    # is required

    # solid color line
    p = make_test_plot3d_list_use_cm_False(PB, False, False)
    assert p.fig.data[0].mode == "lines"
    assert p.fig.data[0].line.color == "#636EFA"

    # solid color markers with empty faces
    p = make_test_plot3d_list_use_cm_False(PB, True, False)
    assert p.fig.data[0].mode == "markers"
    assert p.fig.data[0].marker.color == "#E5ECF6"
    assert p.fig.data[0].marker.line.color == "#636EFA"

    # solid color markers with filled faces
    p = make_test_plot3d_list_use_cm_False(PB, True, True)
    assert p.fig.data[0].marker.color == "#636EFA"


def test_plot3d_list_use_cm_color_func():
    # verify that use_cm=True and color_func do their job

    # line with colormap
    # if color_func is not provided, the same parameter will be used
    # for all points
    p1 = make_test_plot3d_list_use_cm_color_func(PB, False, False, None)
    c1 = p1.fig.data[0].line.color
    p2 = make_test_plot3d_list_use_cm_color_func(
        PB, False, False, lambda x, y, z: x
    )
    c2 = p2.fig.data[0].line.color
    assert not np.allclose(c1, c2)

    # markers with empty faces
    p1 = make_test_plot3d_list_use_cm_color_func(PB, False, False, None)
    c1 = p1.fig.data[0].line.color
    p2 = make_test_plot3d_list_use_cm_color_func(
        PB, False, False, lambda x, y, z: x
    )
    c2 = p2.fig.data[0].line.color
    assert not np.allclose(c1, c2)


def test_plot3d_list_interactive():
    # verify that no errors are raises while updating a plot3d_list

    p = make_test_plot3d_list_interactive(PB)
    p.backend.update_interactive({t: 1})


def test_contour_show_clabels():
    p = make_test_contour_show_clabels_1(PB, False)
    assert not p.fig.data[0].contours.showlabels

    p = make_test_contour_show_clabels_1(PB, True)
    assert p.fig.data[0].contours.showlabels

    p = make_test_contour_show_clabels_2(PB, False)
    p.backend.update_interactive({Symbol("u"): 2})
    assert not p.fig.data[0].contours.showlabels

    p = make_test_contour_show_clabels_2(PB, True)
    p.backend.update_interactive({Symbol("u"): 2})
    assert p.fig.data[0].contours.showlabels


def test_color_func_expr():
    # verify that passing an expression to color_func is supported

    with warns(
        UserWarning,
        match="PlotlyBackend doesn't support custom coloring"
    ):
        p3 = make_test_color_func_expr_2(PB)
        # compute the original figure: no errors should be raised
        p3.fig
        # update the figure with new parameters: no errors should be raised
        p3.backend.update_interactive({u: 0.5})


def test_domain_coloring_2d():
    # verify that at_infinity=True flips the image

    p1 = make_test_domain_coloring_2d(PB, False)
    _, _, abs1a, arg1a, img1a, _ = p1[0].get_data()
    abs1b = p1.fig.data[0].customdata[:, :, 0]
    arg1b = p1.fig.data[0].customdata[:, :, 1]
    img1b = p1.fig.data[0].z
    assert np.allclose(abs1a, abs1b)
    assert np.allclose(arg1a, arg1b)
    assert np.allclose(img1a, img1b)

    p2 = make_test_domain_coloring_2d(PB, True)
    _, _, abs2a, arg2a, img2a, _ = p2[0].get_data()
    abs2b = p2.fig.data[0].customdata[:, :, 0]
    arg2b = p2.fig.data[0].customdata[:, :, 1]
    img2b = p2.fig.data[0].z
    assert np.allclose(abs2b, np.flip(np.flip(abs2a, axis=0), axis=1))
    assert np.allclose(arg2b, np.flip(np.flip(arg2a, axis=0), axis=1))
    assert np.allclose(img2b, np.flip(np.flip(img2a, axis=0), axis=1))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore:The following keyword arguments are unused.")
@pytest.mark.filterwarnings("ignore:NumPy is unable to evaluate with complex numbers")
def test_show_hide_colorbar():
    x, y, z = symbols("x, y, z")
    options = dict(use_cm=True, n=5, adaptive=False, backend=PB, show=False)

    p = lambda c: plot_parametric(
        cos(x), sin(x), (x, 0, 2 * pi), colorbar=c, **options
    )
    assert p(True).fig.data[0].marker.showscale
    assert not p(False).fig.data[0].marker.showscale
    p = lambda c: plot_parametric(
        (cos(x), sin(x)),
        (cos(x) / 2, sin(x) / 2),
        (x, 0, 2 * pi),
        colorbar=c,
        **options
    )
    assert all(t.marker.showscale for t in p(True).fig.data)
    assert not all(t.marker.showscale for t in p(False).fig.data)

    p = lambda c: plot(cos(x), color_func=lambda t: t, colorbar=c, **options)
    assert p(True).fig.data[0].marker.showscale
    assert not p(False).fig.data[0].marker.showscale

    p = lambda c: plot3d_parametric_line(
        cos(x), sin(x), x, (x, 0, 2 * pi), colorbar=c, **options
    )
    assert p(True).fig.data[0].line.showscale
    assert not p(False).fig.data[0].line.showscale
    p = lambda c: plot3d_parametric_line(
        (cos(x), sin(x), x),
        (cos(x) / 2, sin(x) / 2, x),
        (x, 0, 2 * pi),
        colorbar=c,
        **options
    )
    assert all(t.line.showscale for t in p(True).fig.data)
    assert not all(t.line.showscale for t in p(False).fig.data)

    p = lambda c: plot3d(
        cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), colorbar=c, **options
    )
    assert p(True).fig.data[0].showscale
    assert not p(False).fig.data[0].showscale
    p = lambda c: plot3d(
        cos(x**2 + y**2), x * y, (x, -pi, pi), (y, -pi, pi),
        colorbar=c, **options
    )
    assert all(t.showscale for t in p(True).fig.data)
    assert not all(t.showscale for t in p(False).fig.data)

    p = lambda c: plot_contour(
        cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), colorbar=c, **options
    )
    assert p(True).fig.data[0].showscale
    assert not p(False).fig.data[0].showscale

    p = lambda c: plot3d_parametric_surface(
        x * cos(y),
        x * sin(y),
        x * cos(4 * y) / 2,
        (x, 0, pi),
        (y, 0, 2 * pi),
        colorbar=c,
        **options
    )
    assert p(True).fig.data[0].showscale
    assert not p(False).fig.data[0].showscale

    p = lambda c: plot3d_spherical(
        1, (x, 0, 0.7 * pi), (y, 0, 1.8 * pi), colorbar=c, **options
    )
    assert p(True).fig.data[0].showscale
    assert not p(False).fig.data[0].showscale

    p = lambda c: plot3d_revolution(cos(t), (t, 0, pi), colorbar=c, **options)
    assert p(True).fig.data[0].showscale
    assert not p(False).fig.data[0].showscale

    p = lambda c: plot_vector(
        [sin(x - y), cos(x + y)], (x, -3, 3), (y, -3, 3),
        colorbar=c, **options
    )
    assert p(True).fig.data[0].showscale
    assert not p(False).fig.data[0].showscale

    p = lambda c: plot_vector(
        [z, y, x], (x, -10, 10), (y, -10, 10), (z, -10, 10),
        colorbar=c, **options
    )
    assert p(True).fig.data[0].showscale
    assert not p(False).fig.data[0].showscale

    p = lambda c: plot_complex(
        cos(x) + sin(I * x), "f", (x, -2, 2), colorbar=c, **options
    )
    assert p(True).fig.data[0].marker.showscale
    assert not p(False).fig.data[0].marker.showscale
    p = lambda c: plot_complex(
        sin(z), (z, -3 - 3j, 3 + 3j), colorbar=c, **options
    )
    assert len(p(True).fig.data) == 2
    assert len(p(False).fig.data) == 1
    p = lambda c: plot_complex(
        sin(z), (z, -3 - 3j, 3 + 3j), colorbar=c, threed=True, **options
    )
    assert p(True).fig.data[0].showscale
    assert not p(False).fig.data[0].showscale

    p = lambda c: plot_real_imag(
        sqrt(x), (x, -3 - 3j, 3 + 3j), threed=True, colorbar=c, **options
    )
    assert p(True).fig.data[0].showscale
    assert not p(False).fig.data[0].showscale

    expr = (z - 1) / (z**2 + z + 1)
    p = lambda c: plot_riemann_sphere(expr, threed=True, colorbar=c, **options)
    assert p(True).fig.data[1].showscale
    assert not p(False).fig.data[1].showscale


def test_show_in_legend():
    # verify that ability of hiding traces from the legend

    p1, p2 = make_test_show_in_legend_2d(PB)
    p3, p4 = make_test_show_in_legend_3d(PB)

    assert [t.showlegend for t in p1.fig.data] == [True, False, True]
    assert [t.showlegend for t in p2.fig.data] == [True, False, True]
    assert [t.showlegend for t in p3.fig.data] == [True, False, True]
    assert [t.showlegend for t in p4.fig.data] == [True, False, True]


def test_legend_plot_sum():
    # when summing up plots together, the first plot dictates if legend
    # is visible or not

    # first case: legend is specified on the first plot
    # if legend is not specified, the resulting plot will show the legend
    p = make_test_legend_plot_sum_1(PB, None)
    assert p.fig.layout.showlegend
    p = make_test_legend_plot_sum_1(PB, True)
    assert p.fig.layout.showlegend
    # first plot has legend=False: output plot won't show the legend
    p = make_test_legend_plot_sum_1(PB, False)
    assert not p.fig.layout.showlegend

    # second case: legend is specified on the second plot
    # the resulting plot will always show the legend
    p = make_test_legend_plot_sum_2(PB, None)
    assert p.fig.layout.showlegend
    p = make_test_legend_plot_sum_2(PB, True)
    assert p.fig.layout.showlegend
    p = make_test_legend_plot_sum_2(PB, False)
    assert p.fig.layout.showlegend


def test_show_legend():
    # if there is only one data series, don't show the legend

    x = symbols("x")
    p = plot(
        sin(x), (x, -pi, pi),
        backend=PB, adaptive=False, n=5, show=False
    )
    assert not p.fig.layout.showlegend

    p = plot(
        sin(x), cos(x), (x, -pi, pi),
        backend=PB, adaptive=False, n=5, show=False
    )
    assert p.fig.layout.showlegend


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_make_analytic_landscape_black_and_white():
    # verify that the backend doesn't raise an error when grayscale coloring
    # schemes are required

    p = make_test_analytic_landscape(PB)
    with warns(
        UserWarning,
        match="The visualization could be wrong becaue Plotly doesn't support custom coloring",
    ):
        p.fig


def test_xaxis_inverted():
    # verify that for a plot containing a LineOver1DRangeSeries,
    # if range is given as (symb, max, min) then x-axis is inverted.

    x = symbols("x")
    p = plot(sin(x), (x, 0, 3), backend=PB, show=False, n=10)
    assert p.fig.layout.xaxis.autorange is None

    p = plot(sin(x), (x, 3, 0), backend=PB, show=False, n=10)
    assert p.fig.layout.xaxis.autorange == "reversed"


def test_detect_poles():
    # no detection: only one line is visible
    p = make_test_detect_poles(PB, False)
    assert len(p.fig.data) == 1
    assert len(p.fig.layout.shapes) == 0

    # detection is done only with numerical data
    # only one line is visible
    p = make_test_detect_poles(PB, True)
    assert len(p.fig.data) == 1
    assert len(p.fig.layout.shapes) == 0

    # detection is done only both with numerical data
    # and symbolic analysis. Multiple lines are visible
    p = make_test_detect_poles(PB, "symbolic")
    assert len(p.fig.data) == 1
    assert len(p.fig.layout.shapes) > 0
    assert all(t.line.color == "black" for t in p.fig.layout.shapes)


def test_detect_poles_interactive():
    # no detection: only one line is visible
    ip = make_test_detect_poles_interactive(PB, False)
    p = ip.backend
    assert len(p.fig.data) == 1
    assert len(p.fig.layout.shapes) == 0

    # detection is done only with numerical data
    # only one line is visible
    ip = make_test_detect_poles_interactive(PB, True)
    p = ip.backend
    assert len(p.fig.data) == 1
    assert len(p.fig.layout.shapes) == 0

    # no errors are raised
    p.update_interactive({y: 1})
    p.update_interactive({y: -1})

    # detection is done only both with numerical data
    # and symbolic analysis. Multiple lines are visible
    ip = make_test_detect_poles_interactive(PB, "symbolic")
    p = ip.backend
    p.draw()
    assert len(p.fig.data) == 1
    assert len(p.fig.layout.shapes) == 6

    # one more discontinuity is getting into the visible range
    p.update_interactive({y: 1})
    assert len(p.fig.data) == 1
    assert len(p.fig.layout.shapes) == 7

    p.update_interactive({y: -1})
    assert len(p.fig.data) == 1
    assert len(p.fig.layout.shapes) == 7


def test_plot_riemann_sphere():
    p = make_test_plot_riemann_sphere(PB, True)
    p.fig
    f1 = p._new_plots[0].fig
    f2 = p._new_plots[1].fig
    # 1 image + 1 unit disk line + 2 scatters/annotation
    assert len(f1.data) == 4
    # 1 image + 1 unit disk line + 2 scatters/annotation + 1 invisible
    # scatter to place the colorbar
    assert len(f2.data) == 5

    p = make_test_plot_riemann_sphere(PB, False)
    p.fig
    f1 = p._new_plots[0].fig
    f2 = p._new_plots[1].fig
    # 1 image + 1 unit disk line
    assert len(f1.data) == 2
    # 1 image + 1 unit disk line + 1 invisible
    # scatter to place the colorbar
    assert len(f2.data) == 3


def test_parametric_texts():
    # verify that xlabel, ylabel, zlabel, title accepts parametric texts
    wrapper = "<b>%s</b>"
    x, y, p = make_test_parametric_texts_2d(PB)
    assert p.fig.layout.title.text == wrapper % "y=1.0, z=0.000"
    assert p.fig.layout.xaxis.title.text == "test y+z=1.00"
    assert p.fig.layout.yaxis.title.text == "test z=0.00"
    p.backend.update_interactive({y: 1.5, z: 2})
    assert p.fig.layout.title.text == wrapper % "y=1.5, z=2.000"
    assert p.fig.layout.xaxis.title.text == "test y+z=3.50"
    assert p.fig.layout.yaxis.title.text == "test z=2.00"

    a, b, p = make_test_parametric_texts_3d(PB)
    assert p.fig.layout.title.text == wrapper % "a=1.0, a+b=1.000"
    assert p.fig.layout.scene.xaxis.title.text == "test a=1.00"
    assert p.fig.layout.scene.yaxis.title.text == "test b=0.00"
    assert p.fig.layout.scene.zaxis.title.text == "test a=1.00, b=0.00"
    p.backend.update_interactive({a: 1.5, b: 2})
    assert p.fig.layout.title.text == wrapper % "a=1.5, a+b=3.500"
    assert p.fig.layout.scene.xaxis.title.text == "test a=1.50"
    assert p.fig.layout.scene.yaxis.title.text == "test b=2.00"
    assert p.fig.layout.scene.zaxis.title.text == "test a=1.50, b=2.00"


def test_arrow_2d():
    a, b = symbols("a, b")
    p = make_test_arrow_2d(PB, "test", {"arrowcolor": "red"}, True)
    fig = p.fig
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0]["text"] == "test"
    assert fig.layout.annotations[0]["arrowcolor"] == "red"
    p._backend.update_interactive({a: 4, b: 5})

    p = make_test_arrow_2d(PB, "test", {"arrowcolor": "red"}, False)
    fig = p.fig
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0]["text"] == ""
    assert fig.layout.annotations[0]["arrowcolor"] == "red"


def test_existing_figure_lines():
    # verify that user can provide an existing figure containing lines
    # and plot over it

    fig = go.Figure()
    xx = np.linspace(-np.pi, np.pi, 10)
    yy = np.cos(xx)
    fig.add_trace(go.Scatter(x=xx, y=yy, name="l1"))
    assert len(fig.data) == 1

    t = symbols("t")
    p = plot(sin(t), (t, -pi, pi), "l2", n=10,
        backend=PB, show=False, fig=fig)
    assert p.fig is fig
    assert len(fig.data) == 2
    assert fig.data[0].name == "l1"
    assert fig.data[0].line.color is None
    assert fig.data[1].name == "l2"
    assert fig.data[1].line.color == '#EF553B'


def test_existing_figure_surfaces():
    # verify that user can provide an existing figure containing surfaces
    # and plot over it

    fig = go.Figure()
    xx, yy = np.mgrid[-3:3:10j, -3:3:10j]
    fig.add_trace(go.Surface(x=xx, y=yy, z=xx*yy, name="s1"))
    assert len(fig.data) == 1

    x, y = symbols("x, y")
    p = plot3d(x*y, (x, -3, 3), (y, -3, 3), n=10, backend=PB,
        fig=fig, use_cm=False, show=False)
    assert p.fig is fig
    assert len(fig.data) == 2
    assert fig.data[0].colorscale is None
    assert fig.data[1].colorscale[0][1] == "#EF553B"


@pytest.mark.parametrize("update_event, fig_type", [
    (False, go.Figure),
    (True, go.FigureWidget)
])
def test_plotly_update_ranges(update_event, fig_type):
    # verify that `update_event` doesn't raise errors

    x, y = symbols("x, y")
    p = plot(cos(x), (x, -pi, pi), n=10, backend=PB,
        show=False, update_event=update_event)
    assert isinstance(p.fig, fig_type)

    if update_event:
        assert len(p.fig.layout._change_callbacks) == 1
        f = list(p.fig.layout._change_callbacks.values())[0][0]
        f(None, (-5, 4), (-3, 2))
    else:
        assert len(p.fig.layout._change_callbacks) == 0

    p = plot_contour(cos(x**2+y**2), (x, -pi, pi), (y, -pi, pi),
        n=10, backend=PB, show=False, update_event=update_event)
    assert isinstance(p.fig, fig_type)

    if update_event:
        assert len(p.fig.layout._change_callbacks) == 1
        f = list(p.fig.layout._change_callbacks.values())[0][0]
        f(None, (-5, 4), (-3, 2))
    else:
        assert len(p.fig.layout._change_callbacks) == 0


def test_hvlines():
    a, b = symbols("a, b")
    p = make_test_hvlines(PB)
    fig = p._backend.fig
    assert len(fig.layout.shapes) == 2
    l1, l2 = fig.layout.shapes
    assert l1["xref"] == "x domain"
    assert l2["yref"] == "y domain"
    assert not np.allclose(
        [l1.x0, l1.x1, l1.y0, l1.y1],
        [l2.x0, l2.x1, l2.y0, l2.y1]
    )
    p._backend.update_interactive({a: 3, b: 4})
