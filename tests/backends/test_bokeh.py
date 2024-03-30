import pytest
from pytest import raises
bokeh = pytest.importorskip("bokeh")
from bokeh.models import (
    ColumnDataSource, Span, Arrow, LabelSet, Label, Line as BLine, Scatter
)
import os
from tempfile import TemporaryDirectory
import numpy as np
from spb import (
    BB, plot, plot_complex, plot_vector, plot_contour,
    plot_parametric, plot_geometry, plot_nyquist, plot_nichols
)
from spb.series import RootLocusSeries, SGridLineSeries, ZGridLineSeries
from sympy import (
    sin, cos, I, pi, Circle, Polygon, sqrt, Matrix, Line, latex, symbols
)
from sympy.abc import x, y, z, u, t
from sympy.external import import_module
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
    make_plot_contour_1,
    make_plot_vector_2d_quiver,
    make_plot_vector_2d_streamlines_1,
    make_plot_vector_2d_streamlines_2,
    make_plot_vector_3d_quiver,
    make_plot_vector_3d_streamlines_1,
    make_plot_vector_2d_normalize_1,
    make_plot_vector_2d_normalize_2,
    make_plot_vector_2d_quiver_color_func_1,
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
    make_test_aspect_ratio_2d_issue_11764,
    make_test_plot_size,
    make_test_plot_scale_lin_log,
    make_test_backend_latex_labels_1,
    make_test_plot_use_latex,
    make_test_plot_parametric_use_latex,
    make_test_plot_contour_use_latex,
    make_test_plot_vector_2d_quivers_use_latex,
    make_test_plot_vector_2d_streamlines_custom_scalar_field_custom_label_use_latex,
    make_test_plot_vector_2d_streamlines_custom_scalar_field_use_latex,
    make_test_plot_vector_2d_use_latex_colorbar,
    make_test_plot_complex_use_latex_1,
    make_test_plot_complex_use_latex_2,
    make_test_plot_real_imag_use_latex,
    make_test_plot_polar,
    make_test_plot_polar_use_cm,
    make_test_plot3d_implicit,
    make_test_line_interactive_color_func,
    make_test_line_color_plot,
    make_test_color_func_expr_1,
    make_test_legend_plot_sum_1,
    make_test_legend_plot_sum_2,
    make_test_domain_coloring_2d,
    make_test_show_in_legend_2d,
    make_test_detect_poles,
    make_test_detect_poles_interactive,
    make_test_plot_riemann_sphere,
    make_test_parametric_texts_2d,
    make_test_line_color_func,
    make_test_plot_list_color_func,
    make_test_real_imag,
    make_test_arrow_2d,
    make_test_root_locus_1,
    make_test_root_locus_2,
    make_test_poles_zeros_sgrid,
    make_test_zgrid,
    make_test_sgrid,
    make_test_ngrid,
    make_test_mcircles
)
ipy = import_module("ipywidgets")
ct = import_module("control")


# NOTE
# While BB, PB, KB creates the figure at instantiation, MB creates the figure
# once the `show()` method is called. All backends do not populate the figure
# at instantiation. Numerical data is added only when `show()` or `fig` is
# called.
# In the following tests, we will use `show=False`, hence the `show()` method
# won't be executed. To add numerical data to the plots we either call `fig`
# or `draw()`.


class BBchild(BB):
    colorloop = ["red", "green", "blue"]


def test_colorloop_colormaps():
    # verify that backends exposes important class attributes enabling
    # automatic coloring

    assert hasattr(BB, "colorloop")
    assert isinstance(BB.colorloop, (list, tuple))
    assert hasattr(BB, "colormaps")
    assert isinstance(BB.colormaps, (list, tuple))


def test_bokeh_tools():
    # verify tools and tooltips on empty Bokeh figure (populated figure
    # might have different tooltips, tested later on)

    f = plot(backend=BB, show=False).fig
    assert len(f.toolbar.tools) == 5
    assert isinstance(f.toolbar.tools[0], bokeh.models.PanTool)
    assert isinstance(f.toolbar.tools[1], bokeh.models.WheelZoomTool)
    assert isinstance(f.toolbar.tools[2], bokeh.models.BoxZoomTool)
    assert isinstance(f.toolbar.tools[3], bokeh.models.ResetTool)
    assert isinstance(f.toolbar.tools[4], bokeh.models.SaveTool)


def test_custom_colorloop():
    # verify that it is possible to modify the backend's class attributes
    # in order to change custom coloring

    assert len(BBchild.colorloop) != len(BB.colorloop)
    _p1 = custom_colorloop_1(BB)
    _p2 = custom_colorloop_1(BBchild)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert all(
        [isinstance(t.glyph, bokeh.models.glyphs.Line) for t in f1.renderers]
    )
    assert all(
        [isinstance(t.glyph, bokeh.models.glyphs.Line) for t in f2.renderers]
    )
    # there are 6 unique colors in _p1 and 3 unique colors in _p2
    assert len(set([r.glyph.line_color for r in f1.renderers])) == 6
    assert len(set([r.glyph.line_color for r in f2.renderers])) == 3


def test_plot():
    # verify that the backends produce the expected results when `plot()`
    # is called and `rendering_kw` overrides the default line settings

    p = make_plot_1(BB, rendering_kw=dict(line_color="red"))
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

    p = make_plot_1(BB, rendering_kw=dict(line_color="red"), use_latex=True)
    f = p.fig
    assert f.legend[0].items[0].label["value"] == "$\\sin{\\left(x \\right)}$"


def test_plot_parametric():
    # verify that the backends produce the expected results when
    # `plot_parametric()` is called and `rendering_kw` overrides the default
    # line settings

    p = make_plot_parametric_1(BB, rendering_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.MultiLine)
    assert f.renderers[0].glyph.line_color == "red"
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "x"
    assert f.toolbar.tools[-1].tooltips == [("x", "@xs"), ("y", "@ys"), ("u", "@us")]


def test_plot3d_parametric_line():
    # verify that the backends produce the expected results when
    # `plot3d_parametric_line()` is called and `rendering_kw` overrides the
    # default line settings

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: make_plot3d_parametric_line_1(
            BB, rendering_kw=dict(line_color="red")
        ).draw(),
    )


def test_plot3d():
    # verify that the backends produce the expected results when
    # `plot3d()` is called and `rendering_kw` overrides the default surface
    # settings

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: make_plot3d_1(
            BB, rendering_kw=dict(colorscale=[[0, "cyan"], [1, "cyan"]])
        ).draw(),
    )


def test_plot3d_2():
    # verify that the backends uses string labels when `plot3d()` is called
    # with `use_latex=False` and `use_cm=True`

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError, lambda: make_plot3d_2(BB).draw())


def test_plot3d_wireframe():
    # verify that wireframe=True is going to add the expected number of line
    # data series and that appropriate keyword arguments work as expected

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError, lambda: make_plot3d_wireframe_1(BB).draw())
    raises(
        NotImplementedError,
        lambda: make_plot3d_wireframe_2(BB, {}).draw()
    )
    raises(
        NotImplementedError,
        lambda: make_plot3d_wireframe_3(BB, {}).draw()
    )


def test_plot_contour():
    # verify that the backends produce the expected results when
    # `plot_contour()` is called and `rendering_kw` overrides the default
    # surface settings

    # Bokeh doesn't use rendering_kw dictionary. Nothing to customize yet.
    p = make_plot_contour_1(BB, rendering_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Image)
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == str(cos(x**2 + y**2))


def test_plot_vector_2d_quivers():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`quiver_kw` overrides the
    # default settings

    p = make_plot_vector_2d_quiver(
        BB, contour_kw=dict(), quiver_kw=dict(line_color="red")
    )
    assert len(p.series) == 2
    f = p.fig
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Image)
    assert isinstance(f.renderers[1].glyph, bokeh.models.glyphs.Segment)
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "Magnitude"
    assert f.renderers[1].glyph.line_color == "red"


def test_plot_vector_2d_streamlines_custom_scalar_field():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`stream_kw` overrides the
    # default settings

    p = make_plot_vector_2d_streamlines_1(
        BB, stream_kw=dict(line_color="red"), contour_kw=dict()
    )
    assert len(p.series) == 2
    f = p.fig
    assert len(f.renderers) == 2
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Image)
    assert isinstance(f.renderers[1].glyph, bokeh.models.glyphs.MultiLine)
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "x + y"
    assert f.renderers[1].glyph.line_color == "red"


def test_plot_vector_2d_streamlines_custom_scalar_field_custom_label():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`stream_kw` overrides the
    # default settings

    p = make_plot_vector_2d_streamlines_2(
        BB, stream_kw=dict(line_color="red"), contour_kw=dict()
    )
    f = p.fig
    assert f.right[0].title == "test"


def test_plot_vector_3d_quivers():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `quiver_kw` overrides the
    # default settings

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: make_plot_vector_3d_quiver(
            BB, quiver_kw=dict(sizeref=5)).draw(),
    )


def test_plot_vector_3d_streamlines():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `stream_kw` overrides the
    # default settings

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: make_plot_vector_3d_streamlines_1(
            BB, stream_kw=dict(colorscale=[[0, "red"], [1, "red"]])
        ).draw(),
    )


def test_plot_vector_2d_normalize():
    # verify that backends are capable of normalizing a vector field before
    # plotting it. Since all backend are different from each other, let's test
    # that data in the figures is different in the two cases normalize=True
    # and normalize=False

    p1 = make_plot_vector_2d_normalize_1(BB, False)
    p2 = make_plot_vector_2d_normalize_1(BB, True)
    x01 = p1.fig.renderers[0].data_source.data["x0"]
    x11 = p1.fig.renderers[0].data_source.data["x1"]
    y01 = p1.fig.renderers[0].data_source.data["y0"]
    y11 = p1.fig.renderers[0].data_source.data["y1"]
    m1 = p1.fig.renderers[0].data_source.data["color_val"]
    x02 = p2.fig.renderers[0].data_source.data["x0"]
    x12 = p2.fig.renderers[0].data_source.data["x1"]
    y02 = p2.fig.renderers[0].data_source.data["y0"]
    y12 = p2.fig.renderers[0].data_source.data["y1"]
    m2 = p2.fig.renderers[0].data_source.data["color_val"]
    assert not np.allclose(x01, x02)
    assert not np.allclose(x11, x12)
    assert not np.allclose(y01, y02)
    assert not np.allclose(y11, y12)
    assert np.allclose(m1, m2)

    p1 = make_plot_vector_2d_normalize_2(BB, False)
    p1.backend.update_interactive({u: 1.5})
    p2 = make_plot_vector_2d_normalize_2(BB, True)
    p2.backend.update_interactive({u: 1.5})
    x01 = p1.fig.renderers[0].data_source.data["x0"]
    x11 = p1.fig.renderers[0].data_source.data["x1"]
    y01 = p1.fig.renderers[0].data_source.data["y0"]
    y11 = p1.fig.renderers[0].data_source.data["y1"]
    m1 = p1.fig.renderers[0].data_source.data["color_val"]
    x02 = p2.fig.renderers[0].data_source.data["x0"]
    x12 = p2.fig.renderers[0].data_source.data["x1"]
    y02 = p2.fig.renderers[0].data_source.data["y0"]
    y12 = p2.fig.renderers[0].data_source.data["y1"]
    m2 = p2.fig.renderers[0].data_source.data["color_val"]
    assert not np.allclose(x01, x02)
    assert not np.allclose(x11, x12)
    assert not np.allclose(y01, y02)
    assert not np.allclose(y11, y12)
    assert np.allclose(m1, m2)


def test_plot_vector_2d_quiver_color_func():
    # verify that color_func gets applied to 2D quivers

    p1 = make_plot_vector_2d_quiver_color_func_1(BB, None)
    p2 = make_plot_vector_2d_quiver_color_func_1(BB, lambda x, y, u, v: x)
    a1 = p1.fig.renderers[0].data_source.data["color_val"]
    a2 = p2.fig.renderers[0].data_source.data["color_val"]
    assert not np.allclose(a1, a2)


def test_plot_implicit_adaptive_true():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True`

    # BokehBackend doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_test_plot_implicit_adaptive_true(
            BB, rendering_kw=dict()).draw(),
    )


def test_plot_implicit_adaptive_false():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True` and `contour_kw`
    # overrides the default settings

    # BokehBackend doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_test_plot_implicit_adaptive_false(
            BB, rendering_kw=dict()).draw(),
    )


def test_plot_real_imag():
    # verify that the backends produce the expected results when
    # `plot_real_imag()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_real_imag(BB, rendering_kw=dict(line_color="red"))
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


def test_plot_complex_1d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_plot_complex_1d(BB, rendering_kw=dict(line_color="red"))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.MultiLine)
    assert f.renderers[0].glyph.line_color == "red"
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "Arg(sqrt(x))"


def test_plot_complex_2d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_plot_complex_2d(BB, rendering_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.renderers) == 1
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.ImageRGBA)
    assert f.right[0].title == "Argument"
    assert f.toolbar.tools[-1].tooltips == [
        ("x", "$x"),
        ("y", "$y"),
        ("Abs", "@abs"),
        ("Arg", "@arg"),
    ]


def test_plot_complex_3d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: make_test_plot_complex_3d(BB, rendering_kw=dict()).draw(),
    )


def test_plot_list_is_filled_false():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=False`

    p = make_test_plot_list_is_filled_false(BB)
    assert len(p.series) == 1
    f = p.fig
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Scatter)
    assert f.renderers[0].glyph.line_color != f.renderers[0].glyph.fill_color


def test_plot_list_is_filled_true():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=True`

    p = make_test_plot_list_is_filled_true(BB)
    assert len(p.series) == 1
    f = p.fig
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Scatter)
    assert f.renderers[0].glyph.line_color == f.renderers[0].glyph.fill_color


def test_plot_list_color_func():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `color_func`

    p = make_test_plot_list_color_func(BB)
    assert len(p.series) == 1
    f = p.fig
    assert isinstance(f.renderers[0].glyph, bokeh.models.glyphs.Scatter)
    assert np.allclose(f.renderers[0].data_source.data["us"], [0, 1, 2])


def test_plot_piecewise_single_series():
    # verify that plot_piecewise plotting 1 piecewise composed of N
    # sub-expressions uses only 1 color

    p = make_test_plot_piecewise_single_series(BB)
    assert len(p.series) == 4
    colors = set()
    for l in p.fig.renderers:
        colors.add(l.glyph.line_color)
    assert len(colors) == 1
    assert not p.legend


def test_plot_piecewise_multiple_series():
    # verify that plot_piecewise plotting N piecewise expressions uses
    # only N different colors

    p = make_test_plot_piecewise_multiple_series(BB)
    assert len(p.series) == 9
    colors = set()
    for l in p.fig.renderers:
        colors.add(l.glyph.line_color)
    assert len(colors) == 2


def test_plot_geometry_1():
    # verify that the backends produce the expected results when
    # `plot_geometry()` is called

    p = make_test_plot_geometry_1(BB)
    assert len(p.series) == 3
    f = p.fig
    assert len(f.renderers) == 3
    assert all(
        isinstance(r.glyph, bokeh.models.glyphs.Line) for r in f.renderers
    )
    assert f.legend[0].items[0].label["value"] == str(Line((1, 2), (5, 4)))
    assert f.legend[0].items[1].label["value"] == str(Circle((0, 0), 4))
    assert f.legend[0].items[2].label["value"] == str(Polygon((2, 2), 3, n=6))
    assert len(f.legend[0].items) == 3


def test_plot_geometry_2():
    # verify that is_filled works correctly

    p = make_test_plot_geometry_2(BB, False)
    assert (
        len([t.glyph for t in p.fig.renderers
            if isinstance(t.glyph, bokeh.models.glyphs.Line)])
        == 4
    )
    assert (
        len([t.glyph for t in p.fig.renderers
            if isinstance(t.glyph, bokeh.models.glyphs.MultiLine)])
        == 1
    )
    assert (
        len([t.glyph for t in p.fig.renderers
            if isinstance(t.glyph, bokeh.models.glyphs.Scatter)])
        == 1
    )
    assert len(p.fig.legend[0].items) == 5
    p = make_test_plot_geometry_2(BB, True)
    assert (
        len([t.glyph for t in p.fig.renderers
            if isinstance(t.glyph, bokeh.models.glyphs.Line)])
        == 1
    )
    assert (
        len([t.glyph for t in p.fig.renderers
            if isinstance(t.glyph, bokeh.models.glyphs.Patch)])
        == 3
    )
    assert (
        len([t.glyph for t in p.fig.renderers
            if isinstance(t.glyph, bokeh.models.glyphs.MultiLine)])
        == 1
    )
    assert (
        len([t.glyph for t in p.fig.renderers
            if isinstance(t.glyph, bokeh.models.glyphs.Scatter)])
        == 4
    )
    assert len(p.fig.legend[0].items) == 5


def test_save(mocker):
    # Verify the save method. Note that:
    #    Bokeh and Plotly should not be able to save static pictures because
    #    by default they need additional libraries. See the documentation of
    #    the save methods for each backends to see what is required.
    #    Hence, if in our system those libraries are installed, tests will
    #    fail!
    x, y, z = symbols("x:z")
    options = dict(backend=BB, show=False, adaptive=False, n=5)

    with TemporaryDirectory(prefix="sympy_") as tmpdir:
        # Bokeh requires additional libraries to save static pictures.
        # Raise an error because their are not installed.
        mocker.patch("bokeh.io.export_png", side_effect=RuntimeError)
        p = plot(sin(x), cos(x), **options)
        filename = "test_bokeh_save_1.png"
        raises(RuntimeError, lambda: p.save(os.path.join(tmpdir, filename)))

        mocker.patch("bokeh.io.export_svg", side_effect=RuntimeError)
        p = plot(sin(x), cos(x), **options)
        filename = "test_bokeh_save_2.svg"
        raises(RuntimeError, lambda: p.save(os.path.join(tmpdir, filename)))

        p = plot(sin(x), cos(x), **options)
        filename = "test_bokeh_save_3.html"
        p.save(os.path.join(tmpdir, filename))

        p = plot(sin(x), cos(x), **options)
        filename = "test_bokeh_save_4.html"
        p.save(
            os.path.join(tmpdir, filename),
            resources=bokeh.resources.INLINE
        )


def test_aspect_ratio_2d_issue_11764():
    # verify that the backends apply the provided aspect ratio.
    # NOTE: read the backend docs to understand which options are available.

    p = make_test_aspect_ratio_2d_issue_11764(BB)
    assert p.aspect == "auto"
    assert not p.fig.match_aspect

    p = make_test_aspect_ratio_2d_issue_11764(BB, "equal")
    assert p.aspect == "equal"
    assert p.fig.match_aspect


def test_plot_size():
    # verify that the keyword `size` is doing it's job
    # NOTE: K3DBackend doesn't support custom size

    p = make_test_plot_size(BB)
    assert p.fig.sizing_mode == "stretch_width"

    p = make_test_plot_size(BB, (400, 200))
    assert p.fig.sizing_mode == "fixed"
    assert (p.fig.width == 400) and (p.fig.height == 200)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_plot_scale_lin_log():
    # verify that backends are applying the correct scale to the axes
    # NOTE: none of the 3D libraries currently support log scale.

    p = make_test_plot_scale_lin_log(BB, "linear", "linear")
    assert isinstance(p.fig.x_scale, bokeh.models.scales.LinearScale)
    assert isinstance(p.fig.y_scale, bokeh.models.scales.LinearScale)

    p = make_test_plot_scale_lin_log(BB, "log", "linear")
    assert isinstance(p.fig.x_scale, bokeh.models.scales.LogScale)
    assert isinstance(p.fig.y_scale, bokeh.models.scales.LinearScale)

    p = make_test_plot_scale_lin_log(BB, "linear", "log")
    assert isinstance(p.fig.x_scale, bokeh.models.scales.LinearScale)
    assert isinstance(p.fig.y_scale, bokeh.models.scales.LogScale)


def test_backend_latex_labels():
    # verify that backends are going to set axis latex-labels in the
    # 2D and 3D case

    p1 = make_test_backend_latex_labels_1(BB, True)
    p2 = make_test_backend_latex_labels_1(BB, False)
    assert p1.xlabel == p1.fig.xaxis.axis_label == "$x^{2}_{1}$"
    assert p2.xlabel == p2.fig.xaxis.axis_label == "x_1^2"
    assert p1.ylabel == p1.fig.yaxis.axis_label == "$f\\left(x^{2}_{1}\\right)$"
    assert p2.ylabel == p2.fig.yaxis.axis_label == "f(x_1^2)"


def test_plot_use_latex():
    # verify that the backends produce the expected results when `plot()`
    # is called and `rendering_kw` overrides the default line settings

    p = make_test_plot_use_latex(BB)
    f = p.fig
    assert f.legend[0].items[0].label["value"] == "$\\sin{\\left(x \\right)}$"
    assert f.legend[0].items[1].label["value"] == "$\\cos{\\left(x \\right)}$"
    assert f.legend[0].visible is True


def test_plot_parametric_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_parametric_use_latex(BB)
    f = p.fig
    # 1 colorbar
    assert len(f.right) == 1
    assert f.right[0].title == "$x$"


def test_plot_contour_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_contour_use_latex(BB)
    f = p.fig
    assert f.right[0].title == "$%s$" % latex(cos(x**2 + y**2))


def test_plot_vector_2d_quivers_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_2d_quivers_use_latex(BB)
    f = p.fig
    assert f.right[0].title == "Magnitude"


def test_plot_vector_2d_streamlines_custom_scalar_field_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_2d_streamlines_custom_scalar_field_use_latex(BB)
    f = p.fig
    assert f.right[0].title == "$x + y$"


def test_plot_vector_2d_streamlines_custom_scalar_field_custom_label_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_2d_streamlines_custom_scalar_field_custom_label_use_latex(
        BB
    )
    f = p.fig
    assert f.right[0].title == "test"


def test_plot_vector_2d_use_latex_colorbar():
    # verify that the colorbar uses latex label

    p = make_test_plot_vector_2d_use_latex_colorbar(BB, True, False)
    assert p.fig.right[0].title == "Magnitude"

    p = make_test_plot_vector_2d_use_latex_colorbar(BB, True, True)
    assert p.fig.right[0].title == "Magnitude"

    p = make_test_plot_vector_2d_use_latex_colorbar(BB, False, False)
    assert p.fig.right[0].title == "$\\left( x, \\  y\\right)$"

    # Bokeh doesn't support gradient streamlines, hence no colorbar
    p = make_test_plot_vector_2d_use_latex_colorbar(BB, False, True)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_plot_complex_use_latex():
    # complex plot function should return the same result (for axis labels)
    # wheter use_latex is True or False

    p = make_test_plot_complex_use_latex_1(BB)
    assert p.fig.right[0].title == "Arg(cos(x) + I*sinh(x))"
    assert p.fig.xaxis.axis_label == "Real"
    assert p.fig.yaxis.axis_label == "Abs"

    p = make_test_plot_complex_use_latex_2(BB)
    assert p.fig.right[0].title == "Argument"
    assert p.fig.xaxis.axis_label == "Re"
    assert p.fig.yaxis.axis_label == "Im"


def test_plot_real_imag_use_latex():
    # real/imag plot function should return the same result (for axis labels)
    # wheter use_latex is True or False

    p = make_test_plot_real_imag_use_latex(BB)
    assert p.fig.xaxis.axis_label == "$x$"
    assert p.fig.yaxis.axis_label == r"$f\left(x\right)$"
    assert p.fig.legend[0].items[0].label["value"] == "Re(sqrt(x))"
    assert p.fig.legend[0].items[1].label["value"] == "Im(sqrt(x))"


def test_plot_polar():
    # verify that 2D polar plot can create plots with cartesian axis and
    #  polar axis

    p3 = make_test_plot_polar(BB, False)
    plotly_data = p3[0].get_data()
    bokeh_data = p3.fig.renderers[0].data_source.data
    assert np.allclose(plotly_data[0], bokeh_data["xs"])
    assert np.allclose(plotly_data[1], bokeh_data["ys"])

    # Bokeh doesn't have polar projection. Here we check that the backend
    # transforms the data.
    raises(ValueError, lambda: make_test_plot_polar(BB, True))


def test_plot_polar_use_cm():
    # verify the correct behavior of plot_polar when color_func
    # or use_cm are applied

    p = make_test_plot_polar_use_cm(BB, False, False)
    assert len(p.fig.renderers) == 1
    assert isinstance(p.fig.renderers[0].glyph, bokeh.models.glyphs.Line)

    p = make_test_plot_polar_use_cm(BB, False, True)
    assert len(p.fig.renderers) == 1
    assert isinstance(p.fig.renderers[0].glyph, bokeh.models.glyphs.MultiLine)


def test_plot3d_implicit():
    # verify that plot3d_implicit don't raise errors

    raises(NotImplementedError, lambda: make_test_plot3d_implicit(BB).draw())


def test_line_color_func():
    # Verify that backends do not raise errors when plotting lines and that
    # the color function is applied.

    p1 = make_test_line_color_func(BB, None)
    p2 = make_test_line_color_func(BB, lambda x, y: np.cos(x))
    assert isinstance(p1.fig.renderers[0].glyph, bokeh.models.glyphs.Line)
    assert isinstance(p2.fig.renderers[0].glyph, bokeh.models.glyphs.MultiLine)


def test_line_interactive_color_func():
    # Verify that backends do not raise errors when updating lines and a
    # color function is applied.

    p = make_test_line_interactive_color_func(BB)
    p.update_interactive({t: 2})
    assert isinstance(p.fig.renderers[0].glyph, bokeh.models.glyphs.Line)
    assert isinstance(p.fig.renderers[1].glyph, bokeh.models.glyphs.MultiLine)


def test_line_color_plot():
    # verify back-compatibility with old sympy.plotting module when using
    # line_color

    p = make_test_line_color_plot(BB, "red")
    assert p.fig.renderers[0].glyph.line_color == "red"
    p = make_test_line_color_plot(BB, lambda x: -x)
    assert len(p.fig.right) == 1
    assert p.fig.right[0].title == "sin(x)"


def test_update_interactive():
    # quick round down of test to verify that _update_interactive doesn't
    # raise errors

    u, v, x, y, z = symbols("u, v, x:z")

    p = plot(
        sin(u * x),
        (x, -pi, pi),
        adaptive=False,
        n=5,
        backend=BB,
        show=False,
        params={u: (1, 0, 2)},
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    p = plot_parametric(
        cos(u * x),
        sin(u * x),
        (x, 0, 2 * pi),
        adaptive=False,
        n=5,
        backend=BB,
        show=False,
        params={u: (1, 0, 2)},
        use_cm=True,
        is_point=False,
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    p = plot_parametric(
        cos(u * x),
        sin(u * x),
        (x, 0, 2 * pi),
        adaptive=False,
        n=5,
        backend=BB,
        show=False,
        params={u: (1, 0, 2)},
        use_cm=True,
        is_point=True,
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    p = plot_parametric(
        cos(u * x),
        sin(u * x),
        (x, 0, 2 * pi),
        adaptive=False,
        n=5,
        backend=BB,
        show=False,
        params={u: (1, 0, 2)},
        use_cm=False,
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    p = plot_contour(
        cos(u * x**2 + y**2),
        (x, -2, 2),
        (y, -2, 2),
        backend=BB,
        show=False,
        adaptive=False,
        n=5,
        params={u: (1, 0, 2)},
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    p = plot_vector(
        Matrix([-u * y, x]),
        (x, -5, 5),
        (y, -4, 4),
        backend=BB,
        n=4,
        show=False,
        params={u: (1, 0, 2)},
        streamlines=True,
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    p = plot_vector(
        Matrix([-u * y, x]),
        (x, -5, 5),
        (y, -4, 4),
        backend=BB,
        n=4,
        show=False,
        params={u: (1, 0, 2)},
        streamlines=False,
        scalar=True,
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    p = plot_vector(
        Matrix([-u * y, x]),
        (x, -5, 5),
        (y, -4, 4),
        backend=BB,
        n=4,
        show=False,
        params={u: (1, 0, 2)},
        streamlines=False,
        scalar=False,
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    p = plot_complex(
        sqrt(u * x),
        (x, -5 - 5 * I, 5 + 5 * I),
        show=False,
        backend=BB,
        threed=False,
        n=5,
        params={u: (1, 0, 2)},
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    from sympy.geometry import Line as SymPyLine

    p = plot_geometry(
        SymPyLine((u, 2), (5, 4)),
        Polygon((2, u), 3, n=6),
        backend=BB,
        show=False,
        is_filled=True,
        use_latex=False,
        params={u: (1, 0, 2)},
    )
    p.backend.draw()
    p.backend.update_interactive({u: 2})


def test_generic_data_series():
    # verify that backends do not raise errors when generic data series
    # are used

    x = symbols("x")
    source = ColumnDataSource(data=dict(x=[0], y=[0], text=["test"]))

    p = plot(
        x,
        backend=BB,
        show=False,
        adaptive=False,
        n=5,
        markers=[{"x": [0, 1], "y": [0, 1], "marker": "square"}],
        annotations=[{"x": "x", "y": "y", "source": source}],
        fill=[{"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3], "y2": [0, 0, 0, 0]}],
        rectangles=[{"x": 0, "y": -3, "width": 5, "height": 2}],
    )
    p.draw()


def test_color_func_expr():
    # verify that passing an expression to color_func is supported

    p1 = make_test_color_func_expr_1(BB, False)
    p2 = make_test_color_func_expr_1(BB, True)

    # compute the original figure: no errors should be raised
    p1.fig
    p2.fig

    # update the figure with new parameters: no errors should be raised
    p1.backend.update_interactive({u: 0.5})
    # Bokeh don't raise an error because it doesn't apply a colormap
    # to streamlines
    p2.backend.update_interactive({u: 0.5})


def test_domain_coloring_2d():
    # verify that at_infinity=True flips the image

    p1 = make_test_domain_coloring_2d(BB, False)
    _, _, abs1a, arg1a, img1a, _ = p1[0].get_data()
    img1a = p1._get_img(img1a)
    abs1b = p1.fig.renderers[0].data_source.data["abs"][0]
    arg1b = p1.fig.renderers[0].data_source.data["arg"][0]
    img1b = p1.fig.renderers[0].data_source.data["image"][0]
    assert np.allclose(abs1a, abs1b)
    assert np.allclose(arg1a, arg1b)
    assert np.allclose(img1a, img1b)

    p2 = make_test_domain_coloring_2d(BB, True)
    _, _, abs2a, arg2a, img2a, _ = p2[0].get_data()
    img2a = p1._get_img(img2a)
    abs2b = p2.fig.renderers[0].data_source.data["abs"][0]
    arg2b = p2.fig.renderers[0].data_source.data["arg"][0]
    img2b = p2.fig.renderers[0].data_source.data["image"][0]
    assert np.allclose(abs2b, np.flip(np.flip(abs2a, axis=0), axis=1))
    assert np.allclose(arg2b, np.flip(np.flip(arg2a, axis=0), axis=1))
    assert np.allclose(img2b, np.flip(img2a, axis=0))


@pytest.mark.filterwarnings("ignore:The following keyword arguments are unused.")
def test_show_hide_colorbar():
    x, y, z = symbols("x, y, z")
    options = dict(use_cm=True, n=5, adaptive=False, backend=BB, show=False)

    p = lambda c: plot_parametric(
        cos(x), sin(x), (x, 0, 2 * pi), colorbar=c, **options
    )
    assert len(p(True).fig.right) == 1
    assert len(p(False).fig.right) == 0
    p = lambda c: plot_parametric(
        (cos(x), sin(x)),
        (cos(x) / 2, sin(x) / 2),
        (x, 0, 2 * pi),
        colorbar=c,
        **options
    )
    assert len(p(True).fig.right) == 2
    assert len(p(False).fig.right) == 0

    p = lambda c: plot(cos(x), color_func=lambda t: t, colorbar=c, **options)
    assert len(p(True).fig.right) == 1
    assert len(p(False).fig.right) == 0

    p = lambda c: plot_contour(
        cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), colorbar=c, **options
    )
    assert len(p(True).fig.right) == 1
    assert len(p(False).fig.right) == 0

    # in plot_vector, use_cm is not set by default.
    mod_options = options.copy()
    mod_options.pop("use_cm")
    p = lambda c: plot_vector(
        [sin(x - y), cos(x + y)], (x, -3, 3), (y, -3, 3),
        colorbar=c, **mod_options
    )
    assert len(p(True).fig.right) == 1
    assert len(p(False).fig.right) == 0

    p = lambda c: plot_complex(
        cos(x) + sin(I * x), "f", (x, -2, 2), colorbar=c, **options
    )
    assert len(p(True).fig.right) == 1
    assert len(p(False).fig.right) == 0
    p = lambda c: plot_complex(
        sin(z), (z, -3 - 3j, 3 + 3j), colorbar=c, **options)
    assert len(p(True).fig.right) == 1
    assert len(p(False).fig.right) == 0


def test_show_in_legend():
    # verify that ability of hiding traces from the legend

    p1, p2 = make_test_show_in_legend_2d(BB)

    assert len(p1.fig.legend[0].items) == 2
    assert len(p2.fig.legend[0].items) == 2


def test_legend_plot_sum():
    # when summing up plots together, the first plot dictates if legend
    # is visible or not

    # first case: legend is specified on the first plot
    # if legend is not specified, the resulting plot will show the legend
    p = make_test_legend_plot_sum_1(BB, None)
    assert len(p.fig.legend[0].items) == 3
    assert p.fig.legend[0].visible
    p = make_test_legend_plot_sum_1(BB, True)
    assert len(p.fig.legend[0].items) == 3
    assert p.fig.legend[0].visible
    # first plot has legend=False: output plot won't show the legend
    p = make_test_legend_plot_sum_1(BB, False)
    assert not p.fig.legend[0].visible

    # second case: legend is specified on the second plot
    # the resulting plot will always show the legend
    p = make_test_legend_plot_sum_2(BB, None)
    assert len(p.fig.legend[0].items) == 3
    assert p.fig.legend[0].visible
    p = make_test_legend_plot_sum_2(BB, True)
    assert len(p.fig.legend[0].items) == 3
    assert p.fig.legend[0].visible
    p = make_test_legend_plot_sum_2(BB, False)
    assert len(p.fig.legend[0].items) == 3
    assert p.fig.legend[0].visible


def test_xaxis_inverted():
    # verify that for a plot containing a LineOver1DRangeSeries,
    # if range is given as (symb, max, min) then x-axis is inverted.

    x = symbols("x")
    p = plot(sin(x), (x, 0, 3), backend=BB, show=False, n=10)
    assert not p.fig.x_range.flipped

    p = plot(sin(x), (x, 3, 0), backend=BB, show=False, n=10)
    assert p.fig.x_range.flipped


def test_detect_poles():
    # no detection: only one line is visible
    p = make_test_detect_poles(BB, False)
    assert len(p.fig.renderers) == 1
    assert len([t for t in p.fig.center if isinstance(t, Span)]) == 0

    # detection is done only with numerical data
    # only one line is visible
    p = make_test_detect_poles(BB, True)
    assert len(p.fig.renderers) == 1
    assert len([t for t in p.fig.center if isinstance(t, Span)]) == 0

    # detection is done only both with numerical data
    # and symbolic analysis. Multiple lines are visible
    p = make_test_detect_poles(BB, "symbolic")
    assert len([t for t in p.fig.center if isinstance(t, Span)]) == 6
    assert len([t for t in p.fig.center if isinstance(t, Span) and t.visible]) == 6
    assert all(
        [
            t.line_color == "#000000" for t in p.fig.center
            if isinstance(t, Span)
        ]
    )


def test_detect_poles_interactive():
    # no detection: only one line is visible
    ip = make_test_detect_poles_interactive(BB, False)
    p = ip.backend
    assert len(p.fig.renderers) == 1

    # detection is done only with numerical data
    # only one line is visible
    ip = make_test_detect_poles_interactive(BB, True)
    p = ip.backend
    assert len(p.fig.renderers) == 1

    # no errors are raised
    p.update_interactive({y: 1})
    p.update_interactive({y: -1})

    # detection is done only both with numerical data
    # and symbolic analysis. Multiple lines are visible
    ip = make_test_detect_poles_interactive(BB, "symbolic")
    p = ip.backend
    assert len(p.fig.renderers) == 1
    assert len([t for t in p.fig.center if isinstance(t, Span)]) == 6
    assert all([t.visible for t in p.fig.center if isinstance(t, Span)])
    assert all(
        [t.line_color == "#000000" for t in p.fig.center
        if isinstance(t, Span)]
    )

    # one more discontinuity is getting into the visible range
    p.update_interactive({y: 1})
    assert len(p.fig.renderers) == 1
    assert len([t for t in p.fig.center if isinstance(t, Span)]) == 7
    assert all([t.visible for t in p.fig.center if isinstance(t, Span)])

    p.update_interactive({y: -0.8})
    assert len(p.fig.renderers) == 1
    assert len([t for t in p.fig.center if isinstance(t, Span)]) == 7
    assert (
        len(
            [t for t in p.fig.center if isinstance(t, Span) and not t.visible]
            ) == 1
    )


def test_plot_riemann_sphere():
    p = make_test_plot_riemann_sphere(BB, True)
    p.fig
    f1 = p._new_plots[0].fig
    f2 = p._new_plots[1].fig
    # 1 image + 2 scatters + 1 line for black unit circle
    assert len(f1.renderers) == 4
    assert len(f2.renderers) == 4

    p = make_test_plot_riemann_sphere(BB, False)
    p.fig
    f1 = p._new_plots[0].fig
    f2 = p._new_plots[1].fig
    # 1 image + 1 line for black unit circle
    assert len(f1.renderers) == 2
    assert len(f2.renderers) == 2


def test_parametric_texts():
    # verify that xlabel, ylabel, zlabel, title accepts parametric texts
    x, y, p = make_test_parametric_texts_2d(BB)
    assert p.fig.title.text == "y=1.0, z=0.000"
    assert p.fig.xaxis.axis_label == "test y+z=1.00"
    assert p.fig.yaxis.axis_label == "test z=0.00"
    p.backend.update_interactive({y: 1.5, z: 2})
    assert p.fig.title.text == "y=1.5, z=2.000"
    assert p.fig.xaxis.axis_label == "test y+z=3.50"
    assert p.fig.yaxis.axis_label == "test z=2.00"


def test_arrow_2d():
    p = make_test_arrow_2d(BB, "test", {"line_color": "red"}, True)
    p.fig
    assert len(p.fig.center) == 3
    arrows = [t for t in p.fig.center if isinstance(t, Arrow)]
    assert len(arrows) == 1
    assert arrows[0].line_color == "red"


def test_existing_figure_lines():
    # verify that user can provide an existing figure containing lines
    # and plot over it

    from bokeh.plotting import figure
    fig = figure()
    xx = np.linspace(-np.pi, np.pi, 10)
    yy = np.cos(xx)
    fig.line(xx, yy, legend_label="l1")
    assert len(fig.renderers) == 1

    t = symbols("t")
    p = plot(sin(t), (t, -pi, pi), "l2", n=10,
        backend=BB, show=False, fig=fig, legend=True)
    assert p.fig is fig
    assert len(fig.renderers) == 2
    assert fig.right[0].items[0].label.value == "l1"
    assert fig.renderers[0].glyph.line_color == '#1f77b4'
    assert fig.right[0].items[1].label.value == "l2"
    assert fig.renderers[1].glyph.line_color == '#ff7f0e'


@pytest.mark.parametrize("update_event", [
    False,
    True
])
def test_bokeh_update_ranges(update_event):
    # verify that `update_event` doesn't raise errors

    class Event:
        def __init__(self, x0, x1, y0, y1):
            self.x0 = x0
            self.x1 = x1
            self.y0 = y0
            self.y1 = y1

    x, y = symbols("x, y")
    p = plot(cos(x), (x, -pi, pi), n=10, backend=BB,
        show=False, update_event=update_event)

    if update_event:
        p._ranges_update(Event(-1, 1, -2, 2))

    p = plot_contour(cos(x**2+y**2), (x, -pi, pi), (y, -pi, pi),
        n=10, backend=BB, show=False, update_event=update_event)

    if update_event:
        p._ranges_update(Event(-1, 1, -2, 2))


@pytest.mark.parametrize(
    "xi, wn, tp, ts, show_control_axis, params, n_lines, n_hvlines, n_lblsets, n_texts",
    [
        (None, None, None, None, True, None, 30, 2, 4, [10, 10, 0, 0]),
        (None, None, None, None, False, None, 30, 0, 4, [10, 10, 0, 0]),
        (0.5, False, None, None, False, None, 2, 0, 4, [1, 0, 0, 0]),
        ([0.5, 0.75], False, None, None, False, None, 4, 0, 4, [2, 0, 0, 0]),
        (False, 2/3, None, None, False, None, 1, 0, 4, [0, 1, 0, 0]),
        (False, [2/3, 3/4], None, None, False, None, 2, 0, 4, [0, 2, 0, 0]),
        (False, False, 2, None, False, None, 1, 0, 4, [0, 0, 1, 0]),
        (False, False, None, 3, False, None, 1, 0, 4, [0, 0, 0, 1]),
        (False, False, 2, 3, False, None, 2, 0, 4, [0, 0, 1, 1]),
        (False, False, [2, 3], 3, False, None, 3, 0, 4, [0, 0, 2, 1]),
        (False, False, [2, 3], [3, 4], False, None, 4, 0, 4, [0, 0, 2, 2]),
        (x, y, z, x+y, False,
            {x:(0.5, 0, 1), y:(0.75, 0, 4), z: (0.8, 0, 5)},
            5, 0, 4, [1, 1, 1, 1])
    ]
)
def test_zgrid(
    xi, wn, tp, ts, show_control_axis, params, n_lines, n_hvlines, n_lblsets, n_texts
):
    kw = {}
    if params:
        if ipy is None:
            return
        kw["params"] = params

    p = make_test_zgrid(BB, xi, wn, tp, ts, show_control_axis, **kw)
    fig = p._backend.fig if params else p.fig
    assert len(fig.renderers) == n_lines
    assert len([t for t in fig.center if isinstance(t, Span)]) == n_hvlines
    assert len([t for t in fig.center if isinstance(t, LabelSet)]) == n_lblsets
    assert np.allclose(
        [len(t.source.data["x"]) for t in fig.center if isinstance(t, LabelSet)],
        n_texts
    )
    if params:
        p._backend.update_interactive({x: 0.75, y: 0.8, z: 0.85})


@pytest.mark.parametrize(
    "xi, wn, tp, ts, auto, show_control_axis, params, n_lines, n_hvlines, n_lblsets, n_texts",
    [
        (None, None, None, None, False, True, None, 32, 2, 2, [11, 10]),
        (None, None, None, None, False, False, None, 35, 0, 2, [11, 10]),
        (None, None, None, None, True, True, None, 15, 2, 2, [5, 5]),
        (None, None, None, None, True, False, None, 18, 0, 2, [5, 5]),
        (0.5, False, None, None, False, False, None, 2, 0, 2, [1, 0]),
        ([0.5, 0.75], False, None, None, False, False, None, 4, 0, 2, [2, 0]),
        (False, 2, None, None, False, False, None, 1, 0, 2, [0, 1]),
        (False, [2, 3], None, None, False, False, None, 2, 0, 2, [0, 2]),
        (False, False, 2, None, False, False, None, 0, 1, 2, [0, 0]),
        (False, False, None, 3, False, False, None, 0, 1, 2, [0, 0]),
        (False, False, 2, 3, False, False, None, 0, 2, 2, [0, 0]),
        (False, False, [2, 3], 3, False, False, None, 0, 3, 2, [0, 0]),
        (False, False, [2, 3], [3, 4], False, False, None, 0, 4, 2, [0, 0]),
        (x, y, z, x+y, False, False,
            {x:(0.5, 0, 1), y:(2, 0, 4), z: (3, 0, 5)},
            3, 2, 2, [1, 1])
    ]
)
def test_sgrid(
    xi, wn, tp, ts, auto, show_control_axis, params, n_lines, n_hvlines,
    n_lblsets, n_texts
):
    kw = {}
    if params:
        if ipy is None:
            return
        kw["params"] = params

    p = make_test_sgrid(BB, xi, wn, tp, ts, auto, show_control_axis, **kw)
    fig = p._backend.fig if params else p.fig
    assert len(fig.renderers) == n_lines
    assert len([t for t in fig.center if isinstance(t, Span)]) == n_hvlines
    assert len([t for t in fig.center if isinstance(t, LabelSet)]) == n_lblsets
    assert np.allclose(
        [len(t.source.data["x"]) for t in fig.center if isinstance(t, LabelSet)],
        n_texts
    )
    if params:
        p._backend.update_interactive({x: 0.75, y: 0.8, z: 0.85})


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
    p = make_test_ngrid(BB, cl_mags, cl_phases, label_cl_phases)
    fig = p.fig
    assert len(fig.renderers) == n_lines
    assert len([t for t in fig.center if isinstance(t, Label)]) == n_texts


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_poles_zeros_sgrid():
    # verify that SGridLineSeries is rendered with "proper" axis limits

    p = make_test_poles_zeros_sgrid(BB)
    xlim = p.fig.x_range.start, p.fig.x_range.end
    ylim = p.fig.y_range.start, p.fig.y_range.end
    assert (xlim is not None) and (ylim is not None)
    # these are eyeball numbers, it should allows a little bit of tweeking at
    # the code for better positioning the grid...
    assert xlim[0] > -5 and xlim[1] < 2
    assert ylim[0] > -5 and ylim[1] < 5
    p.update_interactive({})


@pytest.mark.skipif(ct is None, reason="control is not installed")
def test_plot_root_locus_1():
    p = make_test_root_locus_1(BB, True, False)
    assert isinstance(p, BB)
    assert len(p.series) == 2
    # NOTE: the backend is going to reorder data series such that grid
    # series are placed at the end.
    assert isinstance(p[0], RootLocusSeries)
    assert isinstance(p[1], SGridLineSeries)
    fig = p.fig
    assert len(fig.renderers) == 18
    assert len([t for t in fig.renderers if isinstance(t.glyph, BLine)]) == 16
    assert len([t for t in fig.renderers if isinstance(t.glyph, Scatter)]) == 2
    assert len([t for t in fig.center if isinstance(t, LabelSet)]) == 2
    assert len([t for t in fig.center if isinstance(t, Span)]) == 2
    line_colors = {'#1f77b4', '#aaa'}
    assert all(t.glyph.line_color in line_colors for t in fig.renderers)
    p.update_interactive({})

    p = make_test_root_locus_1(BB, False, True)
    assert isinstance(p, BB)
    assert len(p.series) == 2
    assert isinstance(p[0], RootLocusSeries)
    assert isinstance(p[1], ZGridLineSeries)
    fig = p.fig
    assert len(fig.renderers) == 33
    assert len([t for t in fig.renderers if isinstance(t.glyph, BLine)]) == 31
    assert len([t for t in fig.renderers if isinstance(t.glyph, Scatter)]) == 2
    assert len([t for t in fig.center if isinstance(t, LabelSet)]) == 4
    assert len([t for t in fig.center if isinstance(t, Span)]) == 2
    line_colors = {'#1f77b4', '#aaa'}
    assert all(t.glyph.line_color in line_colors for t in fig.renderers)
    p.update_interactive({})


@pytest.mark.skipif(ct is None, reason="control is not installed")
def test_plot_root_locus_2():
    p = make_test_root_locus_2(BB)
    assert isinstance(p, BB)
    assert len(p.series) == 3
    assert isinstance(p[0], RootLocusSeries)
    assert isinstance(p[1], RootLocusSeries)
    assert isinstance(p[2], SGridLineSeries)
    fig = p.fig
    assert len(fig.renderers) == 21
    assert len([t for t in fig.renderers if isinstance(t.glyph, BLine)]) == 17
    assert len([t for t in fig.renderers if isinstance(t.glyph, Scatter)]) == 4
    assert len([t for t in fig.center if isinstance(t, LabelSet)]) == 2
    assert len([t for t in fig.center if isinstance(t, Span)]) == 2
    line_colors = {'#1f77b4', '#ff7f0e', '#aaa'}
    assert all(t.glyph.line_color in line_colors for t in fig.renderers)
    assert fig.legend[0].items[0].label["value"] == "a"
    assert fig.legend[0].items[1].label["value"] == "b"
    p.update_interactive({})


@pytest.mark.parametrize(
    "mag, n_lines, n_vlines, n_labels",
    [
        (None, 11, 1, 11),
        (-3, 2, 0, 1),
        (0, 1, 1, 1),
    ]
)
def test_mcircles(mag, n_lines, n_vlines, n_labels):
    p = make_test_mcircles(BB, mag)
    fig = p.fig
    assert len(fig.renderers) == n_lines
    assert len([t for t in fig.center if isinstance(t, Span)]) == n_vlines
    assert len([t for t in fig.center if isinstance(t, Label)]) == n_labels


@pytest.mark.skipif(ct is None, reason="control is not installed")
@pytest.mark.parametrize(
    "m_circles, start_marker, mirror_style, arrows, n_lines, n_vlines, "
    "n_arrows_sets, n_arrows, n_texts",
    [
        (False, "+", None, None, 3, 0, 2, 0, 0),  # no m-circles, no arrows
        (False, None, None, None, 2, 0, 2, 0, 0), # no m-circles, no arrows, no start marker
        (False, "+", None, 2, 3, 0, 2, 2, 0),     # no m-circles
        (False, "+", False, 2, 2, 0, 1, 2, 0),    # no m-circles, no mirror image
        (True, "+", None, 3, 14, 1, 2, 3, 11),    # m-circles, mirror image, arrows, start marker
    ]
)
def test_plot_nyquist_bokeh(
    m_circles, start_marker, mirror_style, arrows, n_lines, n_vlines,
    n_arrows_sets, n_arrows, n_texts
):
    # verify that plot_nyquist adds the necessary objects to the plot

    s = symbols("s")
    tf1 = 1 / (s**2 + 0.5*s + 2)

    p = plot_nyquist(tf1, show=False, n=10, m_circles=m_circles, arrows=arrows,
        mirror_style=mirror_style, start_marker=start_marker, backend=BB)
    fig = p.fig
    assert len(p.fig.renderers) == n_lines
    assert len([t for t in p.fig.center if isinstance(t, Span)]) == n_vlines
    assert len([t for t in p.fig.center if isinstance(t, Arrow)]) == n_arrows_sets
    ad = [t for t in p.fig.center if isinstance(t, Arrow)][0].source.data
    assert len(ad["x_start"]) == n_arrows
    assert len([t for t in p.fig.center if isinstance(t, Label)]) == n_texts


@pytest.mark.skipif(ct is None, reason="control is not installed")
@pytest.mark.parametrize(
    "primary_style, mirror_style",
    [
        ("solid", "dotted"),
        (["solid", "dashdot"], ["dashed", "dotted"]),
        ({"line_dash": "solid"}, {"line_dash": "dotted"}),
        (
            [{"line_dash": "solid"}, {"line_dash": "dotted"}],
            [{"line_dash": "dashed"}, {"line_dash": "dashdot"}]
        ),
        (2, 2),
    ]
)
def test_plot_nyquist_bokeh_linestyles(primary_style, mirror_style):
    s = symbols("s")
    tf1 = 1 / (s**2 + 0.5*s + 2)

    p = plot_nyquist(tf1, show=False, n=10,
        primary_style=primary_style, mirror_style=mirror_style, backend=BB)
    if not isinstance(primary_style, int):
        fig = p.fig
    else:
        raises(ValueError, lambda: p.fig)


@pytest.mark.skipif(ct is None, reason="control is not installed")
def test_plot_nyquist_bokeh_interactive():
    # verify that interactive update doesn't raise errors

    a, s = symbols("a, s")
    tf = 1 / (s + a)
    pl = plot_nyquist(
        tf, xlim=(-2, 1), ylim=(-1, 1),
        aspect="equal", m_circles=True,
        params={a: (1, 0, 2)},
        arrows=4, n=10, show=False, backend=BB
    )
    fig = pl.backend.fig # force first draw
    pl.backend.update_interactive({a: 2}) # update with new value


def test_plot_nichols_bokeh():
    s = symbols("s")
    tf = (5 * (s - 1)) / (s**2 * (s**2 + s + 4))

    # with nichols grid lines
    p = plot_nichols(tf, ngrid=True, show=False, n=10, backend=BB)
    fig = p.fig
    assert len(p.fig.renderers) > 2
    assert len([t for t in p.fig.center if isinstance(t, Label)]) > 0

    # no nichols grid lines
    p = plot_nichols(tf, ngrid=False, show=False, n=10, backend=BB)
    fig = p.fig
    assert len(p.fig.renderers) == 1
    assert len([t for t in p.fig.center if isinstance(t, Label)]) == 0


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
    p = plot_nichols(tf, ngrid=False, show=False, n=10,
        backend=BB, arrows=arrows)
    fig = p.fig
    arrows_glyphs = [t for t in p.fig.center if isinstance(t, Arrow)]
    assert len(arrows_glyphs) == 1
    data = arrows_glyphs[0].source.data
    k = list(data.keys())[0]
    assert len(data[k]) == n_arrows
