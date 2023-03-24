import bokeh
from bokeh.models import ColumnDataSource
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
    assert len(f.toolbar.tools) == 6
    assert isinstance(f.toolbar.tools[0], bokeh.models.PanTool)
    assert isinstance(f.toolbar.tools[1], bokeh.models.WheelZoomTool)
    assert isinstance(f.toolbar.tools[2], bokeh.models.BoxZoomTool)
    assert isinstance(f.toolbar.tools[3], bokeh.models.ResetTool)
    assert isinstance(f.toolbar.tools[4], bokeh.models.HoverTool)
    assert isinstance(f.toolbar.tools[5], bokeh.models.SaveTool)
    assert f.toolbar.tools[4].tooltips == [('x', '$x'), ('y', '$y')]


def test_custom_colorloop():
    # verify that it is possible to modify the backend's class attributes
    # in order to change custom coloring

    assert len(BBchild.colorloop) != len(BB.colorloop)
    _p1 = custom_colorloop_1(BB)
    _p2 = custom_colorloop_1(BBchild)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert all([isinstance(t.glyph, bokeh.models.glyphs.Line) for t in f1.renderers])
    assert all([isinstance(t.glyph, bokeh.models.glyphs.Line) for t in f2.renderers])
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
    assert f.toolbar.tools[-2].tooltips == [('x', '$x'), ('y', '$y'), ("u", "@us")]


def test_plot3d_parametric_line():
    # verify that the backends produce the expected results when
    # `plot3d_parametric_line()` is called and `rendering_kw` overrides the
    # default line settings

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError, lambda: make_plot3d_parametric_line_1(BB,
        rendering_kw=dict(line_color="red")).process_series())


def test_plot3d():
    # verify that the backends produce the expected results when
    # `plot3d()` is called and `rendering_kw` overrides the default surface
    # settings

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: make_plot3d_1(BB, rendering_kw=dict(
            colorscale=[[0, "cyan"], [1, "cyan"]])).process_series())


def test_plot3d_2():
    # verify that the backends uses string labels when `plot3d()` is called
    # with `use_latex=False` and `use_cm=True`

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: make_plot3d_2(BB).process_series())


def test_plot3d_wireframe():
    # verify that wireframe=True is going to add the expected number of line
    # data series and that appropriate keyword arguments work as expected

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: make_plot3d_wireframe_1(BB).process_series())
    raises(
        NotImplementedError,
        lambda: make_plot3d_wireframe_2(BB, {}).process_series())
    raises(
        NotImplementedError,
        lambda: make_plot3d_wireframe_3(BB, {}).process_series())


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
    assert f.right[0].title == str(cos(x ** 2 + y ** 2))


def test_plot_vector_2d_quivers():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`quiver_kw` overrides the
    # default settings

    p = make_plot_vector_2d_quiver(
        BB,contour_kw=dict(), quiver_kw=dict(line_color="red"))
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
        BB, stream_kw=dict(line_color="red"), contour_kw=dict())
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
        BB, stream_kw=dict(line_color="red"), contour_kw=dict())
    f = p.fig
    assert f.right[0].title == "test"


def test_plot_vector_3d_quivers():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `quiver_kw` overrides the
    # default settings

    # Bokeh doesn't support 3D plots
    raises(NotImplementedError,
        lambda: make_plot_vector_3d_quiver(
            BB, quiver_kw=dict(sizeref=5)).process_series())


def test_plot_vector_3d_streamlines():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `stream_kw` overrides the
    # default settings

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: make_plot_vector_3d_streamlines_1(BB, stream_kw=dict(
            colorscale=[[0, "red"], [1, "red"]])).process_series())


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
    raises(NotImplementedError,
        lambda: make_test_plot_implicit_adaptive_true(
            BB, contour_kw=dict()).process_series())


def test_plot_implicit_adaptive_false():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True` and `contour_kw`
    # overrides the default settings

    # BokehBackend doesn't support 2D plots
    raises(NotImplementedError,
        lambda: make_test_plot_implicit_adaptive_false(
            BB, contour_kw=dict()).process_series())


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
    assert (f.toolbar.tools[-2].tooltips == [('x', '$x'), ('y', '$y'),
        ("Abs", "@abs"), ("Arg", "@arg")])


def test_plot_complex_3d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    # Bokeh doesn't support 3D plots
    raises(
        NotImplementedError,
        lambda: make_test_plot_complex_3d(BB, rendering_kw=dict()).process_series())


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
    assert all(isinstance(r.glyph, bokeh.models.glyphs.Line) for r in f.renderers)
    assert f.legend[0].items[0].label["value"] == str(Line((1, 2), (5, 4)))
    assert f.legend[0].items[1].label["value"] == str(Circle((0, 0), 4))
    assert f.legend[0].items[2].label["value"] == str(Polygon((2, 2), 3, n=6))


def test_plot_geometry_2():
    # verify that is_filled works correctly

    p = make_test_plot_geometry_2(BB, False)
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Line)]) == 4
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.MultiLine)]) == 1
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Scatter)]) == 1
    p = make_test_plot_geometry_2(BB, True)
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Line)]) == 1
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Patch)]) == 3
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.MultiLine)]) == 1
    assert len([t.glyph for t in p.fig.renderers if isinstance(t.glyph, bokeh.models.glyphs.Scatter)]) == 1


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
        mocker.patch(
            "bokeh.io.export_png",
            side_effect=RuntimeError
        )
        p = plot(sin(x), cos(x), **options)
        filename = "test_bokeh_save_1.png"
        raises(RuntimeError, lambda: p.save(os.path.join(tmpdir, filename)))

        mocker.patch(
            "bokeh.io.export_svg",
            side_effect=RuntimeError
        )
        p = plot(sin(x), cos(x), **options)
        filename = "test_bokeh_save_2.svg"
        raises(RuntimeError, lambda: p.save(os.path.join(tmpdir, filename)))

        p = plot(sin(x), cos(x), **options)
        filename = "test_bokeh_save_3.html"
        p.save(os.path.join(tmpdir, filename))

        p = plot(sin(x), cos(x), **options)
        filename = "test_bokeh_save_4.html"
        p.save(os.path.join(tmpdir, filename), resources=bokeh.resources.INLINE)


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
    assert p1.xlabel == p1.fig.xaxis.axis_label == '$x^{2}_{1}$'
    assert p2.xlabel == p2.fig.xaxis.axis_label == 'x_1^2'
    assert p1.ylabel == p1.fig.yaxis.axis_label == '$f\\left(x^{2}_{1}\\right)$'
    assert p2.ylabel == p2.fig.yaxis.axis_label == 'f(x_1^2)'


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
    assert f.right[0].title == "$%s$" % latex(cos(x ** 2 + y ** 2))


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

    p = make_test_plot_vector_2d_streamlines_custom_scalar_field_custom_label_use_latex(BB)
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


def test_plot_complex_use_latex():
    # complex plot function should return the same result (for axis labels)
    # wheter use_latex is True or False

    p = make_test_plot_complex_use_latex_1(BB)
    assert p.fig.right[0].title == 'Arg(cos(x) + I*sinh(x))'
    assert p.fig.xaxis.axis_label == "Real"
    assert p.fig.yaxis.axis_label == "Abs"

    p = make_test_plot_complex_use_latex_2(BB)
    assert p.fig.right[0].title == 'Argument'
    assert p.fig.xaxis.axis_label == "Re"
    assert p.fig.yaxis.axis_label == "Im"


def test_plot_real_imag_use_latex():
    # real/imag plot function should return the same result (for axis labels)
    # wheter use_latex is True or False

    p = make_test_plot_real_imag_use_latex(BB)
    assert p.fig.xaxis.axis_label == "$x$"
    assert p.fig.yaxis.axis_label == r"$f\left(x\right)$"
    assert p.fig.legend[0].items[0].label["value"] == 'Re(sqrt(x))'
    assert p.fig.legend[0].items[1].label["value"] == 'Im(sqrt(x))'


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

    raises(NotImplementedError,
        lambda : make_test_plot3d_implicit(BB).process_series())


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


def testupdate_interactive():
    # quick round down of test to verify that _update_interactive doesn't
    # raise errors

    u, v, x, y, z = symbols("u, v, x:z")

    p = plot(sin(u * x), (x, -pi, pi), adaptive=False, n=5,
        backend=BB, show=False, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend.update_interactive({u: 2})

    p = plot_parametric(cos(u * x), sin(u * x), (x, 0, 2*pi), adaptive=False,
        n=5, backend=BB, show=False, params={u: (1, 0, 2)},
        use_cm=True, is_point=False)
    p.backend.process_series()
    p.backend.update_interactive({u: 2})

    p = plot_parametric(cos(u * x), sin(u * x), (x, 0, 2*pi), adaptive=False,
        n=5, backend=BB, show=False, params={u: (1, 0, 2)},
        use_cm=True, is_point=True)
    p.backend.process_series()
    p.backend.update_interactive({u: 2})

    p = plot_parametric(cos(u * x), sin(u * x), (x, 0, 2*pi), adaptive=False,
        n=5, backend=BB, show=False, params={u: (1, 0, 2)}, use_cm=False)
    p.backend.process_series()
    p.backend.update_interactive({u: 2})

    p = plot_contour(cos(u * x**2 + y**2), (x, -2, 2), (y, -2, 2), backend=BB,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend.update_interactive({u: 2})

    p = plot_vector(Matrix([-u * y, x]), (x, -5, 5), (y, -4, 4),
        backend=BB, n=4, show=False, params={u: (1, 0, 2)}, streamlines=True)
    p.backend.process_series()
    p.backend.update_interactive({u: 2})

    p = plot_vector(Matrix([-u * y, x]), (x, -5, 5), (y, -4, 4),
        backend=BB, n=4, show=False, params={u: (1, 0, 2)}, streamlines=False,
        scalar=True)
    p.backend.process_series()
    p.backend.update_interactive({u: 2})

    p = plot_vector(Matrix([-u * y, x]), (x, -5, 5), (y, -4, 4),
        backend=BB, n=4, show=False, params={u: (1, 0, 2)}, streamlines=False,
        scalar=False)
    p.backend.process_series()
    p.backend.update_interactive({u: 2})

    p = plot_complex(sqrt(u * x), (x, -5 - 5 * I, 5 + 5 * I), show=False,
        backend=BB, threed=False, n=5, params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend.update_interactive({u: 2})

    from sympy.geometry import Line as SymPyLine
    p = plot_geometry(
        Polygon((2, u), 3, n=6),
        backend=BB, show=False, is_filled=True, use_latex=False,
        params={u: (1, 0, 2)})
    p.backend.process_series()
    p.backend.update_interactive({u: 2})


def test_generic_data_series():
    # verify that backends do not raise errors when generic data series
    # are used

    x = symbols("x")
    source = ColumnDataSource(data=dict(x=[0], y=[0], text=["test"]))

    p = plot(x, backend=BB, show=False, adaptive=False, n=5,
        markers=[{"x": [0, 1], "y": [0, 1], "marker": "square"}],
        annotations=[{"x": "x", "y": "y", "source": source}],
        fill=[{"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3], "y2": [0, 0, 0, 0]}],
        rectangles=[{"x": 0, "y": -3, "width": 5, "height": 2}])
    p.process_series()


def test_color_func_expr():
    # verify that passing an expression to color_func is supported

    p1 = make_test_color_func_expr_1(BB, False)
    p2 = make_test_color_func_expr_1(BB, True)

    # compute the original figure: no errors should be raised
    f1 = p1.fig
    f2 = p2.fig

    # update the figure with new parameters: no errors should be raised
    p1.backend.update_interactive({u: 0.5})
    # Bokeh don't raise an error because it doesn't apply a colormap
    # to streamlines
    p2.backend.update_interactive({u: 0.5})
