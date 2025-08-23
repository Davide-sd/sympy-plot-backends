# import mayavi
# from mayavi import mlab
# from pytest import raises
# import os
# from tempfile import TemporaryDirectory
# from .make_tests import *


# # NOTE: mayavi is extremely difficult to install on any setup
# # (OS + Python version + conda/pip, ...). There was a moment in time
# # where it was easy to install: as a matter of fact, v1.4.0 of this module
# # succedeed Github actions with mayavi installed. However, something
# # changed, and it became extremely difficult to install it. Hence, from
# # v1.5.0 of this module, MayaviBackend will only be tested locally on an
# # Ubuntu system with Python 3.10.
# # Consider removing MayaviBackend when merging this module into SymPy.


# # NOTE
# # While BB, PB, KB creates the figure at instantiation, MB creates the figure
# # once the `show()` method is called. All backends do not populate the figure
# # at instantiation. Numerical data is added only when `show()` or `fig` is
# # called.
# # In the following tests, we will use `show=False`, hence the `show()` method
# # won't be executed. To add numerical data to the plots we either call `fig`
# # or `draw()`.


# # prevent Mayavi window from opening
# mlab.options.offscreen = True


# class MABchild(MAB):
#     colorloop = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


# def test_colorloop_colormaps():
#     # verify that backends exposes important class attributes enabling
#     # automatic coloring

#     assert hasattr(MAB, "colorloop")
#     assert isinstance(MAB.colorloop, (list, tuple))
#     assert hasattr(MAB, "colormaps")
#     assert isinstance(MAB.colormaps, (list, tuple))


# def test_custom_colorloop():
#     # verify that it is possible to modify the backend's class attributes
#     # in order to change custom coloring

#     # NOTE: Mayavi applies colors only when the plot is shown. Luckily, we
#     # turned off the window from opening at the beginning of this module.
#     assert len(MAB.colorloop) != len(MABchild.colorloop)
#     _p1 = custom_colorloop_2(MAB, show=True)
#     _p2 = custom_colorloop_2(MABchild, show=True)
#     assert len(_p1.series) == len(_p2.series)
#     f1 = _p1.fig
#     f2 = _p2.fig
#     # there are 6 unique colors in _p1 and 3 unique colors in _p2
#     assert len(set([r.handles[0].actor.property.color for r in _p1.renderers])) == 6
#     assert len(set([r.handles[0].actor.property.color for r in _p2._renderers])) == 3


# def test_plot():
#     # verify that the backends produce the expected results when `plot()`
#     # is called and `rendering_kw` overrides the default line settings

#     # Mayavi doesn't support 2D plots
#     raises(
#         NotImplementedError,
#         lambda: make_test_plot(MAB,
#             rendering_kw=dict(color=(1, 0, 0))).draw())


# def test_plot_parametric():
#     # verify that the backends produce the expected results when
#     # `plot_parametric()` is called and `rendering_kw` overrides the default
#     # line settings

#     raises(
#         NotImplementedError,
#         lambda: make_test_plot_parametric(MAB,
#             rendering_kw=dict(color=(1, 0, 0))).draw())


# def test_plot3d_parametric_line():
#     # verify that the backends produce the expected results when
#     # `plot3d_parametric_line()` is called and `rendering_kw` overrides the
#     # default line settings

#     p = make_test_plot3d_parametric_line(MAB, rendering_kw=dict(color=(1, 0, 0)),
#         show=True)
#     assert len(p.series) == 1
#     f = p.fig
#     assert p.renderers[0].handles[0].actor.property.color == (1, 0, 0)


# def test_plot3d():
#     # verify that the backends produce the expected results when
#     # `plot3d()` is called and `rendering_kw` overrides the default surface
#     # settings

#     p = make_plot3d_1(MAB, rendering_kw=dict(color=(1, 0, 0)), show=True)
#     assert len(p.series) == 1
#     f = p.fig
#     assert p.renderers[0].handles[0].actor.property.color == (1, 0, 0)


# def test_plot3d_2():
#     # verify that the backends uses string labels when `plot3d()` is called
#     # with `use_latex=False` and `use_cm=True`

#     p = make_plot3d_2(MAB, show=True)
#     assert len(p.series) == 2
#     assert len(p.renderers) == 2
#     # orientation axis
#     o = p.fig.children[1].children[0].children[0].children[3]
#     xlabel = o.axes.x_axis_label_text
#     ylabel = o.axes.y_axis_label_text
#     zlabel = o.axes.z_axis_label_text
#     assert [xlabel, ylabel, zlabel] == ["x", "y", "f(x, y)"]



# def test_plot3d_wireframe():
#     # verify that wireframe=True is going to add the expected number of line
#     # data series and that appropriate keyword arguments work as expected

#     # TODO: make this test
#     pass


# def test_plot_contour():
#     # verify that the backends produce the expected results when
#     # `plot_contour()` is called and `rendering_kw` overrides the default
#     # surface settings

#     # Mayavi doesn't support 2D plots
#     raises(
#         NotImplementedError,
#         lambda: make_plot_contour_1(MAB,
#             rendering_kw=dict()).draw())


# def test_plot_vector_2d_quivers():
#     # verify that the backends produce the expected results when
#     # `plot_vector()` is called and `contour_kw`/`quiver_kw` overrides the
#     # default settings

#     # Mayavi doesn't support 2D plots
#     raises(
#         NotImplementedError,
#         lambda: make_test_plot_vector_2d_quiver(MAB, quiver_kw=dict(),
#             contour_kw=dict()).draw())


# def test_plot_vector_2d_streamlines_custom_scalar_field():
#     # verify that the backends produce the expected results when
#     # `plot_vector()` is called and `contour_kw`/`stream_kw` overrides the
#     # default settings

#     # Mayavi doesn't support 2D plots
#     raises(
#         NotImplementedError,
#         lambda: make_test_plot_vector_2d_streamlines(MAB, stream_kw=dict(),
#             contour_kw=dict()).draw())


# def test_plot_vector_3d_quivers():
#     # verify that the backends produce the expected results when
#     # `plot_vector()` is called and `quiver_kw` overrides the
#     # default settings

#     p = make_test_plot_vector_3d_quiver(MAB, quiver_kw=dict(color=(1, 0, 0)), show=True)
#     assert len(p.series) == 1
#     assert len(p.fig.children) == 1
#     assert p.renderers[0].handles[0].actor.property.color == (1, 0, 0)


# def test_plot_vector_3d_streamlines():
#     # verify that the backends produce the expected results when
#     # `plot_vector()` is called and `stream_kw` overrides the
#     # default settings

#     p = make_test_plot_vector_3d_streamlines(MAB, stream_kw=dict(color=(1, 0, 0)), show=True)
#     assert len(p.series) == 1
#     assert len(p.fig.children) == 1
#     assert p.renderers[0].handles[0].actor.property.color == (1, 0, 0)


# def test_plot_implicit_adaptive_true():
#     # verify that the backends produce the expected results when
#     # `plot_implicit()` is called with `adaptive=True`

#     # Mayavi doesn't support 2D plots
#     raises(NotImplementedError,
#         lambda: make_test_plot_implicit_adaptive_true(
#             MAB, rendering_kw=dict()).draw())


# def test_plot_implicit_adaptive_false():
#     # verify that the backends produce the expected results when
#     # `plot_implicit()` is called with `adaptive=True` and `contour_kw`
#     # overrides the default settings

#     # Mayavi doesn't support 2D plots
#     raises(NotImplementedError,
#         lambda: make_test_plot_implicit_adaptive_false(
#             MAB, contour_kw=dict()).draw())


# def test_plot_real_imag():
#     # verify that the backends produce the expected results when
#     # `plot_real_imag()` is called and `rendering_kw` overrides the default
#     # settings

#     # Mayavi doesn't support 2D plots
#     raises(NotImplementedError,
#         lambda: make_test_real_imag(MAB, rendering_kw=dict()).draw())


# def test_plot_complex_1d():
#     # verify that the backends produce the expected results when
#     # `plot_complex()` is called and `rendering_kw` overrides the default
#     # settings

#     # Mayavi doesn't support 2D plots
#     raises(NotImplementedError,
#         lambda: make_test_plot_complex_1d(MAB, rendering_kw=dict()).draw())


# def test_plot_complex_2d():
#     # verify that the backends produce the expected results when
#     # `plot_complex()` is called and `rendering_kw` overrides the default
#     # settings

#     # Mayavi doesn't support 2D plots
#     raises(NotImplementedError,
#         lambda: make_test_plot_complex_2d(MAB, rendering_kw=dict()).draw())


# def test_plot_complex_3d():
#     # verify that the backends produce the expected results when
#     # `plot_complex()` is called and `rendering_kw` overrides the default
#     # settings

#     # TODO
#     pass


# def test_plot_list_is_filled_false():
#     # verify that the backends produce the expected results when
#     # `plot_list()` is called with `is_filled=False`

#     # Mayavi doesn't support 2D plots
#     raises(
#         NotImplementedError,
#         lambda: make_test_plot_list_is_filled(MAB).draw())


# def test_plot_list_is_filled_true():
#     # verify that the backends produce the expected results when
#     # `plot_list()` is called with `is_filled=True`

#     # Mayavi doesn't support 2D plots
#     raises(
#         NotImplementedError,
#         lambda: make_test_plot_list_is_filled_true(MAB).draw())


# def test_plot_list_color_func():
#     # verify that the backends produce the expected results when
#     # `plot_list()` is called with `color_func`

#     # Mayavi doesn't support 2D plots
#     raises(
#         NotImplementedError,
#         lambda: make_test_plot_list_color_func(MAB).draw())


# def test_plot_piecewise_single_series():
#     # verify that plot_piecewise plotting 1 piecewise composed of N
#     # sub-expressions uses only 1 color

#     # Mayavi doesn't support 2D plots
#     raises(NotImplementedError, lambda: make_test_plot_piecewise_single_series(MAB))


# def test_plot_piecewise_multiple_series():
#     # verify that plot_piecewise plotting N piecewise expressions uses
#     # only N different colors

#     # Mayavi doesn't support 2D plots
#     raises(NotImplementedError, lambda: make_test_plot_piecewise_multiple_series(MAB))


# def test_plot_geometry_1():
#     # verify that the backends produce the expected results when
#     # `plot_geometry()` is called

#     # Mayavi doesn't support 2D plots
#     raises(NotImplementedError, lambda: make_test_plot_geometry_1(MAB).draw())


# def test_save():
#     # Verify that:
#     # 1. the save method accepts keyword arguments.
#     # 2. Bokeh and Plotly should not be able to save static pictures because
#     #    by default they need additional libraries. See the documentation of
#     #    the save methods for each backends to see what is required.
#     #    Hence, if in our system those libraries are installed, tests will
#     #    fail!
#     x, y, z = symbols("x:z")

#     with TemporaryDirectory(prefix="sympy_") as tmpdir:
#         p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), backend=MAB,
#             adaptive=False, n1=5, n2=5)
#         filename = "test_mab_save_1.png"
#         p.save(os.path.join(tmpdir, filename))


# def test_aspect_ratio_3d():
#     # verify that the backends apply the provided aspect ratio.
#     # NOTE:
#     # 1. read the backend docs to understand which options are available.
#     # 2. K3D doesn't use the `aspect` keyword argument.

#     p = make_test_aspect_ratio_3d(MAB, "equal", show=True)
#     assert p.aspect == "equal"
#     assert np.allclose(
#         p.fig.children[0].children[0].children[0].children[1].axes.bounds,
#         [-2, 2, -2, 2, -1, 1], rtol=1e-02)

#     p = make_test_aspect_ratio_3d(MAB, "auto", show=True)
#     assert p.aspect == "auto"
#     assert np.allclose(
#         p.fig.children[0].children[0].children[0].children[1].axes.bounds,
#         [0, 1, 0, 1, 0, 1], rtol=1e-02)


# def test_backend_latex_labels():
#     # verify that backends are going to set axis latex-labels in the
#     # 2D and 3D case

#     p1 = make_test_backend_latex_labels_2(MAB, True, show=True)
#     p2 = make_test_backend_latex_labels_2(MAB, False, show=True)
#     o1 = p1.fig.children[0].children[0].children[0].children[3]
#     xlabel1 = o1.axes.x_axis_label_text
#     ylabel1 = o1.axes.y_axis_label_text
#     zlabel1 = o1.axes.z_axis_label_text
#     assert p1.xlabel == xlabel1 == '$x^{2}_{1}$'
#     assert p1.ylabel == ylabel1 == '$x_{2}$'
#     assert p1.zlabel == zlabel1 == '$f\\left(x^{2}_{1}, x_{2}\\right)$'
#     o2 = p2.fig.children[0].children[0].children[0].children[3]
#     xlabel2 = o2.axes.x_axis_label_text
#     ylabel2 = o2.axes.y_axis_label_text
#     zlabel2 = o2.axes.z_axis_label_text
#     assert p2.xlabel == xlabel2 == 'x_1^2'
#     assert p2.ylabel == ylabel2 == 'x_2'
#     assert p2.zlabel == zlabel2 == 'f(x_1^2, x_2)'


# def test_plot3d_parametric_line_use_latex():
#     # verify that the colorbar uses latex label

#     p = make_test_plot3d_parametric_line_use_latex(MAB, True)
#     assert p.fig.children[0].children[0].children[0].children[0].scalar_lut_manager.scalar_bar.title == "$x$"


# def test_plot3d_use_latex():
#     # verify that the colorbar uses latex label

#     p = make_test_plot3d_use_latex(MAB, True)
#     assert p.fig.children[0].children[0].children[0].scalar_lut_manager.scalar_bar.title == "$%s$" % latex(cos(x ** 2 + y ** 2))
#     assert p.fig.children[1].children[0].children[0].scalar_lut_manager.scalar_bar.title == "$%s$" % latex(sin(x ** 2 + y ** 2))


# def test_plot_vector_3d_quivers_use_latex():
#     # verify that the colorbar uses latex label

#     p = make_test_plot_vector_3d_quivers_use_latex(MAB, True)
#     assert p.fig.children[0].children[0].vector_lut_manager.scalar_bar.title == '$\\left( z, \\  y, \\  x\\right)$'


# def test_plot_vector_3d_streamlines_use_latex():
#     # verify that the colorbar uses latex label

#     p = make_test_plot_vector_3d_streamlines_use_latex(MAB, True)
#     assert p.fig.children[0].children[0].children[0].scalar_lut_manager.scalar_bar.title == '$\\left( z, \\  y, \\  x\\right)$'


# def test_plot3d_use_cm():
#     # verify that use_cm produces the expected results on plot3d

#     p1 = make_test_plot3d_use_cm(MAB, True, show=True)
#     p2 = make_test_plot3d_use_cm(MAB, False, show=True)
#     assert np.allclose(
#         p1.fig.children[0].children[0].children[0].children[0].actor.property.color,
#         (1, 1, 1)
#     )
#     assert np.allclose(
#         p2.fig.children[0].children[0].children[0].children[0].actor.property.color,
#         (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
#     )


# def test_plot3d_implicit():
#     # verify that plot3d_implicit don't raise errors

#     p = make_test_plot3d_implicit(MAB, True)
#     assert len(p.fig.children) > 0


# def test_surface_color_func():
#     # After the addition of `color_func`, `SurfaceOver2DRangeSeries` and
#     # `ParametricSurfaceSeries` returns different elements.
#     # Verify that backends do not raise errors when plotting surfaces and that
#     # the color function is applied.

#     p1 = make_test_surface_color_func_1(MAB, lambda x, y, z: z, show=True)
#     p2 = make_test_surface_color_func_1(MAB, lambda x, y, z: np.sqrt(x**2 + y**2), show=True)
#     # NOTE: no idea where Mayavi stores the entire scalar field, just the min
#     # and max
#     assert np.allclose(
#         p1.fig.children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
#         [-1,  1],
#         rtol=1e-01
#     )
#     assert np.allclose(
#         p2.fig.children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
#         [0, 4.24264069],
#         rtol=1e-01
#     )

#     p1 = make_test_surface_color_func_2(MAB, lambda x, y, z, u, v: z, show=True)
#     p2 = make_test_surface_color_func_2(MAB, lambda x, y, z, u, v: np.sqrt(x**2 + y**2), show=True)
#     # NOTE: no idea where Mayavi stores the entire scalar field, just the min
#     # and max
#     assert np.allclose(
#         p1.fig.children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
#         [-3,  3],
#         rtol=1e-01
#     )
#     assert np.allclose(
#         p2.fig.children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
#         [0, 3.],
#         rtol=1e-01
#     )


# def test_line_color_plot3d_parametric_line():
#     # verify back-compatibility with old sympy.plotting module when using
#     # line_color

#     p = make_test_line_color_plot3d_parametric_line(MAB, (1, 0, 0), False, True)
#     p.renderers[0].handles[0].actor.property.color == (1, 0, 0)
#     p = make_test_line_color_plot3d_parametric_line(MAB, lambda x: -x, True, True)
#     assert np.allclose(
#         p.fig.children[0].children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
#         [-2*np.pi, 0],
#         rtol=1e-02
#     )


# def test_surface_color_plot3d():
#     # verify back-compatibility with old sympy.plotting module when using
#     # surface_color

#     p = make_test_surface_color_plot3d(MAB, (1, 0, 0), False, True)
#     p.renderers[0].handles[0].actor.property.color == (1, 0, 0)
#     p = make_test_surface_color_plot3d(MAB, lambda x: -x, True, True)
#     assert np.allclose(
#         p.fig.children[0].children[0].children[0].children[0].actor.mapper.scalar_range,
#         [-1, 1],
#         rtol=1e-02
#     )
