from spb.series import Parametric3DLineSeries
import k3d
import numpy as np
import pytest
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
# or `draw()`.


KB.skip_notebook_check = True


class KBchild1(KB):
    colorloop = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def test_colorloop_colormaps():
    # verify that backends exposes important class attributes enabling
    # automatic coloring

    assert hasattr(KB, "colorloop")
    assert isinstance(KB.colorloop, (list, tuple))
    assert hasattr(KB, "colormaps")
    assert isinstance(KB.colormaps, (list, tuple))


def test_custom_colorloop():
    # verify that it is possible to modify the backend's class attributes
    # in order to change custom coloring

    assert len(KB.colorloop) != len(KBchild1.colorloop)
    _p1 = custom_colorloop_2(KB)
    _p2 = custom_colorloop_2(KBchild1)
    assert len(_p1.series) == len(_p2.series)
    f1 = _p1.fig
    f2 = _p2.fig
    assert all([isinstance(t, k3d.objects.Mesh) for t in f1.objects])
    assert all([isinstance(t, k3d.objects.Mesh) for t in f2.objects])
    # there are 6 unique colors in _p1 and 3 unique colors in _p2
    assert len(set([o.color for o in f1.objects])) == 6
    assert len(set([o.color for o in f2.objects])) == 3


def test_plot():
    # verify that the backends produce the expected results when `plot()`
    # is called and `rendering_kw` overrides the default line settings

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_plot_1(KB,
            rendering_kw=dict(line_color="red")).draw())


def test_plot_parametric():
    # verify that the backends produce the expected results when
    # `plot_parametric()` is called and `rendering_kw` overrides the default
    # line settings

    raises(
        NotImplementedError,
        lambda: make_plot_parametric_1(KB,
            rendering_kw=dict(line_color="red")).draw())


def test_plot3d_parametric_line():
    # verify that the backends produce the expected results when
    # `plot3d_parametric_line()` is called and `rendering_kw` overrides the
    # default line settings

    p = make_plot3d_parametric_line_1(KB, rendering_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Line)
    assert f.objects[0].color == 16711680
    assert f.objects[0].name is None


def test_plot3d():
    # verify that the backends produce the expected results when
    # `plot3d()` is called and `rendering_kw` overrides the default surface
    # settings

    p = make_plot3d_1(KB, rendering_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Mesh)
    assert f.objects[0].color == 16711680
    assert f.objects[0].name is None


def test_plot3d_2():
    # verify that the backends uses string labels when `plot3d()` is called
    # with `use_latex=False` and `use_cm=True`

    p = make_plot3d_2(KB)
    assert len(p.series) == 2
    f = p.fig
    assert len(f.objects) == 2
    assert p.fig.axes == ["x", "y", "f(x, y)"]



def test_plot3d_wireframe():
    # verify that wireframe=True is going to add the expected number of line
    # data series and that appropriate keyword arguments work as expected

    p2 = make_plot3d_wireframe_1(KB)
    assert all(p2.fig.objects[1].color == 0 for s in p2.series[1:])

    p3 = make_plot3d_wireframe_2(KB, {"color": 0xff0000})
    assert all(t.color == 0xff0000 for t in p3.fig.objects[1:])

    p4 = make_plot3d_wireframe_3(PB, {"line_color": "red"})
    assert len(p4.series) == 1 + 20 + 40
    assert isinstance(p4[0], SurfaceOver2DRangeSeries)
    assert all(isinstance(s, Parametric3DLineSeries) for s in p4.series[1:])
    assert all((not s.adaptive) and (s.n[0] == p4[0].n[1]) for s in p4.series[1:21])
    assert all((not s.adaptive) and (s.n[0] == p4[0].n[0]) for s in p4.series[21:])
    assert all(t["line"]["color"] == "red" for t in p4.fig.data[1:])
    assert np.allclose(
        [t.x[0] for t in p4.fig.data[1:21]], np.linspace(0, 3.25, 20))
    param = p4.series[1].get_data()[-1]
    assert np.isclose(param.min(), 0) and np.isclose(param.max(), 2*np.pi)
    param = p4.series[21].get_data()[-1]


def test_plot_contour():
    # verify that the backends produce the expected results when
    # `plot_contour()` is called and `rendering_kw` overrides the default
    # surface settings

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_plot_contour_1(KB,
            rendering_kw=dict()).draw())


def test_plot_vector_2d_quivers():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`quiver_kw` overrides the
    # default settings

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_plot_vector_2d_quiver(KB, quiver_kw=dict(),
            contour_kw=dict()).draw())


def test_plot_vector_2d_streamlines_custom_scalar_field():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `contour_kw`/`stream_kw` overrides the
    # default settings

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_plot_vector_2d_streamlines_1(KB, stream_kw=dict(),
            contour_kw=dict()).draw())


def test_plot_vector_3d_quivers():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `quiver_kw` overrides the
    # default settings

    p = make_plot_vector_3d_quiver(KB, quiver_kw=dict(scale=0.5, color=16711680), use_cm=False)
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Vectors)
    assert all([c == 16711680 for c in f.objects[0].colors])


def test_plot_vector_3d_streamlines():
    # verify that the backends produce the expected results when
    # `plot_vector()` is called and `stream_kw` overrides the
    # default settings

    p = make_plot_vector_3d_streamlines_1(KB, stream_kw=dict(color=16711680))
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Line)
    assert f.objects[0].color == 16711680

    # test different combinations for streamlines: it should not raise errors
    p = make_plot_vector_3d_streamlines_1(KB, stream_kw=dict(starts=True))
    p = make_plot_vector_3d_streamlines_1(KB, stream_kw=dict(starts={
        "x": np.linspace(-5, 5, 10),
        "y": np.linspace(-4, 4, 10),
        "z": np.linspace(-3, 3, 10),
    }))

    # other keywords: it should not raise errors
    p = make_plot_vector_3d_streamlines_1(
        KB, stream_kw=dict(), kwargs=dict(use_cm=False))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_plot_vector_3d_normalize():
    # verify that backends are capable of normalizing a vector field before
    # plotting it. Since all backend are different from each other, let's test
    # that data in the figures is different in the two cases normalize=True
    # and normalize=False

    p1 = make_plot_vector_3d_normalize_1(KB, False)
    p2 = make_plot_vector_3d_normalize_1(KB, True)
    assert not np.allclose(p1.fig.objects[0].vectors, p2.fig.objects[0].vectors)

    p1 = make_plot_vector_3d_normalize_2(KB, False)
    p1.backend.update_interactive({u: 1.5})
    p2 = make_plot_vector_3d_normalize_2(KB, True)
    p2.backend.update_interactive({u: 1.5})
    assert not np.allclose(p1.fig.objects[0].vectors, p2.fig.objects[0].vectors)


def test_plot_vector_3d_quivers_color_func():
    # verify that color_func gets applied to 3D quivers

    p1 = make_plot_vector_3d_quiver_color_func_1(KB, None)
    p2 = make_plot_vector_3d_quiver_color_func_1(KB, lambda x, y, z, u, v, w: x)
    assert not np.allclose(p1.fig.objects[0].colors, p2.fig.objects[0].colors)

    p1 = make_plot_vector_3d_quiver_color_func_2(KB, None)
    p2 = make_plot_vector_3d_quiver_color_func_2(KB, lambda x, y, z, u, v, w: np.cos(u))
    p3 = make_plot_vector_3d_quiver_color_func_2(KB, lambda x, y, z, u, v, w: np.cos(u))
    assert not np.allclose(p1.fig.objects[0].colors, p2.fig.objects[0].colors)
    p3.backend.update_interactive({a: 2})
    assert not np.allclose(p2.fig.objects[0].colors, p3.fig.objects[0].colors)


def test_plot_vector_3d_streamlines_color_func():
    # verify that color_func gets applied to 3D quivers

    p1 = make_plot_vector_3d_streamlines_color_func(KB, None)
    p2 = make_plot_vector_3d_streamlines_color_func(KB, lambda x, y, z, u, v, w: x)
    assert not np.allclose(
        p1.fig.objects[0].attribute, p2.fig.objects[0].attribute)


def test_plot_implicit_adaptive_true():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True`

    # K3D doesn't support 2D plots
    raises(NotImplementedError,
        lambda: make_test_plot_implicit_adaptive_true(
            KB, rendering_kw=dict()).draw())


def test_plot_implicit_adaptive_false():
    # verify that the backends produce the expected results when
    # `plot_implicit()` is called with `adaptive=True` and `contour_kw`
    # overrides the default settings

    # K3D doesn't support 2D plots
    raises(NotImplementedError,
        lambda: make_test_plot_implicit_adaptive_false(
            KB, rendering_kw=dict()).draw())


def test_plot_real_imag():
    # verify that the backends produce the expected results when
    # `plot_real_imag()` is called and `rendering_kw` overrides the default
    # settings

    # K3D doesn't support 2D plots
    raises(NotImplementedError,
        lambda: make_test_real_imag(KB, rendering_kw=dict()).draw())


def test_plot_complex_1d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    # K3D doesn't support 2D plots
    raises(NotImplementedError,
        lambda: make_test_plot_complex_1d(KB, rendering_kw=dict()).draw())


def test_plot_complex_2d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    # K3D doesn't support 2D plots
    raises(NotImplementedError,
        lambda: make_test_plot_complex_2d(KB, rendering_kw=dict()).draw())


def test_plot_complex_3d():
    # verify that the backends produce the expected results when
    # `plot_complex()` is called and `rendering_kw` overrides the default
    # settings

    p = make_test_plot_complex_3d(KB, rendering_kw=dict())
    assert len(p.series) == 1
    f = p.fig
    assert len(f.objects) == 1
    assert isinstance(f.objects[0], k3d.objects.Mesh)
    assert f.objects[0].name is None


def test_plot_list_is_filled_false():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=False`

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_test_plot_list_is_filled_false(KB).draw())


def test_plot_list_is_filled_true():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `is_filled=True`

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_test_plot_list_is_filled_true(KB).draw())


def test_plot_list_color_func():
    # verify that the backends produce the expected results when
    # `plot_list()` is called with `color_func`

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_test_plot_list_color_func(KB).draw())


def test_plot_piecewise_single_series():
    # verify that plot_piecewise plotting 1 piecewise composed of N
    # sub-expressions uses only 1 color

    # K3D doesn't support 2D plots
    raises(NotImplementedError, lambda: make_test_plot_piecewise_single_series(KB))


def test_plot_piecewise_multiple_series():
    # verify that plot_piecewise plotting N piecewise expressions uses
    # only N different colors

    # K3D doesn't support 2D plots
    raises(NotImplementedError, lambda: make_test_plot_piecewise_multiple_series(KB))


def test_plot_geometry_1():
    # verify that the backends produce the expected results when
    # `plot_geometry()` is called

    # K3D doesn't support 2D plots
    raises(
        NotImplementedError,
        lambda: make_test_plot_geometry_1(KB).draw())


def test_plot_geometry_3d():
    # verify that no errors are raised when 3d geometric entities are plotted

    p = make_test_plot_geometry_3d(KB)
    p.draw()


def test_save():
    # NOTE: K3D is designed in such a way that the plots need to be shown
    # on the screen before saving them. Since it is not possible to show
    # them on the screen during tests, we are only going to test that it
    # proceeds smoothtly or it raises errors when wrong options are given

    x, y, z = symbols("x:z")
    options = dict(backend=KB, adaptive=False, n=5, show=False)

    with TemporaryDirectory(prefix="sympy_") as tmpdir:

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), **options)
        filename = "test_k3d_save_1.png"
        raises(ValueError, lambda: p.save(os.path.join(tmpdir, filename)))

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), **options)
        filename = "test_k3d_save_2.jpg"
        raises(ValueError, lambda: p.save(os.path.join(tmpdir, filename)))

        # unexpected keyword argument
        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), **options)
        filename = "test_k3d_save_3.html"
        raises(TypeError, lambda: p.save(os.path.join(tmpdir, filename),
            parameter=True))

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), **options)
        filename = "test_k3d_save_4.html"
        p.save(os.path.join(tmpdir, filename))

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), **options)
        filename = "test_k3d_save_4.html"
        p.save(os.path.join(tmpdir, filename), include_js=True)

        p = plot3d(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3), **options)
        filename = "test_k3d_save_4.html"
        raises(TypeError, lambda: p.save(os.path.join(tmpdir, filename),
            include_js=True, parameter=True))


def test_vectors_3d_update_interactive():
    # Some backends do not support streamlines with iplot. Test that the
    # backends raise error.

    p = make_test_vectors_3d_update_interactive(KB)
    raises(NotImplementedError,
        lambda:p.backend.update_interactive({a: 2, b: 2, c: 2}))


def test_backend_latex_labels():
    # verify that backends are going to set axis latex-labels in the
    # 2D and 3D case

    p1 = make_test_backend_latex_labels_2(KB, True)
    p2 = make_test_backend_latex_labels_2(KB, False)
    assert p1.xlabel == p1.fig.axes[0] == 'x^{2}_{1}'
    assert p1.ylabel == p1.fig.axes[1] == 'x_{2}'
    assert p1.zlabel == p1.fig.axes[2] == 'f\\left(x^{2}_{1}, x_{2}\\right)'
    assert p2.xlabel == p2.fig.axes[0] == 'x_1^2'
    assert p2.ylabel == p2.fig.axes[1] == 'x_2'
    assert p2.zlabel == p2.fig.axes[2] == 'f(x_1^2, x_2)'


def test_plot3d_parametric_line_use_latex():
    # verify that the colorbar uses latex label

    # NOTE: K3D doesn't show a label to colorbar
    p = make_test_plot3d_parametric_line_use_latex(KB)


def test_plot3d_use_latex():
    # verify that the colorbar uses latex label

    p = make_test_plot3d_use_latex(KB)
    f = p.fig
    assert p.fig.axes == ["x", "y", "f\\left(x, y\\right)"]


def test_plot_vector_3d_quivers_use_latex():
    # verify that the colorbar uses latex label

    # K3D doesn't show label on colorbar
    p = make_test_plot_vector_3d_quivers_use_latex(KB)
    assert len(p.series) == 1


def test_plot_vector_3d_streamlines_use_latex():
    # verify that the colorbar uses latex label

    # K3D doesn't show labels on colorbar
    p = make_test_plot_vector_3d_streamlines_use_latex(KB)
    assert len(p.series) == 1


def test_plot3d_use_cm():
    # verify that use_cm produces the expected results on plot3d

    p1 = make_test_plot3d_use_cm(KB, True)
    p2 = make_test_plot3d_use_cm(KB, False)
    n1 = len(p1.fig.objects[0].color_map)
    n2 = len(p2.fig.objects[0].color_map)
    if n1 == n2:
        assert not np.allclose(p1.fig.objects[0].color_map, p2.fig.objects[0].color_map)


def test_k3d_vector_pivot():
    # verify that K3DBackend accepts quiver_kw={"pivot": "something"} and
    # produces different results
    x, y, z = symbols("x, y, z")

    _plot_vector = lambda pivot: plot_vector(
        Matrix([z, y, x]),
        (x, -5, 5),
        (y, -4, 4),
        (z, -3, 3),
        n1=3, n2=3, n3=3,
        quiver_kw={"pivot": pivot},
        backend=KB,
        show=False
    )

    p1 = _plot_vector("tail")
    p2 = _plot_vector("mid")
    p3 = _plot_vector("tip")
    assert not np.allclose(p1.fig.objects[0].origins, p2.fig.objects[0].origins, equal_nan=True)
    assert not np.allclose(p1.fig.objects[0].origins, p3.fig.objects[0].origins, equal_nan=True)
    assert not np.allclose(p2.fig.objects[0].origins, p3.fig.objects[0].origins, equal_nan=True)


def test_plot3d_implicit():
    # verify that plot3d_implicit don't raise errors

    p = make_test_plot3d_implicit(KB)
    assert isinstance(p.fig.objects[0], k3d.objects.MarchingCubes)


def test_surface_color_func():
    # After the addition of `color_func`, `SurfaceOver2DRangeSeries` and
    # `ParametricSurfaceSeries` returns different elements.
    # Verify that backends do not raise errors when plotting surfaces and that
    # the color function is applied.

    p1 = make_test_surface_color_func_1(KB, lambda x, y, z: z)
    p2 = make_test_surface_color_func_1(KB, lambda x, y, z: np.sqrt(x**2 + y**2))
    assert not np.allclose(p1.fig.objects[0].attribute, p2.fig.objects[0].attribute)

    p1 = make_test_surface_color_func_2(KB, lambda x, y, z, u, v: z)
    p2 = make_test_surface_color_func_2(KB, lambda x, y, z, u, v: np.sqrt(x**2 + y**2))
    assert not np.allclose(p1.fig.objects[0].attribute, p2.fig.objects[0].attribute)


def test_surface_interactive_color_func():
    # After the addition of `color_func`, `SurfaceInteractiveSeries` and
    # `ParametricSurfaceInteractiveSeries` returns different elements.
    # Verify that backends do not raise errors when updating surfaces and a
    # color function is applied.

    p = make_test_surface_interactive_color_func(KB)
    p.update_interactive({t: 2})
    assert not np.allclose(p.fig.objects[0].attribute, p.fig.objects[1].attribute)
    assert not np.allclose(p.fig.objects[2].attribute, p.fig.objects[3].attribute)


def test_line_color_plot3d_parametric_line():
    # verify back-compatibility with old sympy.plotting module when using
    # line_color

    p = make_test_line_color_plot3d_parametric_line(KB, 0xff0000, False)
    assert p.fig.objects[0].color == 16711680
    p = make_test_line_color_plot3d_parametric_line(KB, lambda x: -x, True)
    assert len(p.fig.objects[0].attribute) > 0


def test_surface_color_plot3d():
    # verify back-compatibility with old sympy.plotting module when using
    # surface_color

    p = make_test_surface_color_plot3d(KB, 0xff0000, False)
    assert p.fig.objects[0].color == 16711680
    p = make_test_surface_color_plot3d(KB, lambda x, y, z: -x, True)
    assert len(p.fig.objects[0].attribute) > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_k3d_high_aspect_ratio_meshes():
    # K3D is not great at dealing with high aspect ratio meshes. So, users
    # should set zlim and the backend should add clipping planes and modify
    # the camera position.

    z = symbols("z")
    p1 = plot_complex(1 / sin(pi + z**3), (z, -2-2j, 2+2j),
        grid=False, threed=True, use_cm=True, backend=KB, coloring="a",
        n=5, show=False)
    p1.draw()
    p2 = plot_complex(1 / sin(pi + z**3), (z, -2-2j, 2+2j),
        grid=False, threed=True, use_cm=True, backend=KB, coloring="a",
        n=5, zlim=(0, 6), show=False)
    p2.draw()

    assert p1._bounds != p2._bounds
    assert p1.fig.camera != p2.fig.camera
    assert len(p1.fig.clipping_planes) == 0
    assert p1.fig.clipping_planes != p2.fig.clipping_planes


def test_update_interactive():
    # quick round down of test to verify that _update_interactive doesn't
    # raise errors

    u, v, x, y, z = symbols("u, v, x:z")

    # points
    p = plot3d_parametric_line(
        cos(u * x), sin(x), x, (x, -pi, pi), backend=KB, is_point=True,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    # line
    p = plot3d_parametric_line(
        cos(u * x), sin(x), x, (x, -pi, pi), backend=KB, is_point=False,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    p = plot3d(cos(u * x**2 + y**2), (x, -2, 2), (y, -2, 2), backend=KB,
        show=False, adaptive=False, n=5, params={u: (1, 0, 2)})
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    u, v = symbols("u, v")
    fx = (1 + v / 2 * cos(u / 2)) * cos(x * u)
    fy = (1 + v / 2 * cos(u / 2)) * sin(x * u)
    fz = v / 2 * sin(u / 2)
    p = plot3d_parametric_surface(fx, fy, fz, (u, 0, 2*pi), (v, -1, 1),
        backend=KB, use_cm=True, n1=5, n2=5, show=False,
        params={x: (1, 0, 2)})
    p.backend.draw()
    p.backend.update_interactive({x: 2})

    p = plot_vector(Matrix([u * z, y, x]), (x, -5, 5), (y, -4, 4), (z, -3, 3),
        backend=KB, n=4, show=False, params={u: (1, 0, 2)})
    p.backend.draw()
    p.backend.update_interactive({u: 2})

    p = plot_complex(sqrt(u * x), (x, -5 - 5 * I, 5 + 5 * I), show=False,
        backend=KB, threed=True, use_cm=True, n=5, params={u: (1, 0, 2)})
    p.backend.draw()
    p.backend.update_interactive({u: 2})


def test_plot3d_list_use_cm_False():
    # verify that plot3d_list produces the expected results when no color map
    # is required

    # solid color line
    p = make_test_plot3d_list_use_cm_False(KB, False, False)
    assert isinstance(p.fig.objects[0], k3d.objects.Line)

    # solid color markers with empty faces
    # NOTE: k3d doesn't support is_filled
    p = make_test_plot3d_list_use_cm_False(KB, True, False)
    assert isinstance(p.fig.objects[0], k3d.objects.Points)

    # solid color markers with filled faces
    # NOTE: k3d doesn't support is_filled
    p = make_test_plot3d_list_use_cm_False(KB, True, True)
    assert isinstance(p.fig.objects[0], k3d.objects.Points)


def test_plot3d_list_use_cm_color_func():
    # verify that use_cm=True and color_func do their job

    # NOTE: k3d doesn't support is_filled

    # line with colormap
    # if color_func is not provided, the same parameter will be used
    # for all points
    p1 = make_test_plot3d_list_use_cm_color_func(KB, False, False, None)
    c1 = p1.fig.objects[0].attribute
    p2 = make_test_plot3d_list_use_cm_color_func(KB, False, False, lambda x, y, z: x)
    c2 = p2.fig.objects[0].attribute
    assert not np.allclose(c1, c2)

    # markers with empty faces
    p1 = make_test_plot3d_list_use_cm_color_func(KB, False, False, None)
    c1 = p1.fig.objects[0].attribute
    p2 = make_test_plot3d_list_use_cm_color_func(KB, False, False, lambda x, y, z: x)
    c2 = p2.fig.objects[0].attribute
    assert not np.allclose(c1, c2)

def test_plot3d_list_interactive():
    # verify that no errors are raises while updating a plot3d_list

    p = make_test_plot3d_list_interactive(MB)
    p.backend.update_interactive({t: 1})


def test_color_func_expr():
    # verify that passing an expression to color_func is supported

    p3 = make_test_color_func_expr_2(KB)
    # compute the original figure: no errors should be raised
    f3 = p3.fig
    # update the figure with new parameters: no errors should be raised
    p3.backend.update_interactive({u: 0.5})


def test_plot_vector_3d_quivers_default_color_func():
    # verify that the default color function based on the magnitude of the
    # vector field is applied correctly

    x, y, z = symbols("x:z")

    p1 = plot_vector([z, y, x], (x, -10, 10), (y, -10, 10), (z, -10, 10),
        backend=KB, n=4, use_cm=True, show=False)
    p2 = plot_vector([z, y, x], (x, -10, 10), (y, -10, 10), (z, -10, 10),
        backend=KB, n=4, use_cm=False, show=False)
    # just test that p2 colors are all equal to each other and p1 colors are
    # different from p2
    assert not np.allclose(
        p1.fig.objects[0].colors, p2.fig.objects[0].colors)
    assert np.allclose(p2.fig.objects[0].colors, p2.fig.objects[0].colors[0])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_make_analytic_landscape_black_and_white():
    # verify that the backend doesn't raise an error when grayscale coloring
    # schemes are required

    p = make_test_analytic_landscape(KB)
    p.fig


def test_parametric_texts():
    # verify that xlabel, ylabel, zlabel, title accepts parametric texts
    a, b, p = make_test_parametric_texts_3d(KB)
    xl, yl, zl = p.fig.axes
    assert p.fig.objects[1].text == "a=1.0, a+b=1.000"
    assert xl == "test a=1.00"
    assert yl == "test b=0.00"
    assert zl == "test a=1.00, b=0.00"
    assert len(p.fig.objects) == 2
    p.backend.update_interactive({a: 1.5, b: 2})
    xl, yl, zl = p.fig.axes
    assert p.fig.objects[1].text == "a=1.5, a+b=3.500"
    assert xl == "test a=1.50"
    assert yl == "test b=2.00"
    assert zl == "test a=1.50, b=2.00"
    assert len(p.fig.objects) == 2
