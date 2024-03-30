from spb import (
    plot, plot_parametric, plot_polar, plot_list, plot_implicit,
    plot_contour, plot_piecewise, plot_geometry,
    plot3d_parametric_line, plot3d, plot3d_list,
    plot3d_implicit, plot3d_parametric_surface,
    plot_vector, plot_complex, plot_real_imag, plot_riemann_sphere,
    graphics, arrow_2d, arrow_3d, plot_root_locus, plot_pole_zero,
    ngrid, sgrid, zgrid, mcircles
)
from spb.series import (
    SurfaceOver2DRangeSeries, ParametricSurfaceSeries, LineOver1DRangeSeries
)
from sympy import (
    symbols, sin, cos, pi, exp, Matrix, sqrt, Heaviside, Piecewise, Eq, I,
    Circle, Line, Polygon, Ellipse, Curve, Point2D, Segment, Rational,
    Point3D, Line3D, Plane, log, gamma, tan
)
from sympy.abc import a, b, c, x, y, z, u, v, t
import numpy as np


def options():
    return dict(n=5, adaptive=False, show=False)


def custom_colorloop_1(B):
    return plot(
        sin(x),
        cos(x),
        sin(x / 2),
        cos(x / 2),
        2 * sin(x),
        2 * cos(x),
        backend=B,
        **options()
    )


def custom_colorloop_2(B, show=False):
    opts = options()
    opts["show"] = show
    return plot3d(
        (cos(x**2 + y**2), (x, -3, -2), (y, -3, 3)),
        (cos(x**2 + y**2), (x, -2, -1), (y, -3, 3)),
        (cos(x**2 + y**2), (x, -1, 0), (y, -3, 3)),
        (cos(x**2 + y**2), (x, 0, 1), (y, -3, 3)),
        (cos(x**2 + y**2), (x, 1, 2), (y, -3, 3)),
        (cos(x**2 + y**2), (x, 2, 3), (y, -3, 3)),
        use_cm=False,
        backend=B,
        **opts
    )


def make_plot_1(B, rendering_kw, use_latex=False):
    return plot(
        sin(x),
        cos(x),
        rendering_kw=rendering_kw,
        backend=B,
        legend=True,
        use_latex=use_latex,
        **options()
    )


def make_plot_parametric_1(B, rendering_kw):
    return plot_parametric(
        cos(x), sin(x),
        (x, 0, 1.5 * pi),
        backend=B,
        rendering_kw=rendering_kw,
        use_latex=False,
        **options()
    )


def make_plot3d_parametric_line_1(B, rendering_kw, show=False):
    opts = options()
    opts["show"] = show
    return plot3d_parametric_line(
        cos(x), sin(x), x,
        (x, -pi, pi),
        backend=B,
        rendering_kw=rendering_kw,
        use_latex=False,
        **opts
    )


def make_plot3d_1(B, rendering_kw, show=False):
    opts = options()
    opts["show"] = show
    return plot3d(
        cos(x**2 + y**2),
        (x, -3, 3), (y, -3, 3),
        use_cm=False,
        backend=B,
        rendering_kw=rendering_kw,
        **opts
    )


def make_plot3d_2(B, show=False):
    opts = options()
    opts["show"] = show
    return plot3d(
        cos(x**2 + y**2),
        sin(x**2 + y**2),
        (x, -3, 3), (y, -3, 3),
        use_cm=True,
        backend=B,
        use_latex=False,
        **opts
    )


def make_plot3d_wireframe_1(B, wf=True):
    return plot3d(
        cos(x**2 + y**2), (x, -2, 2), (y, -3, 3),
        use_cm=True,
        backend=B,
        wireframe=wf,
        **options()
    )


def make_plot3d_wireframe_2(B, rk):
    return plot3d(
        cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        use_cm=True,
        backend=B,
        wireframe=True,
        wf_n1=20,
        wf_n2=30,
        wf_rendering_kw=rk,
        wf_npoints=12,
        **options()
    )


def make_plot3d_wireframe_3(B, wf):
    r, theta = symbols("r, theta")
    return plot3d(
        cos(r**2) * exp(-r / 3), (r, 0, 3.25), (theta, 0, 2 * pi), "r",
        backend=B,
        is_polar=True,
        use_cm=True,
        legend=True,
        color_func=lambda x, y, z: (x**2 + y**2) ** 0.5,
        wireframe=True,
        wf_n1=20,
        wf_n2=40,
        wf_rendering_kw=wf,
        **options()
    )


def make_plot3d_wireframe_4(B, wf=True):
    return plot3d(
        lambda x, y: np.cos(x**2 + y**2),
        ("x", -2, 2), ("y", -3, 3),
        use_cm=True,
        backend=B,
        wireframe=wf,
        **options()
    )


def make_plot3d_wireframe_5(B, wf):
    return plot3d(
        lambda r, theta: np.cos(r**2) * np.exp(-r / 3),
        ("r", 0, 3.25), ("theta", 0, 2 * np.pi), "r",
        backend=B,
        is_polar=True,
        use_cm=True,
        legend=True,
        color_func=lambda x, y, z: (x**2 + y**2) ** 0.5,
        wireframe=True,
        wf_n1=20,
        wf_n2=40,
        wf_rendering_kw=wf,
        **options()
    )


def make_plot3d_parametric_surface_wireframe_1(B, wf):
    u, v = symbols("u, v")
    x = (1 + v / 2 * cos(u / 2)) * cos(u)
    y = (1 + v / 2 * cos(u / 2)) * sin(u)
    z = v / 2 * sin(u / 2)
    opts = options()
    opts.pop("adaptive")
    return plot3d_parametric_surface(
        x, y, z,
        (u, 0, 2 * pi), (v, -1, 1),
        backend=B,
        use_cm=True,
        wireframe=True,
        wf_n1=5,
        wf_n2=6,
        wf_rendering_kw=wf,
        **opts
    )


def make_plot3d_parametric_surface_wireframe_2(B, wf):
    x = lambda u, v: v * np.cos(u)
    y = lambda u, v: v * np.sin(u)
    z = lambda u, v: np.sin(4 * u)
    opts = options()
    opts.pop("adaptive")
    return plot3d_parametric_surface(
        x, y, z,
        ("u", 0, 2 * np.pi), ("v", -1, 0),
        backend=B,
        use_cm=True,
        wireframe=wf,
        wf_n1=5,
        wf_n2=6,
        **opts
    )


def make_plot_contour_1(B, rendering_kw):
    return plot_contour(
        cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        backend=B,
        rendering_kw=rendering_kw,
        use_latex=False,
        **options()
    )


def make_plot_contour_is_filled(B, is_filled):
    return plot_contour(
        cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        backend=B,
        use_latex=False,
        is_filled=is_filled,
        **options()
    )


def make_plot_vector_2d_quiver(B, contour_kw, quiver_kw):
    return plot_vector(
        Matrix([x, y]),
        (x, -5, 5), (y, -4, 4),
        backend=B,
        quiver_kw=quiver_kw,
        contour_kw=contour_kw,
        use_latex=False,
        **options()
    )


def make_plot_vector_2d_streamlines_1(B, stream_kw, contour_kw):
    return plot_vector(
        Matrix([x, y]),
        (x, -5, 5), (y, -4, 4),
        backend=B,
        stream_kw=stream_kw,
        contour_kw=contour_kw,
        scalar=(x + y),
        streamlines=True,
        use_latex=False,
        **options()
    )


def make_plot_vector_2d_streamlines_2(B, stream_kw, contour_kw):
    return plot_vector(
        Matrix([x, y]),
        (x, -5, 5), (y, -4, 4),
        backend=B,
        stream_kw=stream_kw,
        contour_kw=contour_kw,
        scalar=[(x + y), "test"],
        streamlines=True,
        use_latex=False,
        **options()
    )


def make_plot_vector_3d_quiver(B, quiver_kw, show=False, **kwargs):
    opts = options()
    opts["show"] = show
    opts.pop("adaptive")
    return plot_vector(
        Matrix([z, y, x]),
        (x, -5, 5), (y, -4, 4), (z, -3, 3),
        backend=B,
        quiver_kw=quiver_kw,
        use_latex=False,
        **opts,
        **kwargs
    )


def make_plot_vector_3d_streamlines_1(B, stream_kw, show=False, kwargs=dict()):
    opts = options()
    opts["show"] = show
    opts.pop("adaptive")
    return plot_vector(
        Matrix([z, y, x]),
        (x, -5, 5), (y, -4, 4), (z, -3, 3),
        backend=B,
        stream_kw=stream_kw,
        streamlines=True,
        use_latex=False,
        **opts,
        **kwargs
    )


def make_plot_vector_2d_normalize_1(B, norm):
    opts = options()
    opts.pop("adaptive")
    return plot_vector(
        [-sin(y), cos(x)],
        (x, -2, 2), (y, -2, 2),
        backend=B,
        normalize=norm,
        scalar=False,
        use_cm=False,
        **opts
    )


def make_plot_vector_2d_normalize_2(B, norm):
    opts = options()
    opts.pop("adaptive")
    return plot_vector(
        [-u * sin(y), cos(x)],
        (x, -2, 2), (y, -2, 2),
        backend=B,
        normalize=norm,
        scalar=False,
        use_cm=False,
        params={u: (1, 0, 2)},
        **options()
    )


def make_plot_vector_3d_normalize_1(B, norm):
    opts = options()
    opts.pop("adaptive")
    return plot_vector(
        [z, -x, y],
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        backend=B,
        normalize=norm,
        use_cm=False,
        **opts
    )


def make_plot_vector_3d_normalize_2(B, norm):
    opts = options()
    opts.pop("adaptive")
    return plot_vector(
        [u * z, -x, y],
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        backend=B,
        normalize=norm,
        use_cm=False,
        params={u: (1, 0, 2)},
        **opts
    )


def make_plot_vector_2d_quiver_color_func_1(B, cf):
    opts = options()
    opts.pop("adaptive")
    return plot_vector(
        (-y, x), (x, -2, 2), (y, -2, 2),
        scalar=False,
        use_cm=True,
        color_func=cf,
        backend=B,
        **opts
    )


def make_plot_vector_3d_quiver_color_func_1(B, cf):
    opts = options()
    opts.pop("adaptive")
    return plot_vector(
        Matrix([z, y, x]),
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        backend=B,
        color_func=cf,
        **opts
    )


def make_plot_vector_3d_quiver_color_func_2(B, cf):
    opts = options()
    opts.pop("adaptive")
    return plot_vector(
        Matrix([a * z, a * y, a * x]),
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        backend=B,
        color_func=cf,
        params={a: (1, 0, 2)},
        **opts
    )


def make_plot_vector_3d_streamlines_color_func(B, cf):
    # NOTE: better keep a decent number of discretization points in order to
    # be sure to have streamlines
    return plot_vector(
        Matrix([z, y, x]),
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        streamlines=True,
        show=False,
        backend=B,
        color_func=cf,
        n=7,
    )


def make_test_plot_implicit_adaptive_true(B, rendering_kw):
    return plot_implicit(
        x > y, (x, -5, 5), (y, -4, 4),
        backend=B,
        adaptive=True,
        rendering_kw=rendering_kw,
        use_latex=False,
        show=False,
    )


def make_test_plot_implicit_adaptive_false(B, rendering_kw):
    return plot_implicit(
        x > y, (x, -5, 5), (y, -4, 4),
        n=5,
        backend=B,
        adaptive=False,
        show=False,
        rendering_kw=rendering_kw,
        use_latex=False,
    )


def make_test_real_imag(B, rendering_kw):
    return plot_real_imag(
        sqrt(x), (x, -5, 5),
        backend=B,
        rendering_kw=rendering_kw,
        use_latex=False,
        **options()
    )


def make_test_plot_complex_1d(B, rendering_kw):
    return plot_complex(
        sqrt(x), (x, -5, 5),
        backend=B, rendering_kw=rendering_kw,
        **options()
    )


def make_test_plot_complex_2d(B, rendering_kw):
    opts = options()
    opts.pop("adaptive")
    return plot_complex(
        sqrt(x), (x, -5 - 5 * I, 5 + 5 * I),
        backend=B,
        coloring="a",
        rendering_kw=rendering_kw,
        **opts
    )


def make_test_plot_complex_3d(B, rendering_kw):
    opts = options()
    opts.pop("adaptive")
    return plot_complex(
        sqrt(x), (x, -5 - 5 * I, 5 + 5 * I),
        backend=B,
        rendering_kw=rendering_kw,
        threed=True,
        use_cm=False,
        **opts
    )


def make_test_plot_list_is_filled_false(B):
    return plot_list(
        [1, 2, 3],
        [1, 2, 3],
        backend=B,
        is_point=True,
        is_filled=False,
        show=False,
        use_latex=False,
    )


def make_test_plot_list_is_filled_true(B):
    return plot_list(
        [1, 2, 3],
        [1, 2, 3],
        backend=B,
        is_point=True,
        is_filled=True,
        show=False,
        use_latex=False,
    )


def make_test_plot_list_color_func(B):
    return plot_list(
        [1, 2, 3],
        [1, 2, 3],
        backend=B,
        color_func=lambda x, y: np.arange(len(x)),
        use_cm=True,
        show=False,
        use_latex=False,
        is_point=True,
    )


def make_test_plot_piecewise_single_series(B):
    return plot_piecewise(
        Heaviside(x, 0).rewrite(Piecewise),
        (x, -10, 10),
        backend=B,
        use_latex=False,
        **options()
    )


def make_test_plot_piecewise_multiple_series(B):
    return plot_piecewise(
        (Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10)),
        (
            Piecewise(
                (sin(x), x < 0),
                (2, Eq(x, 0)),
                (cos(x), x > 0)),
            (x, -6, 4)
        ),
        backend=B,
        use_latex=False,
        **options()
    )


def make_test_plot_geometry_1(B):
    return plot_geometry(
        Line((1, 2), (5, 4)),
        Circle((0, 0), 4),
        Polygon((2, 2), 3, n=6),
        backend=B,
        show=False,
        is_filled=False,
        use_latex=False,
    )


def make_test_plot_geometry_2(B, is_filled):
    return plot_geometry(
        Circle(Point2D(0, 0), 5),
        Ellipse(Point2D(-3, 2), hradius=3, eccentricity=Rational(4, 5)),
        Polygon((4, 0), 4, n=5),
        Curve((cos(x), sin(x)), (x, 0, 2 * pi)),
        Segment((-4, -6), (6, 6)),
        Point2D(0, 0),
        is_filled=is_filled,
        backend=B,
        show=False,
        use_latex=False,
    )


def make_test_plot_geometry_3d(B):
    return plot_geometry(
        (Point3D(5, 5, 5), "center"),
        (Line3D(Point3D(-2, -3, -4), Point3D(2, 3, 4)), "line"),
        (Plane((0, 0, 0), (1, 1, 1)), (x, -5, 5), (y, -4, 4), (z, -10, 10)),
        show=False,
        backend=B,
    )


def make_test_vectors_3d_update_interactive(B):
    return plot_vector(
        [a * z, b * y, c * x],
        (x, -5, 5), (y, -5, 5), (z, -5, 5),
        params={a: (1, 0, 2), b: (1, 0, 2), c: (1, 0, 2)},
        streamlines=True,
        n=5,
        backend=B,
        show=False,
    )


def make_test_aspect_ratio_2d_issue_11764(B, aspect="auto"):
    return plot_parametric(
        cos(x), sin(x), (x, 0, 2 * pi), backend=B, aspect=aspect, **options()
    )


def make_test_aspect_ratio_3d(B, aspect="auto", show=False):
    opts = options()
    opts["show"] = show
    opts["n"] = 21
    return plot3d(
        cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        backend=B, aspect=aspect,
        **opts
    )


def make_test_plot_size(B, size=None):
    return plot(sin(x), backend=B, size=size, **options())


def make_test_plot_scale_lin_log(B, xscale, yscale):
    return plot(log(x), backend=B, xscale=xscale, yscale=yscale, **options())


def make_test_backend_latex_labels_1(B, use_latex):
    x1 = symbols("x_1^2")
    return plot(
        cos(x1), (x1, -1, 1),
        backend=B, use_latex=use_latex,
        **options()
    )


def make_test_backend_latex_labels_2(B, use_latex, show=False):
    x1, x2 = symbols("x_1^2, x_2")
    opts = options()
    opts["show"] = show
    return plot3d(
        cos(x1**2 + x2**2), (x1, -1, 1), (x2, -1, 1),
        backend=B,
        use_latex=use_latex,
        **opts
    )


def make_test_plot_use_latex(B):
    return plot(
        sin(x), cos(x),
        backend=B, legend=True, use_latex=True,
        **options()
    )


def make_test_plot_parametric_use_latex(B):
    return plot_parametric(
        cos(x), sin(x), (x, 0, 1.5 * pi),
        backend=B, use_latex=True,
        **options()
    )


def make_test_plot_contour_use_latex(B):
    return plot_contour(
        cos(x**2 + y**2),
        (x, -3, 3), (y, -3, 3),
        backend=B,
        use_latex=True,
        **options()
    )


def make_test_plot3d_parametric_line_use_latex(B, show=False):
    opts = options()
    opts["show"] = show
    return plot3d_parametric_line(
        cos(x), sin(x), x, (x, -pi, pi), backend=B, use_latex=True, **opts
    )


def make_test_plot3d_use_latex(B, show=False):
    opts = options()
    opts["show"] = show
    return plot3d(
        cos(x**2 + y**2),
        sin(x**2 + y**2),
        (x, -3, 3), (y, -3, 3),
        use_cm=True,
        backend=B,
        use_latex=True,
        **opts
    )


def make_test_plot_vector_2d_quivers_use_latex(B):
    return plot_vector(
        Matrix([x, y]), (x, -5, 5), (y, -4, 4),
        backend=B, **options()
    )


def make_test_plot_vector_2d_streamlines_custom_scalar_field_use_latex(B):
    return plot_vector(
        Matrix([x, y]),
        (x, -5, 5), (y, -4, 4),
        backend=B,
        scalar=(x + y),
        streamlines=True,
        use_latex=True,
        **options()
    )


def make_test_plot_vector_2d_streamlines_custom_scalar_field_custom_label_use_latex(B):
    return plot_vector(
        Matrix([x, y]),
        (x, -5, 5), (y, -4, 4),
        backend=B,
        scalar=[(x + y), "test"],
        streamlines=True,
        use_latex=True,
        **options()
    )


def make_test_plot_vector_2d_use_latex_colorbar(B, scalar, streamlines):
    opts = options()
    opts.pop("adaptive")
    return plot_vector(
        Matrix([x, y]),
        (x, -5, 5), (y, -4, 4),
        backend=B,
        scalar=scalar,
        streamlines=streamlines,
        use_cm=True,
        use_latex=True,
        **opts
    )


def make_test_plot_vector_3d_quivers_use_latex(B, show=False):
    opts = options()
    opts["show"] = show
    opts.pop("adaptive")
    return plot_vector(
        Matrix([z, y, x]),
        (x, -5, 5), (y, -4, 4), (z, -3, 3),
        backend=B,
        use_cm=True,
        use_latex=True,
        **opts
    )


def make_test_plot_vector_3d_streamlines_use_latex(B, show=False):
    opts = options()
    opts["show"] = show
    opts.pop("adaptive")
    return plot_vector(
        Matrix([z, y, x]),
        (x, -5, 5), (y, -4, 4), (z, -3, 3),
        backend=B,
        streamlines=True,
        use_latex=True,
        **opts
    )


def make_test_plot_complex_use_latex_1(B):
    return plot_complex(
        cos(x) + sin(I * x), (x, -2, 2),
        use_latex=True, backend=B,
        **options()
    )


def make_test_plot_complex_use_latex_2(B):
    opts = options()
    opts.pop("adaptive")
    return plot_complex(
        gamma(z), (z, -3 - 3 * I, 3 + 3 * I),
        use_latex=True, backend=B,
        **opts
    )


def make_test_plot_real_imag_use_latex(B):
    return plot_real_imag(
        sqrt(x), (x, -3, 3),
        backend=B, use_latex=True,
        **options()
    )


def make_test_plot3d_use_cm(B, use_cm, show=False):
    opts = options()
    opts["show"] = show
    return plot3d(
        cos(x**2 + y**2), (x, -1, 1), (y, -1, 1),
        backend=B, use_cm=use_cm, **opts
    )


def make_test_plot_polar(B, pa=False):
    return plot_polar(
        1 + sin(10 * x) / 10, (x, 0, 2 * pi),
        backend=B,
        polar_axis=pa,
        aspect="equal",
        **options()
    )


def make_test_plot_polar_use_cm(B, pa=False, ucm=False, cf=None):
    return plot_polar(
        1 + sin(10 * x) / 10, (x, 0, 2 * pi),
        backend=B,
        polar_axis=pa,
        aspect="equal",
        use_cm=ucm,
        color_func=cf,
        **options()
    )


def make_test_plot3d_implicit(B, show=False):
    opts = options()
    opts["show"] = show
    opts.pop("adaptive")
    return plot3d_implicit(
        x**2 + y**3 - z**2,
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        backend=B, **opts
    )


def make_test_surface_color_func_1(B, col, show=False):
    opts = options()
    opts["show"] = show
    return plot3d(
        cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        backend=B,
        color_func=col,
        use_cm=True,
        **opts
    )


def make_test_surface_color_func_2(B, col, show=False):
    opts = options()
    opts["show"] = show
    opts.pop("adaptive")
    r = 2 + sin(7 * u + 5 * v)
    expr = (r * cos(u) * sin(v), r * sin(u) * sin(v), r * cos(v))
    return plot3d_parametric_surface(
        *expr, (u, 0, 2 * pi), (v, 0, pi),
        use_cm=True,
        backend=B,
        color_func=col,
        **opts
    )


def make_test_surface_interactive_color_func(B):
    expr1 = t * cos(x**2 + y**2)
    r = 2 + sin(7 * u + 5 * v)
    expr2 = (t * r * cos(u) * sin(v), r * sin(u) * sin(v), r * cos(v))

    s1 = SurfaceOver2DRangeSeries(
        expr1, (x, -5, 5), (y, -5, 5),
        n1=5, n2=5,
        params={t: 1},
        use_cm=True,
        color_func=lambda x, y, z: z,
    )
    s2 = SurfaceOver2DRangeSeries(
        expr1, (x, -5, 5), (y, -5, 5),
        n1=5, n2=5,
        params={t: 1},
        use_cm=True,
        color_func=lambda x, y, z: np.sqrt(x**2 + y**2),
    )
    s3 = ParametricSurfaceSeries(
        *expr2, (u, -5, 5), (v, -5, 5),
        n1=5, n2=5,
        params={t: 1},
        use_cm=True,
        color_func=lambda x, y, z, u, v: z
    )
    s4 = ParametricSurfaceSeries(
        *expr2, (u, -5, 5), (v, -5, 5),
        n1=5, n2=5,
        params={t: 1},
        use_cm=True,
        color_func=lambda x, y, z, u, v: np.sqrt(x**2 + y**2)
    )
    return B(s1, s2, s3, s4)


def make_test_line_color_func(B, col):
    return plot(
        cos(x), (x, -3, 3),
        backend=B, color_func=col, legend=True,
        **options()
    )


def make_test_line_interactive_color_func(B):
    expr = t * cos(x * t)
    s1 = LineOver1DRangeSeries(
        expr, (x, -3, 3), n=5, params={t: 1}, color_func=None
    )
    s2 = LineOver1DRangeSeries(
        expr, (x, -3, 3), n=5, params={t: 1},
        color_func=lambda x, y: np.cos(x)
    )
    return B(s1, s2)


def make_test_line_color_plot(B, lc):
    return plot(sin(x), line_color=lc, backend=B, **options())


def make_test_line_color_plot3d_parametric_line(B, lc, use_cm, show=False):
    opts = options()
    opts["show"] = show
    return plot3d_parametric_line(
        cos(x), sin(x), x, (x, 0, 2 * pi),
        line_color=lc,
        backend=B,
        use_cm=use_cm,
        **opts
    )


def make_test_surface_color_plot3d(B, sc, use_cm, show=False):
    opts = options()
    opts["show"] = show
    return plot3d(
        cos(x**2 + y**2), (x, 0, 2), (y, 0, 2),
        surface_color=sc,
        backend=B,
        use_cm=use_cm,
        legend=True,
        **opts
    )


def make_test_plot3d_list_use_cm_False(B, is_point, is_filled=False):
    x = [0, 1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1, 0]
    z = [1, 3, 2, 4, 6, 5]

    return plot3d_list(
        x, y, z,
        backend=B,
        is_point=is_point,
        is_filled=is_filled,
        use_cm=False,
        show=False,
    )


def make_test_plot3d_list_use_cm_color_func(
    B, is_point, is_filled=False, cf=None
):
    x = [0, 1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1, 0]
    z = [1, 3, 2, 4, 6, 5]

    return plot3d_list(
        (x, y, z),
        (z, y, x),
        backend=B,
        is_point=is_point,
        is_filled=is_filled,
        use_cm=True,
        show=False,
        color_func=cf,
    )


def make_test_plot3d_list_interactive(B):
    z1 = np.linspace(0, 6 * np.pi, 10)
    x1 = z1 * np.cos(z1)
    y1 = z1 * np.sin(z1)

    p1 = plot3d_list(x1, y1, z1, show=False, backend=B, is_point=False)
    p2 = plot3d_list(
        [t * cos(t)], [t * sin(t)], [t],
        params={t: (0, 0, 6 * pi)},
        backend=B,
        show=False,
        is_point=True,
    )
    return p2 + p1


def make_test_contour_show_clabels_1(B, clabels):
    return plot_contour(
        cos(x * y), (x, -2, 2), (y, -2, 2),
        backend=B,
        is_filled=False,
        clabels=clabels,
        **options()
    )


def make_test_contour_show_clabels_2(B, clabels):
    return plot_contour(
        cos(u * x * y), (x, -2, 2), (y, -2, 2),
        params={u: (1, 0, 1)},
        backend=B,
        is_filled=False,
        clabels=clabels,
        **options()
    )


def make_test_color_func_expr_1(B, streamlines=False):
    return plot_vector(
        [cos(u * y), -sin(x)],
        (x, -5, 5), (y, -5, 5),
        streamlines=streamlines,
        use_cm=True,
        color_func=cos(x**2 + y**2),
        backend=B,
        params={u: (1, 0, 1)},
        scalar=False,
        **options()
    )


def make_test_color_func_expr_2(B):
    return plot_vector(
        Matrix([u * z, -x, y]),
        (x, -3, 3), (y, -3, 3), (z, -3, 3),
        backend=B,
        use_cm=True,
        color_func=sqrt(x**2 + y**2),
        params={u: (1, 0, 1)},
        **options()
    )


def make_test_domain_coloring_2d(B, at_infinity):
    return plot_complex(
        (z - 1) / (z**2 + z + 1),
        (z, -3 - 3 * I, 3 + 3 * I),
        at_infinity=at_infinity,
        n=10, backend=B, show=False,
    )


def make_test_show_in_legend_3d(B):
    options = dict(backend=B, use_cm=False, show=False, adaptive=False, n=5)

    p1 = plot3d_parametric_line(
        cos(x), sin(x), x, (x, 0, 2 * pi), "a", **options
    )
    p2 = plot3d_parametric_line(
        cos(x) / 2, sin(x) / 2, x, (x, 0, 2 * pi), "b",
        show_in_legend=False, **options
    )
    p3 = plot3d_parametric_line(
        cos(x) / 3, sin(x) / 3, x, (x, 0, 2 * pi), "c", **options
    )
    p4 = p1 + p2 + p3

    p13 = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), "a", **options)
    p14 = plot3d(
        sin(x**2 + y**2),
        (x, -pi, pi),
        (y, -pi, pi),
        "b",
        show_in_legend=False,
        **options
    )
    p15 = plot3d(cos(x * y), (x, -pi, pi), (y, -pi, pi), "c", **options)
    p16 = p13 + p14 + p15

    return p4, p16


def make_test_show_in_legend_2d(B):
    options = dict(backend=B, use_cm=False, show=False, adaptive=False, n=5)

    p5 = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi), "a", **options)
    p6 = plot_parametric(
        cos(x) / 2, sin(x) / 2, (x, 0, 2 * pi), "b",
        show_in_legend=False, **options
    )
    p7 = plot_parametric(
        cos(x) / 3, sin(x) / 3, (x, 0, 2 * pi), "c",
        **options
    )
    p8 = p5 + p6 + p7

    p9 = plot(cos(x), **options)
    p10 = plot(sin(x), show_in_legend=False, **options)
    p11 = plot(sin(x) * cos(x), **options)
    p12 = p9 + p10 + p11

    return p8, p12


def make_test_legend_plot_sum_1(B, l):
    options = dict(show=False, backend=B, adaptive=False, n=5)
    p1 = plot(cos(x), **options, legend=l)
    p2 = plot(sin(x), **options)
    p3 = plot(cos(x) * sin(x), **options)
    return p1 + p2 + p3


def make_test_legend_plot_sum_2(B, l):
    options = dict(show=False, backend=B, adaptive=False, n=5)
    p1 = plot(cos(x), **options)
    p2 = plot(sin(x), **options, legend=l)
    p3 = plot(cos(x) * sin(x), **options)
    return p1 + p2 + p3


def make_test_analytic_landscape(B):
    expr = z**5 + Rational(1, 10)
    return plot_riemann_sphere(
        expr,
        threed=True,
        n1=10, n2=40,
        coloring="k",
        backend=B, legend=False, show=False,
    )


def make_test_detect_poles(B, dp):
    expr = tan(x)
    return plot(expr, (x, -10, 10), backend=B, show=False, detect_poles=dp)


def make_test_detect_poles_interactive(B, dp):
    expr = tan(x - y)
    return plot(
        expr, (x, -10, 10),
        backend=B, show=False,
        params={y: (0, -1, 1)},
        detect_poles=dp,
    )


def make_test_plot_riemann_sphere(B, annotate):
    expr = (z - 1) / (z**2 + z + 2)
    return plot_riemann_sphere(
        expr,
        coloring="b", annotate=annotate, riemann_mask=True,
        n=10, show=False, backend=B,
        imodule="panel",
    )


def make_test_parametric_texts_2d(B):
    expr = sin(y * x + z)
    return (
        x,
        y,
        plot(
            expr,
            n=10, backend=B, show=False,
            params={y: (1, 0, 2), z: (0, -pi, pi)},
            title=("y={}, z={:.3f}", y, z),
            xlabel=("test y+z={:.2f}", y + z),
            ylabel=("test z={:.2f}", z),
        ),
    )


def make_test_parametric_texts_3d(B):
    expr = sin(a * y * x + b)
    return (
        a,
        b,
        plot3d(
            expr, (x, -pi, pi), (y, -pi, pi),
            n=5, backend=B, show=False,
            params={a: (1, 0, 2), b: (0, -pi, pi)},
            title=("a={}, a+b={:.3f}", a, b + a),
            xlabel=("test a={:.2f}", a),
            ylabel=("test b={:.2f}", b),
            zlabel=("test a={:.2f}, b={:.2f}", a, b),
        ),
    )


def make_test_arrow_2d(B, lbl, rkw, sil):
    return graphics(
        arrow_2d(
            (1, 2), (3, 4), label=lbl, rendering_kw=rkw, show_in_legend=sil),
        show=False, backend=B, legend=True
    )


def make_test_arrow_3d(B, lbl, rkw, sil):
    return graphics(
        arrow_3d(
            (1, 2, 3), (4, 5, 6), label=lbl, rendering_kw=rkw, show_in_legend=sil),
        show=False, backend=B, legend=True
    )


def make_test_root_locus_1(B, sgrid, zgrid):
    s = symbols("s")
    G = (s**2 + 1) / (s**3 + 2*s**2 + 3*s + 4)
    return plot_root_locus(G, backend=B, show=False,
        sgrid=sgrid, zgrid=zgrid)


def make_test_root_locus_2(B):
    s = symbols("s")
    G1 = (s**2 + 1) / (s**3 + 2*s**2 + 3*s + 4)
    G2 = (s**2 - 4) / (s**3 + 2*s - 3)
    return plot_root_locus((G1, "a"), (G2, "b"), backend=B, show=False)


def make_test_poles_zeros_sgrid(B):
    s = symbols("s")
    G = (s**2 + 1) / (s**4 + 4*s**3 + 6*s**2 + 5*s + 2)
    return plot_pole_zero(G, sgrid=True, show=False, backend=B)


def make_test_ngrid(B, cl_mags, cl_phases, label_cl_phases):
    return graphics(
        ngrid(cl_mags, cl_phases, label_cl_phases),
        grid=False, show=False, backend=B
    )


def make_test_sgrid(B, xi, wn, tp, ts, auto, show_control_axis, **kwargs):
    return graphics(
        sgrid(xi, wn, tp, ts, auto=auto,
            show_control_axis=show_control_axis, **kwargs),
        grid=False, show=False, backend=B
    )


def make_test_zgrid(B, xi, wn, tp, ts, show_control_axis, **kwargs):
    return graphics(
        zgrid(xi, wn, tp, ts,
            show_control_axis=show_control_axis, **kwargs),
        grid=False, show=False, backend=B
    )


def make_test_mcircles(B, mag):
    return graphics(
        mcircles(mag), backend=B, show=False, grid=False
    )
