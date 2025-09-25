from spb import (
    plot, plot_parametric, plot_polar, plot_list, plot_implicit,
    plot_contour, plot_piecewise, plot_geometry,
    plot3d_parametric_line, plot3d, plot3d_list,
    plot3d_implicit, plot3d_parametric_surface,
    plot_vector, plot_complex, plot_real_imag, plot_riemann_sphere,
    graphics, arrow_2d, arrow_3d, plot_root_locus, plot_pole_zero,
    ngrid, sgrid, zgrid, mcircles, surface, surface_parametric, line,
    root_locus, contour, list_2d, line_parametric_2d
)
from spb.series import (
    SurfaceOver2DRangeSeries, ParametricSurfaceSeries, LineOver1DRangeSeries,
    HVLineSeries
)
from sympy import (
    symbols, sin, cos, pi, exp, Matrix, sqrt, Heaviside, Piecewise, Eq, I,
    Circle, Line, Polygon, Ellipse, Curve, Point2D, Segment, Rational,
    Point3D, Line3D, Plane, log, gamma, tan
)
from sympy.abc import a, b, c, x, y, z, u, v, s, t
import numpy as np


def options():
    return dict(n=5, show=False)


def custom_colorloop_1(B):
    opts = options()
    return plot(
        sin(x),
        cos(x),
        sin(x / 2),
        cos(x / 2),
        2 * sin(x),
        2 * cos(x),
        backend=B,
        **opts
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


def make_test_plot(B, rendering_kw, use_latex=False):
    opts = options()
    return plot(
        sin(a * x),
        cos(b * x),
        rendering_kw=rendering_kw,
        backend=B,
        legend=True,
        use_latex=use_latex,
        params={a: (1, 0, 2), b: (1, 0, 2)},
        **opts
    )


def make_test_plot_parametric(B, use_cm, rendering_kw={}):
    opts = options()
    return plot_parametric(
        cos(a * x), sin(b * x),
        (x, 0, 1.5 * pi),
        use_cm=use_cm,
        backend=B,
        rendering_kw=rendering_kw,
        use_latex=False,
        params={a: (1, 0, 2), b: (1, 0, 2)},
        **opts
    )


def make_test_plot3d_parametric_line(B, rendering_kw, use_latex, use_cm, show=False):
    opts = options()
    opts["show"] = show
    return plot3d_parametric_line(
        cos(a * x), sin(b * x), x,
        (x, -pi, pi),
        backend=B,
        rendering_kw=rendering_kw,
        use_latex=use_latex,
        use_cm=use_cm,
        params={a: (1, 0, 2), b: (1, 0, 2)},
        **opts
    )


def make_test_plot3d(B, rendering_kw, use_cm, use_latex, show=False):
    opts = options()
    opts["show"] = show
    return plot3d(
        cos(a * x**2 + y**2),
        sin(b * x**2 + y**2),
        (x, -3, 3), (y, -3, 3),
        use_latex=use_latex,
        use_cm=use_cm,
        backend=B,
        rendering_kw=rendering_kw,
        params={a: (1, 0, 2), b: (1, 0, 2)},
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
    return plot3d_parametric_surface(
        x, y, z,
        (u, 0, 2 * pi), (v, -1, 1),
        backend=B,
        use_cm=True,
        wireframe=True,
        wf_n1=5,
        wf_n2=6,
        wf_rendering_kw=wf,
        **options()
    )


def make_plot3d_parametric_surface_wireframe_2(B, wf):
    x = lambda u, v: v * np.cos(u)
    y = lambda u, v: v * np.sin(u)
    z = lambda u, v: np.sin(4 * u)
    return plot3d_parametric_surface(
        x, y, z,
        ("u", 0, 2 * np.pi), ("v", -1, 0),
        backend=B,
        use_cm=True,
        wireframe=wf,
        wf_n1=5,
        wf_n2=6,
        **options()
    )


def make_test_plot_contour(B, rendering_kw, use_latex):
    return plot_contour(
        cos(a * x**2 + y**2), (x, -3, 3), (y, -3, 3),
        backend=B,
        rendering_kw=rendering_kw,
        use_latex=use_latex,
        params={a: (1, 0, 2)},
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


def make_test_plot_vector_2d_quiver(B, contour_kw, quiver_kw):
    return plot_vector(
        Matrix([a * x, y]),
        (x, -5, 5), (y, -4, 4),
        backend=B,
        quiver_kw=quiver_kw,
        contour_kw=contour_kw,
        use_latex=False,
        params={a: (1, 0, 2)},
        **options()
    )


def make_test_plot_vector_2d_streamlines(
    B, stream_kw, contour_kw, scalar, use_latex=False
):
    return plot_vector(
        Matrix([a * x, y]),
        (x, -5, 5), (y, -4, 4),
        backend=B,
        stream_kw=stream_kw,
        contour_kw=contour_kw,
        scalar=scalar,
        streamlines=True,
        use_latex=use_latex,
        params={a: (1, 0, 2)},
        **options()
    )


def make_test_plot_vector_3d_quiver_streamlines(
    B, streamlines, quiver_kw={}, stream_kw={}, show=False, **kwargs
):
    opts = options()
    opts["show"] = show
    return plot_vector(
        Matrix([a * z, y, x]),
        (x, -5, 5), (y, -4, 4), (z, -3, 3),
        backend=B,
        quiver_kw=quiver_kw,
        stream_kw=stream_kw,
        streamlines=streamlines,
        params={a: (1, 0, 2)},
        **opts,
        **kwargs
    )


def make_test_plot_vector_2d_normalize(B, norm):
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


def make_test_plot_vector_3d_normalize(B, norm):
    return plot_vector(
        [u * z, -x, y],
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        backend=B,
        normalize=norm,
        use_cm=False,
        params={u: (1, 0, 2)},
        **options()
    )


def make_test_plot_vector_2d_color_func(B, streamlines, cf):
    return plot_vector(
        (-a * y, x), (x, -2, 2), (y, -2, 2),
        scalar=False,
        streamlines=streamlines,
        use_cm=True,
        color_func=cf,
        show=False,
        backend=B,
        n=3,
        params={a: (1, 0, 2)},
    )


def make_test_plot_vector_3d_quiver_color_func(B, cf):
    return plot_vector(
        Matrix([a * z, a * y, a * x]),
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        backend=B,
        color_func=cf,
        params={a: (1, 0, 2)},
        **options()
    )


def make_test_plot_vector_3d_streamlines_color_func(B, cf):
    # NOTE: better keep a decent number of discretization points in order to
    # be sure to have streamlines
    return plot_vector(
        Matrix([a*z, y, x]),
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        streamlines=True,
        show=False,
        backend=B,
        color_func=cf,
        params={a: (1, 0, 2)},
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


def make_test_real_imag(B, rendering_kw, use_latex):
    return plot_real_imag(
        sqrt(x), (x, -5, 5),
        backend=B,
        rendering_kw=rendering_kw,
        use_latex=use_latex,
        **options()
    )


def make_test_plot_complex_1d(B, rendering_kw, use_latex):
    return plot_complex(
        sqrt(x), (x, -5, 5),
        backend=B, rendering_kw=rendering_kw, use_latex=use_latex,
        **options()
    )


def make_test_plot_complex_2d(B, rendering_kw, use_latex=False):
    return plot_complex(
        sqrt(x), (x, -5 - 5 * I, 5 + 5 * I),
        backend=B,
        coloring="a",
        rendering_kw=rendering_kw,
        use_latex=use_latex,
        **options()
    )


def make_test_plot_complex_3d(B, rendering_kw):
    return plot_complex(
        sqrt(x), (x, -5 - 5 * I, 5 + 5 * I),
        backend=B,
        rendering_kw=rendering_kw,
        threed=True,
        use_cm=False,
        **options()
    )


def make_test_plot_list_is_filled(B, is_filled):
    return plot_list(
        [1, 2, 3],
        [1, 2, 3],
        backend=B,
        is_scatter=True,
        is_filled=is_filled,
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
        is_scatter=True,
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
    return plot3d_implicit(
        x**2 + y**3 - z**2,
        (x, -2, 2), (y, -2, 2), (z, -2, 2),
        backend=B, **opts
    )


def make_test_surface_color_func(B):
    expr1 = t * cos(x**2 + y**2)
    r = 2 + sin(7 * u + 5 * v)
    expr2 = (t * r * cos(u) * sin(v), r * sin(u) * sin(v), r * cos(v))
    params = {t: (1, 0, 2)}

    return graphics(
        surface(expr1, (x, -5, 5), (y, -5, 5),
            n1=5, n2=5,
            params=params,
            use_cm=True,
            color_func=lambda x, y, z: z),
        surface(expr1, (x, -5, 5), (y, -5, 5),
            n1=5, n2=5,
            params=params,
            use_cm=True,
            color_func=lambda x, y, z: np.sqrt(x**2 + y**2)),
        surface_parametric(
            *expr2, (u, -5, 5), (v, -5, 5),
            n1=5, n2=5,
            params=params,
            use_cm=True,
            color_func=lambda x, y, z, u, v: z
        ),
        surface_parametric(
            *expr2, (u, -5, 5), (v, -5, 5),
            n1=5, n2=5,
            params=params,
            use_cm=True,
            color_func=lambda x, y, z, u, v: np.sqrt(x**2 + y**2)
        ),
        backend=B, show=False
    )


def make_test_line_color_func(B):
    expr = t * cos(x * t)
    params = {t: (1, 0, 2)}
    return graphics(
        line(expr, (x, -3, 3), n=5, params=params, color_func=None),
        line(expr, (x, -3, 3), n=5, params=params,
            color_func=lambda x, y: np.cos(x)),
        backend=B, show=False
    )


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


def make_test_plot3d_list(B, is_filled, cf):
    z1 = np.linspace(0, 6 * np.pi, 10)
    c = np.cos(z1)
    s = np.sin(z1)
    x1 = z1 * c
    y1 = z1 * s

    p1 = plot3d_list(x1, y1, z1, show=False, backend=B, is_scatter=False,
        use_cm=False)
    p2 = plot3d_list(x1, y1, z1, show=False, backend=B, is_scatter=True,
        is_filled=is_filled, use_cm=False)
    p3 = plot3d_list(
        [t * coeff1*coeff2 for coeff1, coeff2 in zip(c, z1)],
        [t * coeff1*coeff2 for coeff1, coeff2 in zip(s, z1)],
        [t * coeff for coeff in z1],
        params={t: (1, 0, 6 * pi)},
        backend=B,
        show=False,
        is_scatter=True,
        is_filled=is_filled,
        use_cm=True,
        color_func=cf,
    )
    return p3 + p2 + p1


def make_test_contour_show_clabels(B, clabels):
    return plot_contour(
        cos(a * x * y), (x, -2, 2), (y, -2, 2),
        params={a: (1, 0, 1)},
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


def make_test_plot3d_use_cm(B, use_cm, show=False):
    opts = options()
    opts["show"] = show
    return plot3d(
        cos(x**2 + y**2), (x, -1, 1), (y, -1, 1),
        backend=B, use_cm=use_cm, **opts
    )


def make_test_domain_coloring_2d(B, at_infinity):
    return plot_complex(
        (z - 1) / (z**2 + z + 1),
        (z, -3 - 3 * I, 3 + 3 * I),
        at_infinity=at_infinity,
        n=10, backend=B, show=False,
    )


def make_test_show_in_legend_3d(B):
    options = dict(backend=B, use_cm=False, show=False, n=5)

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
    options = dict(backend=B, use_cm=False, show=False, n=5)

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
    options = dict(show=False, backend=B, n=5)
    p1 = plot(cos(x), **options, legend=l)
    p2 = plot(sin(x), **options)
    p3 = plot(cos(x) * sin(x), **options)
    return p1 + p2 + p3


def make_test_legend_plot_sum_2(B, l):
    options = dict(show=False, backend=B, n=5)
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
    params = {a: (3, 0, 5), b: (4, 0, 5)}
    return graphics(
        arrow_2d(
            (1, 2), (a, b), label=lbl, rendering_kw=rkw,
            show_in_legend=sil, params=params),
        show=False, backend=B, legend=True
    )


def make_test_arrow_3d(B, lbl, rkw, sil):
    params = {a: (4, 0, 6), b: (5, 0, 6), c: (6, 0, 6)}
    return graphics(
        arrow_3d(
            (1, 2, 3), (a, b, c), label=lbl, rendering_kw=rkw,
            show_in_legend=sil, params=params),
        show=False, backend=B, legend=True
    )


def make_test_root_locus_1(B, sgrid, zgrid):
    G = (a * s**2 + 1) / (s**3 + 2*s**2 + 3*s + 4)
    return plot_root_locus(G, backend=B, show=False,
        sgrid=sgrid, zgrid=zgrid, params={a: (1, 0, 2)})


def make_test_root_locus_2(B):
    G1 = (s**2 + 1) / (s**3 + 2*s**2 + 3*s + 4)
    G2 = (s**2 - 4) / (s**3 + 2*s - 3)
    return plot_root_locus((G1, "a"), (G2, "b"), backend=B, show=False)


def make_test_root_locus_3(B, _sgrid, _zgrid):
    G = (a * s**2 + 1) / (s**3 + 2*s**2 + 3*s + 4)
    grid = []
    if _sgrid:
        grid = sgrid(auto=False)
    elif _zgrid:
        grid = zgrid()
    return graphics(
        grid,
        root_locus(G, sgrid=False, zgrid=False, params={a: (1, 0, 2)}),
        show=False, backend=B)


def make_test_plot_pole_zero(B, sgrid, zgrid, T, is_filled):
    G = (a * s**2 + 1) / (s**4 + 4*s**3 + 6*s**2 + 5*s + 2)
    return plot_pole_zero(G, show=False, backend=B,
        params={a: (1, 0, 2)}, T=T, sgrid=sgrid, zgrid=zgrid,
        is_filled=is_filled)


def make_test_poles_zeros_sgrid(B):
    G = (a * s**2 + 1) / (s**4 + 4*s**3 + 6*s**2 + 5*s + 2)
    return plot_pole_zero(G, sgrid=True, show=False, backend=B,
        params={a: (1, 0, 2)})


def make_test_root_locus_4(B, sgrid, zgrid):
    G1 = 1 / (s + 2)
    G2 = (a * s**2 + 1) / (s**4 + 4*s**3 + 6*s**2 + 5*s + 2)
    return plot_root_locus(G1, G2, show=False,
        backend=B, sgrid=sgrid, zgrid=zgrid, params={a: (1, 0, 2)})


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


def make_test_hvlines(B):
    p = {a: (1, 0, 5), b: (2, 0, 5)}
    return graphics(
        HVLineSeries(a, is_horizontal=True, params=p),
        HVLineSeries(b, is_horizontal=False, params=p),
        backend=B, show=False
    )


def make_test_grid_minor_grid(B, grid, minor_grid):
    return graphics(
        line(cos(x), (x, -pi, pi)),
        show=False, backend=B, grid=grid, minor_grid=minor_grid
    )


def make_test_tick_formatters_2d(B, x_ticks_formatter, y_ticks_formatter):
    return graphics(
        contour(cos(x**2 + y**2), (x, -pi, pi), (y, -2*pi, 2*pi), n=10),
        show=False, backend=B, grid=False,
        x_ticks_formatter=x_ticks_formatter,
        y_ticks_formatter=y_ticks_formatter
    )


def make_test_tick_formatters_3d(B, x_ticks_formatter, y_ticks_formatter):
    return graphics(
        contour(cos(x**2 + y**2), (x, -pi, pi), (y, -2*pi, 2*pi), n=10),
        show=False, backend=B, grid=False,
        x_ticks_formatter=x_ticks_formatter,
        y_ticks_formatter=y_ticks_formatter
    )


def make_test_tick_formatter_polar_axis(B, x_ticks_formatter):
    # plots of the soluzion of sin(z**3 + 1) = 0 on the complex plane
    # in polar form
    x = [
        0.8708338169785833, 0.8708338169785833, -1.7416676339571666,
        0.6444891748561715, 0.6444891748561715, -1.288978349712343,
        1.0, -0.5, -0.5, 1.6059146520513297, -0.8029573260256648,
        -0.8029573260256648
    ]
    y = [
        1.508328415956043, -1.508328415956043, 0.0, 1.1162879957790313,
        -1.1162879957790313, 0.0, 0.0, -0.8660254037844386,
        0.8660254037844386, 0.0, -1.3907628849860991, 1.3907628849860991
    ]
    return graphics(
        list_2d(x, y, is_scatter=True),
        backend=B, polar_axis=True, show=False,
        x_ticks_formatter=x_ticks_formatter
    )


def make_test_hooks_2d(B, hooks):
    return graphics(
        line_parametric_2d(cos(x), sin(x), (x, 0, 3*pi/2), n=10),
        backend=B, hooks=hooks, show=False
    )


def make_test_hooks_3d(B, hooks):
    return graphics(
        surface(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), n=10),
        backend=B, hooks=hooks, show=False, title="title"
    )


def make_test_surface_use_cm_cmin_cmax_zlim(B, zlim, color_func=None):
    x, y, a, b = symbols("x, y, a, b")
    expr = x**4 + y**4 + y**3 - (4 * x**2 * y) + y**2 - (a * x) + (b * y)
    x_range = (x, -3, 3)
    y_range = (y, -3, 3)
    params = {a: (0, -2, 2), b: (0, -2, 2)}
    kwargs = {}
    if color_func:
        kwargs["color_func"] = color_func
    return graphics(
        surface(
            expr, x_range, y_range,
            params=params, use_cm=True, label="z", n=10, **kwargs
        ),
        backend=B, zlim=zlim, show=False
    )
