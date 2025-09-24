from sympy import (
    pi, Symbol, sin, cos, sqrt, atan2, Tuple, Plane
)
from spb.doc_utils.docstrings import _PARAMS
from spb.doc_utils.ipython import modify_graphics_series_doc
from spb.series import (
    Parametric3DLineSeries, SurfaceOver2DRangeSeries, ParametricSurfaceSeries,
    Implicit3DSeries, List3DSeries, ComplexSurfaceBaseSeries, PlaneSeries
)
from spb.utils import (
    _create_missing_ranges, _preprocess_multiple_ranges,
    prange, spherical_to_cartesian
)
from spb.graphics.utils import _plot3d_wireframe_helper, _plot_sympify
import warnings


@modify_graphics_series_doc(Parametric3DLineSeries, replace={"params": _PARAMS})
def line_parametric_3d(
    expr_x, expr_y, expr_z, range_p=None, label=None,
    rendering_kw=None, colorbar=True, use_cm=True, **kwargs
):
    """
    Plots a 3D parametric curve.

    Returns
    =======

    series : list
        A list containing one instance of ``Parametric3DLineSeries``.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi, root
       >>> from spb import *
       >>> t = symbols('t')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(line_parametric_3d(cos(t), sin(t), t, (t, -5, 5)))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(t), sin(t), t) for t over (-5, 5)

    Customize the appearance by setting a label to the colorbar, changing the
    colormap and the line width.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     line_parametric_3d(
       ...         3 * sin(t) + 2 * sin(3 * t), cos(t) - 2 * cos(3 * t), cos(5 * t),
       ...         (t, 0, 2 * pi), "t [rad]", {"cmap": "hsv", "lw": 1.5}
       ...     )
       ... )
       Plot object containing:
       [0]: 3D parametric cartesian line: (3*sin(t) + 2*sin(3*t), cos(t) - 2*cos(3*t), cos(5*t)) for t over (0, 2*pi)

    Plot multiple parametric 3D lines with different ranges:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> a, b, n = 2, 1, 4
       >>> p, r, s = symbols("p r s")
       >>> xp = a * cos(p) * cos(n * p)
       >>> yp = a * sin(p) * cos(n * p)
       >>> zp = b * cos(n * p)**2 + pi
       >>> xr = root(r, 3) * cos(r)
       >>> yr = root(r, 3) * sin(r)
       >>> zr = 0
       >>> graphics(
       ...     line_parametric_3d(
       ...         xp, yp, zp, (p, 0, pi if n % 2 == 1 else 2 * pi), "petals",
       ...         use_cm=False),
       ...     line_parametric_3d(xr, yr, zr, (r, 0, 6*pi), "roots",
       ...         use_cm=False),
       ...     line_parametric_3d(-sin(s)/3, 0, s, (s, 0, pi), "stem",
       ...         use_cm=False)
       ... )
       Plot object containing:
       [0]: 3D parametric cartesian line: (2*cos(p)*cos(4*p), 2*sin(p)*cos(4*p), cos(4*p)**2 + pi) for p over (0, 2*pi)
       [1]: 3D parametric cartesian line: (r**(1/3)*cos(r), r**(1/3)*sin(r), 0) for r over (0, 6*pi)
       [2]: 3D parametric cartesian line: (-sin(s)/3, 0, s) for s over (0, pi)

    Plotting a numerical function instead of a symbolic expression, using
    Plotly:

    .. plotly::

       from spb import *
       import numpy as np
       fx = lambda t: (1 + 0.25 * np.cos(75 * t)) * np.cos(t)
       fy = lambda t: (1 + 0.25 * np.cos(75 * t)) * np.sin(t)
       fz = lambda t: t + 2 * np.sin(75 * t)
       graphics(
           line_parametric_3d(fx, fy, fz, ("t", 0, 6 * np.pi),
               rendering_kw={"line": {"colorscale": "bluered"}},
               adaptive=False, n=1e04),
           title="Helical Toroid", backend=PB)

    Interactive-widget plot of the parametric line over a tennis ball.
    Refer to the interactive sub-module documentation to learn more about the
    ``params`` dictionary. This plot illustrates:

    * combining together different plots.
    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::

       from sympy import *
       from spb import *
       import k3d
       a, b, s, e, t = symbols("a, b, s, e, t")
       c = 2 * sqrt(a * b)
       r = a + b
       params = {
           a: (1.5, 0, 2),
           b: (1, 0, 2),
           s: (0, 0, 2),
           e: (2, 0, 2)
       }
       graphics(
           surface_revolution(
               (r * cos(t), r * sin(t)), (t, 0, pi),
               params=params, n=50, parallel_axis="x", show_curve=False,
               rendering_kw={"color":0x353535},
               force_real_eval=True
           ),
           line_parametric_3d(
               a * cos(t) + b * cos(3 * t),
               a * sin(t) - b * sin(3 * t),
               c * sin(2 * t), prange(t, s*pi, e*pi),
               rendering_kw={"color_map": k3d.matplotlib_color_maps.Summer},
               params=params
           ),
           backend=KB
       )

    See Also
    ========

    spb.graphics.functions_2d.line_parametric_2d, list_3d

    """
    expr_x, expr_y, expr_z = map(_plot_sympify, [expr_x, expr_y, expr_z])
    params = kwargs.get("params", {})
    range_p = _create_missing_ranges(
        [expr_x, expr_y, expr_z], [range_p] if range_p else [], 1, params)[0]
    s = Parametric3DLineSeries(
        expr_x, expr_y, expr_z, range_p, label,
        rendering_kw=rendering_kw, colorbar=colorbar,
        use_cm=use_cm, **kwargs)
    return [s]


@modify_graphics_series_doc(SurfaceOver2DRangeSeries, replace={"params": _PARAMS})
def surface(
    expr, range_x=None, range_y=None, label=None, rendering_kw=None,
    colorbar=True, use_cm=False, **kwargs
):
    """
    Creates the surface of a function of 2 variables.

    Parameters
    ==========

    wireframe : boolean, optional
        Enable or disable a wireframe over the surface. Depending on the number
        of wireframe lines (see ``wf_n1`` and ``wf_n2``), activating this
        option might add a considerable overhead during the plot's creation.
        Default to False (disabled).
    wf_n1, wf_n2 : int, optional
        Number of wireframe lines along the two ranges, respectively.
        Default to 10. Note that increasing this number might considerably
        slow down the plot's creation. Related parameter: ``wireframe``.
    wf_npoints : int or None, optional
        Number of discretization points for the wireframe lines. Default to
        None, meaning that each wireframe line will have ``n1`` or ``n2``
        number of points, depending on the line direction.
        Related parameter: ``wireframe``.
    wf_rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of wireframe lines.
        Related parameter: ``wireframe``.

    Returns
    =======

    series : list
        A list containing one instance of ``SurfaceOver2DRangeSeries`` and
        possibly multiple instances of ``Parametric3DLineSeries``, if
        ``wireframe=True``.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi, exp
       >>> from spb import *
       >>> x, y = symbols('x y')

    Single plot with Matplotlib, with ticks formatted as multiples of `pi/2`.


    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     surface(cos((x**2 + y**2)), (x, -pi, pi), (y, -pi, pi)),
       ...     x_ticks_formatter=multiples_of_pi_over_2(),
       ...     y_ticks_formatter=multiples_of_pi_over_2(),
       ... )
       Plot object containing:
       [0]: cartesian surface: cos(x**2 + y**2) for x over (-pi, pi) and y over (-pi, pi)


    Single plot with Plotly, illustrating how to apply:

    * a color map: by default, it will map colors to the z values.
    * wireframe lines to better understand the discretization and curvature.
    * transformation to the discretized ranges in order to convert radians to
      degrees.
    * custom aspect ratio with Plotly.

    .. plotly::
       :context: reset

       from sympy import symbols, sin, cos, pi
       from spb import *
       import numpy as np
       x, y = symbols("x, y")
       expr = (cos(x) + sin(x) * sin(y) - sin(x) * cos(y))**2
       graphics(
           surface(expr, (x, 0, pi), (y, 0, 2 * pi), use_cm=True,
               tx=np.rad2deg, ty=np.rad2deg,
               wireframe=True, wf_n1=20, wf_n2=20),
           backend=PB, xlabel="x [deg]", ylabel="y [deg]",
           aspect=dict(x=1.5, y=1.5, z=0.5))

    Multiple plots with same range using color maps. By default, colors are
    mapped to the z values:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     surface(x*y, (x, -5, 5), (y, -5, 5), use_cm=True),
       ...     surface(-x*y, (x, -5, 5), (y, -5, 5), use_cm=True))
       Plot object containing:
       [0]: cartesian surface: x*y for x over (-5, 5) and y over (-5, 5)
       [1]: cartesian surface: -x*y for x over (-5, 5) and y over (-5, 5)

    Multiple plots with different ranges and solid colors.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> f = x**2 + y**2
       >>> graphics(
       ...     surface(f, (x, -3, 3), (y, -3, 3)),
       ...     surface(-f, (x, -5, 5), (y, -5, 5)))
       Plot object containing:
       [0]: cartesian surface: x**2 + y**2 for x over (-3, 3) and y over (-3, 3)
       [1]: cartesian surface: -x**2 - y**2 for x over (-5, 5) and y over (-5, 5)

    Single plot with a polar discretization, a color function mapping a
    colormap to the radius. Note that the same result can be achieved with
    ``plot3d_revolution``.

    .. k3d-screenshot::
       :camera: 4.6, -3.6, 3.86, 2.55, -2.06, 0.36, -0.6, 0.5, 0.63

       from sympy import *
       from spb import *
       import numpy as np
       r, theta = symbols("r, theta")
       expr = cos(r**2) * exp(-r / 3)
       graphics(
           surface(expr, (r, 0, 5), (theta, 1.6 * pi, 2 * pi),
               use_cm=True, color_func=lambda x, y, z: np.sqrt(x**2 + y**2),
               is_polar=True,
               wireframe=True, wf_n1=30, wf_n2=10,
               wf_rendering_kw={"width": 0.005}),
           backend=KB, legend=True, grid=False)

    Plotting a numerical function instead of a symbolic expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     surface(lambda x, y: x * np.exp(-x**2 - y**2),
       ...         ("x", -3, 3), ("y", -3, 3), use_cm=True))  # doctest: +SKIP

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).
    * the use of :py:class:`panel.widgets.slider.RangeSlider`, which is a
      2-values widget.

    .. panel-screenshot::
       :small-size: 800, 600

       from sympy import *
       from spb import *
       import panel as pn
       x, y, a, b, c, d, e = symbols("x y a b c d e")
       graphics(
           surface(
               cos(x**2 + y**2) * exp(-(x**2 + y**2) * a),
               prange(x, b, c), prange(y, d, e),
               params={
                   a: (0.25, 0, 1),
                   (b, c): pn.widgets.RangeSlider(
                       value=(-2, 2), start=-4, end=4, step=0.1),
                   (d, e): pn.widgets.RangeSlider(
                       value=(-2, 2), start=-4, end=4, step=0.1),
               },
               use_cm=True, n=100,
               wireframe=True, wf_n1=15, wf_n2=15),
           backend=PB, aspect=dict(x=1.5, y=1.5, z=0.75))

    See Also
    ========

    spb.graphics.functions_2d.contour, surface_parametric, surface_spherical,
    surface_revolution, wireframe, plane

    """
    # back-compatibility
    range_x = kwargs.pop("range1", range_x)
    range_y = kwargs.pop("range2", range_y)

    expr = _plot_sympify(expr)
    params = kwargs.get("params", {})
    if not (range_x and range_y):
        warnings.warn(
            "No ranges were provided. This function will attempt to find "
            "them, however the order will be arbitrary, which means the "
            "visualization might be flipped."
        )
    kwargs_without_wireframe = _remove_wireframe_kwargs(kwargs)
    ranges = _preprocess_multiple_ranges([expr], [range_x, range_y], 2, params)
    s = SurfaceOver2DRangeSeries(
        expr, *ranges, label,
        rendering_kw=rendering_kw, colorbar=colorbar,
        use_cm=use_cm, **kwargs_without_wireframe)
    s = [s]
    s += _plot3d_wireframe_helper(s, **kwargs)
    return s


def _remove_wireframe_kwargs(kwargs):
    kwargs_without_wireframe = kwargs.copy()
    for k in ["wireframe", "wf_n1", "wf_n2", "wf_npoints", "wf_rendering_kw"]:
        kwargs_without_wireframe.pop(k, None)
    return kwargs_without_wireframe


@modify_graphics_series_doc(ParametricSurfaceSeries, replace={"params": _PARAMS})
def surface_parametric(
    expr_x, expr_y, expr_z, range_u=None, range_v=None,
    label=None, rendering_kw=None, **kwargs
):
    """
    Creates a 3D parametric surface.

    Parameters
    ==========

    wireframe : boolean, optional
        Enable or disable a wireframe over the surface. Depending on the number
        of wireframe lines (see ``wf_n1`` and ``wf_n2``), activating this
        option might add a considerable overhead during the plot's creation.
        Default to False (disabled).
    wf_n1, wf_n2 : int, optional
        Number of wireframe lines along the two ranges, respectively.
        Default to 10. Note that increasing this number might considerably
        slow down the plot's creation. Related parameter: ``wireframe``.
    wf_npoints : int or None, optional
        Number of discretization points for the wireframe lines. Default to
        None, meaning that each wireframe line will have ``n1`` or ``n2``
        number of points, depending on the line direction.
        Related parameter: ``wireframe``.
    wf_rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of wireframe lines.
        Related parameter: ``wireframe``.

    Returns
    =======

    series : list
        A list containing one instance of ``ParametricSurfaceSeries`` and
        possibly multiple instances of ``Parametric3DLineSeries``, if
        ``wireframe=True``.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi, I, sqrt, atan2, re, im
       >>> from spb import *
       >>> u, v = symbols('u v')

    Plot a parametric surface:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     surface_parametric(
       ...         u * cos(v), u * sin(v), u * cos(4 * v) / 2,
       ...         (u, 0, pi), (v, 0, 2*pi), use_cm=False),
       ...     title="Sinusoidal Cone")
       Plot object containing:
       [0]: parametric cartesian surface: (u*cos(v), u*sin(v), u*cos(4*v)/2) for u over (0, pi) and v over (0, 2*pi)

    Customize the appearance of the surface by changing the colormap. Apply a
    color function mapping the `v` values. Activate the wireframe to better
    visualize the parameterization.

    .. k3d-screenshot::
       :camera: 2.215, -2.945, 2.107, 0.06, -0.374, -0.459, -0.365, 0.428, 0.827

       from sympy import *
       from spb import *
       import k3d
       var("u, v")

       x = (1 + v / 2 * cos(u / 2)) * cos(u)
       y = (1 + v / 2 * cos(u / 2)) * sin(u)
       z = v / 2 * sin(u / 2)
       graphics(
           surface_parametric(x, y, z, (u, 0, 2*pi), (v, -1, 1),
               "v", {"color_map": k3d.colormaps.paraview_color_maps.Hue_L60},
               use_cm=True, color_func=lambda u, v: u,
               wireframe=True, wf_n1=20, wf_rendering_kw={"width": 0.004}),
           backend=KB, title="Möbius \\, strip")

    Riemann surfaces of the real part of the multivalued function `z**n`,
    using Plotly:

    .. plotly::
       :context: reset

       from sympy import symbols, sqrt, re, im, pi, atan2, sin, cos, I
       from spb import *
       r, theta, x, y = symbols("r, theta, x, y", real=True)
       mag = lambda z: sqrt(re(z)**2 + im(z)**2)
       phase = lambda z, k=0: atan2(im(z), re(z)) + 2 * k * pi
       n = 2 # exponent (integer)
       z = x + I * y # cartesian
       d = {x: r * cos(theta), y: r * sin(theta)} # cartesian to polar
       branches = [(mag(z)**(1 / n) * cos(phase(z, i) / n)).subs(d)
           for i in range(n)]
       exprs = [(r * cos(theta), r * sin(theta), rb) for rb in branches]
       series = [
           surface_parametric(*e, (r, 0, 3), (theta, -pi, pi),
               label="branch %s" % (i + 1), wireframe=True, wf_n2=20)
           for i, e in enumerate(exprs)]
       graphics(*series, backend=PB, zlabel="f(z)")

    Plotting a numerical function instead of a symbolic expression.

    .. k3d-screenshot::
       :camera: 5.3, -7.6, 4, -0.2, -0.9, -1.3, -0.25, 0.4, 0.9

       from spb import *
       import numpy as np
       fx = lambda u, v: (4 + np.cos(u)) * np.cos(v)
       fy = lambda u, v: (4 + np.cos(u)) * np.sin(v)
       fz = lambda u, v: np.sin(u)
       graphics(
           surface_parametric(fx, fy, fz, ("u", 0, 2 * np.pi),
               ("v", 0, 2 * np.pi)),
           zlim=(-2.5, 2.5), title="Torus", backend=KB, grid=False)

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::

       from sympy import *
       from spb import *
       import k3d
       alpha, u, v, up, vp = symbols("alpha u v u_p v_p")
       graphics(
           surface_parametric(
               exp(u) * cos(v - alpha) / 2 + exp(-u) * cos(v + alpha) / 2,
               exp(u) * sin(v - alpha) / 2 + exp(-u) * sin(v + alpha) / 2,
               cos(alpha) * u + sin(alpha) * v,
               prange(u, -up, up), prange(v, 0, vp * pi),
               n=50, use_cm=True, color_func=lambda u, v: v,
               rendering_kw={"color_map": k3d.colormaps.paraview_color_maps.Hue_L60},
               wireframe=True, wf_n2=15, wf_rendering_kw={"width": 0.005},
               params={
                   alpha: (0, 0, pi),
                   up: (1, 0, 2),
                   vp: (2, 0, 2),
               }),
           backend=KB, grid=False,
           title="Catenoid \\, to \\, Right \\, Helicoid \\, Transformation"
       )

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. Note that the plot's
    creation might be slow due to the wireframe lines.

    .. panel-screenshot::

       from sympy import *
       from spb import *
       import panel as pn
       n, u, v = symbols("n, u, v")
       x = v * cos(u)
       y = v * sin(u)
       z = sin(n * u)
       graphics(
           surface_parametric(x, y, z, (u, 0, 2*pi), (v, -1, 0),
               params = {
                   n: pn.widgets.IntInput(value=3, name="n")
               },
               use_cm=True, wireframe=True, wf_n1=75, wf_n2=6),
           backend=PB,
           title="Plücker's conoid",
           imodule="panel"
       )

    See Also
    ========

    surface, surface_spherical, surface_revolution, wireframe

    """
    # back-compatibility
    range_u = kwargs.pop("range1", range_u)
    range_v = kwargs.pop("range2", range_v)

    expr_x, expr_y, expr_z = map(_plot_sympify, [expr_x, expr_y, expr_z])
    params = kwargs.get("params", {})
    if not (range_u and range_v):
        warnings.warn(
            "No ranges were provided. This function will attempt to find "
            "them, however the order will be arbitrary, which means the "
            "visualization might be flipped."
        )
    kwargs_without_wireframe = _remove_wireframe_kwargs(kwargs)
    ranges = _preprocess_multiple_ranges(
        [expr_x, expr_y, expr_z], [range_u, range_v], 2, params)
    s = ParametricSurfaceSeries(
        expr_x, expr_y, expr_z, *ranges, label,
        rendering_kw=rendering_kw, **kwargs_without_wireframe)
    return [s] + _plot3d_wireframe_helper([s], **kwargs)


@modify_graphics_series_doc(
    ParametricSurfaceSeries,
    replace={"params": _PARAMS},
    exclude=["expr_x", "expr_y", "expr_z", "range_u", "range_v"]
)
def surface_spherical(
    r, range_theta=None, range_phi=None, label=None,
    rendering_kw=None, **kwargs
):
    """
    Plots a radius as a function of the spherical coordinates theta and phi.

    Parameters
    ==========

    r : Expr or callable
        Expression representing the radius. It can be a:

        * Symbolic expression.
        * Numerical function of two variable, f(theta, phi), supporting
          vectorization. In this case the following keyword arguments are
          not supported: ``params``.
    range_theta : tuple
        A 3-tuple (symbol, min, max) denoting the range of the polar angle,
        which is limited in [0, pi]. Consider a sphere:

        * ``theta=0`` indicates the north pole.
        * ``theta=pi/2`` indicates the equator.
        * ``theta=pi`` indicates the south pole.
    range_phi : tuple
        A 3-tuple (symbol, min, max) denoting the range of the azimuthal angle,
        which is limited in [0, 2*pi].

    Returns
    =======

    series : list
        A list containing one instance of ``ParametricSurfaceSeries`` and
        possibly multiple instances of ``Parametric3DLineSeries``, if
        ``wireframe=True``.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi, Ynm, re, lambdify
       >>> from spb import *
       >>> theta, phi = symbols('theta phi')

    Sphere cap:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     surface_spherical(1, (theta, 0, 0.7 * pi), (phi, 0, 1.8 * pi)))
       Plot object containing:
       [0]: parametric cartesian surface: (sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta)) for theta over (0, 0.7*pi) and phi over (0, 1.8*pi)

    Plot real spherical harmonics, highlighting the regions in which the
    real part is positive and negative, using Plotly:

    .. plotly::
       :context: reset

       from sympy import symbols, sin, pi, Ynm, re, lambdify
       from spb import *
       theta, phi = symbols('theta phi')
       r = re(Ynm(3, 3, theta, phi).expand(func=True).rewrite(sin).expand())
       graphics(
           surface_spherical(
               abs(r), (theta, 0, pi), (phi, 0, 2 * pi), "real",
               use_cm=True, n2=200,
               color_func=lambdify([theta, phi], r)),
           backend=PB)

    Multiple surfaces with wireframe lines, using Plotly. Note that activating
    the wireframe option might add a considerable overhead during the plot's
    creation.

    .. plotly::

       from sympy import symbols, sin, pi
       from spb import *
       theta, phi = symbols('theta phi')
       r1 = 1
       r2 = 1.5 + sin(5 * phi) * sin(10 * theta) / 10
       graphics(
           surface_spherical(r1, (theta, 0, pi / 2), (phi, 0.35 * pi, 2 * pi),
               label="r1", wireframe=True, wf_n2=25),
           surface_spherical(r2, (theta, 0, pi / 2), (phi, 0.35 * pi, 2 * pi),
               label="r2", wireframe=True, wf_n2=25),
           backend=PB)

    Interactive-widget plot of real spherical harmonics, highlighting the
    regions in which the real part is positive and negative.
    Note that the plot's creation and update might be slow and that
    it must be ``m < n`` at all times.
    Refer to the interactive sub-module documentation to learn more about the
    ``params`` dictionary.

    .. panel-screenshot::

       from sympy import *
       from spb import *
       import panel as pn
       n, m = symbols("n, m")
       phi, theta = symbols("phi, theta", real=True)
       r = re(Ynm(n, m, theta, phi).expand(func=True).rewrite(sin).expand())
       graphics(
           surface_spherical(abs(r), (theta, 0, pi), (phi, 0, 2*pi),
               label="real",
               params = {
                   n: pn.widgets.IntInput(value=2, name="n"),
                   m: pn.widgets.IntInput(value=0, name="m"),
               },
               use_cm=True, color_func=r, force_real_eval=True),
           backend=PB, imodule="panel")

    See Also
    ========

    surface, surface_parametric, surface_revolution, wireframe

    """
    r = _plot_sympify(r)
    params = kwargs.get("params", {})
    if not (range_theta or range_phi):
        warnings.warn(
            "No ranges were provided. This function will attempt to find "
            "them, however the order will be arbitrary, which means the "
            "visualization might be flipped."
        )

    # deal with symbolic min/max values of ranges
    def rel(t, s, threshold, a):
        try:
            if t == "<":
                if s < threshold:
                    return threshold
            elif t == ">":
                if s > threshold:
                    return threshold
        except Exception:
            return a
        return a

    range_theta, range_phi = _preprocess_multiple_ranges(
        [r], [range_theta, range_phi], 2, params)

    theta, phi = range_theta[0], range_phi[0]
    x, y, z = spherical_to_cartesian(r, theta, phi)

    # enforce polar and azimuthal condition and convert spherical to cartesian
    range_theta = prange(
        theta,
        rel("<", range_theta[1], 0, range_theta[1]),
        rel(">", range_theta[2], pi, range_theta[2]))
    range_phi = prange(
        phi,
        rel("<", range_phi[1], 0, range_phi[1]),
        rel(">", range_phi[2], 2*pi, range_phi[2]))

    return surface_parametric(
        x, y, z, range_theta, range_phi, label,
        rendering_kw=rendering_kw, **kwargs)


@modify_graphics_series_doc(Implicit3DSeries, replace={"params": _PARAMS})
def implicit_3d(
    expr, range_x=None, range_y=None, range_z=None, label=None,
    rendering_kw=None, **kwargs
):
    """
    Plots an isosurface of a function.

    Notes
    =====
    1. the number of discretization points is crucial as the algorithm will
       discretize a volume. A high number of discretization points creates a
       smoother mesh, at the cost of a much higher memory consumption and
       slower computation.
    2. Only ``PlotlyBackend`` and ``K3DBackend`` support 3D implicit plotting.
    3. To plot ``f(x, y, z) = c`` either write ``expr = f(x, y, z) - c`` or
       pass the appropriate keyword to ``rendering_kw``. Read the backends
       documentation to find out the available options.

    Returns
    =======

    series : list
        A list containing one instance of ``Implicit3DSeries``.

    Examples
    ========

    .. plotly::
       :context: reset

       from sympy import symbols
       from spb import *
       x, y, z = symbols('x, y, z')
       graphics(
           implicit_3d(x**2 + y**3 - z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2)),
           backend=PB)

    .. plotly::
       :context: close-figs

       graphics(
           implicit_3d(x**4 + y**4 + z**4 - (x**2 + y**2 + z**2 - 0.3),
               (x, -2, 2), (y, -2, 2), (z, -2, 2)),
           backend=PB)

    Visualize the isocontours from `isomin=0` to `isomax=2` by providing a
    ``rendering_kw`` dictionary:

    .. plotly::
       :context: close-figs

       graphics(
           implicit_3d(1/x**2 - 1/y**2 + 1/z**2,
               (x, -2, 2), (y, -2, 2), (z, -2, 2),
               rendering_kw={
                   "isomin": 0, "isomax": 2,
                   "colorscale":"aggrnyl", "showscale":True
               }),
           backend=PB
       )

    See Also
    ========

    surface, spb.graphics.functions_2d.implicit_2d

    """
    # back-compatibility
    range_x = kwargs.pop("range1", range_x)
    range_y = kwargs.pop("range2", range_y)
    range_z = kwargs.pop("range3", range_z)

    expr = _plot_sympify(expr)
    params = kwargs.get("params", {})

    check = [range_x is None, range_y is None, range_z is None]
    if sum(check) >= 2:
        pre = "Not enough ranges were provided. "
        if sum(check) == 3:
            pre = "No ranges were provided. "
        warnings.warn(
            pre + "This function will attempt to find "
            "them, however the order will be arbitrary, which means the "
            "visualization might be flipped."
        )

    ranges = _preprocess_multiple_ranges(
        [expr], [range_x, range_y, range_z], 3, params)
    s = Implicit3DSeries(
        expr, *ranges, label, rendering_kw=rendering_kw, **kwargs)
    return [s]


@modify_graphics_series_doc(
    ParametricSurfaceSeries,
    replace={"params": _PARAMS},
    exclude=["expr_x", "expr_y", "expr_z", "range_u", "range_v"]
)
def surface_revolution(
    curve, range_t, range_phi=None, axis=(0, 0),
    parallel_axis='z', show_curve=False, curve_kw={}, **kwargs
):
    """
    Creates a surface of revolution by rotating a curve around an axis of
    rotation.

    Parameters
    ==========

    curve : Expr, list ortuple of 2 or 3 elements
        The curve to be revolved, which can be either:

        * a symbolic expression
        * a 2-tuple representing a parametric curve in 2D space
        * a 3-tuple representing a parametric curve in 3D space
    range_t : tuple
        A 3-tuple (symbol, min, max) denoting the range of the parameter of
        the curve.
    range_phi : tuple
        A 3-tuple (symbol, min, max) denoting the range of the azimuthal angle
        where the curve will be revolved. Default to ``(phi, 0, 2*pi)``.
    axis : tuple
        A 2-tuple (coord1, coord2) that specifies the position of the rotation
        axis. Depending on the value of ``parallel_axis``:

        * ``"x"``: the rotation axis intersects the YZ plane at
          (coord1, coord2).
        * ``"y"``: the rotation axis intersects the XZ plane at
          (coord1, coord2).
        * ``"z"``: the rotation axis intersects the XY plane at
          (coord1, coord2).

        Default to ``(0, 0)``.
    parallel_axis : str
        Specify the axis parallel to the axis of rotation. Must be one of the
        following options: "x", "y" or "z". Default to "z".
    show_curve : bool
        Add the initial curve to the plot. Default to False.
    curve_kw : dict
        A dictionary of options that will be passed to
        ``plot3d_parametric_line`` if ``show_curve=True`` in order to customize
        the appearance of the initial curve. Refer to its documentation for
        more information.

    Returns
    =======

    series : list
        A list containing one instance of ``ParametricSurfaceSeries``,
        possibly multiple instances of ``Parametric3DLineSeries`` representing
        wireframe lines (if ``wireframe=True``) and possible another instance
        of ``Parametric3DLineSeries`` representing the curve
        (if ``show_curve=True``).

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi
       >>> from spb import *
       >>> t, phi = symbols('t phi')

    Revolve a function around the z axis:

    .. plot::
       :context: close-figs
       :format: python
       :include-source: True

       graphics(
           surface_revolution(
               cos(t), (t, 0, pi),
               use_cm=True, color_func=lambda t, phi: phi,
               rendering_kw={"alpha": 0.6, "cmap": "twilight"},
               # indicates the azimuthal angle on the colorbar label
               label="$\\phi$ [rad]",
               show_curve=True,
               # this dictionary will be passes to plot3d_parametric_line in
               # order to draw the initial curve
               curve_kw=dict(rendering_kw={"color": "r", "label": "cos(t)"}),
               # activate the wireframe to visualize the parameterization
               wireframe=True, wf_n1=15, wf_n2=15,
               wf_rendering_kw={"lw": 0.5, "alpha": 0.75}
           )
       )

    Revolve the same function around an axis parallel to the x axis, using
    Plotly:

    .. plotly::
       :context: reset

       from sympy import symbols, cos, sin, pi
       from spb import *
       t, phi = symbols('t phi')
       graphics(
           surface_revolution(
               cos(t), (t, 0, pi), parallel_axis="x", axis=(1, 0),
               label="phi [rad]", rendering_kw={"colorscale": "twilight"},
               use_cm=True, color_func=lambda t, phi: phi,
               show_curve=True,
               curve_kw=dict(rendering_kw={"line": {"color": "red", "width": 8},
                   "name": "cos(t)"}),
               wireframe=True, wf_n1=15, wf_n2=15,
               wf_rendering_kw={"line_width": 1}
           ),
           backend=PB
       )

    Revolve a 2D parametric circle around the z axis:

    .. k3d-screenshot::
       :camera: 4.3, -5.82, 4.95, 0.4, -0.25, -0.67, -0.32, 0.5, 0.8

       from sympy import *
       from spb import *
       t = symbols("t")
       circle = (3 + cos(t), sin(t))
       graphics(
           surface_revolution(circle, (t, 0, 2 * pi),
               show_curve=True, rendering_kw={"opacity": 0.65},
               curve_kw={"rendering_kw": {"width": 0.05}}),
           backend=KB)

    Revolve a 3D parametric curve around the z axis for a given azimuthal
    angle, using Plotly:

    .. plotly::
       :context: close-figs

       from sympy import *
       from spb import *
       t = symbols("t")
       graphics(
           surface_revolution(
               (cos(t), sin(t), t), (t, 0, 2*pi), (phi, 0, pi),
               use_cm=True, color_func=lambda t, phi: t, label="t [rad]",
               show_curve=True,
               wireframe=True, wf_n1=2, wf_n2=5),
           backend=PB, aspect="cube")

    Interactive-widget plot of a goblet. Refer to the interactive sub-module
    documentation to learn more about the ``params`` dictionary.
    This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::

       from sympy import *
       from spb import *
       t, phi, u, v, w = symbols("t phi u v w")
       graphics(
           surface_revolution(
               (t, cos(u * t), t**2), prange(t, 0, v), prange(phi, 0, w*pi),
               axis=(1, 0.2), n=50,
               wireframe=True, wf_n1=15, wf_n2=15,
               wf_rendering_kw={"width": 0.004},
               show_curve=True, curve_kw={"rendering_kw": {"width": 0.025}},
               params={
                   u: (2.5, 0, 6),
                   v: (2, 0, 3),
                   w: (2, 0, 2)
               }),
           backend=KB, force_real_eval=True)

    See Also
    ========

    surface, surface_spherical, surface_parametric, wireframe

    """
    if parallel_axis.lower() not in ["x", "y", "z"]:
        raise ValueError(
            "`parallel_axis` must be either 'x' 'y' or 'z'. "
            "Received: %s " % parallel_axis)

    params = kwargs.get("params", {})

    # NOTE: a surface of revolution is a particular case of 3D parametric
    # surface
    if isinstance(curve, (tuple, list, Tuple)):
        if len(curve) == 2:     # curve is a 2D parametric line
            x, z = curve
            y = 0
        elif len(curve) == 3:   # curve is a 3D parametric line
            x, y, z = curve
    else:  # curve is an expression
        x = range_t[0]
        y = 0
        z = curve

    phi = range_phi[0] if range_phi else Symbol("phi")
    if range_phi is None:
        range_phi = (phi, 0, 2*pi)

    phase = 0
    if parallel_axis == "x":
        y0, z0 = axis
        phase = atan2(z - z0, y - y0)
        r = sqrt((y - y0)**2 + (z - z0)**2)
        v = (x, r * cos(phi + phase) + y0, r * sin(phi + phase) + z0)
    elif parallel_axis == "y":
        x0, z0 = axis
        phase = atan2(z - z0, x - x0)
        r = sqrt((x - x0)**2 + (z - z0)**2)
        v = (r * cos(phi + phase) + x0, y, r * sin(phi + phase) + z0)
    else:
        x0, y0 = axis
        phase = atan2(y - y0, x - x0)
        r = sqrt((x - x0)**2 + (y - y0)**2)
        v = (r * cos(phi + phase) + x0, r * sin(phi + phase) + y0, z)

    surface = surface_parametric(*v, range_t, range_phi, **kwargs)
    if not isinstance(surface, list):
        surface = [surface]

    if show_curve:
        curve_kw["params"] = params
        # link the number of discretization points between the two series
        curve_kw["n"] = surface[0].n[0]
        curve_kw.setdefault("use_cm", False)
        curve_kw.setdefault("force_real_eval", surface[0].force_real_eval)
        line = line_parametric_3d(x, y, z, range_t, **curve_kw)

        surface.extend(line)
    return surface


@modify_graphics_series_doc(List3DSeries, replace={"params": _PARAMS})
def list_3d(
    list_x, list_y, list_z, label=None, rendering_kw=None, **kwargs
):
    """
    Plots lists of coordinates in 3D space.

    Returns
    =======

    series : list
        A list containing one instance of ``List3DSeries``.


    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import *
       >>> from spb import *
       >>> import numpy as np

    Plot the coordinates of a single function:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> z = np.linspace(0, 6*np.pi, 100)
       >>> x = z * np.cos(z)
       >>> y = z * np.sin(z)
       >>> graphics(list_3d(x, y, z))
       Plot object containing:
       [0]: 3D list plot

    Plotting multiple functions with custom rendering keywords:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     list_3d(x, y, z, "A"),
       ...     list_3d(x, y, -z, "B", {"linestyle": "--"}))
       Plot object containing:
       [0]: 3D list plot
       [1]: 3D list plot

    Interactive-widget plot of a dot following a path. Refer to the
    interactive sub-module documentation to learn more about the ``params``
    dictionary.

    .. panel-screenshot::
       :small-size: 800, 625

       from sympy import *
       from spb import *
       import numpy as np
       t = symbols("t")
       z = np.linspace(0, 6*np.pi, 100)
       x = z * np.cos(z)
       y = z * np.sin(z)
       graphics(
           list_3d(x, y, z, scatter=False),
           list_3d(t * cos(t), t * sin(t), t,
               params={t: (3*pi, 0, 6*pi)}, scatter=True),
           backend=PB
       )

    See Also
    ========

    spb.graphics.functions_2d.line, line_parametric_3d

    """
    if not hasattr(list_x, "__iter__"):
        list_x = [list_x]
    if not hasattr(list_y, "__iter__"):
        list_y = [list_y]
    if not hasattr(list_z, "__iter__"):
        list_z = [list_z]
    s = List3DSeries(
        list_x, list_y, list_z, label, rendering_kw=rendering_kw, **kwargs)
    return [s]


@modify_graphics_series_doc(
    Parametric3DLineSeries,
    replace={"params": _PARAMS},
    exclude=["expr_x", "expr_y", "expr_z", "range_p"]
)
def wireframe(
    surface_series, n1=10, n2=10, n=None, rendering_kw=None, **kwargs
):
    """
    Creates a wireframe of a 3D surface.

    Parameters
    ==========
    surface_series : BaseSeries
        A data series representing a surface.
    n1 : int, optional
        Number of wireframe lines along the first direction (or range).
    n2: int, optional
        Number of wireframe lines along the second direction (or range).
    n : int, optional
        Number of evaluation points for each wireframe line. If not provided,
        the algorithm will chose the number of discretization points from
        the ``surface_series``. The higher this number, the slower the
        creation of the plot.

    Returns
    =======

    series : list
        A list containing instances of ``Parametric3DLineSeries``.

    Examples
    ========

    Provide better code separation between the surface and wireframe:

    .. plotly::

       from sympy import *
       from spb import *
       import numpy as np
       x, y = symbols("x y")
       surf_series = surface(
           cos(x*y) * exp(-sqrt(x**2 + y**2) / 4), (x, -pi, pi), (y, -pi, pi),
           use_cm=True, color_func=lambda x, y, z: np.sqrt(x**2 + y**2)
       )
       graphics(
           surf_series,
           wireframe(surf_series[0], n1=11, n2=11),
           backend=PB, aspect=dict(x=1.5, y=1.5, z=1)
       )

    Show wireframe without showing the actual surface. Here, dotted
    wireframe represent the half-unit sphere.

    .. plotly::
       :camera: 1.5, 1.5, 0.25, 0, 0, 0, 0, 0, 1

       from sympy import *
       from spb import *
       var("t u v theta phi")
       r_sphere = 1
       sphere = surface_spherical(1, (theta, 0, pi), (phi, pi, 2*pi))[0]
       t = pi / 3 # half-cone angle
       r_cone = r_sphere * sin(t)
       graphics(
           wireframe(sphere, n1=13, rendering_kw={"line_dash": "dot"}),
           surface_spherical(1, (theta, pi - t, pi), (phi, pi, 2*pi),
               label="sphere cap", wireframe=True, wf_n1=5),
           surface_parametric(
               u * cos(v), u * sin(v), -u / tan(t), (u, 0, r_cone), (v, pi , 2*pi),
               label="cone", wireframe=True, wf_n1=7),
           backend=PB, grid=False
       )

    See Also
    ========

    surface, surface_parametric, surface_spherical, surface_revolution

    """
    allowed = (
        ComplexSurfaceBaseSeries, SurfaceOver2DRangeSeries,
        ParametricSurfaceSeries
    )
    if not isinstance(surface_series, allowed):
        raise TypeError(
            f"Wireframe lines are supported only for instances of {allowed}. "
            f"Received: type(surface_series) = {type(surface_series)}")
    if not surface_series.is_3Dsurface:
        # ComplexSurfaceBaseSeries can also be 2D
        raise ValueError("Wireframe lines are supported only for 3D series.")

    kw = kwargs.copy()
    kw["wf_n1"] = n1
    kw["wf_n2"] = n2
    kw["wf_rendering_kw"] = rendering_kw
    kw["wireframe"] = True
    kw["wf_npoints"] = n
    return _plot3d_wireframe_helper([surface_series], **kw)


@modify_graphics_series_doc(PlaneSeries, replace={"params": _PARAMS}, exclude=["plane"])
def plane(
    p, range_x=None, range_y=None, range_z=None, label=None,
    rendering_kw=None, **kwargs
):
    """
    Plot a plane in a 3D space.

    Parameters
    ==========

    p : Plane
        The plane to be plotted.

    Returns
    =======

    series : list
        A list containing an instance of ``PlaneSeries``.

    Examples
    ========

    .. plotly::

       from sympy import *
       from spb import *
       from sympy.abc import x, y, z
       ranges = [(x, -5, 5), (y, -5, 5), (z, -5, 5)]
       graphics(
           plane(Plane((0, 0, 0), (1, 0, 0)), *ranges, label="yz"),
           plane(Plane((0, 0, 0), (0, 1, 0)), *ranges, label="xz"),
           plane(Plane((0, 0, 0), (0, 0, 1)), *ranges, label="yz"),
           plane(Plane((0, 0, 0), (1, 1, 1)), *ranges, label="inclined", n=150),
           backend=PB, xlabel="x", ylabel="y", zlabel="z"
       )

    See Also
    ========

    surface

    """
    p = _plot_sympify(p)
    if not isinstance(p, Plane):
        raise TypeError(
            f"`p` must be an instance of `Plane`. Received type(p)={type(p)}")
    params = kwargs.get("params", {})

    check = [range_x is None, range_y is None, range_z is None]
    if sum(check) >= 2:
        pre = "Not enough ranges were provided. "
        if sum(check) == 3:
            pre = "No ranges were provided. "
        warnings.warn(
            pre + "This function will attempt to find "
            "them, however the order will be arbitrary, which means the "
            "visualization might be flipped."
        )

    ranges = _preprocess_multiple_ranges(
        [p], [range_x, range_y, range_z], 3, params)
    s = PlaneSeries(
        p, *ranges, label, rendering_kw=rendering_kw, **kwargs)
    return [s]
