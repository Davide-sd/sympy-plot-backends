"""Plotting module for Sympy.

A plot is represented by the ``Plot`` class that contains a list of the data
series to be plotted. The data series are responsible to generate numerical
data from sympy expressions.

This module gives only the essential. Especially if you need publication ready
graphs and this module is not enough for you, use directly the backend, which
can be accessed with the ``fig`` attribute:
* MatplotlibBackend.fig: returns a Matplotlib figure.
* BokehBackend.fig: return the Bokeh figure object.
* PlotlyBackend.fig: return the Plotly figure object.
* K3DBackend.fig: return the K3D plot object.

Simplicity of code takes much greater importance than performance. Don't use
it if you care at all about performance.
"""

from spb.graphics import (
    graphics, line_parametric_3d,
    surface_parametric, surface_revolution, surface_spherical,
    implicit_3d, list_3d
)
from spb.utils import (
    _plot_sympify, _check_arguments
)
from spb.plot_functions.functions_2d import (
    _set_labels, _plot3d_plot_contour_helper
)


def plot3d_parametric_line(*args, **kwargs):
    """
    Plots a 3D parametric line plot.

    Typical usage examples are in the followings:

    - Plotting a single expression:

      .. code-block::

         plot3d_parametric_line(expr_x, expr_y, expr_z, range, **kwargs)

    - Plotting a single expression with a custom label and rendering options:

      .. code-block::

         plot3d_parametric_line(expr_x, expr_y, expr_z, range,
            label [opt], rendering_kw [opt], **kwargs)

    - Plotting multiple expressions with the same ranges:

      .. code-block::

         plot3d_parametric_line((expr_x1, expr_y1, expr_z1),
            (expr_x2, expr_y2, expr_z2), ..., range, **kwargs)

    - Plotting multiple expressions with different ranges, custom labels and
      rendering options:

      .. code-block::

         plot3d_parametric_line(
            (expr_x1, expr_y1, expr_z1, range1, label1, rendering_kw1),
            (expr_x2, expr_y2, expr_z2, range2, label1, rendering_kw2),
            ..., **kwargs)

    Refer to :func:`~spb.graphics.functions_3d.line_parametric_3d` for a full
    list of keyword arguments to customize the appearances of lines.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi, root
       >>> from spb import plot3d_parametric_line
       >>> t = symbols('t')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line(cos(t), sin(t), t, (t, -5, 5))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(t), sin(t), t) for t over (-5.0, 5.0)

    Customize the appearance by setting a label to the colorbar, changing the
    colormap and the line width.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line(
       ...     3 * sin(t) + 2 * sin(3 * t), cos(t) - 2 * cos(3 * t), cos(5 * t),
       ...     (t, 0, 2 * pi), "t [rad]", {"cmap": "hsv", "lw": 1.5},
       ...     aspect="equal")
       Plot object containing:
       [0]: 3D parametric cartesian line: (3*sin(t) + 2*sin(3*t), cos(t) - 2*cos(3*t), cos(5*t)) for t over (0.0, 6.283185307179586)

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
       >>> plot3d_parametric_line(
       ...     (xp, yp, zp, (p, 0, pi if n % 2 == 1 else 2 * pi), "petals"),
       ...     (xr, yr, zr, (r, 0, 6*pi), "roots"),
       ...     (-sin(s)/3, 0, s, (s, 0, pi), "stem"), use_cm=False)
       Plot object containing:
       [0]: 3D parametric cartesian line: (2*cos(p)*cos(4*p), 2*sin(p)*cos(4*p), cos(4*p)**2 + pi) for p over (0.0, 6.283185307179586)
       [1]: 3D parametric cartesian line: (r**(1/3)*cos(r), r**(1/3)*sin(r), 0) for r over (0.0, 18.84955592153876)
       [2]: 3D parametric cartesian line: (-sin(s)/3, 0, s) for s over (0.0, 3.141592653589793)

    Plotting a numerical function instead of a symbolic expression, using
    Plotly:

    .. plotly::

       from spb import plot3d_parametric_line, PB
       import numpy as np
       fx = lambda t: (1 + 0.25 * np.cos(75 * t)) * np.cos(t)
       fy = lambda t: (1 + 0.25 * np.cos(75 * t)) * np.sin(t)
       fz = lambda t: t + 2 * np.sin(75 * t)
       plot3d_parametric_line(fx, fy, fz, ("t", 0, 6 * np.pi),
           {"line": {"colorscale": "bluered"}},
           title="Helical Toroid", backend=PB, adaptive=False, n=1e04)

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
       sphere = plot3d_revolution(
           (r * cos(t), r * sin(t)), (t, 0, pi),
           params=params, n=50, parallel_axis="x",
           backend=KB,
           show_curve=False, show=False,
           rendering_kw={"color":0x353535})
       line = plot3d_parametric_line(
           a * cos(t) + b * cos(3 * t),
           a * sin(t) - b * sin(3 * t),
           c * sin(2 * t), prange(t, s*pi, e*pi),
           {"color_map": k3d.matplotlib_color_maps.Summer}, params=params,
           backend=KB, show=False, use_latex=False)
       (line + sphere).show()

    See Also
    ========

    plot3d, plot3d_parametric_surface, plot3d_spherical,
    plot3d_revolution, plot3d_implicit, plot3d_list

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 3, 1, **kwargs)
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)
    lines = []

    for pe in plot_expr:
        e1, e2, e3, r, label, rendering_kw = pe
        lines.extend(
            line_parametric_3d(e1, e2, e3, r, label, rendering_kw, **kwargs))
    _set_labels(lines, global_labels, global_rendering_kw)
    return graphics(*lines, **kwargs)


def plot3d(*args, **kwargs):
    """
    Plots a 3D surface plot.

    Typical usage examples are in the followings:

    - Plotting a single expression:

      .. code-block::

         plot3d(expr, range_x, range_y, **kwargs)

    - Plotting multiple expressions with the same ranges:

      .. code-block::

         plot3d(expr1, expr2, range_x, range_y, **kwargs)

    - Plotting multiple expressions with different ranges, custom labels and
      rendering options:

      .. code-block::

         plot3d(
            (expr1, range_x1, range_y1, label1 [opt], rendering_kw1 [opt]),
            (expr2, range_x2, range_y2, label2 [opt], rendering_kw2 [opt]),
            ..., **kwargs)

    Note: it is important to specify at least the ``range_x``, otherwise the
    function might create a rotated plot.

    Refer to :func:`~spb.graphics.functions_3d.surface` for a full
    list of keyword arguments to customize the appearances of surfaces.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Parameters
    ==========

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of expr will be used. The number of labels must be
        equal to the number of expressions.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the plotting
        library (backend) manual for more informations. If a list of
        dictionaries is provided, the number of dictionaries must be equal
        to the number of expressions.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi, exp
       >>> from spb import plot3d
       >>> x, y = symbols('x y')

    Single plot with Matplotlib:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(cos((x**2 + y**2)), (x, -3, 3), (y, -3, 3))
       Plot object containing:
       [0]: cartesian surface: cos(x**2 + y**2) for x over (-3.0, 3.0) and y over (-3.0, 3.0)


    Single plot with Plotly, illustrating how to apply:

    * a color map: by default, it will map colors to the z values.
    * wireframe lines to better understand the discretization and curvature.
    * transformation to the discretized ranges in order to convert radians to
      degrees.
    * custom aspect ratio with Plotly.

    .. plotly::
       :context: reset

       from sympy import symbols, sin, cos, pi
       from spb import plot3d, PB
       import numpy as np
       x, y = symbols("x, y")
       expr = (cos(x) + sin(x) * sin(y) - sin(x) * cos(y))**2
       plot3d(
           expr, (x, 0, pi), (y, 0, 2 * pi), backend=PB, use_cm=True,
           tx=np.rad2deg, ty=np.rad2deg, wireframe=True, wf_n1=20, wf_n2=20,
           xlabel="x [deg]", ylabel="y [deg]",
           aspect=dict(x=1.5, y=1.5, z=0.5))

    Multiple plots with same range using color maps. By default, colors are
    mapped to the z values:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(x*y, -x*y, (x, -5, 5), (y, -5, 5), use_cm=True)
       Plot object containing:
       [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)
       [1]: cartesian surface: -x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    Multiple plots with different ranges and solid colors.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> f = x**2 + y**2
       >>> plot3d((f, (x, -3, 3), (y, -3, 3)),
       ...     (-f, (x, -5, 5), (y, -5, 5)))
       Plot object containing:
       [0]: cartesian surface: x**2 + y**2 for x over (-3.0, 3.0) and y over (-3.0, 3.0)
       [1]: cartesian surface: -x**2 - y**2 for x over (-5.0, 5.0) and y over (-5.0, 5.0)

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
       plot3d(expr, (r, 0, 5), (theta, 1.6 * pi, 2 * pi),
           backend=KB, is_polar=True, legend=True, grid=False,
           use_cm=True, color_func=lambda x, y, z: np.sqrt(x**2 + y**2),
           wireframe=True, wf_n1=30, wf_n2=10,
           wf_rendering_kw={"width": 0.005})

    Plotting a numerical function instead of a symbolic expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(lambda x, y: x * np.exp(-x**2 - y**2),
       ...     ("x", -3, 3), ("y", -3, 3), use_cm=True)  # doctest: +SKIP

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 600

       from sympy import *
       from spb import *
       x, y, a, b, d = symbols("x y a b d")
       plot3d(
           cos(x**2 + y**2) * exp(-(x**2 + y**2) * d),
           prange(x, -2*a, 2*a), prange(y, -2*b, 2*b),
           params={
               a: (1, 0, 3),
               b: (1, 0, 3),
               d: (0.25, 0, 1),
           },
           backend=PB, use_cm=True, n=100, aspect=dict(x=1.5, y=1.5, z=0.75),
           wireframe=True, wf_n1=15, wf_n2=15, throttled=True, use_latex=False)

    See Also
    ========

    plot3d_parametric_list, plot3d_parametric_surface, plot3d_spherical,
    plot3d_revolution, plot3d_implicit, plot3d_list, plot_contour

    """
    return _plot3d_plot_contour_helper(True, *args, **kwargs)


def plot3d_parametric_surface(*args, **kwargs):
    """
    Plots a 3D parametric surface plot.

    Typical usage examples are in the followings:

    - Plotting a single expression:

      .. code-block::

         plot3d_parametric_surface(
                expr_x, expr_y, expr_z, range_u, range_v, label, **kwargs)

    - Plotting multiple expressions with the same ranges:

      .. code-block::

         plot3d_parametric_surface((expr_x1, expr_y1, expr_z1),
            (expr_x2, expr_y2, expr_z2), range_u, range_v, **kwargs)

    - Plotting multiple expressions with different ranges, custom labels and
      rendering option:

      .. code-block::

         plot3d_parametric_surface(
            (expr_x1, expr_y1, expr_z1, range_u1, range_v1,
                label1 [opt], rendering_kw1 [opt]),
            (expr_x2, expr_y2, expr_z2, range_u2, range_v2,
                label2 [opt], rendering_kw2 [opt]), **kwargs)`

    Note: it is important to specify both the ranges.

    Refer to :func:`~spb.graphics.functions_3d.surface_parametric` for a full
    list of keyword arguments to customize the appearances of surfaces.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Parameters
    ==========

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of expr will be used. The number of labels must be
        equal to the number of expressions.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the plotting
        library (backend) manual for more informations. If a list of
        dictionaries is provided, the number of dictionaries must be equal
        to the number of expressions.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi, I, sqrt, atan2, re, im
       >>> from spb import plot3d_parametric_surface
       >>> u, v = symbols('u v')

    Plot a parametric surface:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_surface(
       ...     u * cos(v), u * sin(v), u * cos(4 * v) / 2,
       ...     (u, 0, pi), (v, 0, 2*pi),
       ...     use_cm=False, title="Sinusoidal Cone")
       Plot object containing:
       [0]: parametric cartesian surface: (u*cos(v), u*sin(v), u*cos(4*v)/2) for u over (0.0, 3.141592653589793) and v over (0.0, 6.283185307179586)

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
       plot3d_parametric_surface(
           x, y, z, (u, 0, 2*pi), (v, -1, 1),
           "v", {"color_map": k3d.colormaps.paraview_color_maps.Hue_L60},
           backend=KB,
           use_cm=True, color_func=lambda u, v: u,
           title=r"Möbius \, strip",
           wireframe=True, wf_n1=20, wf_rendering_kw={"width": 0.004})

    Riemann surfaces of the real part of the multivalued function `z**n`,
    using Plotly:

    .. plotly::
       :context: reset

       from sympy import symbols, sqrt, re, im, pi, atan2, sin, cos, I
       from spb import plot3d_parametric_surface, PB
       r, theta, x, y = symbols("r, theta, x, y", real=True)
       mag = lambda z: sqrt(re(z)**2 + im(z)**2)
       phase = lambda z, k=0: atan2(im(z), re(z)) + 2 * k * pi
       n = 2 # exponent (integer)
       z = x + I * y # cartesian
       d = {x: r * cos(theta), y: r * sin(theta)} # cartesian to polar
       branches = [(mag(z)**(1 / n) * cos(phase(z, i) / n)).subs(d)
           for i in range(n)]
       exprs = [(r * cos(theta), r * sin(theta), rb) for rb in branches]
       plot3d_parametric_surface(*exprs, (r, 0, 3), (theta, -pi, pi),
           backend=PB, wireframe=True, wf_n2=20, zlabel="f(z)",
           label=["branch %s" % (i + 1) for i in range(len(branches))])

    Plotting a numerical function instead of a symbolic expression.

    .. k3d-screenshot::
       :camera: 5.3, -7.6, 4, -0.2, -0.9, -1.3, -0.25, 0.4, 0.9

       from spb import *
       import numpy as np
       fx = lambda u, v: (4 + np.cos(u)) * np.cos(v)
       fy = lambda u, v: (4 + np.cos(u)) * np.sin(v)
       fz = lambda u, v: np.sin(u)
       plot3d_parametric_surface(fx, fy, fz, ("u", 0, 2 * np.pi),
           ("v", 0, 2 * np.pi), zlim=(-2.5, 2.5), title="Torus",
           backend=KB, grid=False)

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
       plot3d_parametric_surface((
               exp(u) * cos(v - alpha) / 2 + exp(-u) * cos(v + alpha) / 2,
               exp(u) * sin(v - alpha) / 2 + exp(-u) * sin(v + alpha) / 2,
               cos(alpha) * u + sin(alpha) * v
           ),
           prange(u, -up, up), prange(v, 0, vp * pi),
           backend=KB,
           use_cm=True,
           color_func=lambda u, v: v,
           rendering_kw={"color_map": k3d.colormaps.paraview_color_maps.Hue_L60},
           wireframe=True, wf_n2=15, wf_rendering_kw={"width": 0.005},
           grid=False, n=50, use_latex=False,
           params={
               alpha: (0, 0, pi),
               up: (1, 0, 2),
               vp: (2, 0, 2),
           },
           title=r"Catenoid \, to \, Right \, Helicoid \, Transformation")

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. Note that the plot's
    creation might be slow due to the wireframe lines.

    .. panel-screenshot::

       from sympy import *
       from spb import *
       import param
       n, u, v = symbols("n, u, v")
       x = v * cos(u)
       y = v * sin(u)
       z = sin(n * u)
       plot3d_parametric_surface(
           (x, y, z, (u, 0, 2*pi), (v, -1, 0)),
           params = {
               n: param.Integer(3, label="n")
           },
           backend=KB,
           use_cm=True,
           title=r"Plücker's \, conoid",
           wireframe=True,
           wf_rendering_kw={"width": 0.004},
           wf_n1=75, wf_n2=6, imodule="panel"
       )

    See Also
    ========

    plot3d, plot3d_parametric_list, plot3d_spherical,
    plot3d_revolution, plot3d_implicit, plot3d_list

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 3, 2, **kwargs)
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)
    surfaces = []
    indeces = []

    for i, pe in enumerate(plot_expr):
        indeces.append(len(surfaces))
        e1, e2, e3, r1, r2, label, rendering_kw = pe
        surfaces.extend(
            surface_parametric(
                e1, e2, e3, r1, r2, label, rendering_kw, **kwargs))
    actual_surfaces = [s for i, s in enumerate(surfaces) if i in indeces]
    _set_labels(actual_surfaces, global_labels, global_rendering_kw)
    return graphics(*surfaces, **kwargs)


def plot3d_spherical(*args, **kwargs):
    """
    Plots a radius as a function of the spherical coordinates theta and phi.

    Typical usage examples are in the followings:

    - Plotting a single expression.:

      .. code-block::

         plot3d_spherical(r, range_theta, range_phi, **kwargs)

    - Plotting multiple expressions with the same ranges.:

      .. code-block::

         plot3d_spherical(r1, r2, range_theta, range_phi, **kwargs)

    - Plotting multiple expressions with different ranges, custom labels and
      rendering options:

      .. code-block::

         plot3d_spherical(
            (r1, range_theta1, range_phi1, label1 [opt], rendering_kw1 [opt]),
            (r2, range_theta2, range_phi2, label2 [opt], rendering_kw2 [opt]),
            ..., **kwargs)

    Note: it is important to specify both the ranges.

    Refer to :func:`~spb.graphics.functions_3d.surface_spherical` for a full
    list of keyword arguments to customize the appearances of surfaces.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Parameters
    ==========

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of expr will be used. The number of labels must be
        equal to the number of expressions.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the plotting
        library (backend) manual for more informations. If a list of
        dictionaries is provided, the number of dictionaries must be equal
        to the number of expressions.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi, Ynm, re, lambdify
       >>> from spb import plot3d_spherical
       >>> theta, phi = symbols('theta phi')

    Sphere cap:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_spherical(1, (theta, 0, 0.7 * pi), (phi, 0, 1.8 * pi))
       Plot object containing:
       [0]: parametric cartesian surface: (sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta)) for theta over (0.0, 2.199114857512855) and phi over (0.0, 5.654866776461628)

    Plot real spherical harmonics, highlighting the regions in which the
    real part is positive and negative, using Plotly:

    .. plotly::
       :context: reset

       from sympy import symbols, sin, pi, Ynm, re, lambdify
       from spb import plot3d_spherical, PB
       theta, phi = symbols('theta phi')
       r = re(Ynm(3, 3, theta, phi).expand(func=True).rewrite(sin).expand())
       plot3d_spherical(
           abs(r), (theta, 0, pi), (phi, 0, 2 * pi), "radius",
           use_cm=True, n2=200, backend=PB,
           color_func=lambdify([theta, phi], r))

    Multiple surfaces with wireframe lines, using Plotly. Note that activating
    the wireframe option might add a considerable overhead during the plot's
    creation.

    .. plotly::

       from sympy import symbols, sin, pi
       from spb import plot3d_spherical, PB
       theta, phi = symbols('theta phi')
       r1 = 1
       r2 = 1.5 + sin(5 * phi) * sin(10 * theta) / 10
       plot3d_spherical(r1, r2, (theta, 0, pi / 2), (phi, 0.35 * pi, 2 * pi),
           wireframe=True, wf_n2=25, backend=PB, label=["r1", "r2"])

    Interactive-widget plot of real spherical harmonics, highlighting the
    regions in which the real part is positive and negative.
    Note that the plot's creation and update might be slow and that
    it must be ``m < n`` at all times.
    Refer to the interactive sub-module documentation to learn more about the
    ``params`` dictionary.

    .. panel-screenshot::

       from sympy import *
       from spb import *
       import param
       n, m = symbols("n, m")
       phi, theta = symbols("phi, theta", real=True)
       r = re(Ynm(n, m, theta, phi).expand(func=True).rewrite(sin).expand())
       plot3d_spherical(
           abs(r), (theta, 0, pi), (phi, 0, 2*pi),
           params = {
               n: param.Integer(2, label="n"),
               m: param.Integer(0, label="m"),
           },
           force_real_eval=True,
           use_cm=True, color_func=r,
           backend=KB, imodule="panel")

    See Also
    ========

    plot3d, plot3d_parametric_list, plot3d_parametric_surface,
    plot3d_revolution, plot3d_implicit, plot3d_list

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 2, **kwargs)
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)
    surfaces = []
    indeces = []

    for i, pe in enumerate(plot_expr):
        indeces.append(len(surfaces))
        expr, r1, r2, label, rendering_kw = pe
        surfaces.extend(
            surface_spherical(expr, r1, r2, label, rendering_kw, **kwargs))
    actual_surfaces = [s for i, s in enumerate(surfaces) if i in indeces]
    _set_labels(actual_surfaces, global_labels, global_rendering_kw)
    return graphics(*surfaces, **kwargs)


def plot3d_implicit(*args, **kwargs):
    """
    Plots an isosurface of a function.

    Typical usage examples are in the followings:

    - Plotting a single expression:

      .. code-block::

         plot3d_implicit(
            expr, range_x, range_y, range_z, rendering_kw [optional], **kwargs)

    - Plotting a multiple expression over the same range:

      .. code-block::

         plot3d_implicit(
            expr1, expr2, range_x, range_y, range_z,
            rendering_kw [optional], **kwargs)`

    - Plotting a multiple expression with different range and
      rendering options:

      .. code-block::

         plot3d_implicit(
            (expr1, range_x1, range_y1, range_z1, rendering_kw1 [opt]),
            (expr2, range_x2, range_y2, range_z2, rendering_kw2 [opt]),
            **kwargs)`

    Refer to :func:`~spb.graphics.functions_3d.implicit_3d` for a full
    list of keyword arguments to customize the appearances of surfaces.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

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

    Examples
    ========

    .. plotly::
       :context: reset

       from sympy import symbols
       from spb import plot3d_implicit, PB, KB
       x, y, z = symbols('x, y, z')
       plot3d_implicit(
           x**2 + y**3 - z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2), backend=PB)

    .. plotly::
       :context: close-figs

       plot3d_implicit(
           x**4 + y**4 + z**4 - (x**2 + y**2 + z**2 - 0.3),
           (x, -2, 2), (y, -2, 2), (z, -2, 2), backend=PB)

    Visualize the isocontours from `isomin=0` to `isomax=2` by providing a
    ``rendering_kw`` dictionary:

    .. plotly::
       :context: close-figs

       plot3d_implicit(
           1/x**2 - 1/y**2 + 1/z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2),
           {
               "isomin": 0, "isomax": 2,
               "colorscale":"aggrnyl", "showscale":True
           },
           backend=PB
       )

    See Also
    ========

    plot3d, plot3d_parametric_list, plot3d_parametric_surface,
    plot3d_spherical, plot3d_revolution, plot3d_list

    """
    if kwargs.pop("params", None) is not None:
        raise NotImplementedError(
            "plot3d_implicit doesn't support interactive widgets.")

    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 3, **kwargs)
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)
    surfaces = []
    indeces = []

    for i, pe in enumerate(plot_expr):
        indeces.append(len(surfaces))
        expr, r1, r2, r3, label, rendering_kw = pe
        surfaces.extend(
            implicit_3d(expr, r1, r2, r3, label, rendering_kw, **kwargs))
    actual_surfaces = [s for i, s in enumerate(surfaces) if i in indeces]
    _set_labels(actual_surfaces, global_labels, global_rendering_kw)
    return graphics(*surfaces, **kwargs)


def plot3d_revolution(
    curve, range_t, range_phi=None, axis=(0, 0),
    parallel_axis="z", show_curve=False, curve_kw={}, **kwargs
):
    """Generate a surface of revolution by rotating a curve around an axis of
    rotation.

    Refer to :func:`~spb.graphics.functions_3d.surface_revolution` for a full
    list of keyword arguments to customize the appearances of surfaces.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Parameters
    ==========

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of expr will be used. The number of labels must be
        equal to the number of expressions.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the plotting
        library (backend) manual for more informations. If a list of
        dictionaries is provided, the number of dictionaries must be equal
        to the number of expressions.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi
       >>> from spb import plot3d_revolution
       >>> t, phi = symbols('t phi')

    Revolve a function around the z axis:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_revolution(
       ...     cos(t), (t, 0, pi),
       ...     # use a color map on the surface to indicate the azimuthal angle
       ...     use_cm=True, color_func=lambda t, phi: phi,
       ...     rendering_kw={"alpha": 0.6, "cmap": "twilight"},
       ...     # indicates the azimuthal angle on the colorbar label
       ...     label=r"$\phi$ [rad]",
       ...     show_curve=True,
       ...     # this dictionary will be passes to plot3d_parametric_line in
       ...     # order to draw the initial curve
       ...     curve_kw=dict(rendering_kw={"color": "r", "label": "cos(t)"}),
       ...     # activate the wireframe to visualize the parameterization
       ...     wireframe=True, wf_n1=15, wf_n2=15,
       ...     wf_rendering_kw={"lw": 0.5, "alpha": 0.75})  # doctest: +SKIP

    Revolve the same function around an axis parallel to the x axis, using
    Plotly:

    .. plotly::
       :context: reset

       from sympy import symbols, cos, sin, pi
       from spb import plot3d_revolution, PB
       t, phi = symbols('t phi')
       plot3d_revolution(
           cos(t), (t, 0, pi), parallel_axis="x", axis=(1, 0),
           backend=PB, use_cm=True, color_func=lambda t, phi: phi,
           rendering_kw={"colorscale": "twilight"},
           label="phi [rad]",
           show_curve=True,
           curve_kw=dict(rendering_kw={"line": {"color": "red", "width": 8},
               "name": "cos(t)"}),
           wireframe=True, wf_n1=15, wf_n2=15,
           wf_rendering_kw={"line_width": 1})

    Revolve a 2D parametric circle around the z axis:

    .. k3d-screenshot::
       :camera: 4.3, -5.82, 4.95, 0.4, -0.25, -0.67, -0.32, 0.5, 0.8

       from sympy import *
       from spb import *
       t = symbols("t")
       circle = (3 + cos(t), sin(t))
       plot3d_revolution(circle, (t, 0, 2 * pi),
           backend=KB, show_curve=True,
           rendering_kw={"opacity": 0.65},
           curve_kw={"rendering_kw": {"width": 0.05}})

    Revolve a 3D parametric curve around the z axis for a given azimuthal
    angle, using Plotly:

    .. plotly::
       :context: close-figs

       plot3d_revolution(
           (cos(t), sin(t), t), (t, 0, 2*pi), (phi, 0, pi),
           use_cm=True, color_func=lambda t, phi: t, label="t [rad]",
           show_curve=True, backend=PB, aspect="cube",
           wireframe=True, wf_n1=2, wf_n2=5)

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
       plot3d_revolution(
           (t, cos(u * t), t**2), prange(t, 0, v), prange(phi, 0, w*pi),
           axis=(1, 0.2),
           params={
               u: (2.5, 0, 6),
               v: (2, 0, 3),
               w: (2, 0, 2)
           }, n=50, backend=KB, force_real_eval=True,
           wireframe=True, wf_n1=15, wf_n2=15,
           wf_rendering_kw={"width": 0.004},
           show_curve=True, curve_kw={"rendering_kw": {"width": 0.025}},
           use_latex=False)

    See Also
    ========

    plot3d, plot3d_parametric_list, plot3d_parametric_surface,
    plot3d_spherical, plot3d_implicit, plot3d_list

    """
    surfaces = surface_revolution(
        curve, range_t, range_phi,
        axis=axis, parallel_axis=parallel_axis, show_curve=show_curve,
        curve_kw=curve_kw, **kwargs)
    return graphics(*surfaces, **kwargs)


def plot3d_list(*args, **kwargs):
    """Plots lists of coordinates (ie, lists of numbers) in 3D space.

    Typical usage examples are in the followings:

    - Plotting coordinates of a single function:

      .. code-block::

         plot3d_list(x, y, **kwargs)

    - Plotting coordinates of multiple functions, adding custom labels and
      rendering options:

      .. code-block::

         plot3d_list(
            (x1, y1, z1, label1 [opt], rendering_kw1 [opt]),
            (x2, y2, z1, label2 [opt], rendering_kw2 [opt]),
            ..., **kwargs)

    Refer to :func:`~spb.graphics.functions_3d.list_3d` for a full
    list of keyword arguments to customize the appearances of lines.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import *
       >>> from spb import plot3d_list
       >>> import numpy as np

    Plot the coordinates of a single function:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> z = np.linspace(0, 6*np.pi, 100)
       >>> x = z * np.cos(z)
       >>> y = z * np.sin(z)
       >>> plot3d_list(x, y, z)
       Plot object containing:
       [0]: 3D list plot

    Plotting multiple functions with custom rendering keywords:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_list(
       ...     (x, y, z, "A"),
       ...     (x, y, -z, "B", {"linestyle": "--"}))
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
       p1 = plot3d_list(x, y, z,
           show=False, scatter=False)
       p2 = plot3d_list(
           [t * cos(t)], [t * sin(t)], [t],
           params={t: (3*pi, 0, 6*pi)},
           backend=PB, show=False, scatter=True, use_latex=False,
           imodule="panel")
       (p2 + p1).show()

    See Also
    ========

    plot3d, plot3d_parametric_list, plot3d_parametric_surface,
    plot3d_spherical, plot3d_revolution, plot3d_implicit

    """
    g_labels = kwargs.pop("label", [])
    g_rendering_kw = kwargs.pop("rendering_kw", None)
    series = []

    def is_tuple(t):
        # verify that t is a tuple of the form (x, y, label [opt],
        # rendering_kw [opt])
        if hasattr(t, "__iter__"):
            if isinstance(t, (str, dict)):
                return False
            if (len(t) >= 3) and all(hasattr(t[i], "__iter__") for i in [0, 1, 2]):
                return True
        return False

    if not any(is_tuple(e) for e in args):
        # in case we are plotting a single line
        args = [args]

    for a in args:
        if not isinstance(a, (list, tuple)):
            raise TypeError(
                "Each argument must be a list or tuple.\n"
                "Received type(a) = {}".format(type(a)))
        if (len(a) < 3) or (len(a) > 5):
            raise ValueError(
                "Each argument must contain from 3 to 5 elements.\n"
                "Received {} elements.".format(len(a)))
        label = [b for b in a if isinstance(b, str)]
        label = "" if not label else label[0]
        rendering_kw = [b for b in a if isinstance(b, dict)]
        rendering_kw = None if not rendering_kw else rendering_kw[0]
        series.extend(
            list_3d(*a[:3], label, rendering_kw, **kwargs))

    _set_labels(series, g_labels, g_rendering_kw)
    return graphics(*series, **kwargs)
