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

from spb.defaults import TWO_D_B
from spb.graphics import (
    graphics, line, line_parametric_2d, line_parametric_3d,
    surface, surface_parametric, surface_revolution, surface_spherical,
    contour, implicit_2d, implicit_3d, list_2d, list_3d, geometry
)
from spb.series import (
    Parametric2DLineSeries, PlaneSeries, GenericDataSeries,
    LineOver1DRangeSeries
)
from spb.utils import (
    _plot_sympify, _check_arguments, _unpack_args, _instantiate_backend,
)
from sympy import latex, Tuple, Symbol, oo, cos, sin
from sympy.external import import_module


def _set_labels(series, labels, rendering_kw):
    """Apply the label keyword argument to the series.
    """
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    if len(labels) > 0:
        if len(series) != len(labels):
            raise ValueError(
                "The number of labels must be equal to the "
                "number of expressions being plotted.\nReceived "
                f"{len(series)} expressions and {len(labels)} labels")

        for s, l in zip(series, labels):
            s.label = l

    if rendering_kw is not None:
        if isinstance(rendering_kw, dict):
            rendering_kw = [rendering_kw]
        if len(rendering_kw) == 1:
            rendering_kw *= len(series)
        elif len(series) != len(rendering_kw):
            raise ValueError(
                "The number of rendering dictionaries must be "
                "equal to the number of expressions being plotted.\nReceived "
                f"{len(series)} expressions and {len(rendering_kw)} labels")
        for s, r in zip(series, rendering_kw):
            s.rendering_kw = r


def _create_series(series_type, plot_expr, **kwargs):
    series = []
    for args in plot_expr:
        kw = kwargs.copy()
        if args[-1] is not None:
            kw["rendering_kw"] = args[-1]
        series.append(series_type(*args[:-1], **kw))
    return series


def _create_generic_data_series(**kwargs):
    keywords = ["annotations", "markers", "fill", "rectangles"]
    series = []
    for kw in keywords:
        dictionaries = kwargs.pop(kw, [])
        if isinstance(dictionaries, dict):
            dictionaries = [dictionaries]
        for d in dictionaries:
            args = d.pop("args", [])
            series.append(GenericDataSeries(kw, *args, **d))
    return series


def plot(*args, **kwargs):
    """Plots a function of a single variable as a curve.

    Typical usage examples are in the followings:

    - Plotting a single expression with the default range:

      .. code-block::

         plot(expr, **kwargs)

    - Plotting a single expression with a custom range, custom label and
      rendering options.

      .. code-block::

         plot(expr, range, label [opt], rendering_kw [opt], **kwargs)

    - Plotting multiple expressions with a single range.

      .. code-block::

         plot(expr1, expr2, ..., range, **kwargs)

    - Plotting multiple expressions with different ranges, custom labels and
      rendering options.

      .. code-block::

         plot(
            (expr1, range1, label1 [opt], rendering_kw1 [opt]),
            (expr2, range2, label2 [opt], rendering_kw2 [opt]),
            ..., **kwargs)

    Refer to :func:`~spb.graphics.functions_2d.line` for a full list of
    keyword arguments to customize the appearances of lines.

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
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, sin, pi, tan, exp, cos, log, floor
       >>> from spb import plot
       >>> x, y = symbols('x, y')

    Single Plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x**2, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-5.0, 5.0)

    Multiple functions over the same range with custom rendering options:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x, log(x), exp(x), (x, -3, 3), aspect="equal", ylim=(-3, 3),
       ...    rendering_kw=[{}, {"linestyle": "--"}, {"linestyle": ":"}])
       Plot object containing:
       [0]: cartesian line: x for x over (-3.0, 3.0)
       [1]: cartesian line: log(x) for x over (-3.0, 3.0)
       [2]: cartesian line: exp(x) for x over (-3.0, 3.0)

    Plotting a summation in which the free symbol of the expression is not
    used in the lower/upper bounds:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import Sum, oo, latex
       >>> expr = Sum(1 / x ** y, (x, 1, oo))
       >>> plot(expr, (y, 2, 10), sum_bound=1e03, title="$%s$" % latex(expr))
       Plot object containing:
       [0]: cartesian line: Sum(x**(-y), (x, 1, 1000)) for y over (2.0, 10.0)

    Plotting a summation in which the free symbol of the expression is
    used in the lower/upper bounds. Here, the discretization variable must
    assume integer values:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = Sum(1 / x, (x, 1, y))
       >>> plot(expr, (y, 2, 10), adaptive=False,
       ...     scatter=True, is_filled=True, title="$%s$" % latex(expr))
       Plot object containing:
       [0]: cartesian line: Sum(1/x, (x, 1, y)) for y over (2.0, 10.0)

    Using an adaptive algorithm, detect and plot vertical lines at
    singularities. Also, apply a transformation function to the discretized
    domain in order to convert radians to degrees:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> plot(tan(x), (x, -1.5*pi, 1.5*pi),
       ...      adaptive=True, adaptive_goal=0.001,
       ...      detect_poles="symbolic", tx=np.rad2deg, ylim=(-7, 7),
       ...      xlabel="x [deg]", grid=False)
       Plot object containing:
       [0]: cartesian line: tan(x) for x over (-4.71238898038469, 4.71238898038469)

    Introducing discontinuities by excluding specified points:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(floor(x) / x, (x, -3.25, 3.25), ylim=(-1, 5),
       ...     exclude=list(range(-4, 5)))
       Plot object containing:
       [0]: cartesian line: floor(x)/x for x over (-3.25, 3.25)

    Advanced example showing:

    * detect singularities by setting ``adaptive=False`` (better performance),
      increasing the number of discretization points (in order to have
      'vertical' segments on the lines) and reducing the threshold for the
      singularity-detection algorithm.
    * application of color function.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> expr = 1 / cos(10 * x) + 5 * sin(x)
       >>> def cf(x, y):
       ...     # map a colormap to the distance from the origin
       ...     d = np.sqrt(x**2 + y**2)
       ...     # visibility of the plot is limited: ylim=(-10, 10). However,
       ...     # some of the y-values computed by the function are much higher
       ...     # (or lower). Filter them out in order to have the entire
       ...     # colormap spectrum visible in the plot.
       ...     offset = 12 # 12 > 10 (safety margin)
       ...     d[(y > offset) | (y < -offset)] = 0
       ...     return d
       >>> p1 = plot(expr, (x, -5, 5),
       ...         "distance from (0, 0)", {"cmap": "plasma"},
       ...         ylim=(-10, 10), adaptive=False, detect_poles=True, n=3e04,
       ...         eps=1e-04, color_func=cf, title="$%s$" % latex(expr))

    Combining multiple plots together:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> p2 = plot(5 * sin(x), (x, -5, 5), {"linestyle": "--"}, show=False)
       >>> (p1 + p2).show()

    Plotting a numerical function instead of a symbolic expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> plot(lambda t: np.cos(np.exp(-t)), ("t", -pi, 0))   # doctest: +SKIP


    Interactive-widget plot of an oscillator. Refer to the interactive
    sub-module documentation to learn more about the ``params`` dictionary.
    This plot illustrates:

    * plotting multiple expressions, each one with its own label and
      rendering options.
    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).
    * the use of a parametric title, specified with a tuple of the form:
      ``(title_str, param_symbol1, ...)``, where:

      * ``title_str`` must be a formatted string, for example:
        ``"test = {:.2f}"``.
      * ``param_symbol1, ...`` must be a symbol or a symbolic expression
        whose free symbols are contained in the ``params`` dictionary.

    .. panel-screenshot::
       :small-size: 800, 625

       from sympy import *
       from spb import *
       x, a, b, c, n = symbols("x, a, b, c, n")
       plot(
           (cos(a * x + b) * exp(-c * x), "oscillator"),
           (exp(-c * x), "upper limit", {"linestyle": ":"}),
           (-exp(-c * x), "lower limit", {"linestyle": ":"}),
           prange(x, 0, n * pi),
           params={
               a: (1, 0, 10),     # frequency
               b: (0, 0, 2 * pi), # phase
               c: (0.25, 0, 1),   # damping
               n: (2, 0, 4)       # multiple of pi
           },
           ylim=(-1.25, 1.25),
           title=("Frequency = {:.2f} Hz", a)
       )

    See Also
    ========

    plot_implicit, plot_polar, plot_parametric, plot_list, plot_contour,
    plot_geometry, plot_piecewise

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 1, **kwargs)
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)
    lines = []

    for pe in plot_expr:
        expr, r, label, rendering_kw = pe
        lines.extend(line(expr, r, label, rendering_kw, **kwargs))
    _set_labels(lines, global_labels, global_rendering_kw)
    gs = _create_generic_data_series(**kwargs)
    return graphics(*lines, gs, **kwargs)


def plot_parametric(*args, **kwargs):
    """
    Plots a 2D parametric curve.

    Typical usage examples are in the followings:

    - Plotting a single parametric curve with a range:

      .. code-block::

         plot_parametric(expr_x, expr_y, range)

    - Plotting multiple parametric curves with the same range:

      .. code-block::

         plot_parametric(
            (expr_x1, expr_y1), (expr_x2, expr_y2), ..., range)

    - Plotting multiple curves with different ranges, custom labels and
      rendering options:

      .. code-block::

         plot_parametric(
            (expr_x1, expr_y1, range1, label1 [opt], rendering_kw1 [opt]),
            (expr_x2, expr_y2, range2, label2 [opt], rendering_kw2 [opt]), ...)

    Refer to :func:`~spb.graphics.functions_2d.line_parametric_2d` for a full
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

       >>> from sympy import symbols, cos, sin, pi, floor, log
       >>> from spb import plot_parametric
       >>> t, u, v = symbols('t, u, v')

    A parametric plot of a single expression (a Hypotrochoid using an equal
    aspect ratio):

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric(
       ...      2 * cos(u) + 5 * cos(2 * u / 3),
       ...      2 * sin(u) - 5 * sin(2 * u / 3),
       ...      (u, 0, 6 * pi), aspect="equal")
       Plot object containing:
       [0]: parametric cartesian line: (5*cos(2*u/3) + 2*cos(u), -5*sin(2*u/3) + 2*sin(u)) for u over (0.0, 18.84955592153876)

    A parametric plot with multiple expressions with the same range with solid
    line colors:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((2 * cos(t), sin(t)), (cos(t), 2 * sin(t)),
       ...    (t, 0, 2*pi), use_cm=False)
       Plot object containing:
       [0]: parametric cartesian line: (2*cos(t), sin(t)) for t over (0.0, 6.283185307179586)
       [1]: parametric cartesian line: (cos(t), 2*sin(t)) for t over (0.0, 6.283185307179586)

    A parametric plot with multiple expressions with different ranges,
    custom labels, custom rendering options and a transformation function
    applied to the discretized parameter to convert radians to degrees:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> plot_parametric(
       ...      (3 * cos(u), 3 * sin(u), (u, 0, 2 * pi), "u [deg]", {"lw": 3}),
       ...      (3 * cos(2 * v), 5 * sin(4 * v), (v, 0, pi), "v [deg]"),
       ...      aspect="equal", tp=np.rad2deg)
       Plot object containing:
       [0]: parametric cartesian line: (3*cos(u), 3*sin(u)) for u over (0.0, 6.283185307179586)
       [1]: parametric cartesian line: (3*cos(2*v), 5*sin(4*v)) for v over (0.0, 3.141592653589793)

    Introducing discontinuities by excluding specified points:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> e1 = log(floor(t))*cos(t)
       >>> e2 = log(floor(t))*sin(t)
       >>> plot_parametric(e1, e2, (t, 1, 4*pi),
       ...     exclude=list(range(1, 13)), grid=False)
       Plot object containing:
       [0]: parametric cartesian line: (log(floor(t))*cos(t), log(floor(t))*sin(t)) for t over (1.0, 12.566370614359172)

    Plotting a numerical function instead of a symbolic expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> fx = lambda t: np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12)**5)
       >>> fy = lambda t: np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12)**5)
       >>> p = plot_parametric(fx, fy, ("t", 0, 12 * pi),
       ...     title="Butterfly Curve", use_cm=False, n=2000)

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 600

       from sympy import *
       from spb import *
       x, a, s, e = symbols("x a s, e")
       plot_parametric(
           cos(a * x), sin(x), prange(x, s*pi, e*pi),
           params={
               a: (0.5, 0, 2),
               s: (0, 0, 2),
               e: (2, 0, 2),
           },
           aspect="equal",
           xlim=(-1.25, 1.25), ylim=(-1.25, 1.25)
       )

    See Also
    ========

    plot, plot_implicit, plot_polar, plot_list, plot_contour,
    plot_geometry, plot_piecewise

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 2, 1, **kwargs)
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)
    lines = []

    for pe in plot_expr:
        e1, e2, r, label, rendering_kw = pe
        lines.extend(
            line_parametric_2d(e1, e2, r, label, rendering_kw, **kwargs))
    _set_labels(lines, global_labels, global_rendering_kw)
    gs = _create_generic_data_series(**kwargs)
    return graphics(*lines, gs, **kwargs)


def plot_parametric_region(*args, **kwargs):
    """
    Plots a 2D parametric region.

    NOTE: this is an experimental plotting function as it only draws lines
    without fills. The resulting visualization might change when new features
    will be implemented.

    Typical usage examples are in the followings:

    - Plotting a single parametric curve with a range:

      .. code-block::

         plot_parametric(expr_x, expr_y, range_u, range_v)

    - Plotting multiple parametric curves with the same range:

      .. code-block::

         plot_parametric((expr_x, expr_y), ..., range_u, range_v)

    - Plotting multiple parametric curves with different ranges:

      .. code-block::

         plot_parametric((expr_x, expr_y, range_u, range_v), ...)

    Parameters
    ==========

    args :
        `expr_x`, `expr_y` : Expr
            The expression representing x and y component, respectively, of
            the parametric function. It can be a:

            * Symbolic expression representing the function of one variable
              to be plotted.
            * Numerical function of one variable, supporting vectorization.
              In this case the following keyword arguments are not supported:
              ``params``.

        `range_u`, `range_v` : (symbol, min, max)
            A 3-tuple denoting the parameter symbols, start and stop. For
            example, `(u, 0, 5), (v, 0, 5)`. If the ranges are not specified,
            then they default to (-10, 10).

            However, if the arguments are specified as
            `(expr_x, expr_y, range_u, range_v), ...`, you must specify the
            ranges for each expressions manually.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines. Refer to the
            plotting library (backend) manual for more informations.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    n : int, optional
        The functions are uniformly sampled at ``n`` number of points.
        Default value to 1000.

    n1, n2 : int, optional
        Number of lines to create along each direction. Default to 10.
        Note: the higher the number, the slower the rendering.

    rkw_u, rkw_v : dict
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines along the u and v
        directions, respectively. These overrides ``rendering_kw`` if provided.
        Refer to the plotting library (backend) manual for more informations.

    show : bool, optional
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the backend.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel : str, optional
        Label for the x-axis or y-axis, respectively.

    xscale, yscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis or y-axis, respectively.
        Default to ``'linear'``.

    xlim, ylim : (float, float), optional
        Denotes the x-axis or y-axis limits, ``(min, max)``, visible in the
        chart.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi, I, re, im, latex
       >>> from spb import plot_parametric_region

    Plot a slice of a ring, applying the same style to all lines:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> r, theta = symbols("r theta")
       >>> p = plot_parametric_region(r * cos(theta), r * sin(theta),
       ...     (r, 1, 2), (theta, 0, 2*pi/3),
       ...     {"color": "k", "linewidth": 0.75},
       ...     n1=5, n2=15, aspect="equal")  # doctest: +SKIP

    Complex mapping, applying to different line styles:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> x, y, z = symbols("x y z")
       >>> f = 1 / z**2
       >>> f_cart = f.subs(z, x + I * y)
       >>> r, i = re(f_cart), im(f_cart)
       >>> n1, n2 = 30, 30
       >>> p = plot_parametric_region(r, i, (x, -2, 2), (y, -2, 2),
       ...     rkw_u={"color": "r", "linewidth": 0.75},
       ...     rkw_v={"color": "b", "linewidth": 0.75},
       ...     n1=20, n2=20, aspect="equal", xlim=(-2, 2), ylim=(-2, 2),
       ...     xlabel="Re", ylabel="Im", title="$f(z)=%s$" % latex(f))

    """
    np = import_module('numpy')
    n1 = kwargs.pop("n1", 10)
    n2 = kwargs.pop("n2", 10)
    rkw_u = kwargs.pop("rkw_u", None)
    rkw_v = kwargs.pop("rkw_v", None)
    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    args = _plot_sympify(args)
    kwargs["adaptive"] = False
    kwargs["use_cm"] = False
    kwargs["legend"] = False
    plot_expr = _check_arguments(args, 2, 2, **kwargs)
    series = []

    if "params" in kwargs.keys():
        raise NotImplementedError

    for pe in plot_expr:
        fx, fy, urange, vrange, lbl, rkw = pe
        u, umin, umax = urange
        v, vmin, vmax = vrange
        new_pe = []
        for uv in np.linspace(float(umin), float(umax), n1):
            new_pe.append((
                fx.subs(u, uv), fy.subs(u, uv),
                (v, vmin, vmax), lbl, rkw if rkw_u is None else rkw_u))
        for vv in np.linspace(float(vmin), float(vmax), n2):
            new_pe.append((
                fx.subs(v, vv), fy.subs(v, vv),
                (u, umin, umax), rkw if rkw_v is None else rkw_v))

        series += _create_series(Parametric2DLineSeries, new_pe, **kwargs)

    Backend = kwargs.pop("backend", TWO_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def _plot3d_plot_contour_helper(threed, *args, **kwargs):
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 2, **kwargs)
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)
    surfaces = []
    func = surface if threed else contour
    indeces = []
    for i, pe in enumerate(plot_expr):
        indeces.append(len(surfaces))
        expr, r1, r2, label, rendering_kw = pe
        surfaces.extend(
            func(expr, r1, r2, label, rendering_kw, **kwargs))
    actual_surfaces = [s for i, s in enumerate(surfaces) if i in indeces]
    _set_labels(actual_surfaces, global_labels, global_rendering_kw)
    return graphics(*surfaces, **kwargs)


def plot_contour(*args, **kwargs):
    """
    Draws contour plot of a function of two variables.

    This function signature is almost identical to :func:`~plot3d`: refer
    to its documentation for a full list of available argument and keyword
    arguments.

    Refer to :func:`~spb.graphics.functions_2d.contour` for a full
    list of keyword arguments to customize the appearances of contours.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Parameters
    ==========

    clabels : bool, optional
        Visualize labels of contour lines. Only works when ``is_filled=False``.
        Default to True. Note that some backend might not implement this
        feature.

    is_filled : bool, optional
        Choose between filled contours or line contours. Default to True
        (filled contours).

    polar_axis : boolean, optional
        If True, attempt to create a plot with polar axis. Default to False,
        which creates a plot with cartesian axis.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, exp, sin, pi, Eq, Add
       >>> from spb import plot_contour
       >>> x, y = symbols('x, y')

    Filled contours of a function of two variables.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_contour(cos((x**2 + y**2)) * exp(-(x**2 + y**2) / 10),
       ...      (x, -5, 5), (y, -5, 5))
       Plot object containing:
       [0]: contour: exp(-x**2/10 - y**2/10)*cos(x**2 + y**2) for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    Line contours of a function of two variables.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = 5 * (cos(x) - 0.2 * sin(y))**2 + 5 * (-0.2 * cos(x) + sin(y))**2
       >>> plot_contour(expr, (x, 0, 2 * pi), (y, 0, 2 * pi), is_filled=False)
       Plot object containing:
       [0]: contour: 5*(-0.2*sin(y) + cos(x))**2 + 5*(sin(y) - 0.2*cos(x))**2 for x over (0.0, 6.283185307179586) and y over (0.0, 6.283185307179586)

    Combining together filled and line contours. Use a custom label on the
    colorbar of the filled contour.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = 5 * (cos(x) - 0.2 * sin(y))**2 + 5 * (-0.2 * cos(x) + sin(y))**2
       >>> p1 = plot_contour(expr, (x, 0, 2 * pi), (y, 0, 2 * pi), "z",
       ...      {"cmap": "coolwarm"}, show=False, grid=False)
       >>> p2 = plot_contour(expr, (x, 0, 2 * pi), (y, 0, 2 * pi),
       ...      {"colors": "k", "cmap": None, "linewidths": 0.75},
       ...      show=False, is_filled=False)
       >>> (p1 + p2).show()

    Visually inspect the solutions of a system of 2 non-linear equations.
    The intersections between the contour lines represent the solutions.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> eq1 = Eq((cos(x) - sin(y) / 2)**2 + 3 * (-sin(x) + cos(y) / 2)**2, 2)
       >>> eq2 = Eq((cos(x) - 2 * sin(y))**2 - (sin(x) + 2 * cos(y))**2, 3)
       >>> plot_contour(eq1.rewrite(Add), eq2.rewrite(Add), {"levels": [0]},
       ...      (x, 0, 2 * pi), (y, 0, 2 * pi), is_filled=False, clabels=False)
       Plot object containing:
       [0]: contour: 3*(-sin(x) + cos(y)/2)**2 + (-sin(y)/2 + cos(x))**2 - 2 for x over (0.0, 6.283185307179586) and y over (0.0, 6.283185307179586)
       [1]: contour: -(sin(x) + 2*cos(y))**2 + (-2*sin(y) + cos(x))**2 - 3 for x over (0.0, 6.283185307179586) and y over (0.0, 6.283185307179586)

    Contour plot with polar axis:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> r, theta = symbols("r, theta")
       >>> plot_contour(sin(2 * r) * cos(theta), (theta, 0, 2*pi), (r, 0, 7),
       ...     {"levels": 100}, polar_axis=True, aspect="equal")
       Plot object containing:
       [0]: contour: sin(2*r)*cos(theta) for theta over (0.0, 6.283185307179586) and r over (0.0, 7.0)

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 600

       from sympy import *
       from spb import *
       x, y, a, b, xp, yp = symbols("x y a b x_p y_p")
       expr = (cos(x) + a * sin(x) * sin(y) - b * sin(x) * cos(y))**2
       plot_contour(expr, prange(x, 0, xp*pi), prange(y, 0, yp * pi),
           params={a: (1, 0, 2), b: (1, 0, 2), xp: (1, 0, 2), yp: (2, 0, 2)},
           grid=False)

    See Also
    ========

    plot, plot_implicit, plot_polar, plot_parametric, plot_list, plot3d,
    plot_geometry, plot_piecewise

    """
    return _plot3d_plot_contour_helper(False, *args, **kwargs)


def plot_implicit(*args, **kwargs):
    """Plot implicit equations / inequalities.

    ``plot_implicit``, by default, generates a contour using a mesh grid of
    fixednumber of points. The greater the number of points, the better the
    results, but also the greater the memory used.
    By setting ``adaptive=True``, interval arithmetic will be used to plot
    functions. If the expression cannot be plotted using interval arithmetic,
    it defaults to generating a contour using a mesh grid. With interval
    arithmetic, the line width can become very small; in those cases, it is
    better to use the mesh grid approach.

    Typical usage examples are in the following:

    - Plot a single expression:

      .. code-block::

         plot_implicit(expr, range_x, range_y)

    - Plot multiple expressions over the same ranges:

      .. code-block::

         plot_implicit(expr1, expr2, range_x, range_y)

    - Plot multiple expressions over different ranges:

      .. code-block::

         plot_implicit(
            (expr1, range_x1, range_y1, label1 [opt]),
            (expr2, range_x2, range_y2, label2 [opt]))

    Refer to :func:`~spb.graphics.functions_2d.implicit_2d` for a full
    list of keyword arguments to customize the appearances of lines and
    regions.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Parameters
    ==========

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of expr will be used. The number of labels must be
        equal to the number of expressions.

    Examples
    ========

    Plot expressions:

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import symbols, Ne, Eq, And, sin, cos, pi, log, latex
        >>> from spb import plot_implicit
        >>> x, y = symbols('x y')

    Providing only the symbol for the horizontal axis:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> p = plot_implicit(x - 1, x)

    Specify both ranges, set the number of discretization points and plot a
    region:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_implicit(y > x**2, (x, -5, 5), (y, -10, 10), n=150, grid=False)
       Plot object containing:
       [0]: Implicit expression: y > x**2 for x over (-5.0, 5.0) and y over (-10.0, 10.0)

    Plot a region using a custom color, highlights the limiting border and
    customize its appearance. In this particular case, the content of
    ``rendering_kw`` will be sent to matplotlib's ``contour`` of ``contourf``
    commands.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = 4 * (cos(x) - sin(y) / 5)**2 + 4 * (-cos(x) / 5 + sin(y))**2
       >>> plot_implicit(expr <= pi, (x, -pi, pi), (y, -pi, pi),
       ...     grid=False, color="gold", border_color="k",
       ...     rendering_kw={"linestyles": "-.", "linewidths": 1})
       Plot object containing:
       [0]: Implicit expression: 4*(-sin(y)/5 + cos(x))**2 + 4*(sin(y) - cos(x)/5)**2 <= pi for x over (-3.141592653589793, 3.141592653589793) and y over (-3.141592653589793, 3.141592653589793)
       [1]: Implicit expression: Eq(-4*(-sin(y)/5 + cos(x))**2 - 4*(sin(y) - cos(x)/5)**2 + pi, 0) for x over (-3.141592653589793, 3.141592653589793) and y over (-3.141592653589793, 3.141592653589793)

    Boolean expressions will be plotted with the adaptive algorithm. Note the
    thin width of lines:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_implicit(
       ...     Eq(y, sin(x)) & (y > 0),
       ...     Eq(y, sin(x)) & (y < 0),
       ...     (x, -2 * pi, 2 * pi), (y, -4, 4))
       Plot object containing:
       [0]: Implicit expression: (y > 0) & Eq(y, sin(x)) for x over (-6.283185307179586, 6.283185307179586) and y over (-4.0, 4.0)
       [1]: Implicit expression: (y < 0) & Eq(y, sin(x)) for x over (-6.283185307179586, 6.283185307179586) and y over (-4.0, 4.0)

    Plotting multiple implicit expressions and setting labels:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> V, t, b, L = symbols("V, t, b, L")
       >>> L_array = [5, 10, 15, 20, 25]
       >>> b_val = 0.0032
       >>> expr = b * V * 0.277 * t - b * L - log(1 + b * V * 0.277 * t)
       >>> expr_list = [expr.subs({b: b_val, L: L_val}) for L_val in L_array]
       >>> labels = ["L = %s" % L_val for L_val in L_array]
       >>> plot_implicit(*expr_list, (t, 0, 3), (V, 0, 1000), label=labels)
       Plot object containing:
       [0]: Implicit expression: Eq(0.0008864*V*t - log(0.0008864*V*t + 1) - 0.016, 0) for t over (0.0, 3.0) and V over (0.0, 1000.0)
       [1]: Implicit expression: Eq(0.0008864*V*t - log(0.0008864*V*t + 1) - 0.032, 0) for t over (0.0, 3.0) and V over (0.0, 1000.0)
       [2]: Implicit expression: Eq(0.0008864*V*t - log(0.0008864*V*t + 1) - 0.048, 0) for t over (0.0, 3.0) and V over (0.0, 1000.0)
       [3]: Implicit expression: Eq(0.0008864*V*t - log(0.0008864*V*t + 1) - 0.064, 0) for t over (0.0, 3.0) and V over (0.0, 1000.0)
       [4]: Implicit expression: Eq(0.0008864*V*t - log(0.0008864*V*t + 1) - 0.08, 0) for t over (0.0, 3.0) and V over (0.0, 1000.0)

    Comparison of similar expressions plotted with different algorithms. Note:

    1. Adaptive algorithm (``adaptive=True``) can be used with any expression,
       but it usually creates lines with variable thickness. The ``depth``
       keyword argument can be used to improve the accuracy, but reduces line
       thickness even further.
    2. Mesh grid algorithm (``adaptive=False``) creates lines with constant
       thickness.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> expr1 = Eq(x * y - 20, 15 * y)
        >>> expr2 = Eq((x - 3) * y - 20, 15 * y)
        >>> expr3 = Eq((x - 6) * y - 20, 15 * y)
        >>> ranges = (x, 15, 30), (y, 0, 50)
        >>> p1 = plot_implicit(expr1, *ranges, adaptive=True, depth=0,
        ...     label="adaptive=True, depth=0", grid=False, show=False)
        >>> p2 = plot_implicit(expr2, *ranges, adaptive=True, depth=1,
        ...     label="adaptive=True, depth=1", grid=False, show=False)
        >>> p3 = plot_implicit(expr3, *ranges, adaptive=False,
        ...     label="adaptive=False", grid=False, show=False)
        >>> (p1 + p2 + p3).show()

    If the expression is plotted with the adaptive algorithm and it produces
    "low-quality" results, maybe it's possible to rewrite it in order to use
    the mesh grid approach (contours). For example:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from spb import plotgrid
        >>> expr = Ne(x*y, 1)
        >>> p1 = plot_implicit(
        ...     expr, (x, -10, 10), (y, -10, 10),
        ...     grid=False, aspect="equal", show=False,
        ...     title="$%s$ : First approach" % latex(expr))
        >>> # plot the entire visible region
        >>> p2 = plot_implicit(
        ...     x < 20, (x, -10, 10), (y, -10, 10),
        ...     show=False, grid=False, aspect="equal",
        ...     title="$%s$ : Second approach" % latex(expr))
        >>> # plot the excluded contour
        >>> p3 = plot_implicit(
        ...     Eq(*expr.args), (x, -10, 10), (y, -10, 10),
        ...     color="w", show_in_legend=False, show=False)
        >>> plotgrid(p1, (p2 + p3), nc=2)  # doctest: +SKIP

    Interactive-widget implicit plot. Refer to the interactive sub-module
    documentation to learn more about the ``params`` dictionary.
    This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 700

       from sympy import *
       from spb import *
       x, y, a, b, c, d, e = symbols("x, y, a, b, c, d, e")
       expr = Eq(a * x**2 - b * x + c, d * y + y**2)
       plot_implicit(expr, (x, -2, 2), prange(y, -e, e),
           params={
               a: (10, -15, 15),
               b: (7, -15, 15),
               c: (3, -15, 15),
               d: (2, -15, 15),
               e: (10, 1, 15),
           }, n=150, ylim=(-10, 10))

    See Also
    ========

    plot, plot_polar, plot_parametric, plot_list, plot_contour,
    plot_geometry, plot_piecewise

    """
    # if the user is plotting a single expression, then he can pass in one
    # or two symbols to sort the axis. Ranges will then be automatically
    # created.
    args = list(args)
    if (len(args) == 2) and isinstance(args[1], Symbol):
        args[1] = Tuple(args[1], -10, 10)
    elif (
        (len(args) >= 3) and isinstance(args[1], Symbol)
        and isinstance(args[2], Symbol)
    ):
        args[1] = Tuple(args[1], -10, 10)
        args[2] = Tuple(args[2], -10, 10)

    args = _plot_sympify(args)
    args = _check_arguments(args, 1, 2, **kwargs)
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)
    color = kwargs.pop("color", None)
    border_color = kwargs.pop("border_color", None)

    series = []
    # attempt to compute the area that should be visible on the plot.
    xmin, xmax, ymin, ymax = oo, -oo, oo, -oo
    for (expr, r1, r2, label, rendering_kw) in args:
        series.extend(
            implicit_2d(
                expr, r1, r2, label, rendering_kw,
                color=color, border_color=border_color, **kwargs))
        s = series[-1]
        if (not s.start_x.free_symbols) and (s.start_x < xmin):
            xmin = s.start_x
        if (not s.end_x.free_symbols) and (s.end_x > xmax):
            xmax = s.end_x
        if (not s.start_y.free_symbols) and (s.start_y < ymin):
            ymin = s.start_y
        if (not s.end_y.free_symbols) and (s.end_y > ymax):
            ymax = s.end_y

    _set_labels(series, global_labels, global_rendering_kw)
    series += _create_generic_data_series(**kwargs)
    if (xmin != oo) and (xmax != -oo):
        kwargs.setdefault("xlim", (xmin, xmax))
    if (ymin != oo) and (ymax != -oo):
        kwargs.setdefault("ylim", (ymin, ymax))
    return graphics(*series, **kwargs)


def plot_polar(*args, **kwargs):
    """The following function creates a 2D polar plot.

    By default, it uses an equal aspect ratio and doesn't apply a colormap.

    Typical usage examples are in the followings:

    - Plotting a single polar curve with a range:

      .. code-block::

         plot_polar(expr, range)

    - Plotting multiple polar curves with the same range:

      .. code-block::

         plot_polar(expr1, expr2, ..., range)

    - Plotting multiple curves with different ranges, custom labels and
      rendering options:

      .. code-block::

         plot_polar(
            (expr1, range1, label1 [opt], rendering_kw1 [opt]),
            (expr2, range2, label2 [opt], rendering_kw2 [opt]), ..., **kwargs)

    This function is going to execute
    :func:`~spb.graphics.functions_2d.line_parametric_2d`. Refer to its
    documentation for a full list of keyword arguments to customize the
    appearances of lines.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Parameters
    ==========

    polar_axis : boolean, optional
        If True, attempt to create a plot with polar axis. Default to False,
        which creates a plot with cartesian axis.

    Examples
    ========

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import symbols, sin, cos, exp, pi
        >>> from spb import plot_polar
        >>> theta = symbols('theta')

    Plot with cartesian axis:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_polar(3 * sin(2 * theta), (theta, 0, 2*pi))
       Plot object containing:
       [0]: parametric cartesian line: (3*sin(2*theta)*cos(theta), 3*sin(theta)*sin(2*theta)) for theta over (0.0, 6.283185307179586)

    Plot with polar axis:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_polar(
       ...     exp(sin(theta)) - 2 * cos(4 * theta), (theta, 0, 2 * pi),
       ...     polar_axis=True)
       Plot object containing:
       [0]: parametric cartesian line: ((exp(sin(theta)) - 2*cos(4*theta))*cos(theta), (exp(sin(theta)) - 2*cos(4*theta))*sin(theta)) for theta over (0.0, 6.283185307179586)

    Interactive-widget plot of Guilloché Pattern. Refer to the interactive
    sub-module documentation to learn more about the ``params`` dictionary.
    This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify the widgets to be
      created by Holoviz's Panel.

    .. panel-screenshot::

       from sympy import *
       from spb import *
       import panel as pn
       a, b, c, d, e, f, theta, tp = symbols("a:f theta tp")
       def func(n):
           t1 = (c + sin(a * theta + d))
           t2 = ((b + sin(b * theta + e)) - (c + sin(a * theta + d)))
           t3 = (f + sin(a * theta + n / pi))
           return t1 + t2 * t3 / 2
       exprs = [func(n) for n in range(20)]
       plot_polar(
           *exprs, prange(theta, 0, tp*pi),
           {"line_color": "black", "line_width": 0.5},
           params={
               a: pn.widgets.IntInput(value=6, name="a"),
               b: pn.widgets.IntInput(value=12, name="b"),
               c: pn.widgets.IntInput(value=18, name="c"),
               d: (4.7, 0, 2*pi),
               e: (1.8, 0, 2*pi),
               f: (3, 0, 5),
               tp: (2, 0, 2)
           },
           layout = "sbl",
           ncols = 1,
           title="Guilloché Pattern Explorer",
           backend=BB,
           legend=False,
           servable=True,
           imodule="panel"
       )

    See Also
    ========

    plot, plot_implicit, plot_parametric, plot_list, plot_contour,
    plot_geometry, plot_piecewise

    """
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)
    kwargs.setdefault("polar_axis", False)
    kwargs.setdefault("aspect", "equal")
    kwargs.setdefault("xlabel", "")
    kwargs.setdefault("ylabel", "")

    polar_axis = kwargs.get("polar_axis", False)
    if polar_axis:
        kwargs.setdefault("is_polar", True)

    kwargs.setdefault("use_cm", False)
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 1, **kwargs)
    lines = []
    # apply polar transformation
    for i, pe in enumerate(plot_expr):
        r = pe[0]
        theta = pe[1][0]
        e1, e2 = (r * cos(theta), r * sin(theta))
        lines.extend(line_parametric_2d(e1, e2, *pe[1:], **kwargs))
    _set_labels(lines, global_labels, global_rendering_kw)
    gs = _create_generic_data_series(**kwargs)
    return graphics(*lines, gs, **kwargs)


def plot_geometry(*args, **kwargs):
    """Plot entities from the sympy.geometry module.

    Typical usage examples are in the following:

    - Plotting a single geometric entity:

      .. code-block::

         plot_geometry(geom, label [opt], rendering_kw [opt])

    - Plotting multiple geometric entities:

      .. code-block::

         plot_geometry(geom1, geom2, ...)

    - Plotting multiple geometric entities, setting custom labels and
      rendering options:

      .. code-block::

         plot_geometry(
            (geom1, label1 [opt], rendering_kw1 [opt]),
            (geom2, label2 [opt], rendering_kw2 [opt]),
            **kwargs)

    Refer to :func:`~spb.graphics.functions_2d.geometry` for a full
    list of keyword arguments to customize the appearances of lines and
    regions.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).


    Examples
    ========

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import (symbols, Circle, Ellipse, Polygon,
        ...      Curve, Segment, Point2D, Point3D, Line3D, Plane,
        ...      Rational, pi, Point, cos, sin)
        >>> from spb import plot_geometry
        >>> x, y, z = symbols('x, y, z')

    Plot a single geometry, customizing its color:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_geometry(
       ...     Ellipse(Point(-3, 2), hradius=3, eccentricity=Rational(4, 5)),
       ...     {"color": "tab:orange"}, grid=False)
       Plot object containing:
       [0]: geometry entity: Ellipse(Point2D(-3, 2), 3, 9/5)

    Plot several numeric geometric entitiesy. By default, circles, ellipses and
    polygons are going to be filled. Plotting Curve objects is the same as
    `plot_parametric`.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_geometry(
       ...      Circle(Point(0, 0), 5),
       ...      Ellipse(Point(-3, 2), hradius=3, eccentricity=Rational(4, 5)),
       ...      Polygon((4, 0), 4, n=5),
       ...      Curve((cos(x), sin(x)), (x, 0, 2 * pi)),
       ...      Segment((-4, -6), (6, 6)),
       ...      Point2D(0, 0))
       Plot object containing:
       [0]: geometry entity: Circle(Point2D(0, 0), 5)
       [1]: geometry entity: Ellipse(Point2D(-3, 2), 3, 9/5)
       [2]: geometry entity: RegularPolygon(Point2D(4, 0), 4, 5, 0)
       [3]: parametric cartesian line: (cos(x), sin(x)) for x over (0.0, 6.283185307179586)
       [4]: geometry entity: Segment2D(Point2D(-4, -6), Point2D(6, 6))
       [5]: geometry entity: Point2D(0, 0)

    Plot several numeric geometric entities defined by numbers only, turn off
    fill. Every entity is represented as a line.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_geometry(
       ...      Circle(Point(0, 0), 5),
       ...      Ellipse(Point(-3, 2), hradius=3, eccentricity=Rational(4, 5)),
       ...      Polygon((4, 0), 4, n=5),
       ...      Curve((cos(x), sin(x)), (x, 0, 2 * pi)),
       ...      Segment((-4, -6), (6, 6)),
       ...      Point2D(0, 0), is_filled=False)
       Plot object containing:
       [0]: geometry entity: Circle(Point2D(0, 0), 5)
       [1]: geometry entity: Ellipse(Point2D(-3, 2), 3, 9/5)
       [2]: geometry entity: RegularPolygon(Point2D(4, 0), 4, 5, 0)
       [3]: parametric cartesian line: (cos(x), sin(x)) for x over (0.0, 6.283185307179586)
       [4]: geometry entity: Segment2D(Point2D(-4, -6), Point2D(6, 6))
       [5]: geometry entity: Point2D(0, 0)

    Plot several symbolic geometric entities. We need to pass in the `params`
    dictionary, which will be used to substitute symbols before numerical
    evaluation. Note: here we also set custom labels:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> a, b, c, d = symbols("a, b, c, d")
       >>> plot_geometry(
       ...      (Polygon((a, b), c, n=d), "triangle"),
       ...      (Polygon((a + 2, b + 3), c, n=d + 1), "square"),
       ...      params = {a: 0, b: 1, c: 2, d: 3})
       Plot object containing:
       [0]: geometry entity: RegularPolygon(Point2D(0, 1), 2, 3, 0)
       [1]: geometry entity: RegularPolygon(Point2D(2, 4), 2, 4, 0)

    Plot 3D geometric entities. Note: when plotting a Plane, we must always
    provide the x/y/z ranges:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_geometry(
       ...      (Point3D(5, 5, 5), "center"),
       ...      (Line3D(Point3D(-2, -3, -4), Point3D(2, 3, 4)), "line"),
       ...      (Plane((0, 0, 0), (1, 1, 1)),
       ...          (x, -5, 5), (y, -4, 4), (z, -10, 10)))
       Plot object containing:
       [0]: geometry entity: Point3D(5, 5, 5)
       [1]: geometry entity: Line3D(Point3D(-2, -3, -4), Point3D(2, 3, 4))
       [2]: plane series: Plane(Point3D(0, 0, 0), (1, 1, 1)) over (x, -5, 5), (y, -4, 4), (z, -10, 10)

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary.

    .. panel-screenshot::
       :small-size: 800, 625

       from sympy import *
       from spb import *
       import panel as pn
       a, b, c, d = symbols("a, b, c, d")
       plot_geometry(
           (Polygon((a, b), c, n=d), "a"),
           (Polygon((a + 2, b + 3), c, n=d + 1), "b"),
           params = {
               a: (0, -1, 1),
               b: (1, -1, 1),
               c: (2, 1, 2),
               d: pn.widgets.IntInput(value=3, start=3, end=8, name="n")
           },
           aspect="equal", is_filled=False,
           xlim=(-2.5, 5.5), ylim=(-3, 6.5), imodule="panel")

    See Also
    ========

    plot, plot_implicit, plot_polar, plot_parametric, plot_list,
    plot_contour, plot_piecewise

    """
    args = _plot_sympify(args)
    global_labels = kwargs.pop("label", [])
    global_rendering_kw = kwargs.pop("rendering_kw", None)

    if not all([isinstance(a, (list, tuple, Tuple)) for a in args]):
        args = [args]

    params = kwargs.pop("params", {})
    is_interactive = False
    if len(params) > 0:
        param = import_module("param")
        ipywidgets = import_module("ipywidgets")
        has_param, has_ipywidgets, has_tuples = False, False, False
        if param and any(isinstance(t, param.Parameter) for t in params.values()):
            has_param = True
        if ipywidgets and any(isinstance(t, ipywidgets.Widget)
            for t in params.values()):
            has_ipywidgets = True
        if any(hasattr(t, "__iter__") for t in params.values()):
            has_tuples = True
        is_interactive = any([has_param, has_ipywidgets, has_tuples])

    series = []
    if is_interactive:
        kwargs["params"] = params
    for a in args:
        exprs, ranges, label, rkw = _unpack_args(*a)

        if len(ranges) > 0:
            # assume it is a plane series
            for e in exprs:
                lbl = str(e) if not label else label
                kw = kwargs.copy()
                kw["rendering_kw"] = rkw
                kw["label"] = lbl
                if not is_interactive and params:
                    e = e.subs(params)
                series.append(PlaneSeries(e, *ranges, **kw))
        else:
            for e in exprs:
                lbl = str(e) if not label else label
                kw = kwargs.copy()
                kw["rendering_kw"] = rkw
                kw["label"] = lbl
                if not is_interactive and params:
                    e = e.subs(params)
                series.extend(geometry(e, **kw))
    _set_labels(series, global_labels, global_rendering_kw)
    return graphics(*series, **kwargs)


def plot_list(*args, **kwargs):
    """Plots lists of coordinates (ie, lists of numbers).

    Typical usage examples are in the followings:

    - Plotting coordinates of a single function:

      .. code-block::

         plot_list(x, y, **kwargs)

    - Plotting coordinates of multiple functions, adding custom labels and
      rendering options:

      .. code-block::

         plot_list(
            (x1, y1, label1 [opt], rendering_kw1 [opt]),
            (x2, y2, label2 [opt], rendering_kw2 [opt]),
            ..., **kwargs)

    Refer to :func:`~spb.graphics.functions_2d.list_2d` for a full
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

       >>> from sympy import symbols, sin, cos
       >>> from spb import plot_list
       >>> x = symbols('x')

    Plot the coordinates of a single function:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> xx = [t / 100 * 6 - 3 for t in list(range(101))]
       >>> yy = [cos(x).evalf(subs={x: t}) for t in xx]
       >>> plot_list(xx, yy)
       Plot object containing:
       [0]: 2D list plot

    Plot individual points with custom labels:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_list(
       ...     ([0], [0], "A"), ([1], [1], "B"), ([2], [0], "C"),
       ...     scatter=True, is_filled=True)
       Plot object containing:
       [0]: 2D list plot
       [1]: 2D list plot
       [2]: 2D list plot

    Scatter plot of the coordinates of multiple functions, with custom
    rendering keywords:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> xx = [t / 70 * 6 - 3 for t in list(range(71))]
       >>> yy1 = [cos(x).evalf(subs={x: t}) for t in xx]
       >>> yy2 = [sin(x).evalf(subs={x: t}) for t in xx]
       >>> plot_list(
       ...     (xx, yy1, "cos"),
       ...     (xx, yy2, "sin", {"marker": "*", "markerfacecolor": None}),
       ...     scatter=True)
       Plot object containing:
       [0]: 2D list plot
       [1]: 2D list plot

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary.

    .. panel-screenshot::
       :small-size: 800, 625

       from sympy import *
       from spb import *
       x, t = symbols("x, t")
       params = {t: (0, 0, 2*pi)}
       # plot trajectories
       p1 = plot_parametric(
           (cos(x), sin(x), (x, 0, 2*pi), {"linestyle": ":"}),
           (cos(2 * x) / 2, sin(2 * x) / 2, (x, 0, pi), {"linestyle": ":"}),
           use_cm=False, aspect="equal", show=False,
           params=params, imodule="panel")
       # plot points
       p2 = plot_list(
           ([cos(t)], [sin(t)], "A"),
           ([cos(2 * t) / 2], [sin(2 * t) / 2], "B"),
           rendering_kw={"marker": "s", "markerfacecolor": None},
           params=params, scatter=True, show=False)
       (p1 + p2).show()

    See Also
    ========

    plot, plot_implicit, plot_polar, plot_parametric, plot_contour,
    plot_geometry, plot_piecewise, plot3d_list

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
            if (len(t) >= 2) and all(hasattr(t[i], "__iter__") for i in [0, 1]):
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
        if (len(a) < 2) or (len(a) > 4):
            raise ValueError(
                "Each argument must contain from 2 to 3 elements.\n"
                "Received {} elements.".format(len(a)))
        label = [b for b in a if isinstance(b, str)]
        label = "" if not label else label[0]
        rendering_kw = [b for b in a if isinstance(b, dict)]
        rendering_kw = None if not rendering_kw else rendering_kw[0]
        series.extend(list_2d(*a[:2], label, rendering_kw, **kwargs))

    _set_labels(series, g_labels, g_rendering_kw)
    return graphics(*series, **kwargs)


def plot_piecewise(*args, **kwargs):
    """Plots univariate piecewise functions.

    Typical usage examples are in the followings:

    - Plotting a single expression with the default range (-10, 10):

      .. code-block::

         plot_piecewise(expr, **kwargs)

    - Plotting multiple expressions with a single range:

      .. code-block::

         plot_piecewise(expr1, expr2, ..., range, **kwargs)

    - Plotting multiple expressions with multiple ranges, custom labels and
      rendering options:

      .. code-block::

         plot_piecewise(
            (expr1, range1, label1 [opt], rendering_kw1 [opt]),
            (expr2, range2, label2 [opt], rendering_kw2 [opt]),
            ..., **kwargs)`

    Refer to :func:`~spb.graphics.functions_2d.line` for a full
    list of keyword arguments to customize the appearances of lines.

    Refer to :func:`~spb.graphics.graphics.graphics` for a full list of
    keyword arguments to customize the appearances of the figure (title,
    axis labels, ...).

    Parameters
    ==========

    dots : boolean
        Wheter to show circular markers at the endpoints. Default to True.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, sin, cos, pi, Heaviside, Piecewise, Eq
       >>> from spb import plot_piecewise
       >>> x = symbols('x')

    Single Plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> f = Piecewise((x**2, x < 2), (5, Eq(x, 2)), (10 - x, True))
       >>> plot_piecewise(f, (x, -2, 5))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-2.0, 1.999999)
       [1]: 2D list plot
       [2]: cartesian line: 10 - x for x over (2.000001, 5.0)
       [3]: 2D list plot
       [4]: 2D list plot

    Single plot without dots (circular markers):

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_piecewise(Heaviside(x, 0).rewrite(Piecewise),
       ...     (x, -10, 10), dots=False)
       Plot object containing:
       [0]: cartesian line: 0 for x over (-10.0, 0.0)
       [1]: cartesian line: 1 for x over (1e-06, 10.0)

    Plot multiple expressions in which the second piecewise expression has
    a dotted line style. Use the ``label`` keyword argument to set the
    appropriate entries for the legend:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_piecewise(
       ...   (Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10)),
       ...   (Piecewise(
       ...      (sin(x), x < -5),
       ...      (cos(x), x > 5),
       ...      (1 / x, True)), (x, -8, 8), {"linestyle": ":"}),
       ...   ylim=(-2, 2), detect_poles=True, legend=True, label=["A", "B"])
       Plot object containing:
       [0]: cartesian line: 0 for x over (-10.0, 0.0)
       [1]: cartesian line: 1 for x over (1e-06, 10.0)
       [2]: 2D list plot
       [3]: 2D list plot
       [4]: cartesian line: sin(x) for x over (-8.0, -5.000001)
       [5]: 2D list plot
       [6]: cartesian line: cos(x) for x over (5.000001, 8.0)
       [7]: 2D list plot
       [8]: cartesian line: 1/x for x over (-5.0, 5.0)
       [9]: 2D list plot
       [10]: 2D list plot

    See Also
    ========

    plot, plot_implicit, plot_polar, plot_parametric, plot_list,
    plot_contour, plot_geometry

    """
    if kwargs.pop("params", None) is not None:
        raise NotImplementedError(
            "plot_piecewise doesn't support interactive widgets.")

    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 1)
    if any(callable(p[0]) for p in plot_expr):
        raise TypeError("plot_piecewise requires symbolic expressions.")
    free = set()
    for p in plot_expr:
        free |= p[0].free_symbols
    x = free.pop() if free else Symbol("x")

    fx = lambda use_latex: x.name if not use_latex else latex(x)
    wrap = lambda use_latex: "f(%s)" if not use_latex else r"f\left(%s\right)"
    fy = lambda use_latex: wrap(use_latex) % fx(use_latex)
    kwargs.setdefault("xlabel", fx)
    kwargs.setdefault("ylabel", fy)
    kwargs.setdefault("legend", False)
    kwargs["process_piecewise"] = True

    labels = kwargs.pop("label", [])
    kwargs.pop("rendering_kw", None)
    if isinstance(labels, str):
        labels = [labels] * len(plot_expr)

    # NOTE: rendering_kw keyword argument is not implemented in this function
    # because it would override the optimal settings chosen by the backend.
    # If a user want to set custom rendering keywords, just use the notation
    # (expr, range, label [optional], rendering_kw [optional])
    color_series_dict = dict()
    for i, a in enumerate(plot_expr):
        expr, r, lbl, rkw = a
        series = line(expr, r, lbl, rkw, **kwargs)
        if i < len(labels):
            # this solve issue 32:
            # https://github.com/Davide-sd/sympy-plot-backends/issues/32
            already_set = False
            for s in series:
                if isinstance(s, LineOver1DRangeSeries) and not already_set:
                    s.label = labels[i]
                    already_set = True
                else:
                    s.label = None
        color_series_dict[i] = series

    # NOTE: let's overwrite this keyword argument: the dictionary will be used
    # by the backend to assign the proper colors to the pieces
    kwargs["process_piecewise"] = color_series_dict
    return graphics(**kwargs)
