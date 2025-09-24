from sympy import (
    sin, cos, Piecewise, Sum, Wild, sign, piecewise_fold, Interval, Union,
    FiniteSet, Eq, Ne, Expr, Plane, Curve, Point3D
)
from sympy.core.relational import Relational
from sympy.geometry.line import LinearEntity3D
# NOTE: from sympy import EmptySet is a different thing!!!
from sympy.sets.sets import EmptySet
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, ContourSeries,
    ImplicitSeries, List2DSeries, Geometry2DSeries, Geometry3DSeries,
    HLineSeries, VLineSeries
)
from spb.doc_utils.docstrings import _PARAMS
from spb.doc_utils.ipython import modify_graphics_series_doc
from spb.graphics.utils import _plot_sympify
from spb.utils import (
    _create_missing_ranges, _preprocess_multiple_ranges
)
import warnings
import param


def _process_piecewise(piecewise, _range, label, **kwargs):
    """
    Extract the pieces of an univariate Piecewise function and create the
    necessary series for an univariate plot.

    Notes
    =====

    As a design choice, the following implementation reuses the existing
    classes, instead of creating a new one to deal with Piecewise. Here, each
    piece is going to create at least one series. If a piece is using a union
    of coditions (for example, ``((x < 0) | (x > 2))``), than two or more
    series of the same expression are created (for example, one covering
    ``x < 0`` and the other covering ``x > 2``), both having the same label.

    However, if a piece is outside of the provided plotting range, then it
    will not be added to the plot. This may lead to not-complete plots in some
    backend, such as BokehBackend, which is capable of auto-recompute the data
    on mouse drag. If the user drags the mouse over an area previously not
    shown (thus no data series), there won't be any line on the plot in this
    area.
    """
    # initial range
    irange = Interval(_range[1], _range[2], False, False)
    # ultimately it will contain all the series
    series = []
    # only contains Line2DSeries with fill=True. They have higher
    # rendering priority, as such they will be added to `series` at last.
    filled_series = []
    dots = kwargs.pop("dots", True)

    def func(expr, _set, c, from_union=False):
        if isinstance(_set, Interval):
            start, end = _set.args[0], _set.args[1]

            # arbitrary small offset
            offset = 1e-06
            # offset must be small even if the interval is small
            diff = end - start
            if diff < 1:
                e = 0
                while diff < 1:
                    diff *= 10
                    e -= 1
                offset *= e

            # prevent NaNs from happening at the ends of the interval
            if _set.left_open:
                start += offset
            if _set.right_open:
                end -= offset

            main_series = LineOver1DRangeSeries(
                expr, (_range[0], start, end), label, **kwargs)
            series.append(main_series)

            if dots:
                xx, yy = main_series.get_data()
                # NOTE: starting with SymPy 1.13, == means structural
                # equality also for Float/Integer/Rational
                if not _range[1].equals(xx[0]):
                    correct_list = series if _set.left_open else filled_series
                    correct_list.append(
                        List2DSeries(
                            [xx[0]], [yy[0]], scatter=True,
                            fill=not _set.left_open, **kwargs)
                    )
                if not _range[2].equals(xx[-1]):
                    correct_list = series if _set.right_open else filled_series
                    correct_list.append(
                        List2DSeries(
                            [xx[-1]], [yy[-1]], scatter=True,
                            fill=not _set.right_open, **kwargs)
                    )
        elif isinstance(_set, FiniteSet):
            loc, val = [], []
            for _loc in _set.args:
                loc.append(float(_loc))
                val.append(float(expr.evalf(subs={_range[0]: _loc})))
            filled_series.append(
                List2DSeries(loc, val, scatter=True, fill=True, **kwargs))
            if not from_union:
                c += 1
        elif isinstance(_set, Union):
            for _s in _set.args:
                c = func(expr, _s, c, from_union=True)
        elif isinstance(_set, EmptySet):
            # in this case, some pieces are outside of the provided range.
            # don't add any series, but increase the counter nonetheless so
            # that there is one-to-one correspondance between the expression
            # and what is plotted.
            if not from_union:
                c += 1
        else:
            raise TypeError(
                "Unhandle situation:\n" +
                "expr: {}\ncond: {}\ntype(cond): {}\n".format(
                    str(expr), _set, type(_set)) +
                "See if you can rewrite the piecewise without "
                "this type of condition and then plot it again.")

        return c

    piecewise = piecewise_fold(piecewise)
    expr_cond = piecewise.as_expr_set_pairs()
    # for the label, attach the number of the piece
    count = 1
    for expr, cond in expr_cond:
        count = func(expr, irange.intersection(cond), count)

    series += filled_series
    return series


def _build_line_series(expr, r, label, **kwargs):
    """
    Loop over the provided arguments. If a piecewise function is found,
    decompose it in such a way that each argument gets its own series.
    """
    series = []
    pp = kwargs.get("process_piecewise", False)
    sum_bound = int(kwargs.get("sum_bound", 1000))
    if not callable(expr) and expr.has(Piecewise) and pp:
        series += _process_piecewise(expr, r, label, **kwargs)
    else:
        series.append(LineOver1DRangeSeries(expr, r, label, **kwargs))
    return series


@modify_graphics_series_doc(LineOver1DRangeSeries, replace={"params": _PARAMS})
def line(expr, range_x=None, label=None, rendering_kw=None, **kwargs):
    """
    Plot a function of one variable over a 2D space.

    Returns
    =======

    series : list
        A list containing one instance of ``LineOver1DRangeSeries``.

    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, sin, pi, tan, exp, cos, log, floor
       >>> from spb import *
       >>> x, y = symbols('x, y')

    Single Plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(line(x**2, (x, -5, 5)))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-5, 5)

    Multiple functions over the same range with custom rendering options:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     line(x, (x, -3, 3)),
       ...     line(log(x), (x, -3, 3), rendering_kw={"linestyle": "--"}),
       ...     line(exp(x), (x, -3, 3), rendering_kw={"linestyle": ":"}),
       ...     aspect="equal", ylim=(-3, 3)
       ... )
       Plot object containing:
       [0]: cartesian line: x for x over (-3, 3)
       [1]: cartesian line: log(x) for x over (-3, 3)
       [2]: cartesian line: exp(x) for x over (-3, 3)

    Plotting a summation in which the free symbol of the expression is not
    used in the lower/upper bounds:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import Sum, oo, latex
       >>> expr = Sum(1 / x ** y, (x, 1, oo))
       >>> graphics(
       ...     line(expr, (y, 2, 10), sum_bound=1e03),
       ...     title="$%s$" % latex(expr)
       ... )
       Plot object containing:
       [0]: cartesian line: Sum(x**(-y), (x, 1, oo)) for y over (2, 10)

    Plotting a summation in which the free symbol of the expression is
    used in the lower/upper bounds. Here, the discretization variable must
    assume integer values:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = Sum(1 / x, (x, 1, y))
       >>> graphics(
       ...     line(expr, (y, 2, 10), scatter=True),
       ...     title="$%s$" % latex(expr)
       ... )
       Plot object containing:
       [0]: cartesian line: Sum(1/x, (x, 1, y)) for y over (2, 10)

    Detect essential singularities and visualize them with vertical lines.
    Also, apply a tick formatter on the x-axis is order to show ticks at
    multiples of pi/2:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> graphics(
       ...     line(tan(x), (x, -1.5*pi, 1.5*pi), detect_poles="symbolic"),
       ...     x_ticks_formatter=multiples_of_pi_over_2(),
       ...     ylim=(-7, 7), xlabel="x [deg]", grid=False
       ... )
       Plot object containing:
       [0]: cartesian line: tan(x) for x over (-1.5*pi, 1.5*pi)

    Introducing discontinuities by excluding specified points:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     line(floor(x) / x, (x, -3.25, 3.25), exclude=list(range(-4, 5))),
       ...     ylim=(-1, 5)
       ... )
       Plot object containing:
       [0]: cartesian line: floor(x)/x for x over (-3.25000000000000, 3.25000000000000)

    Creating a step plot:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     line(x-2, (x, 0, 10), only_integers=True, steps="pre", label="pre"),
       ...     line(x, (x, 0, 10), only_integers=True, steps="mid", label="mid"),
       ...     line(x+2, (x, 0, 10), only_integers=True, steps="post", label="post"),
       ... )
       Plot object containing:
       [0]: cartesian line: x - 2 for x over (0, 10)
       [1]: cartesian line: x for x over (0, 10)
       [2]: cartesian line: x + 2 for x over (0, 10)

    Advanced example showing:

    * detect singularities by setting ``adaptive=False`` (better performance),
      increasing the number of discretization points (in order to have
      'vertical' segments on the lines) and reducing the threshold for the
      singularity-detection algorithm.
    * application of color function.
    * combining together multiple lines.

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
       >>> graphics(
       ...     line(
       ...         expr, (x, -5, 5), "distance from (0, 0)",
       ...         rendering_kw={"cmap": "plasma"},
       ...         adaptive=False, detect_poles=True, n=3e04,
       ...         eps=1e-04, color_func=cf),
       ...     line(5 * sin(x), (x, -5, 5), rendering_kw={"linestyle": "--"}),
       ...     ylim=(-10, 10), title="$%s$" % latex(expr)
       ... )
       Plot object containing:
       [0]: cartesian line: 5*sin(x) + 1/cos(10*x) for x over (-5, 5)
       [1]: cartesian line: 5*sin(x) for x over (-5, 5)

    Interactive-widget plot of an oscillator. Refer to the interactive
    sub-module documentation to learn more about the ``params`` dictionary.
    This plot illustrates:

    * plotting multiple expressions, each one with its own label and
      rendering options.
    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).
    * the use of :py:class:`panel.widgets.slider.RangeSlider`, which is a
      2-values widget. In this case it is used to enforce the condition
      `f1 < f2`.
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
       import panel as pn
       x, y, f1, f2, d, n = symbols("x, y, f_1, f_2, d, n")
       params = {
           (f1, f2): pn.widgets.RangeSlider(
               value=(1, 2), start=0, end=10, step=0.1),     # frequencies
           d: (0.25, 0, 1),   # damping
           n: (2, 0, 4)       # multiple of pi
       }
       graphics(
           line(cos(f1 * x) * exp(-d * x), prange(x, 0, n * pi),
               label="oscillator 1", params=params),
           line(cos(f2 * x) * exp(-d * x), prange(x, 0, n * pi),
               label="oscillator 2", params=params),
           line(exp(-d * x), prange(x, 0, n * pi), label="upper limit",
               rendering_kw={"linestyle": ":"}, params=params),
           line(-exp(-d * x), prange(x, 0, n * pi), label="lower limit",
               rendering_kw={"linestyle": ":"}, params=params),
           ylim=(-1.25, 1.25),
           title=("$f_1$ = {:.2f} Hz", f1)
       )

    See Also
    ========

    line_parametric_2d, line_polar, implicit_2d, list_2d, geometry,
    spb.graphics.functions_3d.line_parametric_3d

    """
    expr = _plot_sympify(expr)
    params = kwargs.get("params", {})
    range_x = _create_missing_ranges(
        [expr], [range_x] if range_x else [], 1, params)[0]
    return _build_line_series(
        expr, range_x, label, rendering_kw=rendering_kw, **kwargs)


@modify_graphics_series_doc(Parametric2DLineSeries, replace={"params": _PARAMS})
def line_parametric_2d(
    expr_x, expr_y, range_p=None, label=None, rendering_kw=None,
    colorbar=True, use_cm=True, **kwargs
):
    """
    Plots a 2D parametric curve.

    Returns
    =======

    series : list
        A list containing one instance of ``Parametric2DLineSeries``.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi, floor, log
       >>> from spb import *
       >>> t, u, v = symbols('t, u, v')

    A parametric plot of a single expression (a Hypotrochoid using an equal
    aspect ratio):

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     line_parametric_2d(
       ...         2 * cos(u) + 5 * cos(2 * u / 3),
       ...         2 * sin(u) - 5 * sin(2 * u / 3),
       ...         (u, 0, 6 * pi)
       ...     ),
       ...     aspect="equal"
       ... )
       Plot object containing:
       [0]: parametric cartesian line: (5*cos(2*u/3) + 2*cos(u), -5*sin(2*u/3) + 2*sin(u)) for u over (0, 6*pi)

    A parametric plot with multiple expressions with the same range with solid
    line colors:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     line_parametric_2d(2 * cos(t), sin(t), (t, 0, 2*pi), use_cm=False),
       ...     line_parametric_2d(cos(t), 2 * sin(t), (t, 0, 2*pi), use_cm=False),
       ... )
       Plot object containing:
       [0]: parametric cartesian line: (2*cos(t), sin(t)) for t over (0, 2*pi)
       [1]: parametric cartesian line: (cos(t), 2*sin(t)) for t over (0, 2*pi)

    A parametric plot with multiple expressions with different ranges,
    custom labels, custom rendering options and a transformation function
    applied to the discretized parameter to convert radians to degrees:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> graphics(
       ...     line_parametric_2d(
       ...         3 * cos(u), 3 * sin(u), (u, 0, 2 * pi), "u [deg]",
       ...         rendering_kw={"lw": 3}, tp=np.rad2deg),
       ...     line_parametric_2d(
       ...         3 * cos(2 * v), 5 * sin(4 * v), (v, 0, pi), "v [deg]",
       ...         tp=np.rad2deg
       ...     ),
       ...     aspect="equal"
       ... )
       Plot object containing:
       [0]: parametric cartesian line: (3*cos(u), 3*sin(u)) for u over (0, 2*pi)
       [1]: parametric cartesian line: (3*cos(2*v), 5*sin(4*v)) for v over (0, pi)

    Introducing discontinuities by excluding specified points:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> e1 = log(floor(t))*cos(t)
       >>> e2 = log(floor(t))*sin(t)
       >>> graphics(
       ...     line_parametric_2d(
       ...         e1, e2, (t, 1, 4*pi), exclude=list(range(1, 13))),
       ...     grid=False
       ... )
       Plot object containing:
       [0]: parametric cartesian line: (log(floor(t))*cos(t), log(floor(t))*sin(t)) for t over (1, 4*pi)

    Plotting a numerical function instead of a symbolic expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> fx = lambda t: np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12)**5)
       >>> fy = lambda t: np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12)**5)
       >>> graphics(
       ...     line_parametric_2d(fx, fy, ("t", 0, 12 * pi),
       ...         use_cm=False, n=2000),
       ...     title="Butterfly Curve",
       ... )  # doctest: +SKIP

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
       graphics(
           line_parametric_2d(
               cos(a * x), sin(x), prange(x, s*pi, e*pi),
                   params={
                   a: (0.5, 0, 2),
                   s: (0, 0, 2),
                   e: (2, 0, 2),
               }),
           aspect="equal",
           xlim=(-1.25, 1.25), ylim=(-1.25, 1.25)
       )

    See Also
    ========

    spb.graphics.functions_3d.line_parametric_3d, line_polar, line

    """
    expr_x, expr_y = map(_plot_sympify, [expr_x, expr_y])
    params = kwargs.get("params", {})
    range_p = _create_missing_ranges(
        [expr_x, expr_y], [range_p] if range_p else [], 1, params)[0]
    s = Parametric2DLineSeries(
        expr_x, expr_y, range_p, label,
        rendering_kw=rendering_kw, colorbar=colorbar,
        use_cm=use_cm, **kwargs)
    return [s]


@modify_graphics_series_doc(Parametric2DLineSeries, replace={"params": _PARAMS})
def line_polar(expr, range_p=None, label=None, rendering_kw=None, **kwargs):
    """
    Creates a 2D polar plot of a function of one variable.

    Examples
    ========

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import symbols, sin, cos, exp, pi
        >>> from spb import *
        >>> theta = symbols('theta')

    Plot with cartesian axis:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     line_polar(3 * sin(2 * theta), (theta, 0, 2*pi)),
       ...     aspect="equal"
       ... )
       Plot object containing:
       [0]: parametric cartesian line: (3*sin(2*theta)*cos(theta), 3*sin(theta)*sin(2*theta)) for theta over (0, 2*pi)

    Plot with polar axis:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     line_polar(exp(sin(theta)) - 2 * cos(4 * theta), (theta, 0, 2 * pi)),
       ...     polar_axis=True, aspect="equal"
       ... )
       Plot object containing:
       [0]: parametric cartesian line: ((exp(sin(theta)) - 2*cos(4*theta))*cos(theta), (exp(sin(theta)) - 2*cos(4*theta))*sin(theta)) for theta over (0, 2*pi)

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
       params = {
           a: pn.widgets.IntInput(value=6, name="a"),
           b: pn.widgets.IntInput(value=12, name="b"),
           c: pn.widgets.IntInput(value=18, name="c"),
           d: (4.7, 0, 2*pi),
           e: (1.8, 0, 2*pi),
           f: (3, 0, 5),
           tp: (2, 0, 2)
       }
       series = []
       for n in range(20):
           series += line_polar(
               func(n), prange(theta, 0, tp*pi), params=params,
               rendering_kw={"line_color": "black", "line_width": 0.5})
       graphics(
           *series,
           aspect="equal",
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

    line_parametric_2d, line

    """
    expr = _plot_sympify(expr)
    params = kwargs.get("params", {})
    kwargs.setdefault("use_cm", False)
    range_p = _create_missing_ranges(
        [expr], [range_p] if range_p else [], 1, params)[0]
    theta = range_p[0]
    return line_parametric_2d(
        expr * cos(theta), expr * sin(theta), range_p,
        label, rendering_kw, **kwargs)


@modify_graphics_series_doc(ContourSeries, replace={"params": _PARAMS})
def contour(
    expr, range_x=None, range_y=None, label=None, rendering_kw=None,
    colorbar=True, clabels=True, fill=True, **kwargs
):
    """
    Plots contour lines or filled contours of a function of two variables.

    Returns
    =======
    series : list
        A list containing one instance of ``ContourSeries``.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, exp, sin, pi, Eq, Add
       >>> from spb import *
       >>> x, y = symbols('x, y')

    Filled contours of a function of two variables.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     contour(
       ...         cos((x**2 + y**2)) * exp(-(x**2 + y**2) / 10),
       ...         (x, -5, 5), (y, -5, 5)
       ...     ),
       ...     grid=False
       ... )
       Plot object containing:
       [0]: contour: exp(-x**2/10 - y**2/10)*cos(x**2 + y**2) for x over (-5, 5) and y over (-5, 5)

    Line contours of a function of two variables, with ticks formatted as
    multiples of `pi/n`.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = 5 * (cos(x) - 0.2 * sin(y))**2 + 5 * (-0.2 * cos(x) + sin(y))**2
       >>> graphics(
       ...     contour(expr, (x, 0, 2 * pi), (y, 0, 2 * pi), fill=False),
       ...     x_ticks_formatter=multiples_of_pi_over_2(),
       ...     y_ticks_formatter=multiples_of_pi_over_3(),
       ...     aspect="equal"
       ... )
       Plot object containing:
       [0]: contour: 5*(-0.2*sin(y) + cos(x))**2 + 5*(sin(y) - 0.2*cos(x))**2 for x over (0, 2*pi) and y over (0, 2*pi)

    Combining together filled and line contours. Use a custom label on the
    colorbar of the filled contour.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = 5 * (cos(x) - 0.2 * sin(y))**2 + 5 * (-0.2 * cos(x) + sin(y))**2
       >>> graphics(
       ...     contour(expr, (x, 0, 2 * pi), (y, 0, 2 * pi), "z",
       ...         rendering_kw={"cmap": "coolwarm"}),
       ...     contour(expr, (x, 0, 2 * pi), (y, 0, 2 * pi),
       ...         rendering_kw={"colors": "k", "cmap": None, "linewidths": 0.75},
       ...         fill=False),
       ...     x_ticks_formatter=multiples_of_pi_over_2(),
       ...     y_ticks_formatter=multiples_of_pi_over_3(),
       ...     aspect="equal", grid=False
       ... )
       Plot object containing:
       [0]: contour: 5*(-0.2*sin(y) + cos(x))**2 + 5*(sin(y) - 0.2*cos(x))**2 for x over (0, 2*pi) and y over (0, 2*pi)
       [1]: contour: 5*(-0.2*sin(y) + cos(x))**2 + 5*(sin(y) - 0.2*cos(x))**2 for x over (0, 2*pi) and y over (0, 2*pi)

    Visually inspect the solutions of a system of 2 non-linear equations.
    The intersections between the contour lines represent the solutions.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> eq1 = Eq((cos(x) - sin(y) / 2)**2 + 3 * (-sin(x) + cos(y) / 2)**2, 2)
       >>> eq2 = Eq((cos(x) - 2 * sin(y))**2 - (sin(x) + 2 * cos(y))**2, 3)
       >>> graphics(
       ...     contour(
       ...         eq1.lhs - eq1.rhs, (x, 0, 2 * pi), (y, 0, 2 * pi),
       ...         rendering_kw={"levels": [0]},
       ...         fill=False, clabels=False),
       ...     contour(
       ...         eq2.lhs - eq2.rhs, (x, 0, 2 * pi), (y, 0, 2 * pi),
       ...         rendering_kw={"levels": [0]},
       ...         fill=False, clabels=False),
       ... )
       Plot object containing:
       [0]: contour: 3*(-sin(x) + cos(y)/2)**2 + (-sin(y)/2 + cos(x))**2 - 2 for x over (0, 2*pi) and y over (0, 2*pi)
       [1]: contour: -(sin(x) + 2*cos(y))**2 + (-2*sin(y) + cos(x))**2 - 3 for x over (0, 2*pi) and y over (0, 2*pi)

    Contour plot with polar axis:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> r, theta = symbols("r, theta")
       >>> graphics(
       ...     contour(
       ...         sin(2 * r) * cos(theta), (theta, 0, 2*pi), (r, 0, 7),
       ...         rendering_kw={"levels": 100}
       ...     ),
       ...     polar_axis=True, aspect="equal"
       ... )
       Plot object containing:
       [0]: contour: sin(2*r)*cos(theta) for theta over (0, 2*pi) and r over (0, 7)

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
       x, y, a, b = symbols("x y a b")
       x_min, x_max, y_min, y_max = symbols("x_min x_max y_min y_max")
       expr = (cos(x) + a * sin(x) * sin(y) - b * sin(x) * cos(y))**2
       graphics(
           contour(
               expr, prange(x, x_min*pi, x_max*pi), prange(y, y_min*pi, y_max*pi),
               params={
                   a: (1, 0, 2), b: (1, 0, 2),
                   (x_min, x_max): pn.widgets.RangeSlider(
                       value=(-1, 1), start=-3, end=3, step=0.1),
                   (y_min, y_max): pn.widgets.RangeSlider(
                       value=(-1, 1), start=-3, end=3, step=0.1),
               }),
           grid=False
       )

    See Also
    ========

    spb.graphics.functions_3d.surface

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
    ranges = _preprocess_multiple_ranges([expr], [range_x, range_y], 2, params)
    s = ContourSeries(
        expr, *ranges, label, rendering_kw=rendering_kw,
        colorbar=colorbar, fill=fill, clabels=clabels, **kwargs)
    return [s]


@modify_graphics_series_doc(ImplicitSeries, replace={"params": _PARAMS})
def implicit_2d(
    f, range_x=None, range_y=None, label=None, rendering_kw=None,
    color=None, border_color=None, border_kw=None, **kwargs
):
    """
    Plot implicit equations / inequalities.

    ``implicit_2d``, by default, generates a contour using a mesh grid of
    fixednumber of points. The greater the number of points, the better the
    results, but also the greater the memory used.
    By setting ``adaptive=True``, interval arithmetic will be used to plot
    functions. If the expression cannot be plotted using interval arithmetic,
    it defaults to generating a contour using a mesh grid. With interval
    arithmetic, the line width can become very small; in those cases, it is
    better to use the mesh grid approach.

    Parameters
    ==========

    color : str, optional
        Specify the color of lines/regions. Default to None (automatic
        coloring by the backend).
    border_color : str or bool, optional
        If given, a limiting border will be added when plotting inequalities
        (<, <=, >, >=). Use ``border_kw`` if more customization options are
        required.
    border_kw : dict, optional
        If given, a limiting border will be added when plotting inequalities
        (<, <=, >, >=). This is a dictionary of keywords/values which is
        passed to the backend's function to customize the appearance of the
        limiting border. Refer to the plotting library (backend) manual for
        more informations.

    Returns
    =======

    series : list
        A list containing at most two instances of ``Implicit2DSeries``.

    Examples
    ========

    Plot expressions:

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import symbols, Ne, Eq, And, sin, cos, pi, log, latex
        >>> from spb import *
        >>> x, y = symbols('x y')

    Plot a line representing an equality:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(implicit_2d(x - 1, (x, -5, 5), (y, -5, 5)))
       Plot object containing:
       [0]: Implicit expression: Eq(x - 1, 0) for x over (-5, 5) and y over (-5, 5)

    Plot a region:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     implicit_2d(y > x**2, (x, -5, 5), (y, -10, 10), n=150),
       ...     grid=False)
       Plot object containing:
       [0]: Implicit expression: y > x**2 for x over (-5, 5) and y over (-10, 10)

    Plot a region using a custom color, highlights the limiting border and
    customize its appearance.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = 4 * (cos(x) - sin(y) / 5)**2 + 4 * (-cos(x) / 5 + sin(y))**2
       >>> graphics(
       ...     implicit_2d(
       ...         expr <= pi, (x, -pi, pi), (y, -pi, pi),
       ...         color="gold", border_color="k",
       ...         border_kw={"linestyles": "-.", "linewidths": 1}
       ...     ),
       ...     grid=False
       ... )
       Plot object containing:
       [0]: Implicit expression: 4*(-sin(y)/5 + cos(x))**2 + 4*(sin(y) - cos(x)/5)**2 <= pi for x over (-pi, pi) and y over (-pi, pi)
       [1]: Implicit expression: Eq(-4*(-sin(y)/5 + cos(x))**2 - 4*(sin(y) - cos(x)/5)**2 + pi, 0) for x over (-pi, pi) and y over (-pi, pi)

    Boolean expressions will be plotted with the adaptive algorithm. Note the
    thin width of lines:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     implicit_2d(
       ...         Eq(y, sin(x)) & (y > 0), (x, -2 * pi, 2 * pi), (y, -4, 4)),
       ...     implicit_2d(
       ...         Eq(y, sin(x)) & (y < 0), (x, -2 * pi, 2 * pi), (y, -4, 4)),
       ...     ylim=(-2, 2)
       ... )
       Plot object containing:
       [0]: Implicit expression: (y > 0) & Eq(y, sin(x)) for x over (-2*pi, 2*pi) and y over (-4, 4)
       [1]: Implicit expression: (y < 0) & Eq(y, sin(x)) for x over (-2*pi, 2*pi) and y over (-4, 4)

    Plotting multiple implicit expressions and setting labels:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> V, t, b, L = symbols("V, t, b, L")
       >>> L_array = [5, 10, 15, 20, 25]
       >>> b_val = 0.0032
       >>> expr = b * V * 0.277 * t - b * L - log(1 + b * V * 0.277 * t)
       >>> series = []
       >>> for L_val in L_array:
       ...     series += implicit_2d(
       ...         expr.subs({b: b_val, L: L_val}), (t, 0, 3), (V, 0, 1000),
       ...         label="L = %s" % L_val)
       >>> graphics(*series)
       Plot object containing:
       [0]: Implicit expression: Eq(0.0008864*V*t - log(0.0008864*V*t + 1) - 0.016, 0) for t over (0, 3) and V over (0, 1000)
       [1]: Implicit expression: Eq(0.0008864*V*t - log(0.0008864*V*t + 1) - 0.032, 0) for t over (0, 3) and V over (0, 1000)
       [2]: Implicit expression: Eq(0.0008864*V*t - log(0.0008864*V*t + 1) - 0.048, 0) for t over (0, 3) and V over (0, 1000)
       [3]: Implicit expression: Eq(0.0008864*V*t - log(0.0008864*V*t + 1) - 0.064, 0) for t over (0, 3) and V over (0, 1000)
       [4]: Implicit expression: Eq(0.0008864*V*t - log(0.0008864*V*t + 1) - 0.08, 0) for t over (0, 3) and V over (0, 1000)

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
       >>> graphics(
       ...    implicit_2d(
       ...        expr1, *ranges, adaptive=True, depth=0,
       ...        label="adaptive=True, depth=0"),
       ...    implicit_2d(
       ...        expr2, *ranges, adaptive=True, depth=1,
       ...        label="adaptive=True, depth=1"),
       ...    implicit_2d(
       ...        expr3, *ranges, adaptive=False, label="adaptive=False"),
       ...    grid=False
       ... )
       Plot object containing:
       [0]: Implicit expression: Eq(x*y - 20, 15*y) for x over (15, 30) and y over (0, 50)
       [1]: Implicit expression: Eq(y*(x - 3) - 20, 15*y) for x over (15, 30) and y over (0, 50)
       [2]: Implicit expression: Eq(y*(x - 6) - 20, 15*y) for x over (15, 30) and y over (0, 50)

    If the expression is plotted with the adaptive algorithm and it produces
    "low-quality" results, maybe it's possible to rewrite it in order to use
    the mesh grid approach (contours). For example:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from spb import plotgrid
       >>> expr = Ne(x*y, 1)
       >>> p1 = graphics(
       ...     implicit_2d(expr, (x, -10, 10), (y, -10, 10)),
       ...     grid=False, title="$%s$ : First approach" % latex(expr),
       ...     aspect="equal", show=False)
       >>> p2 = graphics(
       ...     implicit_2d(x < 20, (x, -10, 10), (y, -10, 10)),
       ...     implicit_2d(Eq(*expr.args), (x, -10, 10), (y, -10, 10),
       ...         color="w", show_in_legend=False),
       ...     grid=False, title="$%s$ : Second approach" % latex(expr),
       ...     aspect="equal", show=False)
       >>> plotgrid(p1, p2, nc=2)  # doctest: +SKIP

    Interactive-widget implicit plot. Refer to the interactive sub-module
    documentation to learn more about the ``params`` dictionary.
    This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).
    * the use of :py:class:`panel.widgets.slider.RangeSlider`, which is a
      2-values widget.

    .. panel-screenshot::
       :small-size: 800, 700

       from sympy import *
       from spb import *
       import panel as pn
       x, y, a, b, c, d = symbols("x, y, a, b, c, d")
       y_min, y_max = symbols("y_min, y_max")
       expr = Eq(a * x**2 - b * x + c, d * y + y**2)
       graphics(
           implicit_2d(expr, (x, -2, 2), prange(y, y_min, y_max),
               params={
                   a: (10, -15, 15),
                   b: (7, -15, 15),
                   c: (3, -15, 15),
                   d: (2, -15, 15),
                   (y_min, y_max): pn.widgets.RangeSlider(
                       value=(-10, 10), start=-15, end=15, step=0.1)
               }, n=150),
           ylim=(-10, 10))

    See Also
    ========

    line, spb.graphics.functions_3d.implicit_3d

    """
    # back-compatibility
    range_x = kwargs.pop("range1", range_x)
    range_y = kwargs.pop("range2", range_y)

    expr = _plot_sympify(f)
    params = kwargs.get("params", {})

    if not (range_x and range_y):
        warnings.warn(
            "No ranges were provided. This function will attempt to find "
            "them, however the order will be arbitrary, which means the "
            "visualization might be flipped."
        )

    series = []
    ranges = _preprocess_multiple_ranges([expr], [range_x, range_y], 2, params)
    series.append(ImplicitSeries(
        expr, *ranges, label, color=color,
        rendering_kw=rendering_kw, **kwargs))

    if (border_color or border_kw) and (not isinstance(expr, (Expr, Eq, Ne))):
        kw = kwargs.copy()
        kw["color"] = border_color
        kw["rendering_kw"] = border_kw
        kw.setdefault("show_in_legend", False)
        if isinstance(expr, Relational):
            expr = expr.rhs - expr.lhs
        series.append(
            ImplicitSeries(expr, *ranges, label, **kw))
    return series


@modify_graphics_series_doc(List2DSeries, replace={"params": _PARAMS})
def list_2d(list_x, list_y, label=None, rendering_kw=None, **kwargs):
    """
    Plots lists of coordinates.

    Returns
    =======

    series : list
        A list containing one instance of ``List2DSeries``.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, sin, cos
       >>> from spb import *
       >>> x = symbols('x')

    Plot the coordinates of a single function:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> xx = [t / 100 * 6 - 3 for t in list(range(101))]
       >>> yy = [cos(x).evalf(subs={x: t}) for t in xx]
       >>> graphics(list_2d(xx, yy))
       Plot object containing:
       [0]: 2D list plot

    Plot individual points with custom labels. Each point will be converted
    to a list by the algorithm:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     list_2d(0, 0, "A", scatter=True),
       ...     list_2d(1, 1, "B", scatter=True),
       ...     list_2d(2, 0, "C", scatter=True),
       ... )
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
       >>> graphics(
       ...     list_2d(xx, yy1, "cos", scatter=True),
       ...     list_2d(xx, yy2, "sin", {"marker": "*", "markerfacecolor": None},
       ...             scatter=True),
       ... )
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
       graphics(
           line_parametric_2d(
               cos(x), sin(x), (x, 0, 2*pi), rendering_kw={"linestyle": ":"},
               use_cm=False),
           line_parametric_2d(
               cos(2 * x) / 2, sin(2 * x) / 2, (x, 0, pi),
               rendering_kw={"linestyle": ":"}, use_cm=False),
           list_2d(
               cos(t), sin(t), "A",
               rendering_kw={"marker": "s", "markerfacecolor": None},
               params=params, scatter=True),
           list_2d(
               cos(2 * t) / 2, sin(2 * t) / 2, "B",
               rendering_kw={"marker": "s", "markerfacecolor": None},
               params=params, scatter=True),
           aspect="equal"
       )

    See Also
    ========

    line, spb.graphics.functions_3d.list_3d

    """
    if not hasattr(list_x, "__iter__"):
        list_x = [list_x]
    if not hasattr(list_y, "__iter__"):
        coord_y = [list_y]
    s = List2DSeries(
        list_x, list_y, label, rendering_kw=rendering_kw, **kwargs)
    return [s]


@modify_graphics_series_doc(Geometry2DSeries, replace={"params": _PARAMS})
def geometry(geom, label=None, rendering_kw=None, fill=True, **kwargs):
    """
    Plot entities from the sympy.geometry module.

    Returns
    =======

    series : list
        A list containing one instance of ``GeometrySeries``.

    Examples
    ========

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import (symbols, Circle, Ellipse, Polygon,
        ...      Curve, Segment, Point2D, Point3D, Line3D, Plane,
        ...      Rational, pi, Point, cos, sin)
        >>> from spb import *
        >>> x, y, z = symbols('x, y, z')

    Plot a single geometry, customizing its color:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     geometry(
       ...         Ellipse(Point(-3, 2), hradius=3, eccentricity=Rational(4, 5)),
       ...         rendering_kw={"color": "tab:orange"}),
       ...     grid=False, aspect="equal"
       ... )
       Plot object containing:
       [0]: 2D geometry entity: Ellipse(Point2D(-3, 2), 3, 9/5)

    Plot several numeric geometric entitiesy. By default, circles, ellipses and
    polygons are going to be filled. Plotting Curve objects is the same as
    `plot_parametric`.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     geometry(Circle(Point(0, 0), 5)),
       ...     geometry(Ellipse(Point(-3, 2), hradius=3, eccentricity=Rational(4, 5))),
       ...     geometry(Polygon((4, 0), 4, n=5)),
       ...     geometry(Curve((cos(x), sin(x)), (x, 0, 2 * pi))),
       ...     geometry(Segment((-4, -6), (6, 6))),
       ...     geometry(Point2D(0, 0)),
       ...     aspect="equal", grid=False
       ... )
       Plot object containing:
       [0]: 2D geometry entity: Circle(Point2D(0, 0), 5)
       [1]: 2D geometry entity: Ellipse(Point2D(-3, 2), 3, 9/5)
       [2]: 2D geometry entity: RegularPolygon(Point2D(4, 0), 4, 5, 0)
       [3]: parametric cartesian line: (cos(x), sin(x)) for x over (0, 2*pi)
       [4]: 2D geometry entity: Segment2D(Point2D(-4, -6), Point2D(6, 6))
       [5]: 2D geometry entity: Point2D(0, 0)

    Plot several numeric geometric entities defined by numbers only, turn off
    fill. Every entity is represented as a line.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     geometry(Circle(Point(0, 0), 5), fill=False),
       ...     geometry(
       ...         Ellipse(Point(-3, 2), hradius=3, eccentricity=Rational(4, 5)),
       ...         fill=False),
       ...     geometry(Polygon((4, 0), 4, n=5), fill=False),
       ...     geometry(Curve((cos(x), sin(x)), (x, 0, 2 * pi)), fill=False),
       ...     geometry(Segment((-4, -6), (6, 6)), fill=False),
       ...     geometry(Point2D(0, 0), fill=False),
       ...     aspect="equal", grid=False
       ... )
       Plot object containing:
       [0]: 2D geometry entity: Circle(Point2D(0, 0), 5)
       [1]: 2D geometry entity: Ellipse(Point2D(-3, 2), 3, 9/5)
       [2]: 2D geometry entity: RegularPolygon(Point2D(4, 0), 4, 5, 0)
       [3]: parametric cartesian line: (cos(x), sin(x)) for x over (0, 2*pi)
       [4]: 2D geometry entity: Segment2D(Point2D(-4, -6), Point2D(6, 6))
       [5]: 2D geometry entity: Point2D(0, 0)

    Plot 3D geometric entities. Instances of ``Plane`` must be plotted with
    ``implicit_3d`` or with ``plane`` (with the necessary ranges).

    .. k3d-screenshot::

       from sympy import *
       from spb import *
       x, y, z = symbols("x, y, z")
       graphics(
            geometry(
                Point3D(0, 0, 0), label="center",
                rendering_kw={"point_size": 1}),
            geometry(Line3D(Point3D(-2, -3, -4), Point3D(2, 3, 4)), "line"),
            plane(
                Plane((0, 0, 0), (1, 1, 1)),
                (x, -5, 5), (y, -4, 4), (z, -10, 10)),
            backend=KB
        )

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary.

    .. panel-screenshot::
       :small-size: 800, 625

       from sympy import *
       from spb import *
       import panel as pn
       a, b, c, d = symbols("a, b, c, d")
       params = {
           a: (0, -1, 1),
           b: (1, -1, 1),
           c: (2, 1, 2),
           d: pn.widgets.IntInput(value=6, start=3, end=8, name="n")
       }
       graphics(
           geometry(Polygon((a, b), c, n=d), "a", params=params),
           geometry(
               Polygon((a + 2, b + 3), c, n=d + 1), "b",
               params=params, fill=False),
           aspect="equal",
           xlim=(-2.5, 5.5), ylim=(-3, 6.5), imodule="panel")

    See Also
    ========

    line, spb.graphics.functions_3d.plane

    """
    if isinstance(geom, Plane):
        raise TypeError("")

    kwargs.setdefault("is_filled", fill)

    if isinstance(geom, Curve):
        new_cls = (
            Parametric2DLineSeries
            if len(geom.functions) == 2
            else Parametric3DLineSeries
        )
        s = new_cls(
            *geom.functions, geom.limits,
            label=label, rendering_kw=rendering_kw, **kwargs)
    else:
        if isinstance(geom, (LinearEntity3D, Point3D)):
            new_cls = Geometry3DSeries
        else:
            new_cls = Geometry2DSeries
        s = new_cls(
            geom, label=label, rendering_kw=rendering_kw, **kwargs)
    return [s]


@modify_graphics_series_doc(HLineSeries, replace={"params": _PARAMS})
def hline(y, label=None, rendering_kw=None, show_in_legend=True, **kwargs):
    """
    Create an horizontal line at a given location in a 2D space.

    Returns
    =======

    A list containing one instance of ``HVLineSeries``.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import *
       >>> from spb import *
       >>> x = symbols("x")
       >>> graphics(
       ...     line(cos(x), (x, -pi, pi)),
       ...     hline(
       ...         0.5, rendering_kw={"linestyle": ":"},
       ...         show_in_legend=False),
       ...     grid=False
       ... )
       Plot object containing:
       [0]: cartesian line: cos(x) for x over (-pi, pi)
       [1]: horizontal line at y = 0.500000000000000

    Interactive widget plot:

    .. panel-screenshot::
       :small-size: 800, 625

       from sympy import *
       from spb import *
       x, u, v, w = symbols("x u v w")
       params = {
           u: (1, 0, 2),
           v: (1, 0, 2),
           w: (0.5, -1, 1)
       }
       graphics(
           line(u * cos(v * x), (x, -pi, pi), params=params),
           hline(
               w, rendering_kw={"linestyle": ":"},
               show_in_legend=False, params=params),
           grid=False, ylim=(-2, 2)
       )

    See Also
    ========

    line

    """
    return [
        HLineSeries(
            y, label,
            rendering_kw=rendering_kw,
            show_in_legend=show_in_legend, **kwargs)
    ]


@modify_graphics_series_doc(VLineSeries, replace={"params": _PARAMS})
def vline(x, label=None, rendering_kw=None, show_in_legend=True, **kwargs):
    """
    Create an vertical line at a given location in a 2D space.

    Returns
    =======

    A list containing one instance of ``HVLineSeries``.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import *
       >>> from spb import *
       >>> x = symbols("x")
       >>> graphics(
       ...     line(cos(x), (x, -pi, pi)),
       ...     vline(1, rendering_kw={"linestyle": ":"}, show_in_legend=False),
       ...     vline(-1, rendering_kw={"linestyle": ":"}, show_in_legend=False),
       ...     grid=False
       ... )
       Plot object containing:
       [0]: cartesian line: cos(x) for x over (-pi, pi)
       [1]: vertical line at x = 1
       [2]: vertical line at x = -1

    Interactive widget plot:

    .. panel-screenshot::
       :small-size: 800, 625

       from sympy import *
       from spb import *
       x, u, v, w = symbols("x u v w")
       params = {
           u: (1, 0, 2),
           v: (1, 0, 2),
           w: (1, -pi, pi)
       }
       graphics(
           line(u * cos(v * x), (x, -pi, pi), params=params),
           vline(
               w, rendering_kw={"linestyle": ":"},
               show_in_legend=False, params=params),
           grid=False, ylim=(-2, 2)
       )

    See Also
    ========

    line

    """
    return [
        VLineSeries(
            x, label,
            rendering_kw=rendering_kw,
            show_in_legend=show_in_legend, **kwargs)
    ]
