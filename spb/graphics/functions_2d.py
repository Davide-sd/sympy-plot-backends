from sympy import (
    sin, cos, Piecewise, Sum, Wild, sign, piecewise_fold, Interval, Union,
    FiniteSet, Eq, Ne, Expr
)
from sympy.core.relational import Relational
# NOTE: from sympy import EmptySet is a different thing!!!
from sympy.sets.sets import EmptySet
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, ContourSeries,
    ImplicitSeries, List2DSeries, GeometrySeries, HVLineSeries
)
from spb.graphics.utils import _plot_sympify
from spb.utils import (
    _create_missing_ranges, _preprocess_multiple_ranges
)
import warnings
import param


def _process_piecewise(piecewise, _range, label, **kwargs):
    """Extract the pieces of an univariate Piecewise function and create the
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


def _process_summations(sum_bound, expr):
    """Substitute oo (infinity) lower/upper bounds of a summation with an
    arbitrary big integer number.

    Parameters
    ==========

    NOTE:
    Let's consider the following summation: ``Sum(1 / x**2, (x, 1, oo))``.
    The current implementation of lambdify (SymPy 1.9 at the time of
    writing this) will create something of this form:
    ``sum(1 / x**2 for x in range(1, INF))``
    The problem is that ``type(INF)`` is float, while ``range`` requires
    integers, thus the evaluation will fails.
    Instead of modifying ``lambdify`` (which requires a deep knowledge),
    let's apply this quick dirty hack: substitute symbolic ``oo`` with an
    arbitrary large number.
    """
    def new_bound(t, bound):
        if (not t.is_number) or t.is_finite:
            return t
        if sign(t) >= 0:
            return bound
        return -bound

    # select summations whose lower/upper bound is infinity
    w = Wild("w", properties=[
        lambda t: isinstance(t, Sum),
        lambda t: any((not a[1].is_finite) or (not a[2].is_finite) for i, a in enumerate(t.args) if i > 0)
    ])

    for t in list(expr.find(w)):
        sums_args = list(t.args)
        for i, a in enumerate(sums_args):
            if i > 0:
                sums_args[i] = (
                    a[0], new_bound(a[1], sum_bound),
                    new_bound(a[2], sum_bound)
                )
        s = Sum(*sums_args)
        expr = expr.subs(t, s)
    return expr


def _build_line_series(expr, r, label, **kwargs):
    """Loop over the provided arguments. If a piecewise function is found,
    decompose it in such a way that each argument gets its own series.
    """
    series = []
    pp = kwargs.get("process_piecewise", False)
    sum_bound = int(kwargs.get("sum_bound", 1000))
    if not callable(expr) and expr.has(Piecewise) and pp:
        series += _process_piecewise(expr, r, label, **kwargs)
    else:
        if not callable(expr):
            expr = _process_summations(sum_bound, expr)
        series.append(LineOver1DRangeSeries(expr, r, label, **kwargs))
    return series


def line(expr, range=None, label=None, rendering_kw=None, **kwargs):
    """Plot a function of one variable over a 2D space.

    Parameters
    ==========

    expr : Expr or callable
        It can either be a symbolic expression representing the function of
        one variable to be plotted, or a numerical function of one variable,
        supporting vectorization. In the latter case the following keyword
        arguments are not supported: ``params``, ``sum_bound``.
    range : (symbol, min, max)
        A 3-tuple denoting the range of the x variable. Default values:
        `min=-10` and `max=10`.
    label : str, optional
        The label to be shown in the legend. If not provided, the string
        representation of ``expr`` will be used.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    adaptive : bool, optional
        Setting ``adaptive=True`` activates the adaptive algorithm
        implemented in [python-adaptive]_ to create smooth plots.
        Use ``adaptive_goal`` and ``loss_fn`` to further customize the output.

        The default value is ``False``, which uses an uniform sampling
        strategy, where the number of discretization points is specified by
        the ``n`` keyword argument.
    adaptive_goal : callable, int, float or None
        Controls the "smoothness" of the evaluation. Possible values:

        * ``None`` (default):  it will use the following goal:
          ``lambda l: l.loss() < 0.01``
        * number (int or float). The lower the number, the more
          evaluation points. This number will be used in the following goal:
          ``lambda l: l.loss() < number``
        * callable: a function requiring one input element, the learner. It
          must return a float number. Refer to [python-adaptive]_ for
          more information.
    color_func : callable or Expr, optional
        Define the line color mapping. It can either be:

        * A numerical function of 2 variables, x, y (the points computed by
          the internal algorithm) supporting vectorization.
        * A symbolic expression having at most as many free symbols as
          ``expr``.
        * None: the default value (no color mapping).
    detect_poles : boolean or str, optional
        Chose whether to detect and correctly plot poles. There are two
        algorithms at work:

        1. based on the gradient of the numerical data, it introduces NaN
           values at locations where the steepness is greater than some
           threshold. This splits the line into multiple segments. To improve
           detection, increase the number of discretization points ``n``
           and/or change the value of ``eps``.
        2. a symbolic approach based on the ``continuous_domain`` function
           from the ``sympy.calculus.util`` module, which computes the
           locations of discontinuities. If any is found, vertical lines
           will be shown.

        Possible options:

        * ``True``: activate poles detection computed with the numerical
          gradient.
        * ``False``: no poles detection.
        * ``"symbolic"``: use both numerical and symbolic algorithms.

        Default to ``False``.
    eps : float, optional
        An arbitrary small value used by the ``detect_poles`` algorithm.
        Default value to 0.1. Before changing this value, it is recommended to
        increase the number of discretization points.
    exclude : list, optional
        A list of numerical values in the horizontal coordinate which are
        going to be excluded from the plot. In practice, it introduces
        discontinuities in the resulting line.
    force_real_eval : boolean, optional
        Default to False, with which the numerical evaluation is attempted
        over a complex domain, which is slower but produces correct results.
        Set this to True if performance is of paramount importance, but be
        aware that it might produce wrong results. It only works with
        ``adaptive=False``.
    scatter : boolean, optional
        Default to False, which will render a line connecting all the points.
        If True, a scatter plot will be generated.
    fill : boolean, optional
        Default to True, which will render empty circular markers. It only
        works if ``scatter=True``.
        If False, filled circular markers will be rendered.
    loss_fn : callable or None
        The loss function to be used by the ``adaptive`` learner.
        Possible values:

        * ``None`` (default): it will use the ``default_loss`` from the
          adaptive module.
        * callable : Refer to [python-adaptive]_ for more information.
          Specifically, look at ``adaptive.learner.learner1D`` to find
          more loss functions.
    n : int, optional
        Used when the ``adaptive=False``: the function is uniformly
        sampled at ``n`` number of points. Default value to 1000.
        If the ``adaptive=True``, this parameter will be ignored.
    only_integers : boolean, optional
        Default to ``False``. If ``True``, discretize the domain with integer
        numbers. It only works when ``adaptive=False``.
        When ``only_integers=True``, the number of discretization points is
        choosen by the algorithm.
    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot, which doesn't support the
        adaptive algorithm (meaning it will use ``adaptive=False``).
        Learn more by reading the documentation of the interactive sub-module.
    steps : {'pre', 'post', 'mid', False}, default: False, optional
        If set, it connects consecutive points with steps rather than
        straight segments.
    sum_bound : int, optional
        When plotting sums, the expression will be pre-processed in order
        to replace lower/upper bounds set to +/- infinity with this +/-
        numerical value. Default value to 1000. Note: the higher this number,
        the slower the evaluation.
    tx, ty : callable, optional
        Apply a numerical function to the discretized x-direction or to the
        output of the numerical evaluation, the y-direction.
    xscale : 'linear' or 'log', optional
        Sets the scaling of the discretized range. Default to ``'linear'``.

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
       [0]: cartesian line: x**2 for x over (-5.0, 5.0)

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
       >>> graphics(
       ...     line(expr, (y, 2, 10), sum_bound=1e03),
       ...     title="$%s$" % latex(expr)
       ... )
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
       >>> graphics(
       ...     line(expr, (y, 2, 10), scatter=True),
       ...     title="$%s$" % latex(expr)
       ... )
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
       >>> graphics(
       ...     line(
       ...         tan(x), (x, -1.5*pi, 1.5*pi),
       ...         adaptive=True, adaptive_goal=0.001,
       ...         detect_poles="symbolic", tx=np.rad2deg
       ...     ),
       ...     ylim=(-7, 7), xlabel="x [deg]", grid=False
       ... )
       Plot object containing:
       [0]: cartesian line: tan(x) for x over (-4.71238898038469, 4.71238898038469)

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
       [0]: cartesian line: floor(x)/x for x over (-3.25, 3.25)

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
       [0]: cartesian line: x - 2 for x over (0.0, 10.0)
       [1]: cartesian line: x for x over (0.0, 10.0)
       [2]: cartesian line: x + 2 for x over (0.0, 10.0)

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
       [0]: cartesian line: 5*sin(x) + 1/cos(10*x) for x over (-5.0, 5.0)
       [1]: cartesian line: 5*sin(x) for x over (-5.0, 5.0)

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
    range = _create_missing_ranges(
        [expr], [range] if range else [], 1, params)[0]
    return _build_line_series(
        expr, range, label, rendering_kw=rendering_kw, **kwargs)



# class line(param.ParameterizedFunction, LineOver1DRangeSeries):
#     """Plot a function of one variable over a 2D space.

#     Returns
#     =======

#     series : list
#         A list containing one instance of ``LineOver1DRangeSeries``.

#     Examples
#     ========

#     .. plot::
#        :context: close-figs
#        :format: doctest
#        :include-source: True

#        >>> from sympy import symbols, sin, pi, tan, exp, cos, log, floor
#        >>> from spb import *
#        >>> x, y = symbols('x, y')

#     Single Plot

#     .. plot::
#        :context: close-figs
#        :format: doctest
#        :include-source: True

#        >>> graphics(line(x**2, (x, -5, 5)))
#        Plot object containing:
#        [0]: cartesian line: x**2 for x over (-5.0, 5.0)

#     Multiple functions over the same range with custom rendering options:

#     .. plot::
#        :context: close-figs
#        :format: doctest
#        :include-source: True

#        >>> graphics(
#        ...     line(x, (x, -3, 3)),
#        ...     line(log(x), (x, -3, 3), rendering_kw={"linestyle": "--"}),
#        ...     line(exp(x), (x, -3, 3), rendering_kw={"linestyle": ":"}),
#        ...     aspect="equal", ylim=(-3, 3)
#        ... )
#        Plot object containing:
#        [0]: cartesian line: x for x over (-3.0, 3.0)
#        [1]: cartesian line: log(x) for x over (-3.0, 3.0)
#        [2]: cartesian line: exp(x) for x over (-3.0, 3.0)

#     Plotting a summation in which the free symbol of the expression is not
#     used in the lower/upper bounds:

#     .. plot::
#        :context: close-figs
#        :format: doctest
#        :include-source: True

#        >>> from sympy import Sum, oo, latex
#        >>> expr = Sum(1 / x ** y, (x, 1, oo))
#        >>> graphics(
#        ...     line(expr, (y, 2, 10), sum_bound=1e03),
#        ...     title="$%s$" % latex(expr)
#        ... )
#        Plot object containing:
#        [0]: cartesian line: Sum(x**(-y), (x, 1, 1000)) for y over (2.0, 10.0)

#     Plotting a summation in which the free symbol of the expression is
#     used in the lower/upper bounds. Here, the discretization variable must
#     assume integer values:

#     .. plot::
#        :context: close-figs
#        :format: doctest
#        :include-source: True

#        >>> expr = Sum(1 / x, (x, 1, y))
#        >>> graphics(
#        ...     line(expr, (y, 2, 10), scatter=True),
#        ...     title="$%s$" % latex(expr)
#        ... )
#        Plot object containing:
#        [0]: cartesian line: Sum(1/x, (x, 1, y)) for y over (2.0, 10.0)

#     Using an adaptive algorithm, detect and plot vertical lines at
#     singularities. Also, apply a transformation function to the discretized
#     domain in order to convert radians to degrees:

#     .. plot::
#        :context: close-figs
#        :format: doctest
#        :include-source: True

#        >>> import numpy as np
#        >>> graphics(
#        ...     line(
#        ...         tan(x), (x, -1.5*pi, 1.5*pi),
#        ...         adaptive=True, adaptive_goal=0.001,
#        ...         detect_poles="symbolic", tx=np.rad2deg
#        ...     ),
#        ...     ylim=(-7, 7), xlabel="x [deg]", grid=False
#        ... )
#        Plot object containing:
#        [0]: cartesian line: tan(x) for x over (-4.71238898038469, 4.71238898038469)

#     Introducing discontinuities by excluding specified points:

#     .. plot::
#        :context: close-figs
#        :format: doctest
#        :include-source: True

#        >>> graphics(
#        ...     line(floor(x) / x, (x, -3.25, 3.25), exclude=list(range(-4, 5))),
#        ...     ylim=(-1, 5)
#        ... )
#        Plot object containing:
#        [0]: cartesian line: floor(x)/x for x over (-3.25, 3.25)

#     Creating a step plot:

#     .. plot::
#        :context: close-figs
#        :format: doctest
#        :include-source: True

#        >>> graphics(
#        ...     line(x-2, (x, 0, 10), only_integers=True, steps="pre", label="pre"),
#        ...     line(x, (x, 0, 10), only_integers=True, steps="mid", label="mid"),
#        ...     line(x+2, (x, 0, 10), only_integers=True, steps="post", label="post"),
#        ... )
#        Plot object containing:
#        [0]: cartesian line: x - 2 for x over (0.0, 10.0)
#        [1]: cartesian line: x for x over (0.0, 10.0)
#        [2]: cartesian line: x + 2 for x over (0.0, 10.0)

#     Advanced example showing:

#     * detect singularities by setting ``adaptive=False`` (better performance),
#       increasing the number of discretization points (in order to have
#       'vertical' segments on the lines) and reducing the threshold for the
#       singularity-detection algorithm.
#     * application of color function.
#     * combining together multiple lines.

#     .. plot::
#        :context: close-figs
#        :format: doctest
#        :include-source: True

#        >>> import numpy as np
#        >>> expr = 1 / cos(10 * x) + 5 * sin(x)
#        >>> def cf(x, y):
#        ...     # map a colormap to the distance from the origin
#        ...     d = np.sqrt(x**2 + y**2)
#        ...     # visibility of the plot is limited: ylim=(-10, 10). However,
#        ...     # some of the y-values computed by the function are much higher
#        ...     # (or lower). Filter them out in order to have the entire
#        ...     # colormap spectrum visible in the plot.
#        ...     offset = 12 # 12 > 10 (safety margin)
#        ...     d[(y > offset) | (y < -offset)] = 0
#        ...     return d
#        >>> graphics(
#        ...     line(
#        ...         expr, (x, -5, 5), "distance from (0, 0)",
#        ...         rendering_kw={"cmap": "plasma"},
#        ...         adaptive=False, detect_poles=True, n=3e04,
#        ...         eps=1e-04, color_func=cf),
#        ...     line(5 * sin(x), (x, -5, 5), rendering_kw={"linestyle": "--"}),
#        ...     ylim=(-10, 10), title="$%s$" % latex(expr)
#        ... )
#        Plot object containing:
#        [0]: cartesian line: 5*sin(x) + 1/cos(10*x) for x over (-5.0, 5.0)
#        [1]: cartesian line: 5*sin(x) for x over (-5.0, 5.0)

#     Interactive-widget plot of an oscillator. Refer to the interactive
#     sub-module documentation to learn more about the ``params`` dictionary.
#     This plot illustrates:

#     * plotting multiple expressions, each one with its own label and
#       rendering options.
#     * the use of ``prange`` (parametric plotting range).
#     * the use of the ``params`` dictionary to specify sliders in
#       their basic form: (default, min, max).
#     * the use of :py:class:`panel.widgets.slider.RangeSlider`, which is a
#       2-values widget. In this case it is used to enforce the condition
#       `f1 < f2`.
#     * the use of a parametric title, specified with a tuple of the form:
#       ``(title_str, param_symbol1, ...)``, where:

#       * ``title_str`` must be a formatted string, for example:
#         ``"test = {:.2f}"``.
#       * ``param_symbol1, ...`` must be a symbol or a symbolic expression
#         whose free symbols are contained in the ``params`` dictionary.

#     .. panel-screenshot::
#        :small-size: 800, 625

#        from sympy import *
#        from spb import *
#        import panel as pn
#        x, y, f1, f2, d, n = symbols("x, y, f_1, f_2, d, n")
#        params = {
#            (f1, f2): pn.widgets.RangeSlider(
#                value=(1, 2), start=0, end=10, step=0.1),     # frequencies
#            d: (0.25, 0, 1),   # damping
#            n: (2, 0, 4)       # multiple of pi
#        }
#        graphics(
#            line(cos(f1 * x) * exp(-d * x), prange(x, 0, n * pi),
#                label="oscillator 1", params=params),
#            line(cos(f2 * x) * exp(-d * x), prange(x, 0, n * pi),
#                label="oscillator 2", params=params),
#            line(exp(-d * x), prange(x, 0, n * pi), label="upper limit",
#                rendering_kw={"linestyle": ":"}, params=params),
#            line(-exp(-d * x), prange(x, 0, n * pi), label="lower limit",
#                rendering_kw={"linestyle": ":"}, params=params),
#            ylim=(-1.25, 1.25),
#            title=("$f_1$ = {:.2f} Hz", f1)
#        )

#     See Also
#     ========

#     line_parametric_2d, line_polar, implicit_2d, list_2d, geometry,
#     spb.graphics.functions_3d.line_parametric_3d

#     """

#     range = param.Tuple(doc="""
#         A 3-tuple (symbol, min, max) denoting the range of the variable
#         to be shown on the horizontal axis.
#         Default values: `min=-10` and `max=10`.""")
#     sum_bound = param.Integer(default=1000, doc="""
#         When plotting sums, the expression will be pre-processed in order
#         to replace lower/upper bounds set to +/- infinity with this +/-
#         numerical value. Default value to 1000. Note: the higher this number,
#         the slower the evaluation.""")

#     def __call__(self, expr, range=None, label=None, rendering_kw=None, **kwargs):
#         p = ParamOverrides(self, kwargs)
#         expr = _plot_sympify(expr)
#         params = kwargs.get("params", {})
#         range = _create_missing_ranges(
#             [expr], [range] if range else [], 1, params)[0]
#         return _build_line_series(
#             expr, range, label, rendering_kw=rendering_kw, **kwargs)


def line_parametric_2d(
    expr1, expr2, range=None, label=None, rendering_kw=None,
    colorbar=True, use_cm=True, **kwargs
):
    """Plots a 2D parametric curve.

    Parameters
    ==========

    expr1, expr2 : Expr or callable
        The expression representing the horizontal and vertical components
        of the parametric function.
        It can either be a symbolic expression representing the function of
        one variable to be plotted, or a numerical function of one variable,
        supporting vectorization. In the latter case the following keyword
        arguments are not supported: ``params``, ``sum_bound``.
    range : (symbol, min, max)
        A 3-tuple denoting the parameter symbol, start value and stop value.
        For example, ``(u, 0, 5)``. If the range is not specified, then a
        default range of (-10, 10) is used.
    label : str, optional
        The label to be shown in the legend. If not provided, the string
        representation of ``expr1`` and ``expr1`` will be used.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    adaptive : bool, optional
        Setting ``adaptive=True`` activates the adaptive algorithm
        implemented in [python-adaptive]_ to create smooth plots.
        Use ``adaptive_goal`` and ``loss_fn`` to further customize the output.

        The default value is ``False``, which uses an uniform sampling
        strategy, where the number of discretization points is specified by
        the ``n`` keyword argument.
    adaptive_goal : callable, int, float or None
        Controls the "smoothness" of the evaluation. Possible values:

        * ``None`` (default):  it will use the following goal:
          ``lambda l: l.loss() < 0.01``
        * number (int or float). The lower the number, the more
          evaluation points. This number will be used in the following goal:
          ``lambda l: l.loss() < number``
        * callable: a function requiring one input element, the learner. It
          must return a float number. Refer to [python-adaptive]_ for
          more information.
    colorbar : boolean, optional
        Show/hide the colorbar. Default to True (colorbar is visible).
        Only works when ``use_cm=True``.
    color_func : callable, optional
        Define the line color mapping when ``use_cm=True``. It can either be:

        * A numerical function supporting vectorization. The arity can be:

          * 1 argument: ``f(t)``, where ``t`` is the parameter.
          * 2 arguments: ``f(x, y)`` where ``x, y`` are the coordinates of
            the points.
          * 3 arguments: ``f(x, y, t)``.

        * A symbolic expression having at most as many free symbols as
          ``expr1`` or ``expr2``.
        * None: the default value (color mapping applied to the parameter).
    exclude : list, optional
        A list of numerical values along the parameter which are going to
        be excluded from the plot. In practice, it introduces discontinuities
        in the resulting line.
    force_real_eval : boolean, optional
        Default to False, with which the numerical evaluation is attempted
        over a complex domain, which is slower but produces correct results.
        Set this to True if performance is of paramount importance, but be
        aware that it might produce wrong results. It only works with
        ``adaptive=False``.
    loss_fn : callable or None
        The loss function to be used by the adaptive learner.
        Possible values:

        * ``None`` (default): it will use the ``default_loss`` from the
          adaptive module.
        * callable : Refer to [python-adaptive]_ for more information.
          Specifically, look at ``adaptive.learner.learner1D`` to find
          more loss functions.
    n : int, optional
        Used when the ``adaptive=False``. The function is uniformly sampled
        at ``n`` number of points. Default value to 1000.
        If the ``adaptive=True``, this parameter will be ignored.
    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot, which doesn't support the
        adaptive algorithm (meaning it will use ``adaptive=False``).
        Learn more by reading the documentation of the interactive sub-module.
    tx, ty, tp : callable, optional
        Apply a numerical function to the x-direction, y-direction and
        parameter, respectively.
    use_cm : boolean, optional
        If True, apply a color map to the parametric lines.
        If False, solid colors will be used instead. Default to True.
    xscale : 'linear' or 'log', optional
        Sets the scaling of the parameter.

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
       [0]: parametric cartesian line: (5*cos(2*u/3) + 2*cos(u), -5*sin(2*u/3) + 2*sin(u)) for u over (0.0, 18.84955592153876)

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
       [0]: parametric cartesian line: (3*cos(u), 3*sin(u)) for u over (0.0, 6.283185307179586)
       [1]: parametric cartesian line: (3*cos(2*v), 5*sin(4*v)) for v over (0.0, 3.141592653589793)

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
       [0]: parametric cartesian line: (log(floor(t))*cos(t), log(floor(t))*sin(t)) for t over (1.0, 12.566370614359172)

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
    expr1, expr2 = map(_plot_sympify, [expr1, expr2])
    params = kwargs.get("params", {})
    range = _create_missing_ranges(
        [expr1, expr2], [range] if range else [], 1, params)[0]
    s = Parametric2DLineSeries(
        expr1, expr2, range, label,
        rendering_kw=rendering_kw, colorbar=colorbar,
        use_cm=use_cm, **kwargs)
    return [s]


def line_polar(expr, range=None, label=None, rendering_kw=None, **kwargs):
    """Creates a 2D polar plot of a function of one variable.

    This function executes ``line_parametric_2d``. Refer to its documentation
    for a full list of keyword arguments.

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
       [0]: parametric cartesian line: (3*sin(2*theta)*cos(theta), 3*sin(theta)*sin(2*theta)) for theta over (0.0, 6.283185307179586)

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
    range = _create_missing_ranges(
        [expr], [range] if range else [], 1, params)[0]
    theta = range[0]
    return line_parametric_2d(
        expr * cos(theta), expr * sin(theta), range,
        label, rendering_kw, **kwargs)


def contour(
    expr, range1=None, range2=None, label=None, rendering_kw=None,
    colorbar=True, clabels=True, fill=True, **kwargs
):
    """Plots contour lines or filled contours of a function of two variables.

    Parameters
    ==========

    clabels : bool, optional
        Visualize labels of contour lines. Only works when ``fill=False``.
        Default to True. Note that some backend might not implement this
        feature.
    colorbar : boolean, optional
        Show/hide the colorbar. Default to True (colorbar is visible).
        Only works when ``use_cm=True``.
    fill : bool, optional
        Choose between filled contours or line contours. Default to True
        (filled contours).
    **kwargs :
        Keyword arguments are the same as
        :func:`~spb.graphics.functions_3d.surface`.
        Refer to its documentation for a for a full list of keyword arguments.

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
       [0]: contour: exp(-x**2/10 - y**2/10)*cos(x**2 + y**2) for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    Line contours of a function of two variables.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = 5 * (cos(x) - 0.2 * sin(y))**2 + 5 * (-0.2 * cos(x) + sin(y))**2
       >>> graphics(
       ...     contour(expr, (x, 0, 2 * pi), (y, 0, 2 * pi), fill=False)
       ... )
       Plot object containing:
       [0]: contour: 5*(-0.2*sin(y) + cos(x))**2 + 5*(sin(y) - 0.2*cos(x))**2 for x over (0.0, 6.283185307179586) and y over (0.0, 6.283185307179586)

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
       ...     grid=False
       ... )
       Plot object containing:
       [0]: contour: 5*(-0.2*sin(y) + cos(x))**2 + 5*(sin(y) - 0.2*cos(x))**2 for x over (0.0, 6.283185307179586) and y over (0.0, 6.283185307179586)
       [1]: contour: 5*(-0.2*sin(y) + cos(x))**2 + 5*(sin(y) - 0.2*cos(x))**2 for x over (0.0, 6.283185307179586) and y over (0.0, 6.283185307179586)

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
       ...         eq1.rewrite(Add), (x, 0, 2 * pi), (y, 0, 2 * pi),
       ...         rendering_kw={"levels": [0]},
       ...         fill=False, clabels=False),
       ...     contour(
       ...         eq2.rewrite(Add), (x, 0, 2 * pi), (y, 0, 2 * pi),
       ...         rendering_kw={"levels": [0]},
       ...         fill=False, clabels=False),
       ... )
       Plot object containing:
       [0]: contour: 3*(-sin(x) + cos(y)/2)**2 + (-sin(y)/2 + cos(x))**2 - 2 for x over (0.0, 6.283185307179586) and y over (0.0, 6.283185307179586)
       [1]: contour: -(sin(x) + 2*cos(y))**2 + (-2*sin(y) + cos(x))**2 - 3 for x over (0.0, 6.283185307179586) and y over (0.0, 6.283185307179586)

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
       [0]: contour: sin(2*r)*cos(theta) for theta over (0.0, 6.283185307179586) and r over (0.0, 7.0)

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
    expr = _plot_sympify(expr)
    params = kwargs.get("params", {})
    if not (range1 and range2):
        warnings.warn(
            "No ranges were provided. This function will attempt to find "
            "them, however the order will be arbitrary, which means the "
            "visualization might be flipped."
        )
    ranges = _preprocess_multiple_ranges([expr], [range1, range2], 2, params)
    s = ContourSeries(
        expr, *ranges, label, rendering_kw=rendering_kw,
        colorbar=colorbar, fill=fill, clabels=clabels, **kwargs)
    return [s]


def implicit_2d(
    expr, range1=None, range2=None, label=None, rendering_kw=None,
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

    expr : Expr, Relational, BooleanFunction
        The equation / inequality that is to be plotted.
    range1, range2 : tuples or Symbol
        Tuple denoting the discretization domain, for example:
        ``(x, -10, 10)``.
    label : str, optional
        The label to be shown when multiple expressions are plotted.
        If not provided, the string representation of the expression
        will be used.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of contours. Refer to the
        plotting library (backend) manual for more informations.
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
    adaptive : bool, optional
        The default value is set to ``False``, meaning that the internal
        algorithm uses a mesh grid approach. In such case, Boolean
        combinations of expressions cannot be plotted.
        If set to ``True``, the internal algorithm uses interval arithmetic.
        If the expression cannot be plotted with interval arithmetic, it
        switches to the meshgrid approach.
    depth : integer
        The depth of recursion for adaptive grid. Default value is 0.
        Takes value in the range (0, 4).
        Think of the resulting plot as a picture composed by pixels. By
        increasing ``depth`` we are increasing the number of pixels, thus
        obtaining a more accurate plot.
    n, n1, n2 : int
        Number of discretization points in the horizontal and vertical
        directions when ``adaptive=False``. Default to 100. ``n`` is a shortcut
        to set the same number of discretization points on both directions.
    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot. Learn more by reading the
        documentation of the interactive sub-module.
    show_in_legend : bool
        If True, add a legend entry for the expression being plotted.
        This option is useful to hide a particular expression when combining
        together multiple plots. Default to True.

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
       [0]: Implicit expression: Eq(x - 1, 0) for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    Plot a region:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> graphics(
       ...     implicit_2d(y > x**2, (x, -5, 5), (y, -10, 10), n=150),
       ...     grid=False)
       Plot object containing:
       [0]: Implicit expression: y > x**2 for x over (-5.0, 5.0) and y over (-10.0, 10.0)

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
       [0]: Implicit expression: 4*(-sin(y)/5 + cos(x))**2 + 4*(sin(y) - cos(x)/5)**2 <= pi for x over (-3.141592653589793, 3.141592653589793) and y over (-3.141592653589793, 3.141592653589793)
       [1]: Implicit expression: Eq(-4*(-sin(y)/5 + cos(x))**2 - 4*(sin(y) - cos(x)/5)**2 + pi, 0) for x over (-3.141592653589793, 3.141592653589793) and y over (-3.141592653589793, 3.141592653589793)

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
       >>> series = []
       >>> for L_val in L_array:
       ...     series += implicit_2d(
       ...         expr.subs({b: b_val, L: L_val}), (t, 0, 3), (V, 0, 1000),
       ...         label="L = %s" % L_val)
       >>> graphics(*series)
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
       [0]: Implicit expression: Eq(x*y - 20, 15*y) for x over (15.0, 30.0) and y over (0.0, 50.0)
       [1]: Implicit expression: Eq(y*(x - 3) - 20, 15*y) for x over (15.0, 30.0) and y over (0.0, 50.0)
       [2]: Implicit expression: Eq(y*(x - 6) - 20, 15*y) for x over (15.0, 30.0) and y over (0.0, 50.0)

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
    expr = _plot_sympify(expr)
    params = kwargs.get("params", {})

    if not (range1 and range2):
        warnings.warn(
            "No ranges were provided. This function will attempt to find "
            "them, however the order will be arbitrary, which means the "
            "visualization might be flipped."
        )

    print("implicit_2d", kwargs)

    series = []
    ranges = _preprocess_multiple_ranges([expr], [range1, range2], 2, params)
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


def list_2d(coord_x, coord_y, label=None, rendering_kw=None, **kwargs):
    """Plots lists of coordinates.

    Parameters
    ==========

    coord_x, coord_y : list or tuple
        List of coordinates.
    label : str, optional
        The label to be shown in the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
    scatter : boolean, optional
        Default to False, which will render a line connecting all the points.
        If True, a scatter plot will be generated.
    fill : boolean, optional
        Default to False, which will render empty circular markers. It only
        works if ``scatter=True``.
        If True, filled circular markers will be rendered.
    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot. Learn more by reading the
        documentation of the interactive sub-module.

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
    if not hasattr(coord_x, "__iter__"):
        coord_x = [coord_x]
    if not hasattr(coord_y, "__iter__"):
        coord_y = [coord_y]
    s = List2DSeries(
        coord_x, coord_y, label, rendering_kw=rendering_kw, **kwargs)
    return [s]


def geometry(geom, label=None, rendering_kw=None, fill=True, **kwargs):
    """Plot entities from the sympy.geometry module.

    Parameters
    ==========

    geom : GeometryEntity
        Represent the geometric entity to be plotted.
    label : str, optional
        The name of the geometry entity to be eventually shown on the
        legend. If not provided, the string representation of ``geom``
        will be used.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines or fills. Refer to
        the plotting library (backend) manual for more informations.
    fill : boolean
        Default to True. Fill the polygon/circle/ellipse.
    A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot. Learn more by reading the
        documentation of the interactive sub-module.

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
       [0]: geometry entity: Ellipse(Point2D(-3, 2), 3, 9/5)

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
       [0]: geometry entity: Circle(Point2D(0, 0), 5)
       [1]: geometry entity: Ellipse(Point2D(-3, 2), 3, 9/5)
       [2]: geometry entity: RegularPolygon(Point2D(4, 0), 4, 5, 0)
       [3]: parametric cartesian line: (cos(x), sin(x)) for x over (0.0, 6.283185307179586)
       [4]: geometry entity: Segment2D(Point2D(-4, -6), Point2D(6, 6))
       [5]: geometry entity: Point2D(0, 0)

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
    s = GeometrySeries(
        geom, label=label, rendering_kw=rendering_kw, fill=fill, **kwargs)
    return [s]


def hline(v, label=None, rendering_kw=None, show_in_legend=True, **kwargs):
    """Create an horizontal line at a given location in a 2D space.

    Parameters
    ==========
    v : float or Expr
        The y-coordinate where to draw the horizontal line.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    show_in_legend : bool, optional
        Show/hide the line from the legend. Default to True (line is visible).

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
       [0]: cartesian line: cos(x) for x over (-3.141592653589793, 3.141592653589793)
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
        HVLineSeries(
            v, True, label,
            rendering_kw=rendering_kw,
            show_in_legend=show_in_legend, **kwargs)
    ]


def vline(v, label=None, rendering_kw=None, show_in_legend=True, **kwargs):
    """Create an horizontal line at a given location in a 2D space.

    Parameters
    ==========
    v : float or Expr
        The x-coordinate where to draw the horizontal line.
    label : str, optional
        The label to be shown on the legend.
    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
    show_in_legend : bool, optional
        Show/hide the line from the legend. Default to True (line is visible).

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
       [0]: cartesian line: cos(x) for x over (-3.141592653589793, 3.141592653589793)
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
        HVLineSeries(
            v, False, label,
            rendering_kw=rendering_kw,
            show_in_legend=show_in_legend, **kwargs)
    ]
