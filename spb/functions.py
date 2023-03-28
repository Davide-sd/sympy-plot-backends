"""Plotting module for Sympy.

A plot is represented by the ``Plot`` class that contains a list of the data
series to be plotted. The data series are instances of classes meant to
simplify getting points and meshes from sympy expressions.

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

from spb.defaults import TWO_D_B, THREE_D_B, cfg
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    SurfaceOver2DRangeSeries, ContourSeries, ParametricSurfaceSeries,
    ImplicitSeries, _set_discretization_points,
    List2DSeries, List3DSeries, GeometrySeries, Implicit3DSeries,
    GenericDataSeries, ComplexParametric3DLineSeries
)
from spb.interactive import create_interactive_plot
from spb.utils import (
    _plot_sympify, _check_arguments, _unpack_args, _instantiate_backend,
    spherical_to_cartesian, prange
)
from sympy import (
    latex, Tuple, Expr, Symbol, Wild, oo, Sum, sign, Piecewise, piecewise_fold,
    Plane, FiniteSet, Interval, Union, cos, sin, pi, sympify, atan2, sqrt,
    Dummy, symbols, I, re, im
)
# NOTE: from sympy import EmptySet is a different thing!!!
from sympy.sets.sets import EmptySet
from sympy.vector import BaseScalar
from sympy.external import import_module

# N.B.
# When changing the minimum module version for matplotlib, please change
# the same in the `SymPyDocTestFinder` in `sympy/testing/runtests.py`


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
    # only contains Line2DSeries with is_filled=True. They have higher
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
                if xx[0] != _range[1]:
                    correct_list = series if _set.left_open else filled_series
                    correct_list.append(
                        List2DSeries([xx[0]], [yy[0]], is_point=True,
                            is_filled=not _set.left_open, **kwargs)
                    )
                if xx[-1] != _range[2]:
                    correct_list = series if _set.right_open else filled_series
                    correct_list.append(
                        List2DSeries([xx[-1]], [yy[-1]], is_point=True,
                            is_filled=not _set.right_open, **kwargs)
                    )
        elif isinstance(_set, FiniteSet):
            loc, val = [], []
            for _loc in _set.args:
                loc.append(float(_loc))
                val.append(float(expr.evalf(subs={_range[0]: _loc})))
            filled_series.append(List2DSeries(loc, val, is_point=True,
                is_filled=True, **kwargs))
            if not from_union:
                c += 1
        elif isinstance(_set, Union):
            for _s in _set.args:
                c = func(expr, _s, c, from_union=True)
        elif isinstance(_set, EmptySet):
            # in this case, some pieces are outside of the provided range.
            # don't add any series, but increase the counter nonetheless so that
            # there is one-to-one correspondance between the expression and
            # what is plotted.
            if not from_union:
                c += 1
        else:
            raise TypeError(
                "Unhandle situation:\n" +
                "expr: {}\ncond: {}\ntype(cond): {}\n".format(str(expr),
                    _set, type(_set)) +
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


def _process_summations(sum_bound, *args):
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

    args = list(args)
    expr = args[0]

    # select summations whose lower/upper bound is infinity
    w = Wild("w", properties=[
        lambda t: isinstance(t, Sum),
        lambda t: any((not a[1].is_finite) or (not a[2].is_finite) for i, a in enumerate(t.args) if i > 0)
    ])

    for t in list(expr.find(w)):
        sums_args = list(t.args)
        for i, a in enumerate(sums_args):
            if i > 0:
                sums_args[i] = (a[0], new_bound(a[1], sum_bound),
                    new_bound(a[2], sum_bound))
        s = Sum(*sums_args)
        expr = expr.subs(t, s)
    args[0] = expr
    return args


def _build_line_series(*args, **kwargs):
    """Loop over the provided arguments. If a piecewise function is found,
    decompose it in such a way that each argument gets its own series.
    """
    series = []
    pp = kwargs.get("process_piecewise", False)
    sum_bound = int(kwargs.get("sum_bound", 1000))
    for arg in args:
        expr, r, label, rendering_kw = arg
        kw = kwargs.copy()
        if rendering_kw is not None:
            kw["rendering_kw"] = rendering_kw
        if not callable(expr) and expr.has(Piecewise) and pp:
            series += _process_piecewise(expr, r, label, **kw)
        else:
            if not callable(expr):
                arg = _process_summations(sum_bound, *arg)
            series.append(LineOver1DRangeSeries(*arg[:-1], **kw))
    return series


def _set_labels(series, labels, rendering_kw):
    """Apply the label keyword argument to the series.
    """
    # NOTE: this function is a workaround until a better integration is
    # achieved between iplot and all other plotting functions.
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    if len(labels) > 0:
        if len(series) != len(labels):
            raise ValueError("The number of labels must be equal to the "
                "number of expressions being plotted.\nReceived "
                "{} expressions and {} labels".format(len(series), len(labels)))

        for s, l in zip(series, labels):
            s.label = l

    if rendering_kw is not None:
        if isinstance(rendering_kw, dict):
            rendering_kw = [rendering_kw]
        if len(rendering_kw) == 1:
            rendering_kw *= len(series)
        elif len(series) != len(rendering_kw):
            raise ValueError("The number of rendering dictionaries must be "
                "equal to the number of expressions being plotted.\nReceived "
                "{} expressions and {} labels".format(len(series), len(rendering_kw)))
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

    - Plotting a single expression with a single range.
        `plot(expr, range, **kwargs)`
    - Plotting a single expression with custom rendering options.
        `plot(expr, range, rendering_kw, **kwargs)`
    - Plotting a single expression with the default range (-10, 10).
        `plot(expr, **kwargs)`
    - Plotting multiple expressions with a single range.
        `plot(expr1, expr2, ..., range, **kwargs)`
    - Plotting multiple expressions with multiple ranges.
        `plot((expr1, range1), (expr2, range2), ..., **kwargs)`
    - Plotting multiple expressions with custom labels and rendering options.
        `plot((expr1, range1, label1, rendering_kw1), (expr2, range2, label2, rendering_kw2), ..., **kwargs)`

    Parameters
    ==========

    args :
        expr : Expr or callable
            It can either be a:

            * Symbolic expression representing the function of one variable
              to be plotted.
            * Numerical function of one variable, supporting vectorization.
              In this case the following keyword arguments are not supported:
              ``params``, ``sum_bound``.

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
        implemented in [#fn1]_ to create smooth plots. Use ``adaptive_goal``
        and ``loss_fn`` to further customize the output.

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
          must return a float number. Refer to [#fn1]_ for more information.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}. Only available with ``MatplotlibBackend``.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    color_func : callable or Expr, optional
        Define the line color mapping. It can either be:

        * A numerical function of 2 variables, x, y (the points computed by
          the internal algorithm) supporting vectorization.
        * A symbolic expression having at most as many free symbols as
          ``expr``.
        * None: the default value (no color mapping).

    detect_poles : boolean, optional
        Chose whether to detect and correctly plot poles.
        Defaulto to ``False``. To improve detection, increase the number of
        discretization points ``n`` and/or change the value of ``eps``.

    eps : float, optional
        An arbitrary small value used by the ``detect_poles`` algorithm.
        Default value to 0.1. Before changing this value, it is recommended to
        increase the number of discretization points.

    force_real_eval : boolean, optional
        Default to False, with which the numerical evaluation is attempted
        over a complex domain, which is slower but produces correct results.
        Set this to True if performance is of paramount importance, but be
        aware that it might produce wrong results. It only works with
        ``adaptive=False``.

    is_point : boolean, optional
        Default to False, which will render a line connecting all the points.
        If True, a scatter plot will be generated.

    is_filled : boolean, optional
        Default to True, which will render empty circular markers. It only
        works if ``is_point=True``.
        If False, filled circular markers will be rendered.

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of ``expr`` will be used. The number of labels must be
        equal to the number of expressions.

    loss_fn : callable or None
        The loss function to be used by the ``adaptive`` learner.
        Possible values:

        * ``None`` (default): it will use the ``default_loss`` from the
          adaptive module.
        * callable : Refer to [#fn1]_ for more information. Specifically,
          look at ``adaptive.learner.learner1D`` to find more loss functions.

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
        Learn more by reading the documentation of ``iplot``.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

    is_polar : boolean, optional
        Default to False. If True, requests the backend to use a 2D polar
        chart.

    show : bool, optional
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the backend.

    steps : boolean, optional
        Default to ``False``. If ``True``, connects consecutive points with
        steps rather than straight segments.

    sum_bound : int, optional
        When plotting sums, the expression will be pre-processed in order
        to replace lower/upper bounds set to +/- infinity with this +/-
        numerical value. Default value to 1000. Note: the higher this number,
        the slower the evaluation.

    title : str, optional
        Title of the plot.

    tx, ty : callable, optional
        Apply a numerical function to the discretized x-direction or to the
        output of the numerical evaluation, the y-direction.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel : str, optional
        Labels for the x-axis or y-axis, respectively.

    xscale, yscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis or y-axis, respectively.
        Default to ``'linear'``.

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)``, visible in the chart.
        Note that the function is still being evaluated over the specified
        ``range``.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)``, visible in the chart.


    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, sin, pi, tan, exp, cos, log
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
       ...     is_point=True, is_filled=True, title="$%s$" % latex(expr))
       Plot object containing:
       [0]: cartesian line: Sum(1/x, (x, 1, y)) for y over (2.0, 10.0)

    Using an adaptive algorithm, detect singularities and apply a
    transformation function to the discretized domain in order to convert
    radians to degrees:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> plot(tan(x), (x, -1.5*pi, 1.5*pi),
       ...      adaptive=True, adaptive_goal=0.001,
       ...      detect_poles=True, tx=np.rad2deg, ylim=(-7, 7),
       ...      xlabel="x [deg]")
       Plot object containing:
       [0]: cartesian line: tan(x) for x over (-4.71238898038469, 4.71238898038469)


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
           ylim=(-1.25, 1.25), use_latex=False
       )

    References
    ==========

    .. [#fn1] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot_polar, plot_parametric, plot_contour, plot3d, plot3d_parametric_line,
    plot_implicit, plot_geometry, plot_piecewise

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 1, **kwargs)
    params = kwargs.get("params", None)
    free = set()
    for p in plot_expr:
        if not isinstance(p[1][0], str):
            free |= p[1][0].free_symbols
        else:
            free |= set([Symbol(p[1][0])])
    if params:
        free = free.difference(params.keys())
    x = free.pop() if free else Symbol("x")

    fx = lambda use_latex: x.name if not use_latex else latex(x)
    wrap = lambda use_latex: "f(%s)" if not use_latex else r"f\left(%s\right)"
    fy = lambda use_latex: wrap(use_latex) % fx(use_latex)
    kwargs.setdefault("xlabel", fx)
    kwargs.setdefault("ylabel", fy)
    kwargs = _set_discretization_points(kwargs, LineOver1DRangeSeries)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _build_line_series(*plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)
    series += _create_generic_data_series(**kwargs)

    if params:
        return create_interactive_plot(*series, **kwargs)

    Backend = kwargs.pop("backend", TWO_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def plot_parametric(*args, **kwargs):
    """
    Plots a 2D parametric curve.

    Typical usage examples are in the followings:

    - Plotting a single parametric curve with a range
        `plot_parametric(expr_x, expr_y, range)`
    - Plotting multiple parametric curves with the same range
        `plot_parametric((expr_x, expr_y), ..., range)`
    - Plotting multiple parametric curves with different ranges
        `plot_parametric((expr_x, expr_y, range), ...)`
    - Plotting multiple curves with different ranges and custom labels
        `plot_parametric((expr_x, expr_y, range, label), ...)`

    Parameters
    ==========

    args :
        `expr_x` : Expr
            The expression representing x component of the parametric
            function. It can be a:

            * Symbolic expression representing the function of one variable
              to be plotted.
            * Numerical function of one variable, supporting vectorization.
              In this case the following keyword arguments are not supported:
              ``params``.

        `expr_y` : Expr
            The expression representing y component of the parametric
            function. It can be a:

            * Symbolic expression representing the function of one variable
              to be plotted.
            * Numerical function of one variable, supporting vectorization.
              In this case the following keyword arguments are not supported:
              ``params``.

        `range` : (symbol, min, max)
            A 3-tuple denoting the parameter symbol, start and stop. For
            example, ``(u, 0, 5)``. If the range is not specified, then a
            default range of (-10, 10) is used.

            However, if the arguments are specified as
            ``(expr_x, expr_y, range), ...``, you must specify the ranges
            for each expressions manually.

        `label` : str, optional
            The label to be shown in the legend. If not provided, the string
            representation of ``expr_x`` and ``expr_y`` will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines. Refer to the
            plotting library (backend) manual for more informations.

    adaptive : bool, optional
        Setting ``adaptive=True`` activates the adaptive algorithm
        implemented in [#fn2]_ to create smooth plots. Use ``adaptive_goal``
        and ``loss_fn`` to further customize the output.

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
          must return a float number. Refer to [#fn2]_ for more information.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}. Only available with ``MatplotlibBackend``.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    color_func : callable, optional
        Define the line color mapping when ``use_cm=True``. It can either be:

        * A numerical function supporting vectorization. The arity can be:

          * 1 argument: ``f(t)``, where ``t`` is the parameter.
          * 2 arguments: ``f(x, y)`` where ``x, y`` are the coordinates of
            the points.
          * 3 arguments: ``f(x, y, t)``.

        * A symbolic expression having at most as many free symbols as
          ``expr_x`` or ``expr_y``.
        * None: the default value (color mapping applied to the parameter).

    force_real_eval : boolean, optional
        Default to False, with which the numerical evaluation is attempted
        over a complex domain, which is slower but produces correct results.
        Set this to True if performance is of paramount importance, but be
        aware that it might produce wrong results. It only works with
        ``adaptive=False``.

    label : str or list/tuple, optional
        The label to be shown in the legend or in the colorbar. If not
        provided, the string representation of `expr` will be used. The number
        of labels must be equal to the number of expressions.

    loss_fn : callable or None
        The loss function to be used by the adaptive learner.
        Possible values:

        * ``None`` (default): it will use the ``default_loss`` from the
          adaptive module.
        * callable : Refer to [#fn2]_ for more information. Specifically,
          look at ``adaptive.learner.learner1D`` to find more loss functions.

    n : int, optional
        Used when the ``adaptive=False``. The function is uniformly sampled
        at ``n`` number of points. Default value to 1000.
        If the ``adaptive=True``, this parameter will be ignored.

    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot, which doesn't support the
        adaptive algorithm (meaning it will use ``adaptive=False``).
        Learn more by reading the documentation of ``iplot``.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

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

    tx, ty, tp : callable, optional
        Apply a numerical function to the x-direction, y-direction and
        parameter, respectively.

    use_cm : boolean, optional
        If True, apply a color map to the parametric lines.
        If False, solid colors will be used instead. Default to True.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel : str, optional
        Labels for the x-axis or y-axis, respectively.

    xscale, yscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis or y-axis, respectively.
        Default to ``'linear'``.

    xlim, ylim : (float, float), optional
        Denotes the x-axis limits or y-axis limits, respectively,
        ``(min, max)``, visible in the chart.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi
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

    Plotting a numerical function instead of a symbolic expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> fx = lambda t: np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12)**5)
       >>> fy = lambda t: np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12)**5)
       >>> p = plot_parametric(fx, fy, ("t", 0, 12 * pi), title="Butterfly Curve",
       ...     use_cm=False, n=2000)

    Interactive-widget plot. Refer to the interactive sub-module documentation
    to learn more about the ``params`` dictionary. This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 575

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
           aspect="equal", use_latex=False,
           xlim=(-1.25, 1.25), ylim=(-1.25, 1.25)
       )

    References
    ==========

    .. [#fn2] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot, plot_polar, plot3d, plot3d_parametric_line, plot_implicit,
    plot_geometry, plot_piecewise, plot_list

    """
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, Parametric2DLineSeries)
    plot_expr = _check_arguments(args, 2, 1, **kwargs)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Parametric2DLineSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)
    series += _create_generic_data_series(**kwargs)

    if kwargs.get("params", None):
        return create_interactive_plot(*series, **kwargs)

    Backend = kwargs.pop("backend", TWO_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def plot_parametric_region(*args, **kwargs):
    """
    Plots a 2D parametric region.

    NOTE: this is an experimental plotting function as it only draws lines
    without fills. The resulting visualization might change when new features
    will be implemented.

    Typical usage examples are in the followings:

    - Plotting a single parametric curve with a range
        `plot_parametric(expr_x, expr_y, range_u, range_v)`
    - Plotting multiple parametric curves with the same range
        `plot_parametric((expr_x, expr_y), ..., range_u, range_v)`
    - Plotting multiple parametric curves with different ranges
        `plot_parametric((expr_x, expr_y, range_u, range_v), ...)`

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
    kwargs = _set_discretization_points(kwargs, Parametric2DLineSeries)
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
            new_pe.append((fx.subs(u, uv), fy.subs(u, uv),
                (v, vmin, vmax), lbl, rkw if rkw_u is None else rkw_u))
        for vv in np.linspace(float(vmin), float(vmax), n2):
            new_pe.append((fx.subs(v, vv), fy.subs(v, vv),
                (u, umin, umax), rkw if rkw_v is None else rkw_v))

        series += _create_series(Parametric2DLineSeries, new_pe, **kwargs)

    Backend = kwargs.pop("backend", TWO_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def plot3d_parametric_line(*args, **kwargs):
    """
    Plots a 3D parametric line plot.

    Typical usage examples are in the followings:

    - Plotting a single expression.
        `plot3d_parametric_line(expr_x, expr_y, expr_z, range, **kwargs)`
    - Plotting a single expression with a custom label and rendering options.
        `plot3d_parametric_line(expr_x, expr_y, expr_z, range, label, rendering_kw, **kwargs)`
    - Plotting multiple expressions with the same ranges.
        `plot3d_parametric_line((expr_x1, expr_y1, expr_z1), (expr_x2, expr_y2, expr_z2), ..., range, **kwargs)`
    - Plotting multiple expressions with different ranges.
        `plot3d_parametric_line((expr_x1, expr_y1, expr_z1, range1), (expr_x2, expr_y2, expr_z2, range2), ..., **kwargs)`
    - Plotting multiple expressions with custom labels and rendering options.
        `plot3d_parametric_line((expr_x1, expr_y1, expr_z1, range1, label1, rendering_kw1), (expr_x2, expr_y2, expr_z2, range2, label1, rendering_kw2), ..., **kwargs)`


    Parameters
    ==========

    args :
        expr_x : Expr
            The expression representing x component of the parametric
            function. It can be a:

            * Symbolic expression representing the function of one variable
              to be plotted.
            * Numerical function of one variable, supporting vectorization.
              In this case the following keyword arguments are not supported:
              ``params``.

        expr_y : Expr
            The expression representing y component of the parametric
            function. It can be a:

            * Symbolic expression representing the function of one variable
              to be plotted.
            * Numerical function of one variable, supporting vectorization.
              In this case the following keyword arguments are not supported:
              ``params``.

        expr_z : Expr
            The expression representing z component of the parametric
            function. It can be a:

            * Symbolic expression representing the function of one variable
              to be plotted.
            * Numerical function of one variable, supporting vectorization.
              In this case the following keyword arguments are not supported:
              ``params``.

        range : (symbol, min, max)
            A 3-tuple denoting the range of the parameter variable.

        label : str, optional
            An optional string denoting the label of the expression
            to be visualized on the legend. If not provided, the string
            representation of the expression will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines. Refer to the
            plotting library (backend) manual for more informations.

    adaptive : bool, optional
        Setting ``adaptive=True`` activates the adaptive algorithm
        implemented in [#fn3]_ to create smooth plots. Use ``adaptive_goal``
        and ``loss_fn`` to further customize the output.

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
          must return a float number. Refer to [#fn3]_ for more information.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    color_func : callable, optional
        Define the line color mapping when ``use_cm=True``. It can either be:

        * A numerical function supporting vectorization. The arity can be:

          * 1 argument: ``f(t)``, where ``t`` is the parameter.
          * 3 arguments: ``f(x, y, z)`` where ``x, y, z`` are the coordinates
            of the points.
          * 4 arguments: ``f(x, y, z, t)``.

        * A symbolic expression having at most as many free symbols as
          ``expr_x`` or ``expr_y`` or ``expr_z``.
        * None: the default value (color mapping applied to the parameter).

    force_real_eval : boolean, optional
        Default to False, with which the numerical evaluation is attempted
        over a complex domain, which is slower but produces correct results.
        Set this to True if performance is of paramount importance, but be
        aware that it might produce wrong results. It only works with
        ``adaptive=False``.

    is_point : boolean, optional
        Default to False, which will render a line connecting all the points.
        If True, a scatter plot will be generated.

    label : str or list/tuple, optional
        The label to be shown in the legend or in the colorbar. If not
        provided, the string representation of ``expr`` will be used.
        The number of labels must be equal to the number of expressions.

    loss_fn : callable or None
        The loss function to be used by the adaptive learner.
        Possible values:

        * ``None`` (default): it will use the ``default_loss`` from the
          ``adaptive`` module.
        * callable : Refer to [#fn3]_ for more information. Specifically,
          look at ``adaptive.learner.learner1D`` to find more loss functions.

    n : int, optional
        Used when the ``adaptive=False``. The function is uniformly
        sampled at ``n`` number of points. Default value to 1000.
        If the ``adaptive=True``, this parameter will be ignored.

    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot, which doesn't support the
        adaptive algorithm (meaning it will use ``adaptive=False``).
        Learn more by reading the documentation of ``iplot``.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

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

    tx, ty, tz, tp : callable, optional
        Apply a numerical function to the x, y, z directions and to the
        discretized parameter.

    use_cm : boolean, optional
        If True, apply a color map to the parametric lines.
        If False, solid colors will be used instead. Default to True.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel, zlabel : str, optional
        Labels for the x-axis, y-axis, z-axis, respectively.

    xlim, ylim, zlim : (float, float), optional
        Denotes the axis limits, `(min, max)`, visible in the chart.


    Examples
    ========

    Note: for documentation purposes, the following examples uses Matplotlib.
    However, Matplotlib's 3D capabilities are rather limited. Consider running
    these examples with a different backend (hence, modify ``rendering_kw``
    to pass the correct options to the backend).

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
           rendering_kw={"color":0x353535}, imodule="panel")
       line = plot3d_parametric_line(
           a * cos(t) + b * cos(3 * t),
           a * sin(t) - b * sin(3 * t),
           c * sin(2 * t), prange(t, s*pi, e*pi),
           {"color_map": k3d.matplotlib_color_maps.Summer}, params=params,
           backend=KB, show=False, imodule="panel")
       (line + sphere).show()

    References
    ==========

    .. [#fn3] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot, plot3d, plot_contour, plot3d_parametric_surface,
    plot3d_spherical, plot3d_revolution, plot3d_implicit

    """
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, Parametric3DLineSeries)
    plot_expr = _check_arguments(args, 3, 1, **kwargs)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Parametric3DLineSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    if kwargs.get("params", None):
        return create_interactive_plot(*series, **kwargs)

    Backend = kwargs.pop("backend", THREE_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def _plot3d_plot_contour_helper(Series, is_threed, Backend, *args, **kwargs):
    """plot3d and plot_contour are structurally identical. Let's reduce
    code repetition.
    """
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, SurfaceOver2DRangeSeries)
    plot_expr = _check_arguments(args, 1, 2, **kwargs)

    if is_threed:
        if any(isinstance(p[0], Plane) for p in plot_expr):
            raise ValueError("Please, use ``plot_geometry`` to visualize "
                "a plane.")

    free_x = set()
    free_y = set()
    for p in plot_expr:
        free_x |= {p[1][0]} if isinstance(p[1][0], (Symbol, BaseScalar)) else {Symbol(p[1][0])}
        free_y |= {p[2][0]} if isinstance(p[2][0], (Symbol, BaseScalar)) else {Symbol(p[2][0])}
    x = free_x.pop() if free_x else Symbol("x")
    y = free_y.pop() if free_y else Symbol("y")
    fx = lambda use_latex: x.name if not use_latex else latex(x)
    fy = lambda use_latex: y.name if not use_latex else latex(y)
    wrap = lambda use_latex: "f(%s, %s)" if not use_latex else r"f\left(%s, %s\right)"
    fz = lambda use_latex: wrap(use_latex) % (fx(use_latex), fy(use_latex))
    kwargs.setdefault("xlabel", fx)
    kwargs.setdefault("ylabel", fy)
    kwargs.setdefault("zlabel", fz)

    # if a polar discretization is requested and automatic labelling has ben
    # applied, hide the labels on the x-y axis.
    if kwargs.get("is_polar", False):
        if callable(kwargs["xlabel"]):
            kwargs["xlabel"] = ""
        if callable(kwargs["ylabel"]):
            kwargs["ylabel"] = ""

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Series, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)
    if is_threed:
        series += _plot3d_wireframe_helper(series, **kwargs)

    if kwargs.get("params", None):
        kwargs["threed"] = is_threed
        kwargs["backend"] = Backend
        return create_interactive_plot(*series, **kwargs)

    return _instantiate_backend(Backend, *series, **kwargs)


def _plot3d_wireframe_helper(surfaces, **kwargs):
    """Create data series representing wireframe lines.

    Parameters
    ==========

    surfaces : list of BaseSeries

    Returns
    =======

    line_series : list of Parametric3DLineSeries
    """
    if not kwargs.get("wireframe", False):
        return []

    np = import_module('numpy')
    lines = []
    wf_n1 = kwargs.get("wf_n1", 10)
    wf_n2 = kwargs.get("wf_n2", 10)
    npoints = kwargs.get("wf_npoints", None)
    wf_rend_kw = kwargs.get("wf_rendering_kw", dict())

    wf_kwargs = dict(
        use_cm=False, show_in_legend=False,
        # use uniform meshing to maximize performance
        adaptive=False, n=npoints,
        rendering_kw=wf_rend_kw
    )

    def create_series(expr, ranges, surface_series, **kw):
        expr = [e if callable(e) else sympify(e) for e in expr]
        kw["tx"] = surface_series._tx
        kw["ty"] = surface_series._ty
        kw["tz"] = surface_series._tz
        kw["tp"] = surface_series._tp
        kw["force_real_eval"] = surface_series._force_real_eval
        if "return" not in kw.keys():
            return Parametric3DLineSeries(*expr, *ranges, "", **kw)
        return ComplexParametric3DLineSeries(*expr, *ranges, "", **kw)

    # NOTE: can't use np.linspace because start, end might be
    # symbolic expressions
    def linspace(start, end , n):
        return [start + (end - start) * i / (n - 1) for i in range(n)]

    for s in surfaces:
        param_expr, ranges = [], []

        if s.is_3Dsurface:
            expr = s.expr

            kw = wf_kwargs.copy()
            if s.is_interactive:
                kw["params"] = s.params.copy()

            if s.is_parametric:
                (x, sx, ex), (y, sy, ey) = s.ranges
                is_callable = any(callable(e) for e in expr)

                for uval in linspace(sx, ex, wf_n1):
                    kw["n"] = s.n[1] if npoints is None else npoints
                    if is_callable:
                        # NOTE: closure on lambda functions
                        param_expr = [lambda t, uv=uval, e=e: e(float(uv), t) for e in expr]
                        ranges = [(y, sy, ey)]
                    else:
                        param_expr = [e.subs(x, uval) for e in expr]
                        ranges = [(y, sy, ey)]
                    lines.append(create_series(param_expr, ranges, s, **kw))
                for vval in linspace(sy, ey, wf_n2):
                    kw["n"] = s.n[0] if npoints is None else npoints
                    if is_callable:
                        # NOTE: closure on lambda functions
                        param_expr = [lambda t, vv=vval, e=e: e(t, float(vv)) for e in expr]
                        ranges = [(x, sx, ex)]
                    else:
                        param_expr = [e.subs(y, vval) for e in expr]
                        ranges = [(x, sx, ex)]
                    lines.append(create_series(param_expr, ranges, s, **kw))

            else:
                if not s.is_complex:
                    (x, sx, ex), (y, sy, ey) = s.ranges
                else:
                    x, y = symbols("x, y", cls=Dummy)
                    z, start, end = s.ranges[0]
                    expr = s.expr.subs(z, x + I * y)
                    sx, ex = re(start), re(end)
                    sy, ey = im(start), im(end)
                    kw["return"] = s._return

                if not s.is_polar:
                    for xval in linspace(sx, ex, wf_n1):
                        kw["n"] = s.n[1] if npoints is None else npoints
                        if callable(expr):
                            # NOTE: closure on lambda functions
                            param_expr = [
                                lambda t, xv=xval: xv,
                                lambda t: t,
                                lambda t, xv=xval: expr(float(xv), t)]
                            ranges = [(y, sy, ey)]
                        else:
                            param_expr = [xval, y, expr.subs(x, xval)]
                            ranges = [(y, sy, ey)]
                        lines.append(create_series(param_expr, ranges, s, **kw))
                    for yval in linspace(sy, ey, wf_n2):
                        kw["n"] = s.n[0] if npoints is None else npoints
                        if callable(expr):
                            # NOTE: closure on lambda functions
                            param_expr = [
                                lambda t: t,
                                lambda t, yv=yval: yv,
                                lambda t, yv=yval: expr(t, float(yv))]
                            ranges = [(x, sx, ex)]
                        else:
                            param_expr = [x, yval, expr.subs(y, yval)]
                            ranges = [(x, sx, ex)]
                        lines.append(create_series(param_expr, ranges, s, **kw))
                else:
                    for rval in linspace(sx, ex, wf_n1):
                        kw["n"] = s.n[1] if npoints is None else npoints
                        if callable(expr):
                            param_expr = [
                                lambda t, rv=rval: float(rv) * np.cos(t),
                                lambda t, rv=rval: float(rv) * np.sin(t),
                                lambda t, rv=rval: expr(float(rv), t)]
                            ranges = [(y, sy, ey)]
                        else:
                            param_expr = [rval * cos(y), rval * sin(y), expr.subs(x, rval)]
                            ranges = [(y, sy, ey)]
                        lines.append(create_series(param_expr, ranges, s, **kw))
                    for tval in linspace(sy, ey, wf_n2):
                        kw["n"] = s.n[0] if npoints is None else npoints
                        if callable(expr):
                            param_expr = [
                                lambda p, tv=tval: p * np.cos(float(tv)),
                                lambda p, tv=tval: p * np.sin(float(tv)),
                                lambda p, tv=tval: expr(p, float(tv))]
                            ranges = [(x, sx, ex)]
                        else:
                            param_expr = [x * cos(tval), x * sin(tval), expr.subs(y, tval)]
                            ranges = [(x, sx, ex)]
                        lines.append(create_series(param_expr, ranges, s, **kw))

    return lines


def plot3d(*args, **kwargs):
    """
    Plots a 3D surface plot.

    Typical usage examples are in the followings:

    - Plotting a single expression.
        `plot3d(expr, range_x, range_y, **kwargs)`
    - Plotting multiple expressions with the same ranges.
        `plot3d(expr1, expr2, range_x, range_y, **kwargs)`
    - Plotting multiple expressions with different ranges.
        `plot3d((expr1, range_x1, range_y1), (expr2, range_x2, range_y2), ..., **kwargs)`
    - Plotting multiple expressions with custom labels and rendering options.
        `plot3d((expr1, range_x1, range_y1, label1, rendering_kw1), (expr2, range_x2, range_y2, label2, rendering_kw2), ..., **kwargs)`

    Note: it is important to specify at least the ``range_x``, otherwise the
    function might create a rotated plot.

    Parameters
    ==========

    args :
        expr : Expr
            Expression representing the function of two variables to be plotted.
            The expression representing the function of two variables to be
            plotted. It can be a:

            * Symbolic expression.
            * Numerical function of two variable, supporting vectorization.
              In this case the following keyword arguments are not supported:
              ``params``.

        range_x: (symbol, min, max)
            A 3-tuple denoting the range of the x variable. Default values:
            `min=-10` and `max=10`.

        range_y: (symbol, min, max)
            A 3-tuple denoting the range of the y variable. Default values:
            `min=-10` and `max=10`.

        label : str, optional
            The label to be shown in the colorbar.  If not provided, the string
            representation of ``expr`` will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of surfaces. Refer to the
            plotting library (backend) manual for more informations.

    adaptive : bool, optional
        The default value is set to ``False``, which uses a uniform sampling
        strategy with number of discretization points ``n1`` and ``n2`` along
        the x and y directions, respectively.

        Set adaptive to ``True`` to use the adaptive algorithm implemented in
        [#fn4]_ to create smooth plots. Use ``adaptive_goal`` and ``loss_fn``
        to further customize the output.

    adaptive_goal : callable, int, float or None
        Controls the "smoothness" of the evaluation. Possible values:

        * ``None`` (default):  it will use the following goal:
          ``lambda l: l.loss() < 0.01``
        * number (int or float). The lower the number, the more
          evaluation points. This number will be used in the following goal:
          ``lambda l: l.loss() < number``
        * callable: a function requiring one input element, the learner. It
          must return a float number. Refer to [#fn4]_ for more information.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    color_func : callable, optional
        Define the surface color mapping when ``use_cm=True``.
        It can either be:

        * A numerical function of 3 variables, x, y, z (the points computed
          by the internal algorithm) supporting vectorization.
        * A symbolic expression having at most as many free symbols as
          ``expr``.
        * None: the default value (color mapping applied to the z-value of
          the surface).

    force_real_eval : boolean, optional
        Default to False, with which the numerical evaluation is attempted
        over a complex domain, which is slower but produces correct results.
        Set this to True if performance is of paramount importance, but be
        aware that it might produce wrong results. It only works with
        ``adaptive=False``.

    is_polar : boolean, optional
        Default to False. If True, requests a polar discretization. In this
        case, ``range_x`` represents the radius, ``range_y`` represents the
        angle.

    label : str or list/tuple, optional
        The label to be shown in the colorbar. If not provided, the string
        representation of ``expr`` will be used. The number of labels must be
        equal to the number of expressions.

    loss_fn : callable or None
        The loss function to be used by the adaptive learner.
        Possible values:

        * ``None`` (default): it will use the ``default_loss`` from the
          ``adaptive`` module.
        * callable : Refer to [#fn4]_ for more information. Specifically,
          look at ``adaptive.learner.learnerND`` to find more loss functions.

    n1, n2 : int, optional
        ``n1`` and ``n2`` set the number of discretization points along the
        x and y ranges, respectively. Default value to 100.

    n : int or two-elements tuple (n1, n2), optional
        If an integer is provided, the x and y ranges are sampled uniformly
        at ``n`` of points. If a tuple is provided, it overrides
        ``n1`` and ``n2``.

    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot, which doesn't support the
        adaptive algorithm (meaning it will use ``adaptive=False``).
        Learn more by reading the documentation of ``iplot``.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of surfaces. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

    show : bool, optional
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the backend.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    tx, ty, tz : callable, optional
        Apply a numerical function to the discretized domain in the
        x, y and z direction, respectively.

    use_cm : boolean, optional
        If True, apply a color map to the surface.
        If False, solid colors will be used instead. Default to False.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    wireframe : boolean, optional
        Enable or disable a wireframe over the surface. Depending on the number
        of wireframe lines (see ``wf_n1`` and ``wf_n2``), activating this
        option might add a considerable overhead during the plot's creation.
        Default to False (disabled).

    wf_n1, wf_n2 : int, optional
        Number of wireframe lines along the x and y ranges, respectively.
        Default to 10. Note that increasing this number might considerably
        slow down the plot's creation.

    wf_npoint : int or None, optional
        Number of discretization points for the wireframe lines. Default to
        None, meaning that each wireframe line will have ``n1`` or ``n2``
        number of points, depending on the line direction.

    wf_rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of wireframe lines.

    xlabel, ylabel, zlabel : str, optional
        Labels for the x-axis or y-axis or z-axis, respectively.

    xlim, ylim : (float, float), optional
        Denotes the x-axis limits or y-axis limits, ``(min, max)``, visible in
        the chart. Note that the function is still being evaluate over
        ``range_x`` and ``range_y``.

    zlim : (float, float), optional
        Denotes the z-axis limits, ``(min, max)``, visible in the chart.

    Examples
    ========

    Note: for documentation purposes, the following examples uses Matplotlib.
    However, Matplotlib's 3D capabilities are rather limited. Consider running
    these examples with a different backend (hence, modify the ``rendering_kw``
    and ``wf_rendering_kw`` to pass the correct options to the backend).

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

    Multiple plots with same range. Set ``use_cm=True`` to distinguish the
    expressions:

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

    References
    ==========

    .. [#fn4] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot, plot_contour, plot3d_parametric_line, plot3d_parametric_surface,
    plot3d_implicit, plot3d_revolution, plot3d_spherical

    """
    Backend = kwargs.pop("backend", THREE_D_B)
    return _plot3d_plot_contour_helper(
        SurfaceOver2DRangeSeries, True, Backend, *args, **kwargs)


def plot3d_parametric_surface(*args, **kwargs):
    """
    Plots a 3D parametric surface plot.

    Typical usage examples are in the followings:

    - Plotting a single expression.
        `plot3d_parametric_surface(expr_x, expr_y, expr_z, range_u, range_v, label, **kwargs)`
    - Plotting multiple expressions with the same ranges.
        `plot3d_parametric_surface((expr_x1, expr_y1, expr_z1), (expr_x2, expr_y2, expr_z2), range_u, range_v, **kwargs)`
    - Plotting multiple expressions with different ranges.
        `plot3d_parametric_surface((expr_x1, expr_y1, expr_z1, range_u1, range_v1), (expr_x2, expr_y2, expr_z2, range_u2, range_v2), **kwargs)`
    - Plotting multiple expressions with different ranges and rendering option.
        `plot3d_parametric_surface((expr_x1, expr_y1, expr_z1, range_u1, range_v1, label1, rendering_kw1), (expr_x2, expr_y2, expr_z2, range_u2, range_v2, label2, rendering_kw2), **kwargs)`

    Note: it is important to specify both the ranges.

    Parameters
    ==========

    args :
        expr_x: Expr
            Expression representing the function along `x`. It can be a:

            * Symbolic expression.
            * Numerical function of two variable, f(u, v), supporting
              vectorization. In this case the following keyword arguments are
              not supported: ``params``.

        expr_y: Expr
            Expression representing the function along `y`. It can be a:

            * Symbolic expression.
            * Numerical function of two variable, f(u, v), supporting
              vectorization. In this case the following keyword arguments are
              not supported: ``params``.

        expr_z: Expr
            Expression representing the function along `z`. It can be a:

            * Symbolic expression.
            * Numerical function of two variable, f(u, v), supporting
              vectorization. In this case the following keyword arguments are
              not supported: ``params``.

        range_u: (symbol, min, max)
            A 3-tuple denoting the range of the `u` variable.

        range_v: (symbol, min, max)
            A 3-tuple denoting the range of the `v` variable.

        label : str, optional
            The label to be shown in the colorbar.  If not provided, the string
            representation of the expression will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of surfaces. Refer to the
            plotting library (backend) manual for more informations.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    color_func : callable, optional
        Define the surface color mapping when ``use_cm=True``.
        It can either be:

        * A numerical function supporting vectorization. The arity can be:

          * 1 argument: ``f(u)``, where ``u`` is the first parameter.
          * 2 arguments: ``f(u, v)`` where ``u, v`` are the parameters.
          * 3 arguments: ``f(x, y, z)`` where ``x, y, z`` are the coordinates of
            the points.
          * 5 arguments: ``f(x, y, z, u, v)``.

        * A symbolic expression having at most as many free symbols as
          ``expr_x`` or ``expr_y`` or ``expr_z``.
        * None: the default value (color mapping applied to the z-value of
          the surface).

    force_real_eval : boolean, optional
        Default to False, with which the numerical evaluation is attempted
        over a complex domain, which is slower but produces correct results.
        Set this to True if performance is of paramount importance, but be
        aware that it might produce wrong results. It only works with
        ``adaptive=False``.

    label : str or list/tuple, optional
        The label to be shown in the colorbar. If not provided, the string
        representation will be used. The number of labels must be
        equal to the number of expressions.

    n1, n2 : int, optional
        ``n1`` and ``n2`` set the number of discretization points along the
        u and v ranges, respectively. Default value to 100.

    n : int or two-elements tuple (n1, n2), optional
        If an integer is provided, the u and v ranges are sampled uniformly
        at ``n`` of points. If a tuple is provided, it overrides
        ``n1`` and ``n2``.

    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot. Learn more by reading the
        documentation of ``iplot``.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of surfaces. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

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

    tx, ty, tz : callable, optional
        Apply a numerical function to the discretized domain in the
        x, y and z direction, respectively.

    use_cm : boolean, optional
        If True, apply a color map to the surface.
        If False, solid colors will be used instead. Default to False.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    wireframe : boolean, optional
        Enable or disable a wireframe over the surface. Depending on the number
        of wireframe lines (see ``wf_n1`` and ``wf_n2``), activating this
        option might add a considerable overhead during the plot's creation.
        Default to False (disabled).

    wf_n1, wf_n2 : int, optional
        Number of wireframe lines along the u and v ranges, respectively.
        Default to 10. Note that increasing this number might considerably
        slow down the plot's creation.

    wf_npoint : int or None, optional
        Number of discretization points for the wireframe lines. Default to
        None, meaning that each wireframe line will have ``n1`` or ``n2``
        number of points, depending on the line direction.

    wf_rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of wireframe lines.

    xlabel, ylabel, zlabel : str, optional
        Label for the x-axis or y-axis or z-axis, respectively.

    xlim, ylim, zlim : (float, float), optional
        Denotes the x-axis limits, or y-axis limits, or z-axis limits,
        respectively, ``(min, max)``, visible in the chart.


    Examples
    ========

    Note: for documentation purposes, the following examples uses Matplotlib.
    However, Matplotlib's 3D capabilities are rather limited. Consider running
    these examples with a different backend (hence, modify the ``rendering_kw``
    and ``wf_rendering_kw`` to pass the correct options to the backend).

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
           title="Möbius \, strip",
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
           backend=PB, wireframe=True, wf_n2=20, zlabel="f(z)")

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
           title="Catenoid \, to \, Right \, Helicoid \, Transformation")

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

    plot, plot3d, plot_contour, plot3d_parametric_line, plot3d_spherical,
    plot3d_revolution, plot3d_implicit

    """
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, ParametricSurfaceSeries)
    plot_expr = _check_arguments(args, 3, 2, **kwargs)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(ParametricSurfaceSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)
    series += _plot3d_wireframe_helper(series, **kwargs)
    Backend = kwargs.get("backend", THREE_D_B)

    if kwargs.get("params", None):
        return create_interactive_plot(*series, **kwargs)

    return _instantiate_backend(Backend, *series, **kwargs)


def plot3d_spherical(*args, **kwargs):
    """
    Plots a radius as a function of the spherical coordinates theta and phi.

    Typical usage examples are in the followings:

    - Plotting a single expression.
        `plot3d_spherical(r, range_theta, range_phi, **kwargs)`
    - Plotting multiple expressions with the same ranges.
        `plot3d_parametric_surface(r1, r2, range_theta, range_phi, **kwargs)`
    - Plotting multiple expressions with different ranges.
        `plot3d_parametric_surface((r1, range_theta1, range_phi1), (r2, range_theta2, range_phi2), **kwargs)`
    - Plotting multiple expressions with different ranges and rendering option.
        `plot3d_parametric_surface((r1, range_theta1, range_phi1, label1, rendering_kw1), (r2, range_theta2, range_phi2, label2, rendering_kw2), **kwargs)`

    Note: it is important to specify both the ranges.

    Parameters
    ==========

    args :
        r: Expr
            Expression representing the radius. It can be a:

            * Symbolic expression.
            * Numerical function of two variable, f(theta, phi), supporting
              vectorization. In this case the following keyword arguments are
              not supported: ``params``.

        theta: (symbol, min, max)
            A 3-tuple denoting the range of the polar angle, which is limited
            in [0, pi]. Consider a sphere:

            * ``theta=0`` indicates the north pole.
            * ``theta=pi/2`` indicates the equator.
            * ``theta=pi`` indicates the south pole.

        range_v: (symbol, min, max)
            A 3-tuple denoting the range of the azimuthal angle, which is
            limited in [0, 2*pi].

        label : str, optional
            The label to be shown in the colorbar. If not provided, the string
            representation of the expression will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of surfaces. Refer to the
            plotting library (backend) manual for more informations.

    Keyword arguments are the same as ``plot3d_parametric_surface``. Refer to
    its documentation for more information.

    Examples
    ========

    Note: for documentation purposes, the following examples uses Matplotlib.
    However, Matplotlib's 3D capabilities are rather limited. Consider running
    these examples with a different backend (hence, modify the ``rendering_kw``
    and ``wf_rendering_kw`` to pass the correct options to the backend).


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
           wireframe=True, wf_n2=25, backend=PB)

    Interactive-widget plot of spherical harmonics. Note that the plot's
    creation and update might be slow and that it must be ``m < n`` at all
    times. Refer to the interactive sub-module documentation to learn more
    about the ``params`` dictionary.

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

    plot3d, plot3d_parametric_surface, plot3d_parametric_line, plot3d_implicit,
    plot3d_revolution

    """
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, ParametricSurfaceSeries)
    plot_expr = _check_arguments(args, 1, 2, **kwargs)

    # deal with symbolic min/max values of ranges
    def rel(t, s, threshold, a):
        try:
            if t == "<":
                if s < threshold:
                    return threshold
            elif t == ">":
                if s > threshold:
                    return threshold
        except:
            return a
        return a

    # enforce polar and azimuthal condition and convert spherical to cartesian
    for i, pe in enumerate(plot_expr):
        r, r1, r2 = pe[0], pe[1], pe[2]
        theta, phi = r1[0], r2[0]
        x, y, z = spherical_to_cartesian(r, theta, phi)
        r1 = prange(theta, rel("<", r1[1], 0, r1[1]), rel(">", r1[2], pi, r1[2]))
        r2 = prange(phi, rel("<", r2[1], 0, r2[1]), rel(">", r2[2], 2*pi, r2[2]))
        plot_expr[i] = (x, y, z, r1, r2, *pe[3:])
    return plot3d_parametric_surface(*plot_expr, **kwargs)


def plot3d_implicit(*args, **kwargs):
    """
    Plots an isosurface of a function.

    Typical usage examples are in the followings:

    - `plot3d_parametric_surface(expr, range_x, range_y, range_z, rendering_kw [optional], **kwargs)`

    Note that:

    1. it is important to specify the ranges, as they will determine the
       orientation of the surface.
    2. the number of discretization points is crucial as the algorithm will
       discretize a volume. A high number of discretization points creates a
       smoother mesh, at the cost of a much higher memory consumption and
       slower computation.
    3. To plot ``f(x, y, z) = c`` either write ``expr = f(x, y, z) - c`` or
       pass the appropriate keyword to ``rendering_kw``. Read the backends
       documentation to find out the available options.
    4. Only ``PlotlyBackend`` and ``K3DBackend`` support 3D implicit plotting.


    Parameters
    ==========

    args :
        expr: Expr
            Implicit expression.  It can be a:

            * Symbolic expression.
            * Numerical function of three variable, f(x, y, z), supporting
              vectorization.

        range_x: (symbol, min, max)
            A 3-tuple denoting the range of the `x` variable.

        range_y: (symbol, min, max)
            A 3-tuple denoting the range of the `y` variable.

        range_z: (symbol, min, max)
            A 3-tuple denoting the range of the `z` variable.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of surfaces. Refer to the
            plotting library (backend) manual for more informations.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.

    n1, n2, n3 : int, optional
        Set the number of discretization points along the x, y and z ranges,
        respectively. Default value is 60.

    n : int or three-elements tuple (n1, n2, n3), optional
        If an integer is provided, the x, y and z ranges are sampled uniformly
        at ``n`` of points. If a tuple is provided, it overrides ``n1``,
        ``n2`` and ``n3``.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of surfaces. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

    show : bool, optional
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel, zlabel : str, optional
        Labels for the x-axis, y-axis or z-axis, respectively.

    xlim, ylim, zlim : (float, float), optional
        Denotes the x-axis limits, y-axis limits or z-axis limits,
        respectively, ``(min, max)``, visible in the chart. Note that the
        function is still being evaluated over the ``range_x``, ``range_y``
        and ``range_z``.


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

    plot, plot3d, plot_contour, plot3d_parametric_line,
    plot3d_parametric_surface, plot3d_revolution, plot3d_spherical

    """
    if kwargs.pop("params", None) is not None:
        raise NotImplementedError(
            "plot3d_implicit doesn't support interactive widgets.")

    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, Implicit3DSeries)
    plot_expr = _check_arguments(args, 1, 3, **kwargs)

    labels = kwargs.pop("label", dict())
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Implicit3DSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    fx = lambda use_latex: series[0].var_x.name if not use_latex else latex(series[0].var_x)
    fy = lambda use_latex: series[0].var_y.name if not use_latex else latex(series[0].var_y)
    fz = lambda use_latex: series[0].var_z.name if not use_latex else latex(series[0].var_z)
    kwargs.setdefault("xlabel", fx)
    kwargs.setdefault("ylabel", fy)
    kwargs.setdefault("zlabel", fz)

    Backend = kwargs.get("backend", THREE_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def plot_contour(*args, **kwargs):
    """
    Draws contour plot of a function of two variables.

    This function signature is almost identical to `plot3d`: refer to its
    documentation for a full list of available argument and keyword arguments.

    Parameters
    ==========

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

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
       :small-size: 800, 575

       from sympy import *
       from spb import *
       x, y, a, b, xp, yp = symbols("x y a b x_p y_p")
       expr = (cos(x) + a * sin(x) * sin(y) - b * sin(x) * cos(y))**2
       plot_contour(expr, prange(x, 0, xp*pi), prange(y, 0, yp * pi),
           params={a: (1, 0, 2), b: (1, 0, 2), xp: (1, 0, 2), yp: (2, 0, 2)},
           grid=False, use_latex=False)

    See Also
    ========

    plot, plot_polar, plot_parametric, plot3d, plot_implicit, plot_geometry,
    plot_piecewise, plot_list

    """
    Backend = kwargs.pop("backend", TWO_D_B)
    return _plot3d_plot_contour_helper(
        ContourSeries, False, Backend, *args, **kwargs)


def plot3d_revolution(curve, range_t, range_phi=None, axis=(0, 0),
    parallel_axis="z", show_curve=False, curve_kw=None, **kwargs):
    """Generate a surface of revolution by rotating a curve around an axis of
    rotation.

    Parameters
    ==========

    curve : Expr, list/tuple of 2 or 3 elements
        The curve to be revolved, which can be either:

        * a symbolic expression
        * a 2-tuple representing a parametric curve in 2D space
        * a 3-tuple representing a parametric curve in 3D space

    range_t : (symbol, min, max)
        A 3-tuple denoting the range of the parameter of the curve.

    range_phi : (symbol, min, max)
        A 3-tuple denoting the range of the azimuthal angle where the curve
        will be revolved. Default to ``(phi, 0, 2*pi)``.

    axis : (coord1, coord2)
        A 2-tuple that specifies the position of the rotation axis.
        Depending on the value of ``parallel_axis``:

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

    **kwargs :
        Keyword arguments are the same as ``plot3d_parametric_surface``.
        Refer to its documentation for more information.

    Examples
    ========

    Note: for documentation purposes, the following examples uses Matplotlib.
    However, Matplotlib's 3D capabilities are rather limited. Consider running
    these examples with a different backend (hence, modify the ``curve_kw``,
    ``rendering_kw`` and ``wf_rendering_kw`` to pass the correct options to
    the backend).

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

    plot3d, plot3d_parametric_surface, plot3d_parametric_line,
    plot3d_spherical, plot_contour

    """
    show = kwargs.pop("show", True)
    kwargs["show"] = False
    imodule = kwargs.get("imodule", cfg["interactive"]["module"])

    if curve_kw is None:
        curve_kw = {}

    if parallel_axis.lower() not in ["x", "y", "z"]:
        raise ValueError("`parallel_axis` must be either 'x' 'y' or 'z'. "
            "Received: %s " % parallel_axis)

    # NOTE: a surface of revolution is a particular case of 3D parametric
    # surface
    if isinstance(curve, (tuple, list, Tuple)):
        if len(curve) == 2:     # curve is a 2D parametric line
            x, z = curve
            y = 0
        elif len(curve) == 3:   # curve is a 3D parametric line
            x, y, z = curve
    else: # curve is an expression
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

    surface = plot3d_parametric_surface(*v, range_t, range_phi, **kwargs)

    params = kwargs.get("params", None)
    if show_curve:
        if params is None:
            backend = type(surface)
            n = surface[0].n[0]
            force_real_eval = surface[0]._force_real_eval
        else:
            n = surface.backend[0].n[0]
            backend = type(surface.backend)
            curve_kw["params"] = params
            curve_kw.setdefault("imodule", imodule)
            force_real_eval = surface.backend[0]._force_real_eval

        curve_kw["show"] = False
        # uniform mesh evaluation is faster
        curve_kw["adaptive"] = False
        # link the number of discretization points between the two series
        curve_kw["n"] = n
        curve_kw.setdefault("use_cm", False)
        curve_kw.setdefault("force_real_eval", force_real_eval)

        line = plot3d_parametric_line(
            x, y, z, range_t, backend=backend, **curve_kw)
        result = surface + line
    else:
        result = surface

    if show:
        if params is None:
            result.show()
        else:
            return result.show()

    return result


def plot_implicit(*args, **kwargs):
    """Plot implicit equations / inequalities.

    plot_implicit, by default, generates a contour using a mesh grid of fixed
    number of points. The greater the number of points, the greater the memory
    used. By setting ``adaptive=True``, interval arithmetic will be used to
    plot functions. If the expression cannot be plotted using interval
    arithmetic, it defaults to generating a contour using a mesh grid.
    With interval arithmetic, the line width can become very small; in those
    cases, it is better to use the mesh grid approach.

    Parameters
    ==========

    args :
        expr : Expr, Relational, BooleanFunction
            The equation / inequality that is to be plotted.

        ranges : tuples or Symbol
            Two tuple denoting the discretization domain, for example:
            ``(x, -10, 10), (y, -10, 10)``
            To get a correct plot, at least the horizontal range must be
            provided. If no range is given, then the free symbols in the
            expression will be assigned in the order they are sorted, which
            could 'invert' the axis.

            Alternatively, a single Symbol corresponding to the horizontal
            axis must be provided, which will be internally converted to a
            range ``(sym, -10, 10)``.

        label : str, optional
            The label to be shown when multiple expressions are plotted.
            If not provided, the string representation of the expression
            will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of contours. Refer to the
            plotting library (backend) manual for more informations.

    adaptive : bool, optional
        The default value is set to ``False``, meaning that the internal
        algorithm uses a mesh grid approach. In such case, Boolean
        combinations of expressions cannot be plotted.
        If set to ``True``, the internal algorithm uses interval arithmetic.
        It switches to the meshgrid approach if the expression cannot be
        plotted using interval arithmetic.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    backend : Plot, optional
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    depth : integer
        The depth of recursion for adaptive mesh grid. Default value is 0.
        Takes value in the range (0, 4).
        Think of the resulting plot as a picture composed by pixels. By
        increasing ``depth`` we are increasing the number of pixels, thus
        obtaining a more accurate plot.

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of ``expr`` will be used. The number of labels must be
        equal to the number of expressions.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance. Refer to the plotting library
        (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

    n1, n2 : int
        Number of discretization points in the horizontal and vertical
        directions when ``adaptive=False``. Default to 1000.

    n : int or two-elements tuple (n1, n2), optional
        If an integer is provided, the x and y ranges are sampled uniformly
        at ``n`` of points. If a tuple is provided, it overrides
        ``n1`` and ``n2``.

    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot. Learn more by reading the
        documentation of ``iplot``.

    show : Boolean
        Default value is True. If set to False, the plot will not be shown.
        See `Plot` for further information.

    title : string
        The title for the plot.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel : string
        The labels for the x-axis or y-axis, respectively.

    Examples
    ========

    Plot expressions:

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import symbols, Eq, And, sin, pi, log
        >>> from spb import plot_implicit
        >>> x, y = symbols('x y')

    Providing only the symbol for the horizontal axis:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> p = plot_implicit(x - 1, x)

    Specify both ranges and set the number of discretization points:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_implicit(
       ...     (x**2 + y**2 - 1)**3 - x**2 * y**3,
       ...     (x, -1.5, 1.5), (y, -1.5, 1.5),
       ...     n = 500)
       Plot object containing:
       [0]: Implicit expression: Eq(-x**2*y**3 + (x**2 + y**2 - 1)**3, 0) for x over (-1.5, 1.5) and y over (-1.5, 1.5)

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

    Comparison of similar expressions plotted with different algorithms. Note:

    1. Adaptive algorithm (``adaptive=True``) can be used with any expression,
       but it usually creates lines with variable thickness. The ``depth``
       keyword argument can be used to improve the accuracy, but reduces line
       thickness even further.
    2. Mesh grid algorithm (``adaptive=False``) creates lines with contant
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

    Plotting regions:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_implicit(y > x**2, (x, -5, 5), grid=False)
       Plot object containing:
       [0]: Implicit expression: y > x**2 for x over (-5.0, 5.0) and y over (-10.0, 10.0)

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


    Interactive-widget implicit plot. Refer to the interactive sub-module
    documentation to learn more about the ``params`` dictionary.
    This plot illustrates:

    * the use of ``prange`` (parametric plotting range).
    * the use of the ``params`` dictionary to specify sliders in
      their basic form: (default, min, max).

    .. panel-screenshot::
       :small-size: 800, 500

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
           }, n=400)

    See Also
    ========

    plot, plot_polar, plot3d, plot_contour, plot3d_parametric_line,
    plot3d_parametric_surface, plot_geometry, plot3d_implicit, plot_list

    """
    # if the user is plotting a single expression, then he can pass in one
    # or two symbols to sort the axis. Ranges will then be automatically
    # created.
    args = list(args)
    if (len(args) == 2) and isinstance(args[1], Symbol):
        args[1] = Tuple(args[1], -10, 10)
    elif (len(args) >= 3) and isinstance(args[1], Symbol) and isinstance(args[2], Symbol):
        args[1] = Tuple(args[1], -10, 10)
        args[2] = Tuple(args[2], -10, 10)

    args = _plot_sympify(args)
    args = _check_arguments(args, 1, 2, **kwargs)
    kwargs = _set_discretization_points(kwargs, ImplicitSeries)

    series_kw = dict(
        n1=kwargs.pop("n1", 1000),
        n2=kwargs.pop("n2", 1000),
        depth=kwargs.pop("depth", 0),
        adaptive=kwargs.pop("adaptive", False),
        contour_kw=kwargs.pop("contour_kw", dict()),
        params=kwargs.get("params", dict())
    )

    series = []
    # attempt to compute the area that should be visible on the plot.
    xmin, xmax, ymin, ymax = oo, -oo, oo, -oo
    for (expr, r1, r2, label, rendering_kw) in args:
        skw = series_kw.copy()
        if rendering_kw is not None:
            skw["rendering_kw"] = rendering_kw
        s = ImplicitSeries(expr, r1, r2, label, **skw)
        if (not s.start_x.free_symbols) and (s.start_x < xmin):
            xmin = s.start_x
        if (not s.end_x.free_symbols) and (s.end_x > xmax):
            xmax = s.end_x
        if (not s.start_y.free_symbols) and (s.start_y < ymin):
            ymin = s.start_y
        if (not s.end_y.free_symbols) and (s.end_y > ymax):
            ymax = s.end_y
        series.append(s)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    _set_labels(series, labels, rendering_kw)
    series += _create_generic_data_series(**kwargs)
    if (xmin != oo) and (xmax != -oo):
        kwargs.setdefault("xlim", (xmin, xmax))
    if (ymin != oo) and (ymax != -oo):
        kwargs.setdefault("ylim", (ymin, ymax))
    kwargs.setdefault("xlabel", lambda use_latex: series[-1].var_x.name if not use_latex else latex(series[0].var_x))
    kwargs.setdefault("ylabel", lambda use_latex: series[-1].var_y.name if not use_latex else latex(series[0].var_y))

    if kwargs.get("params", dict()):
        return create_interactive_plot(*series, **kwargs)

    Backend = kwargs.pop("backend", TWO_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def plot_polar(*args, **kwargs):
    """The following function creates a 2D polar plot.

    By default, it uses an equal aspect ratio and doesn't apply a colormap.

    Parameters
    ==========

    This function is going to call ``plot_parametric``: refer to its
    documentation for the full list of keyword arguments.

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
       :small-size: 800, 500

       from sympy import *
       from spb import *
       import param
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
               a: param.Integer(6, label="a"),
               b: param.Integer(12, label="b"),
               c: param.Integer(18, label="c"),
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
           use_latex=False,
           servable=True,
           imodule="panel"
       )

    See Also
    ========

    plot, plot_parametric, plot3d, plot_implicit, plot_geometry,
    plot_piecewise, plot_list

    """
    # polar_axis = kwargs.pop("polar_axis", False)
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
    # apply polar transformation
    for i, pe in enumerate(plot_expr):
        r = pe[0]
        theta = pe[1][0]
        plot_expr[i] = (r * cos(theta), r * sin(theta), *pe[1:])

    return plot_parametric(*plot_expr, **kwargs)


def plot_geometry(*args, **kwargs):
    """Plot entities from the sympy.geometry module.

    Parameters
    ==========

    args :
        geom : GeometryEntity
            Represent the geometric entity to be plotted.

        label : str, optional
            The name of the geometry entity to be eventually shown on the
            legend. If not provided, the string representation of `geom`
            will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines or fills. Refer to
            the plotting library (backend) manual for more informations.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    is_filled : boolean
        Default to True. Fill the polygon/circle/ellipse.

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of ``geom`` will be used. The number of labels must be
        equal to the number of geometric entities.

    params : dict
        A dictionary in which the keys are symbols, enabling two different
        modes of operation:

        1. If the values are numbers, the dictionary acts like a substitution
           dictionary for the provided geometric entities.

        2. If the values are tuples representing parameters, the dictionary
           enables the interactive-widgets plot, which doesn't support the
           adaptive algorithm (meaning it will use ``adaptive=False``).
           Learn more by reading the documentation of ``iplot``.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}. Only available with ``MatplotlibBackend``.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        functions to customize the appearance of lines and/or fills. Refer to
        the plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

    show : bool, optional
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel, zlabel : str, optional
        Labels for the x-axis, y-axis or z-axis, respectively.

    xlim, ylim, zlim : (float, float), optional
        Denotes the x-axis limits, y-axis limits or z-axis limits,
        respectively, ``(min, max)``, visible in the chart.


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
       [0]: interactive geometry entity: RegularPolygon(Point2D(a, b), c, d, 0) and parameters (a, b, c, d)
       [1]: interactive geometry entity: RegularPolygon(Point2D(a + 2, b + 3), c, d + 1, 0) and parameters (a, b, c, d)

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
       import param
       a, b, c, d = symbols("a, b, c, d")
       plot_geometry(
           (Polygon((a, b), c, n=d), "a"),
           (Polygon((a + 2, b + 3), c, n=d + 1), "b"),
           params = {
               a: (0, -1, 1),
               b: (1, -1, 1),
               c: (2, 1, 2),
               d: param.Integer(3, softbounds=(3, 8), label="n")
           },
           aspect="equal", is_filled=False, use_latex=False,
           xlim=(-2.5, 5.5), ylim=(-3, 6.5), imodule="panel")

    See Also
    ========

    plot, plot_piecewise, plot_polar, plot_list

    """
    args = _plot_sympify(args)

    series = []
    if not all([isinstance(a, (list, tuple, Tuple)) for a in args]):
        args = [args]

    params = kwargs.get("params", None)
    plot_expr = []

    for a in args:
        exprs, ranges, label, rendering_kw = _unpack_args(*a)

        kw = kwargs.copy()
        r = ranges if len(ranges) > 0 else [None]
        if len(exprs) == 1:
            series.append(GeometrySeries(exprs[0], *r, label, **kw))
        else:
            # this is the case where the user provided: v1, v2, ..., range
            # we use the same ranges for each expression
            for e in exprs:
                series.append(GeometrySeries(e, *r, str(e), **kw))

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    _set_labels(series, labels, rendering_kw)

    any_3D = any(s.is_3D for s in series)
    if ("aspect" not in kwargs) and (not any_3D):
        kwargs["aspect"] = "equal"

    is_interactive = False
    if params is not None:
        param = import_module("param")
        ipywidgets = import_module("ipywidgets")
        has_param, has_ipywidgets, has_tuples = False, False, False
        if param and any(isinstance(t, param.Parameter) for t in params.values()):
            has_param = True
        if ipywidgets and any(isinstance(t, ipywidgets.Widget)
            for t in params.values()):
            has_ipywidgets = True
        if any(hasattr(t, "__iter__") for t in params.values()):
            has_tuples     = True

        is_interactive = any([has_param, has_ipywidgets, has_tuples])

    if is_interactive:
        return create_interactive_plot(*series, **kwargs)

    Backend = kwargs.pop("backend", THREE_D_B if any_3D else TWO_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def plot_list(*args, **kwargs):
    """Plots lists of coordinates (ie, lists of numbers).

    Typical usage examples are in the followings:

    - Plotting coordinates of a single function.
        `plot_list(x, y, **kwargs)`
    - Plotting coordinates of multiple functions adding custom labels.
        `plot_list((x1, y1, label1), (x2, y2, label2), **kwargs)`


    Parameters
    ==========

    args :
        x : list or tuple
            x-coordinates

        y : list or tuple
            y-coordinates

        label : str, optional
            The label to be shown in the legend.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines. Refer to the
            plotting library (backend) manual for more informations.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}. Only available with MatplotlibBackend.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    is_point : boolean, optional
        Default to False, which will render a line connecting all the points.
        If True, a scatter plot will be generated.

    is_filled : boolean, optional
        Default to False, which will render empty circular markers. It only
        works if ``is_point=True``.
        If True, filled circular markers will be rendered.

    label : str or list/tuple, optional
        The label to be shown in the legend. The number of labels must be
        equal to the number of expressions.

    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot. Learn more by reading the
        documentation of ``iplot``.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

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
        Labels for the x-axis or y-axis, respectively.

    xscale, yscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis or y-axis, respectively.
        Default to ``'linear'``.

    xlim, ylim : (float, float), optional
        Denotes the x-axis limits or y-axis limits, respectively,
        ``(min, max)``, visible in the chart.


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

       >>> plot_list(([0], [0], "A"), ([1], [1], "B"), ([2], [0], "C"),
       ...     is_point=True, is_filled=True)
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
       ...     is_point=True)
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
           use_cm=False, aspect="equal", show=False, use_latex=False,
           params=params, imodule="panel")
       # plot points
       p2 = plot_list(
           ([cos(t)], [sin(t)], "A"),
           ([cos(2 * t) / 2], [sin(2 * t) / 2], "B"),
           rendering_kw={"marker": "s", "markerfacecolor": None},
           params=params, is_point=True, show=False)
       (p1 + p2).show()

    See Also
    ========

    plot, plot_parametric, plot3d, plot_piecewise

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

        kw = kwargs.copy()
        kw["rendering_kw"] = rendering_kw
        series.append(List2DSeries(*a[:2], label, **kw))

    _set_labels(series, g_labels, g_rendering_kw)
    if kwargs.get("params", None):
        return create_interactive_plot(*series, **kwargs)

    Backend = kwargs.pop("backend", TWO_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def plot3d_list(*args, **kwargs):
    """Plots lists of coordinates (ie, lists of numbers) in 3D space.

    Typical usage examples are in the followings:

    - Plotting coordinates of a single function.
        `plot3d_list(x, y, **kwargs)`
    - Plotting coordinates of multiple functions adding custom labels.
        `plot3d_list((x1, y1, label1), (x2, y2, label2), **kwargs)`


    Parameters
    ==========

    args :
        x : list or tuple or 1D NumPy array
            x-coordinates

        y : list or tuple or 1D NumPy array
            y-coordinates

        z : list or tuple or 1D NumPy array
            z-coordinates

        label : str, optional
            The label to be shown in the legend.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines. Refer to the
            plotting library (backend) manual for more informations.

    backend : Plot, optional
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    color_func : callable, optional
        A numerical function of 3 variables, x, y, z defining the line color.
        Default to None. Requires ``use_cm=True`` in order to be applied.

    is_point : boolean, optional
        Default to False, which will render a line connecting all the points.
        If True, a scatter plot will be generated.

    is_filled : boolean, optional
        Default to True, which will render filled circular markers. It only
        works if `is_point=True`. If True, filled circular markers will be
        rendered. Note that some backend might not support this feature.

    label : str or list/tuple, optional
        The label to be shown in the legend. The number of labels must be
        equal to the number of expressions.

    params : dict
        A dictionary mapping symbols to parameters. This keyword argument
        enables the interactive-widgets plot. Learn more by reading the
        documentation of ``iplot``.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

    show : bool, optional
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    use_cm : boolean, optional
        If True, apply a color map to the parametric lines.
        If False, solid colors will be used instead. Default to True.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel, zlabel : str, optional
        Label for the x-axis, y-axis, z-axis, respectively.

    xlim, ylim, zlim : (float, float), optional
        Denotes the x-axis limits, y-axis limits or z-axis limits,
        respectively, ``(min, max)``, visible in the chart.


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
           show=False, is_point=False)
       p2 = plot3d_list(
           [t * cos(t)], [t * sin(t)], [t],
           params={t: (3*pi, 0, 6*pi)},
           backend=PB, show=False, is_point=True, use_latex=False,
           imodule="panel")
       (p2 + p1).show()

    See Also
    ========

    plot, plot_list, plot3d_parametric_line

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

        kw = kwargs.copy()
        kw["rendering_kw"] = rendering_kw
        series.append(List3DSeries(*a[:3], label, **kw))

    _set_labels(series, g_labels, g_rendering_kw)
    if kwargs.get("params", None):
        return create_interactive_plot(*series, **kwargs)

    Backend = kwargs.pop("backend", TWO_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def plot_piecewise(*args, **kwargs):
    """Plots univariate piecewise functions.

    Typical usage examples are in the followings:

    - Plotting a single expression with a single range.
        `plot_piecewise(expr, range, **kwargs)`
    - Plotting a single expression with the default range (-10, 10).
        `plot_piecewise(expr, **kwargs)`
    - Plotting multiple expressions with a single range.
        `plot_piecewise(expr1, expr2, ..., range, **kwargs)`
    - Plotting multiple expressions with multiple ranges.
        `plot_piecewise((expr1, range1), (expr2, range2), ..., **kwargs)`
    - Plotting multiple expressions with multiple ranges and custom labels.
        `plot_piecewise((expr1, range1, label1), (expr2, range2, label2), ..., **kwargs)`


    Parameters
    ==========

    args :
        expr : Expr
            Expression representing the function of one variable to be
            plotted.

        range: (symbol, min, max)
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
        implemented in [#fn5]_ to create smooth plots. Use ``adaptive_goal``
        and ``loss_fn`` to further customize the output.

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
          must return a float number. Refer to [#fn5]_ for more information.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}. Only available with ``MatplotlibBackend``.

    backend : Plot, optional
        A subclass of ``Plot``, which will perform the rendering.
        Default to ``MatplotlibBackend``.

    detect_poles : boolean, optional
        Chose whether to detect and correctly plot poles.
        Defaulto to ``False``. To improve detection, increase the number of
        discretization points ``n`` and/or change the value of ``eps``.

    dots : boolean
        Wheter to show circular markers at the endpoints. Default to True.

    eps : float, optional
        An arbitrary small value used by the ``detect_poles`` algorithm.
        Default value to 0.1. Before changing this value, it is recommended to
        increase the number of discretization points.

    force_real_eval : boolean, optional
        Default to False, with which the numerical evaluation is attempted
        over a complex domain, which is slower but produces correct results.
        Set this to True if performance is of paramount importance, but be
        aware that it might produce wrong results. It only works with
        ``adaptive=False``.

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of `expr` will be used. If a list/tuple is provided, the
        number of labels must be equal to the number of expressions.

    loss_fn : callable or None
        The loss function to be used by the adaptive learner.
        Possible values:

        * ``None`` (default): it will use the ``default_loss`` from the
          ``adaptive`` module.
        * callable : Refer to [#fn5]_ for more information. Specifically,
          look at ``adaptive.learner.learner1D`` to find more loss functions.

    n : int, optional
        Used when the ``adaptive=False``. The function is uniformly
        sampled at ``n`` number of points. Default value to 1000.
        If the ``adaptive=True``, this parameter will be ignored.

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

    tx, ty : callable, optional
        Apply a numerical function to the discretized domain in the
        x and y directions, respectively.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel, ylabel : str, optional
        Labels for the x-axis or y-axis, respectively.

    xscale, yscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis or y-axis, respectively.
        Default to `'linear'`.

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)``, visible in the chart.
        Note that the function is still being evaluated over the specified
        ``range``.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)``, visible in the chart.


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
    a dotted line style.

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
       ...   ylim=(-2, 2), detect_poles=True)
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


    References
    ==========

    .. [#fn5] https://github.com/python-adaptive/adaptive


    See Also
    ========

    plot, plot_polar, plot_parametric, plot3d, plot_implicit, plot_geometry,
    plot_list

    """
    if kwargs.pop("params", None) is not None:
        raise NotImplementedError(
            "plot_piecewise doesn't support interactive widgets.")

    Backend = kwargs.pop("backend", TWO_D_B)
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 1)
    if any(callable(p[0]) for p in plot_expr):
        raise TypeError("plot_piecewise requires symbolic expressions.")
    show = kwargs.get("show", True)
    free = set()
    for p in plot_expr:
        free |= p[0].free_symbols
    x = free.pop() if free else Symbol("x")

    fx = lambda use_latex: x.name if not use_latex else latex(x)
    wrap = lambda use_latex: "f(%s)" if not use_latex else r"f\left(%s\right)"
    fy = lambda use_latex: wrap(use_latex) % fx(use_latex)
    kwargs.setdefault("xlabel", fx)
    kwargs.setdefault("ylabel", fy)
    kwargs = _set_discretization_points(kwargs, LineOver1DRangeSeries)
    kwargs.setdefault("legend", False)
    kwargs["process_piecewise"] = True

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    if isinstance(labels, str):
        labels = [labels] * len(plot_expr)

    # NOTE: rendering_kw keyword argument is not implemented in this function
    # because it would override the optimal settings chosen by the backend.
    # If a user want to set custom rendering keywords, just use the notation
    # (expr, range, label [optional], rendering_kw [optional])
    color_series_dict = dict()
    for i, a in enumerate(plot_expr):
        series = _build_line_series(a, **kwargs)
        if i < len(labels):
            _set_labels(series, [labels[i]] * len(series), None)
        color_series_dict[i] = series

    # NOTE: let's overwrite this keyword argument: the dictionary will be used
    # by the backend to assign the proper colors to the pieces
    kwargs["process_piecewise"] = color_series_dict

    return _instantiate_backend(Backend, **kwargs)
