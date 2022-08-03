"""Plotting module for Sympy.

A plot is represented by the `Plot` class that contains a list of the data
series to be plotted. The data series are instances of classes meant to
simplify getting points and meshes from sympy expressions.

This module gives only the essential. Especially if you need publication ready
graphs and this module is not enough for you, use directly the backend, which
can be accessed with the `fig` attribute:
* MatplotlibBackend.fig: returns a Matplotlib figure.
* BokehBackend.fig: return the Bokeh figure object.
* PlotlyBackend.fig: return the Plotly figure object.
* K3DBackend.fig: return the K3D plot object.

Simplicity of code takes much greater importance than performance. Don't use
it if you care at all about performance.
"""

from spb.defaults import TWO_D_B, THREE_D_B
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    SurfaceOver2DRangeSeries, ContourSeries, ParametricSurfaceSeries,
    ImplicitSeries, _set_discretization_points,
    List2DSeries, GeometrySeries, Implicit3DSeries,
    InteractiveSeries
)
from spb.utils import (
    _plot_sympify, _check_arguments, _unpack_args, _instantiate_backend
)
from sympy import (
    latex, Tuple, Expr, Symbol, Wild, oo, Sum, sign, Piecewise, piecewise_fold,
    Plane, FiniteSet, Interval, Union
)
# NOTE: from sympy import EmptySet is a different thing!!!
from sympy.sets.sets import EmptySet


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
    of coditions (for example, `((x < 0) | (x > 2))`), than two or more
    series of the same expression are created (for example, one covering
    `x < 0` and the other covering `x > 2`), both having the same label.

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
    Let's consider the following summation: `Sum(1 / x**2, (x, 1, oo))`
    The current implementation of lambdify (SymPy 1.9 at the time of
    writing this) will create something of this form:
    `sum(1 / x**2 for x in range(1, INF))`
    The problem is that type(INF) is float, while `range` requires integers,
    thus the evaluation will fails.
    Instead of modifying `lambdify` (which requires a deep knowledge),
    let's apply this quick dirty hack: substitute symbolic `oo` with an
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


def _create_interactive_plot(*plot_expr, **kwargs):
    # NOTE: the iplot module is really slow to load, so let's load it only when
    # it is necessary
    from spb.interactive import iplot
    return iplot(*plot_expr, **kwargs)


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
            representation of `expr` will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines. Refer to the
            plotting library (backend) manual for more informations.

    adaptive : bool, optional
        The default value is set to `True`, which uses the adaptive algorithm
        implemented in [#fn1]_ to create smooth plots. Use `adaptive_goal`
        and `loss_fn` to further customize the output.

        Set adaptive to `False` and specify `n` if uniform sampling is
        required.

    adaptive_goal : callable, int, float or None
        Controls the "smoothness" of the evaluation. Possible values:

        * `None` (default):  it will use the following goal:
          `lambda l: l.loss() < 0.01`
        * number (int or float). The lower the number, the more
          evaluation points. This number will be used in the following goal:
          `lambda l: l.loss() < number`
        * callable: a function requiring one input element, the learner. It
          must return a float number. Refer to [#fn1]_ for more information.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}. Only available with `MatplotlibBackend`.

    backend : Plot, optional
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    color_func : callable, optional
        A function of 2 variables, x, y (the points computed by the internal
        algorithm) which defines the line color. Default to None.

    detect_poles : boolean
        Chose whether to detect and correctly plot poles.
        Defaulto to `False`. To improve detection, increase the number of
        discretization points `n` and/or change the value of `eps`.

    eps : float
        An arbitrary small value used by the `detect_poles` algorithm.
        Default value to 0.1. Before changing this value, it is recommended to
        increase the number of discretization points.

    is_point : boolean, optional
        Default to False, which will render a line connecting all the points.
        If True, a scatter plot will be generated.

    is_filled : boolean, optional
        Default to True, which will render empty circular markers. It only
        works if `is_point=True`.
        If False, filled circular markers will be rendered.

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of `expr` will be used. The number of labels must be
        equal to the number of expressions.

    loss_fn : callable or None
        The loss function to be used by the `adaptive` learner.
        Possible values:

        * `None` (default): it will use the `default_loss` from the
          `adaptive` module.
        * callable : Refer to [#fn1]_ for more information. Specifically,
          look at `adaptive.learner.learner1D` to find more loss functions.

    n : int, optional
        Used when the `adaptive` is set to `False`. The function is uniformly
        sampled at `n` number of points. Default value to 1000.
        If the `adaptive` flag is set to `True`, this parameter will be
        ignored.

    only_integers : boolean, optional
        Default to `False`. If `True`, discretize the domain with integer
        numbers, which can be useful to plot sums. It only works when
        `adaptive=False`. When `only_integers=True`, the number of
        discretization points is choosen by the algorithm.

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
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    steps : boolean, optional
        Default to `False`. If `True`, connects consecutive points with steps
        rather than straight segments.

    sum_bound : int, optional
        When plotting sums, the expression will be pre-processed in order
        to replace lower/upper bounds set to +/- infinity with this +/-
        numerical value. Default value to 1000. Note: the higher this number,
        the slower the evaluation.

    title : str, optional
        Title of the plot.

    tx : callable, optional
        Apply a numerical function to the discretized x-direction.

    ty : callable, optional
        Apply a numerical function to the output of the numerical evaluation,
        the y-direction.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis. Default to 'linear'.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis. Default to 'linear'.

    xlim : (float, float), optional
        Denotes the x-axis limits, `(min, max)`.

    ylim : (float, float), optional
        Denotes the y-axis limits, `(min, max)`.


    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, sin, pi, tan, exp
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

    Multiple plots with single range.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x, x**2, x**3, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x for x over (-5.0, 5.0)
       [1]: cartesian line: x**2 for x over (-5.0, 5.0)
       [2]: cartesian line: x**3 for x over (-5.0, 5.0)

    Multiple plots with different ranges, custom labels and custom rendering:
    the first expression will have a dashed line style when plotted with
    ``MatplotlibBackend``.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot((x**2, (x, -6, 6), "$f_{1}$", {"linestyle": "--"}),
       ...      (x, (x, -5, 5), "f2"))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-6.0, 6.0)
       [1]: cartesian line: x for x over (-5.0, 5.0)

    No adaptive sampling.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x**2, adaptive=False, n=400)
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-10.0, 10.0)

    Plotting a summation in which the free symbol of the expression is not
    used in the lower/upper bounds:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import Sum, oo
       >>> plot(Sum(1 / x ** y, (x, 1, oo)), (y, 2, 10), sum_bound=1e03)
       Plot object containing:
       [0]: cartesian line: Sum(x**(-y), (x, 1, 1000)) for y over (2.0, 10.0)

    Plotting a summation in which the free symbol of the expression is
    used in the lower/upper bounds. Here, the discretization variable must
    assume integer values:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(Sum(1 / x, (x, 1, y)), (y, 2, 10), adaptive=False,
       ...     only_integers=True)
       Plot object containing:
       [0]: cartesian line: Sum(1/x, (x, 1, y)) for y over (2.0, 10.0)

    Detect singularities and apply a transformation function to the discretized
    domain in order to convert radians to degrees:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> plot(tan(x), (x, -1.5*pi, 1.5*pi), adaptive_goal=0.001,
       ...      detect_poles=True, tx=np.rad2deg, ylim=(-7, 7),
       ...      xlabel="x [deg]")
       Plot object containing:
       [0]: cartesian line: tan(x) for x over (-10.0, 10.0)

    Applying a colormap with a color function:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(cos(exp(-x)), (x, -pi, 0), "frequency",
       ...      adaptive=False, color_func=lambda x, y: np.exp(-x))
       Plot object containing:
       [0]: cartesian line: cos(exp(-x)) for x over (-3.141592653589793, 0.0)

    Plotting a numerical function instead of a symbolic expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> plot(lambda t: np.cos(np.exp(-t)), ("t", -pi, 0))


    Interactive-widget plot of an oscillator. Refer to ``iplot`` documentation
    to learn more about the ``params`` dictionary.

    .. code-block:: python

       from sympy import *
       from spb import *
       x, a, b, c = symbols("x, a, b, c")
       plot(
           (cos(a * x + b) * exp(-c * x), "oscillator"),
           (exp(-c * x), "upper limit", {"linestyle": ":"}),
           (-exp(-c * x), "lower limit", {"linestyle": ":"}),
           (x, 0, 2 * pi),
           params={
               a: (1, 0, 10),     # frequency
               b: (0, 0, 2 * pi), # phase
               c: (0.25, 0, 1)    # damping
           },
           ylim=(-1.25, 1.25)
       )

    References
    ==========

    .. [#fn1] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot_polar, plot_parametric, plot_contour, plot3d, plot3d_parametric_line,
    plot3d_parametric_surface, plot_implicit, plot_geometry, plot_piecewise,
    plot3d_implicit, iplot

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

    if params:
        return _create_interactive_plot(*plot_expr, **kwargs)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _build_line_series(*plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

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
    - Plotting multiple parametric curves with different ranges and
        custom labels
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
            example, `(u, 0, 5)`. If the range is not specified, then a
            default range of (-10, 10) is used.

            However, if the arguments are specified as
            `(expr_x, expr_y, range), ...`, you must specify the ranges
            for each expressions manually.

        `label` : str, optional
            The label to be shown in the legend. If not provided, the string
            representation of `expr_x` and `expr_y` will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines. Refer to the
            plotting library (backend) manual for more informations.

    adaptive : bool, optional
        The default value is set to `True`, which uses the adaptive algorithm
        implemented in [#fn2]_ to create smooth plots. Use `adaptive_goal`
        and `loss_fn` to further customize the output.

        Set adaptive to `False` and specify `n` if uniform sampling is
        required.

    adaptive_goal : callable, int, float or None
        Controls the "smoothness" of the evaluation. Possible values:

        * `None` (default):  it will use the following goal:
          `lambda l: l.loss() < 0.01`
        * number (int or float). The lower the number, the more
          evaluation points. This number will be used in the following goal:
          `lambda l: l.loss() < number`
        * callable: a function requiring one input element, the learner. It
          must return a float number. Refer to [#fn2]_ for more information.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}. Only available with `MatplotlibBackend`.

    backend : Plot, optional
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    color_func : callable, optional
        A function defining the line color. The arity can be:

        * 1 argument: ``f(t)``, where ``t`` is the parameter.
        * 2 arguments: ``f(x, y)`` where ``x, y`` are the coordinates of the
          points.
        * 3 arguments: ``f(x, y, t)``.

        Default to None.

    label : str or list/tuple, optional
        The label to be shown in the legend or in the colorbar. If not
        provided, the string representation of `expr` will be used. The number
        of labels must be equal to the number of expressions.

    loss_fn : callable or None
        The loss function to be used by the `adaptive` learner.
        Possible values:

        * `None` (default): it will use the `default_loss` from the
          `adaptive` module.
        * callable : Refer to [#fn2]_ for more information. Specifically,
          look at `adaptive.learner.learner1D` to find more loss functions.

    n : int, optional
        Used when the `adaptive` is set to `False`. The function is uniformly
        sampled at `n` number of points. Default value to 1000.
        If the `adaptive` flag is set to `True`, this parameter will be
        ignored.

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
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    tz : callable, optional
        Apply a numerical function to the discretized range.

    use_cm : boolean, optional
        If True, apply a color map to the parametric lines.
        If False, solid colors will be used instead. Default to True.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis. Default to 'linear'.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis. Default to 'linear'.

    xlim : (float, float), optional
        Denotes the x-axis limits, `(min, max)`.

    ylim : (float, float), optional
        Denotes the y-axis limits, `(min, max)`.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi
       >>> from spb.functions import plot_parametric
       >>> u, v = symbols('u, v')

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

       >>> plot_parametric((cos(u), sin(u)), (u, cos(u)), (u, -3, 3),
       ...      use_cm=False)
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-3.0, 3.0)
       [1]: parametric cartesian line: (u, cos(u)) for u over (-3.0, 3.0)

    A parametric plot with multiple expressions with different ranges,
    custom labels, custom rendering options and a transformation function
    applied to the discretized ranges to convert radians to degrees:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> plot_parametric(
       ...      (3 * cos(u), 3 * sin(u), (u, 0, 2 * pi), "u [deg]", {"lw": 3}),
       ...      (3 * cos(2 * v), 5 * sin(4 * v), (v, 0, pi), "v [deg]"),
       ...      aspect="equal", tz=np.rad2deg)
       Plot object containing:
       [0]: parametric cartesian line: (3*cos(u), 3*sin(u)) for u over (0.0, 6.283185307179586)
       [1]: parametric cartesian line: (3*cos(2*u), 5*sin(4*u)) for u over (0.0, 3.141592653589793)

    Plotting a numerical function instead of a symbolic expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> fx = lambda t: np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12)**5)
       >>> fy = lambda t: np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12)**5)
       >>> plot_parametric(fx, fy, ("t", 0, 12 * pi), title="Butterfly Curve",
       ...     use_cm=False)

    Interactive-widget plot. Refer to ``iplot`` documentation to learn more
    about the ``params`` dictionary.

    .. code-block:: python

       from sympy import *
       from spb import *
       x, a = symbols("x a")
       plot_parametric(
           cos(a * x), sin(x), (x, 0, 2*pi),
           params={a: (1, 0, 2)},
           aspect="equal", xlim=(-1.25, 1.25), ylim=(-1.25, 1.25)
       )

    References
    ==========

    .. [#fn2] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot, plot_polar, plot_contour, plot3d, plot3d_parametric_line,
    plot3d_parametric_surface, plot_implicit, plot_geometry, plot_piecewise,
    plot3d_implicit, iplot

    """
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, Parametric2DLineSeries)
    plot_expr = _check_arguments(args, 2, 1, **kwargs)

    if kwargs.get("params", None):
        return _create_interactive_plot(*plot_expr, **kwargs)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Parametric2DLineSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

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
        The default value is set to `True`, which uses the adaptive algorithm
        implemented in [#fn3]_ to create smooth plots. Use `adaptive_goal`
        and `loss_fn` to further customize the output.

        Set adaptive to `False` and specify `n` if uniform sampling is
        required.

    adaptive_goal : callable, int, float or None
        Controls the "smoothness" of the evaluation. Possible values:

        * `None` (default):  it will use the following goal:
          `lambda l: l.loss() < 0.01`
        * number (int or float). The lower the number, the more
          evaluation points. This number will be used in the following goal:
          `lambda l: l.loss() < number`
        * callable: a function requiring one input element, the learner. It
          must return a float number. Refer to [#fn3]_ for more information.

    backend : Plot, optional
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    color_func : callable, optional
        A function defining the line color. The arity can be:

        * 1 argument: ``f(t)``, where ``t`` is the parameter.
        * 3 arguments: ``f(x, y, z)`` where ``x, y, z`` are the coordinates of
          the points.
        * 4 arguments: ``f(x, y, z, t)``.

        Default to None.

    label : str or list/tuple, optional
        The label to be shown in the legend or in the colorbar. If not
        provided, the string representation of `expr` will be used. The number
        of labels must be equal to the number of expressions.

    loss_fn : callable or None
        The loss function to be used by the `adaptive` learner.
        Possible values:

        * `None` (default): it will use the `default_loss` from the
          `adaptive` module.
        * callable : Refer to [#fn3]_ for more information. Specifically,
          look at `adaptive.learner.learner1D` to find more loss functions.

    n : int, optional
        Used when the `adaptive` is set to `False`. The function is uniformly
        sampled at `n` number of points. Default value to 1000.
        If the `adaptive` flag is set to `True`, this parameter will be
        ignored.

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
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    tz : callable, optional
        Apply a numerical function to the discretized parameter.

    use_cm : boolean, optional
        If True, apply a color map to the parametric lines.
        If False, solid colors will be used instead. Default to True.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    zlabel : str, optional
        Label for the z-axis.

    xlim : (float, float), optional
        Denotes the x-axis limits, `(min, max)`.

    ylim : (float, float), optional
        Denotes the y-axis limits, `(min, max)`.

    zlim : (float, float), optional
        Denotes the z-axis limits, `(min, max)`.


    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi
       >>> from spb.functions import plot3d_parametric_line
       >>> u, v = symbols('u, v')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line(cos(u), sin(u), u, (u, -5, 5))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)


    Multiple plots with different ranges, custom labels and custom rendering
    options.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line((cos(u), sin(u), u, (u, -5, 5), "u"),
       ...     (sin(v), v**2, v, (v, -3, 3), "v", {"lw": 3, "cmap": "hsv"}))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)
       [1]: 3D parametric cartesian line: (sin(u), u**2, u) for u over (-3.0, 3.0)

    Plotting a numerical function instead of a symbolic expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> fx = lambda t: (1 + 0.25 * np.cos(75 * t)) * np.cos(t)
       >>> fy = lambda t: (1 + 0.25 * np.cos(75 * t)) * np.sin(t)
       >>> fz = lambda t: t + 2 * np.sin(75 * t)
       >>> plot3d_parametric_line(fx, fy, fz, ("t", 0, 6 * pi),
       ...     title="Helical Toroid")

    Interactive-widget plot. Refer to ``iplot`` documentation to learn more
    about the ``params`` dictionary.

    .. code-block:: python

       from sympy import *
       from spb import *
       x, a = symbols("x a")
       plot3d_parametric_line(
           cos(a * x), sin(x), a * x, (x, 0, 2*pi),
           params={a: (1, 0, 2)},
           xlim=(-1.25, 1.25), ylim=(-1.25, 1.25)
       )

    References
    ==========

    .. [#fn3] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot, plot_polar, plot3d, plot_contour, plot3d_parametric_surface,
    plot_implicit, plot_geometry, plot_parametric, plot_piecewise,
    plot3d_implicit, iplot

    """
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, Parametric3DLineSeries)
    plot_expr = _check_arguments(args, 3, 1, **kwargs)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")

    if kwargs.get("params", None):
        return _create_interactive_plot(*plot_expr, **kwargs)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Parametric3DLineSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

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
        free_x |= {p[1][0]} if isinstance(p[1][0], Symbol) else {Symbol(p[1][0])}
        free_y |= {p[2][0]} if isinstance(p[2][0], Symbol) else {Symbol(p[2][0])}
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

    if kwargs.get("params", None):
        kwargs["threed"] = is_threed
        return _create_interactive_plot(*plot_expr, **kwargs)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Series, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    return _instantiate_backend(Backend, *series, **kwargs)


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

    Note that it is important to specify at least the `range_x`, otherwise the
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
            representation of `expr` will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of surfaces. Refer to the
            plotting library (backend) manual for more informations.

    adaptive : bool, optional
        The default value is set to `False`, which uses a uniform sampling
        strategy with number of discretization points `n1` and `n2` along the
        x and y directions, respectively.

        Set adaptive to `True` to use the adaptive algorithm implemented in
        [#fn4]_ to create smooth plots. Use `adaptive_goal` and `loss_fn`
        to further customize the output.

    adaptive_goal : callable, int, float or None
        Controls the "smoothness" of the evaluation. Possible values:

        * `None` (default):  it will use the following goal:
          `lambda l: l.loss() < 0.01`
        * number (int or float). The lower the number, the more
          evaluation points. This number will be used in the following goal:
          `lambda l: l.loss() < number`
        * callable: a function requiring one input element, the learner. It
          must return a float number. Refer to [#fn4]_ for more information.

    backend : Plot, optional
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    color_func : callable, optional
        A function of 3 variables, x, y, z (the points computed by the
        internal algorithm) which defines the surface color when
        ``use_cm=True``. Default to None.

    is_polar : boolean, optional
        Default to False. If True, requests a polar discretization. In this
        case, ``range_x`` represents the radius, ``range_y`` represents the
        angle.

    label : str or list/tuple, optional
        The label to be shown in the colorbar. If not provided, the string
        representation of `expr` will be used. The number of labels must be
        equal to the number of expressions.

    loss_fn : callable or None
        The loss function to be used by the `adaptive` learner.
        Possible values:

        * `None` (default): it will use the `default_loss` from the
          `adaptive` module.
        * callable : Refer to [#fn4]_ for more information. Specifically,
          look at `adaptive.learner.learnerND` to find more loss functions.

    n1 : int, optional
        The x range is sampled uniformly at `n1` of points. Default value
        is 100.

    n2 : int, optional
        The y range is sampled uniformly at `n2` of points. Default value
        is 100.

    n : int or two-elements tuple (n1, n2), optional
        If an integer is provided, the x and y ranges are sampled uniformly
        at `n` of points. If a tuple is provided, it overrides `n1` and `n2`.

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
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    tx : callable, optional
        Apply a numerical function to the discretized domain in the
        x-direction.

    ty : callable, optional
        Apply a numerical function to the discretized domain in the
        y-direction.

    tz : callable, optional
        Apply a numerical function to the results of the numerical evaluation.

    use_cm : boolean, optional
        If True, apply a color map to the surface.
        If False, solid colors will be used instead. Default to True.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    zlabel : str, optional
        Label for the z-axis.

    xlim : (float, float), optional
        Denotes the x-axis limits, `(min, max)`.

    ylim : (float, float), optional
        Denotes the y-axis limits, `(min, max)`.

    zlim : (float, float), optional
        Denotes the z-axis limits, `(min, max)`.


    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi, exp
       >>> from spb.functions import plot3d
       >>> x, y = symbols('x y')

    Single plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(cos((x**2 + y**2)), (x, -3, 3), (y, -3, 3))
       Plot object containing:
       [0]: cartesian surface: cos(x**2 + y**2) for x over (-3.0, 3.0) and y over (-3.0, 3.0)


    Single plot with a polar discretization and a color function mapping a
    colormap to the radius:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import matplotlib.cm as cm
       >>> r, theta = symbols("r, theta")
       >>> p = plot3d(
       ...     (cos(r**2) * exp(-r / 3), (r, 0, 3.25),
       ...         (theta, 0, 2 * pi), "r", {"cmap": cm.winter}),
       ...     is_polar=True, use_cm=True, legend=True,
       ...     color_func=lambda x, y, z: (x**2 + y**2)**0.5)
       Plot object containing:
       [0]: cartesian surface: exp(-r/3)*cos(r**2) for r over (0.0, 3.25) and theta over (0.0, 6.283185307179586)


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


    Multiple plots with different ranges.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d((x**2 + y**2, (x, -5, 5), (y, -5, 5)),
       ...     (x*y, (x, -3, 3), (y, -3, 3)))
       Plot object containing:
       [0]: cartesian surface: x**2 + y**2 for x over (-5.0, 5.0) and y over (-5.0, 5.0)
       [1]: cartesian surface: x*y for x over (-3.0, 3.0) and y over (-3.0, 3.0)

    Apply a transformation to the discretized ranged in order to convert
    radians to degrees:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> expr = (cos(x) + sin(x) * sin(y) - sin(x) * cos(y))**2
       ... plot3d(expr, (x, 0, pi), (y, 0, 2 * pi),
       ...     tx=np.rad2deg, ty=np.rad2deg, use_cm=True,
       ...     xlabel="x [deg]", ylabel="y [deg]")

    Plotting a numerical function instead of a symbolic expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> plot3d(lambda x, y: x * np.exp(-x**2 - y**2),
       ...     ("x", -3, 3), ("y", -3, 3), use_cm=True)

    Interactive-widget plot. Refer to ``iplot`` documentation to learn more
    about the ``params`` dictionary.

    .. code-block:: python

       from sympy import *
       from spb import *
       x, d = symbols("x d")
       plot3d(
           cos(x**2 + y**2) * exp(-(x**2 + y**2) * d), (x, -2, 2), (y, -2, 2),
           params={d: (0.25, 0, 1)}, n=50, zlim=(-1.25, 1.25))

    References
    ==========

    .. [#fn4] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot, plot_polar, plot_contour, plot_parametric, plot3d_parametric_line,
    plot3d_parametric_surface, plot_implicit, plot_geometry, plot_piecewise,
    plot3d_implicit, iplot

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

    Note that it is important to specify both the ranges.

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
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    color_func : callable, optional
        A function defining the surface color when ``use_cm=True``. The arity
        can be:

        * 1 argument: ``f(u)``, where ``u`` is the first parameter.
        * 2 arguments: ``f(u, v)`` where ``u, v`` are the parameters.
        * 3 arguments: ``f(x, y, z)`` where ``x, y, z`` are the coordinates of
          the points.
        * 5 arguments: ``f(x, y, z, u, v)``.

        Default to None.

    label : str or list/tuple, optional
        The label to be shown in the colorbar. If not provided, the string
        representation will be used. The number of labels must be
        equal to the number of expressions.

    n1 : int, optional
        The u range is sampled uniformly at `n1` of points. Default value
        is 100.

    n2 : int, optional
        The v range is sampled uniformly at `n2` of points. Default value
        is 100.

    n : int or two-elements tuple (n1, n2), optional
        If an integer is provided, the u and v ranges are sampled uniformly
        at `n` of points. If a tuple is provided, it overrides `n1` and `n2`.

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
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    use_cm : boolean, optional
        If True, apply a color map to the surface.
        If False, solid colors will be used instead. Default to True.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    zlabel : str, optional
        Label for the z-axis.

    xlim : (float, float), optional
        Denotes the x-axis limits, `(min, max)`.

    ylim : (float, float), optional
        Denotes the y-axis limits, `(min, max)`.

    zlim : (float, float), optional
        Denotes the z-axis limits, `(min, max)`.


    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin, pi
       >>> from spb.functions import plot3d_parametric_surface
       >>> u, v = symbols('u v')

    Single plot with u/v directions discretized with 200 points, showing a
    custom label and using a coloring function.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> r = 2 + sin(7 * u + 5 * v)
       >>> expr = (
       ...      r * cos(u) * sin(v),
       ...          r * sin(u) * sin(v),
       ...          r * cos(v)
       ... )
       >>> plot3d_parametric_surface(*expr, (u, 0, 2 * pi), (v, 0, pi), "u",
       ...      n=200, use_cm=True, color_func=lambda u, v: u)
       Plot object containing:
       [0]: parametric cartesian surface: ((sin(7*u + 5*v) + 2)*sin(v)*cos(u), (sin(7*u + 5*v) + 2)*sin(u)*sin(v), (sin(7*u + 5*v) + 2)*cos(v)) for u over (0.0, 6.283185307179586) and v over (0.0, 3.141592653589793)

    Plotting a numerical function instead of a symbolic expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> import numpy as np
       >>> fx = lambda u, v: (4 + np.cos(u)) * np.cos(v)
       >>> fy = lambda u, v: (4 + np.cos(u)) * np.sin(v)
       >>> fz = lambda u, v: np.sin(u)
       >>> plot3d_parametric_surface(fx, fy, fz, ("u", 0, 2 * pi),
       ...     ("v", 0, 2 * pi))


    See Also
    ========

    plot, plot_polar, plot_parametric, plot3d, plot_contour,
    plot3d_parametric_line, plot_implicit, plot_geometry, plot_piecewise,
    plot3d_implicit, iplot

    """
    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, ParametricSurfaceSeries)
    plot_expr = _check_arguments(args, 3, 2, **kwargs)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")

    if kwargs.get("params", None):
        return _create_interactive_plot(*plot_expr, **kwargs)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(ParametricSurfaceSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)
    Backend = kwargs.pop("backend", THREE_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


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
        A subclass of `Plot`, which will perform the rendering.
        Only PlotlyBackend and K3DBackend support 3D implicit plotting.

    n1 : int, optional
        The x range is sampled uniformly at `n1` of points. Default value
        is 60.

    n2 : int, optional
        The y range is sampled uniformly at `n2` of points. Default value
        is 60.

    n3 : int, optional
        The z range is sampled uniformly at `n3` of points. Default value
        is 60.

    n : int or three-elements tuple (n1, n2, n3), optional
        If an integer is provided, the x, y and z ranges are sampled uniformly
        at `n` of points. If a tuple is provided, it overrides `n1`, `n2` and
        `n3`.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of surfaces. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

    show : bool, optional
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

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

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    zlabel : str, optional
        Label for the z-axis.

    xlim : (float, float), optional
        Denotes the x-axis limits, `(min, max)`.

    ylim : (float, float), optional
        Denotes the y-axis limits, `(min, max)`.

    zlim : (float, float), optional
        Denotes the z-axis limits, `(min, max)`.


    Examples
    ========

    .. jupyter-execute::

       from sympy import symbols
       from spb import plot3d_implicit, PB, KB
       x, y, z = symbols('x, y, z')
       plot3d_implicit(
           x**2 + y**3 - z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2), backend=PB)

    .. jupyter-execute::

       p = plot3d_implicit(
           x**4 + y**4 + z**4 - (x**2 + y**2 + z**2 - 0.3),
           (x, -2, 2), (y, -2, 2), (z, -2, 2), backend=PB)

    Visualize the isocontours from `isomin=0` to `isomax=2` by providing a
    ``rendering_kw`` dictionary:

    .. jupyter-execute::

       plot3d_implicit(
           1/x**2 - 1/y**2 + 1/z**2, (x, -2, 2), (y, -2, 2), (z, -2, 2),
           {
               "isomin": 0, "isomax": 2,
               "colorscale":"aggrnyl", "showscale":True
           },
           backend=PB
       )

    Plotting a numerical function instead of a symbolic expression:

    .. jupyter-execute::

       import numpy as np
       plot3d_implicit(lambda x, y, z: x**2 + y**2 - z**2,
           ("x", -3, 3), ("y", -3, 3), ("z", 0, 3), backend=PB)

    See Also
    ========

    plot, plot_polar, plot_parametric, plot3d, plot_contour,
    plot3d_parametric_line, plot_implicit, plot_geometry, plot_piecewise,
    plot3d_parametric_surface

    """
    if kwargs.pop("params", None) is not None:
        raise NotImplementedError(
            "plot3d_implicit doesn't support interactive widgets.")

    args = _plot_sympify(args)
    kwargs = _set_discretization_points(kwargs, Implicit3DSeries)
    plot_expr = _check_arguments(args, 1, 3)

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

    Backend = kwargs.pop("backend", THREE_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def plot_contour(*args, **kwargs):
    """
    Draws contour plot of a function of two variables.

    This function signature is identical to `plot3d`: refer to its
    documentation for a list of available argument and keyword arguments.

    Parameters
    ==========

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, exp
       >>> from spb.functions import plot_contour
       >>> x, y = symbols('x, y')

    Contour of a function of two variables.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_contour(cos((x**2 + y**2)) * exp(-(x**2 + y**2) / 10),
       ...      (x, -5, 5), (y, -5, 5))
       Plot object containing:
       [0]: contour: exp(-x**2/10 - y**2/10)*cos(x**2 + y**2) for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    Interactive-widget plot. Refer to ``iplot`` documentation to learn more
    about the ``params`` dictionary.

    .. code-block:: python

       from sympy import *
       from spb import *
       x, y, a = symbols("x y a")
       plot_contour(
           cos(a * x**2 + y**2), (x, -2, 2), (y, -2, 2),
           params={a: (1, 0, 2)}, grid=False)

    See Also
    ========

    plot, plot_polar, plot_parametric, plot3d, plot3d_parametric_line,
    plot3d_parametric_surface, plot_implicit, plot_geometry, plot_piecewise,
    plot3d_implicit, iplot

    """
    Backend = kwargs.pop("backend", TWO_D_B)
    return _plot3d_plot_contour_helper(
        ContourSeries, False, Backend, *args, **kwargs)


def plot_implicit(*args, **kwargs):
    """Plot implicit equations / inequalities.

    plot_implicit, by default, generates a contour using a mesh grid of fixed
    number of points. The greater the number of points, the greater the memory
    used. By setting `adaptive=True`, interval arithmetic will be used to plot
    functions. If the expression cannot be plotted using interval arithmetic,
    it defaults to generating a contour using a mesh grid. With interval
    arithmetic, the line width can become very small; in those cases, it is
    better to use the mesh grid approach.

    Parameters
    ==========

    args :
        expr : Expr, Relational, BooleanFunction
            The equation / inequality that is to be plotted.

        ranges : tuples or Symbol
            Two tuple denoting the discretization domain, for example:
            `(x, -10, 10), (y, -10, 10)`
            To get a correct plot, at least the horizontal range must be
            provided. If no range is given, then the free symbols in the
            expression will be assigned in the order they are sorted, which
            could 'invert' the axis.

            Alternatively, a single Symbol corresponding to the horizontal
            axis must be provided, which will be internally converted to a
            range `(sym, -10, 10)`.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of contours. Refer to the
            plotting library (backend) manual for more informations.

    adaptive : Boolean
        The default value is set to False, meaning that the internal
        algorithm uses a mesh grid approach. In such case, Boolean
        combinations of expressions cannot be plotted.
        If set to True, the internal algorithm uses interval arithmetic.
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
        increasing `depth` we are increasing the number of pixels, thus
        obtaining a more accurate plot.

    n1, n2 : int
        Number of discretization points in the horizontal and vertical
        directions when `adaptive=False`. Default to 1000.

    n : int or two-elements tuple (n1, n2), optional
        If an integer is provided, the x and y ranges are sampled uniformly
        at `n` of points. If a tuple is provided, it overrides `n1` and `n2`.

    show : Boolean
        Default value is True. If set to False, the plot will not be shown.
        See `Plot` for further information.

    title : string
        The title for the plot.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel : string
        The label for the x-axis

    ylabel : string
        The label for the y-axis

    Examples
    ========

    Plot expressions:

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import symbols, Eq, And, sin, pi
        >>> from spb.functions import plot_implicit
        >>> x, y = symbols('x y')

    Providing only the symbol for the horizontal axis:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_implicit(x - 1, x)

    With the range for both symbols:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_implicit(Eq(x**2 + y**2, 3), (x, -3, 3), (y, -3, 3))

    Specify the number of discretization points for the contour algorithm:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_implicit(
        ...     (x**2 + y**2 - 1)**3 - x**2 * y**3,
        ...     (x, -1.5, 1.5), (y, -1.5, 1.5),
        ...     n = 500)

    Using adaptive meshing and Boolean expressions:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_implicit(
        ...     Eq(y, sin(x)) & (y > 0),
        ...     Eq(y, sin(x)) & (y < 0),
        ...     (x, -2 * pi, 2 * pi), (y, -4, 4),
        ...     adaptive=True)

    Using adaptive meshing with depth of recursion as argument:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_implicit(
        ...     Eq(x**2 + y**2, 5), (x, -4, 4), (y, -4, 4),
        ...     adaptive=True, depth=2)

    Plotting regions:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_implicit(y > x**2, (x, -5, 5))

    See Also
    ========

    plot, plot_polar, plot3d, plot_contour, plot3d_parametric_line,
    plot3d_parametric_surface, plot_geometry, plot3d_implicit

    """
    if kwargs.pop("params", None) is not None:
        raise NotImplementedError(
            "plot_implicit doesn't support interactive widgets.")

    # if the user is plotting a single expression, then he can pass in one
    # or two symbols to sort the axis. Ranges will then be automatically
    # created.
    args = list(args)
    if (len(args) == 2) and isinstance(args[1], Symbol):
        args[1] = Tuple(args[1], -10, 10)
    if (len(args) == 3) and isinstance(args[1], Symbol) and isinstance(args[2], Symbol):
        args[1] = Tuple(args[1], -10, 10)
        args[2] = Tuple(args[2], -10, 10)

    args = _plot_sympify(args)
    args = _check_arguments(args, 1, 2)
    kwargs = _set_discretization_points(kwargs, ImplicitSeries)

    series_kw = dict(
        n1=kwargs.pop("n1", 1000),
        n2=kwargs.pop("n2", 1000),
        depth=kwargs.pop("depth", 0),
        adaptive=kwargs.pop("adaptive", False),
        contour_kw=kwargs.pop("contour_kw", dict())
    )

    series = []
    # compute the area that should be visible on the plot
    xmin, xmax, ymin, ymax = oo, -oo, oo, -oo
    for (expr, r1, r2, label, rendering_kw) in args:
        skw = series_kw.copy()
        if rendering_kw is not None:
            skw["rendering_kw"] = rendering_kw
        s = ImplicitSeries(expr, r1, r2, label, **skw)
        if s.start_x < xmin:
            xmin = s.start_x
        if s.end_x > xmax:
            xmax = s.end_x
        if s.start_y < ymin:
            ymin = s.start_y
        if s.end_y > ymax:
            ymax = s.end_y
        series.append(s)

    # kwargs.setdefault("backend", TWO_D_B)
    kwargs.setdefault("xlim", (xmin, xmax))
    kwargs.setdefault("ylim", (ymin, ymax))
    kwargs.setdefault("xlabel", lambda use_latex: series[-1].var_x.name if not use_latex else latex(series[0].var_x))
    kwargs.setdefault("ylabel", lambda use_latex: series[-1].var_y.name if not use_latex else latex(series[0].var_y))
    Backend = kwargs.pop("backend", TWO_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def plot_polar(*args, **kwargs):
    """The following function creates a 2D polar plot.

    This function signature is identical to `plot`: refer to its
    documentation for a list of available argument and keyword arguments.

    Examples
    ========

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import symbols, sin, pi
        >>> from spb.functions import plot_polar
        >>> x = symbols('x')


    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> plot_polar(1 + sin(10 * x) / 10, (x, 0, 2 * pi))

    See Also
    ========

    plot, plot_parametric, plot3d, plot_contour, plot3d_parametric_line,
    plot3d_parametric_surface, plot_implicit, plot_geometry, plot_piecewise,
    plot3d_implicit, iplot

    """
    kwargs["is_polar"] = True
    kwargs.setdefault("axis", "equal")
    kwargs.setdefault("xlabel", "")
    kwargs.setdefault("ylabel", "")
    return plot(*args, **kwargs)


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
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    is_filled : boolean
        Default to True. Fill the polygon/circle/ellipse.

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of `geom` will be used. The number of labels must be
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
        {'center', 'auto'}. Only available with MatplotlibBackend.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        functions to customize the appearance of lines and/or fills. Refer to
        the plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

    show : bool, optional
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

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

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    zlabel : str, optional
        Label for the z-axis.

    xlim : (float, float), optional
        Denotes the x-axis limits, `(min, max)`.

    ylim : (float, float), optional
        Denotes the y-axis limits, `(min, max)`.

    zlim : (float, float), optional
        Denotes the z-axis limits, `(min, max)`.


    Examples
    ========

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import (symbols, Circle, Ellipse, Polygon,
        ...      Curve, Segment, Point2D, Point3D, Line3D, Plane,
        ...      Rational, pi, Point, cos, sin)
        >>> from spb.functions import plot_geometry
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
       [0]: geometry entity: RegularPolygon(Point2D(a, b), c, d, 0)
       [1]: geometry entity: RegularPolygon(Point2D(a + 2, b + 3), c, d + 1, 0)

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
       [2]: plane series over (x, -5, 5), (y, -4, 4), (z, -10, 10)

    Interactive-widget plot. Refer to ``iplot`` documentation to learn more
    about the ``params`` dictionary.

    .. code-block:: python

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
           aspect="equal", is_filled=False,
           xlim=(-2.5, 5.5), ylim=(-3, 6.5))

    See Also
    ========

    plot, plot_piecewise, plot_polar, iplot

    """
    args = _plot_sympify(args)

    series = []
    if not all([isinstance(a, (list, tuple, Tuple)) for a in args]):
        args = [args]

    params = kwargs.get("params", None)
    is_interactive = False if params is None else any(hasattr(t, "__iter__") for t in params.values())
    plot_expr = []

    for a in args:
        exprs, ranges, label, rendering_kw = _unpack_args(*a)

        if not is_interactive:
            kw = kwargs.copy()
            if rendering_kw is not None:
                kw["rendering_kw"] = rendering_kw
            r = ranges if len(ranges) > 0 else [None]
            if len(exprs) == 1:
                series.append(GeometrySeries(exprs[0], *r, label, **kw))
            else:
                # this is the case where the user provided: v1, v2, ..., range
                # we use the same ranges for each expression
                for e in exprs:
                    series.append(GeometrySeries(e, *r, str(e), **kw))
        else:
            plot_expr.append([*exprs, *ranges, label, rendering_kw])

    if is_interactive:
        return _create_interactive_plot(*plot_expr, **kwargs)

    # TODO: apply line_kw and fill_kw
    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    _set_labels(series, labels, rendering_kw)

    any_3D = any(s.is_3D for s in series)
    if ("aspect" not in kwargs) and (not any_3D):
        kwargs["aspect"] = "equal"

    Backend = kwargs.pop("backend", THREE_D_B if any_3D else TWO_D_B)
    return _instantiate_backend(Backend, *series, **kwargs)


def plot_list(*args, **kwargs):
    """Plots lists of coordinates (ie, lists of numbers).

    Typical usage examples are in the followings:

    - Plotting coordinates of a single function.
        `plot(x, y, **kwargs)`
    - Plotting coordinates of multiple functions adding custom labels.
        `plot((x1, y1, label1), (x2, y2, label2), **kwargs)`


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
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    is_point : boolean, optional
        Default to False, which will render a line connecting all the points.
        If True, a scatter plot will be generated.

    is_filled : boolean, optional
        Default to True, which will render empty circular markers. It only
        works if `is_point=True`.
        If False, filled circular markers will be rendered.

    label : str or list/tuple, optional
        The label to be shown in the legend. The number of labels must be
        equal to the number of expressions.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of expressions.

    show : bool, optional
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

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

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis. Default to 'linear'.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis. Default to 'linear'.

    xlim : (float, float), optional
        Denotes the x-axis limits, `(min, max)`.

    ylim : (float, float), optional
        Denotes the y-axis limits, `(min, max)`.


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
       [0]: list plot

    Scatter plot of the coordinates of multiple functions:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> xx = [t / 100 * 6 - 3 for t in list(range(101))]
       >>> yy1 = [cos(x).evalf(subs={x: t}) for t in xx]
       >>> yy2 = [sin(x).evalf(subs={x: t}) for t in xx]
       >>> plot_list((xx, yy1, "cos"), (xx, yy2, "sin"), is_point=True)
       Plot object containing:
       [0]: list plot
       [1]: list plot

    """
    if kwargs.pop("params", None) is not None:
        raise NotImplementedError(
            "plot_list doesn't support interactive widgets.")

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
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
        kw["line_kw"] = rendering_kw
        series.append(List2DSeries(*a[:2], label, **kw))

    _set_labels(series, labels, rendering_kw)

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
            representation of `expr` will be used.

        rendering_kw : dict, optional
            A dictionary of keywords/values which is passed to the backend's
            function to customize the appearance of lines. Refer to the
            plotting library (backend) manual for more informations.

    adaptive : bool, optional
        The default value is set to `True`, which uses the adaptive algorithm
        implemented in [#fn5]_ to create smooth plots. Use `adaptive_goal`
        and `loss_fn` to further customize the output.

        Set adaptive to `False` and specify `n` if uniform sampling is
        required.

    adaptive_goal : callable, int, float or None
        Controls the "smoothness" of the evaluation. Possible values:

        * `None` (default):  it will use the following goal:
          `lambda l: l.loss() < 0.01`
        * number (int or float). The lower the number, the more
          evaluation points. This number will be used in the following goal:
          `lambda l: l.loss() < number`
        * callable: a function requiring one input element, the learner. It
          must return a float number. Refer to [#fn5]_ for more information.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}. Only available with `MatplotlibBackend`.

    backend : Plot, optional
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    detect_poles : boolean
        Chose whether to detect and correctly plot poles.
        Defaulto to `False`. To improve detection, increase the number of
        discretization points `n` and/or change the value of `eps`.

    eps : float
        An arbitrary small value used by the `detect_poles` algorithm.
        Default value to 0.1. Before changing this value, it is recommended to
        increase the number of discretization points.

    label : str or list/tuple, optional
        The label to be shown in the legend. If not provided, the string
        representation of `expr` will be used. If a list/tuple is provided, the
        number of labels must be equal to the number of expressions.

    loss_fn : callable or None
        The loss function to be used by the `adaptive` learner.
        Possible values:

        * `None` (default): it will use the `default_loss` from the
          `adaptive` module.
        * callable : Refer to [#fn5]_ for more information. Specifically,
          look at `adaptive.learner.learner1D` to find more loss functions.

    n : int, optional
        Used when the `adaptive` is set to `False`. The function is uniformly
        sampled at `n` number of points. Default value to 1000.
        If the `adaptive` flag is set to `True`, this parameter will be
        ignored.

    show : bool, optional
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    tx : callable, optional
        Apply a numerical function to the discretized x-direction.

    ty : callable, optional
        Apply a numerical function to the output of the numerical evaluation,
        the y-direction.

    use_latex : boolean, optional
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis. Default to 'linear'.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis. Default to 'linear'.

    xlim : (float, float), optional
        Denotes the x-axis limits, `(min, max)`.

    ylim : (float, float), optional
        Denotes the y-axis limits, `(min, max)`.


    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, sin, cos, pi, Heaviside, Piecewise
       >>> from spb import plot_piecewise
       >>> x = symbols('x')

    Single Plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_piecewise(Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10))
       Plot object containing:
       [0]: cartesian line: 0 for x over (-10.0, 0.0)
       [1]: cartesian line: 1 for x over (1e-06, 10.0)
       [2]: list plot
       [3]: list plot

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
       ...   ylim=(-2, 2))
       Plot object containing:
       [0]: cartesian line: 0 for x over (-10.0, 0.0)
       [1]: cartesian line: 1 for x over (1e-06, 10.0)
       [2]: list plot
       [3]: list plot
       [4]: cartesian line: sin(x) for x over (-10.0, -5.000001)
       [5]: list plot
       [6]: cartesian line: cos(x) for x over (5.000001, 10.0)
       [7]: list plot
       [8]: cartesian line: 1/x for x over (-5.0, 5.0)
       [9]: list plot
       [10]: list plot


    References
    ==========

    .. [#fn5] https://github.com/python-adaptive/adaptive


    See Also
    ========

    plot, plot_polar, plot_parametric, plot_contour, plot3d,
    plot3d_parametric_line, plot3d_parametric_surface,
    plot_implicit, plot_geometry, plot_list, plot3d_implicit

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
