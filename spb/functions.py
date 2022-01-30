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

from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, Wild
from sympy.core.numbers import oo
from sympy.concrete.summations import Sum
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.sets.sets import EmptySet, FiniteSet, Interval, Union
from spb.backends.base_backend import Plot
from spb.utils import _plot_sympify, _check_arguments, _unpack_args
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    SurfaceOver2DRangeSeries, ContourSeries, ParametricSurfaceSeries,
    ImplicitSeries, _set_discretization_points,
    List2DSeries, GeometrySeries
)

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

            current_label = str(expr) if "Piecewise(" in label else label
            main_series = LineOver1DRangeSeries(
                expr, (_range[0], start, end), current_label, **kwargs)
            series.append(main_series)
            xx, yy = main_series.get_data()

            if xx[0] != _range[1]:
                correct_list = series if _set.left_open else filled_series
                correct_list.append(
                    List2DSeries([xx[0]], [yy[0]], is_point=True,
                        is_filled=not _set.left_open)
                )
            if xx[-1] != _range[2]:
                correct_list = series if _set.right_open else filled_series
                correct_list.append(
                    List2DSeries([xx[-1]], [yy[-1]], is_point=True,
                        is_filled=not _set.right_open)
                )
        elif isinstance(_set, FiniteSet):
            loc, val = [], []
            for _loc in _set.args:
                loc.append(float(_loc))
                val.append(float(expr.evalf(subs={_range[0]: _loc})))
            filled_series.append(List2DSeries(loc, val, is_point=True,
                is_filled=True))
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
        expr, r, label = arg
        if expr.has(Piecewise) and pp:
            series += _process_piecewise(expr, r, label, **kwargs)
        else:
            arg = _process_summations(sum_bound, *arg)
            series.append(LineOver1DRangeSeries(*arg, **kwargs))
    return series


def plot(*args, show=True, **kwargs):
    """Plots a function of a single variable as a curve.

    Typical usage examples are in the followings:

    - Plotting a single expression with a single range.
        `plot(expr, range, **kwargs)`
    - Plotting a single expression with the default range (-10, 10).
        `plot(expr, **kwargs)`
    - Plotting multiple expressions with a single range.
        `plot(expr1, expr2, ..., range, **kwargs)`
    - Plotting multiple expressions with multiple ranges.
        `plot((expr1, range1), (expr2, range2), ..., **kwargs)`
    - Plotting multiple expressions with multiple ranges and custom labels.
        `plot((expr1, range1, label1), (expr2, range2, label2), ..., legend=True, **kwargs)`


    Parameters
    ==========

    args :
        expr : Expr
            Expression representing the function of one variable to be
            plotted.

        range : (symbol, min, max)
            A 3-tuple denoting the range of the x variable. Default values:
            `min=-10` and `max=10`.

        label : str, optional
            The label to be shown in the legend. If not provided, the string
            representation of `expr` will be used.

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

    line_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's function to customize the appearance of the lines. Refer to the
        plotting library (backend) manual for more informations.

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

    polar : boolean
        Default to `False`. If `True`, generate a polar plot of a curve
        with radius `expr` as a function of the range.

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

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

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

       >>> from sympy import symbols, sin, pi
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

    Multiple plots with different ranges and custom labels.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot((x**2, (x, -6, 6), "$f_{1}$"),
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

    Plotting a summation in which the free symbols of the expression is not
    used in the lower/upper bounds:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(Sum(1 / x ** y, (x, 1, oo)), (y, 2, 10), sum_bound=1e03)
       Plot object containing:
       [0]: cartesian line: Sum(x**(-y), (x, 1, 1000)) for y over (2.0, 10.0)

    Plotting a summation in which the free symbols of the expression is
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


    References
    ==========

    .. [#fn1] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot_polar, plot_parametric, plot_contour, plot3d, plot3d_parametric_line,
    plot3d_parametric_surface, plot_implicit, plot_geometry, plot_piecewise

    """
    from spb.defaults import TWO_D_B

    args = _plot_sympify(args)
    free = set()
    for a in args:
        if isinstance(a, Expr):
            free |= a.free_symbols
            if len(free) > 1:
                raise ValueError(
                    "The same variable should be used in all "
                    "univariate expressions being plotted."
                )
    x = free.pop() if free else Symbol("x")
    kwargs.setdefault("backend", TWO_D_B)
    kwargs.setdefault("xlabel", x.name)
    kwargs.setdefault("ylabel", "f(%s)" % x.name)
    kwargs = _set_discretization_points(kwargs, LineOver1DRangeSeries)
    series = []
    plot_expr = _check_arguments(args, 1, 1)
    series = _build_line_series(*plot_expr, **kwargs)

    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot_parametric(*args, show=True, **kwargs):
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
            The expression representing $x$ component of the parametric
            function.

        `expr_y` : Expr
            The expression representing $y$ component of the parametric
            function.

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

    line_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's function to customize the appearance of the lines. Refer to the
        plotting library (backend) manual for more informations.

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

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

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

    A parametric plot with a single expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, -5, 5))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)

    A parametric plot of a Hypotrochoid using an equal aspect ratio:

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

    A parametric plot with multiple expressions with the same range:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, cos(u)), (u, -3, 3))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-3.0, 3.0)
       [1]: parametric cartesian line: (u, cos(u)) for u over (-3.0, 3.0)

    A parametric plot with multiple expressions with different ranges and
    custom labels for each curve:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric(
       ...      (3 * cos(u), 3 * sin(u), (u, 0, 2 * pi), "u"),
       ...      (3 * cos(2 * v), 5 * sin(4 * v), (v, 0, pi), "v"),
       ...      aspect="equal")
       Plot object containing:
       [0]: parametric cartesian line: (3*cos(u), 3*sin(u)) for u over (0.0, 6.283185307179586)
       [1]: parametric cartesian line: (3*cos(2*u), 5*sin(4*u)) for u over (0.0, 3.141592653589793)

    References
    ==========

    .. [#fn2] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot, plot_polar, plot_contour, plot3d, plot3d_parametric_line,
    plot3d_parametric_surface, plot_implicit, plot_geometry, plot_piecewise

    """
    from spb.defaults import TWO_D_B

    args = _plot_sympify(args)
    series = []
    kwargs.setdefault("backend", TWO_D_B)
    kwargs = _set_discretization_points(kwargs, Parametric2DLineSeries)
    plot_expr = _check_arguments(args, 2, 1)
    series = [Parametric2DLineSeries(*arg, **kwargs) for arg in plot_expr]
    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot3d_parametric_line(*args, show=True, **kwargs):
    """
    Plots a 3D parametric line plot.

    Typical usage examples are in the followings:

    - Plotting a single expression.
        `plot3d_parametric_line(expr_x, expr_y, expr_z, range, **kwargs)`
    - Plotting a single expression with a custom label.
        `plot3d_parametric_line(expr_x, expr_y, expr_z, range, label, **kwargs)`
    - Plotting multiple expressions with the same ranges.
        `plot3d_parametric_line((expr_x1, expr_y1, expr_z1), (expr_x2, expr_y2, expr_z2), ..., range, **kwargs)`
    - Plotting multiple expressions with different ranges.
        `plot3d_parametric_line((expr_x1, expr_y1, expr_z1, range1), (expr_x2, expr_y2, expr_z2, range2), ..., **kwargs)`
    - Plotting multiple expressions with different ranges and custom labels.
        `plot3d_parametric_line((expr_x1, expr_y1, expr_z1, range1, label1), (expr_x2, expr_y2, expr_z2, range2, label1), ..., **kwargs)`


    Parameters
    ==========

    args :
        expr_x : Expr
            Expression representing the function along x.

        expr_y : Expr
            Expression representing the function along y.

        expr_z : Expr
            Expression representing the function along z.

        range : (symbol, min, max)
            A 3-tuple denoting the range of the parameter variable.

        label : str, optional
            An optional string denoting the label of the expression
            to be visualized on the legend. If not provided, the string
            representation of the expression will be used.

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

    line_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's function to customize the appearance of the lines. Refer to the
        plotting library (backend) manual for more informations.

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

       >>> from sympy import symbols, cos, sin
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


    Multiple plots with different ranges and custom labels.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line((cos(u), sin(u), u, (u, -5, 5), "u"),
       ...     (sin(v), v**2, v, (v, -3, 3), "v"), legend=True)
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)
       [1]: 3D parametric cartesian line: (sin(u), u**2, u) for u over (-3.0, 3.0)

    References
    ==========

    .. [#fn3] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot, plot_polar, plot3d, plot_contour, plot3d_parametric_surface,
    plot_implicit, plot_geometry, plot_parametric, plot_piecewise

    """
    from spb.defaults import THREE_D_B

    args = _plot_sympify(args)
    kwargs.setdefault("backend", THREE_D_B)
    kwargs = _set_discretization_points(kwargs, Parametric3DLineSeries)
    series = []
    plot_expr = _check_arguments(args, 3, 1)
    series = [Parametric3DLineSeries(*arg, **kwargs) for arg in plot_expr]
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")
    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot3d(*args, show=True, **kwargs):
    """
    Plots a 3D surface plot.

    Typical usage examples are in the followings:

    - Plotting a single expression.
        `plot3d(expr, range_x, range_y, **kwargs)`
    - Plotting multiple expressions with the same ranges.
        `plot3d(expr1, expr2, range_x, range_y, **kwargs)`
    - Plotting multiple expressions with different ranges.
        `plot3d((expr1, range_x1, range_y1), (expr2, range_x2, range_y2), ..., **kwargs)`
    - Plotting multiple expressions with different ranges and custom labels.
        `plot3d((expr1, range_x1, range_y1, label1), (expr2, range_x2, range_y2, label2), ..., **kwargs)`

    Note that it is important to specify at least the `range_x`, otherwise the
    function might create a rotated plot.

    Parameters
    ==========

    args :
        expr : Expr
            Expression representing the function of two variables to be plotted.

        range_x: (symbol, min, max)
            A 3-tuple denoting the range of the x variable. Default values:
            `min=-10` and `max=10`.

        range_y: (symbol, min, max)
            A 3-tuple denoting the range of the y variable. Default values:
            `min=-10` and `max=10`.

        label : str, optional
            The label to be shown in the legend.  If not provided, the string
            representation of `expr` will be used.

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

    n : int, optional
        The x and y ranges are sampled uniformly at `n` of points.
        It overrides `n1` and `n2`.

    show : bool, optional
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    surface_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's function to customize the appearance of surfaces. Refer to the
        plotting library (backend) manual for more informations.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

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

       >>> from sympy import symbols, cos
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


    Multiple plots with same range

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(x*y, -x*y, (x, -5, 5), (y, -5, 5))
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

    References
    ==========

    .. [#fn4] https://github.com/python-adaptive/adaptive

    See Also
    ========

    plot, plot_polar, plot_contour, plot_parametric, plot3d_parametric_line,
    plot3d_parametric_surface, plot_implicit, plot_geometry, plot_piecewise

    """
    from spb.defaults import THREE_D_B

    args = _plot_sympify(args)
    kwargs.setdefault("backend", THREE_D_B)
    kwargs = _set_discretization_points(kwargs, SurfaceOver2DRangeSeries)
    series = []
    plot_expr = _check_arguments(args, 1, 2)
    series = [SurfaceOver2DRangeSeries(*arg, **kwargs) for arg in plot_expr]
    xlabel = series[0].var_x.name
    ylabel = series[0].var_y.name
    kwargs.setdefault("xlabel", xlabel)
    kwargs.setdefault("ylabel", ylabel)
    kwargs.setdefault("zlabel", "f(%s, %s)" % (xlabel, ylabel))
    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot3d_parametric_surface(*args, show=True, **kwargs):
    """
    Plots a 3D parametric surface plot.

    Typical usage examples are in the followings:

    - Plotting a single expression.
        `plot3d_parametric_surface(expr_x, expr_y, expr_z, range_u, range_v, label, **kwargs)`
    - Plotting multiple expressions with the same ranges.
        `plot3d_parametric_surface((expr_x1, expr_y1, expr_z1), (expr_x2, expr_y2, expr_z2), range_u, range_v, **kwargs)`
    - Plotting multiple expressions with different ranges.
        `plot3d_parametric_surface((expr_x1, expr_y1, expr_z1, range_u1, range_v1), (expr_x2, expr_y2, expr_z2, range_u2, range_v2), **kwargs)`
    - Plotting multiple expressions with different ranges and custom labels.
        `plot3d_parametric_surface((expr_x1, expr_y1, expr_z1, range_u1, range_v1, label1), (expr_x2, expr_y2, expr_z2, range_u2, range_v2, label2), **kwargs)`

    Note that it is important to specify both the ranges.

    Parameters
    ==========

    args :
        expr_x: Expr
            Expression representing the function along `x`.

        expr_y: Expr
            Expression representing the function along `y`.

        expr_z: Expr
            Expression representing the function along `z`.

        range_u: (symbol, min, max)
            A 3-tuple denoting the range of the `u` variable.

        range_v: (symbol, min, max)
            A 3-tuple denoting the range of the `v` variable.

        label : str, optional
            The label to be shown in the legend.  If not provided, the string
            representation of the expression will be used.

    backend : Plot, optional
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    n1 : int, optional
        The u range is sampled uniformly at `n1` of points. Default value
        is 100.

    n2 : int, optional
        The v range is sampled uniformly at `n2` of points. Default value
        is 100.

    n : int, optional
        The u and v ranges are sampled uniformly at `n` of points.
        It overrides `n1` and `n2`.

    show : bool, optional
        The default value is set to `True`. Set show to `False` and
        the function will not display the plot. The returned instance of
        the `Plot` class can then be used to save or display the plot
        by calling the `save()` and `show()` methods respectively.

    size : (float, float), optional
        A tuple in the form (width, height) to specify the size of
        the overall figure. The default value is set to `None`, meaning
        the size will be set by the backend.

    surface_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's function to customize the appearance of surfaces. Refer to the
        plotting library (backend) manual for more informations.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

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

    Single plot with u/v directions discretized with 200 points and a custom
    label.

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
       >>> plot3d_parametric_surface(*expr, (u, 0, 2 * pi), (v, 0, pi), "f",
       ...      n=200)
       Plot object containing:
       [0]: parametric cartesian surface: ((sin(7*u + 5*v) + 2)*sin(v)*cos(u), (sin(7*u + 5*v) + 2)*sin(u)*sin(v), (sin(7*u + 5*v) + 2)*cos(v)) for u over (0.0, 6.283185307179586) and v over (0.0, 3.141592653589793)


    See Also
    ========

    plot, plot_polar, plot_parametric, plot3d, plot_contour,
    plot3d_parametric_line, plot_implicit, plot_geometry, plot_piecewise

    """
    from spb.defaults import THREE_D_B

    args = _plot_sympify(args)
    kwargs.setdefault("backend", THREE_D_B)
    kwargs = _set_discretization_points(kwargs, ParametricSurfaceSeries)
    plot_expr = _check_arguments(args, 3, 2)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")
    series = [ParametricSurfaceSeries(*arg, **kwargs) for arg in plot_expr]
    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot_contour(*args, show=True, **kwargs):
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

    contour_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's function to customize the appearance of contours. Refer to the
        plotting library (backend) manual for more informations.

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


    See Also
    ========

    plot, plot_polar, plot_parametric, plot3d, plot3d_parametric_line,
    plot3d_parametric_surface, plot_implicit, plot_geometry, plot_piecewise

    """
    from spb.defaults import TWO_D_B

    args = _plot_sympify(args)
    kwargs.setdefault("backend", TWO_D_B)
    kwargs = _set_discretization_points(kwargs, ContourSeries)
    plot_expr = _check_arguments(args, 1, 2)
    series = [ContourSeries(*arg, **kwargs) for arg in plot_expr]
    xlabel = series[0].var_x.name
    ylabel = series[0].var_y.name
    kwargs.setdefault("xlabel", xlabel)
    kwargs.setdefault("ylabel", ylabel)
    plot_contours = Plot(*series, **kwargs)
    if show:
        plot_contours.show()
    return plot_contours


def plot_implicit(*args, show=True, **kwargs):
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

        ranges : tuples
            Two tuple denoting the discretization domain, for example:
            `(x, -10, 10), (y, -10, 10)`
            To get a correct plot, at least the horizontal range must be
            provided. If no range is given, then the free symbols in the
            expression will be assigned in the order they are sorted, which
            could 'invert' the axis.

        label : str, optional
            The name of the expression to be eventually shown on the legend.
            If not provided, the string representation of `expr` will be used.

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

    n : integer
        Set the number of discretization points when `adaptive=False` in
        both direction simultaneously. Default value is 1000.
        The greater the value the more accurate the plot, but the more
        memory will be used.

    show : Boolean
        Default value is True. If set to False, the plot will not be shown.
        See `Plot` for further information.

    title : string
        The title for the plot.

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

    Without any ranges for the symbols in the expression:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p1 = plot_implicit(Eq(x**2 + y**2, 5))

    With the range for the symbols:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p2 = plot_implicit(
        ...     Eq(x**2 + y**2, 3), (x, -3, 3), (y, -3, 3))

    Using mesh grid without adaptive meshing with number of points
    specified:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p3 = plot_implicit(
        ...     (x**2 + y**2 - 1)**3 - x**2 * y**3,
        ...     (x, -1.5, 1.5), (y, -1.5, 1.5),
        ...     n = 1000)

    Using adaptive meshing and Boolean expressions:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p4 = plot_implicit(
        ...     Eq(y, sin(x)) & (y > 0),
        ...     Eq(y, sin(x)) & (y < 0),
        ...     (x, -2 * pi, 2 * pi), (y, -4, 4),
        ...     adaptive=True)

    Using adaptive meshing with depth of recursion as argument:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p5 = plot_implicit(
        ...     Eq(x**2 + y**2, 5), (x, -4, 4), (y, -4, 4),
        ...     adaptive=True, depth = 2)

    Plotting regions:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p6 = plot_implicit(y > x**2, (x, -5, 5))

    See Also
    ========

    plot, plot_polar, plot3d, plot_contour, plot3d_parametric_line,
    plot3d_parametric_surface, plot_geometry

    """
    from spb.defaults import TWO_D_B

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
    xmin, xmax, ymin, ymax = oo, -oo, oo, -oo
    for a in args:
        s = ImplicitSeries(*a, **series_kw)
        if s.start_x < xmin:
            xmin = s.start_x
        if s.end_x > xmax:
            xmax = s.end_x
        if s.start_y < ymin:
            ymin = s.start_y
        if s.end_y > ymax:
            ymax = s.end_y
        series.append(s)

    kwargs.setdefault("backend", TWO_D_B)
    kwargs.setdefault("xlim", (xmin, xmax))
    kwargs.setdefault("ylim", (ymin, ymax))
    kwargs.setdefault("xlabel", series[-1].var_x.name)
    kwargs.setdefault("ylabel", series[-1].var_y.name)
    p = Plot(*series, **kwargs)
    if show:
        p.show()
    return p


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

        >>> plot_polar(1 + sin(10 * x) / 10, (x, 0, 2 * pi),
        ...     polar=True, aspect="equal")

    See Also
    ========

    plot, plot_parametric, plot3d, plot_contour, plot3d_parametric_line,
    plot3d_parametric_surface, plot_implicit, plot_geometry, plot_piecewise

    """
    kwargs["polar"] = True
    kwargs.setdefault("axis", "equal")
    return plot(*args, **kwargs)


def plot_geometry(*args, show=True, **kwargs):
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

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    backend : Plot, optional
        A subclass of `Plot`, which will perform the rendering.
        Default to `MatplotlibBackend`.

    is_filled : boolean
        Default to True. Fill the polygon/circle/ellipse.

    params : dict
        Substitution dictionary to properly evaluate symbolic geometric
        entities. The keys represents symbols, the values represents the
        numeric number associated to the symbol.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}. Only available with MatplotlibBackend.

    fill_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's fill
        function to customize the appearance of fills. Refer to the
        plotting library (backend) manual for more informations.

    line_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.

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
    """
    from spb.defaults import TWO_D_B, THREE_D_B

    args = _plot_sympify(args)

    series = []
    if not all([isinstance(a, (list, tuple, Tuple)) for a in args]):
        args = [args]

    for a in args:
        exprs, ranges, label = _unpack_args(*a)
        r = ranges if len(ranges) > 0 else [None]
        if len(exprs) == 1:
            series.append(GeometrySeries(exprs[0], *r, label, **kwargs))
        else:
            # this is the case where the user provided: v1, v2, ..., range
            # we use the same ranges for each expression
            for e in exprs:
                series.append(GeometrySeries(e, *r, str(e), **kwargs))

    any_3D = any(s.is_3D for s in series)
    if ("aspect" not in kwargs) and (not any_3D):
        kwargs["aspect"] = "equal"

    if any_3D:
        kwargs.setdefault("backend", THREE_D_B)
    else:
        kwargs.setdefault("backend", TWO_D_B)

    p = Plot(*series, **kwargs)
    if show:
        p.show()
    return p


def plot_list(*args, show=True, **kwargs):
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

    line_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.

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

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

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
    from spb.defaults import TWO_D_B

    series = []

    if (
        ((len(args) == 2) and (not hasattr(args[0][1], "__iter__"))) or
        ((len(args) == 3) and isinstance(args[-1], str))
    ):
        series.append(List2DSeries(*args, **kwargs))
    else:
        for a in args:
            if not isinstance(a, (list, tuple)):
                raise TypeError(
                    "Each argument must be a list or tuple.\n"
                    "Received type(a) = {}".format(type(a)))
            if (len(a) < 2) or (len(a) > 3):
                raise ValueError(
                    "Each argument must contain 2 or 3 elements.\n"
                    "Received {} elements.".format(len(a)))
            if (len(a) == 3) and (not isinstance(a[-1], str)):
                raise TypeError(
                    "The label must be of type string.\n"
                    "Received: {}".format(type(a[-1]))
                )
            series.append(List2DSeries(*a, **kwargs))

    kwargs.setdefault("backend", TWO_D_B)
    p = Plot(*series, **kwargs)
    if show:
        p.show()
    return p


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
        `plot_piecewise((expr1, range1, label1), (expr2, range2, label2), ..., legend=True, **kwargs)`


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

    line_kw : dict, optional
        A dictionary of keywords/values which is passed to the backend's function to customize the appearance of lines. Refer to the
        plotting library (backend) manual for more informations.

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

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

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

       >>> from sympy import symbols, cos, pi, Heaviside, Piecewise
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

    Multiple plots with different ranges and custom labels.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_piecewise(
       ...   (Heaviside(x, 0).rewrite(Piecewise), (x, -10, 10)),
       ...   (Piecewise(
       ...      (sin(x), x < -5),
       ...      (cos(x), x > 5),
       ...      (1 / x, True)), (x, -8, 8)),
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
    plot_implicit, plot_geometry, plot_list

    """
    from spb.defaults import TWO_D_B
    Backend = kwargs.pop("backend", TWO_D_B)

    args = _plot_sympify(args)
    free = set()
    for a in args:
        if isinstance(a, Expr):
            free |= a.free_symbols
            if len(free) > 1:
                raise ValueError(
                    "The same variable should be used in all "
                    "univariate expressions being plotted."
                )

    show = kwargs.get("show", True)
    kwargs["show"] = False

    x = free.pop() if free else Symbol("x")
    kwargs.setdefault("xlabel", x.name)
    kwargs.setdefault("ylabel", "f(%s)" % x.name)
    kwargs.setdefault("legend", False)
    kwargs["process_piecewise"] = True
    kwargs = _set_discretization_points(kwargs, LineOver1DRangeSeries)
    series = []

    plots = []
    plot_expr = _check_arguments(args, 1, 1)
    color_series_dict = dict()
    for i, a in enumerate(plot_expr):
        series = _build_line_series(a, **kwargs)
        color_series_dict[i] = series

    # NOTE: let's overwrite this keyword argument
    kwargs["process_piecewise"] = color_series_dict
    p = Backend(**kwargs)

    if show:
        p.show()
    return p
