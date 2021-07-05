"""Plotting module for Sympy.

A plot is represented by the ``Plot`` class that contains a reference to the
backend and a list of the data series to be plotted. The data series are
instances of classes meant to simplify getting points and meshes from sympy
expressions. ``plot_backends`` is a dictionary with all the backends.

This module gives only the essential. For all the fancy stuff use directly
the backend. You can get the backend wrapper for every plot from the
``_backend`` attribute. Moreover the data series classes have various useful
methods like ``get_points``, ``get_meshes``, etc, that may
be useful if you wish to use another plotting library.

Especially if you need publication ready graphs and this module is not enough
for you - just get the ``_backend`` attribute and add whatever you want
directly to it. In the case of matplotlib (the common way to graph data in
python) just copy ``_backend.fig`` which is the figure and ``_backend.ax``
which is the axis and work on them as you would on any other matplotlib object.

Simplicity of code takes much greater importance than performance. Don't use it
if you care at all about performance. A new backend instance is initialized
every time you call ``show()`` and the old one is left to the garbage collector.
"""


from sympy import (
    Expr, Tuple, Symbol, oo, Piecewise, piecewise_fold,
    UniversalSet, EmptySet, UniversalSet, FiniteSet, Interval, Union
)

from spb.backends.base_backend import Plot
from spb.utils import _is_range, _plot_sympify, _check_arguments

# N.B.
# When changing the minimum module version for matplotlib, please change
# the same in the `SymPyDocTestFinder`` in `sympy/testing/runtests.py`

from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries,
    Parametric3DLineSeries, SurfaceOver2DRangeSeries,
    ParametricSurfaceSeries, ContourSeries, ImplicitSeries,
    _set_discretization_points, List2DSeries
)

def _get_endpoints(i, _min, _max):
    """ Given the end points of a local range, compute the 
    appropriate end points of the interval in such a way that the interval is
    contained in the local range.

    Returns
    =======
        a, b : float
            The appropriate end points
        skip : boolean
            Indicate whether to add the series or to skip it. Series whose
            domain doesn't overlap with the provided local range [_min, _max]
            will be discarded.
    """
    a, b, lopen, ropen = i.args
    
    skip = False
    if (a >= _max) or (b <= _min):
        skip = True
        
    if not skip:
        if a < _min:
            a = _min
        if b > _max:
            b = _max
    
        eps = (b - a) / 1e06
        if lopen:
            a += eps
        if ropen:
            b -= eps
    return a, b, skip

def _process_piecewise(piecewise, _range, label, **kwargs):
    s = EmptySet
    series = []
    _min, _max = _range[1], _range[2]

    # for the label, attach the number of the piece
    count = 1
    if "Piecewise(" in label:
        # piecewise string representation are usually very long. Cut it short if
        # the user didn't specify any custom label.
        label = "P"

    for arg in piecewise.args:
        expr, cond = arg.args
        cond = cond.as_set()
        if isinstance(cond, Interval):
            s = s.union(cond)
            a, b, skip = _get_endpoints(cond, _min, _max)
            if not skip:
                series.append(LineOver1DRangeSeries(expr, (_range[0], a, b), 
                        label + str(count)))
                count += 1
        elif isinstance(cond, FiniteSet):
            s = s.union(cond)
            loc, val = [], []
            for _loc in cond.args:
                loc.append(float(_loc))
                val.append(float(piecewise.evalf(subs={_range[0]: _loc})))
            series.append(List2DSeries(loc, val, label + str(count), 
                    is_point=True))
            count += 1
        elif isinstance(cond, UniversalSet.func):
            # at this point the condition should be UniversalSet (or True)
            # meaning the complementary part of the domain.
            # NOTE: there should not be any more arg after this one...
            s = s.complement(Interval(_min, _max, True, True))
            if isinstance(s, Union):
                s1 = [t for t in s.args if isinstance(t, Interval)]
                s2 = [t for t in s.args if isinstance(t, FiniteSet)]
            else:
                s1, s2 = [s], []

            for t in s1:
                a, b, skip = _get_endpoints(t, _min, _max)
                if not skip:
                    series.append(LineOver1DRangeSeries(expr, 
                            (_range[0], a, b), label + str(count)))
                    count += 1
            for t in s2:
                loc, val = [], []
                for _loc in t.args:
                    loc.append(float(_loc))
                    val.append(float(expr.evalf(subs={_range[0]: _loc})))
                series.append(List2DSeries(loc, val, label + str(count), 
                        is_point=True))
                count += 1
    return series

def _build_line_series(*args, **kwargs):
    """ Loop over the provided arguments. If a piecewise function is found,
    decompose it in such a way that each argument gets its own series.
    """
    series = []
    for arg in args:
        expr, r, label = arg
        if expr.has(Piecewise):
            expr = piecewise_fold(expr)
            series += _process_piecewise(expr, r, label, **kwargs)
        else:
            series.append(LineOver1DRangeSeries(*arg, **kwargs))
    return series

def plot(*args, show=True, **kwargs):
    """Plots a function of a single variable as a curve.

    Parameters
    ==========

    args :
        The first argument is the expression representing the function
        of single variable to be plotted.

        The next argument is a 3-tuple denoting the range of the free
        variable. e.g. ``(x, 0, 5)``

        The last optional argument is a string denoting the label of the
        expression to be visualized on the legend. If not provided, the label
        will be the string representation of the expression.

        Typical usage examples are in the followings:

        - Plotting a single expression with a single range.
            ``plot(expr, range, **kwargs)``
        - Plotting a single expression with the default range (-10, 10).
            ``plot(expr, **kwargs)``
        - Plotting multiple expressions with a single range.
            ``plot(expr1, expr2, ..., range, **kwargs)``
        - Plotting multiple expressions with multiple ranges.
            ``plot((expr1, range1), (expr2, range2), ..., **kwargs)``
        - Plotting multiple expressions with multiple ranges and custom labels.
            ``plot((expr1, range1, label1), (expr2, range2, label2), ..., legend=True, **kwargs)``

        It is best practice to specify range explicitly because default
        range may change in the future if a more advanced default range
        detection algorithm is implemented.

    show : bool, optional
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

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

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.

    annotations : list, optional
        A list of dictionaries specifying the type of annotation
        required. The keys in the dictionary should be equivalent
        to the arguments of the matplotlib's annotate() function.

    markers : list, optional
        A list of dictionaries specifying the type the markers required.
        The keys in the dictionary should be equivalent to the arguments
        of the matplotlib's plot() function along with the marker
        related keyworded arguments.

    rectangles : list, optional
        A list of dictionaries specifying the dimensions of the
        rectangles to be plotted. The keys in the dictionary should be
        equivalent to the arguments of the matplotlib's
        patches.Rectangle class.

    fill : dict, optional
        A dictionary specifying the type of color filling required in
        the plot. The keys in the dictionary should be equivalent to the
        arguments of the matplotlib's fill_between() function.

    adaptive : bool, optional
        The default value is set to ``True``. Set adaptive to ``False``
        and specify ``n`` if uniform sampling is required.

        The plotting uses an adaptive algorithm which samples
        recursively to accurately plot. The adaptive algorithm uses a
        random point near the midpoint of two points that has to be
        further sampled. Hence the same plots can appear slightly
        different.

    depth : int, optional
        Recursion depth of the adaptive algorithm. A depth of value
        ``n`` samples a maximum of `2^{n}` points.

        If the ``adaptive`` flag is set to ``False``, this will be
        ignored.
    
    detect_poles : boolean
            Chose whether to detect and correctly plot poles. Defaulto to False.
            This improve detection, increase the number of discretization points
            and/or change the value of `eps`.
        
    eps : float
        An arbitrary small value used by the `detect_poles` algorithm.
        Default value to 0.1. Before changing this value, it is better to
        increase the number of discretization points.

    n : int, optional
        Used when the ``adaptive`` is set to ``False``. The function
        is uniformly sampled at ``n`` number of points.

        If the ``adaptive`` flag is set to ``True``, this will be
        ignored.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import symbols
       >>> from sympy.plotting import plot
       >>> x = symbols('x')

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

       >>> plot((x**2, (x, -6, 6), "$f_{1}$"), (x, (x, -5, 5), "f2"), legend=True)
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

    See Also
    ========

    Plot, LineOver1DRangeSeries

    """
    from spb.defaults import TWO_D_B
    args = _plot_sympify(args)
    free = set()
    for a in args:
        if isinstance(a, Expr):
            free |= a.free_symbols
            if len(free) > 1:
                raise ValueError(
                    'The same variable should be used in all '
                    'univariate expressions being plotted.')
    x = free.pop() if free else Symbol('x')
    kwargs.setdefault('backend', TWO_D_B)
    kwargs.setdefault('xlabel', x.name)
    kwargs.setdefault('ylabel', 'f(%s)' % x.name)
    kwargs = _set_discretization_points(kwargs, LineOver1DRangeSeries)
    series = []
    plot_expr = _check_arguments(args, 1, 1)
    # series = [LineOver1DRangeSeries(*arg, **kwargs) for arg in plot_expr]
    series = _build_line_series(*plot_expr, **kwargs)

    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot_parametric(*args, show=True, **kwargs):
    """
    Plots a 2D parametric curve.

    Parameters
    ==========

    args
        Common specifications are:

        - Plotting a single parametric curve with a range
            ``plot_parametric(expr_x, expr_y, range)``
        - Plotting multiple parametric curves with the same range
            ``plot_parametric((expr_x, expr_y), ..., range)``
        - Plotting multiple parametric curves with different ranges
            ``plot_parametric((expr_x, expr_y, range), ...)``
        - Plotting multiple parametric curves with different ranges and
            custom labels
            ``plot_parametric((expr_x, expr_y, range, label), ...)``

        ``expr_x`` is the expression representing $x$ component of the
        parametric function.

        ``expr_y`` is the expression representing $y$ component of the
        parametric function.

        ``range`` is a 3-tuple denoting the parameter symbol, start and
        stop. For example, ``(u, 0, 5)``. If the range is not specified, then
        a default range of (-10, 10) is used.

        However, if the arguments are specified as
        ``(expr_x, expr_y, range), ...``, you must specify the ranges
        for each expressions manually.

        Default range may change in the future if a more advanced
        algorithm is implemented.

        ``label`` : An optional string denoting the label of the expression
        to be visualized on the legend. If not provided, the label will be the
        string representation of the expression.

    adaptive : bool, optional
        Specifies whether to use the adaptive sampling or not.

        The default value is set to ``True``. Set adaptive to ``False``
        and specify ``n`` if uniform sampling is required.

    depth :  int, optional
        The recursion depth of the adaptive algorithm. A depth of
        value $n$ samples a maximum of $2^n$ points.

    n : int, optional
        Used when the ``adaptive`` flag is set to ``False``.

        Specifies the number of the points used for the uniform
        sampling.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot_parametric
       >>> u = symbols('u')

    A parametric plot with a single expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, -5, 5))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)

    A parametric plot with multiple expressions with the same range:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, cos(u)), (u, -10, 10))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-10.0, 10.0)
       [1]: parametric cartesian line: (u, cos(u)) for u over (-10.0, 10.0)

    A parametric plot with multiple expressions with different ranges and
    custom labels for each curve:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u), (u, -5, 5), "a"),
       ...     (cos(u), u, "b"))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)
       [1]: parametric cartesian line: (cos(u), u) for u over (-10.0, 10.0)

    Notes
    =====

    The plotting uses an adaptive algorithm which samples recursively to
    accurately plot the curve. The adaptive algorithm uses a random point
    near the midpoint of two points that has to be further sampled.
    Hence, repeating the same plot command can give slightly different
    results because of the random sampling.

    If there are multiple plots, then the same optional arguments are
    applied to all the plots drawn in the same canvas. If you want to
    set these options separately, you can index the returned ``Plot``
    object and set it.

    For example, when you specify ``line_color`` once, it would be
    applied simultaneously to both series.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> from sympy import pi
        >>> expr1 = (u, cos(2*pi*u)/2 + 1/2)
        >>> expr2 = (u, sin(2*pi*u)/2 + 1/2)
        >>> p = plot_parametric(expr1, expr2, (u, 0, 1), line_color='blue')

    If you want to specify the line color for the specific series, you
    should index each item and apply the property manually.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> p[0].line_color = 'red'
        >>> p.show()

    See Also
    ========

    Plot, Parametric2DLineSeries
    """
    from spb.defaults import TWO_D_B
    args = _plot_sympify(args)
    series = []
    kwargs.setdefault('backend', TWO_D_B)
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

    Usage
    =====

    Single plot:

    ``plot3d_parametric_line(expr_x, expr_y, expr_z, range, label, **kwargs)``

    If the range is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_line((expr_x, expr_y, expr_z, range, label), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    ``expr_x`` : Expression representing the function along x.

    ``expr_y`` : Expression representing the function along y.

    ``expr_z`` : Expression representing the function along z.

    ``range``: ``(u, 0, 5)``, A 3-tuple denoting the range of the parameter
    variable.

    ``label`` : An optional string denoting the label of the expression
        to be visualized on the legend. If not provided, the label will be the
        string representation of the expression.

    Keyword Arguments
    =================

    Arguments for ``Parametric3DLineSeries`` class.

    ``n``: The range is uniformly sampled at ``n`` number of points.

    Aesthetics:

    ``line_color``: string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class.

    ``title`` : str. Title of the plot.

    ``size`` : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_line
       >>> u = symbols('u')

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

       >>> plot3d_parametric_line((cos(u), sin(u), u, (u, -5, 5), "a"),
       ...     (sin(u), u**2, u, (u, -3, 3), "b"), legend=True)
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)
       [1]: 3D parametric cartesian line: (sin(u), u**2, u) for u over (-3.0, 3.0)


    See Also
    ========

    Plot, Parametric3DLineSeries

    """
    from spb.defaults import THREE_D_B
    args = _plot_sympify(args)
    kwargs.setdefault('backend', THREE_D_B)
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

    Usage
    =====

    Single plot

    ``plot3d(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot3d(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot3d((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    ``expr`` : Expression representing the function along x.

    ``range_x``: (x, 0, 5), A 3-tuple denoting the range of the x
    variable.

    ``range_y``: (y, 0, 5), A 3-tuple denoting the range of the y
     variable.

    Keyword Arguments
    =================

    Arguments for ``SurfaceOver2DRangeSeries`` class:

    ``n1``: int. The x range is sampled uniformly at ``n1`` of points.

    ``n2``: int. The y range is sampled uniformly at ``n2`` of points.

    ``n``: int. The x and y ranges are sampled uniformly at ``n`` of points.
    It overrides ``n1`` and ``n2``.

    Aesthetics:

    ``surface_color``: Function which returns a float. Specifies the color for
    the surface of the plot. See ``sympy.plotting.Plot`` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    ``title`` : str. Title of the plot.
    ``size`` : (float, float), optional
    A tuple in the form (width, height) in inches to specify the size of the
    overall figure. The default value is set to ``None``, meaning the size will
    be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols
       >>> from sympy.plotting import plot3d
       >>> x, y = symbols('x y')

    Single plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(x*y, (x, -5, 5), (y, -5, 5))
       Plot object containing:
       [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)


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


    See Also
    ========

    Plot, SurfaceOver2DRangeSeries

    """
    from spb.defaults import THREE_D_B
    args = _plot_sympify(args)
    kwargs.setdefault('backend', THREE_D_B)
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

    Explanation
    ===========

    Single plot.

    ``plot3d_parametric_surface(expr_x, expr_y, expr_z, range_u, range_v, **kwargs)``

    If the ranges is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_surface((expr_x, expr_y, expr_z, range_u, range_v), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    ``expr_x``: Expression representing the function along ``x``.

    ``expr_y``: Expression representing the function along ``y``.

    ``expr_z``: Expression representing the function along ``z``.

    ``range_u``: ``(u, 0, 5)``,  A 3-tuple denoting the range of the ``u``
    variable.

    ``range_v``: ``(v, 0, 5)``,  A 3-tuple denoting the range of the v
    variable.

    Keyword Arguments
    =================

    Arguments for ``ParametricSurfaceSeries`` class:

    ``n1``: int. The ``u`` range is sampled uniformly at ``n1`` of points

    ``n2``: int. The ``v`` range is sampled uniformly at ``n2`` of points

    ``n``: int. The u and v ranges are sampled uniformly at ``n`` of points.
    It overrides ``n1`` and ``n2``.

    Aesthetics:

    ``surface_color``: Function which returns a float. Specifies the color for
    the surface of the plot. See ``sympy.plotting.Plot`` for more details.

    If there are multiple plots, then the same series arguments are applied for
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.


    Arguments for ``Plot`` class:

    ``title`` : str. Title of the plot.
    ``size`` : (float, float), optional
    A tuple in the form (width, height) in inches to specify the size of the
    overall figure. The default value is set to ``None``, meaning the size will
    be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_surface
       >>> u, v = symbols('u v')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_surface(cos(u + v), sin(u - v), u - v,
       ...     (u, -5, 5), (v, -5, 5))
       Plot object containing:
       [0]: parametric cartesian surface: (cos(u + v), sin(u - v), u - v) for u over (-5.0, 5.0) and v over (-5.0, 5.0)


    See Also
    ========

    Plot, ParametricSurfaceSeries

    """
    from spb.defaults import THREE_D_B
    args = _plot_sympify(args)
    kwargs.setdefault('backend', THREE_D_B)
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
    Draws contour plot of a function

    Usage
    =====

    Single plot

    ``plot_contour(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot_contour(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot_contour((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    ``expr`` : Expression representing the function along x.

    ``range_x``: (x, 0, 5), A 3-tuple denoting the range of the x
    variable.

    ``range_y``: (y, 0, 5), A 3-tuple denoting the range of the y
     variable.

    Keyword Arguments
    =================

    Arguments for ``ContourSeries`` class:

    ``n1``: int. The x range is sampled uniformly at ``n1`` of points.

    ``n2``: int. The y range is sampled uniformly at ``n2`` of points.

    ``n``: int. The x and y ranges are sampled uniformly at ``n`` of points.
    It overrides ``n1`` and ``n2``.

    Aesthetics:

    ``surface_color``: Function which returns a float. Specifies the color for
    the surface of the plot. See ``sympy.plotting.Plot`` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    ``title`` : str. Title of the plot.
    ``size`` : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    See Also
    ========

    Plot, ContourSeries

    """
    from spb.defaults import TWO_D_B
    args = _plot_sympify(args)
    kwargs.setdefault('backend', TWO_D_B)
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
    used. By setting `adaptive=True` interval arithmetic will be used to plot
    functions. If the expression cannot be plotted using interval arithmetic, 
    it defaults to generating a contour using a mesh grid. With interval
    arithmetic, the line width can become very small; in those cases, it is
    better to use the mesh grid approach.

    Arguments
    =========
        expr : Expr, Relational, BooleanFunction
            The equation / inequality that is to be plotted.
        
        ranges : tuples
            Two tuple denoting the discretization domain, for example:
            `(x, -10, 10), (y, -10, 10)`
            If no range is given, then the free symbols in the expression will
            be assigned in the order they are sorted.
        
        label : str
            The name of the expression to be eventually shown on the legend.
            If none is provided, the string representation will be used.

    Keyword Arguments
    =================

        adaptive : Boolean
            The default value is set to False, meaning that the internal
            algorithm uses a mesh grid approach. In such case,
            Boolean combinations of expressions cannot be plotted.
            If set to True, the internal algorithm uses interval arithmetic.
            It switches to a fall back algorithm (meshgrid approach) if the
            expression cannot be plotted using interval arithmetic.

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
            See ``Plot`` for further information.

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

        >>> from sympy import plot_implicit, symbols, Eq, And
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

        >>> p6 = plot_implicit(y > x**2)

    """
    from spb.defaults import TWO_D_B
    args = _plot_sympify(args)
    args = _check_arguments(args, 1, 2)
    
    kwargs = _set_discretization_points(kwargs, ImplicitSeries)

    series_kw = dict()
    series_kw["n1"] = kwargs.pop("n1", 1000)
    series_kw["n2"] = kwargs.pop("n2", 1000)
    series_kw["depth"] = kwargs.pop("depth", 0)
    series_kw["adaptive"] = kwargs.pop("adaptive", False)

    series = []
    xmin, xmax, ymin, ymax = oo, -oo, oo, -oo
    for a in args:
        s = ImplicitSeries(*a, **series_kw)
        if s.start_x < xmin: xmin = s.start_x
        if s.end_x > xmax: xmax = s.end_x
        if s.start_y < ymin: ymin = s.start_y
        if s.end_y > ymax: ymax = s.end_y
        series.append(s)

    kwargs.setdefault('backend', TWO_D_B)
    kwargs.setdefault('xlim', (xmin, xmax))
    kwargs.setdefault('ylim', (ymin, ymax))
    kwargs.setdefault('xlabel', series[-1].var_x.name)
    kwargs.setdefault('ylabel', series[-1].var_y.name)
    p = Plot(*series, **kwargs)
    if show:
        p.show()
    return p
