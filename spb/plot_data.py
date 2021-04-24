from sympy import Tuple
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.plotting.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries,
    Parametric3DLineSeries, SurfaceOver2DRangeSeries,
    ParametricSurfaceSeries, _set_discretization_points
)
from sympy.plotting.plot_implicit import ImplicitSeries
from sympy.plotting.plot import (
    Plot, _plot_sympify, _is_range, _check_arguments    
)

def _build_series(*args, **kwargs):
    # In the following dictionary the key is composed of two characters:
    # 1. The first represents the number of sub-expressions. For example,
    #    a line plot or a surface plot have 1 expression. A 2D parametric line
    #    does have 2 parameters, ...
    # 2. The second represent the number of parameters.
    # These categorization doesn't work for ImplicitSeries, in which case a
    # random number was assigned.
    mapping = {
        "99": ImplicitSeries,
        "11": LineOver1DRangeSeries,
        "21": Parametric2DLineSeries,
        "31": Parametric3DLineSeries,
        "32": ParametricSurfaceSeries,
        "12": SurfaceOver2DRangeSeries,
    }

    # TODO: this is an hack because _set_discretization_points is defined into
    # plot.py. I would really like to send in the series classes to that method,
    # however I can't because of circular inport with plot_implicit.py
    # Need to separate plot.py code into several modules.
    cls_to_pt = {
        ImplicitSeries: "pt",
        LineOver1DRangeSeries: "p",
        Parametric2DLineSeries: "pp",
        Parametric3DLineSeries: "p3dpl",
        SurfaceOver2DRangeSeries: "p3d",
        ParametricSurfaceSeries: "p3dps",
    }

    args = _plot_sympify(args)
    # select the expressions
    res = [not (_is_range(a) or isinstance(a, str)) for a in args]
    exprs = [a for a, b in zip(args, res) if b]

    skip_check = False
    if isinstance(exprs[0], (Boolean, Relational)):
        skip_check = True
        npar = 9
        nexpr = 9
    if isinstance(exprs[0], (list, tuple, Tuple)):
        fs = set().union(*[e.free_symbols for e in exprs[0]])
        npar = len(fs)
        nexpr = len(exprs[0])
    else:
        fs = set().union(*[e.free_symbols for e in exprs])
        npar = len(fs)
        nexpr = len(exprs)
    
    if not skip_check:
        args = _check_arguments(args, nexpr, npar)[0]

    k = str(nexpr) + str(npar)
    if k not in mapping.keys():
        raise ValueError(
            "Don't know how to plot your expression:\n" +
            "Received: {}\n".format(args) +
            "Number of subexpressions: {}\n".format(nexpr) +
            "Number of parameters: {}".format(npar)
        )
    _cls = mapping[k]
    kwargs = _set_discretization_points(kwargs, cls_to_pt[_cls])
    s = _cls(*args, **kwargs)
    return s

def get_plot_data(*args, **kwargs):
    """ Return the numerical data associated to the a symbolic expression that
    we would like to plot. If a symbolic expression can be plotted with any of
    the plotting functions exposed by sympy.plotting.plot or 
    sympy.plotting.plot_implicit, then numerical data will be returned.

    Only one expression can be processed by this function. The shape of the
    numerical data depends on the symbolic expression. Arguments and keywords
    have the same format used in other plotting functions.

    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, pi
       >>> from sympy.plotting.plot_data import get_plot_data
       >>> u, v, x, y = symbols('u, v, x, y')
       
    Data from a function with a single variable (2D line):

    .. get_plot_data::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> xx, yy = get_plot_data(cos(x), (x, -5, 5))
    
    Here, `xx` and `yy` are two lists of coordinates.

    Data from a function with two variables (surface):

    .. get_plot_data::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> xx, yy, zz = get_plot_data(cos(x * y), (x, -5, 5), (y, -10, 10))

    Here, `xx, yy, zz` are two-dimensional numpy arrays. `xx, yy` represent the
    mesh grid.

    Data from a 2D parametric function with one variable (2D line):

    .. get_plot_data::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> xx, yy = get_plot_data(cos(u), sin(u), (u, 0, 2 * pi))

    Here, `xx` and `yy` are two lists of coordinates.

    Data from a 3D parametric function with one variables (3D line):

    .. get_plot_data::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> xx, yy, zz = get_plot_data(cos(u), sin(u), u, (u, -5, 5))

    Here, `xx, yy, zz` are three lists of coordinates.

    Data from an implicit relation:

    .. get_plot_data::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> data = get_plot_data(x > y)

    Here, `data` depends on the specific case. Its shape could be:
    `data = ((xx, yy), 'fill')` where `xx, yy` are two lists of coordinates.
    `data = (xx, yy, zz, 'contour') where `xx, yy, zz` are two-dimensional numpy
        arrays. `xx, yy` represent the mesh grid. This is returned by objects of
        type Equality.
    `data = (xx, yy, zz, 'contourf') where `xx, yy, zz` are two-dimensional numpy
        arrays. `xx, yy` represent the mesh grid. This is returned by objects of
        type non-equalities (greater than, less than, ...).

    See also
    ========

    plot, plot_parametric, plot3d, plot3d_parametric_line,
    plot3d_parametric_surface, plot_contour, plot_implicit
    """
    return _build_series(*args, **kwargs).get_data()

def smart_plot(*args, **kwargs):
    """ Smart plot interface. Using the same interface of the other plot
    functions, namely (expr, range, label), it unifies the plotting experience.
    With this function, we can create the most common plots: line plots,
    parametric plots, 3d surface plots. Contour plots are not supported.

    Examples
    ========

    Plotting multiple expressions:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> smart_plot((cos(u), sin(u), (u, -5, 5), "a"),
       ...     (cos(u), u, "b"))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)
       [1]: parametric cartesian line: (cos(u), u) for u over (-10.0, 10.0)

    """

    args = _plot_sympify(args)
    if not all([isinstance(a, (list, tuple, Tuple)) for a in args]):
        args = [args]
    series = []
    is_2D, is_3D, is_contour = False, False, False

    for arg in args:
        series.append(_build_series(*arg, **kwargs))

    plots = Plot(*series, **kwargs)
    plots.show()
    return plots