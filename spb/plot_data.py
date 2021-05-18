from sympy import Tuple, S
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.matrices.dense import DenseMatrix
from sympy.vector import Vector
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries,
    Parametric3DLineSeries, SurfaceOver2DRangeSeries,
    ParametricSurfaceSeries, ImplicitSeries, InteractiveSeries,
    _set_discretization_points, Vector2DSeries, Vector3DSeries
)
from spb.plot import (
    Plot, _check_arguments    
)
from spb.utils import _unpack_args, _plot_sympify
from spb.vectors import _preprocess, _split_vector

"""
TODO:
    1. Implement smart_plot
    2. See if it's possible to move the logic of plot_implicit into
        ImplicitSeries. Doing that would allow get_plot_data and smart_plot
        to also work with implicit series.
"""

def _build_series(*args, **kwargs):
    """ Read the docstring of get_plot_data to unsertand what args and kwargs
    are.
    """
    # In the following dictionary the key is composed of two characters:
    # 1. The first represents the number of sub-expressions. For example,
    #    a line plot or a surface plot have 1 expression. A 2D parametric line
    #    does have 2 parameters, ...
    # 2. The second represent the number of parameters.
    # This categorization doesn't work for ImplicitSeries and InteractiveSeries,
    # in which case a random number was assigned.
    mapping = {
        "11": LineOver1DRangeSeries,
        "21": Parametric2DLineSeries,
        "31": Parametric3DLineSeries,
        "32": ParametricSurfaceSeries,
        "12": SurfaceOver2DRangeSeries,
        "99": ImplicitSeries,
        "00": InteractiveSeries,
        "22": Vector2DSeries,
        "33": Vector3DSeries,
    }

    args = _plot_sympify(args)
    exprs, ranges, label = _unpack_args(*args)
    args = [*exprs, *ranges, label]

    pt = kwargs.get("pt", None)
    if pt is None:
        # Automatic detection based on the number of free symbols and the number
        # of expressions

        skip_check = False
        if isinstance(exprs[0], (Boolean, Relational)):
            # implicit series
            skip_check = True
            npar = 9
            nexpr = 9
        elif isinstance(exprs[0], (DenseMatrix, Vector)):
            split_expr, ranges = _split_vector(exprs[0], ranges)
            nexpr = 2
            if split_expr[-1] is S.Zero:
                nexpr = 2
                args = [split_expr[:2], *ranges, label]
            else:
                nexpr = 3
                args = [split_expr, *ranges, label]
            npar = len(ranges)
        elif isinstance(exprs[0], (list, tuple, Tuple)):
            # Two possible cases:
            # 1. The actual parametric expression has been provided in the form
            #    (expr1, expr2, expr3 [optional]), range1, range2 [optional]
            # 2. A vector has been written as a tuple/list
            fs = set().union(*[e.free_symbols for e in exprs[0]])
            npar = len(fs)
            nexpr = len(exprs[0])
        else:
            # the actual expression (parametric or not) is not contained in a
            # tuple. For example:
            # expr1, expr2, expr3 [optional]), range1, range2 [optional]
            fs = set().union(*[e.free_symbols for e in exprs])
            npar = len(fs)
            nexpr = len(exprs)

        k = str(nexpr) + str(npar)
        params = kwargs.get("params", dict())
        if len(params) > 0:
            # we are most likely dealing with an interactive series
            skip_check = True
            k = "00"
            args = [exprs, ranges, label]

        if not skip_check:
            # In case of LineOver1DRangeSeries, Parametric2DLineSeries,
            # Parametric3DLineSeries, ParametricSurfaceSeries,
            # SurfaceOver2DRangeSeries, validate the provided expressions/ranges
            args = _check_arguments(args, nexpr, npar)[0]

        
        if k not in mapping.keys():
            raise ValueError(
                "Don't know how to plot your expression:\n" +
                "Received: {}\n".format(args) +
                "Number of subexpressions: {}\n".format(nexpr) +
                "Number of parameters: {}".format(npar)
            )
    else:
        if pt == "p":
            k = "11"
            args = _check_arguments(args, 1, 1)[0]
        elif pt == "pp":
            k = "21"
            args = _check_arguments(args, 2, 1)[0]
        elif pt == "p3dl":
            k = "31"
            args = _check_arguments(args, 3, 1)[0]
        elif pt == "p3d":
            k = "12"
            args = _check_arguments(args, 1, 2)[0]
        elif pt == "p3ds":
            k = "32"
            args = _check_arguments(args, 3, 2)[0]
        elif pt == "ip":
            k = "00"
            args = [exprs, ranges, label]
        elif pt == "v2d":
            k = "22"
            if isinstance(args[0], Vector):
                split_expr, ranges = _split_vector(exprs[0], ranges)
                args = [split_expr[:2], *ranges, label]
            args = _check_arguments(args, 2, 2)[0]
        elif pt == "v3d":
            k = "33"
            if isinstance(args[0], Vector):
                split_expr, ranges = _split_vector(exprs[0], ranges)
                args = [split_expr, *ranges, label]
            args = _check_arguments(args, 3, 3)[0]
        else:
            raise ValueError("Wrong `pt` value. Please, check the docstring " +
                "of `get_plot_data` to list the possible values.")

    _cls = mapping[k]
    if _cls == ImplicitSeries:
        raise ValueError(
            "_build_series is currently not able to deal with ImplicitSeries")

    kwargs = _set_discretization_points(kwargs, _cls)
    return _cls(*args, **kwargs)


def get_plot_data(*args, **kwargs):
    """ Return the numerical data associated to the a symbolic expression that
    we would like to plot. If a symbolic expression can be plotted with any of
    the plotting functions exposed by sympy.plotting.plot or 
    sympy.plotting.plot_implicit, then numerical data will be returned.

    Only one expression at a time can be processed by this function.
    The shape of the numerical data depends on the provided expression.

    Keyword Arguments
    =================

        pt : str
            Specify which kind of data you would like to obtain. Default value
            is None, indicating the function will use automatic detection.
            Possible values are:
                "p": to specify a line plot.
                "pp": to specify a 2d parametric line plot.
                "p3dl": to specify a 3d parametric line plot.
                "p3d": to specify a 3d plot.
                "p3s": to specify a 3d parametric surface plot.
                "ip": to specify an interactive plot. In such a case, you will
                        also have to provide a `param` dictionary mapping the
                        parameters to their values.
                "v2d": to specify a 2D vector plot.
                "v3d": to specify a 3D vector plot.

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
