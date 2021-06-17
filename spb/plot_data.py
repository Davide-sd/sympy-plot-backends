from sympy import Tuple, S, I
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.matrices.dense import DenseMatrix
from sympy.vector import Vector
from spb.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries,
    Parametric3DLineSeries, SurfaceOver2DRangeSeries,
    ParametricSurfaceSeries, ImplicitSeries, InteractiveSeries,
    _set_discretization_points, Vector2DSeries, Vector3DSeries,
    ContourSeries, ComplexSeries, ComplexInteractiveSeries
)
from spb.backends.base_backend import Plot 
from spb.utils import _unpack_args, _plot_sympify, _check_arguments
from spb.vectors import _preprocess, _split_vector
from spb.defaults import TWO_D_B, THREE_D_B

"""
TODO:
    1. Implement smart_plot
"""

def _build_series(*args, **kwargs):
    """ Read the docstring of get_plot_data to unsertand what args and kwargs
    are.
    """

    mapping = {
        "p": [LineOver1DRangeSeries, 1, 1], # [cls, nexpr, npar]
        "pp": [Parametric2DLineSeries, 2, 1],
        "p3dl": [Parametric3DLineSeries, 3, 1],
        "p3ds": [ParametricSurfaceSeries, 3, 2],
        "p3d": [SurfaceOver2DRangeSeries, 1, 2],
        "pi": [ImplicitSeries, 1, 2],
        "pinter": [InteractiveSeries, 0, 0],
        "v2d": [Vector2DSeries, 2, 2],
        "v3d": [Vector3DSeries, 3, 3],
        "pc": [ContourSeries, 1, 2],
        "c": [ComplexSeries, 1, 1]
    }

    # In the following dictionary the key is composed of two characters:
    # 1. The first represents the number of sub-expressions. For example,
    #    a line plot or a surface plot have 1 expression. A 2D parametric line
    #    does have 2 parameters, ...
    # 2. The second represent the number of parameters.
    # There will be ambiguities due to the fact that some series use the same
    # number of expressions and parameters. This dictionary set a default series
    reverse_mapping = {
        "11": "p",
        "21": "pp",
        "31": "p3dl",
        "32": "p3ds",
        "12": "p3d",
        "22": "v2d",
        "33": "v3d",
    }
    
    args = _plot_sympify(args)
    exprs, ranges, label = _unpack_args(*args)
    args = [*exprs, *ranges, label]

    pt = kwargs.pop("pt", None)
    if pt is None:
        # Automatic detection based on the number of free symbols and the number
        # of expressions

        pt = ""
        skip_check = False

        if ((len(ranges) > 0) and 
            (ranges[0][1].has(I) or ranges[0][2].has(I))):
            pt = "c"
        elif isinstance(exprs[0], (Boolean, Relational)):
            # implicit series
            pt = "pi"
        elif isinstance(exprs[0], (DenseMatrix, Vector)):
            split_expr, ranges = _split_vector(exprs[0], ranges)
            if split_expr[-1] is S.Zero:
                args = [split_expr[:2], *ranges, label]
                pt = "v2d"
            else:
                args = [split_expr, *ranges, label]
                pt = "v3d"
        elif isinstance(exprs[0], (list, tuple, Tuple)):
            if any(t.has(I) for t in exprs[0]):
                # list of complex points
                pt = "c"
                skip_check = True
                if len(args) == 2:
                    # no range has been provided
                    args = [args[0], None, args[1]]
            else:
                # Two possible cases:
                # 1. The actual parametric expression has been provided in the form
                #    (expr1, expr2, expr3 [optional]), range1, range2 [optional]
                # 2. A vector has been written as a tuple/list
                fs = set().union(*[e.free_symbols for e in exprs[0]])
                npar = len(fs)
                nexpr = len(exprs[0])
                tmp = str(nexpr) + str(npar)
                if tmp in reverse_mapping.keys():
                    pt = reverse_mapping[tmp]
        elif exprs[0].has(I):
            # complex series -> return complex numbers
            pt = "c"
            # if absargs=True, by setting the following to True, I can return 
            # (x, magn, args) rather then (x, mag) which is usually used by
            # backends
            kwargs["gpd"] = True
        else:
            # the actual expression (parametric or not) is not contained in a
            # tuple. For example:
            # expr1, expr2, expr3 [optional]), range1, range2 [optional]
            fs = set().union(*[e.free_symbols for e in exprs])
            npar = len(fs)
            nexpr = len(exprs)
            tmp = str(nexpr) + str(npar)
            if tmp in reverse_mapping.keys():
                pt = reverse_mapping[tmp]

        params = kwargs.get("params", dict())
        if len(params) > 0:
            # we are most likely dealing with an interactive series
            skip_check = True
            pt = "pinter"
            args = [exprs, ranges, label]

        if pt not in mapping.keys():
            raise ValueError(
                "Don't know how to plot the expression:\n" +
                "Received: {}\n".format(args) +
                "Number of subexpressions: {}\n".format(nexpr) +
                "Number of parameters: {}".format(npar)
            )
        
        _cls, nexpr, npar = mapping[pt]
        if not skip_check:
            # In case of LineOver1DRangeSeries, Parametric2DLineSeries,
            # Parametric3DLineSeries, ParametricSurfaceSeries,
            # SurfaceOver2DRangeSeries, validate the provided expressions/ranges
            args = _check_arguments(args, nexpr, npar)[0]

    else:
        if pt in mapping.keys():
            _cls, nexpr, npar = mapping[pt]
            k = str(nexpr) + str(npar)
            if k == "00":
                args = [exprs, ranges, label]
                if kwargs.get("is_complex", False):
                    _cls = ComplexInteractiveSeries
                    args = [exprs[0], ranges[0], label]
            elif k == "22":
                if isinstance(args[0], (Vector, DenseMatrix)):
                    split_expr, ranges = _split_vector(exprs[0], ranges)
                    args = [split_expr[:2], *ranges, label]
                args = _check_arguments(args, 2, 2)[0]
            elif k == "33":
                if isinstance(args[0], (Vector, DenseMatrix)):
                    split_expr, ranges = _split_vector(exprs[0], ranges)
                    args = [split_expr, *ranges, label]
                args = _check_arguments(args, 3, 3)[0]
            else:
                args = _check_arguments(args, nexpr, npar)[0]
        else:
            raise ValueError("Wrong `pt` value. Please, check the docstring " +
                "of `get_plot_data` to list the possible values.")
    kwargs = _set_discretization_points(kwargs, _cls)
    kwargs["gpd"] = True
    return _cls(*args, **kwargs)


def get_plot_data(*args, **kwargs):
    """ Return the numerical data associated to the a symbolic expression that
    we would like to plot. If a symbolic expression can be plotted with any of
    the plotting functions exposed by spb.functions or spb.vectors, then 
    numerical data will be returned.

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
                "p3ds": to specify a 3d parametric surface plot.
                "pi": to specify an implificit plot.
                "pinter": to specify an interactive plot. In such a case, you
                        will also have to provide a `param` dictionary mapping
                        theparameters to their values.
                "v2d": to specify a 2D vector plot.
                "v3d": to specify a 3D vector plot.
                "c": to specify a complex plot.

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

    Get the real and imaginary part of a complex function over a real range:

    .. code-block:: python
        z = symbols("z")
        xx, real, imag = get_plot_data(sqrt(z), (z, -3, 3), pt="c")
    
    Note the use of pt="c" to specify a complex plot: the expression doesn't 
    contain the imaginary unit, hence we need to aid the detection algorithm.
    
    Get the magnitude and argument of a complex function over a real range:

    .. code-block:: python
        z = symbols("z")
        expr = 1 + exp(-Abs(z)) * sin(I * sin(5 * z))
        xx, mag, arg = get_plot_data(expr, (z, -3, 3), absarg=True)

    Compute a complex function over a complex range:

    .. code-block:: python
        z = symbols("z")
        xx, yy, zz, abs, arg = get_plot_data(gamma(z), (z, -3 - 3*I, 3 + 3*I))
    
    Here, `xx, yy, zz` are 2D arrays. `xx` is the real part of the domain.
    `yy` is the complex part of the domain. `zz` contains complex numbers.
    `abs` and `arg` are the absolute value and argument of `zz`.

    See also
    ========

    plot, plot_parametric, plot3d, plot3d_parametric_line,
    plot3d_parametric_surface, plot_contour, plot_implicit,
    vector_plot, complex_plot
    """
    return _build_series(*args, **kwargs).get_data()

def smart_plot(*args, show=True, **kwargs):
    """ Smart plot interface. Using the same interface of the other plot
    functions, namely (expr, range, label), it unifies the plotting experience.
    If a symbolic expression can be plotted with any of the plotting functions
    exposed by spb.functions or spb.vectors, then this function will be able
    to plot it as well.

    Keyword Arguments
    =================
        The usual keyword arguments available on every other plotting functions
        are available (`xlabel`, ..., `adaptive`, `n`, ...). On top of that
        we can set:

        pt : str
            Specify which kind of plot we are intereseted. Default value
            is None, indicating the function will use automatic detection.
            Possible values are:
                "p": to specify a line plot.
                "pp": to specify a 2d parametric line plot.
                "p3dl": to specify a 3d parametric line plot.
                "p3d": to specify a 3d plot.
                "p3ds": to specify a 3d parametric surface plot.
                "pi": to specify an implificit plot.
                "pinter": to specify an interactive plot. In such a case, you
                        will also have to provide a `param` dictionary mapping
                        theparameters to their values.
                        To specify a complex-interactive plot, set 
                        `is_complex=True`.
                "v2d": to specify a 2D vector plot.
                "v3d": to specify a 3D vector plot.
                "c": to specify a complex plot.

    Examples
    ========

    Plotting different types of expressions with automatic detection:

    .. code-block:: python
        from sympy import symbols, sin, cos, Matrix
        from spb.backends.plotly import PB
        x, y = symbols("x, y")
        smart_plot(
            (Matrix([-sin(y), cos(x)]), (x, -5, 5), (y, -3, 3), "vector"),
            (sin(x), (x, -5, 5)),
            aspect="equal", n=20, legend=True,
            quiver_kw=dict(scale=0.25), line_kw=dict(line_color="cyan"),
            backend=PB
        )

    Specify the kind of plot we are interested in:

    .. code-block:: python
        from sympy import symbols, cos
        x, y = symbols("x, y")
        plot(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
            xlabel="x", ylabel="y", pt="pc")

    See also
    ========

    plot, plot_parametric, plot3d, plot3d_parametric_line,
    plot3d_parametric_surface, plot_contour, plot_implicit,
    vector_plot, complex_plot
    """

    args = _plot_sympify(args)
    if not all([isinstance(a, (list, tuple, Tuple)) for a in args]):
        args = [args]
    series = []

    for arg in args:
        series.append(_build_series(*arg, **kwargs))
    
    if "backend" not in kwargs.keys():
        is_3D = any([s.is_3D for s in series])
        kwargs["backend"] = THREE_D_B if is_3D else TWO_D_B
    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots
