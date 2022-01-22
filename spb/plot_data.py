from sympy import Tuple, S, I, Plane
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.matrices.dense import DenseMatrix
from sympy.vector import Vector
from sympy.geometry.entity import GeometryEntity
from spb.series import (
    LineOver1DRangeSeries,
    Parametric2DLineSeries,
    Parametric3DLineSeries,
    SurfaceOver2DRangeSeries,
    ParametricSurfaceSeries,
    ImplicitSeries,
    InteractiveSeries,
    _set_discretization_points,
    Vector2DSeries,
    Vector3DSeries,
    ContourSeries,
    ComplexSurfaceBaseSeries,
    ComplexInteractiveBaseSeries,
    SliceVector3DSeries,
    GeometrySeries,
    PlaneInteractiveSeries,
)
from spb.backends.base_backend import Plot
from spb.utils import _unpack_args, _plot_sympify, _check_arguments
from spb.vectors import _split_vector
from spb.ccomplex.complex import _build_series as _build_complex_series


def _deal_with_complex_series(exprs, ranges, interactive, kwargs, pt):
    """ Look for complex-related keyword arguments. If found, build and return
    a complex-related data series.
    """
    keys = ["real", "imag", "abs", "arg", "absarg"]
    for t in keys:
        kwargs.setdefault(t, False)
    real, imag, _abs, _arg, absarg = [kwargs.get(t, False) for t in keys]
    
    if any([real, imag, _abs, _arg, absarg]) or (pt == "c"):
        
        series = _build_complex_series(*exprs, *ranges,
                interactive=interactive, **kwargs)
        if len(series) > 1:
            raise ValueError("Multiple data complex-related series have been " +
                "generated: this is an ambiguous situation because it's " +
                "impossible to determine what will be returned. Please, set " +
                "to True only one of the following keyword arguments:\n" +
                "real={}, imag={}, abs={}, arg={}, absarg={}".format(
                    real, imag, _abs, _arg, absarg
                ))
        return series[0]
    return None

def _build_series(*args, **kwargs):
    """Read the docstring of get_plot_data to unsertand what args and kwargs
    are.
    """

    mapping = {
        "p": [LineOver1DRangeSeries, 1, 1],  # [cls, nexpr, npar]
        "pp": [Parametric2DLineSeries, 2, 1],
        "p3dl": [Parametric3DLineSeries, 3, 1],
        "p3ds": [ParametricSurfaceSeries, 3, 2],
        "p3d": [SurfaceOver2DRangeSeries, 1, 2],
        "pi": [ImplicitSeries, 1, 2],
        "pinter": [InteractiveSeries, 0, 0],
        "v2d": [Vector2DSeries, 2, 2],
        "v3d": [Vector3DSeries, 3, 3],
        "v3ds": [SliceVector3DSeries, 3, 3],
        "pc": [ContourSeries, 1, 2],
        "c": [ComplexSurfaceBaseSeries, 1, 1],
        "g": [GeometrySeries, 9, 9],
    }

    # In the following dictionary the key is composed of two characters:
    # 1. The first represents the number of sub-expressions. For example,
    #    a line plot or a surface plot have 1 expression. A 2D parametric line
    #    does have 2 parameters, ...
    # 2. The second represent the number of parameters.
    # There will be ambiguities due to the fact that some series use the
    # same number of expressions and parameters. This dictionary set a
    # default series
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

    params = kwargs.get("params", dict())
    pt = kwargs.pop("pt", None)
    if pt is None:
        # Automatic detection based on the number of free symbols and the
        # number of expressions
        pt = ""
        skip_check = False

        if (len(ranges) > 0) and (ranges[0][1].has(I) or ranges[0][2].has(I)):
            pt = "c"
        elif isinstance(exprs[0], GeometryEntity):
            if len(ranges) == 0:
                ranges = [None]
            args = [*exprs, *ranges, label]
            pt = "g"
            skip_check = True
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

        # intercept complex-related series
        interactive = True if len(params) > 0 else False
        r = _deal_with_complex_series(exprs, ranges, interactive, kwargs, pt)
        if r is not None:
            return r

        if len(params) > 0:
            # we are most likely dealing with an interactive series
            skip_check = True
            pt = "pinter"
            args = [exprs, ranges, label]

        if pt not in mapping.keys():
            raise ValueError(
                "Don't know how to plot the expression:\n"
                + "Received: {}\n".format(args)
                + "Number of subexpressions: {}\n".format(nexpr)
                + "Number of parameters: {}".format(npar)
            )

        _cls, nexpr, npar = mapping[pt]
        if not skip_check:
            # In case of LineOver1DRangeSeries, Parametric2DLineSeries,
            # Parametric3DLineSeries, ParametricSurfaceSeries,
            # SurfaceOver2DRangeSeries, validate the provided
            # expressions/ranges
            args = _check_arguments(args, nexpr, npar)[0]

        _slice = kwargs.pop("slice", None)
        if pt == "v3d" and (_slice is not None):
            args = [_slice] + list(args)
            _cls, nexpr, npar = mapping["v3ds"]

    else:
        if pt in mapping.keys():
            # intercept complex-related series
            interactive = True if len(params) > 0 else False
            r = _deal_with_complex_series(exprs, ranges, interactive, kwargs, pt)
            if r is not None:
                return r

            _cls, nexpr, npar = mapping[pt]
            k = str(nexpr) + str(npar)
            if k == "00":
                args = [exprs, ranges, label]
                
                s, e = [complex(t) for t in ranges[0][1:]]
                if s.imag != e.imag:
                    # we are dealing with 2D/3D domain coloring
                    _cls = ComplexInteractiveBaseSeries
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

                _slice = kwargs.pop("slice", None)
                if _slice is not None:
                    args = [_slice] + list(args)
                    _cls = SliceVector3DSeries
            elif k == "99":
                _cls = GeometrySeries
                if len(ranges) == 0:
                    ranges = [None]
                args = [*exprs, *ranges, label]
                if (
                    isinstance(exprs[0], Plane)
                    and len(kwargs.get("params", dict())) > 0
                ):
                    _cls = PlaneInteractiveSeries
                    args = [exprs, ranges, label]
            else:
                args = _check_arguments(args, nexpr, npar)[0]
        else:
            raise ValueError(
                "Wrong `pt` value. Please, check the docstring "
                + "of `get_plot_data` to list the possible values."
            )
    kwargs = _set_discretization_points(kwargs, _cls)
    return _cls(*args, **kwargs)


def get_plot_data(*args, **kwargs):
    """
    Return the numerical data associated to the a symbolic expression that we
    would like to plot. If a symbolic expression can be plotted with any of
    the plotting functions exposed by spb.functions or spb.vectors, then numerical data will be returned.

    Only one expression at a time can be processed by this function.
    The shape of the numerical data depends on the provided expression.

    Parameters
    ==========

    args :
        expr : Expr
            Expression (or expressions) representing the function to evaluate.

        range: (symbol, min, max)
            A 3-tuple (or multiple 3-tuple) denoting the range of the
            variable.

    pt : str, optional
        Specify which kind of data the user would like to obtain. Default
        value is `None`, indicating the function will use automatic
        detection. Possible values are:

        - `"p"`: to specify a line plot.
        - `"pp"`: to specify a 2d parametric line plot.
        - `"p3dl"`: to specify a 3d parametric line plot.
        - `"p3d"`: to specify a 3d plot.
        - `"p3ds"`: to specify a 3d parametric surface plot.
        - `"pc"`: to specify a contour plot.
        - `"pi"`: to specify an implificit plot.
        - `"pinter"`: to specify an interactive plot. In such a case, the
          user will also have to provide a `param` dictionary mapping the
          parameters to their values.
        - `"v2d"`: to specify a 2D vector plot.
        - `"v3d"`: to specify a 3D vector plot.

    get_series : boolean, optional
        If False, it returns the numerical data associated to the provided
        expression. If True, it returns the data series object which can be
        used to generate the data. Default to False.

    Examples
    ========

    >>> from sympy import (symbols, pi, sin, cos, exp, Plane,
    ...     Matrix, gamma, I, sqrt, Abs)
    >>> from spb.plot_data import get_plot_data
    >>> u, v, x, y, z = symbols('u, v, x:z')

    Data from a function with a single variable (2D line):

    >>> xx, yy = get_plot_data(cos(x), (x, -5, 5))

    Here, `xx` and `yy` are two lists of coordinates.

    Data from a function with two variables (surface):

    >>> xx, yy, zz = get_plot_data(cos(x * y), (x, -5, 5), (y, -10, 10))

    Here, `xx, yy, zz` are two-dimensional numpy arrays. `
    `xx, yy` represent the mesh grid.

    Data from a 2D parametric function with one variable (2D line):

    >>> xx, yy, param = get_plot_data(cos(u), sin(u), (u, 0, 2 * pi))

    Here, `xx, yy` are two lists of coordinates.

    Data from a 3D parametric function with one variables (3D line):

    >>> xx, yy, zz, param = get_plot_data(cos(u), sin(u), u, (u, -5, 5))

    Here, `xx, yy, zz` are three lists of coordinates.

    Data from an implicit relation:

    >>> data = get_plot_data(x > y)

    Here, `data` depends on the specific case. Its shape could be:

    - `data = ((xx, yy), 'fill')` where `xx, yy` are two lists of
      coordinates.
    - `data = (xx, yy, zz, 'contour')` where `xx, yy, zz` are
      two-dimensional numpy arrays. `xx, yy` represent the mesh grid.
      This is returned by objects of type `Equality`.
    - `data = (xx, yy, zz, 'contourf')` where `xx, yy, zz` are
      two-dimensional numpy arrays. `xx, yy` represent the mesh grid.
      This is returned by objects of type non-equalities (greater than,
      less than, ...).

    Get data from a symbolic geometry entity:

    >>> xx, yy = get_plot_data(Ellipse(Point(0, 0), 5, 1), n=10)

    Get data from a plane: being a 3D entity, it requires three ranges, even
    if the plane is parallel to one of the xy, yz, xz planes.

    >>> xx, yy, zz = get_plot_data(Plane((0, 0, 0), (1, 1, 0)),
    ...     (x, -10, 10), (y, -10, 10), (z, -10, 10))

    Get the necessary data to plot a 3D vector field over a slice plane:

    >>> xx, yy, zz, uu, vv, ww = get_plot_data(
    ...     Matrix([z, y, x]), (x, -5, 5), (y, -5, 5), (z, -5, 5),
    ...     slice = Plane((-2, 0, 0), (1, 1, 1)), n=5)

    Here `xx, yy, zz, uu, vv, ww` are three dimensional numpy arrays.

    Get the real part of a complex function over a real range:

    >>> xx, real = get_plot_data(sqrt(x), (x, -3, 3),
    ...     real=True, imag=False)

    Get the magnitude and argument of a complex function over a real range:

    >>> expr = 1 + exp(-Abs(x)) * sin(I * sin(5 * x))
    >>> xx, mag, arg = get_plot_data(expr, (x, -3, 3), absarg=True)

    Compute a complex function over a complex range:

    >>> xx, yy, abs, arg, img, colorscale = get_plot_data(gamma(z),
    ...     (z, -3 - 3*I, 3 + 3*I))

    Here, `xx, yy` are 2D arrays representing the real and the imaginary part
    of the domain, respectively. `abs` and `arg` are the absolute value and
    argument of the complex function. `img` is a matrix [n x m x 3] of RGB
    colors, whereas `colorscale` is a [N x 3] matrix of RGB colors
    representing the colorscale being used by `img`.


    See also
    ========

    smart_plot

    """
    get_series = kwargs.pop("get_series", False)
    s = _build_series(*args, **kwargs)
    if get_series:
        return s
    return s.get_data()


def smart_plot(*args, show=True, **kwargs):
    """
    Smart plot interface. Using the same interface of the other plot functions,
    namely `(expr, range, label)`, it unifies the plotting experience.
    If a symbolic expression can be plotted with any of the plotting functions
    exposed by `spb.functions` or `spb.vectors` or `spb.ccomplex.complex`,
    then `smart_plot` will be able to plot it as well.

    The usual keyword arguments available on every other plotting functions
    are available (`xlabel`, ..., `adaptive`, `n`, etc.).

    Parameters
    ==========

    pt : str
        Specify which kind of plot we are intereseted. Default value is
        `None`, indicating the function will use automatic detection.
        Possible values are:

        - `"p"`: to specify a line plot.
        - `"pp"`: to specify a 2d parametric line plot.
        - `"p3dl"`: to specify a 3d parametric line plot.
        - `"p3d"`: to specify a 3d plot.
        - `"p3ds"`: to specify a 3d parametric surface plot.
        - `"pc"`: to specify a contour plot.
        - `"pi"`: to specify an implificit plot.
        - `"pinter"`: to specify an interactive plot. In such a case, the
          user will also have to provide a `param` dictionary mapping the
          parameters to their values.
        - `"v2d"`: to specify a 2D vector plot.
        - `"v3d"`: to specify a 3D vector plot.
        - `"g"`: to specify a geometric entity plot.

    Examples
    ========

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import symbols, sin, cos, Matrix, Plane, gamma
        >>> from spb.plot_data import smart_plot
        >>> x, y, z = symbols("x, y, z")
        Plot object containing:
        [0]: 2D vector series: [-sin(y), cos(x)] over (x, -5.0, 5.0), (y, -3.0, 3.0)
        [1]: cartesian line: sin(x) for x over (-5.0, 5.0)

    Plotting a vector field and a line plot with automatic detection.
    Note that we are also setting the number of discretization point and
    the aspect ratio.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> smart_plot(
        ...     (Matrix([-sin(y), cos(x)]), (x, -5, 5), (y, -3, 3), "vector"),
        ...     (sin(x), (x, -5, 5)),
        ...     aspect="equal", n=20)

    Create a contour plot by specifying the plot type.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> smart_plot(cos(x**2 + y**2), (x, -3, 3), (y, -3, 3),
        ...     xlabel="x", ylabel="y", pt="pc")
        Plot object containing:
        [0]: contour: cos(x**2 + y**2) for x over (-3.0, 3.0) and y over (-3.0, 3.0)

    Create a 2D domain coloring plot.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> smart_plot(gamma(z), (z, -3-3j, 3+3j), coloring="b")
        Plot object containing:
        [0]: domain coloring: gamma(z) for re(z) over (-3.0, 3.0) and im(z) over (-3.0, 3.0)

    Create a 3D vector plot.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> smart_plot(
        ...     Matrix([z, y, x]), (x, -5, 5), (y, -5, 5), (z, -5, 5),
        ...     slice=Plane((0, 0, 0), (1, 1, 1)),
        ...     quiver_kw={"length": 0.25})
        Plot object containing:
        [0]: sliced 3D vector series: [z, y, x] over (x, -5.0, 5.0), (y, -5.0, 5.0), (z, -5.0, 5.0) at Plane(Point3D(0, 0, 0), (1, 1, 1))

    See also
    ========

    get_plot_data

    """

    args = _plot_sympify(args)
    if not all([isinstance(a, (list, tuple, Tuple)) for a in args]):
        args = [args]
    series = []

    for arg in args:
        series.append(_build_series(*arg, **kwargs))

    if "backend" not in kwargs.keys():
        from spb.defaults import TWO_D_B, THREE_D_B

        is_3D = any([s.is_3D for s in series])
        kwargs["backend"] = THREE_D_B if is_3D else TWO_D_B
    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots
