from spb.defaults import cfg
from sympy.core.containers import Tuple
from sympy.core.sympify import sympify
from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.core.singleton import S
from sympy.matrices.dense import DenseMatrix
from sympy.vector import Vector
from sympy.vector.operators import _get_coord_systems
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction
from sympy.external import import_module


def _create_ranges(exprs, ranges, npar, label="", params=None):
    """This function does two things:
    1. Check if the number of free symbols is in agreement with the type of
       plot chosen. For example, plot() requires 1 free symbol;
       plot3d() requires 2 free symbols.
    2. Sometime users create plots without providing ranges for the variables.
       Here we create the necessary ranges.

    Parameters
    ==========

    exprs : iterable
        The expressions from which to extract the free symbols

    ranges : iterable
        The limiting ranges provided by the user

    npar : int
        The number of free symbols required by the plot functions.
        For example,
        npar=1 for plot, npar=2 for plot3d, ...

    params : dict
        A dictionary mapping symbols to parameters for iplot.

    """

    get_default_range = lambda symbol: Tuple(symbol, cfg["plot_range"]["min"], cfg["plot_range"]["max"])
    free_symbols = set().union(*[e.free_symbols for e in exprs])

    if params is not None:
        free_symbols = free_symbols.difference(params.keys())

    if len(free_symbols) > npar:
        raise ValueError(
            "Too many free symbols.\n"
            + "Expected {} free symbols.\n".format(npar)
            + "Received {}: {}".format(len(free_symbols), free_symbols)
        )

    if len(ranges) > npar:
        raise ValueError(
            "Too many ranges. Received %s, expected %s" % (len(ranges), npar))

    # free symbols in the ranges provided by the user
    rfs = set().union([r[0] for r in ranges])
    if len(rfs) != len(ranges):
        raise ValueError("Multiple ranges with the same symbol")

    if len(ranges) < npar:
        symbols = free_symbols.difference(rfs)
        if symbols != set():
            # add a range for each missing free symbols
            for s in symbols:
                ranges.append(get_default_range(s))
        # if there is still room, fill them with dummys
        for i in range(npar - len(ranges)):
            ranges.append(get_default_range(Dummy()))

    if len(free_symbols) == npar:
        # there could be times when this condition is not met, for example
        # plotting the function f(x, y) = x (which is a plane); in this case,
        # free_symbols = {x} whereas rfs = {x, y} (or x and Dummy)
        rfs = set().union([r[0] for r in ranges])
        if free_symbols.difference(rfs) != set():
            raise ValueError(
                "Incompatible free symbols of the expressions with "
                "the ranges.\n"
                + "Free symbols in the expressions: {}\n".format(free_symbols)
                + "Free symbols in the ranges: {}".format(rfs)
            )
    return ranges


def _check_interactive_fs(exprs, ranges, label, params):
    """ Checks if there are enogh parameters and free symbols in order to
    build the interactive series.
    """
    # from the expression's free symbols, remove the ones used in
    # the parameters and the ranges
    fs = set().union(*[e.free_symbols for e in exprs])
    fs = fs.difference(params.keys())
    if ranges is not None:
        fs = fs.difference([r[0] for r in ranges])

    if len(fs) > 0:
        raise ValueError(
            "Incompatible expression and parameters.\n"
            + "Expression: {}\n".format(
                (exprs, ranges, label) if ranges is not None else (exprs, label))
            + "params: {}\n".format(params)
            + "Specify what these symbols represent: {}\n".format(fs)
            + "Are they ranges or parameters?"
        )


def _check_arguments(args, nexpr, npar, **kwargs):
    """Checks the arguments and converts into tuples of the
    form (exprs, ranges, name_expr).

    Parameters
    ==========

    args
        The arguments provided to the plot functions

    nexpr
        The number of sub-expression forming an expression to be plotted.
        For example:
        nexpr=1 for plot.
        nexpr=2 for plot_parametric: a curve is represented by a tuple of two
            elements.
        nexpr=1 for plot3d.
        nexpr=3 for plot3d_parametric_line: a curve is represented by a tuple
            of three elements.

    npar
        The number of free symbols required by the plot functions. For example,
        npar=1 for plot, npar=2 for plot3d, ...

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import cos, sin, symbols
       >>> from sympy.plotting.plot import _check_arguments
       >>> x = symbols('x')
       >>> _check_arguments([cos(x), sin(x)], 2, 1)
           [(cos(x), sin(x), (x, -10, 10), '(cos(x), sin(x))')]

       >>> _check_arguments([x, x**2], 1, 1)
           [(x, (x, -10, 10), 'x'), (x**2, (x, -10, 10), 'x**2')]
    """
    if not args:
        return []
    output = []
    params = kwargs.get("params", None)

    if all([isinstance(a, (Expr, Relational, BooleanFunction)) for a in args[:nexpr]]):
        # In this case, with a single plot command, we are plotting either:
        #   1. one expression
        #   2. multiple expressions over the same range

        exprs, ranges, label, rendering_kw = _unpack_args(*args)
        free_symbols = set().union(*[e.free_symbols for e in exprs])
        ranges = _create_ranges(exprs, ranges, npar, label, params)

        if nexpr > 1:
            # in case of plot_parametric or plot3d_parametric_line, there will
            # be 2 or 3 expressions defining a curve. Group them together.
            if len(exprs) == nexpr:
                exprs = (tuple(exprs),)
        for expr in exprs:
            # need this if-else to deal with both plot/plot3d and
            # plot_parametric/plot3d_parametric_line
            is_expr = isinstance(expr, (Expr, Relational, BooleanFunction))
            e = (expr,) if is_expr else expr
            current_label = (
                label if label else str(expr) if is_expr else str(e)
            )
            if ((not label) and (current_label != label) and
                (nexpr in [2, 3]) and (npar == 1)):
                # in case of parametric 2d/3d line plots, use the parameter
                # as the label
                current_label = str(ranges[0][0])
            output.append((*e, *ranges, current_label, rendering_kw))

    else:
        # In this case, we are plotting multiple expressions, each one with its
        # range. Each "expression" to be plotted has the following form:
        # (expr, range, label) where label is optional

        _, ranges, labels, rendering_kw = _unpack_args(*args)
        labels = [labels] if labels else []

        # number of expressions
        n = (len(ranges) + len(labels) +
            (len(rendering_kw) if rendering_kw is not None else 0))
        new_args = args[:-n] if n > 0 else args

        # at this point, new_args might just be [expr]. But I need it to be
        # [[expr]] in order to be able to loop over [expr, range [opt], label [opt]]
        if not isinstance(new_args[0], (list, tuple, Tuple)):
            new_args = [new_args]

        # Each arg has the form (expr1, expr2, ..., range1 [optional], ...,
        #   label [optional], rendering_kw [optional])
        for arg in new_args:
            # look for "local" range and label. If there is not, use "global".
            l = [a for a in arg if isinstance(a, str)]
            if not l:
                l = labels
            r = [a for a in arg if _is_range(a)]
            if not r:
                r = ranges.copy()
            rend_kw = [a for a in arg if isinstance(a, dict)]
            rend_kw = rendering_kw if len(rend_kw) == 0 else rend_kw[0]

            arg = arg[:nexpr]
            free_symbols = set().union(*[a.free_symbols for a in arg])
            if len(r) != npar:
                r = _create_ranges(arg, r, npar, "", params)
            label = ""
            if not l:
                label = str(arg[0]) if nexpr == 1 else str(arg)
                if (nexpr in [2, 3]) and (npar == 1):
                    # in case of parametric 2d/3d line plots, use the
                    # parameter as the label
                    label = str(r[0][0])
            else:
                label = l[0]
            output.append((*arg, *r, label, rend_kw))
    return output


def _plot_sympify(args):
    """By allowing the users to set custom labels to the expressions being
    plotted, a critical issue is raised: whenever a special character like $,
    {, }, ... is used in the label (type string), sympify will raise an error.
    This function recursively loop over the arguments passed to the plot
    functions: the sympify function will be applied to all arguments except
    those of type string.
    """
    if isinstance(args, Expr):
        return args

    args = list(args)
    for i, a in enumerate(args):
        if isinstance(a, (list, tuple)):
            args[i] = Tuple(*_plot_sympify(a), sympify=False)
        elif not isinstance(a, (str, dict)):
            args[i] = sympify(a)
    return args


def _is_range(r):
    """A range is defined as (symbol, start, end). start and end should
    be numbers.
    """
    return (
        isinstance(r, Tuple)
        and (len(r) == 3)
        and (not isinstance(r.args[1], str)) and r.args[1].is_number
        and (not isinstance(r.args[2], str)) and r.args[2].is_number
    )


def _unpack_args(*args):
    """Given a list/tuple of arguments previously processed by _plot_sympify()
    and/or _check_arguments(), separates and returns its components:
    expressions, ranges, label and rendering keywords.

    Examples
    ========

    >>> from sympy import cos, sin, symbols
    >>> x, y = symbols('x, y')
    >>> args = (sin(x), (x, -10, 10), "f1")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
        ([sin(x)], [(x, -2, 2)], 'f1')

    >>> args = (sin(x**2 + y**2), (x, -2, 2), (y, -3, 3), "f2")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
        ([sin(x**2 + y**2)], [(x, -2, 2), (y, -3, 3)], 'f2')

    >>> args = (sin(x + y), cos(x - y), x + y, (x, -2, 2), (y, -3, 3), "f3")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
        ([sin(x + y), cos(x - y), x + y], [(x, -2, 2), (y, -3, 3)], 'f3')
    """
    ranges = [t for t in args if _is_range(t)]
    labels = [t for t in args if isinstance(t, str)]
    label = "" if not labels else labels[0]
    rendering_kw = [t for t in args if isinstance(t, dict)]
    rendering_kw = None if not rendering_kw else rendering_kw[0]
    # NOTE: why None? because args might have been preprocessed by
    # _check_arguments, so None might represent the rendering_kw
    results = [not (_is_range(a) or isinstance(a, (str, dict)) or (a is None)) for a in args]
    exprs = [a for a, b in zip(args, results) if b]
    return exprs, ranges, label, rendering_kw


def _unpack_args_extended(*args, matrices=False, fill_ranges=True):
    """Extendend _unpack_args to deal with vectors expressed as matrices.

    Parameters
    ==========
    matrices : boolean
        Default to False. If True, when a single DenseMatrix is given as
        the expression, it will be converted to a list. This is useful in
        order to deal with vectors (written in form of matrices) for
        iplot.

    fill_ranges : boolean
        Default to True. If not enough ranges are provided, the algorithm
        will try to create the missing ones.

    See also
    ========
    _unpack_args
    """
    exprs, ranges, label, rendering_kw = _unpack_args(*args)

    if label == "":
        if len(exprs) == 1:
            label = str(exprs[0])
        else:
            label = str(tuple(exprs))

    if matrices and (len(exprs) == 1):
        if isinstance(exprs[0], (list, tuple, Tuple, DenseMatrix)):
            exprs = list(exprs[0])
        elif isinstance(exprs[0], Vector):
            exprs, ranges = _split_vector(exprs[0], ranges, fill_ranges)
            if exprs[-1] is S.Zero:
                exprs = exprs[:-1]
    return exprs, ranges, label, rendering_kw


def ij2k(cols, i, j):
    """Create the connectivity for the mesh.
    https://github.com/K3D-tools/K3D-jupyter/issues/273
    """
    return cols * i + j


def get_vertices_indices(x, y, z):
    """Compute the vertices matrix (Nx3) and the connectivity list for
    triangular faces.

    Parameters
    ==========
        x, y, z : np.array
            2D arrays
    """
    np = import_module('numpy')

    rows, cols = x.shape
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    vertices = np.vstack([x, y, z]).T
    indices = []
    for i in range(1, rows):
        for j in range(1, cols):
            indices.append(
                [ij2k(cols, i, j), ij2k(cols, i - 1, j), ij2k(cols, i, j - 1)]
            )
            indices.append(
                [ij2k(cols, i - 1, j - 1), ij2k(cols, i, j - 1), ij2k(cols, i - 1, j)]
            )
    return vertices, indices


def _split_vector(expr, ranges, fill_ranges=True):
    """Extract the components of the given vector or matrix.

    Parameters
    ==========
        expr : Vector, DenseMatrix or list/tuple
        ranges : list/tuple

    Returns
    =======
        split_expr : tuple
            Tuple of the form (x_expr, y_expr, z_expr). If a 2D vector is
            provided, z_expr = S.Zero.
        ranges : list/tuple

    NOTE: this function is located in utils.py module (and not in vectors.py)
    in order to avoid circular import.
    """
    if isinstance(expr, Vector):
        N = list(_get_coord_systems(expr))[0]
        expr = expr.to_matrix(N)
    elif not isinstance(expr, (DenseMatrix, list, tuple, Tuple)):
        raise TypeError(
            "The provided expression must be a symbolic vector, or a "
            "symbolic matrix, or a tuple/list with 2 or 3 symbolic "
            + "elements.\nReceived type = {}".format(type(expr))
        )
    elif (len(expr) < 2) or (len(expr) > 3):
        raise ValueError(
            "This function only plots 2D or 3D vectors.\n"
            + "Received: {}. Number of elements: {}".format(expr, len(expr))
        )

    if fill_ranges:
        ranges = list(ranges)
        fs = set().union(*[e.free_symbols for e in expr])
        if len(ranges) < len(fs):
            fs_ranges = set().union([r[0] for r in ranges])
            for s in fs:
                if s not in fs_ranges:
                    ranges.append(Tuple(s, cfg["plot_range"]["min"], cfg["plot_range"]["max"]))

    if len(expr) == 2:
        xexpr, yexpr = expr
        zexpr = S.Zero
    else:
        xexpr, yexpr, zexpr = expr
    split_expr = xexpr, yexpr, zexpr
    return split_expr, ranges


def _instantiate_backend(Backend, *series, **kwargs):
    p = Backend(*series, **kwargs)
    if kwargs.get("show", True):
        p.show()
    return p
