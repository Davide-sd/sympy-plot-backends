from spb.defaults import cfg
from sympy import Tuple, sympify, Expr, Dummy, S
from sympy.matrices.dense import DenseMatrix
from sympy.physics.mechanics import Vector as MechVector
from sympy.vector import Vector
from sympy.vector.operators import _get_coord_systems
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction
from sympy.external import import_module
import warnings

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
    free_symbols = set()
    if all(not callable(e) for e in exprs):
        free_symbols = free_symbols.union(*[e.free_symbols for e in exprs])

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
            output.append((*e, *ranges, label, rendering_kw))

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
        # [[expr]] in order to be able to loop over
        # [expr, range [opt], label [opt]]
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
            free_symbols = set()
            if all(not callable(a) for a in arg):
                free_symbols = free_symbols.union(*[a.free_symbols for a in arg])
            if len(r) != npar:
                r = _create_ranges(arg, r, npar, "", params)

            label = None if not l else l[0]
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
        elif not (isinstance(a, (str, dict, MechVector)) or callable(a)):
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
    label = None if not labels else labels[0]
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
    elif isinstance(expr, MechVector):
        expr = expr.args[0][0]
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
        fs = set()
        if all(not callable(e) for e in expr):
            fs = fs.union(*[e.free_symbols for e in expr])
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
    _validate_kwargs(p, **kwargs)

    if kwargs.get("show", True):
        p.show()
    return p


def _validate_kwargs(backend_obj, **kwargs):
    """Find the user-provided keywords arguments that might contain spelling
    errors and informs the user of possible alternatives.

    Parameters
    ==========
    backend_obj : Plot
        An instance of the Plot class

    Notes
    =====
    To keep development "agile", I extensively used ``**kwargs`` everywhere.
    The problem is that there are "multiple levels" of keyword arguments:

    * some keyword arguments get intercepted at the plotting function level.
      Think for example to ``scalar`` in ``plot_vector``, or ``sum_bound`` in
      ``plot``.
    * some plotting function might insert useful keyword arguments, for example
      ``real``, ``imag``, etc., on complex-related functions.
    * many of the keyword arguments get passed down to the Series and to
      the Backend classes.

    There are many approaches to tackle keyword arguments validation:

    1. Replace **kwargs everywhere with the actual expected keywords.
       This is very time consuming and hard to maintain as the module gets
       developed even further. Morover, Python will raise an error everytime
       something get mispelled, which I think is annoying.
    2. Perform "multiple levels" of keyword validation, at a plotting function
       level (on each function), at a series level and at a backend level.
       Again, time consuming.
    3. The laziest and most simple approach I could think of: create the
       ``_allowed_keys`` attribute on Series and Backend classes. Implement
       this function to perform some validation. It is definitely not as good
       as the previous approaches, in particular:

       * the validation is actually done after the creation of Series and
         Backend. This is not a problem as the validation is only meant to show
         a warning message.
       * needs to be careful when modifying Series and Backend, as the
         ``_allowed_keys`` attribute must be update.
       * function-level keyword arguments must be listed inside this function.
         Again, not so great in terms of further development.

       But it's a quick approach and surely better than nothing.
    """
    # find the user-provided keywords arguments that might contain
    # spelling errors and inform the user of possible alternatives.
    allowed_keys = set(backend_obj._allowed_keys)
    for s in backend_obj.series:
        allowed_keys = allowed_keys.union(s._allowed_keys)
    # some functions injects the following keyword arguments that will be
    # processed by other functions before instantion of Series and Backend.
    allowed_keys = allowed_keys.union([
        "is_interactive", "abs", "absarg", "arg", "real", "imag",
        "is_complex", "is_vector", "slice", "threed", "sum_bound", "n",
        "phaseres", "is_polar", "label"
    ])
    # params is a keyword argument that is also checked before instantion of
    # Series and Backend.
    allowed_keys = allowed_keys.union(["params", "layout", "ncols",
        "use_latex", "throttled", "servable", "custom_css", "pane_kw",
        "is_iplot"])
    user_provided_keys = set(kwargs.keys())
    unused_keys = user_provided_keys.difference(allowed_keys)
    if len(unused_keys) > 0:
        msg = "\nThe following keyword arguments are unused.\n"
        for k in unused_keys:
            possible_match = find_closest_string(k, allowed_keys)
            msg += "* '%s'" % k
            msg += ": did you mean '%s'?\n" % possible_match
        warnings.warn(msg)
        # this returns helps with tests
        return msg


# taken from
# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)  # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            # than s2
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# taken from plotly.py/packages/python/plotly/_plotly_utils/utils.py
def find_closest_string(string, strings):
    def _key(s):
        # sort by levenshtein distance and lexographically to maintain a stable
        # sort for different keys with the same levenshtein distance
        return (levenshtein(s, string), s)

    return sorted(strings, key=_key)[0]
