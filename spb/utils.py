from spb.defaults import cfg
from sympy import (
    Tuple, sympify, Expr, Dummy, sin, cos, Symbol, Indexed, ImageSet,
    FiniteSet, Basic, Float, Integer, Rational, Poly, fraction, exp,
    NumberSymbol, IndexedBase
)
from sympy.vector import BaseScalar
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction
from sympy.external import import_module
import param
import warnings
import inspect


def _create_missing_ranges(exprs, ranges, npar, params=None, imaginary=False):
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
    imaginary : bool
        Include the imaginary part. Default to False.

    """

    def get_default_range(symbol):
        _min = cfg["plot_range"]["min"]
        _max = cfg["plot_range"]["max"]
        if not imaginary:
            return Tuple(symbol, _min, _max)
        return Tuple(symbol, _min + _min * 1j, _max + _max * 1j)

    free_symbols = _get_free_symbols(exprs)
    if params is not None:
        if any(isinstance(t, (list, tuple)) for t in params.keys()):
            # take care of RangeSlider
            p_symbols = set()
            for k in params.keys():
                if isinstance(k, (list, tuple)):
                    p_symbols = p_symbols.union(k)
                else:
                    p_symbols = p_symbols.union([k])
        else:
            p_symbols = params.keys()
        free_symbols = free_symbols.difference(p_symbols)

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
        if len(free_symbols.difference(rfs)) > 0:
            raise ValueError(
                "Incompatible free symbols of the expressions with "
                "the ranges.\n"
                + "Free symbols in the expressions: {}\n".format(free_symbols)
                + "Free symbols in the ranges: {}".format(rfs)
            )
    return ranges


def _create_ranges_iterable(*ranges):
    """Create a list of ranges. If a range is not provided, it won't be
    included in this list.

    Returns
    -------
    provided_ranges : list
        A list, for example `[r1, r2, r3]`. If `r2` is not provide, the list
        looks like `[r1, r2]`. If no range is provided, `[]` is returned.
    mapping : dict
        Maps the i-th provided range to its position in `provided_ranges`.
    """
    provided_ranges = []
    mapping = {}
    for i, r in enumerate(ranges):
        if r is not None:
            provided_ranges.append(r)
            mapping[i] = len(provided_ranges) - 1
    return provided_ranges, mapping


def _preprocess_multiple_ranges(exprs, ranges, npar, params={}):
    """Users might not provide the necessary ranges to create a 3D plot.
    This function looks at what has been provided, eventually add missing
    ranges and sort them to the appropriate order.

    Parameters
    ----------
    exprs : iterable
        The expressions from which to extract the free symbols
    ranges : iterable
        The limiting ranges provided by the user
    npar : int
        The number of free symbols required by the plot functions.
        For example, npar=1 for plot, npar=2 for plot3d, ...
    params : dict
        A dictionary mapping symbols to parameters for iplot.
    """
    provided_ranges, mapping = _create_ranges_iterable(*ranges)
    # add missing ranges
    ranges = _create_missing_ranges(
        exprs, provided_ranges.copy(), npar, params)
    # sort the ranges in order to get [range1, range2, ranges3 [optional]]
    sorted_ranges = [None] * npar
    for k, v in mapping.items():
        sorted_ranges[k] = provided_ranges[v]
        ranges.remove(provided_ranges[v])
    for r in ranges:
        i = sorted_ranges.index(None)
        sorted_ranges[i] = r
    return sorted_ranges


def _get_free_symbols(exprs):
    """Returns the free symbols of a symbolic expression.

    If the expression contains any of these elements, assume that they are
    the "free symbols" of the expression:

    * indexed objects
    * applied undefined function (useful for sympy.physics.mechanics module)
    """
    # TODO: this function gets called 3 times to generate a single plot.
    # See if its possible to remove one functions call inside series.py
    if not isinstance(exprs, (list, tuple, set)):
        exprs = [exprs]
    if all(callable(e) for e in exprs):
        return set()

    # NOTE:
    # 1. srepr(IndexedBase("a")) is "IndexedBase(Symbol('a'))"
    #    So, if expr = IndexedBase("a")[0] + 1, it follows that
    #    expr.free_symbols is {IndexedBase("a")[0], Symbol("a")}
    #    This must be filtered to {IndexedBase("a")[0]}
    # 2. Let a = IndexedBase("a"). Even though as of sympy 1.14.0 it is
    #    possible to write expressions like a + 1, for simplicity,
    #    I don't allow them, because of Note 1, which would increase
    #    complexity in this code.

    undefined_func = set().union(*[e.atoms(AppliedUndef) for e in exprs])
    undefined_func_args = set().union(*[f.args for f in undefined_func])
    indexed_base = set().union(*[e.atoms(IndexedBase) for e in exprs])
    indexed_base_args = set().union(*[i.args for i in indexed_base])

    # select all free symbols, be them instances of Symbol, Indexed
    # or the arguments of IndexedBase
    free_symbols = set().union(*[e.free_symbols for e in exprs])
    # remove instances of IndexedBase
    free_symbols = free_symbols.difference(indexed_base)
    # remove free symbols that are arguments of applied undef functions
    # it is unlikely that these symbols are being used as parameters as well.
    free_symbols = free_symbols.difference(undefined_func_args)
    # remove free symbols that are arguments of indexed base
    free_symbols = free_symbols.difference(indexed_base_args)

    free = free_symbols.union(undefined_func)

    return free


def _check_arguments(args, nexpr, npar, **kwargs):
    """Checks the arguments and converts into tuples of the
    form (exprs, ranges, label, rendering_kw).

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
    **kwargs :
        keyword arguments passed to the plotting function. It will be used to
        verify if ``params`` has ben provided.

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
        ranges = _create_missing_ranges(exprs, ranges, npar, params)

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
        n = (
            len(ranges) + len(labels) +
            (len(rendering_kw) if rendering_kw is not None else 0)
        )
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

            # NOTE: arg = arg[:nexpr] may raise an exception if lambda
            # functions are used. Execute the following instead:
            arg = [arg[i] for i in range(nexpr)]
            free_symbols = set()
            if all(not callable(a) for a in arg):
                free_symbols = free_symbols.union(*[
                    a.free_symbols for a in arg])
            if len(r) != npar:
                r = _create_missing_ranges(arg, r, npar, params)

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
        elif not (
            isinstance(a, (str, dict)) or callable(a)
            or (
                (a.__class__.__name__ == "Vector") and
                not isinstance(a, Basic)
            )
        ):
            args[i] = sympify(a)
    return args


def _is_range(r):
    """A range is defined as (symbol, start, end). start and end should
    be numbers.
    """
    if isinstance(r, prange):
        return True
    return (
        isinstance(r, Tuple)
        and (len(r) == 3)
        and (not isinstance(r.args[1], str)) and r.args[1].is_number
        and (not isinstance(r.args[2], str)) and r.args[2].is_number
    )


class prange(Tuple):
    """Represents a plot range, an entity describing what interval a
    particular variable is allowed to vary. It is a 3-elements tuple:
    (symbol, minimum, maximum).

    Notes
    =====

    Why does the plotting module needs this class instead of providing a
    plotting range with ordinary tuple/list? After all, ordinary plots
    works just fine.

    If a plotting range is provided with a 3-elements tuple/list, the internal
    algorithm looks at the tuple and tries to determine what it is.
    If minimum and maximum are numeric values, than it is a plotting range.

    Hovewer, there are some plotting functions in which the expression consists
    of 3-elements tuple/list. The plotting module is also interactive, meaning
    that minimum and maximum can also be expressions containing parameters.
    In these cases, the plotting range is indistinguishable from a 3-elements
    tuple describing an expression.

    This class is meant to solve that ambiguity: it only represents a plotting
    range.

    Examples
    ========

    Let x be a symbol and u, v, t be parameters. An example plotting range is:

    .. doctest::

       >>> from sympy import symbols
       >>> from spb import prange
       >>> x, u, v, t = symbols("x, u, v, t")
       >>> prange(x, u * v, v**2 + t)
       (x, u*v, t + v**2)

    """
    def __new__(cls, *args):
        if len(args) != 3:
            raise ValueError(
                "`%s` requires 3 elements. Received " % cls.__name__ +
                "%s elements: %s" % (len(args), args))
        if not isinstance(args[0], (str, Symbol, BaseScalar, Indexed)):
            raise TypeError(
                "The first element of a plotting range must "
                "be a symbol. Received: %s" % type(args[0])
            )
        args = [sympify(a) for a in args]
        if (
            (args[0] in args[1].free_symbols) or
            (args[0] in args[2].free_symbols)
        ):
            raise ValueError(
                "Symbol `%s` representing the range can only " % args[0] +
                "be specified in the first element of %s" % cls.__name__)
        return Tuple.__new__(cls, *args, sympify=False)


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
    results = [
        not (_is_range(a) or isinstance(a, (str, dict)) or (a is None))
        for a in args
    ]
    exprs = [a for a, b in zip(args, results) if b]
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


def _instantiate_backend(Backend, *series, **kwargs):
    show = kwargs.pop("show", True)
    p = Backend(*series, **kwargs)

    if show:
        p.show()
    return p


def _check_misspelled_kwargs(
    obj, additional_keys=[], exclude_keys=[], **kwargs):
    """Find the user-provided keywords arguments that might contain spelling
    errors and informs the user of possible alternatives.

    Parameters
    ==========
    obj : param.Parameterized
        The object holding the correct parameter names.
    additional_keys : list
        List of string representing additional keyword arguments that might be
        involved in the instantiation of `obj`.
    exclude_keys : list
        List of string representing parameter names that should not be
        considered while performing the validation.
    **kwargs : dict
        Keyword arguments passed to `obj` __init__ method.

    Notes
    =====

    Within this module, there are "multiple levels" of keyword arguments:

    * some keyword arguments get intercepted at the plotting function level.
      Think for example to ``scalar`` in ``plot_vector``, or ``sum_bound`` in
      ``plot``.
    * some plotting function might insert useful keyword arguments, for example
      ``real``, ``imag``, etc., on complex-related functions.
    * many of the keyword arguments get passed down to the Series and/or to
      the Backend classes (for example, ``xscale, ...``).

    After porting this module to param, I have implemented the validation
    of keyword arguments at the ``*Series`` and ``graphics``. I was unable
    to perform it inside the ``Plot`` class because the ``graphics`` function
    removes unused keyword arguments.
    Hopefully one day I'll implement on the interactive level too
    (interactive plots and animations).

    The plotting module offers two main approaches:

    * spb.plot_function, inherithed from sympy.plotting. Here, the problem is
      that keyword arguments from a specific plot function get directed
      both at series as well as the backend. For example, the Plot class could
      receive arguments that are meant to go to LineOver1DRangeSeries, and
      vice-versa. It's a mess. In order to deal with this mess, I introduced
      the `plot_function` keyword argument: this will enable validation on data
      series but not on the ``graphics`` function.
    * spb.graphics: here, there is a clear separation between data series
      and backend. I can implement the validation on both ends.

    With this in mind, this function is executed at the ``__init__``
    of *Series, and inside the ``graphics`` function.
    """
    if isinstance(obj, param.Parameterized):
        # do not consider private attributes
        allowed_keys = [
            t for t in obj.param.objects('existing')
            if t[0] != "_"
        ] + additional_keys
    else:
        allowed_keys = additional_keys

    allowed_keys = list(set(allowed_keys))
    print(allowed_keys)
    # allowed_keys = list(set(allowed_keys))
    kwargs = [k for k in kwargs if k[0] != "_"]
    user_provided_keys = set(kwargs).difference(exclude_keys)
    unused_keys = user_provided_keys.difference(allowed_keys)

    if len(unused_keys) > 0:
        t = type(obj).__name__
        msg = f"The following keyword arguments are unused by `{t}`.\n"
        for k in unused_keys:
            possible_match = find_closest_string(k, allowed_keys)
            msg += "* '%s'" % k
            msg += ": did you mean '%s'?\n" % possible_match
        warnings.warn(msg, stacklevel=2)
        # this "return" helps with tests
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
            # j+1 instead of j since previous_row and current_row are
            # one character longer than s2
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


def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to cartesian coordinates.

    Parameters
    ==========
        r :
            Radius.
        theta :
            Polar angle. Must be in [0, pi]. 0 is the north pole, pi/2 is the
            equator, pi is the south pole.
        phi :
            Azimuthal angle. Must be in [0, 2*pi].

    Returns
    =======
        x, y, z
    """
    if callable(r):
        np = import_module('numpy')
        x = lambda t, p: r(t, p) * np.sin(t) * np.cos(p)
        y = lambda t, p: r(t, p) * np.sin(t) * np.sin(p)
        z = lambda t, p: r(t, p) * np.cos(t)
    else:
        x = r * sin(theta) * cos(phi)
        y = r * sin(theta) * sin(phi)
        z = r * cos(theta)
    return x, y, z


def unwrap(angle, period=None):
    """Unwrap a phase angle to give a continuous curve

    Parameters
    ----------
    angle : array_like
        Array of angles to be unwrapped
    period : float, optional
        Period (defaults to `2*pi`)

    Returns
    -------
    angle_out : array_like
        Output array, with jumps of period/2 eliminated

    Examples
    --------
    >>> # Already continuous
    >>> theta1 = np.array([1.0, 1.5, 2.0, 2.5, 3.0]) * np.pi
    >>> theta2 = ct.unwrap(theta1)
    >>> theta2/np.pi                                          # doctest: +SKIP
    array([1. , 1.5, 2. , 2.5, 3. ])

    >>> # Wrapped, discontinuous
    >>> theta1 = np.array([1.0, 1.5, 0.0, 0.5, 1.0]) * np.pi
    >>> theta2 = ct.unwrap(theta1)
    >>> theta2/np.pi                                          # doctest: +SKIP
    array([1. , 1.5, 2. , 2.5, 3. ])

    Notes
    -----

    This function comes from the `control` package, specifically the
    `control.ctrlutil.py` module.

    """
    np = import_module('numpy')
    if period is None:
        period = 2 * np.pi
    dangle = np.diff(angle)
    dangle_desired = (dangle + period/2.) % period - period/2.
    correction = np.cumsum(dangle_desired - dangle)
    angle[1:] += correction
    return angle


def extract_solution(set_sol, n=10):
    """Extract numerical solutions from a set solution (computed by solveset,
    linsolve, nonlinsolve). Often, it is not trivial do get something useful
    out of them.

    Parameters
    ==========

    n : int, optional
        In order to replace ImageSet with FiniteSet, an iterator is created
        for each ImageSet contained in `set_sol`, starting from 0 up to `n`.
        Default value: 10.
    """
    images = set_sol.find(ImageSet)
    for im in images:
        it = iter(im)
        s = FiniteSet(*[next(it) for n in range(0, n)])
        set_sol = set_sol.subs(im, s)
    return set_sol


def is_number(t, allow_complex=True):
    if allow_complex:
        number_types = (NumberSymbol, Float, Integer, Rational,
            float, int, complex)
    else:
        number_types = (NumberSymbol, Float, Integer, Rational, float, int)
    return isinstance(t, number_types) or (isinstance(t, Expr) and t.is_number)


def tf_to_control(tf):
    """Convert a transfer function to a ``control.TransferFunction``.

    Parameters
    ==========
    tf :
        The transfer function's type can be:

        * an instance of :py:class:`sympy.physics.control.lti.TransferFunction`
          or :py:class:`sympy.physics.control.lti.TransferFunctionMatrix`
        * an instance of :py:class:`scipy.signal.TransferFunction`
        * a symbolic expression in rational form.
        * a tuple of two or three elements: ``(num, den, generator [opt])``.
          ``num, den`` can be symbolic expressions or list of coefficients.

    Returns
    =======
    tf : :py:class:`ct.TransferFunction`
    """
    ct = import_module("control")
    sp = import_module("scipy")
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})

    def _from_sympy_to_ct(num, den):
        fs = num.free_symbols.union(den.free_symbols)
        if len(fs) != 1:
            raise ValueError(
                "In order to convert a SymPy trasfer function to a "
                "``control`` transfer function, there must only be "
                "one free-symbol.\nReceived: %s" % fs
            )
        s = fs.pop()

        delays = tf_find_time_delay(num / den)
        if len(delays) > 0:
            raise ValueError(
                "The symbolic transfer function contains the following "
                "time delays: %s. "
                "Time delays are not supported by the ``control`` module. "
                "Consider applying a Padé approximation." % delays
            )

        n, d = [Poly(t, s).all_coeffs() for t in [num, den]]
        try:
            n = [float(t) for t in n]
            d = [float(t) for t in d]
        except TypeError as err:
            raise TypeError(
                str(err) + "\nYou are trying to convert a transfer function to "
                "``control.TransferFunction``. It appears like some of the "
                "coefficients are complex. At the time of coding this "
                "message, the ``control`` module doesn't support complex "
                "coefficents. You might still be able to achieve your goal "
                "by setting ``control=False`` in your function call."
            )
        return ct.tf(n, d)

    if ct and isinstance(tf, ct.TransferFunction):
        return tf

    if isinstance(tf, Expr):
        gen = tf.free_symbols.pop()
        tf = sm.control.TransferFunction.from_rational_expression(
            tf, gen)
        return _from_sympy_to_ct(tf.num, tf.den)

    elif isinstance(tf, sm.control.TransferFunction):
        return _from_sympy_to_ct(tf.num, tf.den)

    elif isinstance(tf, sm.control.TransferFunctionMatrix):
        num, den = [], []
        for i in range(tf.num_outputs):
            row_num, row_den = [], []
            for j in range(tf.num_inputs):
                tmp = _from_sympy_to_ct(tf[i, j].num, tf[i, j].den)
                row_num.append(list(tmp.num[0][0]))
                row_den.append(list(tmp.den[0][0]))
            num.append(row_num)
            den.append(row_den)
        return ct.tf(num, den)

    elif sp and isinstance(tf, sp.signal.TransferFunction):
        return ct.tf(tf.num, tf.den, dt=0 if tf.dt is None else tf.dt)

    elif isinstance(tf, (list, tuple)):
        tf = tf_to_sympy(tf)
        return _from_sympy_to_ct(tf.num, tf.den)

    else:
        raise TypeError(
            "Transfer function's type not recognized.\n" +
            "Received: type(tf) = %s\n" % type(tf) +
            "Expected: Expr or sympy.physics.control.TransferFunction or " +
            "sympy.physics.control.TransferFunctionMatrix"
        )


def tf_to_sympy(tf, var=None, skip_check_dt=False, params={}):
    """Convert a transfer function from the control module or from scipy.signal
    to a sympy ``TransferFunction`` or ``TransferFunctionMatrix``.

    Parameters
    ==========
    tf : control.TransferFunction, scipy.signal.TransferFunction

        * an instance of :py:class:`sympy.physics.control.lti.TransferFunction`
          or :py:class:`sympy.physics.control.lti.TransferFunctionMatrix`
        * an instance of :py:class:`control.TransferFunction`
        * an instance of :py:class:`scipy.signal.TransferFunction`
        * a symbolic expression in rational form.
        * a tuple of two or three elements: ``(num, den, generator [opt])``.
          ``num, den`` can be symbolic expressions or list of coefficients.

    var : Symbol or None
        The s-variable (or z-variable) when ``tf`` is a symbolic expression.
        If not provided, it will be automatically selected.
    skip_check_dt : bool
        If True, don't raise a warning about sympy not supporting discrete-time
        systems.
    params : dict
        A dictionary whose keys are symbols.

    Returns
    =======
    tf : TransferFunction or TransferFunctionMatrix
    """
    ct = import_module("control")
    sp = import_module("scipy")
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})

    gen = Symbol("z") if is_discrete_time(tf) else Symbol("s")
    TransferFunction = sm.control.lti.TransferFunction
    TransferFunctionMatrix = sm.control.lti.TransferFunctionMatrix
    Series = sm.control.lti.Series
    Parallel = sm.control.lti.Parallel

    def _check_dt(system):
        if system.dt and (not skip_check_dt):
            warnings.warn(
                "At the time of writing this message, SymPy doesn't "
                "implement discrete-time transfer functions. Returning "
                "a continuous-time transfer function."
            )

    if isinstance(tf, (TransferFunction, TransferFunctionMatrix)):
        return tf

    elif isinstance(tf, Expr):
        if var is None:
            fs = list(tf.free_symbols.difference(params.keys()))
            if len(fs) > 1:
                warnings.warn(
                    "Multiple free symbols found in transfer function: %s. "
                    "Selecting the first as the s-variable "
                    "(or z-variable). Use the ``var=`` keyword argument "
                    "to specify the appropriate symbol." % fs
                )
            var = fs[0] if len(tf.free_symbols) > 0 else Symbol("s")
        return TransferFunction.from_rational_expression(tf, var)

    elif isinstance(tf, (Series, Parallel)):
        return tf.doit()

    if (ct is not None) and isinstance(tf, ct.TransferFunction):
        if (tf.ninputs == 1) and (tf.noutputs == 1):
            n, d = tf.num[0][0], tf.den[0][0]
            n = Poly.from_list(n, gen).as_expr()
            d = Poly.from_list(d, gen).as_expr()
            _check_dt(tf)
            return TransferFunction(n, d, gen)
        rows = []
        for o in range(tf.noutputs):
            row = []
            for i in range(tf.ninputs):
                n = tf.num[o][i]
                d = tf.den[o][i]
                new_tf = tf_to_sympy(ct.tf(n, d, dt=tf.dt))
                row.append(new_tf)
            rows.append(row)
        return TransferFunctionMatrix(rows)

    elif (sp is not None) and isinstance(tf, sp.signal.TransferFunction):
        n = Poly.from_list(tf.num, gen).as_expr()
        d = Poly.from_list(tf.den, gen).as_expr()
        _check_dt(tf)
        return TransferFunction(n, d, gen)

    if isinstance(tf, (list, tuple)):
        powers = lambda e, s: [t * s**(len(e) - (k + 1))
            for k, t in enumerate(e)]
        if len(tf) == 2:
            num, den = tf
            if all(isinstance(e, Expr) for e in tf):
                gen = Tuple(num, den).free_symbols.difference(params.keys()).pop()
            else:
                num = sum(powers(num, gen))
                den = sum(powers(den, gen))
            return TransferFunction(num, den, gen)
        elif len(tf) == 3:
            num, den, gen = tf
            if not all(isinstance(e, Expr) for e in tf):
                num = sum(powers(num, gen))
                den = sum(powers(den, gen))
            return TransferFunction(num, den, gen)
        else:
            raise ValueError(
                "If a tuple/list is provided, it must have "
                "two or three elements: (num, den, free_symbol [opt]). "
                f"Received len(system) = {len(tf)}, system = {tf}"
            )

    else:
        raise TypeError(
            "Transfer function's type not recognized.\n" +
            "Received: type(tf) = %s\n" % type(tf) +
            "Expected: Expr or sympy.physics.control.TransferFunction"
        )


def _get_initial_params(params):
    """Extract the initial values of parameters from the ``params`` dictionary
    used on interactive-widget plots.
    """
    return {
        k: (v[0] if hasattr(v, "__iter__") else v) for k, v in params.items()
    }


def is_discrete_time(system):
    """Verify if ``system`` is a discrete-time control system.
    """
    ct = import_module("control")
    sp = import_module("scipy")
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})

    if isinstance(system, sm.control.lti.SISOLinearTimeInvariant):
        return False
    if (sp is not None) and isinstance(system, sp.signal.TransferFunction):
        return False if system.dt is None else True
    if (ct is not None) and isinstance(system, ct.TransferFunction):
        return system.isdtime()
    return False


def tf_find_time_delay(tf, var=None):
    """Find time delays contained in a symbolic TransferFunction.
    """
    sympy = import_module("sympy")

    if isinstance(tf, Expr):
        tf = tf_to_sympy(tf, var=var)

    if not isinstance(tf, sympy.physics.control.TransferFunction):
        raise TypeError(
            "``tf_find_time_delay`` only works with instances of "
            "sympy.physics.control.lti.TransferFunction."
        )

    num, den, s = tf.args
    exp_num = [t for t in num.find(exp) if t.has(s)]
    exp_den = [t for t in den.find(exp) if t.has(s)]
    return exp_num + exp_den


def is_siso(system):
    """Check if a control system is SISO or not.
    """
    ct = import_module("control")
    sp = import_module("scipy")
    sm = import_module("sympy.physics", import_kwargs={'fromlist':['control']})
    if isinstance(system, sm.control.lti.SISOLinearTimeInvariant):
        return True
    if sp and isinstance(system, sp.signal.TransferFunction):
        return True
    if (
        ct and isinstance(system, ct.TransferFunction) and
        (system.ninputs == 1) and (system.noutputs == 1)
    ):
        return True
    if isinstance(system, Expr):
        return True
    return False



def _aggregate_parameters(params, series):
    """Loop over data series to extract the `params` dictionaries provided by
    the user. This is necessary when dealing with the ``graphics`` module.

    Parameters
    ==========
    params : dict
        Whatever was provided by the user in the main function call (be it
        plot(), plot_paramentric(), ..., graphics())
    series : list
        Data series of the current interactive widget plot.

    Returns
    =======
    params : dict
    """
    if params is None:
        params = {}
    # if len(params) == 0:
    #     # this is the case when an interactive widget plot is build with
    #     # the `graphics` interface.
    for s in series:
        if s.is_interactive:
            # use s._original_params instead of s.params in order to
            # keep track of multi-values widgets
            params.update(s._original_params)
    if len(params) == 0:
        raise ValueError(
            "In order to create an interactive plot, "
            "the `params` dictionary must be provided.")
    return params


def get_environment():
    """Find which environment is used to run the code.

    Returns
    =======
    mode : int
        0 - the code is running on Jupyter Notebook or qtconsole
        1 - terminal running IPython
        2 - other type (?)
        3 - probably standard Python interpreter

    References
    ==========

    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return 0  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return 1  # Terminal running IPython
        else:
            return 2  # Other type (?)
    except NameError:
        return 3  # Probably standard Python interpreter


def _correct_shape(a, b):
    """Convert ``a`` to a np.ndarray of the same shape of ``b``.

    Parameters
    ==========
    a : int, float, complex, np.ndarray
        Usually, this is the result of a numerical evaluation of a
        symbolic expression. Even if a discretized domain was used to
        evaluate the function, the result can be a scalar (int, float,
        complex).
    b : np.ndarray
        It represents the correct shape that ``a`` should have.

    Returns
    =======
    new_a : np.ndarray
        An array with the correct shape.
    """
    np = import_module('numpy')

    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.shape != b.shape:
        if a.shape == ():
            a = a * np.ones_like(b)
        else:
            a = a.reshape(b.shape)
    return a
