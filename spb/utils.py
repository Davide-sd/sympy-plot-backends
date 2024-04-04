from spb.defaults import cfg
from sympy import (
    Tuple, sympify, Expr, Dummy, sin, cos, Symbol, Indexed, ImageSet,
    FiniteSet, Basic, Float, Integer, Rational, Poly, fraction, exp,
    NumberSymbol
)
from sympy.vector import BaseScalar
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction
from sympy.external import import_module
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

    free = set().union(*[e.atoms(Indexed) for e in exprs])
    free = free.union(*[e.atoms(AppliedUndef) for e in exprs])
    if len(free) > 0:
        return free
    return set().union(*[e.free_symbols for e in exprs])


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
       developed even further. Moreover, Python will raise an error everytime
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
        for c in inspect.getmro(type(s)):
            if hasattr(c, "_allowed_keys"):
                allowed_keys = allowed_keys.union(getattr(c, "_allowed_keys"))
    # some functions injects the following keyword arguments that will be
    # processed by other functions before instantion of Series and Backend.
    allowed_keys = allowed_keys.union([
        "abs", "absarg", "arg", "real", "imag", "force_real_eval",
        "slice", "threed", "sum_bound", "n",
        "phaseres", "is_point", "is_polar", "label",
        "wireframe", "wf_n1", "wf_n2", "wf_npoints", "wf_rendering_kw",
        "dots", "show_in_legend", "fig", "ax",
        "sgrid", "ngrid", "zgrid", "m_circles"
    ])
    # from interactive sub-module
    allowed_keys = allowed_keys.union([
        "layout", "ncols", "use_latex", "throttled", "servable",
        "custom_css", "pane_kw", "is_iplot", "series", "template"
    ])
    # from graphics.control sub-module
    allowed_keys = allowed_keys.union([
        "omega_limits", "input", "output"
    ])
    user_provided_keys = set(kwargs.keys())
    unused_keys = user_provided_keys.difference(allowed_keys)
    if len(unused_keys) > 0:
        msg = "The following keyword arguments are unused.\n"
        for k in unused_keys:
            possible_match = find_closest_string(k, allowed_keys)
            msg += "* '%s'" % k
            msg += ": did you mean '%s'?\n" % possible_match
        warnings.warn(msg, stacklevel=3)
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


def tf_to_control(tf, gen=None):
    """Convert a transfer function to a ``control.TransferFunction``.

    Parameters
    ==========
    tf :
        The transfer function's type can be:

        * Expr: it must be a rational expression.
        * sympy.physics.control.TransferFunction
        * sympy.physics.control.TransferFunctionMatrix
        * scipy.signal.TransferFunction

    gen : Symbol or None
        The s or z variable. It is only used if ``tf`` is a symbolic
        expression containing multiple free symbols.

    Returns
    =======
    tf : ct.TransferFunction
    """
    ct = import_module("control")
    sp = import_module("scipy")
    sympy = import_module("sympy")

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

    if isinstance(tf, ct.TransferFunction):
        return tf
    if isinstance(tf, Expr):
        if gen is None:
            gen = tf.free_symbols.pop()
        tf = sympy.physics.control.TransferFunction.from_rational_expression(
            tf, gen)
        return _from_sympy_to_ct(tf.num, tf.den)
    elif isinstance(tf, sympy.physics.control.TransferFunction):
        return _from_sympy_to_ct(tf.num, tf.den)
    elif isinstance(tf, sympy.physics.control.TransferFunctionMatrix):
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
    elif isinstance(tf, sp.signal.TransferFunction):
        return ct.tf(tf.num, tf.den, dt=0 if tf.dt is None else tf.dt)
    else:
        raise TypeError(
            "Transfer function's type not recognized.\n" +
            "Received: type(tf) = %s\n" % type(tf) +
            "Expected: Expr or sympy.physics.control.TransferFunction or " +
            "sympy.physics.control.TransferFunctionMatrix"
        )


def tf_to_sympy(tf, var=None, skip_check_dt=False):
    """Convert a transfer function from the control module or from scipy.signal
    to a sympy ``TransferFunction`` or ``TransferFunctionMatrix``.

    Parameters
    ==========
    tf : control.TransferFunction, scipy.signal.TransferFunction
    var : Symbol or None
        The s-variable (or z-variable).
    skip_check_dt : bool
        If True, don't raise a warning about sympy not supporting discrete-time
        systems.

    Returns
    =======
    tf : TransferFunction or TransferFunctionMatrix
    """
    ct = import_module("control")
    sp = import_module("scipy")
    sympy = import_module("sympy")

    gen = Symbol("z") if is_discrete_time(tf) else Symbol("s")
    TransferFunction = sympy.physics.control.lti.TransferFunction
    TransferFunctionMatrix = sympy.physics.control.lti.TransferFunctionMatrix
    Series = sympy.physics.control.lti.Series
    Parallel = sympy.physics.control.lti.Parallel

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
            fs = list(tf.free_symbols)
            if len(fs) > 1:
                warnings.warn(
                    "Multiple free symbols found in transfer function: %s. "
                    "Selecting the first as the s-variable "
                    "(or z-variable). Use the ``var=`` keyword argument "
                    "to specify the appropriate symbol." % fs
                )
                var = fs[0]
            else:
                var = tf.free_symbols.pop() if len(tf.free_symbols) > 0 else Symbol("s")
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
    sy = import_module("sympy")

    if isinstance(system, sy.physics.control.lti.SISOLinearTimeInvariant):
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
