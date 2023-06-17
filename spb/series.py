from inspect import signature
from spb.ccomplex.wegert import wegert
from spb.defaults import cfg
from spb.utils import prange, _get_free_symbols, spherical_to_cartesian
from sympy import (
    latex, Tuple, arity, symbols, sympify, solve, Expr, lambdify,
    Equality, Ne, GreaterThan, LessThan, StrictLessThan, StrictGreaterThan,
    Plane, Polygon, Circle, Ellipse, Segment, Ray, Curve, Point2D, Point3D,
    atan2, floor, ceiling, Sum, Product, Symbol, frac, im, re, zeta, Poly
)
from sympy.geometry.entity import GeometryEntity
from sympy.geometry.line import LinearEntity2D, LinearEntity3D
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction
from sympy.plotting.intervalmath import interval
from sympy.external import import_module
from sympy.physics.control.lti import TransferFunction
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.precedence import precedence
from sympy.core.sorting import default_sort_key
from sympy.utilities.exceptions import sympy_deprecation_warning
import warnings

class IntervalMathPrinter(PythonCodePrinter):
    """A printer to be used inside `plot_implicit` when `adaptive=True`,
    in which case the interval arithmetic module is going to be used, which
    requires the following edits.
    """
    def _print_And(self, expr):
        PREC = precedence(expr)
        return " & ".join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Or(self, expr):
        PREC = precedence(expr)
        return " | ".join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))


def _adaptive_eval(wrapper_func, free_symbols, expr, bounds, *args,
        modules=None, adaptive_goal=None, loss_fn=None):
    """Numerical evaluation of a symbolic expression with an adaptive
    algorithm [#fn1]_.

    Note: this is an experimental function, as such it is prone to changes.
    Please, do not use it in your code.

    Parameters
    ==========

    wrapper_func : callable
        The function to be evaluated, which will return any number of
        elements, depending on the computation to be done. The signature
        must be as follow: ``wrapper_func(f, *args)``
        where ``f`` is the lambda function representing the symbolic
        expression; ``*args`` is a list of arguments necessary to perform
        the evaluation.

    free_symbols : tuple or list
        The free symbols associated to ``expr``.

    expr : Expr
        The symbolic expression to be evaluated.

    bounds : tuple (min, max) or list of tuples
        The bounds for the numerical evaluation. Let `f(x)` be the function
        to be evaluated, then `x` will assume values between [min, max].
        For multivariate functions there is a correspondance between the
        symbols in ``free_symbols`` and the tuples in ``bounds``.

    args :
        The necessary arguments to perform the evaluation.

    modules : str or None
        The evaluation module. Refer to ``lambdify`` for a list of possible
        values. If ``None``, the evaluation will be done with Numpy/Scipy.

    adaptive_goal : callable, int, float or None
        Controls the "smoothness" of the evaluation. Possible values:

        * ``None`` (default):  it will use the following goal:
          ``lambda l: l.loss() < 0.01``
        * number (int or float). The lower the number, the more
          evaluation points. This number will be used in the following goal:
          `lambda l: l.loss() < number`
        * callable: a function requiring one input element, the learner. It
          must return a float number.

    loss_fn : callable or None
        The loss function to be used by the learner. Possible values:

        * ``None`` (default): it will use the ``default_loss`` from the
          adaptive module.
        * callable : look at adaptive.learner.learner1D or
          adaptive.learner.learnerND to find more loss functions.

    Returns
    =======

    data : np.ndarray
        A Numpy array containing the evaluation results. The shape is [NxM],
        where N is the random number of evaluation points and M is the sum
        between the number of free symbols and the number of elements
        returned by ``wrapper_func``.
        No matter the evaluation ``modules``, the array type is going to be
        complex.

    References
    ==========

    .. [#fn1] `adaptive module <https://github.com/python-adaptive/adaptive`_.
    """
    np = import_module('numpy')
    adaptive = import_module(
        'adaptive',
        import_kwargs={'fromlist': ['runner', 'learner']},
        min_module_version='0.12.0',
        warn_not_installed=True)
    simple = adaptive.runner.simple
    Learner1D = adaptive.learner.learner1D.Learner1D
    LearnerND = adaptive.learner.learnerND.LearnerND
    default_loss_1d = adaptive.learner.learner1D.default_loss
    default_loss_nd = adaptive.learner.learnerND.default_loss
    from functools import partial

    if not callable(expr):
        # expr is a single symbolic expressions or a tuple of symb expressions
        one_d = hasattr(free_symbols, "__iter__") and (len(free_symbols) == 1)
    else:
        # expr is a user-provided lambda function
        one_d = len(signature(expr).parameters) == 1

    goal = lambda l: l.loss() < 0.01
    if adaptive_goal is not None:
        if isinstance(adaptive_goal, (int, float)):
            goal = lambda l: l.loss() < adaptive_goal
        if callable(adaptive_goal):
            goal = adaptive_goal

    lf = default_loss_1d if one_d else default_loss_nd
    if loss_fn is not None:
        lf = loss_fn
    k = "loss_per_interval" if one_d else "loss_per_simplex"
    d = {k: lf}
    Learner = Learner1D if one_d else LearnerND

    if not callable(expr):
        # expr is a single symbolic expressions or a tuple of symb expressions
        try:
            # TODO: set cse=True once this issue is solved:
            # https://github.com/sympy/sympy/issues/24246
            f = lambdify(free_symbols, expr, modules=modules, cse=False)
            learner = Learner(partial(wrapper_func, f, *args),
                bounds=bounds, **d)
            simple(learner, goal)
        except Exception as err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not modules else modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
            f = lambdify(free_symbols, expr, modules="sympy", cse=False)
            learner = Learner(partial(wrapper_func, f, *args),
                bounds=bounds, **d)
            simple(learner, goal)
    else:
        # expr is a user-provided lambda function
        learner = Learner(partial(wrapper_func,expr, *args), bounds=bounds, **d)
        simple(learner, goal)

    if one_d:
        return learner.to_numpy()

    # For multivariate functions, create a meshgrid where to interpolate the
    # results. Taken from adaptive.learner.learnerND.plot
    x, y = learner._bbox
    scale_factor = np.product(np.diag(learner._transform))
    a_sq = np.sqrt(np.min(learner.tri.volumes()) * scale_factor)
    n = max(10, int(0.658 / a_sq) * 2)
    xs = ys = np.linspace(0, 1, n)
    xs = xs * (x[1] - x[0]) + x[0]
    ys = ys * (y[1] - y[0]) + y[0]
    z = learner._ip()(xs[:, None], ys[None, :]).squeeze()
    xs, ys = np.meshgrid(xs, ys)
    return xs, ys, np.rot90(z)


def _uniform_eval(f1, f2, *args, modules=None,
    force_real_eval=False, has_sum=False):
    """
    Note: this is an experimental function, as such it is prone to changes.
    Please, do not use it in your code.
    """
    np = import_module('numpy')

    def wrapper_func(func, *args):
        try:
            return complex(func(*args))
        except (ZeroDivisionError, OverflowError):
            return complex(np.nan, np.nan)

    # NOTE: np.vectorize is much slower than numpy vectorized operations.
    # However, this modules must be able to evaluate functions also with
    # mpmath or sympy.
    wrapper_func = np.vectorize(wrapper_func, otypes=[complex])

    skip_fast_eval = False
    if modules == "sympy":
        skip_fast_eval = True

    try:
        if skip_fast_eval:
            raise Exception
        return wrapper_func(f1, *args)
    except Exception as err:
        if f2 is None:
            raise RuntimeError(
                "Impossible to evaluate the provided numerical function "
                "because the following exception was raised:\n"
                "{}: {}".format(type(err).__name__, err))
        if not skip_fast_eval:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not modules else modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
        return wrapper_func(f2, *args)


def _get_wrapper_for_expr(ret):
    wrapper = "%s"
    if ret == "real":
        wrapper = "re(%s)"
    elif ret == "imag":
        wrapper = "im(%s)"
    elif ret == "abs":
        wrapper = "abs(%s)"
    elif ret == "arg":
        wrapper = "arg(%s)"
    return wrapper


class BaseSeries:
    """Base class for the data objects containing stuff to be plotted.

    Notes
    =====

    The backend should check if it supports the data series that it's given.
    It's the backend responsibility to know how to use the data series that
    it's given.
    """

    # Some flags follow. The rationale for using flags instead of checking
    # base classes is that setting multiple flags is simpler than multiple
    # inheritance.

    is_2Dline = False

    is_3Dline = False

    is_3Dsurface = False

    is_contour = False

    is_implicit = False
    # Both contour and implicit series uses colormap, but they are different.
    # Hence, a different attribute

    is_parametric = False

    is_interactive = False
    # An interactive series can update its data.

    is_vector = False
    is_2Dvector = False
    is_3Dvector = False
    is_slice = False
    # Represents a 2D or 3D vector

    is_complex = False
    # Represent a complex expression
    is_domain_coloring = False

    is_geometry = False
    # If True, it represents an object of the sympy.geometry module

    is_generic = False
    # Implement back-compatibility with sympy.plotting <= 1.11
    # Please, read NOTE section on GenericDataSeries

    _allowed_keys = []
    # contains a list of keyword arguments supported by the series. It will be
    # used to validate the user-provided keyword arguments.

    _N = 100
    # default number of discretization points for uniform sampling. Each
    # subclass can set its number.

    def __init__(self, *args, **kwargs):
        kwargs = _set_discretization_points(kwargs.copy(), type(self))
        # discretize the domain using only integer numbers
        self.only_integers = kwargs.get("only_integers", False)
        # represents the evaluation modules to be used by lambdify
        self.modules = kwargs.get("modules", None)
        # plot functions might create data series that might not be useful to
        # be shown on the legend, for example wireframe lines on 3D plots.
        self.show_in_legend = kwargs.get("show_in_legend", True)
        # line and surface series can show data with a colormap, hence a
        # colorbar is essential to understand the data. However, sometime it
        # is useful to hide it on series-by-series base. The following keyword
        # controls wheter the series should show a colorbar or not.
        self.colorbar = kwargs.get("colorbar", True)
        # Some series might use a colormap as default coloring. Setting this
        # attribute to False will inform the backends to use solid color.
        self.use_cm = kwargs.get("use_cm", False)
        # If True, the backend will attempt to render it on a polar-projection
        # axis, or using a polar discretization if a 3D plot is requested
        self.is_polar = kwargs.get("is_polar", False)
        # If True, the rendering will use points, not lines.
        self.is_point = kwargs.get("is_point", False)

        self._label = self._latex_label = ""
        self._ranges = []
        self._n = [
            int(kwargs.get("n1", self._N)),
            int(kwargs.get("n2", self._N)),
            int(kwargs.get("n3", self._N))
        ]
        self._scales = [
            kwargs.get("xscale", "linear"),
            kwargs.get("yscale", "linear"),
            kwargs.get("zscale", "linear")
        ]

        self._params = kwargs.get("params", dict())
        if not isinstance(self._params, dict):
            raise TypeError("`params` must be a dictionary mapping symbols "
                "to numeric values.")
        if len(self._params) > 0:
            self.is_interactive = True

        self.rendering_kw = kwargs.get("rendering_kw", dict())

        # numerical transformation functions to be applied to the output data
        self._tx = kwargs.get("tx", None)
        self._ty = kwargs.get("ty", None)
        self._tz = kwargs.get("tz", None)
        self._tp = kwargs.get("tp", None)
        if not all(callable(t) or (t is None) for t in
            [self._tx, self._ty, self._tz, self._tp]):
            raise TypeError("`tx`, `ty`, `tz`, `tp` must be functions.")

        # list of numerical functions representing the expressions to evaluate
        self._functions = []
        # signature of for the numerical functions
        self._signature = []
        # some expressions don't like to be evaluated over complex data.
        # if that's the case, set this to True
        self._force_real_eval = kwargs.get("force_real_eval", None)
        # eventually it will contain a dictionary with the discretized ranges
        self._discretized_domain = None
        # wheter the series contains any interactive range
        self._interactive_ranges = False
        # NOTE: consider a generic summation, for example:
        #   s = Sum(cos(pi * x), (x, 1, y))
        # This gets lambdified to something:
        #   sum(cos(pi*x) for x in range(1, y+1))
        # Hence, y needs to be an integer, otherwise it raises:
        #   TypeError: 'complex' object cannot be interpreted as an integer
        # This list will contains symbols that are upper bound to summations
        # or products
        self._needs_to_be_int = []
        # subclasses that requires this attribute must set an appropriate
        # default value.
        self.color_func = None
        # NOTE: color_func usually receives numerical functions that are going
        # to be evaluated over the coordinates of the computed points (or the
        # discretized meshes).
        # However, if an expression is given to color_func, then it will be
        # lambdified with symbols in self._signature, and it will be evaluated
        # with the same data used to evaluate the plotted expression.
        self._eval_color_func_with_signature = False

    def _block_lambda_functions(self, *exprs):
        if any(callable(e) for e in exprs):
            raise TypeError(type(self).__name__ + " requires a symbolic "
                "expression.")

    def _check_fs(self):
        """ Checks if there are enogh parameters and free symbols.
        """
        exprs, ranges = self.expr, self.ranges
        params, label = self.params, self.label
        exprs = exprs if hasattr(exprs, "__iter__") else [exprs]
        if any(callable(e) for e in exprs):
            return

        # from the expression's free symbols, remove the ones used in
        # the parameters and the ranges
        fs = _get_free_symbols(exprs)
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

        # verify that all symbols are known (they either represent plotting
        # ranges or parameters)
        range_symbols = [r[0] for r in ranges]
        for r in ranges:
            fs = set().union(*[e.free_symbols for e in r[1:]])
            if any(t in fs for t in range_symbols):
                raise ValueError("Range symbols can't be included into "
                    "minimum and maximum of a range. "
                    "Received range: %s" % str(r))
            if len(fs) > 0:
                self._interactive_ranges = True
            remaining_fs = fs.difference(params.keys())
            if len(remaining_fs) > 0:
                raise ValueError(
                    "Unkown symbols found in plotting range: %s. " % (r,) +
                    "Are the following parameters? %s" % remaining_fs)

    def _create_lambda_func(self):
        """Create the lambda functions to be used by the uniform meshing
        strategy.
        """
        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        if not any(callable(e) for e in exprs):
            fs = _get_free_symbols(exprs)
            self._signature = sorted(fs, key=lambda t: t.name)

            # Generate a list of lambda functions, two for each expression:
            # 1. the default one.
            # 2. the backup one, in case of failures with the default one.
            self._functions = []
            for e in exprs:
                # TODO: set cse=True once this issue is solved:
                # https://github.com/sympy/sympy/issues/24246
                self._functions.append([
                    lambdify(self._signature, e, modules=self.modules),
                    lambdify(self._signature, e, modules="sympy", dummify=True),
                ])
        else:
            self._signature = sorted([r[0] for r in self.ranges], key=lambda t: t.name)
            self._functions = [(e, None) for e in exprs]

        # deal with symbolic color_func
        if isinstance(self.color_func, Expr):
            self.color_func = lambdify(self._signature, self.color_func)
            self._eval_color_func_with_signature = True

    def _update_range_value(self, t):
        if not self._interactive_ranges:
            return complex(t)
        return complex(t.subs(self.params))

    def _create_discretized_domain(self):
        """Discretize the ranges for uniform meshing strategy.
        """
        # NOTE: the goal is to create a dictionary stored in
        # self._discretized_domain, mapping symbols to a numpy array
        # representing the discretization
        discr_symbols = []
        discretizations = []

        # create a 1D discretization
        for i, r in enumerate(self.ranges):
            discr_symbols.append(r[0])
            c_start = self._update_range_value(r[1])
            c_end = self._update_range_value(r[2])
            start = c_start.real if c_start.imag == c_end.imag == 0 else c_start
            end = c_end.real if c_start.imag == c_end.imag == 0 else c_end
            needs_integer_discr = self.only_integers or (r[0] in self._needs_to_be_int)
            d = BaseSeries._discretize(start, end, self.n[i],
                scale=self.scales[i],
                only_integers=needs_integer_discr)

            if ((not self._force_real_eval) and (not needs_integer_discr) and
                (d.dtype != "complex")):
                d = d + 1j * c_start.imag

            if needs_integer_discr:
                d = d.astype(int)

            discretizations.append(d)

        # create 2D or 3D
        self._create_discretized_domain_helper(discr_symbols, discretizations)

    def _create_discretized_domain_helper(self, discr_symbols, discretizations):
        """Create 2D or 3D discretized grids.

        Subclasses should override this method in order to implement a
        different behaviour.
        """
        np = import_module('numpy')

        # discretization suitable for 2D line plots, 3D surface plots,
        # contours plots, vector plots
        # NOTE: why indexing='ij'? Because it produces consistent results with
        # np.mgrid. This is important as Mayavi requires this indexing
        # to correctly compute 3D streamlines. VTK is able to compute them
        # nonetheless, but it produces "strange" results with "voids" into the
        # discretization volume. This indexing solves the problem.
        # Also note that matplotlib 2D streamlines requires indexing='xy'.
        indexing = "xy"
        if self.is_3Dvector or (self.is_3Dsurface and self.is_implicit):
            indexing = "ij"
        meshes = np.meshgrid(*discretizations, indexing=indexing)
        self._discretized_domain = {k: v for k, v in zip(discr_symbols, meshes)}

    def _evaluate(self, cast_to_real=True):
        """Evaluation of the symbolic expression (or expressions) with the
        uniform meshing strategy, based on current values of the parameters.
        """
        np = import_module('numpy')

        # create lambda functions
        if not self._functions:
            self._create_lambda_func()
        # create (or update) the discretized domain
        if (not self._discretized_domain) or self._interactive_ranges:
            self._create_discretized_domain()
        # ensure that discretized domains are returned with the proper order
        discr = [self._discretized_domain[s[0]] for s in self.ranges]

        args = self._aggregate_args()

        results = []
        for f in self._functions:
            r = _uniform_eval(*f, *args)
            # the evaluation might produce an int/float. Need this correction.
            r = self._correct_shape(np.array(r), discr[0])
            # sometime the evaluation is performed over arrays of type object.
            # hence, `result` might be of type object, which don't work well
            # with numpy real and imag functions.
            r = r.astype(complex)
            results.append(r)

        if cast_to_real:
            discr = [np.real(d.astype(complex)) for d in discr]
        return [*discr, *results]

    def _aggregate_args(self):
        args = []
        for s in self._signature:
            if s in self._params.keys():
                args.append(
                    int(self._params[s]) if s in self._needs_to_be_int else
                    self._params[s] if self._force_real_eval
                    else complex(self._params[s]))
            else:
                args.append(self._discretized_domain[s])
        return args

    @property
    def expr(self):
        """Set the expression (or expressions) of the series."""
        return self._expr

    @expr.setter
    def expr(self, e):
        """Set the expression (or expressions) of the series."""
        is_iter = hasattr(e, "__iter__")
        is_callable = callable(e) if not is_iter else any(callable(t) for t in e)
        if is_callable:
            self._expr = e
        else:
            self._expr = sympify(e) if not is_iter else Tuple(*e)
            s = set()
            for e in self._expr.atoms(Sum, Product):
                for a in e.args[1:]:
                    if isinstance(a[-1], Symbol):
                        s.add(a[-1])
            self._needs_to_be_int = list(s)

            # list of sympy functions that when lambdified, the corresponding
            # numpy functions don't like complex-type arguments
            pf = [ceiling, floor, atan2, frac, zeta]
            if self._force_real_eval is not True:
                check_res = [self._expr.has(f) for f in pf]
                self._force_real_eval = any(check_res)
                if self._force_real_eval and ((self.modules is None) or
                    (isinstance(self.modules, str) and "numpy" in self.modules)):
                    funcs = [f for f, c in zip(pf, check_res) if c]
                    warnings.warn("NumPy is unable to evaluate with complex "
                        "numbers some of the functions included in this "
                        "symbolic expression: %s. " % funcs +
                        "Hence, the evaluation will use real numbers. "
                        "If you believe the resulting plot is incorrect, "
                        "change the evaluation module by setting the "
                        "`modules` keyword argument.")
            if self._functions:
                # update lambda functions
                self._create_lambda_func()


    def get_expr(self):
        """Set the expression (or expressions) of the series."""
        warnings.warn("This method is deprecated and will be remove in the "
            "future. Use the `expr` attribute instead.")
        return self.expr

    @property
    def is_3D(self):
        flags3D = [self.is_3Dline, self.is_3Dsurface, self.is_3Dvector]
        return any(flags3D)

    @property
    def is_line(self):
        flagslines = [self.is_2Dline, self.is_3Dline]
        return any(flagslines)

    def _line_surface_color(self, prop, val):
        """This method enables back-compatibility with old sympy.plotting"""
        if val:
            if callable(val) or isinstance(val, Expr):
                use = "color_func"
            else:
                use = "rendering_kw"
            # TODO: follow sympy doc procedure to create this deprecation
            sympy_deprecation_warning(
                f"`{prop[1:]}` is deprecated. Use `{use}` instead.",
                deprecated_since_version="1.12",
                active_deprecations_target='---')
        # NOTE: color_func is set inside the init method of the series.
        # If line_color/surface_color is not a callable, then color_func will
        # be set to None.
        setattr(self, prop, val)
        if callable(val) or isinstance(val, Expr):
            self.color_func = val
            setattr(self, prop, None)
        elif val is not None:
            self.color_func = None

    @property
    def line_color(self):
        return self._line_color

    @line_color.setter
    def line_color(self, val):
        self._line_surface_color("_line_color", val)

    @property
    def n(self):
        """Returns a list [n1, n2, n3] of numbers of discratization points.
        """
        return self._n

    @n.setter
    def n(self, v):
        """Set the numbers of discretization points. ``v`` must be an int or
        a list.

        Let ``s`` be a series. Then:

        * to set the number of discretization points along the x direction (or
          first parameter): ``s.n = 10``
        * to set the number of discretization points along the x and y
          directions (or first and second parameters): ``s.n = [10, 15]``
        * to set the number of discretization points along the x, y and z
          directions: ``s.n = [10, 15, 20]``

        Note that the following is highly unreccomended, because it prevents
        the execution of necessary code in order to keep updated data:
        ``s.n[1] = 15``
        """
        if not hasattr(v, "__iter__"):
            self._n[0] = v
        else:
            self._n[:len(v)] = v
        if self._discretized_domain:
            # update the discretized domain
            self._create_discretized_domain()

    @property
    def params(self):
        """Get or set the current parameters dictionary.

        Parameters
        ==========

        p : dict

            * key: symbol associated to the parameter
            * val: the numeric value
        """
        return self._params

    @params.setter
    def params(self, p):
        self._params = p

    def _post_init(self):
        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        if any(callable(e) for e in exprs) and self.params:
            raise TypeError("`params` was provided, hence an interactive plot "
                "is expected. However, interactive plots do not support "
                "user-provided numerical functions.")

        # if the expressions is a lambda function and no label has been
        # provided, then its better to do the following in order to avoid
        # suprises on  the backend
        if any(callable(e) for e in exprs):
            if self._label == str(self.expr):
                self.label = ""

        self._check_fs()

        if hasattr(self, "adaptive") and self.adaptive and self.params:
            warnings.warn("`params` was provided, hence an interactive plot "
                "is expected. However, interactive plots do not support "
                "adaptive evaluation. Automatically switched to "
                "adaptive=False.")
            self.adaptive = False

    @property
    def scales(self):
        return self._scales

    @scales.setter
    def scales(self, v):
        if isinstance(v, str):
            self._scales[0] = v
        else:
            self._scales[:len(v)] = v

    @property
    def surface_color(self):
        return self._surface_color

    @surface_color.setter
    def surface_color(self, val):
        self._line_surface_color("_surface_color", val)

    @property
    def rendering_kw(self):
        return self._rendering_kw

    @rendering_kw.setter
    def rendering_kw(self, kwargs):
        if isinstance(kwargs, dict):
            self._rendering_kw = kwargs
        else:
            self._rendering_kw = dict()
            if kwargs is not None:
                warnings.warn(
                    "`rendering_kw` must be a dictionary, instead an "
                    "object of type %s was received. " % type(kwargs) +
                    "Automatically setting `rendering_kw` to an empty "
                    "dictionary")

    @staticmethod
    def _discretize(start, end, N, scale="linear", only_integers=False):
        """Discretize a 1D domain.

        Returns
        =======

        domain : np.ndarray with dtype=float or complex
            The domain's dtype will be float or complex (depending on the
            type of start/end) even if only_integers=True. It is left for
            the downstream code to perform further casting, if necessary.
        """
        np = import_module('numpy')

        if only_integers is True:
            start, end = int(start), int(end)
            N = end - start + 1

        if scale == "linear":
            return np.linspace(start, end, N)
        return np.geomspace(start, end, N)

    @staticmethod
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

    def eval_color_func(self, *args):
        """Evaluate the color function.

        Parameters
        ==========

        args : tuple
            Arguments to be passed to the coloring function. Can be coordinates
            or parameters or both.

        Notes
        =====

        The backend will request the data series to generate the numerical
        data. Depending on the data series, either the data series itself or
        the backend will eventually execute this function to generate the
        appropriate coloring value.
        """
        np = import_module('numpy')
        if self.color_func is None:
            # NOTE: with the line_color and surface_color attributes
            # (back-compatibility with the old sympy.plotting module) it is
            # possible to create a plot with a callable line_color (or
            # surface_color). For example:
            # p = plot(sin(x), line_color=lambda x, y: -y)
            # This will create a ColoredLineOver1DRangeSeries, which
            # efffectively is a parametric series. Later we could change
            # it to a string value:
            # p[0].line_color = "red"
            # However, this won't apply the red color, because we can't ask
            # a parametric series to be non-parametric!
            warnings.warn("This is likely not the result you were "
                "looking for. Please, re-execute the plot command, this time "
                "with the appropriate line_color or surface_color")
            return np.ones_like(args[0])

        if self._eval_color_func_with_signature:
            args = self._aggregate_args()
            color = self.color_func(*args)
            _re, _im = np.real(color), np.imag(color)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            return _re

        nargs = arity(self.color_func)
        if nargs == 1:
            if self.is_2Dline and self.is_parametric:
                if len(args) == 2:
                    # ColoredLineOver1DRangeSeries
                    return self._correct_shape(self.color_func(args[0]), args[0])
                # Parametric2DLineSeries
                return self._correct_shape(self.color_func(args[2]), args[2])
            elif self.is_3Dline and self.is_parametric:
                return self._correct_shape(self.color_func(args[3]), args[3])
            elif self.is_3Dsurface and self.is_parametric:
                return self._correct_shape(self.color_func(args[3]), args[3])
            return self._correct_shape(self.color_func(args[0]), args[0])
        elif nargs == 2:
            if self.is_3Dsurface and self.is_parametric:
                return self._correct_shape(self.color_func(*args[3:]), args[3])
            return self._correct_shape(self.color_func(*args[:2]), args[0])
        return self._correct_shape(self.color_func(*args[:nargs]), args[0])

    def get_data(self):
        """Compute and returns the numerical data.

        The number of parameters returned by this method depends on the
        specific instance. If ``s`` is the series, make sure to read
        ``help(s.get_data)`` to understand what it returns.
        """
        raise NotImplementedError

    def _get_wrapped_label(self, label, wrapper):
        """Given a latex representation of an expression, wrap it inside
        some characters. Matplotlib needs "$%s%$", K3D-Jupyter needs "%s".
        """
        return wrapper % label

    def get_label(self, use_latex=False, wrapper="$%s$"):
        """Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
        """
        if use_latex is False:
            return self._label
        if self._label == str(self.expr):
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._latex_label

    @property
    def label(self):
        return self.get_label()

    @label.setter
    def label(self, val):
        """Set the labels associated to this series."""
        # NOTE: the init method of any series requires a label. If the user do
        # not provide it, the preprocessing function will set label=None, which
        # informs the series to initialize two attributes:
        # _label contains the string representation of the expression.
        # _latex_label contains the latex representation of the expression.
        self._label = self._latex_label = val

    @property
    def ranges(self):
        return self._ranges

    @ranges.setter
    def ranges(self, val):
        new_vals = []
        for v in val:
            if v is not None:
                new_vals.append(tuple([sympify(t) for t in v]))
        self._ranges = new_vals

    def _apply_transform(self, *args):
        """Apply transformations to the results of numerical evaluation.

        Parameters
        ==========
        args : tuple
            Results of numerical evaluation.

        Returns
        =======
        transformed_args : tuple
            Tuple containing the transformed results.
        """
        t = lambda x, transform: x if transform is None else transform(x)
        x, y, z = None, None, None
        if len(args) == 2:
            x, y = args
            return t(x, self._tx), t(y, self._ty)
        elif (len(args) == 3) and isinstance(self, Parametric2DLineSeries):
            x, y, u = args
            return (t(x, self._tx), t(y, self._ty), t(u, self._tp))
        elif len(args) == 3:
            x, y, z = args
            return t(x, self._tx), t(y, self._ty), t(z, self._tz)
        elif (len(args) == 4) and isinstance(self, Parametric3DLineSeries):
            x, y, z, u = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), t(u, self._tp))
        elif len(args) == 4: # 2D vector plot
            x, y, u, v = args
            return (
                t(x, self._tx), t(y, self._ty),
                t(u, self._tx), t(v, self._ty)
            )
        elif (len(args) == 5) and isinstance(self, ParametricSurfaceSeries):
            x, y, z, u, v = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), u, v)
        elif (len(args) == 6) and self.is_3Dvector: # 3D vector plot
            x, y, z, u, v, w = args
            return (
                t(x, self._tx), t(y, self._ty), t(z, self._tz),
                t(u, self._tx), t(v, self._ty), t(w, self._tz)
            )
        elif len(args) == 6: # complex plot
            x, y, _abs, _arg, img, colors = args
            return (
                x, y, t(_abs, self._tz), _arg, img, colors)
        return args

    def _str_helper(self, s):
        pre, post = "", ""
        if self.is_interactive:
            pre = "interactive "
            post = " and parameters " + str(tuple(self.params.keys()))
        return pre + s + post


def _detect_poles_helper(x, y, eps=0.01):
    """Compute the steepness of each segment. If it's greater than a
    threshold, set the right-point y-value non NaN.
    """
    np = import_module('numpy')

    yy = y.copy()
    threshold = np.pi / 2 - eps
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = abs(y[i + 1] - y[i])
        angle = np.arctan(dy / dx)
        if abs(angle) >= threshold:
            yy[i + 1] = np.nan
    return x, yy


class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines."""

    is_2Dline = True
    _N = 1000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = kwargs.get("steps", False)
        self.is_point = kwargs.get("is_point", False)
        self.is_filled = kwargs.get("is_filled", True)
        self.adaptive = kwargs.get("adaptive", cfg["adaptive"]["used_by_default"])
        self.adaptive_goal = kwargs.get("adaptive_goal", cfg["adaptive"]["goal"])
        self.loss_fn = kwargs.get("loss_fn", None)
        self.use_cm = kwargs.get("use_cm", False)
        self.color_func = kwargs.get("color_func", None)
        self.line_color = kwargs.get("line_color", None)
        self.detect_poles = kwargs.get("detect_poles", False)
        self.eps = kwargs.get("eps", 0.01)
        self.is_polar = kwargs.get("is_polar", False)

    def get_data(self):
        """Return coordinates for plotting the line.

        Returns
        =======

        x: np.ndarray
            x-coordinates

        y: np.ndarray
            y-coordinates

        z: np.ndarray (optional)
            z-coordinates in case of Parametric3DLineSeries,
            Parametric3DLineInteractiveSeries

        param : np.ndarray (optional)
            The parameter in case of Parametric2DLineSeries,
            Parametric3DLineSeries or AbsArgLineSeries (and their
            corresponding interactive series).
        """
        np = import_module('numpy')
        points = self._get_data_helper()

        # postprocessing
        points = self._apply_transform(*points)

        if self.is_2Dline and self.detect_poles:
            if len(points) == 2:
                x, y = points
                x, y = _detect_poles_helper(x, y, self.eps)
                points = (x, y)
            else:
                x, y, p = points
                x, y = _detect_poles_helper(x, y, self.eps)
                points = (x, y, p)

        if self.steps is True:
            if self.is_2Dline:
                x, y = points[0], points[1]
                x = np.array((x, x)).T.flatten()[1:]
                y = np.array((y, y)).T.flatten()[:-1]
                if self.is_parametric:
                    points = (x, y, points[2])
                else:
                    points = (x, y)
            elif self.is_3Dline:
                x = np.repeat(points[0], 3)[2:]
                y = np.repeat(points[1], 3)[:-2]
                z = np.repeat(points[2], 3)[1:-1]
                if len(points) > 3:
                    points = (x, y, z, points[3])
                else:
                    points = (x, y, z)
        return points

    @property
    def var(self):
        return None if not self.ranges else self.ranges[0][0]

    @property
    def start(self):
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][1])
        except:
            return self.ranges[0][1]

    @property
    def end(self):
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][2])
        except:
            return self.ranges[0][2]


class List2DSeries(Line2DBaseSeries):
    """Representation for a line consisting of list of points."""

    _allowed_keys = ["adaptive", "adaptive_goal", "color_func", "is_filled",
    "is_point", "is_polar", "line_color", "loss_fn", "modules", "n",
    "only_integers", "rendering_kw", "steps", "use_cm", "xscale",
    "tx", "ty", "tz", "use_cm"]

    def __init__(self, list_x, list_y, label="", **kwargs):
        super().__init__(**kwargs)
        np = import_module('numpy')
        if len(list_x) != len(list_y):
            raise ValueError(
                "The two lists of coordinates must have the same "
                "number of elements.\n"
                "Received: len(list_x) = {} ".format(len(list_x)) +
                "and len(list_y) = {}".format(len(list_y))
            )
        self._block_lambda_functions(list_x, list_y)
        check = lambda l: [isinstance(t, Expr) and (not t.is_number) for t in l]
        if any(check(list_x) + check(list_y)) or self.params:
            if not self.params:
                raise ValueError("Some or all elements of the provided lists "
                    "are symbolic expressions, but the ``params`` dictionary "
                    "was not provided: those elements can't be evaluated.")
            self.list_x = Tuple(*list_x)
            self.list_y = Tuple(*list_y)
        else:
            self.list_x = np.array(list_x, dtype=np.float64)
            self.list_y = np.array(list_y, dtype=np.float64)

        self._expr = (self.list_x, self.list_y)
        if not any(isinstance(t, np.ndarray) for t in [self.list_x, self.list_y]):
            self._check_fs()
        self.is_polar = kwargs.get("is_polar", False)
        self.label = label
        self.rendering_kw = kwargs.get("rendering_kw", dict())
        if self.use_cm and self.color_func:
            self.is_parametric = True
            if isinstance(self.color_func, Expr):
                raise TypeError(
                    "%s don't support symbolic " % self.__class__.__name__ +
                    "expression for `color_func`.")

    def __str__(self):
        return "2D list plot"

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed."""
        lx, ly = self.list_x, self.list_y

        if not self.is_interactive:
            return self._eval_color_func_and_return(lx, ly)

        np = import_module('numpy')
        lx = np.array([t.evalf(subs=self.params) for t in lx], dtype=float)
        ly = np.array([t.evalf(subs=self.params) for t in ly], dtype=float)
        return self._eval_color_func_and_return(lx, ly)

    def _eval_color_func_and_return(self, *data):
        if self.use_cm and callable(self.color_func):
            return [*data, self.eval_color_func(*data)]
        return data


class List3DSeries(List2DSeries):
    is_2Dline = False
    is_3Dline = True

    def __init__(self, list_x, list_y, list_z, label="", **kwargs):
        # TODO: this can definitely be done better
        super().__init__(list_x, list_y, label, **kwargs)
        np = import_module('numpy')
        if len(list_z) != len(list_x):
            raise ValueError(
                "The three lists of coordinates must have the same "
                "number of elements.\n"
                "Received: len(list_x) = len(list_y) = {} ".format(len(list_x)) +
                "and len(list_z) = {}".format(len(list_z))
            )
        self._block_lambda_functions(list_z)
        check = lambda l: [isinstance(t, Expr) and (not t.is_number) for t in l]
        if any(check(list_z)):
            if not self.params:
                raise ValueError("Some or all elements of the provided lists "
                    "are symbolic expressions, but the ``params`` dictionary "
                    "was not provided: those elements can't be evaluated.")
            self.list_z = Tuple(*list_z)
            self._check_fs()
        else:
            self.list_z = np.array(list_z, dtype=np.float64)

        self._expr = (self.list_x, self.list_y, self.list_z)

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed."""
        lx, ly, lz = self.list_x, self.list_y, self.list_z

        if not self.is_interactive:
            return self._eval_color_func_and_return(lx, ly, lz)

        np = import_module('numpy')
        lx = np.array([t.evalf(subs=self.params) for t in lx], dtype=float)
        ly = np.array([t.evalf(subs=self.params) for t in ly], dtype=float)
        lz = np.array([t.evalf(subs=self.params) for t in lz], dtype=float)
        return self._eval_color_func_and_return(lx, ly, lz)

    def __str__(self):
        return "3D list plot"


class LineOver1DRangeSeries(Line2DBaseSeries):
    """Representation for a line consisting of a SymPy expression over a
    real range."""

    _allowed_keys = ["absarg", "adaptive", "adaptive_goal", "color_func",
    "detect_poles", "eps","is_complex", "is_filled", "is_point", "line_color",
    "loss_fn", "modules", "n", "only_integers", "rendering_kw", "steps",
    "use_cm", "xscale", "tx", "ty", "tz", "is_polar"]

    def __new__(cls, *args, **kwargs):
        if kwargs.get("absarg", False):
            return super().__new__(AbsArgLineSeries)
        cf = kwargs.get("color_func", None)
        if (callable(cf) or  callable(kwargs.get("line_color", None)) or
            isinstance(cf, Expr)):
            return super().__new__(ColoredLineOver1DRangeSeries)
        return object.__new__(cls)

    def __init__(self, expr, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self._label = str(self.expr) if label is None else label
        self._latex_label = latex(self.expr) if label is None else label
        self.ranges = [var_start_end]
        self._cast = complex
        # for complex-related data series, this determines what data to return
        # on the y-axis
        self._return = kwargs.get("return", None)
        self._post_init()

        if not self._interactive_ranges:
            # NOTE: the following check is only possible when the minimum and
            # maximum values of a plotting range are numeric
            start, end = [complex(t) for t in self.ranges[0][1:]]
            if im(start) != im(end):
                raise ValueError(
                    "%s requires the imaginary " % self.__class__.__name__ +
                    "part of the start and end values of the range "
                    "to be the same.")

    def __str__(self):
        def f(t):
            if isinstance(t, complex):
                if t.imag != 0:
                    return t
                return t.real
            return t
        pre = "interactive " if self.is_interactive else ""
        post = ""
        if self.is_interactive:
            post = " and parameters " + str(tuple(self.params.keys()))
        wrapper = _get_wrapper_for_expr(self._return)
        return pre + "cartesian line: %s for %s over %s" % (
            wrapper % self.expr,
            str(self.var),
            str((f(self.start), f(self.end))),
        ) + post

    def _adaptive_sampling(self):
        np = import_module('numpy')

        def func(f, imag, x):
            try:
                w = complex(f(x + 1j * imag))
                return w.real, w.imag
            except (ZeroDivisionError, OverflowError):
                return np.nan, np.nan

        data = _adaptive_eval(
            func, [self.var], self.expr,
            [complex(self.start).real, complex(self.end).real],
            complex(self.start).imag,
            modules=self.modules,
            adaptive_goal=self.adaptive_goal,
            loss_fn=self.loss_fn)
        return data[:, 0], data[:, 1], data[:, 2]

    def _uniform_sampling(self):
        np = import_module('numpy')

        x, result = self._evaluate()
        _re, _im = np.real(result), np.imag(result)
        _re = self._correct_shape(_re, x)
        _im = self._correct_shape(_im, x)
        return x, _re, _im

    def _get_real_imag(self):
        """ By evaluating the function over a complex range it should
        return complex values. The imaginary part can be used to mask out the
        unwanted values.
        """
        if self.adaptive:
            return self._adaptive_sampling()
        return self._uniform_sampling()

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        """
        np = import_module('numpy')

        x, _re, _im = self._get_real_imag()

        if self._return is None:
            # The evaluation could produce complex numbers. Set real elements
            # to NaN where there are non-zero imaginary elements
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        elif self._return == "real":
            pass
        elif self._return == "imag":
            _re = _im
        elif self._return == "abs":
            _re = np.sqrt(_re**2 + _im**2)
        elif self._return == "arg":
            _re = np.arctan2(_im, _re)
        else:
            raise ValueError("`_return` not recognized. Received: %s" % self._return)

        return x, _re


class ColoredLineOver1DRangeSeries(LineOver1DRangeSeries):
    """Represents a 2D line series in which `color_func` is a callable.
    """
    is_parametric = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cm = kwargs.get("use_cm", True)

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        x, y = super()._get_data_helper()
        return x, y, self.eval_color_func(x, y)


class AbsArgLineSeries(LineOver1DRangeSeries):
    """Represents the absolute value of a complex function colored by its
    argument over a complex range (a + I*b, c + I * b). Note that the imaginary part of the start and end must be the same.
    """

    is_parametric = True
    is_complex = True

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cm = kwargs.get("use_cm", True)

    def __str__(self):
        return self._str_helper("cartesian abs-arg line: %s for %s over %s" % (
            str(self.expr),
            str(self.var),
            str((self.start, self.end))))

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        np = import_module('numpy')

        x, _re, _im = self._get_real_imag()
        _abs = np.sqrt(_re**2 + _im**2)
        _angle = np.arctan2(_im, _re)
        return x, _abs, _angle


class ParametricLineBaseSeries(Line2DBaseSeries):
    is_parametric = True
    _allowed_keys = ["adaptive", "adaptive_goal", "color_func", "is_filled",
    "is_point", "line_color", "loss_fn", "modules", "n", "only_integers",
    "rendering_kw", "use_cm", "xscale", "tx", "ty", "tz", "tp", "colorbar"]

    def _set_parametric_line_label(self, label):
        """Logic to set the correct label to be shown on the plot.
        If `use_cm=True` there will be a colorbar, so we show the parameter.
        If `use_cm=False`, there might be a legend, so we show the expressions.

        Parameters
        ==========
        label : str
            label passed in by the pre-processor or the user
        """
        self._label = str(self.var) if label is None else label
        self._latex_label = latex(self.var) if label is None else label
        if (self.use_cm is False) and (self._label == str(self.var)):
            self._label = str(self.expr)
            self._latex_label = latex(self.expr)
        # if the expressions is a lambda function and use_cm=False and no label
        # has been provided, then its better to do the following in order to
        # avoid suprises on the backend
        if any(callable(e) for e in self.expr) and (not self.use_cm):
            if self._label == str(self.expr):
                self._label = ""

    def _adaptive_sampling(self):
        np = import_module('numpy')

        def func(f, is_2Dline, x):
            try:
                w = [complex(t) for t in f(complex(x))]
                return [t.real if np.isclose(t.imag, 0) else np.nan for t in w]
            except (ZeroDivisionError, OverflowError):
                return [np.nan for t in range(2 if is_2Dline else 3)]

        if all(not callable(e) for e in self.expr):
            expr = Tuple(self.expr_x, self.expr_y)
            if not self.is_2Dline:
                expr = Tuple(self.expr_x, self.expr_y, self.expr_z)
        else:
            # expr is user-provided lambda functions
            expr = lambda x: (self.expr_x(x), self.expr_y(x))
            if not self.is_2Dline:
                expr = lambda x: (self.expr_x(x), self.expr_y(x), self.expr_z(x))

        data = _adaptive_eval(
            func, [self.var], expr,
            [float(self.start), float(self.end)],
            self.is_2Dline,
            modules=self.modules,
            adaptive_goal=self.adaptive_goal,
            loss_fn=self.loss_fn)

        if self.is_2Dline:
            return data[:, 1], data[:, 2], data[:, 0]
        return data[:, 1], data[:, 2], data[:, 3], data[:, 0]

    def get_label(self, use_latex=False, wrapper="$%s$"):
        # parametric lines returns the representation of the parameter to be
        # shown on the colorbar if `use_cm=True`, otherwise it returns the
        # representation of the expression to be placed on the legend.
        if self.use_cm:
            if str(self.var) == self._label:
                if use_latex:
                    return self._get_wrapped_label(latex(self.var), wrapper)
                return str(self.var)
            # here the user has provided a custom label
            return self._label
        if use_latex:
            if self._label != str(self.expr):
                return self._latex_label
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._label

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        if self.adaptive:
            coords = self._adaptive_sampling()
        else:
            coords = self._uniform_sampling()

        if self.is_2Dline and self.is_polar:
            # when plot_polar is executed with polar_axis=True
            np = import_module('numpy')
            x, y, _ = coords
            r = np.sqrt(x**2 + y**2)
            t = np.arctan2(y, x)
            coords = [t, r, coords[-1]]

        if callable(self.color_func):
            coords = list(coords)
            coords[-1] = self.eval_color_func(*coords)

        return coords

    def _uniform_sampling(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')

        results = self._evaluate()
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        return [*results[1:], results[0]]


class Parametric2DLineSeries(ParametricLineBaseSeries):
    """Representation for a line consisting of two parametric sympy expressions
    over a range."""

    is_parametric = True

    def __init__(self, expr_x, expr_y, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr = (self.expr_x, self.expr_y)
        self.ranges = [var_start_end]
        self._cast = float
        self.use_cm = kwargs.get("use_cm", True)
        self._set_parametric_line_label(label)
        self._post_init()

    def __str__(self):
        return self._str_helper(
            "parametric cartesian line: (%s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.var),
            str((self.start, self.end))
        ))


class Parametric3DLineSeries(ParametricLineBaseSeries):
    """Representation for a 3D line consisting of three parametric sympy
    expressions and a range."""

    is_2Dline = False
    is_3Dline = True

    def __init__(self, expr_x, expr_y, expr_z, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end]
        self._cast = float
        self.use_cm = kwargs.get("use_cm", True)
        self._set_parametric_line_label(label)
        self._post_init()

    def __str__(self):
        return self._str_helper(
            "3D parametric cartesian line: (%s, %s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.expr_z),
            str(self.var),
            str((self.start, self.end))
        ))


class SurfaceBaseSeries(BaseSeries):
    """A base class for 3D surfaces."""

    is_3Dsurface = True
    _allowed_keys = ["adaptive", "adaptive_goal", "color_func", "is_polar",
        "loss_fn", "modules", "n1", "n2", "only_integers", "rendering_kw",
        "surface_color", "use_cm", "xscale", "yscale", "tx", "ty", "tz",
        "colorbar"]

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.adaptive = kwargs.get("adaptive", False)
        self.adaptive_goal = kwargs.get("adaptive_goal", cfg["adaptive"]["goal"])
        self.loss_fn = kwargs.get("loss_fn", None)
        self.use_cm = kwargs.get("use_cm", cfg["plot3d"]["use_cm"])
        # NOTE: why should SurfaceOver2DRangeSeries support is polar?
        # After all, the same result can be achieve with
        # ParametricSurfaceSeries. For example:
        # sin(r) for (r, 0, 2 * pi) and (theta, 0, pi/2) can be parameterized
        # as (r * cos(theta), r * sin(theta), sin(t)) for (r, 0, 2 * pi) and
        # (theta, 0, pi/2).
        # Because it is faster to evaluate (important for interactive plots).
        self.is_polar = kwargs.get("is_polar", False)
        self.surface_color = kwargs.get("surface_color", None)
        self.color_func = kwargs.get("color_func", lambda x, y, z: z)
        if callable(self.surface_color):
            self.color_func = self.surface_color
            self.surface_color = None

    def _set_surface_label(self, label):
        exprs = self.expr
        self._label = str(exprs) if label is None else label
        self._latex_label = latex(exprs) if label is None else label
        # if the expressions is a lambda function and no label
        # has been provided, then its better to do the following to avoid
        # suprises on the backend
        is_lambda = (callable(exprs) if not hasattr(exprs, "__iter__")
            else any(callable(e) for e in exprs))
        if is_lambda and (self._label == str(exprs)):
                self._label = ""


class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of a sympy expression and 2D
    range."""

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self.ranges = [var_start_end_x, var_start_end_y]
        self._set_surface_label(label)
        self._post_init()

    @property
    def var_x(self):
        return self.ranges[0][0]

    @property
    def var_y(self):
        return self.ranges[1][0]

    @property
    def start_x(self):
        try:
            return float(self.ranges[0][1])
        except:
            return self.ranges[0][1]

    @property
    def end_x(self):
        try:
            return float(self.ranges[0][2])
        except:
            return self.ranges[0][2]

    @property
    def start_y(self):
        try:
            return float(self.ranges[1][1])
        except:
            return self.ranges[1][1]

    @property
    def end_y(self):
        try:
            return float(self.ranges[1][2])
        except:
            return self.ranges[1][2]

    def __str__(self):
        series_type = "cartesian surface" if self.is_3Dsurface else "contour"
        return self._str_helper(
            series_type + ": %s for" " %s over %s and %s over %s" % (
            str(self.expr),
            str(self.var_x), str((self.start_x, self.end_x)),
            str(self.var_y), str((self.start_y, self.end_y)),
        ))

    def _adaptive_sampling(self):
        np = import_module('numpy')

        def func(f, xy):
            try:
                w = f(*[complex(t) for t in xy])
                return w.real if np.isclose(w.imag, 0) else np.nan
            except (ZeroDivisionError, OverflowError):
                return np.nan

        return _adaptive_eval(
            func, [self.var_x, self.var_y], self.expr,
            [(self.start_x, self.end_x), (self.start_y, self.end_y)],
            modules=self.modules,
            adaptive_goal=self.adaptive_goal,
            loss_fn=self.loss_fn)

    def _uniform_sampling(self):
        np = import_module('numpy')

        results = self._evaluate()
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        return results

    def get_data(self):
        """Return arrays of coordinates for plotting. Depending on the
        `adaptive` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        Returns
        =======

        mesh_x : np.ndarray [n2 x n1]
            Real Discretized x-domain.

        mesh_y : np.ndarray [n2 x n1]
            Real Discretized y-domain.

        z : np.ndarray [n2 x n1]
            Results of the evaluation.
        """
        np = import_module('numpy')

        if self.adaptive:
            res = self._adaptive_sampling()
        else:
            res = self._uniform_sampling()

        x, y, z = res
        r = x.copy()
        if self.is_polar and self.is_3Dsurface:
            x = r * np.cos(y)
            y = r * np.sin(y)

        return self._apply_transform(x, y, z)


class ParametricSurfaceSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of three parametric sympy
    expressions and a range."""

    is_parametric = True

    def __init__(self, expr_x, expr_y, expr_z,
        var_start_end_u, var_start_end_v, label="", **kwargs
    ):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end_u, var_start_end_v]
        self.color_func = kwargs.get("color_func", lambda x, y, z, u, v: z)
        self._set_surface_label(label)

        if self.adaptive:
            # NOTE: turns out that it is difficult to interpolate over 3
            # parameters in order to get a uniform grid out of the adaptive
            # results. As a consequence, let's not implement adaptive for this
            # class.
            warnings.warn(
                "ParametricSurfaceSeries does not support adaptive algorithm. "
                "Automatically switching to a uniform spacing algorithm.")
            self.adaptive = False

        self._post_init()

    @property
    def var_u(self):
        return self.ranges[0][0]

    @property
    def var_v(self):
        return self.ranges[1][0]

    @property
    def start_u(self):
        try:
            return float(self.ranges[0][1])
        except:
            return self.ranges[0][1]

    @property
    def end_u(self):
        try:
            return float(self.ranges[0][2])
        except:
            return self.ranges[0][2]

    @property
    def start_v(self):
        try:
            return float(self.ranges[1][1])
        except:
            return self.ranges[1][1]

    @property
    def end_v(self):
        try:
            return float(self.ranges[1][2])
        except:
            return self.ranges[1][2]

    def __str__(self):
        return self._str_helper(
            "parametric cartesian surface: (%s, %s, %s) for"
            " %s over %s and %s over %s" % (
            str(self.expr_x), str(self.expr_y), str(self.expr_z),
            str(self.var_u), str((self.start_u, self.end_u)),
            str(self.var_v), str((self.start_v, self.end_v)),
        ))

    def get_data(self):
        """Return arrays of coordinates for plotting. Depending on the
        `adaptive` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        Returns
        =======

        x : np.ndarray [n2 x n1]
            x-coordinates.
        y : np.ndarray [n2 x n1]
            y-coordinates.
        z : np.ndarray [n2 x n1]
            z-coordinates.
        mesh_u : np.ndarray [n2 x n1]
            Discretized u range.
        mesh_v : np.ndarray [n2 x n1]
            Discretized v range.
        """
        np = import_module('numpy')

        results = self._evaluate()
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        return self._apply_transform(*results[2:], *results[:2])


class ContourSeries(SurfaceOver2DRangeSeries):
    """Representation for a contour plot."""

    is_3Dsurface = False
    is_contour = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._allowed_keys += ["contour_kw", "is_filled", "clabels", "colorbar"]
        self.is_filled = kwargs.get("is_filled", True)
        self.show_clabels = kwargs.get("clabels", True)

        # NOTE: contour plots are used by plot_contour, plot_vector and
        # plot_complex_vector. By implementing contour_kw we are able to
        # quickly target the contour plot.
        self.rendering_kw = kwargs.get("contour_kw",
            kwargs.get("rendering_kw", dict()))


class ImplicitSeries(BaseSeries):
    """Representation for Implicit plot

    References
    ==========

    .. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for
    Mathematical Formulae with Two Free Variables.

    .. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval
    Arithmetic. Master's thesis. University of Toronto, 1996
    """

    is_implicit = True
    use_cm = False
    _allowed_keys = ["adaptive", "depth", "n1", "n2", "rendering_kw",
    "xscale", "yscale", "color"]
    _N = 100

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__(**kwargs)
        self._block_lambda_functions(expr)
        # these are needed for adaptive evaluation
        expr, has_equality = self._has_equality(sympify(expr))
        self._adaptive_expr = expr
        self.has_equality = has_equality
        self.ranges = [var_start_end_x, var_start_end_y]
        self.var_x, self.start_x, self.end_x = self.ranges[0]
        self.var_y, self.start_y, self.end_y = self.ranges[1]
        self._label = str(expr) if label is None else label
        self._latex_label = latex(expr) if label is None else label
        self.adaptive = kwargs.get("adaptive", False)
        self.color = kwargs.get("color", kwargs.get("line_color", None))

        if ((isinstance(expr, BooleanFunction) or isinstance(expr, Ne))
            and (not self.adaptive)):
            self.adaptive = True
            msg = "contains Boolean functions. "
            if isinstance(expr, Ne):
                msg = "is an unequality. "
            warnings.warn(
                "The provided expression " + msg
                + "In order to plot the expression, the algorithm "
                + "automatically switched to an adaptive sampling."
            )

        if isinstance(expr, BooleanFunction):
            self._non_adaptive_expr = None
            self._is_equality = False
        else:
            # these are needed for uniform meshing evaluation
            expr, is_equality = self._preprocess_meshgrid_expression(expr, self.adaptive)
            self._non_adaptive_expr = expr
            self._is_equality = is_equality

        if self.is_interactive and self.adaptive:
            raise NotImplementedError("Interactive plot with `adaptive=True` "
                "is not supported.")

        # Check whether the depth is greater than 4 or less than 0.
        depth = kwargs.get("depth", 0)
        if depth > 4:
            depth = 4
        elif depth < 0:
            depth = 0
        self.depth = 4 + depth
        self._post_init()

    @property
    def expr(self):
        if self.adaptive:
            return self._adaptive_expr
        return self._non_adaptive_expr

    def _has_equality(self, expr):
        # Represents whether the expression contains an Equality, GreaterThan
        # or LessThan
        has_equality = False

        def arg_expand(bool_expr):
            """Recursively expands the arguments of an Boolean Function"""
            for arg in bool_expr.args:
                if isinstance(arg, BooleanFunction):
                    arg_expand(arg)
                elif isinstance(arg, Relational):
                    arg_list.append(arg)

        arg_list = []
        if isinstance(expr, BooleanFunction):
            arg_expand(expr)
            # Check whether there is an equality in the expression provided.
            if any(isinstance(e, (Equality, GreaterThan, LessThan)) for e in arg_list):
                has_equality = True
        elif not isinstance(expr, Relational):
            expr = Equality(expr, 0)
            has_equality = True
        elif isinstance(expr, (Equality, GreaterThan, LessThan)):
            has_equality = True

        return expr, has_equality

    def __str__(self):
        f = lambda t: float(t) if len(t.free_symbols) == 0 else t

        return self._str_helper(
            "Implicit expression: %s for %s over %s and %s over %s") % (
            str(self._adaptive_expr),
            str(self.var_x),
            str((f(self.start_x), f(self.end_x))),
            str(self.var_y),
            str((f(self.start_y), f(self.end_y))),
        )

    def get_data(self):
        """Returns numerical data.

        Returns
        =======

        If the series is evaluated with the `adaptive=True` it returns:

        interval_list : list
            List of bounding rectangular intervals to be postprocessed and
            eventually used with Matplotlib's ``fill`` command.
        dummy : str
            A string containing ``"fill"``.

        Otherwise, it returns 2D numpy arrays to be used with Matplotlib's
        ``contour`` or ``contourf`` commands:

        x_array : np.ndarray
        y_array : np.ndarray
        z_array : np.ndarray
        plot_type : str
            A string specifying which plot command to use, ``"contour"``
            or ``"contourf"``.
        """
        if self.adaptive:
            data = self._adaptive_eval()
            if data is not None:
                return data

        return self._get_meshes_grid()

    def _adaptive_eval(self):
        import sympy.plotting.intervalmath.lib_interval as li

        user_functions = {}
        printer = IntervalMathPrinter({
            'fully_qualified_modules': False, 'inline': True,
            'allow_unknown_functions': True,
            'user_functions': user_functions})

        keys = [t for t in dir(li) if ("__" not in t) and (t not in ["import_module", "interval"])]
        vals = [getattr(li, k) for k in keys]
        d = {k: v for k, v in zip(keys, vals)}
        func = lambdify((self.var_x, self.var_y), self.expr, modules=[d], printer=printer)
        data = None

        try:
            data = self._get_raster_interval(func)
        except NameError as err:
            warnings.warn(
                "Adaptive meshing could not be applied to the"
                " expression, as some functions are not yet implemented"
                " in the interval math module:\n\n"
                "NameError: %s\n\n" % err +
                "Proceeding with uniform meshing."
                )
            self.adaptive = False
        except (AttributeError, TypeError):
            # XXX: AttributeError("'list' object has no attribute 'is_real'")
            # That needs fixing somehow - we shouldn't be catching
            # AttributeError here.
            warnings.warn(
                "Adaptive meshing could not be applied to the"
                " expression. Using uniform meshing.")
            self.adaptive = False

        return data

    def _get_raster_interval(self, func):
        """Uses interval math to adaptively mesh and obtain the plot"""
        np = import_module('numpy')

        k = self.depth
        interval_list = []
        sx, sy = [float(t) for t in [self.start_x, self.start_y]]
        ex, ey = [float(t) for t in [self.end_x, self.end_y]]
        # Create initial 32 divisions
        xsample = np.linspace(sx, ex, 33)
        ysample = np.linspace(sy, ey, 33)

        # Add a small jitter so that there are no false positives for equality.
        # Ex: y==x becomes True for x interval(1, 2) and y interval(1, 2)
        # which will draw a rectangle.
        jitterx = (
            (np.random.rand(len(xsample)) * 2 - 1)
            * (ex - sx)
            / 2 ** 20
        )
        jittery = (
            (np.random.rand(len(ysample)) * 2 - 1)
            * (ey - sy)
            / 2 ** 20
        )
        xsample += jitterx
        ysample += jittery

        xinter = [interval(x1, x2) for x1, x2 in zip(xsample[:-1], xsample[1:])]
        yinter = [interval(y1, y2) for y1, y2 in zip(ysample[:-1], ysample[1:])]
        interval_list = [[x, y] for x in xinter for y in yinter]
        plot_list = []

        # recursive call refinepixels which subdivides the intervals which are
        # neither True nor False according to the expression.
        def refine_pixels(interval_list):
            """Evaluates the intervals and subdivides the interval if the
            expression is partially satisfied."""
            temp_interval_list = []
            plot_list = []
            for intervals in interval_list:

                # Convert the array indices to x and y values
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                # The expression is valid in the interval. Change the contour
                # array values to 1.
                if func_eval[1] is False or func_eval[0] is False:
                    pass
                elif func_eval == (True, True):
                    plot_list.append([intervalx, intervaly])
                elif func_eval[1] is None or func_eval[0] is None:
                    # Subdivide
                    avgx = intervalx.mid
                    avgy = intervaly.mid
                    a = interval(intervalx.start, avgx)
                    b = interval(avgx, intervalx.end)
                    c = interval(intervaly.start, avgy)
                    d = interval(avgy, intervaly.end)
                    temp_interval_list.append([a, c])
                    temp_interval_list.append([a, d])
                    temp_interval_list.append([b, c])
                    temp_interval_list.append([b, d])
            return temp_interval_list, plot_list

        while k >= 0 and len(interval_list):
            interval_list, plot_list_temp = refine_pixels(interval_list)
            plot_list.extend(plot_list_temp)
            k = k - 1
        # Check whether the expression represents an equality
        # If it represents an equality, then none of the intervals
        # would have satisfied the expression due to floating point
        # differences. Add all the undecided values to the plot.
        if self.has_equality:
            for intervals in interval_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                if func_eval[1] and func_eval[0] is not False:
                    plot_list.append([intervalx, intervaly])
        return plot_list, "fill"

    def _get_meshes_grid(self):
        """Generates the mesh for generating a contour.

        In the case of equality, ``contour`` function of matplotlib can
        be used. In other cases, matplotlib's ``contourf`` is used.
        """
        np = import_module('numpy')

        xarray, yarray, z_grid = self._evaluate()
        _re, _im = np.real(z_grid), np.imag(z_grid)
        _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        if self._is_equality:
            return xarray, yarray, _re, 'contour'
        return xarray, yarray, _re, 'contourf'

    @staticmethod
    def _preprocess_meshgrid_expression(expr, adaptive):
        """If the expression is a Relational, rewrite it as a single
        expression.

        Returns
        =======

        expr : Expr
            The rewritten expression

        equality : Boolean
            Wheter the original expression was an Equality or not.
        """
        equality = False
        if isinstance(expr, Equality):
            expr = expr.lhs - expr.rhs
            equality = True
        elif isinstance(expr, (GreaterThan, StrictGreaterThan)):
            expr = expr.lhs - expr.rhs
        elif isinstance(expr, (LessThan, StrictLessThan)):
            expr = expr.rhs - expr.lhs
        elif not adaptive:
            raise NotImplementedError(
                "The expression is not supported for "
                "plotting in uniform meshed plot."
            )
        return expr, equality

    def get_label(self, use_latex=False, wrapper="$%s$"):
        """Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
        """
        if use_latex is False:
            return self._label
        if self._label == str(self._adaptive_expr):
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._latex_label


class Implicit3DSeries(SurfaceBaseSeries):
    is_implicit = True
    _N = 60

    def __init__(self, expr, range_x, range_y, range_z, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self.ranges = [range_x, range_y, range_z]
        self.var_x, self.start_x, self.end_x = self.ranges[0]
        self.var_y, self.start_y, self.end_y = self.ranges[1]
        self.var_z, self.start_z, self.end_z = self.ranges[2]
        self._set_surface_label(label)
        self._allowed_keys += ["n3", "zscale"]

    def __str__(self):
        var_x, start_x, end_x = self.ranges[0]
        var_y, start_y, end_y = self.ranges[1]
        var_z, start_z, end_z = self.ranges[2]
        return ("implicit surface series: %s for %s over %s and %s over %s"
            " and %s over %s") % (
                str(self.expr),
                str(var_x), str((float(start_x), float(end_x))),
                str(var_y), str((float(start_y), float(end_y))),
                str(var_z), str((float(start_z), float(end_z)))
            )

    def get_data(self):
        """Evaluate the expression over the provided domain. The backend will
        then try to compute and visualize the final result, if it support this
        data series.

        Returns
        =======
        mesh_x : np.ndarray [n1 x n2 x n3]
        mesh_y : np.ndarray [n1 x n2 x n3]
        mesh_z : np.ndarray [n1 x n2 x n3]
        f : np.ndarray [n1 x n2 x n3]
        """
        np = import_module('numpy')

        results = self._evaluate()
        for i, r in enumerate(results):
            re_v, im_v = np.real(r), np.imag(r)
            re_v[np.invert(np.isclose(im_v, np.zeros_like(im_v)))] = np.nan
            results[i] = re_v

        return self._apply_transform(*results)


class ComplexPointSeries(Line2DBaseSeries):
    """Representation for a line in the complex plane consisting of
    list of points."""

    _allowed_keys = ["color_func", "is_filled", "is_point", "line_color",
    "rendering_kw", "steps", "tx", "ty", "tz", "use_cm"]

    def __init__(self, expr, label="", **kwargs):
        super().__init__(**kwargs)
        if isinstance(expr, (list, tuple)):
            self.expr = Tuple(*expr)
        elif isinstance(expr, Expr):
            self.expr = Tuple(expr)
        else:
            self.expr = expr
        self._block_lambda_functions(*self.expr)

        self.is_point = kwargs.get("is_point", True)
        self.is_filled = kwargs.get("is_filled", True)
        self.steps = kwargs.get("steps", False)
        self._label = label
        self._latex_label = label
        self.rendering_kw = kwargs.get("rendering_kw", dict())
        self.use_cm = kwargs.get("use_cm", True)
        self.color_func = kwargs.get("color_func", None)
        if self.use_cm and self.color_func:
            self.is_parametric = True
        self.line_color = kwargs.get("line_color", None)

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')
        points = [complex(p.evalf(subs=self.params)) for p in self.expr]
        points = np.array(points)
        r, i = np.real(points), np.imag(points)
        if self.use_cm and callable(self.color_func):
            return r, i, self.eval_color_func(r, i)
        return r, i

    def __str__(self):
        return self._str_helper("complex points: %s" % self.expr)


class ComplexSurfaceBaseSeries(BaseSeries):
    """Represent a complex function."""
    is_complex = True
    _N = 300
    _allowed_keys = ["absarg", "coloring", "color_func", "modules", "phaseres",
    "is_polar", "n1", "n2", "only_integers", "rendering_kw", "steps", "cmap",
    "surface_color","use_cm", "xscale", "yscale", "tx", "ty", "tz", "threed",
    "blevel", "phaseoffset", "colorbar"]

    def __new__(cls, *args, **kwargs):
        domain_coloring = kwargs.get("absarg", False)
        if domain_coloring:
            return super().__new__(ComplexDomainColoringSeries)
        return super().__new__(ComplexSurfaceSeries)

    def _init_domain_coloring_kw(self, **kwargs):
        self.coloring = kwargs.get("coloring", "a")
        if isinstance(self.coloring, str):
            self.coloring = self.coloring.lower()
        elif not callable(self.coloring):
            raise TypeError(
                "`coloring` must be a character from 'a' to 'j' or a callable.")
        self.phaseres = kwargs.get("phaseres", 20)
        self.cmap = kwargs.get("cmap", None)
        self.blevel = float(kwargs.get("blevel", 0.75))
        if self.blevel < 0:
            warnings.warn("It must be 0 <= blevel <= 1. Automatically "
                "setting blevel = 0.")
            self.blevel = 0
        if self.blevel > 1:
            warnings.warn("It must be 0 <= blevel <= 1. Automatically "
                "setting blevel = 1.")
            self.blevel = 1
        self.phaseoffset = float(kwargs.get("phaseoffset", 0))

    def __init__(self, expr, r, label="", **kwargs):
        super().__init__(**kwargs)
        if kwargs.get("threed", False):
            self.is_3Dsurface = True
        self.expr = expr if callable(expr) else sympify(expr)
        self.ranges = [r]

        if isinstance(self, ComplexSurfaceSeries):
            self._block_lambda_functions(self.expr)

        self._label = str(self.expr) if label is None else label
        self._latex_label = latex(self.expr) if label is None else label
        self.use_cm = kwargs.get("use_cm", cfg["plot3d"]["use_cm"])
        self.is_polar = kwargs.get("is_polar", False)
        self.surface_color = kwargs.get("surface_color", None)
        self.is_filled = kwargs.get("is_filled", True)
        # determines what data to return on the z-axis
        self._return = kwargs.get("return", None)

        # domain coloring mode
        self._init_domain_coloring_kw(**kwargs)

        self._post_init()
        if not self._interactive_ranges:
            # NOTE: the following check is only possible when the minimum and
            # maximum values of a plotting range are numeric
            start, end = [complex(t) for t in self.ranges[0][1:]]
            if start.imag == end.imag:
                raise ValueError(
                    "The same imaginary part has been used for `start` and "
                    "`end`: %s. " % start.imag +
                    "They must be different."
                )

    @property
    def var(self):
        return self.ranges[0][0]

    @property
    def start(self):
        return self.ranges[0][1]

    @property
    def end(self):
        return self.ranges[0][2]

    def __str__(self):
        if self.is_domain_coloring:
            prefix = "complex domain coloring"
            if self.is_3Dsurface:
                prefix = "complex 3D domain coloring"
        else:
            prefix = "complex cartesian surface"
            if self.is_contour:
                prefix = "complex contour"

        wrapper = _get_wrapper_for_expr(self._return)
        res, ree = re(self.start), re(self.end)
        ims, ime = im(self.start), im(self.end)
        f = lambda t: float(t) if len(t.free_symbols) == 0 else t

        return self._str_helper(
            prefix + ": %s for" " re(%s) over %s and im(%s) over %s" % (
                wrapper % self.expr,
                str(self.var),
                str((f(res), f(ree))),
                str(self.var),
                str((f(ims), f(ime))),
        ))

    def _create_discretized_domain(self):
        """Discretize the ranges in case of uniform meshing strategy.
        """
        np = import_module('numpy')
        start_x = self._update_range_value(self.start).real
        end_x = self._update_range_value(self.end).real
        start_y = self._update_range_value(self.start).imag
        end_y = self._update_range_value(self.end).imag
        x = self._discretize(start_x, end_x, self.n[0],
            self.scales[0], self.only_integers)
        y = self._discretize(start_y, end_y, self.n[1],
            self.scales[1], self.only_integers)
        xx, yy = np.meshgrid(x, y)
        domain = xx + 1j * yy
        self._discretized_domain = {self.var: domain}


class ComplexSurfaceSeries(ComplexSurfaceBaseSeries):
    """Represents a 3D surface or contour plot of a complex function over
    the complex plane.
    """
    is_3Dsurface = True
    is_contour = False
    is_domain_coloring = False

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not kwargs.get("threed", False):
            # if not 3D, plot the contours
            self.is_contour = True
            self.is_3Dsurface = False
        self.color_func = kwargs.get("color_func", lambda x, y, z: z)
        self.rendering_kw = kwargs.get("rendering_kw", dict())
        self.is_filled = kwargs.get("is_filled", True)
        self.show_clabels = kwargs.get("clabels", True)
        self._allowed_keys += ["is_filled", "clabels"]

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======

        mesh_x : np.ndarray [n2 x n1]
            Real discretized domain.

        mesh_y : np.ndarray [n2 x n1]
            Imaginary discretized domain.

        z : np.ndarray [n2 x n1]
            Results of the evaluation.
        """
        np = import_module('numpy')

        domain, z = self._evaluate(False)
        if self._return is None:
            pass
        elif self._return == "real":
            z = np.real(z)
        elif self._return == "imag":
            z = np.imag(z)
        elif self._return == "abs":
            z = np.absolute(z)
        elif self._return == "arg":
            z = np.angle(z)
        else:
            raise ValueError("`_return` not recognized. Received: %s" % self._return)

        return np.real(domain), np.imag(domain), z


class ComplexDomainColoringSeries(ComplexSurfaceBaseSeries):
    """Represents a 2D/3D domain coloring plot of a complex function over
    the complex plane.
    """
    is_3Dsurface = False
    is_domain_coloring = True

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get("threed", False):
            self.is_3Dsurface = True
            self.use_cm = kwargs.get("use_cm", True)
        self.rendering_kw = kwargs.get("rendering_kw", dict())
        # apply the transformation z -> 1/z in order to study the behavior
        # of the function at z=infinity
        self.at_infinity = kwargs.get("at_infinity", False)
        if self.at_infinity:
            if callable(self.expr):
                raise ValueError(
                    "``at_infinity=True`` is only supported for symbolic "
                    "expressions. Instead, a callable was provided.")
            z = self.ranges[0][0]
            tmp = self.expr.subs(z, 1 / z)
            if self._label == str(self.expr):
                # adjust labels to prevent the wrong one to be seen on colorbar
                self._label = str(tmp)
                self._latex_label = latex(tmp)
            self.expr = tmp

        self.annotate = kwargs.get("annotate", True)
        self.riemann_mask = kwargs.get("riemann_mask", False)
        self._allowed_keys += ["at_infinity", "riemann_mask", "annotate"]

    def _domain_coloring(self, domain, w):
        if isinstance(self.coloring, str):
            self.coloring = self.coloring.lower()
            return wegert(self.coloring, w, self.phaseres, self.cmap,
                self.blevel, self.phaseoffset,
                self.at_infinity, self.riemann_mask,
                domain=[domain[0, 0], domain[-1, -1]])
        return self.coloring(w)

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======

        mesh_x : np.ndarray [n2 x n1]
            Real discretized domain.

        mesh_y : np.ndarray [n2 x n1]
            Imaginary discretized domain.

        abs : np.ndarray [n2 x n1]
            Absolute value of the function.

        arg : np.ndarray [n2 x n1]
            Argument of the function.

        img : np.ndarray [n2 x n1 x 3]
            RGB image values computed from the argument of the function.
            0 <= R, G, B <= 255

        colors : np.ndarray [256 x 3]
            Color scale associated to `img`.
        """
        np = import_module('numpy')

        domain, z = self._evaluate(False)
        return self._apply_transform(
            np.real(domain), np.imag(domain),
            np.absolute(z), np.angle(z),
            *self._domain_coloring(domain, z),
        )


class ComplexParametric3DLineSeries(Parametric3DLineSeries):
    """Represent a mesh/wireframe line of a complex surface series.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # determines what data to return on the z-axis
        self._return = kwargs.get("return", None)

    def _adaptive_sampling(self):
        raise NotImplementedError

    def _uniform_sampling(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')

        results = self._evaluate()
        for i in range(len(results) - 1):
            _re, _im = np.real(results[i]), np.imag(results[i])
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        if self._return is None:
            pass
        elif self._return == "real":
            results[-1] = np.real(results[-1])
        elif self._return == "imag":
            results[-1] = np.imag(results[-1])
        elif self._return == "abs" or self._return == "absarg":
            results[-1] = np.absolute(results[-1])
        elif self._return == "arg":
            results[-1] = np.angle(results[-1])
        else:
            raise ValueError("`_return` not recognized. Received: %s" % self._return)

        return [*results[1:], results[0]]


def _set_discretization_points(kwargs, pt):
    """Allow the use of the keyword arguments n, n1 and n2 (and n3) to
    specify the number of discretization points in two (or three) directions.

    Parameters
    ==========

    kwargs : dict

    pt : type
        The type of the series, which indicates the kind of plot we are
        trying to create.

    Returns
    =======

    kwargs : dict
    """
    deprecated_keywords = {
        "nb_of_points": "n",
        "nb_of_points_x": "n1",
        "nb_of_points_y": "n2",
        "nb_of_points_u": "n1",
        "nb_of_points_v": "n2",
        "points": "n"
    }
    for k, v in deprecated_keywords.items():
        if k in kwargs.keys():
            kwargs[v] = kwargs.pop(k)
            # TODO: follow sympy doc procedure to create this deprecation
            sympy_deprecation_warning(
                f"`{k}` is deprecated. Use `{v}` instead.",
                deprecated_since_version="1.12",
                active_deprecations_target='---')

    if pt in [LineOver1DRangeSeries, Parametric2DLineSeries,
        Parametric3DLineSeries, AbsArgLineSeries, ColoredLineOver1DRangeSeries,
        ComplexParametric3DLineSeries, NyquistLineSeries]:
        if "n" in kwargs.keys():
            kwargs["n1"] = kwargs["n"]
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 0):
                kwargs["n1"] = kwargs["n"][0]
    elif pt in [
        SurfaceOver2DRangeSeries, ContourSeries, ParametricSurfaceSeries,
        ComplexSurfaceSeries, ComplexDomainColoringSeries,
        Vector2DSeries, ImplicitSeries, RiemannSphereSeries
    ]:
        if "n" in kwargs.keys():
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 1):
                kwargs["n1"] = kwargs["n"][0]
                kwargs["n2"] = kwargs["n"][1]
            else:
                kwargs["n1"] = kwargs["n2"] = kwargs["n"]
    elif pt in [Vector3DSeries, SliceVector3DSeries, Implicit3DSeries,
        PlaneSeries]:
        if "n" in kwargs.keys():
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 2):
                kwargs["n1"] = kwargs["n"][0]
                kwargs["n2"] = kwargs["n"][1]
                kwargs["n3"] = kwargs["n"][2]
            else:
                kwargs["n1"] = kwargs["n2"] = kwargs["n3"] = kwargs["n"]
    return kwargs


class VectorBase(BaseSeries):
    """Represent a vector field."""

    is_vector = True
    is_slice = False
    is_streamlines = False
    _allowed_keys = ["n1", "n2", "n3", "modules", "only_integers",
        "streamlines", "use_cm", "xscale", "yscale", "zscale", "quiver_kw",
        "stream_kw", "rendering_kw", "tx", "ty", "tz", "normalize",
        "color_func", "colorbar"]

    def __init__(self, exprs, ranges, label, **kwargs):
        super().__init__(**kwargs)
        self.expr = tuple([e if callable(e) else sympify(e) for e in exprs])
        self.ranges = ranges
        self._label = str(exprs) if label is None else label
        self._latex_label = latex(exprs) if label is None else label
        self.is_streamlines = kwargs.get("streamlines", False)
        self.use_cm = kwargs.get("use_cm", True)
        self.color_func = kwargs.get("color_func", None)
        # NOTE: normalization is achieved at the backend side: this allows to
        # obtain same length arrows, but colored with the actual magnitude.
        # If normalization is applied on the series get_data(), the coloring
        # by magnitude would not be applicable at the backend.
        self.normalize = kwargs.get("normalize", False)

        # if the expressions are lambda functions and no label has been
        # provided, then its better to do the following in order to avoid
        # suprises on the backend
        if any(callable(e) for e in self.expr):
            if self._label == str(self.expr):
                self._label = "Magnitude"

        # NOTE: when plotting vector fields it might be useful to repeat the
        # plot command switching between quivers and streamlines.
        # Usually, plotting libraries expose different functions for quivers
        # and streamlines, accepting different keyword arguments.
        # The choice to implement separates stream_kw and quiver_kw allows
        # this quick switch.
        if self.is_streamlines:
            self.rendering_kw = kwargs.get("stream_kw",
                kwargs.get("rendering_kw", dict()))
        else:
            self.rendering_kw = kwargs.get("quiver_kw",
                kwargs.get("rendering_kw", dict()))
        self._post_init()

    def get_label(self, use_latex=False, wrapper="$%s$"):
        if use_latex:
            expr = self.expr
            if self._label != str(expr):
                return self._latex_label
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._label

    def get_data(self):
        """Return arrays of coordinates for plotting. Depending on the
        `adaptive` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        Returns
        =======

        mesh_x : np.ndarray [n2 x n1]
            Discretized x-domain.

        mesh_y : np.ndarray [n2 x n1]
            Discretized y-domain.

        mesh_z : np.ndarray [n2 x n1] (optional)
            Discretized z-domain in the case of Vector3DSeries.

        u : np.ndarray [n2 x n1]
            First component of the vector field.

        v : np.ndarray [n2 x n1]
            Second component of the vector field.

        w : np.ndarray [n2 x n1] (optional)
            Third component of the vector field in the case of Vector3DSeries.
        """
        np = import_module('numpy')

        results = self._evaluate()
        for i, r in enumerate(results):
            re_v, im_v = np.real(r), np.imag(r)
            re_v[np.invert(np.isclose(im_v, np.zeros_like(im_v)))] = np.nan
            results[i] = re_v

        return self._apply_transform(*results)


class Vector2DSeries(VectorBase):
    """Represents a 2D vector field."""

    is_2Dvector = True
    # default number of discretization points
    _N = 25

    def __init__(self, u, v, range1, range2, label="", **kwargs):
        super().__init__((u, v), (range1, range2), label, **kwargs)
        self._allowed_keys += ["scalar"]
        self._set_use_quiver_solid_color(**kwargs)

    def _set_use_quiver_solid_color(self, **kwargs):
        # NOTE: this attribute will inform the backend wheter to use a
        # color map or a solid color for the quivers. It is placed here
        # because it simplifies the backend logic when dealing with
        # plot sums.
        self.use_quiver_solid_color = (
            True
            if ("scalar" not in kwargs.keys())
            else (
                False
                if (not kwargs["scalar"]) or (kwargs["scalar"] is None)
                else True
            )
        )

    def __str__(self):
        ranges = []
        f = lambda t: t if len(t.free_symbols) > 0 else float(t)
        for r in self.ranges:
            ranges.append((r[0], f(r[1]), f(r[2])))
        return self._str_helper(
            "2D vector series: [%s, %s] over %s, %s" % (
            *self.expr, *ranges))


class Vector3DSeries(VectorBase):
    """Represents a 3D vector field."""

    is_3D = True
    is_3Dvector = True
    # default number of discretization points
    _N = 10

    def __init__(self, u, v, z, range1, range2, range3, label="", **kwargs):
        super().__init__((u, v, z), (range1, range2, range3), label, **kwargs)
        if self.is_streamlines and isinstance(self.color_func, Expr):
            raise TypeError("Vector3DSeries with streamlines can't use "
                "symbolic `color_func`.")

    def __str__(self):
        ranges = []
        for r in self.ranges:
            ranges.append((r[0], float(r[1]), float(r[2])))
        return self._str_helper(
            "3D vector series: [%s, %s, %s] over %s, %s, %s" % (
            *self.expr, *ranges))


def _build_slice_series(slice_surf, ranges, **kwargs):
    if isinstance(slice_surf, Plane):
        return PlaneSeries(sympify(slice_surf), *ranges, **kwargs)
    elif isinstance(slice_surf, BaseSeries):
        if slice_surf.is_3Dsurface:
            return slice_surf
        raise TypeError("Only 3D surface-related series are supported.")

    # If the vector field is V(x, y, z), the slice expression can be f(x, y)
    # or f(y, z) or f(x, z). Extract the correct ranges.
    fs = slice_surf.free_symbols
    new_ranges = [r for r in ranges if r[0] in fs]
    # apply the correct discretization number
    n = [
        int(kwargs.get("n1", Vector3DSeries._N)),
        int(kwargs.get("n2", Vector3DSeries._N)),
        int(kwargs.get("n3", Vector3DSeries._N))]
    discr_symbols = [r[0] for r in ranges]
    idx = [discr_symbols.index(s) for s in [r[0] for r in new_ranges]]
    kwargs2 = kwargs.copy()
    kwargs2["n1"] = n[idx[0]]
    kwargs2["n2"] = n[idx[1]]

    return SurfaceOver2DRangeSeries(slice_surf, *new_ranges, **kwargs2)


class SliceVector3DSeries(Vector3DSeries):
    """Represents a 3D vector field plotted over a slice. The slice can be
    a Plane or a surface.
    """
    is_slice = True

    def __init__(self, slice_surf, u, v, w, range_x, range_y, range_z, label="", **kwargs):
        self.slice_surf_series = _build_slice_series(slice_surf, [range_x, range_y, range_z], **kwargs)
        super().__init__(u, v, w, range_x, range_y, range_z, label, **kwargs)

    def _discretize(self):
        data = self.slice_surf_series.get_data()
        if isinstance(self.slice_surf_series, PlaneSeries):
            return data
        if self.slice_surf_series.is_parametric:
            return data[:3]

        # symbols used by this vector's discretization
        discr_symbols = [r[0] for r in self.ranges]
        # ordered symbols from slice_surf_series
        order = self._discretize_helper(discr_symbols)
        return [data[k] for k in order]

    def _discretize_helper(self, vec_discr_symbols):
        # NOTE: let's say the vector field is discretized along x, y, z (in
        # this order), and the slice surface is f(y, z). Then, data will be
        # [yy, zz, f(yy, zz)], which has not the correct order expected by
        # the vector field's discretization. Here we are going to fix that.

        if not isinstance(self.slice_surf_series, SurfaceOver2DRangeSeries):
            raise TypeError("This helper function is meant to be used only "
                "with non-parametric slicing surfaces of 2 variables. "
                "type(self.slice_surf_series) = {}".format(
                    type(self.slice_surf_series)))

        # slice surface free symbols
        # don't use self.slice_surf_series.free_symbols as this expression
        # might not use both its discretization symbols
        ssfs = [r[0] for r in self.slice_surf_series.ranges]
        # given f(y, z), we already have y, z (ssfs), now find x
        missing_symbol = list(set(vec_discr_symbols).difference(ssfs))
        # ordered symbols in the returned data
        returned_symbols = ssfs + missing_symbol
        # output order
        order = [returned_symbols.index(s) for s in vec_discr_symbols]
        return order

    def _create_discretized_domain(self):
        """Discretize the ranges in case of uniform meshing strategy.
        """
        # TODO: once InteractiveSeries has been remove, the following can
        # be reorganized in order to remove code repetition, specifically the
        # following line of code.
        # symbols used by this vector's discretization
        discr_symbols = [r[0] for r in self.ranges]
        discretizations = self._discretize()
        self._discretized_domain = {k: v for k, v in zip(discr_symbols, discretizations)}

    @property
    def params(self):
        """Get or set the current parameters dictionary.

        Parameters
        ==========

        p : dict
            key: symbol associated to the parameter
            val: the value
        """
        return self._params

    @params.setter
    def params(self, p):
        self._params = p
        if self.slice_surf_series.is_interactive:
            # update both parameters and discretization ranges
            self.slice_surf_series.params = p
        # symbols used by this vector's discretization
        discr_symbols = [r[0] for r in self.ranges]

        if isinstance(self.slice_surf_series, SurfaceOver2DRangeSeries) and (not self.slice_surf_series.is_parametric):
            # ordered symbols from slice_surf_series
            ordered_symbols = self._discretize_helper(discr_symbols)
            data = self.slice_surf_series.get_data()
            self._discretized_domain = {
                k: data[v] for k, v in zip(discr_symbols, ordered_symbols)
            }
        else:
            self._discretized_domain = {
                k: v for k, v in zip(discr_symbols, self.slice_surf_series.get_data())
            }

    def __str__(self):
        return "sliced " + super().__str__() + " at {}".format(
            self.slice_surf_series)


class PlaneSeries(SurfaceBaseSeries):
    """Represents a plane in a 3D domain."""

    is_3Dsurface = True
    _N = 20

    # a generic plane (for example with normal (1,1,1)) can generate a huge
    # range along the z-direction. With _use_nan=True, every z-value outside
    # of the provided z_range will be set to Nan.
    _use_nan = True

    def __init__(
        self, plane, x_range, y_range, z_range=None, label="", **kwargs
    ):
        super().__init__(**kwargs)
        self._block_lambda_functions(plane)
        self.plane = sympify(plane)
        self.expr = self.plane
        if not isinstance(self.plane, Plane):
            raise TypeError("`plane` must be an instance of sympy.geometry.Plane")
        self.x_range = sympify(x_range)
        self.y_range = sympify(y_range)
        self.z_range = sympify(z_range)
        self.ranges = [self.x_range, self.y_range, self.z_range]
        self.use_cm = kwargs.get("use_cm", cfg["plot3d"]["use_cm"])
        self._set_surface_label(label)
        self.surface_color = kwargs.get("surface_color", None)
        self.color_func = kwargs.get("color_func", lambda x, y, z: z)
        if self.params and not self.plane.free_symbols:
            self.params = dict()
            self.is_interactive = False

    def __str__(self):
        return self._str_helper(
            "plane series: %s over %s, %s, %s" % (
            self.plane, self.x_range, self.y_range, self.z_range))

    def get_data(self):
        np = import_module('numpy')

        x, y, z = symbols("x, y, z")
        plane = self.plane.subs(self._params)
        fs = plane.equation(x, y, z).free_symbols
        xx, yy, zz = None, None, None
        if fs == set([x]):
            # parallel to yz plane (normal vector (1, 0, 0))
            s = SurfaceOver2DRangeSeries(
                plane.p1[0],
                (x, *self.z_range[1:]),
                (y, *self.y_range[1:]),
                "",
                n1=self.n[2],
                n2=self.n[1],
                xscale=self._scales[0],
                yscale=self._scales[1]
            )
            xx, yy, zz = s.get_data()
            xx, yy, zz = zz, yy, xx
        elif fs == set([y]):
            # parallel to xz plane (normal vector (0, 1, 0))
            s = SurfaceOver2DRangeSeries(
                plane.p1[1],
                (x, *self.x_range[1:]),
                (y, *self.z_range[1:]),
                "",
                n1=self.n[0],
                n2=self.n[2],
                xscale=self._scales[0],
                yscale=self._scales[1]
            )
            xx, yy, zz = s.get_data()
            xx, yy, zz = xx, zz, yy
        elif fs == set([x, y]):
            # vertical plane oriented with some angle

            # Get numpy vectors
            p1 = np.array(plane.p1, dtype=float)
            nv = np.array(plane.normal_vector, dtype=float)
            # convert the normal vector to unit normal vector
            nv = nv / np.sqrt(nv.T @ nv)

            # plane has distance to origin as length of projection of p1 onto normal vector
            proj_p2nv = nv.dot(p1)

            s = SurfaceOver2DRangeSeries(
                proj_p2nv,
                (x, *self.x_range[1:]),
                (y, *self.z_range[1:]),
                "",
                n1=self.n[0],
                n2=self.n[2],
                xscale=self._scales[0],
                yscale=self._scales[1]
            )
            xx, yy, zz = s.get_data()
            xx, yy, zz = xx, zz, yy

            # rotate plane corresponding to the normal vector
            def R(t):
                return np.array([
                    [np.cos(t), -np.sin(t), 0],
                    [np.sin(t), np.cos(t), 0],
                    [0, 0, 1]
                ])

            theta = np.arctan2(nv[1], nv[0])
            coords = np.stack([t.flatten() for t in [xx, yy, np.ones_like(xx)]]).T
            coords = np.matmul(coords, R(theta))
            yy, xx = coords[:, 0].reshape(yy.shape), coords[:, 1].reshape(xx.shape)
        else:
            # any other plane
            eq = plane.equation(x, y, z)
            if z in eq.free_symbols:
                eq = solve(eq, z)[0]
            s = SurfaceOver2DRangeSeries(
                eq,
                (x, *self.x_range[1:]),
                (y, *self.y_range[1:]),
                "",
                n1=self.n[0],
                n2=self.n[1],
                xscale=self._scales[0],
                yscale=self._scales[1]
            )
            xx, yy, zz = s.get_data()
            if (len(fs) > 1) and self._use_nan:
                idx = np.logical_or(zz < self.z_range[1], zz > self.z_range[2])
                zz[idx] = np.nan
        return xx, yy, zz


class GeometrySeries(BaseSeries):
    """Represents an entity from the sympy.geometry module.
    Depending on the geometry entity, this class can either represents a
    point, a line, or a parametric line
    """

    is_geometry = True
    _allowed_keys = ["is_filled", "use_cm", "color_func", "line_color",
    "rendering_kw", "colorbar"]

    def __new__(cls, *args, **kwargs):
        if isinstance(args[0], Plane):
            return PlaneSeries(*args, **kwargs)
        elif isinstance(args[0], Curve):
            new_cls = (
                Parametric2DLineSeries
                if len(args[0].functions) == 2
                else Parametric3DLineSeries
            )
            label = [a for a in args if isinstance(a, str)]
            label = label[0] if len(label) > 0 else str(args[0])
            return new_cls(*args[0].functions, args[0].limits, label, **kwargs)
        return object.__new__(cls)

    def __init__(self, expr, _range=None, label="", **kwargs):
        if not isinstance(expr, GeometryEntity):
            raise ValueError(
                "`expr` must be a geomtric entity.\n"
                + "Received: type(expr) = {}\n".format(type(expr))
                + "Expr: {}".format(expr)
            )
        super().__init__(**kwargs)
        r = expr.free_symbols.difference(set(self.params.keys()))
        if len(r) > 0:
            raise ValueError(
                "Too many free symbols. Please, specify the values of the "
                + "following symbols with the `params` dictionary: {}".format(r)
            )
        self._expr = expr
        self.ranges = [_range]
        self._label = str(expr) if label is None else label
        self._latex_label = latex(expr) if label is None else label
        self.is_filled = kwargs.get("is_filled", True)
        self.n = int(kwargs.get("n", 200))
        self.use_cm = kwargs.get("use_cm", False)
        self.color_func = kwargs.get("color_func", None)
        self.line_color = kwargs.get("line_color", None)
        if isinstance(expr, (LinearEntity3D, Point3D)):
            self.is_3Dline = True
            self.is_parametric = False
            self.start = 0
            self.end = 0
            if isinstance(expr, Point3D):
                self.is_point = True
        elif isinstance(expr, LinearEntity2D) or (
            isinstance(expr, (Polygon, Circle, Ellipse)) and (not self.is_filled)
        ):
            self.is_2Dline = True
        elif isinstance(expr, Point2D):
            self.is_point = True
            self.is_2Dline = True
        self.rendering_kw = kwargs.get("rendering_kw", dict())

    def get_data(self):
        np = import_module('numpy')

        expr = self.expr.subs(self.params)
        if isinstance(expr, Point3D):
            return (
                np.array([expr.x], dtype=float),
                np.array([expr.y], dtype=float),
                np.array([expr.z], dtype=float),
                # np.array([0], dtype=float),
            )
        elif isinstance(expr, Point2D):
            return np.array([expr.x], dtype=float), np.array([expr.y], dtype=float)
        elif isinstance(expr, Polygon):
            x = [float(v.x) for v in expr.vertices]
            y = [float(v.y) for v in expr.vertices]
            x.append(x[0])
            y.append(y[0])
            return np.array(x), np.array(y)
        elif isinstance(expr, Circle):
            cx, cy = float(expr.center[0]), float(expr.center[1])
            r = float(expr.radius)
            t = np.linspace(0, 2 * np.pi, self.n[0])
            x, y = cx + r * np.cos(t), cy + r * np.sin(t)
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            return x, y
        elif isinstance(expr, Ellipse):
            cx, cy = float(expr.center[0]), float(expr.center[1])
            a = float(expr.hradius)
            e = float(expr.eccentricity)
            x = np.linspace(-a, a, self.n[0])
            y = np.sqrt((a ** 2 - x ** 2) * (1 - e ** 2))
            x += cx
            x, y = np.concatenate((x, x[::-1])), np.concatenate((cy + y, cy - y[::-1]))
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            return x, y
        elif isinstance(expr, LinearEntity3D):
            p1, p2 = expr.points
            x = np.array([p1.x, p2.x], dtype=float)
            y = np.array([p1.y, p2.y], dtype=float)
            z = np.array([p1.z, p2.z], dtype=float)
            return x, y, z
        elif isinstance(expr, (Segment, Ray)):
            p1, p2 = expr.points
            x = np.array([p1.x, p2.x])
            y = np.array([p1.y, p2.y])
            return x.astype(float), y.astype(float)
        else:  # Line
            p1, p2 = expr.points
            if not self.ranges:
                x = np.array([p1.x, p2.x])
                y = np.array([p1.y, p2.y])
            else:
                _range = self.ranges[0]
                m = expr.slope
                q = p1[1] - m * p1[0]
                x = np.array([_range[1], _range[2]])
                y = m * x + q
            return x.astype(float), y.astype(float)

    def __str__(self):
        return self._str_helper("geometry entity: %s" % str(self.expr))


class GenericDataSeries(BaseSeries):
    """Represents generic numerical data.

    NOTE:
    This class implements back-compatibility with Sympy <=1.11: its plotting
    module accepts the following keyword arguments:

    annotations, markers, rectangles, fill

    Sadly, the developers forgot to properly document them: there are no
    example whatsoever about their usage. This is actually a very good thing
    for this new plotting module, which supports multiple backends.
    Every backend exposes different functions:

    1. For example, to create line plots Matplotlib exposes ``ax.plot``,
       whereas Plotly exposes ``go.Scatter``, whereas Bokeh exposes
       ``fig.line``, etc. But those different ways do not overlap completely:
       with ``go.Scatter`` it's also possible to create filled regions,
       whereas with ``ax.plot`` that's not possible.
    2. Moreover, some plotting library exposes functionalities that are
       unmatched by others. For example, Matplotlib's ``ax.fill_between`` is
       substantially different from Plotly's filled area or whatever Bokeh
       exposes. Similarly, Matplotlib's Rectangle is very specific, whereas
       with Plotly we can add any shape (rectangle, line, ...) with the same
       function call.

    So, the problem is clear: if developers document a feature to do one
    specific thing, users expect it to produce consistent results across
    backends. This is clearly impossible to achieve.

    There is also the problem of when "enough is enough"? Meaning, who is to
    stop anyone from adding new keyword arguments that are just wrappers to
    what a plotting library already can do? For example, I could add the
    ``hex_tile`` keyword: it's beautiful for Bokeh, but very difficult
    to implement on other backends. Or maybe I could add ``hlines`` or
    ``vlines`` keyword arguments to add horizontal or vertical lines. If this
    approach was to be followed, we will end up rewriting multiple plotting
    libraries: for what?

    Instead, the goal of this module is to facilitate the plotting of symbolic
    expressions. If user needs to add numerical data to a plot, he/she can
    easily retrieve the figure object and proceed with the usual commands
    associated to a specific plotting library.
    For example, for ``MatplotlibBackend``:

    .. code-block:: python

       from sympy import *
       from spb import *
       import numpy as np
       var("x")

       # plot symbolic expressions
       p = plot(sin(x), cos(x), backend=MB)
       # extract the axes object
       ax = p.fig.axes[0]
       # add numerical data
       xx = np.linspace(-10, 10)
       f = 1 / (1 + np.exp(-xx))
       ax.plot(xx, f1, "k:", label="numerical data")
       ax.legend()
       p.fig

    Hence, the decision to maintain this back-compatibility (for the moment)
    but not to document those keyword arguments on the plotting functions.
    """
    is_generic = True

    def __init__(self, tp, *args, **kwargs):
        super().__init__(**kwargs)
        self.type = tp
        self.args = args
        self.rendering_kw = kwargs

    def get_data(self):
        return self.args


class RiemannSphereSeries(BaseSeries):
    is_complex = True
    is_domain_coloring = True
    is_3Dsurface = True
    _N = 150
    _allowed_keys = ["n1", "n2", "cmap", "coloring", "blevel", "phaseres",
        "phaseoffset"]

    def __init__(self, f, range_t, range_p, **kwargs):
        self._block_lambda_functions(f)
        super().__init__(**kwargs)
        if len(f.free_symbols) > 1:
            # NOTE: considering how computationally heavy this series is,
            # it is rather unuseful to allow interactive-widgets plot.
            raise ValueError(
                "Complex function can only have one free symbol. "
                "Received free symbols: %s" % f.free_symbols)
        self.expr = f
        # NOTE: we can easily create a sphere with a single data series.
        # However, K3DBackend is unable to properly visualize it, and it
        # would require a few hours of work to apply the necessary edits.
        # Instead, I'm going to create two sphere caps (Northen and Southern
        # hemispheres, respectively), hence the need for ranges :D
        self.ranges = [range_t, range_p]
        ComplexDomainColoringSeries._init_domain_coloring_kw(self, **kwargs)
        if self.n[0] == self.n[1]:
            self.n = [self.n[0], 4 * self.n[0]]
        self.use_cm = True

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======

        x, y, z : np.ndarray [n2 x n1]
            Coordinates on the unit sphere.

        arg : np.ndarray [n2 x n1]
            Argument of the function.

        img : np.ndarray [n2 x n1 x 3]
            RGB image values computed from the argument of the function.
            0 <= R, G, B <= 255

        colors : np.ndarray [256 x 3]
            Color scale associated to `img`.
        """
        np = import_module('numpy')

        # discretize the unit sphere
        r = 1
        # TODO: this parameterization places a lot of points near the poles
        # but not enough near the equator. Can a different parameterization
        # improves the final result, maybe even reducing the computational
        # cost?
        ts, te = [float(t) for t in self.ranges[0][1:]]
        ps, pe = [float(t) for t in self.ranges[1][1:]]
        t, p = np.mgrid[ts:te:self.n[0]*1j, ps:pe:self.n[1]*1j]
        X = r * np.sin(t) * np.cos(p)
        Y = r * np.sin(t) * np.sin(p)
        Z = r * np.cos(t)
        # stereographic projection
        # TODO: suppress warnings
        x = X / (1 - Z)
        y = Y / (1 - Z)
        # evaluation over the complex plane
        # NOTE: _uniform_eval should be used, as it is able to deal with
        # different evaluation modules. However, that method is much slower
        # than vanilla-Numpy with vectorization, even when using Numpy.
        # To get decent results, the function must be evaluated on a big
        # number of discretization points, which automatically precludes
        # mpmath or sympy. Hence, just use bare bones Numpy, even though this
        # module might not implement all the interesting functions.
        z = x + 1j * y
        f = lambdify(list(self.expr.free_symbols)[0], self.expr)
        w = f(z)
        img, cs = wegert(self.coloring, w, self.phaseres, self.cmap)
        return X, Y, Z, np.angle(w), img, cs


class HVLineSeries(BaseSeries):
    """Represent an horizontal or vertical line series.
    In Matplotlib, this will be rendered by axhline or axvline.
    """
    def __init__(self, v, horizontal, label="", **kwargs):
        super().__init__(**kwargs)
        self._expr = sympify(v)
        self.is_horizontal = horizontal
        self._label = str(self.expr) if label is None else label
        self._latex_label = latex(self.expr) if label is None else label

    def get_data(self):
        location = self.expr
        if self.is_interactive:
            location = self.expr.subs(self.params)
        return float(location)


class NyquistLineSeries(Parametric2DLineSeries):
    """Represent a line on a Nyquist plot.

    Differently from Parametric2DLineSeries, which returns real `x, y, param`,
    this class returns real `x, y` but complex `param`.

    This class incorporates code from the `python-control` package,
    specifically the `freqplot.py` module
    """
    def __init__(self, tf, omega_range, label="", **kwargs):
        self.tf = tf
        tf_expr = self.tf.to_expr()
        # NOTE/TODO: this might be fragile, it might results in something
        # different then a fraction, creating a wrong denominator which
        # causes wrong poles to be computed, which trigger warnings.
        # Maybe it's better to adopt `python-control` approach to compute
        # the feedback system transfer function
        tf_cl_expr = tf_expr / (1 + tf_expr).together()
        num, den = tf_cl_expr.as_numer_denom()
        self.tf_cl = TransferFunction(num, den, self.tf.var)

        super().__init__(re(tf_expr), im(tf_expr), omega_range, label, **kwargs)

        self.omega_range_given = kwargs.get("omega_range_given", False)
        self.m_circles = kwargs.get("m_circles", False)
        self.clabels_to_top = kwargs.get("clabels_to_top", True)
        self.arrows_loc = kwargs.get("arrows", 2)
        self.indent_points = kwargs.get("indent_points", 50)
        self.indent_radius = kwargs.get("indent_radius", 1e-04)
        self.indent_direction = kwargs.get("indent_direction", 'right')
        self.max_curve_magnitude = kwargs.get("max_curve_magnitude", 20)
        self.max_curve_offset = kwargs.get("max_curve_offset", 0.02)
        self.encirclement_threshold = kwargs.get("encirclement_threshold", 0.05)
        self.warn_encirclements = kwargs.get("warn_encirclements", True)
        self.start_marker = kwargs.get("start_marker", True)
        self.primary_style = kwargs.get("primary_style", None)
        self.mirror_style = kwargs.get("mirror_style", None)
        self._update_poles()

        self._allowed_keys += [
            "m_circles", "clabels_to_top", "arrows", "indent_points",
            "indent_radius", "indent_direction",
            "max_curve_magnitude", "max_curve_offset",
            "encirclement_threshold", "omega_range_given", "start_marker",
            "primary_style", "mirror_style"]

    def _update_poles(self):
        """Computes the poles of the open-loop transfer function as well as
        the closed-loop transfer function, which will be used to raise
        appropriate warnings, if necessary.
        """
        np = import_module('numpy')

        # NOTE: at instantiation, `self.params` contains the tuples provided
        # by user. Can't proceed in that case.
        go_on = True
        if len(self.params) > 0:
            first_param = list(self.params.values())[0]
            if isinstance(first_param, (list, tuple)):
                go_on = False

        if go_on:
            s = self.tf.var
            den_tf_poly = Poly(self.tf.den.subs(self.params), s)
            den_tf_poly = np.array(den_tf_poly.all_coeffs(), dtype=np.complex128)
            self._poles = np.roots(den_tf_poly)

            den_tf_cl_poly = Poly(self.tf_cl.den.subs(self.params), s)
            den_tf_cl_poly = np.array(den_tf_cl_poly.all_coeffs(), dtype=np.complex128)
            self._poles_cl = np.roots(den_tf_cl_poly)

    @property
    def arrows_loc(self):
        return self._arrows_loc

    @arrows_loc.setter
    def arrows_loc(self, v):
        """Set arrows position."""
        np = import_module('numpy')

        # from https://github.com/python-control/python-control/blob/main/control/freqplot.py
        if not v:
            self._arrows_loc = []
        elif isinstance(v, int):
            N = v
            # Space arrows out, starting midway along each "region"
            self._arrows_loc = np.linspace(0.5/N, 1 + 0.5/N, N, endpoint=False)
        elif isinstance(v, (list, np.ndarray)):
            self._arrows_loc = np.sort(np.atleast_1d(v))
        else:
            raise ValueError("unknown or unsupported arrow location")

    def _create_discretized_domain(self):
        """Discretize the ranges for uniform meshing strategy.
        """
        np = import_module('numpy')

        r = self.ranges[0]
        discr_symbols = [r[0]]

        c_start = self._update_range_value(r[1])
        c_end = self._update_range_value(r[2])
        start = c_start.real if c_start.imag == c_end.imag == 0 else c_start
        end = c_end.real if c_start.imag == c_end.imag == 0 else c_end

        # NOTE: here we generate n points. However, depending on the input
        # arguments and/or on the transfer function, the final number of points
        # will be greater!
        omega = BaseSeries._discretize(
            start, end, self.n[0], scale=self.scales[0])

        if not self.omega_range_given:
            omega[0] = 0
            # TODO: the following should be better, but is it really necessary?
            # it makes testing more difficult...
            # d = np.concatenate((
            #         np.linspace(0, d[0], self.indent_points), d[1:]))

        omega= 1j * omega
        omega = self._modify_discretization(omega)
        discretizations = [omega]
        self._create_discretized_domain_helper(discr_symbols, discretizations)

    @property
    def params(self):
        """Get or set the current parameters dictionary.

        Parameters
        ==========

        p : dict

            * key: symbol associated to the parameter
            * val: the numeric value
        """
        return self._params

    @params.setter
    def params(self, p):
        self._params = p
        self._update_poles()

    def _modify_discretization(self, splane_contour):
        # from https://github.com/python-control/python-control/blob/main/control/freqplot.py

        np = import_module('numpy')
        splane_poles = self._poles
        splane_cl_poles = self._poles_cl
        indent_points = self.indent_points
        indent_radius = self.indent_radius
        indent_direction = self.indent_direction
        warn_encirclements = self.warn_encirclements

        #
        # Check to make sure indent radius is small enough
        #
        # If there is a closed loop pole that is near the imaginary axis
        # at a point that is near an open loop pole, it is possible that
        # indentation might skip or create an extraneous encirclement.
        # We check for that situation here and generate a warning if that
        # could happen.
        #
        for p_cl in splane_cl_poles:
            # See if any closed loop poles are near the imaginary axis
            if abs(p_cl.real) <= indent_radius:
                # See if any open loop poles are close to closed loop poles
                if len(splane_poles) > 0:
                    p_ol = splane_poles[
                        (np.abs(splane_poles - p_cl)).argmin()]

                    if abs(p_ol - p_cl) <= indent_radius and \
                            warn_encirclements:
                        warnings.warn(
                            "indented contour may miss closed loop pole; "
                            "consider reducing indent_radius to below "
                            f"{abs(p_ol - p_cl):5.2g}", stacklevel=2)

        #
        # See if we should add some frequency points near imaginary poles
        #
        for p in splane_poles:
            # print("adding points")
            # See if we need to process this pole (skip if on the negative
            # imaginary axis or not near imaginary axis + user override)
            if p.imag < 0 or abs(p.real) > indent_radius:
                continue

            # Find the frequencies before the pole frequency
            below_points = np.argwhere(
                splane_contour.imag - abs(p.imag) < -indent_radius)
            if below_points.size > 0:
                first_point = below_points[-1].item()
                start_freq = p.imag - indent_radius
            else:
                # Add the points starting at the beginning of the contour
                assert splane_contour[0] == 0
                first_point = 0
                start_freq = 0

            # Find the frequencies after the pole frequency
            above_points = np.argwhere(
                splane_contour.imag - abs(p.imag) > indent_radius)
            last_point = above_points[0].item()

            # Add points for half/quarter circle around pole frequency
            # (these will get indented left or right below)
            splane_contour = np.concatenate((
                splane_contour[0:first_point+1],
                (1j * np.linspace(
                    start_freq, p.imag + indent_radius, indent_points)),
                splane_contour[last_point:]))

        # Indent points that are too close to a pole
        if len(splane_poles) > 0: # accomodate no splane poles if dtime sys
            for i, s in enumerate(splane_contour):
                # Find the nearest pole
                p = splane_poles[(np.abs(splane_poles - s)).argmin()]

                # See if we need to indent around it
                if abs(s - p) < indent_radius:
                    # Figure out how much to offset (simple trigonometry)
                    offset = np.sqrt(indent_radius ** 2 - (s - p).imag ** 2) \
                        - (s - p).real

                    # Figure out which way to offset the contour point
                    if p.real < 0 or (p.real == 0 and
                                    indent_direction == 'right'):
                        # Indent to the right
                        splane_contour[i] += offset

                    elif p.real > 0 or (p.real == 0 and
                                        indent_direction == 'left'):
                        # Indent to the left
                        splane_contour[i] -= offset

                    else:
                        raise ValueError("unknown value for indent_direction")

        return splane_contour

    def _uniform_sampling(self):
        """Returns coordinates that needs to be postprocessed."""
        # print("orci ")
        np = import_module('numpy')

        results = self._evaluate(cast_to_real=False)
        for i, r in enumerate(results):
            if i != 0:
                _re, _im = np.real(r), np.imag(r)
                _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
                results[i] = _re
        # print([t.dtype for t in results])
        return [*results[1:], results[0]]
