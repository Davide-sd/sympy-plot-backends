import param
from inspect import signature, ismodule
from spb.defaults import cfg
from spb.utils import _get_free_symbols, _correct_shape
from sympy import (
    Tuple, symbols, sympify, Expr, lambdify,
    atan2, floor, ceiling, Sum, Product, Symbol, frac, im, re, zeta, Poly,
    Integral, hyper, arity
)
from sympy.external import import_module
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.precedence import precedence
from sympy.core.sorting import default_sort_key
import warnings
from numbers import Number


def format_warnings_on_one_line(
    message, category, filename, lineno, file=None, line=None
):
    # https://stackoverflow.com/a/26433913/2329968
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


warnings.formatwarning = format_warnings_on_one_line


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
        return " | ".join(
            self.parenthesize(a, PREC)
            for a in sorted(expr.args, key=default_sort_key)
        )


def _warning_eval_error(err, modules):
    warnings.warn(
        "The evaluation with %s failed.\n" % (
            "NumPy/SciPy" if not modules else modules) +
        "{}: {}\n".format(type(err).__name__, err) +
        "Trying to evaluate the expression with Sympy, but it might "
        "be a slow operation.",
        stacklevel=2
    )


def _get_wrapper_func():
    """
    Create a 'vectorized' wrapper function for numerical evaluation.
    """
    np = import_module("numpy")
    def wrapper_func(func, *args):
        try:
            return complex(func(*args))
        except (ZeroDivisionError, OverflowError):
            return complex(np.nan, np.nan)

    # NOTE: np.vectorize is much slower than numpy vectorized
    # operations. However, this modules must be able to evaluate
    # functions also with mpmath or sympy.
    wrapper_func = np.vectorize(wrapper_func, otypes=[complex])
    return wrapper_func


def _uniform_eval(evaluator):
    """
    Note: this is an experimental function, as such it is prone to changes.
    Please, do not use it in your code.
    """
    np = import_module('numpy')

    series = evaluator.series
    lambdifier = series.lambdifier
    modules = series.modules
    wrapper_func = _get_wrapper_func()

    results = []
    functions = lambdifier.request_lambda_functions(modules)
    # ensure that discretized domains are returned with the proper order
    discr = [evaluator._discretized_domain[s[0]] for s in series.ranges]
    args = evaluator._aggregate_args()

    for i, f in enumerate(functions):
        try:
            r = f(*args)
        except (ValueError, TypeError):
            # attempt to use numpy.vectorize
            r = wrapper_func(f, *args)
        except Exception as err:
            # fall back to sympy
            _warning_eval_error(err, modules)
            sympy_functions = lambdifier.request_lambda_functions("sympy")
            r = wrapper_func(sympy_functions[i], *args)

        # the evaluation might produce an int/float. Need this correction.
        r = _correct_shape(np.array(r), discr[0])
        # sometime the evaluation is performed over arrays of type object.
        # hence, `r` might be of type object, which don't work well
        # with numpy real and imag functions.
        r = r.astype(complex)
        results.append(r)
    return results


class _NMixin:
    @property
    def n(self):
        n1 = self._N if not hasattr(self, "n1") else self.n1
        n2 = self._N if not hasattr(self, "n2") else self.n2
        n3 = self._N if not hasattr(self, "n3") else self.n3
        return [n1, n2, n3]

    @n.setter
    def n(self, value):
        n = [self._N] * 3
        if value is not None:
            if hasattr(value, "__iter__"):
                for i in range(min(len(value), 3)):
                    n[i] = int(value[i])
            else:
                n = [int(value)] * 3

        if hasattr(self, "n1"):
            self.n1 = n[0]
        if hasattr(self, "n2"):
            self.n2 = n[1]
        if hasattr(self, "n3"):
            self.n3 = n[2]


class _GridEvaluationParameters(param.Parameterized, _NMixin):
    _lambdifier = param.Parameter(doc="""
        The machinery that creates lambda functions which are eventually
        going to be evaluated.""")
    evaluator = param.Parameter(doc="""
        An instance of ``GridEvaluator``, which is the machinery that
        generates numerical data starting from the parameters of the
        current series.""")
    force_real_eval = param.Boolean(False, doc="""
        By default, numerical evaluation is performed over complex numbers,
        which is slower but produces correct results.
        However, when the symbolic expression is converted to a numerical
        function with lambdify, the resulting function may not like to
        be evaluated over complex numbers. In such cases, forcing the
        evaluation to be performed over real numbers might be a good choice.
        The plotting module should be able to detect such occurences and
        automatically activate this option. If that is not the case, or
        evaluation performance is of paramount importance, set this parameter
        to True, but be aware that it might produce wrong results.
        It only works with ``adaptive=False``.""")
    only_integers = param.Boolean(False, doc="""
        Discretize the domain using only integer numbers. It only works when
        ``adaptive=False``. When this parameter is True, the number of
        discretization points is choosen by the algorithm.""")
    modules = param.Parameter(None, doc="""
        Specify the evaluation modules to be used by lambdify.
        If not specified, the evaluation will be done with NumPy/SciPy.""")

    @property
    def lambdifier(self):
        return self._lambdifier


def _discretize(start, end, N, scale="linear", only_integers=False):
    """
    Discretize a 1D domain.

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


def _update_range_value(series, expr):
    """
    Given a symbolic expression, substitutes the parameters if
    the series is interactive.
    """
    if not series._parametric_ranges:
        return complex(expr)
    return complex(expr.subs(series.params))


def _hashify_modules(modules):
    """
    Given a user-provided ``modules`` for the ``lambdify`` function,
    create an hash value so that newly generated lambda functions
    can be stored for future evaluations.
    """
    if (
        isinstance(modules, str)
        or ismodule(modules)
        or (modules is None)
    ):
        return hash(modules)
    elif isinstance(modules, list):
        if all(isinstance(t, str) for t in modules):
            return hash("|".join(modules))
        else:
            keys = []
            for t in modules:
                if isinstance(t, str):
                    keys.append(t)
                elif ismodule(t):
                    keys.append(t.__name__)
                elif isinstance(t, dict):
                    keys.extend([str(k) for k in t])
                else:
                    raise NotImplementedError
        return hash("|".join(keys))
    elif isinstance(modules, dict):
        return hash("|".join([str(k) for k in modules]))
    raise NotImplementedError


class Lambdifier(param.Parameterized):
    series = param.Parameter(doc="Data series to be evaluated.")
    expr = param.Parameter(constant=True, doc="""
        Symbolic expressions (or user-provided numerical functions)
        to be evaluated. This class applies some processing to the
        user-provided series' expression, which is then stored in
        this parameter.""")
    _color_func = param.Dict({}, doc="""
        A dictionary mapping hashes of the ``modules`` keyword argument to
        lambda functions of the ``series.color_func``.""")
    _functions = param.Dict({}, doc="""
        A dictionary mapping hashes of the ``modules`` keyword argument to
        lambda functions created with ``lambdify``.""")
    _signature = param.List([], item_type=Expr, doc="""
        Signature of the numerical functions. It is generated by the
        *Series.""")
    _eval_color_func_with_signature = param.Boolean(False, doc="""
        ``color_func`` usually receives numerical functions that are going
        to be evaluated over the coordinates of the computed points (or the
        discretized meshes).
        However, if ``color_func`` is a symbolic expression, then it will be
        lambdified with symbols in self._signature, and it will be evaluated
        with the same data used to evaluate the plotted expression.""")

    def request_lambda_functions(self, modules):
        """
        Create and returns the lambda functions associated to the
        symbolic expressions, to be used by the uniform meshing evaluation.

        Parameters
        ----------
        module :
            The value to be passed to the ``module`` keyword argument of
            ``lambdify``.

        Returns
        -------
        functions : list
            A list of lambda functions, one for each symbolic expression.
            For examples:

            * LineOver1DRangeSeries has 1 symbolic expression: this list
              contains 1 lambda function.
            * Parametric2DLineSeries has 2 symbolic expressions: this list
              contains 2 lambda functions.
            * Parametric32DLineSeries has 3 symbolic expressions: this list
              contains 3 lambda functions.
        """
        h = _hashify_modules(modules)
        if h in self._functions:
            return self._functions[h]

        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        if not any(callable(e) for e in exprs):
            if len(self._signature) == 0:
                fs = _get_free_symbols(exprs)
                self._signature = sorted(fs, key=lambda t: t.name)

            # NOTE: suppose we are dealing with a Parametric3DLineSeries,
            # which have 3 symbolic expressions. Why creating 3 lambda
            # functions when I can lambdify a Tuple(expr1, expr2, expr3), thus
            # obtaining 1 lambda function?
            # Because I need to maximize evaluation performance. Suppose expr1
            # contain a function which NumPy/SciPy is unable to evaluate, for
            # example hyper, while expr2, expr3 contain ordinary functions
            # that NumPy/Scipy can evaluate. If I was using only one lambda
            # function, then an error in the evaluation of expr1 with
            # NumPy/Scipy would force the evaluation algorithm to use SymPy,
            # which is much slower than NumPy/Scipy. SymPy would have to
            # evaluate 3 expressions instead of 1.
            # Instead, by using this approach, the failure in expr1 would force
            # the evaluation algorithm to use SymPy only to evaluate expr1.
            functions = []
            for e in exprs:
                # TODO: set cse=True once this issue is solved:
                # https://github.com/sympy/sympy/issues/24246
                kwargs = dict(modules=modules, docstring_limit=0)
                if modules == "sympy":
                    kwargs["dummify"] = True
                functions.append(lambdify(self._signature, e, **kwargs))
            self._functions[h] = functions
        else:
            if len(self._signature) == 0:
                self._signature = sorted([
                    r[0] for r in self.series.ranges], key=lambda t: t.name)
            self._functions[h] = [e for e in exprs]

        return self._functions[h]

    def request_color_func(self, modules):
        """
        Create and returns the lambda functions associated to the
        series' ``color_func`` attribute, to be used by the uniform
        meshing evaluation.

        Parameters
        ----------
        module :
            The value to be passed to the ``module`` keyword argument of
            ``lambdify``.

        Returns
        -------
        function : callable
            A single lambda function. Its signature depends on the
            user-provided series' ``color_func`` attribute. If a symbolic
            lambda function was provided, then the signature is dictated by
            order of the ranges used in the definition of the data series.
        """
        h = _hashify_modules(modules)
        if h in self._color_func:
            return self._color_func[h]

        if isinstance(self.series.color_func, Expr):
            self._eval_color_func_with_signature = True
            self._color_func[h] = lambdify(
                self._signature, self.series.color_func,
                modules=modules, docstring_limit=0)
        else:
            self._color_func[h] = self.series.color_func
        return self._color_func[h]

    def set_expressions(self):
        """
        Set the expression (or expressions) to be evaluated.
        """
        e = self.series.expr
        is_iter = hasattr(e, "__iter__")
        is_callable = callable(e) if not is_iter else any(callable(t) for t in e)
        if is_callable:
            with param.edit_constant(self):
                self.expr = e
        else:
            with param.edit_constant(self):
                self.expr = sympify(e) if not is_iter else Tuple(*e)
            s = set()
            for e in self.expr.atoms(Sum, Product):
                for a in e.args[1:]:
                    if isinstance(a[-1], Symbol):
                        s.add(a[-1])
            self.series.evaluator._needs_to_be_int = list(s)

            # TODO: move this into GridEval and only raise the warning
            # if adaptive=False
            # TODO: what if the user set force_real_eval=True (or the
            # algorithm did it with the previous expression) but the new
            # expression doesn't need it?
            if self.series.force_real_eval is not True:
                check_res = [
                    self.expr.has(f) for f
                    in self.series.evaluator._problematic_functions
                ]
                self.series.force_real_eval = any(check_res)
                if self.series.force_real_eval and (
                    (self.series.modules is None) or
                    (
                        isinstance(self.series.modules, str)
                        and "numpy" in self.series.modules
                    )
                ):
                    funcs = [
                        f for f, c in zip(
                            self.series.evaluator._problematic_functions,
                            check_res
                        ) if c
                    ]
                    warnings.warn(
                        "NumPy is unable to evaluate with complex "
                        "numbers some of the functions included in this "
                        "symbolic expression: %s. " % funcs +
                        "Hence, the evaluation will use real numbers. "
                        "If you believe the resulting plot is incorrect, "
                        "change the evaluation module by setting the "
                        "`modules` keyword argument.")
            self._functions = {}


class GridEvaluator(param.Parameterized):
    """
    Many plotting functions resemble this form:

    .. code-block:: python

       plot_function(
           expr1, expr2 [opt], ...,
           range1, range2 [opt], ...,
           params=dict()
       )

    Namely, there are one or more symbolic expressions to represent a curve
    or surface, that should be evaluated over one or more ranges, with zero
    or more parameters (whose values come from interactive widgets).

    This class automates the following processes:

    1. Create lambda functions from symbolic expressions. In particular, it
       creates one lambda function to be evaluated with the specified module
       (usually NumPy), and another lambda function to be evaluated with
       SymPy, in case there are any errors with the first.
    2. Create numerical arrays representing ranges, according to the specified
       discretization strategy (linear or logarithmic). Usually, these arrays
       are of type complex, unless ``force_real_eval=True`` is provided in the
       ``plot_function`` call.
    3. Evaluate each lambda function with the appropriate arrays and
       parameters.

    Let ``evaluator`` be an instance of ``GridEvaluator``. Then,
    series requiring this kind of evaluation should call
    ``evaluator._evaluate()`` in order to get numerical data, which should
    then be post-processed.

    Note: it's not mandatory to use this class. For example, control system
    related data series don't need this machinery.
    """

    series = param.Parameter(doc="Data series to be evaluated.")
    _discretized_domain = param.Dict({}, doc="""
        Contain a dictionary with the discretized ranges, used to evaluate
        the numerical functions.""")
    _needs_to_be_int = param.List([], doc="""
        Consider a generic summation, for example:
            s = Sum(cos(pi * x), (x, 1, y))
        This gets lambdified to something like:
            sum(cos(pi*x) for x in range(1, y+1))
        Hence, y needs to be an integer, otherwise it raises:
            TypeError: 'complex' object cannot be interpreted as an integer
        This list will contains symbols that are upper bound to summations
        or products.""")
    _problematic_functions = param.List(
        default=[ceiling, floor, atan2, frac, zeta, Integral, hyper], doc="""
        List of sympy functions that when lambdified, the corresponding
        numpy functions don't like complex-type arguments, and appropriate
        actions should be taken."""
    )

    def __init__(self, **params):
        super().__init__(**params)
        if not hasattr(self.series, "lambdifier"):
            raise ValueError(
                f"{type(self.series).__name__} must expose the"
                " `lambdifier` attribute."
            )

    def _create_discretized_domain(self):
        """
        Discretize the ranges for uniform meshing strategy.
        """
        # NOTE: the goal is to create a dictionary stored in
        # self._discretized_domain, mapping symbols to a numpy array
        # representing the discretization
        discr_symbols = []
        discretizations = []

        # create a 1D discretization
        for i, r in enumerate(self.series.ranges):
            discr_symbols.append(r[0])
            c_start = _update_range_value(self.series, r[1])
            c_end = _update_range_value(self.series, r[2])
            start = c_start.real if c_start.imag == c_end.imag == 0 else c_start
            end = c_end.real if c_start.imag == c_end.imag == 0 else c_end
            needs_integer_discr = self.series.only_integers or (
                r[0] in self._needs_to_be_int)
            d = _discretize(
                start, end, self.series.n[i],
                scale=self.series.scales[i],
                only_integers=needs_integer_discr
            )

            if (
                (not self.series.force_real_eval) and
                (not needs_integer_discr) and
                (d.dtype != "complex")
            ):
                d = d + 1j * c_start.imag

            if needs_integer_discr:
                d = d.astype(int)

            discretizations.append(d)

        # create 2D or 3D
        self._create_discretized_domain_helper(discr_symbols, discretizations)

    def _create_discretized_domain_helper(self, discr_symbols, discretizations):
        """
        Create 2D or 3D discretized grids.

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
        if (
            self.series.is_3Dvector
            or (self.series.is_3Dsurface and self.series.is_implicit)
        ):
            indexing = "ij"
        meshes = np.meshgrid(*discretizations, indexing=indexing)
        self._discretized_domain = {
            k: v for k, v in zip(discr_symbols, meshes)}

    def _evaluate(self, cast_to_real=True):
        """
        Evaluation of the symbolic expression (or expressions) with the
        uniform meshing strategy, based on current values of the parameters.
        """
        np = import_module('numpy')

        # create (or update) the discretized domain
        if (not self._discretized_domain) or self.series._parametric_ranges:
            self._create_discretized_domain()
        # ensure that discretized domains are returned with the proper order
        discr = [self._discretized_domain[s[0]] for s in self.series.ranges]

        results = _uniform_eval(self)

        if cast_to_real:
            discr = [np.real(d.astype(complex)) for d in discr]
        return [*discr, *results]

    def _aggregate_args(self):
        args = []
        for s in self.series.lambdifier._signature:
            if s in self.series.params.keys():
                args.append(
                    int(self.series.params[s]) if s in self._needs_to_be_int else
                    self.series.params[s] if self.series.force_real_eval
                    else complex(self.series.params[s]))
            else:
                args.append(self._discretized_domain[s])
        return args

    def eval_color_func(self, *args):
        """
        Attempt to evaluate the user-provided color function that are instances
        of the `Expr` class.

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
        cf = self.series.lambdifier.request_color_func(self.series.modules)

        if cf is None:
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
            # Similarly, if a 3D surface plot is created with
            # surface_color="red" and `use_cm=True`, then no color_func will
            # be set, and we would end up here, creating constant color across
            # the surface.
            if self.series.use_cm:
                warnings.warn(
                    "This is likely not the result you were  looking for. "
                    "Please, re-execute the plot command, this time "
                    "with the appropriate line_color or surface_color")
            return np.ones_like(args[0])

        if self.series.lambdifier._eval_color_func_with_signature:
            wrapper_func = _get_wrapper_func()
            args = self._aggregate_args()
            lambdifier = self.series.lambdifier
            try:
                color = cf(*args)
            except (ValueError, TypeError):
                # attempt to use numpy.vectorize
                color = wrapper_func(cf, *args)
            except Exception as err:
                # fall back to sympy
                _warning_eval_error(err, self.series.modules)
                cf = lambdifier.request_color_func("sympy")
                color = wrapper_func(cf, *args)

            _re, _im = np.real(color), np.imag(color)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            return _re


class ComplexGridEvaluator(GridEvaluator):
    def _create_discretized_domain(self):
        """
        Discretize the ranges in case of uniform meshing strategy.
        """
        np = import_module('numpy')
        start_x = _update_range_value(self.series, self.series.start).real
        end_x = _update_range_value(self.series, self.series.end).real
        start_y = _update_range_value(self.series, self.series.start).imag
        end_y = _update_range_value(self.series, self.series.end).imag
        x = _discretize(
            start_x, end_x, self.series.n[0], self.series.scales[0],
            self.series.only_integers)
        y = _discretize(
            start_y, end_y, self.series.n[1], self.series.scales[1],
            self.series.only_integers)
        xx, yy = np.meshgrid(x, y)
        domain = xx + 1j * yy
        self._discretized_domain = {self.series.var: domain}


class SliceVectorGridEvaluator(GridEvaluator):
    def __init__(self, **params):
        super().__init__(**params)

    def _discretize(self):
        data = self.series.slice_surf_series.get_data()
        from spb.series.series_2d_3d import PlaneSeries
        if isinstance(self.series.slice_surf_series, PlaneSeries):
            return data
        if self.series.slice_surf_series.is_parametric:
            return data[:3]

        # symbols used by this vector's discretization
        discr_symbols = [r[0] for r in self.series.ranges]
        # ordered symbols from slice_surf_series
        order = self._discretize_helper(discr_symbols)
        return [data[k] for k in order]

    def _discretize_helper(self, vec_discr_symbols):
        # NOTE: let's say the vector field is discretized along x, y, z (in
        # this order), and the slice surface is f(y, z). Then, data will be
        # [yy, zz, f(yy, zz)], which has not the correct order expected by
        # the vector field's discretization. Here we are going to fix that.

        from spb.series.series_2d_3d import SurfaceOver2DRangeSeries
        if not isinstance(self.series.slice_surf_series, SurfaceOver2DRangeSeries):
            raise TypeError("This helper function is meant to be used only "
                "with non-parametric slicing surfaces of 2 variables. "
                "type(self.slice_surf_series) = {}".format(
                    type(self.series.slice_surf_series)))

        # slice surface free symbols
        # don't use self.slice_surf_series.free_symbols as this expression
        # might not use both its discretization symbols
        ssfs = [r[0] for r in self.series.slice_surf_series.ranges]
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
        discr_symbols = [r[0] for r in self.series.ranges]
        discretizations = self._discretize()
        self._discretized_domain = {
            k: v for k, v in zip(discr_symbols, discretizations)}

    def _update_discretized_domain(self):
        if self.series.slice_surf_series.is_interactive:
            # update both parameters and discretization ranges
            self.series.slice_surf_series.params = self.series.params
        # symbols used by this vector's discretization
        discr_symbols = [r[0] for r in self.series.ranges]

        from spb.series.series_2d_3d import SurfaceOver2DRangeSeries
        if (
            isinstance(self.series.slice_surf_series, SurfaceOver2DRangeSeries) and
            (not self.series.slice_surf_series.is_parametric)
        ):
            # ordered symbols from slice_surf_series
            ordered_symbols = self._discretize_helper(discr_symbols)
            data = self.series.slice_surf_series.get_data()
            self._discretized_domain = {
                k: data[v] for k, v in zip(discr_symbols, ordered_symbols)
            }
        else:
            self._discretized_domain = {
                k: v for k, v in zip(
                    discr_symbols,
                    self.series.slice_surf_series.get_data()
                )
            }
