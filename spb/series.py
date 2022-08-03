from inspect import signature
from spb.defaults import cfg
from sympy import (
    latex, Tuple, arity, symbols, sympify, solve, Expr, lambdify,
    Equality, GreaterThan, LessThan, StrictLessThan, StrictGreaterThan,
    Plane, Polygon, Circle, Ellipse, Segment, Ray, Curve, Point2D, Point3D,
)
from sympy.geometry.entity import GeometryEntity
from sympy.geometry.line import LinearEntity2D, LinearEntity3D
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction
from sympy.plotting.intervalmath import interval
from sympy.external import import_module
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.precedence import precedence
from sympy.core.sorting import default_sort_key
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
        min_module_version='0.12.0')
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

    # TODO:
    # As of adaptive 0.13.0, this warning will be raised if the function to
    # be evaluated returns multiple values. The warning is raised somewhere
    # inside adaptive. Let's ignore it until a PR is done to fix it.
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

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
            f = lambdify(free_symbols, expr, modules=modules)
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
            f = lambdify(free_symbols, expr, modules="sympy")
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


def _uniform_eval(free_symbols, expr, *args, modules=None):
    """Convert the expression to a lambda function using the specified
    module. Perform the evaluation and return the results.

    Note: this is an experimental function, as such it is prone to changes.
    Please, do not use it in your code.

    Parameters
    ==========
    free_symbols : tuple or list
        The free symbols associated to ``expr``.

    expr : Expr
        The symbolic expression to be evaluated.

    args :
        The necessary arguments to perform the evaluation.

    modules : str or None
        The evaluation module. Refer to ``lambdify`` for a list of possible
        values. If ``None``, the evaluation will be done with Numpy/Scipy,
        using vectorized operation whenever possible. With other modules,
        the evaluation might be significantly slower.


    Returns
    =======
    data : np.ndarray (N)
        A 1D array containing the results of the evaluation (type complex).
        If the input arguments are 2D arrays of shape [m, n], then N=(m x n).
        No matter the evaluation ``modules``, the array type is going to be
        complex.
    """
    if callable(expr):
        return expr(*args)

    # generate two lambda functions: the default one, and the backup in case
    # of failures with the default one.
    f1 = lambdify(free_symbols, expr, modules=modules)
    f2 = lambdify(free_symbols, expr, modules="sympy")
    return _uniform_eval_helper(f1, f2, *args, modules=modules)


def _uniform_eval_helper(f1, f2, *args, modules=None):
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
    wrapper_func = np.vectorize(wrapper_func)

    try:
        return wrapper_func(f1, *args)
    except Exception as err:
        warnings.warn(
            "The evaluation with %s failed.\n" % (
                "NumPy/SciPy" if not modules else modules) +
            "{}: {}\n".format(type(err).__name__, err) +
            "Trying to evaluate the expression with Sympy, but it might "
            "be a slow operation."
        )
        return wrapper_func(f2, *args)


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
    # Different from is_contour as the colormap in backend will be
    # different

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

    is_point = False
    # If True, the rendering will use points, not lines.

    is_geometry = False
    # If True, it represents an object of the sympy.geometry module

    is_polar = False
    # If True, the backend will attempt to render it on a polar-projection
    # axis, or using a polar discretization if a 3D plot is requested

    use_cm = True
    # Some series might use a colormap as default coloring. Setting this
    # attribute to False will inform the backends to use solid color.

    _allowed_keys = []
    # contains a list of keyword arguments supported by the series. It will be
    # used to validate the user-provided keyword arguments.

    def __init__(self, *args, **kwargs):
        super().__init__()

    def _init_transforms(self, **kwargs):
        self._tx = kwargs.get("tx", None)
        self._ty = kwargs.get("ty", None)
        self._tz = kwargs.get("tz", None)
        if not all(callable(t) or (t is None) for t in [self._tx, self._ty, self._tz]):
            raise TypeError("`tx`, `ty`, `tz` must be functions.")

    def _block_lambda_functions(self, *exprs):
        if any(callable(e) for e in exprs):
            raise TypeError(type(self).__name__ + " requires a symbolic "
                "expression.")

    @property
    def is_3D(self):
        flags3D = [self.is_3Dline, self.is_3Dsurface, self.is_3Dvector]
        return any(flags3D)

    @property
    def is_line(self):
        flagslines = [self.is_2Dline, self.is_3Dline]
        return any(flagslines)

    def _line_surface_color(self, prop, val):
        """This setter enables back-compatibility with sympy.plotting"""
        # NOTE: color_func is set inside the init method of the series.
        # If line_color/surface_color is not a callable, then color_func will
        # be set to None.
        setattr(self, prop, val)
        if callable(val):
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
            evaluate the function, the result can be a scalar (int, float, complex).

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
            np = import_module('numpy')
            warnings.warn("This is likely not the result you were "
                "looking for. Please, re-execute the plot command, this time "
                "with the appropriate line_color or surface_color")
            return np.ones_like(args[0])

        nargs = arity(self.color_func)
        if nargs == 1:
            if self.is_2Dline and self.is_parametric:
                if len(args) == 2:
                    # ColoredLineOver1DRangeSeries
                    return self.color_func(args[0])
                # Parametric2DLineSeries
                return self.color_func(args[2])
            elif self.is_3Dline and self.is_parametric:
                return self.color_func(args[3])
            elif self.is_3Dsurface and self.is_parametric:
                return self.color_func(args[3])
            return self.color_func(args[0])
        elif nargs == 2:
            if self.is_3Dsurface and self.is_parametric:
                return self.color_func(*args[3:])
            return self.color_func(*args[:2])
        return self.color_func(*args[:nargs])

    def get_data(self):
        """Compute and returns the numerical data.

        The number of parameters returned by this method depends on the
        specific instance. If ``s`` is the series, make sure to read
        ``help(s.get_data)`` to understand what it returns.
        """
        raise NotImplementedError

    def _get_wrapped_label(self, label, wrapper):
        """Given a latex representation of an expression, wrap it inside
        some characters. Matplotlib needs $%s%, K3D-Jupyter needs "%s".
        """
        return wrapper % label

    def get_expr(self):
        """Return the expression (or expressions) of the series."""
        return self.expr

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
        if self._label == str(self.get_expr()):
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
        elif (len(args) == 3) and isinstance(self, (Parametric2DLineSeries, Parametric2DLineInteractiveSeries)):
            x, y, u = args
            return (x, y, t(u, self._tz))
        elif len(args) == 3:
            x, y, z = args
            return t(x, self._tx), t(y, self._ty), t(z, self._tz)
        elif (len(args) == 4) and isinstance(self, (Parametric3DLineSeries, Parametric3DLineInteractiveSeries)):
            x, y, z, u = args
            return (x, y, z, t(u, self._tz))
        elif len(args) == 4: # 2D vector plot
            x, y, u, v = args
            return (
                t(x, self._tx), t(y, self._ty),
                t(u, self._tx), t(v, self._ty)
            )
        elif (len(args) == 6) and (not self.is_complex): # 3D vector plot
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


class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines."""

    is_2Dline = True

    def __init__(self, **kwargs):
        super().__init__()
        self.label = None
        self.steps = kwargs.get("steps", False)
        self.only_integers = kwargs.get("only_integers", False)
        self.is_point = kwargs.get("is_point", False)
        self.is_filled = kwargs.get("is_filled", False)
        self.scale = kwargs.get("xscale", "linear")
        self.n = int(kwargs.get("n", 1000))
        self.modules = kwargs.get("modules", None)
        self.adaptive = kwargs.get("adaptive", cfg["adaptive"]["used_by_default"])
        self.adaptive_goal = kwargs.get("adaptive_goal", cfg["adaptive"]["goal"])
        self.loss_fn = kwargs.get("loss_fn", None)
        self.rendering_kw = kwargs.get("rendering_kw", dict())
        self.use_cm = kwargs.get("use_cm", True)
        self.color_func = kwargs.get("color_func", None)
        self.line_color = kwargs.get("line_color", None)
        self._init_transforms(**kwargs)

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

        points = self._get_points()
        points = self._apply_transform(*points)

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
                points = (x, y, z, points[3])
        return points


class List2DSeries(Line2DBaseSeries):
    """Representation for a line consisting of list of points."""

    _allowed_keys = ["adaptive", "adaptive_goal", "color_func", "is_filled",
    "is_point", "is_polar", "line_color", "loss_fn", "modules", "n",
    "only_integers", "rendering_kw", "steps", "use_cm", "xscale",
    "tx", "ty", "tz"]

    def __init__(self, list_x, list_y, label="", **kwargs):
        super().__init__(**kwargs)
        np = import_module('numpy')
        self._block_lambda_functions(list_x, list_y)
        self.list_x = np.array(list_x, dtype=np.float64)
        self.list_y = np.array(list_y, dtype=np.float64)
        if len(list_x) != len(list_y):
            raise ValueError(
                "The two lists of coordinates must have the same "
                "number of elements.\n"
                "Received: len(list_x) = {} ".format(len(self.list_x)) +
                "and len(list_y) = {}".format(len(self.list_y))
            )
        self.is_polar = kwargs.get("is_polar", False)
        self.label = label

    def get_expr(self):
        return self.list_x, self.list_y

    def __str__(self):
        return "list plot"

    def _get_points(self):
        """Returns coordinates that needs to be postprocessed."""
        return self.list_x, self.list_y


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
        if callable(kwargs.get("color_func", None)) or callable(kwargs.get("line_color", None)):
            return super().__new__(ColoredLineOver1DRangeSeries)
        return object.__new__(cls)

    def __init__(self, expr, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self._label = str(self.get_expr()) if label is None else label
        self._latex_label = latex(self.get_expr()) if label is None else label
        self.var = sympify(var_start_end[0])
        # NOTE: even though this class represents a line over a real range,
        # this class serves as a base class for AbsArgLineSeries, which is
        # capable of plotting a line over a complex range.
        if "is_complex" in kwargs.keys():
            self.is_complex = kwargs["is_complex"]
        self.start = complex(var_start_end[1])
        self.end = complex(var_start_end[2])
        if self.start.imag != self.end.imag:
            raise ValueError(
                "%s requires the imaginary " % self.__class__.__name__ +
                "part of the start and end values of the range "
                "to be the same.")
        self.is_polar = kwargs.get("is_polar", False)
        self.detect_poles = kwargs.get("detect_poles", False)
        self.eps = kwargs.get("eps", 0.01)

        # if the expressions is a lambda function and no label has been
        # provided, then its better to do the following to avoid suprises on
        # the backend
        if callable(self.expr):
            if self._label == str(self.expr):
                self.label = ""

    def __str__(self):
        return "cartesian line: %s for %s over %s" % (
            str(self.expr),
            str(self.var),
            str((self.start.real, self.end.real)),
        )

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
            [self.start.real, self.end.real],
            self.start.imag,
            modules=self.modules,
            adaptive_goal=self.adaptive_goal,
            loss_fn=self.loss_fn)
        return data[:, 0], data[:, 1], data[:, 2]

    def _uniform_sampling(self):
        np = import_module('numpy')

        x = xx = self._discretize(self.start.real, self.end.real, self.n, scale=self.scale, only_integers=self.only_integers)

        if self.is_complex:
            xx = xx + 1j * self.start.imag
        elif self.only_integers:
            # NOTE: likely plotting a Sum. The lambdified function is
            # using ``range``, requiring integer arguments.
            xx = xx.astype(int)
            # HACK:
            # However, now xx is of type np.int64. If the expression contains
            # powers, for example 2**(-np.int64(3)) than ValueError is raised.
            # Turns out that by converting to object the evaluation proceed
            # as expected.
            xx = xx.astype(object)

        data = _uniform_eval([self.var], self.expr, xx, modules=self.modules)
        _re, _im = np.real(data), np.imag(data)

        # with uniform sampling, if self.expr is a constant then only one
        # value will be returned, no matter the shape of x.
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

    @staticmethod
    def _detect_poles(x, y, eps=0.01):
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
            if angle >= threshold:
                yy[i + 1] = np.nan
        return x, yy

    def _get_points(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        np = import_module('numpy')

        x, _re, _im = self._get_real_imag()
        # The evaluation could produce complex numbers. Set real elements
        # to NaN where there are non-zero imaginary elements
        _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan

        if self.detect_poles:
            return self._detect_poles(x, _re, self.eps)
        return x, _re


class ColoredLineOver1DRangeSeries(LineOver1DRangeSeries):
    """Represents a 2D line series in which `color_func` is a callable.
    """
    is_parametric = True

    def _get_points(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        x, y = super()._get_points()
        return x, y, self.eval_color_func(x, y)


class AbsArgLineSeries(LineOver1DRangeSeries):
    """Represents the absolute value of a complex function colored by its
    argument over a complex range (a + I*b, c + I * b). Note that the imaginary part of the start and end must be the same.
    """

    is_parametric = True
    is_complex = True

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __str__(self):
        return "cartesian abs-arg line: %s for %s over %s" % (
            str(self.expr),
            str(self.var),
            str((self.start, self.end)),
        )

    def _get_points(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        np = import_module('numpy')

        x, _re, _im = self._get_real_imag()
        _abs = np.sqrt(_re**2 + _im**2)
        _angle = np.arctan2(_im, _re)
        if self.detect_poles:
            _, _abs = self._detect_poles(x, _abs, self.eps)
        return x, _abs, _angle


class ParametricLineBaseSeries(Line2DBaseSeries):
    is_parametric = True
    _allowed_keys = ["adaptive", "adaptive_goal", "color_func", "is_filled",
    "is_point", "line_color", "loss_fn", "modules", "n", "only_integers",
    "rendering_kw", "use_cm", "xscale", "tx", "ty", "tz"]

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
            self._label = str(self.get_expr())
            self._latex_label = latex(self.get_expr())
        # if the expressions is a lambda function and use_cm=False and no label
        # has been provided, then its better to do the following to avoid
        # suprises on the backend
        if any(callable(e) for e in self.get_expr()) and (not self.use_cm):
            if self._label == str(self.get_expr()):
                self.label = ""

    def _eval_component(self, expr, param):
        """Evaluate the specified expression over a predefined
        param-discretization.
        """
        np = import_module('numpy')

        v = _uniform_eval([self.var], expr, param, modules=self.modules)
        re_v, im_v = np.real(v), np.imag(v)
        re_v = self._correct_shape(re_v, param)
        im_v = self._correct_shape(im_v, param)
        re_v[np.invert(np.isclose(im_v, np.zeros_like(im_v)))] = np.nan
        return re_v

    def _adaptive_sampling(self):
        np = import_module('numpy')

        def func(f, is_2Dline, x):
            try:
                w = [complex(t) for t in f(x)]
                return [t.real if np.isclose(t.imag, 0) else np.nan for t in w]
            except (ZeroDivisionError, OverflowError):
                return [np.nan for t in range(2 if is_2Dline else 3)]

        if all(not callable(e) for e in self.get_expr()):
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
            [self.start, self.end],
            self.is_2Dline,
            modules=self.modules,
            adaptive_goal=self.adaptive_goal,
            loss_fn=self.loss_fn)

        if self.is_2Dline:
            return data[:, 1], data[:, 2], data[:, 0]
        return data[:, 1], data[:, 2], data[:, 3], data[:, 0]

    def get_expr(self):
        if self.is_2Dline:
            return (self.expr_x, self.expr_y)
        return (self.expr_x, self.expr_y, self.expr_z)

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
            if self._label != str(self.get_expr()):
                return self._latex_label
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._label

    def _get_points(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        if self.adaptive:
            coords = self._adaptive_sampling()
        else:
            coords = self._uniform_sampling()

        if callable(self.color_func):
            coords = list(coords)
            coords[-1] = self.eval_color_func(*coords)
        return coords


class Parametric2DLineSeries(ParametricLineBaseSeries):
    """Representation for a line consisting of two parametric sympy expressions
    over a range."""

    is_parametric = True

    def __init__(self, expr_x, expr_y, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.var = sympify(var_start_end[0])
        self.start = float(var_start_end[1])
        self.end = float(var_start_end[2])
        self._set_parametric_line_label(label)

    def __str__(self):
        return "parametric cartesian line: (%s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.var),
            str((self.start, self.end)),
        )

    def _uniform_sampling(self):
        param = self._discretize(self.start, self.end, self.n, scale=self.scale, only_integers=self.only_integers)

        x = self._eval_component(self.expr_x, param)
        y = self._eval_component(self.expr_y, param)
        return x, y, param


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
        self.var = sympify(var_start_end[0])
        self.start = float(var_start_end[1])
        self.end = float(var_start_end[2])
        self._set_parametric_line_label(label)

    def __str__(self):
        return "3D parametric cartesian line: (%s, %s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.expr_z),
            str(self.var),
            str((self.start, self.end)),
        )

    def _uniform_sampling(self):
        param = self._discretize(self.start, self.end, self.n, scale=self.scale, only_integers=self.only_integers)

        x = self._eval_component(self.expr_x, param)
        y = self._eval_component(self.expr_y, param)
        z = self._eval_component(self.expr_z, param)
        return x, y, z, param


class SurfaceBaseSeries(BaseSeries):
    """A base class for 3D surfaces."""

    is_3Dsurface = True
    _allowed_keys = ["adaptive", "adaptive_goal", "color_func", "is_polar",
    "loss_fn", "modules", "n1", "n2", "only_integers", "rendering_kw",
    "surface_color", "use_cm", "xscale", "yscale", "tx", "ty", "tz"]

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.only_integers = kwargs.get("only_integers", False)
        self.n1 = int(kwargs.get("n1", 100))
        self.n2 = int(kwargs.get("n2", 100))
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.adaptive = kwargs.get("adaptive", False)
        self.adaptive_goal = kwargs.get("adaptive_goal", cfg["adaptive"]["goal"])
        self.loss_fn = kwargs.get("loss_fn", None)
        self.modules = kwargs.get("modules", None)
        self.rendering_kw = kwargs.get("rendering_kw", dict())
        self.use_cm = kwargs.get("use_cm", cfg["plot3d"]["use_cm"])
        self.is_polar = kwargs.get("is_polar", False)
        self.surface_color = kwargs.get("surface_color", None)
        self.color_func = kwargs.get("color_func", lambda x, y, z: z)
        if callable(self.surface_color):
            self.color_func = self.surface_color
            self.surface_color = None
        self._init_transforms(**kwargs)

    def _set_surface_label(self, label):
        exprs = self.get_expr()
        self._label = str(exprs) if label is None else label
        self._latex_label = latex(exprs) if label is None else label
        # if the expressions is a lambda function and no label
        # has been provided, then its better to do the following to avoid
        # suprises on the backend
        is_lambda = (callable(exprs) if not hasattr(exprs, "__iter__")
            else any(callable(e) for e in exprs))
        if is_lambda and (self._label == str(exprs)):
                self.label = ""

    def _discretize(self, s1, e1, s2, e2):
        np = import_module('numpy')

        mesh_x = super()._discretize(s1, e1, self.n1,
            self.xscale, self.only_integers)
        mesh_y = super()._discretize(s2, e2, self.n2,
            self.yscale, self.only_integers)
        return np.meshgrid(mesh_x, mesh_y)


class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of a sympy expression and 2D
    range."""

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self.var_x = sympify(var_start_end_x[0])
        self.start_x = float(var_start_end_x[1])
        self.end_x = float(var_start_end_x[2])
        self.var_y = sympify(var_start_end_y[0])
        self.start_y = float(var_start_end_y[1])
        self.end_y = float(var_start_end_y[2])
        self._set_surface_label(label)

    def __str__(self):
        return ("cartesian surface: %s for" " %s over %s and %s over %s") % (
            str(self.expr),
            str(self.var_x), str((self.start_x, self.end_x)),
            str(self.var_y), str((self.start_y, self.end_y)),
        )

    def _adaptive_sampling(self):
        np = import_module('numpy')

        def func(f, xy):
            try:
                return f(*xy)
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

        mesh_x, mesh_y = self._discretize(self.start_x, self.end_x,
            self.start_y, self.end_y)

        v = _uniform_eval([self.var_x, self.var_y], self.expr,
            mesh_x, mesh_y, modules=self.modules)
        re_v, im_v = np.real(v), np.imag(v)
        re_v = self._correct_shape(re_v, mesh_x)
        im_v = self._correct_shape(im_v, mesh_x)
        re_v[np.invert(np.isclose(im_v, np.zeros_like(im_v)))] = np.nan
        return mesh_x, mesh_y, re_v

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
        if self.is_polar:
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
        self.var_u = sympify(var_start_end_u[0])
        self.start_u = float(var_start_end_u[1])
        self.end_u = float(var_start_end_u[2])
        self.var_v = sympify(var_start_end_v[0])
        self.start_v = float(var_start_end_v[1])
        self.end_v = float(var_start_end_v[2])
        self.use_cm = kwargs.get("use_cm", cfg["plot3d"]["use_cm"])
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

    def get_expr(self):
        return (self.expr_x, self.expr_y, self.expr_z)

    def __str__(self):
        return (
            "parametric cartesian surface: (%s, %s, %s) for"
            " %s over %s and %s over %s"
        ) % (
            str(self.expr_x), str(self.expr_y), str(self.expr_z),
            str(self.var_u), str((self.start_u, self.end_u)),
            str(self.var_v), str((self.start_v, self.end_v)),
        )

    def _eval_component(self, expr, *args):
        """ Evaluate the specified expression over a predefined
        param-discretization.
        """
        np = import_module('numpy')

        v = _uniform_eval([self.var_u, self.var_v], expr, *args,
            modules=self.modules)
        re_v, im_v = np.real(v), np.imag(v)
        re_v = self._correct_shape(re_v, args[0])
        im_v = self._correct_shape(im_v, args[0])
        re_v[np.invert(np.isclose(im_v, np.zeros_like(im_v)))] = np.nan
        return re_v

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
        mesh_u, mesh_v = self._discretize(self.start_u, self.end_u,
            self.start_v, self.end_v)
        x = self._eval_component(self.expr_x, mesh_u, mesh_v)
        y = self._eval_component(self.expr_y, mesh_u, mesh_v)
        z = self._eval_component(self.expr_z, mesh_u, mesh_v)
        return x, y, z, mesh_u, mesh_v


class ContourSeries(SurfaceOver2DRangeSeries):
    """Representation for a contour plot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._allowed_keys += ["contour_kw"]

        # NOTE: contour plots are used by plot_contour, plot_vector and
        # plot_complex_vector. By implementing contour_kw we are able to
        # quickly target the contour plot.
        self.rendering_kw = kwargs.get("contour_kw",
            kwargs.get("rendering_kw", dict()))

    is_3Dsurface = False
    is_contour = True

    def __str__(self):
        return ("contour: %s for " "%s over %s and %s over %s") % (
            str(self.expr),
            str(self.var_x), str((self.start_x, self.end_x)),
            str(self.var_y), str((self.start_y, self.end_y)),
        )


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
    _allowed_keys = ["adaptive", "depth", "n1", "n2", "rendering_kw",
    "xscale", "yscale"]

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__()
        self._block_lambda_functions(expr)
        expr, has_equality = self._has_equality(sympify(expr))
        self.expr = expr
        self.var_x = sympify(var_start_end_x[0])
        self.start_x = float(var_start_end_x[1])
        self.end_x = float(var_start_end_x[2])
        self.var_y = sympify(var_start_end_y[0])
        self.start_y = float(var_start_end_y[1])
        self.end_y = float(var_start_end_y[2])
        self.has_equality = has_equality
        self.n1 = int(kwargs.get("n1", 1000))
        self.n2 = int(kwargs.get("n2", 1000))
        self._label = str(expr) if label is None else label
        self._latex_label = latex(expr) if label is None else label
        self.adaptive = kwargs.get("adaptive", False)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.rendering_kw = kwargs.get("rendering_kw", dict())

        if isinstance(expr, BooleanFunction) and (not self.adaptive):
            self.adaptive = True
            warnings.warn(
                "The provided expression contains Boolean functions. "
                + "In order to plot the expression, the algorithm "
                + "automatically switched to an adaptive sampling."
            )

        # Check whether the depth is greater than 4 or less than 0.
        depth = kwargs.get("depth", 0)
        if depth > 4:
            depth = 4
        elif depth < 0:
            depth = 0
        self.depth = 4 + depth

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
        return ("Implicit expression: %s for %s over %s and %s over %s") % (
            str(self.expr),
            str(self.var_x),
            str((self.start_x, self.end_x)),
            str(self.var_y),
            str((self.start_y, self.end_y)),
        )

    def get_data(self):
        if self.adaptive:
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

            if self.adaptive:
                return data

        return self._get_meshes_grid()

    def _get_raster_interval(self, func):
        """Uses interval math to adaptively mesh and obtain the plot"""
        np = import_module('numpy')

        k = self.depth
        interval_list = []
        # Create initial 32 divisions
        xsample = np.linspace(self.start_x, self.end_x, 33)
        ysample = np.linspace(self.start_y, self.end_y, 33)

        # Add a small jitter so that there are no false positives for equality.
        # Ex: y==x becomes True for x interval(1, 2) and y interval(1, 2)
        # which will draw a rectangle.
        jitterx = (
            (np.random.rand(len(xsample)) * 2 - 1)
            * (self.end_x - self.start_x)
            / 2 ** 20
        )
        jittery = (
            (np.random.rand(len(ysample)) * 2 - 1)
            * (self.end_y - self.start_y)
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

        expr, equality = self._preprocess_meshgrid_expression(self.expr)
        xarray = self._discretize(self.start_x, self.end_x, self.n1, self.xscale)
        yarray = self._discretize(self.start_y, self.end_y, self.n2, self.yscale)
        x_grid, y_grid = np.meshgrid(xarray, yarray)
        func = lambdify((self.var_x, self.var_y), expr)
        z_grid = func(x_grid, y_grid)
        z_grid = self._correct_shape(z_grid, x_grid)
        z_grid[np.ma.where(z_grid < 0)] = -1
        z_grid[np.ma.where(z_grid > 0)] = 1
        if equality:
            return xarray, yarray, z_grid, 'contour'
        else:
            return xarray, yarray, z_grid, 'contourf'

    @staticmethod
    def _preprocess_meshgrid_expression(expr):
        """If the expression is a Relational, rewrite it as a single
        expression. This method reduces code repetition.

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
        else:
            raise NotImplementedError(
                "The expression is not supported for "
                "plotting in uniform meshed plot."
            )
        return expr, equality


class Implicit3DSeries(SurfaceBaseSeries):
    is_implicit = True

    def __init__(self, expr, range_x, range_y, range_z, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self.var_x = sympify(range_x[0])
        self.start_x = float(range_x[1])
        self.end_x = float(range_x[2])
        self.var_y = sympify(range_y[0])
        self.start_y = float(range_y[1])
        self.end_y = float(range_y[2])
        self.var_z = sympify(range_z[0])
        self.start_z = float(range_z[1])
        self.end_z = float(range_z[2])
        # keep number of discretization points low as some backend might be
        # slow at processing/rendering
        self.n1 = int(kwargs.get("n1", 60))
        self.n2 = int(kwargs.get("n2", 60))
        self.n3 = int(kwargs.get("n3", 60))
        self.zscale = kwargs.get("zscale", "linear")
        self._set_surface_label(label)
        self._allowed_keys += ["n3", "zscale"]

    def __str__(self):
        return ("implicit surface series: %s for %s over %s and %s over %s"
            " and %s over %s") % (
                str(self.expr),
                str(self.var_x), str((self.start_x, self.end_x)),
                str(self.var_y), str((self.start_y, self.end_y)),
                str(self.var_z), str((self.start_z, self.end_z))
            )

    def _discretize(self, s1, e1, s2, e2, s3, e3):
        np = import_module('numpy')

        mesh_x = BaseSeries._discretize(s1, e1, self.n1,
            self.xscale, self.only_integers)
        mesh_y = BaseSeries._discretize(s2, e2, self.n2,
            self.yscale, self.only_integers)
        mesh_z = BaseSeries._discretize(s3, e3, self.n3,
            self.zscale, self.only_integers)
        return np.meshgrid(mesh_x, mesh_y, mesh_z, indexing='ij')

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

        mesh_x, mesh_y, mesh_z = self._discretize(
            self.start_x, self.end_x,
            self.start_y, self.end_y,
            self.start_z, self.end_z)
        v = _uniform_eval([self.var_x, self.var_y, self.var_z], self.expr,
                mesh_x, mesh_y, mesh_z, modules=self.modules)
        re_v, im_v = np.real(v), np.imag(v)
        re_v = self._correct_shape(re_v, mesh_x)
        im_v = self._correct_shape(im_v, mesh_x)
        re_v[np.invert(np.isclose(im_v, np.zeros_like(im_v)))] = np.nan
        return mesh_x, mesh_y, mesh_z, re_v


class InteractiveSeries(BaseSeries):
    """Base class for interactive series, in which the expressions can be
    either a line, a surface (parametric or not), a vector field, ...
    On top of the usual ranges (x, y or u, v, which must be provided), the
    expressions can use any number of parameters.

    The following class, together with the `interactive.py` module, makes it
    possible to easily plot symbolic expressions with interactive widgets.

    Once new parameters are available, update them by setting the ``params``
    attribute with a dictionary with all the necessary parameters. Then,
    ``get_data()`` can be called.

    Differently from non-interactive series, only uniform sampling is
    implemented here.
    """
    is_interactive = True

    def __new__(cls, exprs, ranges, *args, **kwargs):
        nexpr, npar = len(exprs), len(ranges)

        if nexpr == 0:
            raise ValueError(
                "At least one expression must be provided."
                + "\nReceived: {}".format((exprs, ranges))
            )

        if isinstance(exprs[0], Plane):
            return super().__new__(PlaneInteractiveSeries)
        elif isinstance(exprs[0], GeometryEntity) and (
            not isinstance(exprs[0], Curve)
        ):
            return super().__new__(GeometryInteractiveSeries)

        if (nexpr == 1) and (npar == 1):
            absarg = kwargs.get("absarg", False)
            if not absarg:
                if callable(kwargs.get("color_func", None)) or callable(kwargs.get("line_color", None)):
                    return super().__new__(ColoredLineInteractiveSeries)
                return super().__new__(LineInteractiveSeries)
            return super().__new__(AbsArgLineInteractiveSeries)
        elif (nexpr == 2) and (npar == 1):
            return super().__new__(Parametric2DLineInteractiveSeries)
        elif (nexpr == 3) and (npar == 1):
            return super().__new__(Parametric3DLineInteractiveSeries)
        elif (nexpr == 1) and (npar == 2):
            if kwargs.get("threed", False):
                return super().__new__(SurfaceInteractiveSeries)
            return super().__new__(ContourInteractiveSeries)
        elif (nexpr == 3) and (npar == 2):
            return super().__new__(ParametricSurfaceInteractiveSeries)
        elif (nexpr == 2) and (npar == 2):
            return super().__new__(Vector2DInteractiveSeries)
        elif (nexpr == 3) and (npar == 3):
            if kwargs.get("slice", None) is None:
                return super().__new__(Vector3DInteractiveSeries)
            return super().__new__(SliceVector3DInteractiveSeries)

    def __init__(self, exprs, ranges, label="", **kwargs):
        np = import_module('numpy')
        # NOTE: I believe it takes too much effort to adapt this code to work
        # with lambda functions. Let it be done by someone who is actually
        # interested in that feature.
        self._block_lambda_functions(*exprs)

        # free symbols of the parameters
        self._params = kwargs.get("params", dict())
        self._init_num_discretization_points(**kwargs)
        n = [self.n1, self.n2, self.n3]
        self.modules = kwargs.get("modules", None)
        self.is_polar = kwargs.get("is_polar", False)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.only_integers = kwargs.get("only_integers", False)
        self.is_point = kwargs.get("is_point", False)
        self.is_filled = kwargs.get("is_filled", False)
        self.use_cm = kwargs.get("use_cm", True)
        self.var = None # might be set later on
        self._tx = kwargs.get("tx", None)
        self._ty = kwargs.get("ty", None)
        self._tz = kwargs.get("tz", None)
        if not all(callable(t) or (t is None) for t in [self._tx, self._ty, self._tz]):
            raise TypeError("`tx`, `ty`, `tz` must be functions.")

        nexpr, npar = len(exprs), len(ranges)

        if nexpr == 0:
            raise ValueError(
                "At least one expression must be provided."
                + "\nReceived: {}".format((exprs, ranges, label))
            )

        self._check_fs(exprs, ranges, label, self._params)

        # NOTE: the expressions must have been sympified earlier.
        self.expr = exprs[0] if len(exprs) == 1 else Tuple(*exprs, sympify=False)
        self._label = str(self.expr) if label is None else label
        self._latex_label = latex(self.expr) if label is None else label
        self.signature = sorted(self.expr.free_symbols, key=lambda t: t.name)

        # Generate a list of lambda functions, two for each expression:
        # 1. the default one.
        # 2. the backup one, in case of failures with the default one.
        self.functions = []
        for e in exprs:
            self.functions.append([
                lambdify(self.signature, e, modules=self.modules),
                lambdify(self.signature, e, modules="sympy", dummify=True),
            ])

        # Discretize the ranges. In the dictionary self.ranges:
        #    key: symbol associate to this particular range
        #    val: the numpy array representing the discretization
        discr_symbols = []
        discretizations = []
        for i, r in enumerate(ranges):
            discr_symbols.append(r[0])
            scale = self.xscale
            if i == 1:  # y direction
                scale = self.yscale

            c_start = complex(r[1])
            c_end = complex(r[2])
            start = c_start.real if c_start.imag == c_end.imag == 0 else c_start
            end = c_end.real if c_start.imag == c_end.imag == 0 else c_end
            d = BaseSeries._discretize(start, end, n[i], scale=scale, only_integers=self.only_integers)

            if self.is_complex:
                d = d + 1j * c_start.imag
            elif self.only_integers:
                # NOTE: likely plotting a Sum. The lambdified function is
                # using ``range``, requiring integer arguments.
                d = d.astype(int)
                # HACK:
                # However, now xx is of type np.int64. If the expression
                # contains powers, for example 2**(-np.int64(3)) than
                # ValueError is raised. Turns out that by converting to
                # object the evaluation proceed as expected.
                d = d.astype(object)

            discretizations.append(d)

        self._set_discretization_ranges(discr_symbols, discretizations)

    def _init_num_discretization_points(self, **kwargs):
        """Subclasses should override this method to provide dirrent values."""
        self.n1 = int(kwargs.get("n1", 250))
        self.n2 = int(kwargs.get("n2", 250))
        self.n3 = int(kwargs.get("n3", 250))

    def _set_discretization_ranges(self, discr_symbols, discretizations):
        """Set the discretized ranges that will be used in the numerical
        evaluation.

        This method implements discretizations suitable for contour/surface/
        vector plots. Subclasses should override this method in order to
        implement a different behaviour.
        """
        np = import_module('numpy')

        meshes = np.meshgrid(*discretizations)
        self.ranges = {k: v for k, v in zip(discr_symbols, meshes)}


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

    def _str(self, series_type):
        np = import_module('numpy')

        ranges = [(k, np.amin(v), np.amax(v)) for k, v in self.ranges.items()]
        return ("interactive %s: %s with ranges %s and parameters %s") % (
            series_type,
            str(self.expr),
            ", ".join([str(r) for r in ranges]),
            str(tuple(self._params.keys())),
        )

    def _check_fs(self, exprs, ranges, label, params):
        """ Checks if there are enogh parameters and free symbols.
        This method reduces code repetition.
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

    def _evaluate(self):
        """Evaluate the function based on the current values of the parameters.
        """
        np = import_module('numpy')

        # discretized ranges all have the same shape. Take the first!
        discr = list(self.ranges.values())[0]

        args = []
        for s in self.signature:
            if s in self._params.keys():
                args.append(self._params[s])
            else:
                args.append(self.ranges[s])

        results = []
        for f in self.functions:
            r = _uniform_eval_helper(*f, *args)
            # the evaluation might produce an int/float. Need this correction.
            r = self._correct_shape(np.array(r), discr)
            results.append(r)

        return results


class LineInteractiveBaseSeries(InteractiveSeries):
    _allowed_keys = ["absarg", "color_func", "detect_poles", "eps",
    "is_complex", "is_filled", "is_point", "line_color", "modules", "n",
    "only_integers", "rendering_kw", "steps", "use_cm", "xscale",
    "tx", "ty", "tz"]

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def _set_discretization_ranges(self, discr_symbols, discretizations):
        """Set the discretized ranges that will be used in the numerical
        evaluation.
        """
        self.ranges = {k: v for k, v in zip(discr_symbols, discretizations)}


class LineInteractiveSeries(LineInteractiveBaseSeries, Line2DBaseSeries):
    """Representation for an interactive line consisting of a SymPy
    expression over a real range."""

    def __new__(cls, *args, **kwargs):
        if any(callable(kwargs.get(t, None)) for t in ["color_func", "line_color"]):
            return super().__new__(ColoredLineInteractiveSeries)
        return object.__new__(cls)

    def __init__(self, exprs, ranges, label="", **kwargs):
        if "is_complex" in kwargs.keys():
            self.is_complex = kwargs["is_complex"]
        super().__init__(exprs, ranges, label, **kwargs)
        self.var = sympify(ranges[0][0])
        self.start = complex(ranges[0][1])
        self.end = complex(ranges[0][2])
        self.steps = kwargs.get("steps", False)
        self.detect_poles = kwargs.get("detect_poles", False)
        self.eps = kwargs.get("eps", 0.01)
        self.rendering_kw = kwargs.get("rendering_kw", dict())
        self.color_func = kwargs.get("color_func", None)
        self.line_color = kwargs.get("line_color", None)

    def _get_points(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')

        results = self._evaluate()[0]
        _re, _im = np.real(results), np.imag(results)
        _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        discr = np.real(list(self.ranges.values())[0])

        if self.detect_poles:
            return LineOver1DRangeSeries._detect_poles(discr, _re, self.eps)

        return discr, _re

    def __str__(self):
        return self._str("cartesian line")


class ColoredLineInteractiveSeries(LineInteractiveSeries):
    """Represents a 2D line interactive series in which `color_func` is a
    callable.
    """
    is_parametric = True

    def _get_points(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        x, y = super()._get_points()
        return x, y, self.eval_color_func(x, y)


class AbsArgLineInteractiveSeries(LineInteractiveSeries):
    """Represents the interactive absolute value of a complex function
    colored by its argument over a complex range (a + I*b, c + I * b).
    Note that the imaginary part of the start and end must be the same.
    """
    is_parametric = True
    is_complex = True

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def _get_points(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')

        results = self._evaluate()[0]
        _re, _im = np.real(results), np.imag(results)
        _abs = np.sqrt(_re**2 + _im**2)
        _angle = np.arctan2(_im, _re)
        discr = np.real(list(self.ranges.values())[0])
        if self.detect_poles:
            _, _abs = LineOver1DRangeSeries._detect_poles(discr, _abs, self.eps)
        return discr, _abs, _angle

    def __str__(self):
        return self._str("cartesian abs-arg line")

class Parametric2DLineInteractiveSeries(LineInteractiveBaseSeries, Line2DBaseSeries):
    """Representation for an interactive line consisting of two
    parametric sympy expressions over a range."""
    is_parametric = True

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = kwargs.get("steps", False)
        self.rendering_kw = kwargs.get("rendering_kw", dict())
        self.var = list(self.ranges.keys())[0]
        self.color_func = kwargs.get("color_func", None)
        self.line_color = kwargs.get("line_color", None)
        ParametricLineBaseSeries._set_parametric_line_label(self, args[-1])

    def get_label(self, use_latex=False, wrapper="$%s$"):
        return ParametricLineBaseSeries.get_label(self, use_latex, wrapper)

    def _get_points(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')

        results = self._evaluate()
        _re, _im = np.real(results), np.imag(results)
        _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        discr = [np.real(t) for t in self.ranges.values()]
        if callable(self.color_func):
            discr = [self.eval_color_func(*_re, *discr)]
        return [*_re, *discr]

    def __str__(self):
        return self._str("parametric cartesian line")

class Parametric3DLineInteractiveSeries(Parametric2DLineInteractiveSeries):
    """Representation for a 3D interactive line consisting of three
    parametric sympy expressions and a range."""
    is_2Dline = False
    is_3Dline = True

    def __str__(self):
        return self._str("3D parametric cartesian line")

class SurfaceInteractiveSeries(InteractiveSeries):
    """Representation for a 3D interactive surface consisting of a sympy
    expression and 2D range."""
    is_3Dsurface = True
    _allowed_keys = ["color_func", "is_polar", "modules", "n1", "n2",
    "only_integers", "rendering_kw", "surface_color", "use_cm",
    "xscale", "yscale", "tx", "ty", "tz"]


    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rendering_kw = kwargs.get("rendering_kw", dict())
        self.use_cm = kwargs.get("use_cm", cfg["plot3d"]["use_cm"])
        self.color_func = kwargs.get("color_func", lambda x, y, z: z)
        self.surface_color = kwargs.get("surface_color", None)

    def get_data(self):
        """Return arrays of coordinates for plotting.

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

        results = self._evaluate()[0]
        _re, _im = np.real(results), np.imag(results)
        _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        x, y = [np.real(t) for t in self.ranges.values()]

        r = x.copy()
        if self.is_polar:
            x = r * np.cos(y)
            y = r * np.sin(y)
        return self._apply_transform(x, y, _re)

    def __str__(self):
        return self._str("cartesian surface")

class ContourInteractiveSeries(SurfaceInteractiveSeries):
    """Representation for an interactive contour plot."""
    is_3Dsurface = False
    is_contour = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE: contour plots are used by plot_contour, plot_vector and
        # plot_complex_vector. By implementing contour_kw we are able to
        # quickly target the contour plot.
        self.rendering_kw = kwargs.get("contour_kw",
            kwargs.get("rendering_kw", dict()))

    def __str__(self):
        return self._str("contour")

class ParametricSurfaceInteractiveSeries(SurfaceInteractiveSeries):
    """Representation for a 3D interactive surface consisting of three
    parametric sympy expressions and a range."""
    is_parametric = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_func = kwargs.get("color_func", lambda x, y, z, u, v: z)

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======

        x : np.ndarray [n2 x n1]
            x-coordinates.

        y : np.ndarray [n2 x n1]
            y-coordinates.

        z : np.ndarray [n2 x n1]
            z-coordinates.
        """
        np = import_module('numpy')

        results = self._evaluate()
        for i in range(len(results)):
            _re, _im = np.real(results[i]), np.imag(results[i])
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        discr = [np.real(t) for t in self.ranges.values()]
        return [*results, *discr]

    def __str__(self):
        return self._str("parametric cartesian surface")


class ComplexPointSeries(Line2DBaseSeries):
    """Representation for a line in the complex plane consisting of
    list of points."""

    _allowed_keys = ["color_func", "is_filled", "is_point", "line_color",
    "rendering_kw", "steps", "tx", "ty", "tz"]

    def __init__(self, expr, label="", **kwargs):
        self._init_attributes(expr, label, **kwargs)

    def _init_attributes(self, expr, label, **kwargs):
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
        self.label = label
        self._latex_label = label
        self.rendering_kw = kwargs.get("rendering_kw", dict())
        self.color_func = kwargs.get("color_func", None)
        self.line_color = kwargs.get("line_color", None)
        self._init_transforms(**kwargs)

    @staticmethod
    def _evaluate(points):
        np = import_module('numpy')
        points = np.array([complex(p) for p in points])
        return np.real(points), np.imag(points)

    def _get_points(self):
        """Returns coordinates that needs to be postprocessed."""
        return self._evaluate(self.expr)

    def __str__(self):
        return "complex points: %s" % self.expr


class ComplexPointInteractiveSeries(LineInteractiveBaseSeries, ComplexPointSeries):
    """Representation for an interactive line in the complex plane
    consisting of list of points."""
    _allowed_keys = ComplexPointSeries._allowed_keys

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, expr, label="", **kwargs):
        self._init_attributes(expr, label, **kwargs)
        self._params = kwargs.get("params", dict())
        self._check_fs(expr, None, label, self._params)

    def update_data(self, params):
        self._params = params

    def _get_points(self):
        """Returns coordinates that needs to be postprocessed."""
        points = Tuple(*[p.evalf(subs=self._params) for p in self.expr])
        return ComplexPointSeries._evaluate(points)

    def __str__(self):
        return "interactive complex points: %s with parameters %s" % (
            self.expr, tuple(self._params.keys()))


class ComplexSurfaceBaseSeries(BaseSeries):
    """Represent a complex function."""
    is_complex = True
    _allowed_keys = ["absarg", "coloring", "color_func", "modules", "phaseres",
    "is_polar", "n1", "n2", "only_integers", "rendering_kw", "steps",
    "surface_color","use_cm", "xscale", "yscale", "tx", "ty", "tz", "threed"]

    def __new__(cls, *args, **kwargs):
        domain_coloring = kwargs.get("absarg", False)
        if domain_coloring:
            return super().__new__(ComplexDomainColoringSeries)
        return super().__new__(ComplexSurfaceSeries)

    def __init__(self, expr, r, label="", **kwargs):
        expr = expr if callable(expr) else sympify(expr)
        self._init_attributes(expr, r, label, **kwargs)

    def _init_attributes(self, expr, r, label, **kwargs):
        self.var = sympify(r[0])
        self.start = complex(r[1])
        self.end = complex(r[2])
        if self.start.imag == self.end.imag:
            raise ValueError(
                "The same imaginary part has been used for `start` and "
                "`end`: %s. " % self.start.imag +
                "They must be different."
            )

        if kwargs.get("threed", False):
            self.is_3Dsurface = True

        if isinstance(self, ComplexSurfaceSeries):
            self._block_lambda_functions(expr)
        self.expr = expr
        self._label = str(expr) if label is None else label
        self._latex_label = latex(expr) if label is None else label
        self.n1 = int(kwargs.get("n1", 300))
        self.n2 = int(kwargs.get("n2", 300))
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.modules = kwargs.get("modules", None)
        self.only_integers = kwargs.get("only_integers", False)
        self.use_cm = kwargs.get("use_cm", True)
        self.is_polar = kwargs.get("is_polar", False)
        self.surface_color = kwargs.get("surface_color", None)

        # domain coloring mode
        self.coloring = kwargs.get("coloring", "a")
        if isinstance(self.coloring, str):
            self.coloring = self.coloring.lower()
        elif not callable(self.coloring):
            raise TypeError(
                "`coloring` must be a character from 'a' to 'j' or a callable.")
        self.phaseres = kwargs.get("phaseres", 20)
        self._init_transforms(**kwargs)

    def __str__(self):
        if self.is_domain_coloring:
            prefix = "complex domain coloring"
            if self.is_3Dsurface:
                prefix = "complex 3D domain coloring"
        else:
            prefix = "complex cartesian surface"
            if self.is_contour:
                prefix = "complex contour"

        return (prefix + ": %s for" " re(%s) over %s and im(%s) over %s") % (
                str(self.expr),
                str(self.var),
                str((self.start.real, self.end.real)),
                str(self.var),
                str((self.start.imag, self.end.imag)),
            )

    def _common_eval(self):
        np = import_module('numpy')

        start_x = self.start.real
        end_x = self.end.real
        start_y = self.start.imag
        end_y = self.end.imag
        x = self._discretize(start_x, end_x, self.n1,
            self.xscale, self.only_integers)
        y = self._discretize(start_y, end_y, self.n2,
            self.yscale, self.only_integers)
        xx, yy = np.meshgrid(x, y)
        domain = xx + 1j * yy
        zz = _uniform_eval(self.var, self.expr, domain,
            modules=self.modules)
        zz = self._correct_shape(np.array(zz), domain)
        return domain, zz


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
        self._init_rendering_kw(**kwargs)

    def _init_rendering_kw(self, **kwargs):
        self.color_func = kwargs.get("color_func", lambda x, y, z: z)
        self.rendering_kw = kwargs.get("rendering_kw", dict())

    def _correct_output(self, domain, z):
        np = import_module('numpy')

        return np.real(domain), np.imag(domain), np.real(z)

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
        domain, z = self._common_eval()
        return self._correct_output(domain, z)


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
        self._init_rendering_kw(**kwargs)

    def _init_rendering_kw(self, **kwargs):
        self.rendering_kw = kwargs.get("rendering_kw", dict())

    def _domain_coloring(self, w):
        if isinstance(self.coloring, str):
            from spb.ccomplex.wegert import wegert
            self.coloring = self.coloring.lower()
            return wegert(self.coloring, w, self.phaseres)
        return self.coloring(w)

    def _correct_output(self, domain, z):
        np = import_module('numpy')

        return self._apply_transform(
            np.real(domain), np.imag(domain),
            np.absolute(z), np.angle(z),
            *self._domain_coloring(z),
        )

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
        domain, z = self._common_eval()
        return self._correct_output(domain, z)


class ComplexInteractiveBaseSeries(InteractiveSeries, ComplexSurfaceBaseSeries):
    """Represent an interactive complex function."""
    _allowed_keys = ComplexSurfaceBaseSeries._allowed_keys

    def __new__(cls, *args, **kwargs):
        domain_coloring = kwargs.get("absarg", False)

        if domain_coloring:
            return ComplexDomainColoringInteractiveSeries(*args, **kwargs)
        return ComplexSurfaceInteractiveSeries(*args, **kwargs)

    def __init__(self, expr, r, label="", **kwargs):
        np = import_module('numpy')

        self._params = kwargs.get("params", dict())
        self._init_attributes(expr, r, label, **kwargs)
        self._check_fs([expr], [r], label, self._params)

        self.signature = sorted(self.expr.free_symbols, key=lambda t: t.name)
        # Two lambda functions:
        # 1. the default one.
        # 2. the backup one, in case of failures with the default one.
        self.functions = [[
            lambdify(self.signature, self.expr, modules=self.modules),
            lambdify(self.signature, self.expr, modules="sympy")
        ]]

        x = self._discretize(
            self.start.real, self.end.real, self.n1,
            scale=self.xscale, only_integers=self.only_integers)
        y = self._discretize(
            self.start.imag, self.end.imag, self.n2,
            scale=self.yscale, only_integers=self.only_integers)
        xx, yy = np.meshgrid(x, y)
        zz = xx + 1j * yy
        self.ranges = {self.var: zz}

    def __str__(self):
        if self.is_domain_coloring:
            prefix = "complex domain coloring"
            if self.is_3Dsurface:
                prefix = "complex 3D domain coloring"
        else:
            prefix = "complex cartesian surface"
            if self.is_contour:
                prefix = "complex contour"

        return "interactive %s for expression: %s over %s and parameters %s" % (
            prefix, str(self.expr),
            str((self.var, self.start, self.end)),
            str(tuple(self._params.keys()))
        )


class ComplexSurfaceInteractiveSeries(ComplexInteractiveBaseSeries, ComplexSurfaceSeries):
    """Represents an interactive 3D surface or contour plot of a complex
    function over the complex plane.
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not kwargs.get("threed", False):
            # if not 3D, plot the contours
            self.is_contour = True
            self.is_3Dsurface = False
        self._init_rendering_kw(**kwargs)

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
        domain = list(self.ranges.values())[0]
        results = self._evaluate()[0]
        return self._correct_output(domain, results)


class ComplexDomainColoringInteractiveSeries(ComplexInteractiveBaseSeries, ComplexDomainColoringSeries):
    """Represents an interactive 2D/3D domain coloring plot of a complex
    function over the complex plane.
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_rendering_kw(**kwargs)

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

        domain = list(self.ranges.values())[0]
        results = self._evaluate()[0]
        return self._apply_transform(
            np.real(domain), np.imag(domain),
            np.absolute(results), np.angle(results),
            *self._domain_coloring(results),
        )


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
    if pt in [LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries, LineInteractiveSeries]:
        if "n" in kwargs.keys():
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 0):
                kwargs["n"] = kwargs["n"][0]
        elif "n1" in kwargs.keys():
            kwargs["n"] = kwargs["n1"]
    elif pt in [
        SurfaceOver2DRangeSeries, ContourSeries, ParametricSurfaceSeries,
        ComplexSurfaceBaseSeries, ComplexInteractiveBaseSeries,
        Vector2DSeries, ImplicitSeries,
    ]:
        if "n" in kwargs.keys():
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 1):
                kwargs["n1"] = kwargs["n"][0]
                kwargs["n2"] = kwargs["n"][1]
            else:
                kwargs["n1"] = kwargs["n2"] = kwargs["n"]
    elif pt in [Vector3DSeries, SliceVector3DSeries, InteractiveSeries,
        Implicit3DSeries
    ]:
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
    _allowed_keys = ["n1", "n2", "n3", "modules", "only_integers", "streamlines", "use_cm", "xscale", "yscale", "zscale", "quiver_kw", "stream_kw", "rendering_kw", "tx", "ty", "tz"]

    def _init_num_discretization_points(self, **kwargs):
        """Subclasses should override this method to provide dirrent values."""
        self.n1 = int(kwargs.get("n1", self._n))
        self.n2 = int(kwargs.get("n2", self._n))
        self.n3 = int(kwargs.get("n3", self._n))

    def __init__(self, exprs, ranges, label, **kwargs):
        new_ranges = []
        for r in ranges:
            new_ranges.append((sympify(r[0]), float(r[1]), float(r[2])))
        self.exprs = exprs
        self.ranges = new_ranges
        self._label = str(exprs) if label is None else label
        self._latex_label = latex(exprs) if label is None else label
        self._init_num_discretization_points(**kwargs)
        self.n = [self.n1, self.n2, self.n3]
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.zscale = kwargs.get("zscale", "linear")
        self.scales = [self.xscale, self.yscale, self.zscale]
        self.is_streamlines = kwargs.get("streamlines", False)
        self.modules = kwargs.get("modules", None)
        self.only_integers = kwargs.get("only_integers", False)
        self.use_cm = kwargs.get("use_cm", True)

        # if the expressions are lambda functions and no label has been
        # provided, then its better to do the following to avoid suprises on
        # the backend
        if any(callable(e) for e in self.exprs):
            if self._label == str(self.exprs):
                self.label = "Magnitude"

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
        self._init_transforms(**kwargs)

    def get_expr(self):
        return self.exprs

    def _discretize(self):
        np = import_module('numpy')

        one_d = []
        for r, n, s in zip(self.ranges, self.n, self.scales):
            one_d.append(super()._discretize(r[1], r[2], n, s, self.only_integers))
        return np.meshgrid(*one_d)

    def _eval_component(self, meshes, fs, expr):
        np = import_module('numpy')

        v = _uniform_eval(fs, expr, *meshes, modules=self.modules)
        re_v, im_v = np.real(v), np.imag(v)
        re_v = self._correct_shape(re_v, meshes[0])
        im_v = self._correct_shape(im_v, meshes[0])
        re_v[np.invert(np.isclose(im_v, np.zeros_like(im_v)))] = np.nan
        return re_v

    def _eval_component2(self, meshes, f):
        np = import_module('numpy')

        v = f(*meshes)
        re_v, im_v = np.real(v), np.imag(v)
        re_v = self._correct_shape(re_v, meshes[0])
        im_v = self._correct_shape(im_v, meshes[0])
        re_v[np.invert(np.isclose(im_v, np.zeros_like(im_v)))] = np.nan
        return re_v

    def get_expr(self):
        return self.exprs

    def get_label(self, use_latex=False, wrapper="$%s$"):
        if use_latex:
            expr = self.get_expr()
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
        meshes = self._discretize()
        free_symbols = [r[0] for r in self.ranges]
        results = []
        if any(callable(e) for e in self.exprs):
            for e in self.exprs:
                results.append(self._eval_component2(meshes, e))
        else:
            for e in self.exprs:
                results.append(self._eval_component(meshes, free_symbols, e))
        return self._apply_transform(*meshes, *results)


class Vector2DSeries(VectorBase):
    """Represents a 2D vector field."""

    is_2Dvector = True
    # default number of discretization points
    _n = 25

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
        return "2D vector series: [%s, %s] over %s, %s" % (
            *self.exprs, *self.ranges)


class Vector3DSeries(VectorBase):
    """Represents a 3D vector field."""

    is_3D = True
    is_3Dvector = True
    # default number of discretization points
    _n = 10

    def __init__(self, u, v, z, range1, range2, range3, label="", **kwargs):
        super().__init__((u, v, z), (range1, range2, range3), label, **kwargs)

    def __str__(self):
        return "3D vector series: [%s, %s, %s] over %s, %s, %s" % (
            *self.exprs, *self.ranges)

class VectorInteractiveBaseSeries(InteractiveSeries):
    """Represent an interactive vector field."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def _init_num_discretization_points(self, **kwargs):
        return VectorBase._init_num_discretization_points(self, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_streamlines = kwargs.get("streamlines", False)
        if self.is_streamlines:
            self.rendering_kw = kwargs.get("stream_kw", dict())
        else:
            self.rendering_kw = kwargs.get("quiver_kw", dict())

    def get_expr(self):
        return self.expr

    def get_label(self, use_latex=False, wrapper="$%s$"):
        return VectorBase.get_label(self, use_latex, wrapper)

    def get_data(self):
        np = import_module('numpy')

        discr = [np.real(t) for t in self.ranges.values()]
        results = self._evaluate()

        for i, r in enumerate(results):
            re_v, im_v = np.real(r), np.imag(r)
            re_v[np.invert(np.isclose(im_v, np.zeros_like(im_v)))] = np.nan
            results[i] = re_v

        return self._apply_transform(*discr, *results)

    def __str__(self):
        prefix = "2D" if self.is_2Dvector else "3D"
        prefix += " vector series"
        return self._str(prefix)

class Vector2DInteractiveSeries(VectorInteractiveBaseSeries, Vector2DSeries):
    """Represents an interactive 2D vector field."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_use_quiver_solid_color(**kwargs)

class Vector3DInteractiveSeries(VectorInteractiveBaseSeries, Vector3DSeries):
    """Represents an interactive 3D vector field."""
    pass


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
        int(kwargs.get("n1", Vector3DSeries._n)),
        int(kwargs.get("n2", Vector3DSeries._n)),
        int(kwargs.get("n3", Vector3DSeries._n))]
    discr_symbols = [r[0] for r in ranges]
    idx = [discr_symbols.index(s) for s in [r[0] for r in new_ranges]]
    kwargs2 = kwargs.copy()
    kwargs2["n1"] = n[idx[0]]
    kwargs2["n2"] = n[idx[1]]

    if ("params" in kwargs2.keys() and
    any(s in kwargs2["params"].keys() for s in slice_surf.free_symbols)):
        return SurfaceInteractiveSeries([slice_surf], new_ranges, **kwargs2)
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
        if (isinstance(self.slice_surf_series, PlaneSeries) or
        self.slice_surf_series.is_parametric):
            return data

        # NOTE: let's say the vector field is discretized along x, y, z (in
        # this order), and the slice surface is f(y, z). Then, data will be
        # [yy, zz, f(yy, zz)], which has not the correct order expected by
        # the vector field's discretization. Here we are going to fix that.

        # symbols used by this vector's discretization
        discr_symbols = [r[0] for r in self.ranges]
        # slice surface free symbols
        # don't use self.slice_surf_series.free_symbols as this expression
        # might not use both its discretization symbols
        ssfs = [self.slice_surf_series.var_x, self.slice_surf_series.var_y]
        # given f(y, z), we already have y, z (ssfs), now find x
        missing_symbol = list(set(discr_symbols).difference(ssfs))
        # ordered symbols in the returned data
        returned_symbols = ssfs + missing_symbol
        # output order
        order = [returned_symbols.index(s) for s in discr_symbols]
        return [data[k] for k in order]

    def __str__(self):
        return "sliced " + super().__str__() + " at {}".format(
            self.slice_surf_series)


class SliceVector3DInteractiveSeries(VectorInteractiveBaseSeries, SliceVector3DSeries):
    """Represents an interactive 3D vector field plotted over a slice.
    The slice can be a Plane or a surface.
    """
    def __init__(self, *args, **kwargs):
        slice_surf = kwargs.get("slice", None)
        ranges = args[1]
        kwargs2 = kwargs.copy()
        kwargs2 = _set_discretization_points(kwargs2, SliceVector3DSeries)
        res = _build_slice_series(slice_surf, ranges, **kwargs2)
        self.slice_surf_series = res
        super().__init__(*args, **kwargs)

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
            discr_symbols = self.ranges.keys()
            self.ranges = {
                k: v for k, v in zip(discr_symbols, self.slice_surf_series.get_data())
            }

    def _set_discretization_ranges(self, discr_symbols, discretizations):
        """Set the discretized ranges that will be used in the numerical
        evaluation.
        """
        self.ranges = {
            k: v for k, v in zip(discr_symbols, self.slice_surf_series.get_data())
        }

    def __str__(self):
        return "sliced " + super().__str__() + " at {}".format(
            self.slice_surf_series)


class PlaneSeries(SurfaceBaseSeries):
    """Represents a plane in a 3D domain."""

    is_3Dsurface = True

    def __init__(
        self, plane, x_range, y_range, z_range=None, label="", params=dict(), **kwargs
    ):
        self._block_lambda_functions(plane)
        self.plane = sympify(plane)
        if not isinstance(self.plane, Plane):
            raise TypeError("`plane` must be an instance of sympy.geometry.Plane")
        self.x_range = sympify(x_range)
        self.y_range = sympify(y_range)
        self.z_range = sympify(z_range)
        self.n1 = int(kwargs.get("n1", 20))
        self.n2 = int(kwargs.get("n2", 20))
        self.n3 = int(kwargs.get("n3", 20))
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.zscale = kwargs.get("zscale", "linear")
        self._params = params
        self.rendering_kw = kwargs.get("rendering_kw", dict())
        self.use_cm = kwargs.get("use_cm", True)
        self._set_surface_label(label)
        self.color_func = kwargs.get("color_func", lambda x, y, z: z)

    def __str__(self):
        return "plane series: %s over %s, %s, %s" % (
            self.plane, self.x_range, self.y_range, self.z_range
        )

    def get_expr(self):
        return self.plane

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
                n1=self.n3,
                n2=self.n2,
                xscale=self.xscale,
                yscale=self.yscale
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
                n1=self.n1,
                n2=self.n3,
                xscale=self.xscale,
                yscale=self.yscale
            )
            xx, yy, zz = s.get_data()
            xx, yy, zz = xx, zz, yy
        else:
            # parallel to xy plane, or any other plane
            eq = plane.equation(x, y, z)
            if z in eq.free_symbols:
                eq = solve(eq, z)[0]
            s = SurfaceOver2DRangeSeries(
                eq,
                (x, *self.x_range[1:]),
                (y, *self.y_range[1:]),
                "",
                n1=self.n1,
                n2=self.n2,
                xscale=self.xscale,
                yscale=self.yscale
            )
            xx, yy, zz = s.get_data()
            if len(fs) > 1:
                idx = np.logical_or(zz < self.z_range[1], zz > self.z_range[2])
                zz[idx] = np.nan
        return xx, yy, zz


class PlaneInteractiveSeries(PlaneSeries, InteractiveSeries):
    """Represents an interactive Plane in a 3D domain."""

    # NOTE: In the MRO, PlaneSeries has the precedence over InteractiveSeries.
    # This is because Numpy and Scipy don't have correspondence with Plane.
    # Hence, we have to use get_data() implemented in PlaneSeries.

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, exprs, ranges, label="", **kwargs):
        PlaneSeries.__init__(self, exprs[0], *ranges, label=label, **kwargs)

    def update_data(self, params):
        self._params = params

    def __str__(self):
        s = super().__str__()
        return "interactive " + s + " with parameters " + str(list(self._params.keys()))


class GeometrySeries(BaseSeries):
    """Represents an entity from the sympy.geometry module.
    Depending on the geometry entity, this class can either represents a
    point, a line, or a parametric line
    """

    is_geometry = True
    _allowed_keys = ["is_filled", "use_cm", "color_func", "line_color",
    "rendering_kw"]

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

    def __init__(self, expr, _range=None, label="", params=dict(), **kwargs):
        if not isinstance(expr, GeometryEntity):
            raise ValueError(
                "`expr` must be a geomtric entity.\n"
                + "Received: type(expr) = {}\n".format(type(expr))
                + "Expr: {}".format(expr)
            )

        r = expr.free_symbols.difference(set(params.keys()))
        if len(r) > 0:
            raise ValueError(
                "Too many free symbols. Please, specify the values of the "
                + "following symbols with the `params` dictionary: {}".format(r)
            )

        self.expr = expr
        self._range = _range
        self._label = str(expr) if label is None else label
        self._latex_label = latex(expr) if label is None else label
        self._params = params
        self.is_filled = kwargs.get("is_filled", True)
        self.n = int(kwargs.get("n", 200))
        self.use_cm = kwargs.get("use_cm", True)
        self.color_func = kwargs.get("color_func", None)
        self.line_color = kwargs.get("line_color", None)
        if isinstance(expr, (LinearEntity3D, Point3D)):
            self.is_3Dline = True
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

        expr = self.expr.subs(self._params)
        if isinstance(expr, Point3D):
            return (
                np.array([expr.x], dtype=float),
                np.array([expr.y], dtype=float),
                np.array([expr.z], dtype=float),
                np.array([0], dtype=float),
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
            t = np.linspace(0, 2 * np.pi, self.n)
            x, y = cx + r * np.cos(t), cy + r * np.sin(t)
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            return x, y
        elif isinstance(expr, Ellipse):
            cx, cy = float(expr.center[0]), float(expr.center[1])
            a = float(expr.hradius)
            e = float(expr.eccentricity)
            x = np.linspace(-a, a, self.n)
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
            param = np.zeros_like(x)
            return x, y, z, param
        elif isinstance(expr, (Segment, Ray)):
            p1, p2 = expr.points
            x = np.array([p1.x, p2.x])
            y = np.array([p1.y, p2.y])
            return x.astype(float), y.astype(float)
        else:  # Line
            p1, p2 = expr.points
            if self._range is None:
                x = np.array([p1.x, p2.x])
                y = np.array([p1.y, p2.y])
            else:
                m = expr.slope
                q = p1[1] - m * p1[0]
                x = np.array([self._range[1], self._range[2]])
                y = m * x + q
            return x.astype(float), y.astype(float)

    def __str__(self):
        return "geometry entity: %s" % str(self.expr)


class GeometryInteractiveSeries(GeometrySeries, InteractiveSeries):
    """Represents an interactive entity from the sympy.geometry module."""

    # NOTE: In the MRO, GeometrySeries has the precedence over
    # InteractiveSeries. This is because Numpy and Scipy don't have
    # correspondence with Line, Segment, Polygon, ... Hence, we have to
    # use get_data() implemented in GeometrySeries.

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, exprs, ranges, label="", **kwargs):
        r = ranges[0] if len(ranges) > 0 else None
        GeometrySeries.__init__(self, exprs[0], _range=r, label=label, **kwargs)

    def update_data(self, params):
        self._params = params

    def __str__(self):
        s = super().__str__()
        return "interactive " + s + " with parameters " + str(tuple(self._params.keys()))
