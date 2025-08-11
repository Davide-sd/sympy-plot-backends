import math
import numpy as np
import param
from numbers import Number
from inspect import signature
from spb.wegert import wegert
from spb.defaults import cfg
from spb.utils import (
    _get_free_symbols,
    _correct_shape,
    unwrap,
    extract_solution,
    tf_to_control
)
import sympy
from sympy import (
    latex, Tuple, arity, symbols, sympify, solve, Expr, lambdify,
    Equality, Ne, GreaterThan, LessThan, StrictLessThan, StrictGreaterThan,
    Plane, Polygon, Circle, Ellipse, Segment, Ray, Curve, Point2D, Point3D,
    atan2, floor, ceiling, Sum, Product, Symbol, frac, im, re, zeta, Poly,
    Union, Interval, nsimplify, Set, Integral, hyper, fraction
)
from sympy.core.relational import Relational
from sympy.calculus.util import continuous_domain
from sympy.geometry.entity import GeometryEntity
from sympy.geometry.line import LinearEntity2D, LinearEntity3D
from sympy.logic.boolalg import BooleanFunction
from sympy.plotting.intervalmath import interval
from sympy.external import import_module
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.precedence import precedence
from sympy.core.sorting import default_sort_key
from matplotlib.cbook import (
    pts_to_prestep, pts_to_poststep, pts_to_midstep
)
import warnings
from spb.series.evaluator import (
    IntervalMathPrinter,
    _AdaptiveEvaluationParameters,
    _GridEvaluationParameters,
    GridEvaluator,
    ComplexGridEvaluator,
    _adaptive_eval,
)
from spb.series.base import (
    BaseSeries,
    _TpParameter,
    _TzParameter,
    _get_wrapper_for_expr,
    _raise_color_func_error
)


def _detect_poles_numerical_helper(
    x, y, eps=0.01, expr=None, symb=None, symbolic=False
):
    """Compute the steepness of each segment. If it's greater than a
    threshold, set the right-point y-value non NaN and record the
    corresponding x-location for further processing.

    Returns
    =======
    x : np.ndarray
        Unchanged x-data.
    yy : np.ndarray
        Modified y-data with NaN values.
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


def _detect_poles_symbolic_helper(expr, symb, start, end):
    """Attempts to compute symbolic discontinuities.

    Returns
    =======
    pole : list
        List of symbolic poles, possibily empty.
    """
    poles = []
    interval = Interval(nsimplify(start), nsimplify(end))
    res = continuous_domain(expr, symb, interval)
    res = res.simplify()
    if res == interval:
        pass
    elif (isinstance(res, Union) and
        all(isinstance(t, Interval) for t in res.args)):
        poles = []
        for s in res.args:
            if s.left_open:
                poles.append(s.left)
            if s.right_open:
                poles.append(s.right)
        poles = list(set(poles))
    else:
        raise ValueError(
            f"Could not parse the following object: {res} .\n"
            "Please, submit this as a bug. Consider also to set "
            "`detect_poles=True`."
        )
    return poles


def _check_steps(steps):
    if isinstance(steps, str):
        steps = steps.lower()
    possible_values = ["pre", "post", "mid", True, False, None]
    if not (steps in possible_values):
        warnings.warn(
            "``steps`` not recognized. Possible values are: " % possible_values
        )
    return steps


class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines."""

    is_2Dline = True
    _N = 1000

    _allowed_keys = [
        "steps", "scatter", "is_filled", "fill", "line_color", "detect_poles",
        "eps", "is_polar", "unwrap", "exclude",
    ]

    steps = param.Selector(
        default=False,
        objects=["pre", "post", "mid", True, False, None],
        doc="""
            If set, it connects consecutive points with steps rather than
            straight segments.""")
    # TODO: replace is_point with is_scatter or scatter
    is_point = param.Boolean(False, doc="""
        Whether to create a scatter or a continuous line.""")
    is_filled = param.Boolean(True, doc="""
        Whether scatter's markers are filled or void.""")
    line_color = param.Parameter(default=None, doc="""
        For back-compatibility with old sympy.plotting. Use ``rendering_kw``
        in order to fully customize the appearance of the line/scatter.""")
    detect_poles = param.Selector(
        default=False, objects={
            "No poles detection": False,
            "Poles detection with the numerical algorithm": True,
            "Poles detection with numerical and symbolic algorithms": "symbolic"
        }, doc="""
        Chose whether to detect and correctly plot the roots of the
        denominator. There are two algorithms at work:

        1. based on the gradient of the numerical data, it introduces NaN
           values at locations where the steepness is greater than some
           threshold. This splits the line into multiple segments. To improve
           detection, increase the number of discretization points ``n``
           and/or change the value of ``eps``.
        2. a symbolic approach based on the ``continuous_domain`` function
           from the ``sympy.calculus.util`` module, which computes the
           locations of discontinuities. If any are found, vertical lines
           will be shown.
        """)
    eps = param.Number(default=0.01, bounds=(0, None), doc="""
        An arbitrary small value used by the ``detect_poles`` numerical
        algorithm. Before changing this value, it is recommended to increase
        the number of discretization points.""")
    # TODO: are they excluded from eval or is the result at this particular
    # coordinate set to Nan?
    exclude = param.List([], item_type=float, doc="""
        List of x-coordinates to be excluded from evaluation.""")
    unwrap = param.ClassSelector(default=False, class_=(bool, dict), doc="""
        Whether to use numpy.unwrap() on the computed coordinates in order
        to get rid of discontinuities. It can be:

        * False: do not use ``np.unwrap()``.
        * True: use ``np.unwrap()`` with default keyword arguments.
        * dictionary of keyword arguments passed to ``np.unwrap()``.
        """)
    poles_locations = param.List([], doc="""
        When ``detect_poles="symbolic"``, stores the location of the computed
        poles so that they can be appropriately rendered.""")
    rendering_kw = param.Dict(default={}, doc="""
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the line.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib:

          - for solid lines:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
          - for colormap-based lines:
            https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection
          - for scatters:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

        * Bokeh:

          - for solid lines:
            https://docs.bokeh.org/en/latest/docs/reference/plotting.html#bokeh.plotting.Figure.line
          - for scatter:
            https://docs.bokeh.org/en/latest/docs/reference/plotting/figure.html#bokeh.plotting.Figure.scatter

        * Plotly:
          https://plotly.com/python/line-and-scatter/
        """)

    @param.depends("line_color", watch=True, on_init=True)
    def _update_line_color(self):
        self._line_surface_color("_line_color", self.line_color)

    def __init__(self, *args, **kwargs):
        if "scatter" in kwargs:
            kwargs.setdefault("is_point", kwargs.pop("scatter"))
        if "fill" in kwargs:
            kwargs.setdefault("is_filled", kwargs.pop("fill"))
        if "label" in kwargs:
            if kwargs["label"] is None:
                kwargs["label"] = ""
        kwargs.setdefault("use_cm", False)
        # kwargs.setdefault("n", self._N)


        exclude = kwargs.pop("exclude", [])
        if isinstance(exclude, Set):
            exclude = list(extract_solution(exclude, n=100))
        if not hasattr(exclude, "__iter__"):
            exclude = [exclude]
        kwargs["exclude"] = sorted([float(e) for e in exclude])

        print("Line2DSeries.__init__", kwargs)
        super().__init__(*args, **kwargs)

    def get_data(self):
        """
        Return coordinates for plotting the line.

        Returns
        =======

        x: np.ndarray
            x-coordinates
        y: np.ndarray
            y-coordinates
        z: np.ndarray (optional)
            z-coordinates in case of Parametric3DLineSeries
        param : np.ndarray (optional)
            The parameter in case of Parametric2DLineSeries,
            Parametric3DLineSeries or AbsArgLineSeries.
        """
        np = import_module('numpy')
        points = self._get_data_helper()

        if (
            isinstance(self, LineOver1DRangeSeries) and
            (self.detect_poles == "symbolic")
        ):
            poles = _detect_poles_symbolic_helper(
                self.expr.subs(self.params), *self.ranges[0])
            poles = np.array([float(t) for t in poles])
            t = lambda x, transform: x if transform is None else transform(x)
            self.poles_locations = list(t(np.array(poles), self.tx))

        # postprocessing
        points = self._apply_transform(*points)

        if self.is_2Dline and self.detect_poles:
            if len(points) == 2:
                x, y = points
                x, y = _detect_poles_numerical_helper(
                    x, y, self.eps)
                points = (x, y)
            else:
                x, y, p = points
                x, y = _detect_poles_numerical_helper(x, y, self.eps)
                points = (x, y, p)

        if self.unwrap:
            kw = {}
            if self.unwrap is not True:
                kw = self.unwrap
            if self.is_2Dline:
                if len(points) == 2:
                    x, y = points
                    y = np.unwrap(y, **kw)
                    points = (x, y)
                else:
                    x, y, p = points
                    y = np.unwrap(y, **kw)
                    points = (x, y, p)

        if (self.steps is True) or (self.steps == "pre"):
            points = pts_to_prestep(*points)
        elif self.steps == "post":
            points = pts_to_poststep(*points)
        elif self.steps == "mid":
            points = pts_to_midstep(*points)

        points = self._insert_exclusions(points)
        return points

    def _apply_transform(self, *args):
        t = self._get_transform_helper()
        if len(args) == 2:
            x, y = args
            return t(x, self.tx), t(y, self.ty)
        else:
            if self.is_2Dline:
                x, y, p = args
                return t(x, self.tx), t(y, self.ty), t(p, self.tp)
            else:
                if len(args) == 3:
                    x, y, z = args
                    return t(x, self.tx), t(y, self.ty), t(z, self.tz)
                else:
                    x, y, z, p = args
                    return t(x, self.tx), t(y, self.ty), t(z, self.tz), t(p, self.tp)

    def _insert_exclusions(self, points):
        """Add NaN to each of the exclusion point. Practically, this adds a
        NaN to the exlusion point, plus two other nearby points evaluated with
        the numerical functions associated to this data series.
        These nearby points are important when the number of discretization
        points is low, or the scale is logarithm.

        NOTE: it would be easier to just add exclusion points to the
        discretized domain before evaluation, then after evaluation add NaN
        to the exclusion points. But that's only work with adaptive=False.
        The following approach work even with adaptive=True.
        """
        if len(self.exclude) == 0:
            return points

        np = import_module("numpy")
        points = list(points)
        n = len(points)
        # index of the x-coordinate (for 2d plots) or parameter (for 2d/3d
        # parametric plots)
        k = n - 1
        if n == 2:
            k = 0
        # indeces of the other coordinates
        j_indeces = sorted(set(range(n)).difference([k]))
        # TODO: for now, I assume that numpy functions are going to succeed
        funcs = [f[0] for f in self.evaluator._functions]

        for e in self.exclude:
            res = points[k] - e >= 0
            # if res contains both True and False, ie, if e is found
            if any(res) and any(~res):
                idx = np.nanargmax(res)
                # select the previous point with respect to e
                idx -= 1
                # TODO: what if points[k][idx]==e or points[k][idx+1]==e?

                if idx > 0 and idx < len(points[k]) - 1:
                    delta_prev = abs(e - points[k][idx])
                    delta_post = abs(e - points[k][idx + 1])
                    delta = min(delta_prev, delta_post) / 100
                    prev = e - delta
                    post = e + delta

                    # add points to the x-coord or the parameter
                    points[k] = np.concatenate(
                        (points[k][:idx], [prev, e, post], points[k][idx+1:]))

                    # add points to the other coordinates
                    c = 0
                    for j in j_indeces:
                        values = funcs[c](np.array([prev, post]))
                        c += 1
                        points[j] = np.concatenate(
                            (points[j][:idx], [values[0], np.nan, values[1]], points[j][idx+1:]))
        return points

    @property
    def var(self):
        return None if not self.ranges else self.ranges[0][0]

    @property
    def start(self):
        return self.ranges[0][1]

    @property
    def end(self):
        return self.ranges[0][2]


class List2DSeries(Line2DBaseSeries, _TpParameter):
    """
    Representation for a line consisting of list of points.
    """

    # NOTE: these parameters will eventually hold either Tuple or numpy arrays
    list_x = param.Parameter(default=[], doc="""
        Coordinates for the x-axis.""")
    list_y = param.Parameter(default=[], doc="""
        Coordinates for the y-axis.""")
    color_func = param.Callable(default=None, doc="""
        A color function to be applied to the numerical data. It can be:

        * None: no color function.
        * callable: a function accepting two arguments (the x-y coordinates)
          and returning numerical data.
        """)

    def _check_length(self, list_x, list_y, list_z=None):
        n1 = len(list_x)
        n2 = len(list_y)
        is_different_length = n1 != n2
        if list_z is not None:
            n3 = len(list_z)
            is_different_length = is_different_length or (n1 != n3)

        if is_different_length:
            msg = "Received: len(list_x) = %s, len(list_y) = %s" % (
                len(list_x), len(list_y))
            if list_z is not None:
                msg += ", len(list_z) = %s" % len(list_z)
            raise ValueError(
                "The provided lists of coordinates must have the same "
                "number of elements.\n" + msg
            )

    def _cast_to_appropriate_type(self, list_x, list_y, list_z=None, **kwargs):
        expr_in = lambda _list: [
            isinstance(t, Expr) and (not t.is_number) for t in _list]
        expr_in_list_x = expr_in(list_x)
        expr_in_list_y = expr_in(list_y)
        expr_in_list_z = [False] if list_z is None else expr_in(list_z)
        any_expr = any(expr_in_list_x + expr_in_list_y + expr_in_list_z)
        params = kwargs.get("params", None)
        if any_expr or params:
            if not params:
                raise ValueError(
                    "Some or all elements of the provided lists "
                    "are symbolic expressions, but the ``params`` dictionary "
                    "was not provided: those elements can't be evaluated.")
            kwargs["list_x"] = Tuple(*list_x)
            kwargs["list_y"] = Tuple(*list_y)
            if list_z is not None:
                kwargs["list_z"] = Tuple(*list_z)
        else:
            np = import_module('numpy')
            kwargs["list_x"] = np.array(list_x, dtype=np.float64)
            kwargs["list_y"] = np.array(list_y, dtype=np.float64)
            if list_z is not None:
                kwargs["list_z"] = np.array(list_z, dtype=np.float64)

        return any_expr, kwargs

    def __init__(self, list_x, list_y, label="", **kwargs):
        self._check_length(list_x, list_y)
        self._block_lambda_functions(list_x, list_y)
        any_expr, kwargs = self._cast_to_appropriate_type(
            list_x, list_y, **kwargs)
        kwargs["label"] = label
        super().__init__(**kwargs)

        # any_expr_in = lambda _list: [
        #     isinstance(t, Expr) and (not t.is_number) for t in _list]
        # if any(any_expr_in(list_x) + any_expr_in(list_y)) or self.params:
        #     if not self.params:
        #         raise ValueError(
        #             "Some or all elements of the provided lists "
        #             "are symbolic expressions, but the ``params`` dictionary "
        #             "was not provided: those elements can't be evaluated.")
        #     self.list_x = Tuple(*list_x)
        #     self.list_y = Tuple(*list_y)
        # else:
        #     self.list_x = np.array(list_x, dtype=np.float64)
        #     self.list_y = np.array(list_y, dtype=np.float64)

        if any_expr:
            self._check_fs()

        if self.use_cm and self.color_func:
            self.is_parametric = True

    @property
    def expr(self):
        return self.list_x, self.list_y

    def __str__(self):
        pre = "2D" if self.is_2Dline else "3D"
        return pre + " list plot"

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed."""
        lx, ly = self.list_x, self.list_y

        if not self.is_interactive:
            return self._return_correct_elements(lx, ly)

        np = import_module('numpy')
        lx = np.array([t.evalf(subs=self.params) for t in lx], dtype=float)
        ly = np.array([t.evalf(subs=self.params) for t in ly], dtype=float)
        return self._return_correct_elements(lx, ly)

    def _return_correct_elements(self, *data):
        color = self.eval_color_func(*data)
        if color is None:
            return [*data]
        return [*data, color]

    def eval_color_func(self, *data):
        if self.use_cm and callable(self.color_func):
            nargs = arity(self.color_func)
            if nargs == 0:
                color = self.color_func()
            elif nargs == 1:
                # y-coord for 2D lines
                color = self.color_func(data[1])
            elif nargs == 2:
                color = self.color_func(*data)
            else:
                raise ValueError(
                    "The `color_func` attribute of a `List2DSeries`"
                    " must be a function with 1 or 2 parameters. Instead,"
                    f" {nargs} parameters were provided."
                )
            color = _correct_shape(color, data[0])
            return color
        return None


class List3DSeries(_TzParameter, List2DSeries):
    is_2Dline = False
    is_3Dline = True

    list_z = param.Parameter(default=[], doc="""
        Coordinates for the z-axis.""")
    color_func = param.Callable(default=None, doc="""
        A color function to be applied to the numerical data. It can be:

        * None: no color function.
        * callable: a function accepting thre arguments (the x-y-z coordinates)
          and returning numerical data.
        """)

    def __init__(self, list_x, list_y, list_z, label="", **kwargs):
        self._check_length(list_x, list_y, list_z)
        self._block_lambda_functions(list_x, list_y, list_z)
        any_expr, kwargs = self._cast_to_appropriate_type(
            list_x, list_y, list_z, **kwargs)
        super().__init__(label=label, **kwargs)

        if any_expr:
            self._check_fs()

        if self.use_cm and self.color_func:
            self.is_parametric = True

        # # TODO: this can definitely be done better
        # kwargs["list_z"]
        # super().__init__(list_x, list_y, label, **kwargs)
        # np = import_module('numpy')
        # if len(list_z) != len(list_x):
        #     raise ValueError(
        #         "The three lists of coordinates must have the same "
        #         "number of elements.\n"
        #         "Received: len(list_x) = len(list_y) = {} ".format(len(list_x)) +
        #         "and len(list_z) = {}".format(len(list_z))
        #     )
        # self._block_lambda_functions(list_z)
        # check = lambda l: [isinstance(t, Expr) and (not t.is_number) for t in l]
        # if any(check(list_z)):
        #     if not self.params:
        #         raise ValueError(
        #             "Some or all elements of the provided lists "
        #             "are symbolic expressions, but the ``params`` dictionary "
        #             "was not provided: those elements can't be evaluated.")
        #     self.list_z = Tuple(*list_z)
        #     self._check_fs()
        # else:
        #     self.list_z = np.array(list_z, dtype=np.float64)

    @property
    def expr(self):
        return self.list_x, self.list_y, self.list_z

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed."""
        lx, ly, lz = self.list_x, self.list_y, self.list_z

        if not self.is_interactive:
            return self._return_correct_elements(lx, ly, lz)

        np = import_module('numpy')
        lx = np.array([t.evalf(subs=self.params) for t in lx], dtype=float)
        ly = np.array([t.evalf(subs=self.params) for t in ly], dtype=float)
        lz = np.array([t.evalf(subs=self.params) for t in lz], dtype=float)
        return self._return_correct_elements(lx, ly, lz)

    def eval_color_func(self, *data):
        if self.use_cm and callable(self.color_func):
            nargs = arity(self.color_func)
            if nargs == 0:
                color = self.color_func()
            elif nargs == 1:
                # z-coord for 3D lines
                color = self.color_func(data[2])
            elif nargs == 2:
                color = self.color_func(*data[:2])
            elif nargs == 3:
                color = self.color_func(*data)
            else:
                raise ValueError(
                    "The `color_func` attribute of a `List3DSeries`"
                    " must be a function with 1 or 2 or 3 parameters. Instead,"
                    f" {nargs} parameters were provided."
                )
            color = _correct_shape(color, data[0])
            return color
        return None


class LineOver1DRangeSeries(
    _GridEvaluationParameters,
    _AdaptiveEvaluationParameters,
    Line2DBaseSeries
):
    """
    Representation for a line consisting of a SymPy expression over a
    real range.
    """

    _allowed_keys = [
        "absarg", "is_complex", "is_polar"
    ]
    _exclude_params_from_doc = ["zscale"]

    expr = param.Parameter(doc="""
        It can either be a symbolic expression representing the function
        of one variable to be plotted, or a numerical function of one
        variable, supporting vectorization. In the latter case the following
        keyword arguments are not supported: ``params``, ``sum_bound``.""")
    color_func = param.Parameter(doc="""
        A color function to be applied to the numerical data. It can be:

        * A numerical function of 2 variables, x, y (the points computed by
          the internal algorithm) supporting vectorization.
        * A symbolic expression having at most as many free symbols as
          ``expr``.
        * None: the default value (no color mapping).
        """)
    def __new__(cls, *args, **kwargs):
        if kwargs.get("absarg", False):
            return super().__new__(AbsArgLineSeries)
        cf = kwargs.get("color_func", None)
        lc = kwargs.get("line_color", None)
        if (callable(cf) or callable(lc) or isinstance(cf, Expr)):
            return super().__new__(ColoredLineOver1DRangeSeries)
        return object.__new__(cls)

    def __init__(self, expr, var_start_end, label="", **kwargs):
        _return = kwargs.pop("return", None)
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self.evaluator = GridEvaluator(series=self)
        self._label_str = str(self.expr) if label is None else label
        self._label_latex = latex(self.expr) if label is None else label
        self.ranges = [var_start_end]
        # for complex-related data series, this determines what data to return
        # on the y-axis
        self._return = _return
        self._post_init()

        if not self._parametric_ranges:
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
            goal=self._goal,
            loss_fn=self.loss_fn)
        return data[:, 0], data[:, 1], data[:, 2]

    def _uniform_sampling(self):
        np = import_module('numpy')
        x, result = self.evaluator._evaluate()
        _re, _im = np.real(result), np.imag(result)
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
            raise ValueError(
                "`_return` not recognized. Received: %s" % self._return)

        return x, _re

    # def _apply_transform(self, *args):
    #     t = self._get_transform_helper()
    #     x, y = args
    #     return t(x, self.tx), t(y, self.ty)


class ColoredLineOver1DRangeSeries(_TpParameter, LineOver1DRangeSeries):
    """Represents a 2D line series in which `color_func` is a callable.
    """
    is_parametric = True

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_cm", True)
        super().__init__(*args, **kwargs)

    def eval_color_func(self, x, y):
        color = self.evaluator.eval_color_func(x, y)
        if color is not None:
            return color

        nargs = arity(self.evaluator.color_func)
        if nargs == 1:
            color = self.evaluator.color_func(x)
        elif nargs == 2:
            color = self.evaluator.color_func(x, y)
        else:
            _raise_color_func_error(self, nargs)
        return _correct_shape(color, x)

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        x, y = super()._get_data_helper()
        return x, y, self.eval_color_func(x, y)

    # def _apply_transform(self, *args):
    #     t = self._get_transform_helper()
    #     x, y, p = args
    #     return t(x, self.tx), t(y, self.ty), t(p, self.tp)


class _2DParametricParameters(param.Parameterized):
    expr_x = param.Parameter(doc="""
        The expression representing the component along the x-axis of the
        parametric function.
        It can either be a symbolic expression representing the function of
        one variable to be plotted, or a numerical function of one variable,
        supporting vectorization. In the latter case the following keyword
        arguments are not supported: ``params``, ``sum_bound``.""")
    expr_y = param.Parameter(doc="""
        The expression representing the component along the y-axis of the
        parametric function.
        It can either be a symbolic expression representing the function of
        one variable to be plotted, or a numerical function of one variable,
        supporting vectorization. In the latter case the following keyword
        arguments are not supported: ``params``, ``sum_bound``.""")
    exclude = param.List([], item_type=float, doc="""
        A list of numerical values along the parameter which are going to
        be excluded from the evaluation. In practice, it introduces
        discontinuities in the resulting line.""")


class _3DParametricParameters(_2DParametricParameters):
    expr_z = param.Parameter(doc="""
        The expression representing the component along the z-axis of the
        parametric function.
        It can either be a symbolic expression representing the function of
        one variable to be plotted, or a numerical function of one variable,
        supporting vectorization. In the latter case the following keyword
        arguments are not supported: ``params``, ``sum_bound``.""")


class ParametricLineBaseSeries(
    _GridEvaluationParameters,
    Line2DBaseSeries,
    _TpParameter,
    _AdaptiveEvaluationParameters
):
    is_parametric = True

    is_polar = param.Boolean(False, doc="""
        If True, apply a cartesian to polar transformation.""")

    def _set_parametric_line_label(self, label):
        """Logic to set the correct label to be shown on the plot.
        If `use_cm=True` there will be a colorbar, so we show the parameter.
        If `use_cm=False`, there might be a legend, so we show the expressions.

        Parameters
        ==========
        label : str
            label passed in by the pre-processor or the user
        """
        self._label_str = str(self.var) if label is None else label
        self._label_latex = latex(self.var) if label is None else label
        if (self.use_cm is False) and (self._label_str == str(self.var)):
            self._label_str = str(self.expr)
            self._label_latex = latex(self.expr)
        # if the expressions is a lambda function and use_cm=False and no label
        # has been provided, then its better to do the following in order to
        # avoid suprises on the backend
        if any(callable(e) for e in self.expr) and (not self.use_cm):
            if self._label_str == str(self.expr):
                self._label_str = ""

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
                expr = lambda x: (
                    self.expr_x(x), self.expr_y(x), self.expr_z(x))

        data = _adaptive_eval(
            func, [self.var], expr,
            [float(self.start), float(self.end)],
            self.is_2Dline,
            modules=self.modules,
            goal=self._goal,
            loss_fn=self.loss_fn)

        if self.is_2Dline:
            return data[:, 1], data[:, 2], data[:, 0]
        return data[:, 1], data[:, 2], data[:, 3], data[:, 0]

    def get_label(self, use_latex=False, wrapper="$%s$"):
        # parametric lines returns the representation of the parameter to be
        # shown on the colorbar if `use_cm=True`, otherwise it returns the
        # representation of the expression to be placed on the legend.
        if self.use_cm:
            if str(self.var) == self._label_str:
                if use_latex:
                    return self._get_wrapped_label(latex(self.var), wrapper)
                return str(self.var)
            # here the user has provided a custom label
            return self._label_str
        if use_latex:
            if self._label_str != str(self.expr):
                return self._label_latex
            return self._get_wrapped_label(self._label_latex, wrapper)
        return self._label_str

    def eval_color_func(self, *coords):
        color = self.evaluator.eval_color_func(*coords)
        if color is not None:
            return color

        nargs = arity(self.evaluator.color_func)
        if nargs == 1:
            color = self.evaluator.color_func(coords[-1])
        elif nargs == 2:
            if self.is_2Dline:
                color = self.evaluator.color_func(*coords[:2])
        elif nargs == 3:
            color = self.evaluator.color_func(*coords[:3])
        elif nargs == 4:
            if self.is_3Dline:
                color = self.evaluator.color_func(*coords)
        else:
            _raise_color_func_error(self, nargs)
        return _correct_shape(color, coords[0])

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

        results = self.evaluator._evaluate()
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        return [*results[1:], results[0]]


class Parametric2DLineSeries(
    _2DParametricParameters,
    ParametricLineBaseSeries
):
    """
    Representation for a line consisting of two parametric sympy expressions
    over a range."""

    is_parametric = True

    color_func = param.Parameter(default=None, doc="""
        Define a custom color mapping when ``use_cm=True``. It can either be:

        * A numerical function supporting vectorization. The arity can be:

          * 1 argument: ``f(t)``, where ``t`` is the parameter.
          * 2 arguments: ``f(x, y)`` where ``x, y`` are the coordinates of
            the points.
          * 3 arguments: ``f(x, y, t)``.

        * A symbolic expression having at most as many free symbols as
          ``expr_x`` or ``expr_y``.
        * None: the default value (color mapping according to the parameter).
        """)

    def __init__(self, expr_x, expr_y, var_start_end, label="", **kwargs):
        kwargs["expr_x"] = expr_x if callable(expr_x) else sympify(expr_x)
        kwargs["expr_y"] = expr_y if callable(expr_y) else sympify(expr_y)
        kwargs.setdefault("use_cm", True)
        super().__init__(**kwargs)
        self.expr = (self.expr_x, self.expr_y)
        self.ranges = [var_start_end]
        self.evaluator = GridEvaluator(series=self)
        self._set_parametric_line_label(label)
        self._post_init()

    # def _apply_transform(self, *args):
    #     t = self._get_transform_helper()
    #     x, y, p = args
    #     return t(x, self.tx), t(y, self.ty), t(p, self.tp)

    def __str__(self):
        return self._str_helper(
            "parametric cartesian line: (%s, %s) for %s over %s" % (
                str(self.expr_x),
                str(self.expr_y),
                str(self.var),
                str((self.start, self.end))
            )
        )


class Parametric3DLineSeries(
    _3DParametricParameters,
    _TzParameter,
    ParametricLineBaseSeries
):
    """
    Representation for a 3D line consisting of three parametric sympy
    expressions and a range."""

    is_2Dline = False
    is_3Dline = True

    color_func = param.Parameter(default=None, doc="""
        Define a custom color mapping when ``use_cm=True``. It can either be:

        * A numerical function supporting vectorization. The arity can be:

          * 1 argument: ``f(t)``, where ``t`` is the parameter.
          * 3 arguments: ``f(x, y, z)`` where ``x, y, z`` are the coordinates
            of the points.
          * 4 arguments: ``f(x, y, z, t)``.

        * A symbolic expression having at most as many free symbols as
          ``expr_x`` or ``expr_y`` or ``expr_z``.
        * None: the default value (color mapping according to the parameter).
        """)

    def __init__(
        self, expr_x, expr_y, expr_z, var_start_end, label="", **kwargs
    ):
        kwargs["expr_x"] = expr_x if callable(expr_x) else sympify(expr_x)
        kwargs["expr_y"] = expr_y if callable(expr_y) else sympify(expr_y)
        kwargs["expr_z"] = expr_z if callable(expr_z) else sympify(expr_z)
        kwargs.setdefault("use_cm", True)
        super().__init__(**kwargs)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end]
        self.evaluator = GridEvaluator(series=self)
        self._set_parametric_line_label(label)
        self._post_init()

    # def _apply_transform(self, *args):
    #     t = self._get_transform_helper()
    #     x, y, z, p = args
    #     return t(x, self.tx), t(y, self.ty), t(z, self.tz), t(p, self.tp)

    def __str__(self):
        return self._str_helper(
            "3D parametric cartesian line: (%s, %s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.expr_z),
            str(self.var),
            str((self.start, self.end))
        ))


# class PublicSurfaceParameters(
#     _IsPolarForSurfaceParameter,
#     _TzParameter,
#     BaseSeries
# ):
#     pass


class SurfaceBaseSeries(
    _GridEvaluationParameters,
    _TzParameter,
    BaseSeries
):
    """A base class for 3D surfaces."""

    is_3Dsurface = True
    _allowed_keys = ["surface_color", "is_polar"]

    surface_color = param.Parameter(default=None, doc="""
        For back-compatibility with old sympy.plotting. Use ``rendering_kw``
        in order to fully customize the appearance of the surface.""")
    is_polar = param.Boolean(False, doc="""
        If True, requests a polar discretization. In this case,
        ``range1`` represents the radius, while ``range2`` represents
        the angle.""")
    rendering_kw = param.Dict(default={}, doc="""
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the surface.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib:
          https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface
        * Plotly:
          https://plotly.com/python/3d-surface-plots/
        * K3D-Jupyter: look at the documentation of k3d.mesh.
        """)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("color_func", lambda x, y, z: z)
        super().__init__(**kwargs)
        # NOTE: why should SurfaceOver2DRangeSeries support is polar?
        # After all, the same result can be achieve with
        # ParametricSurfaceSeries. For example:
        # sin(r) for (r, 0, 2 * pi) and (theta, 0, pi/2) can be parameterized
        # as (r * cos(theta), r * sin(theta), sin(t)) for (r, 0, 2 * pi) and
        # (theta, 0, pi/2).
        # Because it is faster to evaluate (important for interactive plots).

        if callable(self.surface_color):
            self.color_func = self.surface_color
            self.surface_color = None

    def _set_surface_label(self, label):
        exprs = self.expr
        self._label_str = str(exprs) if label is None else label
        self._label_latex = latex(exprs) if label is None else label
        # if the expressions is a lambda function and no label
        # has been provided, then its better to do the following to avoid
        # suprises on the backend
        is_lambda = (callable(exprs) if not hasattr(exprs, "__iter__")
            else any(callable(e) for e in exprs))
        if is_lambda and (self._label_str == str(exprs)):
            self._label_str = ""
            self._label_latex = ""

    @param.depends("surface_color", watch=True, on_init=True)
    def _update_surface_color(self):
        self._line_surface_color("_surface_color", self.surface_color)

    def _apply_transform(self, *args):
        t = self._get_transform_helper()
        if len(args) == 3: # general surfaces
            x, y, z = args
            return t(x, self.tx), t(y, self.ty), t(z, self.tz)
        elif len(args) == 5: # parametric surfaces
            x, y, z, u, v = args
            return (
                t(x, self.tx), t(y, self.ty), t(z, self.tz), u, v)
        else: # complex domain coloring surfaces
            x, y, _abs, _arg, img, colors = args
            return (
                t(x, self.tx), t(y, self.ty), t(_abs, self.tz),
                _arg, img, colors
            )


class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """
    Representation for a 3D surface consisting of a sympy expression and 2D
    range.
    """

    color_func = param.Parameter(default=None, doc="""
        Define a custom color mapping. It can either be:

        * A numerical function supporting vectorization. The arity can be:

          * 2 arguments: ``f(x, y)`` where ``x, y`` are the coordinates of
            the points.
          * 3 arguments: ``f(x, y, z)`` where ``x, y, z`` are the coordinates
            of the points.
        * A symbolic expression having at most as many free symbols as
          ``expr``.
        * None: the default value (color mapping according to the
          z coordinate).
        """)

    def __init__(
        self, expr, var_start_end_x, var_start_end_y, label="", **kwargs
    ):
        kwargs["expr"] = expr if callable(expr) else sympify(expr)
        super().__init__(**kwargs)
        self.ranges = [var_start_end_x, var_start_end_y]
        self.evaluator = GridEvaluator(series=self)
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
        return self.ranges[0][1]

    @property
    def end_x(self):
        return self.ranges[0][2]

    @property
    def start_y(self):
        return self.ranges[1][1]

    @property
    def end_y(self):
        return self.ranges[1][2]

    def __str__(self):
        series_type = "cartesian surface" if self.is_3Dsurface else "contour"
        return self._str_helper(
            series_type + ": %s for" " %s over %s and %s over %s" % (
                str(self.expr),
                str(self.var_x), str((self.start_x, self.end_x)),
                str(self.var_y), str((self.start_y, self.end_y)),
            )
        )

    def get_data(self):
        """
        Return arrays of coordinates for plotting. Depending on the
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

        results = self.evaluator._evaluate()
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        x, y, z = results
        if self.is_polar and self.is_3Dsurface:
            r = x.copy()
            x = r * np.cos(y)
            y = r * np.sin(y)

        return self._apply_transform(x, y, z)

    def eval_color_func(self, *coords):
        color = self.evaluator.eval_color_func(*coords)
        if color is not None:
            return color

        nargs = arity(self.evaluator.color_func)
        if nargs == 1:
            color = self.evaluator.color_func(coords[0])
        elif nargs == 2:
            color = self.evaluator.color_func(*coords[:2])
        elif nargs == 3:
            color = self.evaluator.color_func(*coords)
        else:
            _raise_color_func_error(self, nargs)
        return _correct_shape(color, coords[0])


class ParametricSurfaceSeries(
    _3DParametricParameters,
    SurfaceBaseSeries
):
    """
    Representation for a 3D surface consisting of three parametric sympy
    expressions and a range.
    """

    is_parametric = True

    def __init__(
        self, expr_x, expr_y, expr_z,
        var_start_end_u, var_start_end_v, label="", **kwargs
    ):
        kwargs["expr_x"] = expr_x if callable(expr_x) else sympify(expr_x)
        kwargs["expr_y"] = expr_y if callable(expr_y) else sympify(expr_y)
        kwargs["expr_z"] = expr_z if callable(expr_z) else sympify(expr_z)
        kwargs.setdefault("color_func", lambda x, y, z, u, v: z)
        super().__init__(**kwargs)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end_u, var_start_end_v]
        self.evaluator = GridEvaluator(series=self)
        self._set_surface_label(label)
        self._post_init()

    @property
    def var_u(self):
        return self.ranges[0][0]

    @property
    def var_v(self):
        return self.ranges[1][0]

    @property
    def start_u(self):
        return self.ranges[0][1]

    @property
    def end_u(self):
        return self.ranges[0][2]

    @property
    def start_v(self):
        return self.ranges[1][1]

    @property
    def end_v(self):
        return self.ranges[1][2]

    def __str__(self):
        return self._str_helper(
            "parametric cartesian surface: (%s, %s, %s) for"
            " %s over %s and %s over %s" % (
                str(self.expr_x), str(self.expr_y), str(self.expr_z),
                str(self.var_u), str((self.start_u, self.end_u)),
                str(self.var_v), str((self.start_v, self.end_v)),
            )
        )

    def get_data(self):
        """
        Return arrays of coordinates for plotting. Depending on the
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

        results = self.evaluator._evaluate()
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        return self._apply_transform(*results[2:], *results[:2])

    def eval_color_func(self, *coords):
        color = self.evaluator.eval_color_func(*coords)
        if color is not None:
            return color

        nargs = arity(self.evaluator.color_func)
        if nargs == 1:
            color = self.evaluator.color_func(coords[3])
        elif nargs == 2:
            color = self.evaluator.color_func(coords[3], coords[4])
        elif nargs == 3:
            color = self.evaluator.color_func(*coords[:3])
        elif nargs == 5:
            color = self.evaluator.color_func(*coords)
        else:
            _raise_color_func_error(self, nargs)
        return _correct_shape(color, coords[0])


class ContourSeries(SurfaceOver2DRangeSeries):
    """Representation for a contour plot."""

    is_3Dsurface = False
    is_contour = True
    _allowed_keys = [
        "contour_kw", "is_filled", "fill", "clabels"]

    is_filled = param.Boolean(True, doc="""
        If True, use filled contours. Otherwise, use line contours.""")
    show_clabels = param.Boolean(True, doc="""
        Toggle the label's visibility of contour lines. Only works when
        ``is_filled=False``. Note that some backend might not implement
        this feature.""")
    rendering_kw = param.Dict(default={}, doc="""
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the contour.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib:
          https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
        * Plotly:
          https://plotly.com/python/contour-plots/
        """)

    def __init__(self, *args, **kwargs):
        kwargs = kwargs.copy()
        kwargs.setdefault("show_in_legend", False)
        kwargs.setdefault("use_cm", True)
        kwargs.setdefault("is_filled", kwargs.pop("fill", True))
        kwargs.setdefault("show_clabels", kwargs.pop("clabels", True))
        # NOTE: contour plots are used by plot_contour, plot_vector and
        # plot_complex_vector. By implementing contour_kw we are able to
        # quickly target the contour plot.
        kwargs.setdefault("rendering_kw", kwargs.pop("contour_kw", dict()))
        super().__init__(*args, **kwargs)


class ImplicitSeries(
    _GridEvaluationParameters,
    BaseSeries
):
    """
    Representation for Implicit plot

    References
    ==========

    .. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for
    Mathematical Formulae with Two Free Variables.

    .. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval
    Arithmetic. Master's thesis. University of Toronto, 1996
    """

    is_implicit = True
    use_cm = False
    _allowed_keys = ["adaptive", "depth", "color"]
    _N = 100

    # expr = param.Parameter()
    adaptive = param.Boolean(False, doc="""
        Select the evaluation strategy to be used.
        If ``False``, the internal algorithm uses a mesh grid approach.
        In such case, Boolean combinations of expressions cannot be plotted.
        If ``True``, the internal algorithm uses interval arithmetic.
        If the expression cannot be plotted with interval arithmetic, it
        switches to the meshgrid approach.""")
    depth = param.Integer(default=0, bounds=(0, 4), doc="""
        The depth of recursion for adaptive grid. Default value is 0.
        Think of the resulting plot as a picture composed by pixels. By
        increasing ``depth`` we are increasing the number of pixels, thus
        obtaining a more accurate plot, at the cost of evaluation speed
        and possibly readability (if the figure has small size).""")
    _actual_depth = param.Integer(default=4, bounds=(4, 8), doc="""
        The ``depth`` is meant to be user friendly. It will be processed
        by the data series. The new value used by the algorithm is stored
        in this parameter.""")
    _adaptive_expr = param.ClassSelector(
        class_=(Expr, Relational, BooleanFunction), doc="""
        The expression to be evaluated by the adaptive algorithm.""")
    _non_adaptive_expr = param.ClassSelector(
        class_=(Expr, Relational, BooleanFunction), doc="""
        The expression to be evaluated by the uniform mesh
        evaluation algorithm.""")
    _has_relational = param.Boolean(default=False, doc="""
        This flag is set by the algorithm when the symbolic expression
        is going to be set. It indicates the presence of ``Equality``,
        ``GreaterThan``, ``LessThan`` in the symbolic expression.""")
    _is_equality = param.Boolean(default=False, doc="""
        This flag is set by the algorithm when the symbolic expression
        is going to be set. It indicates the presence of ``Equality``
        in the symbolic expression.""")
    rendering_kw = param.Dict(default={}, doc="""
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the contour.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib:
          https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
        * Plotly:
          https://plotly.com/python/contour-plots/
        """)

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        kwargs = kwargs.copy()
        kwargs["label"] = label if label is not None else ""
        # kwargs["_label_str"] = str(expr) if label is None else label
        # kwargs["_label_latex"] = latex(expr) if label is None else label
        color = kwargs.pop("color", kwargs.get("line_color", None))
        kwargs = self._preprocess_expr(expr, kwargs)
        super().__init__(**kwargs)

        self.ranges = [var_start_end_x, var_start_end_y]
        self.evaluator = GridEvaluator(series=self)
        self.var_x, self.start_x, self.end_x = self.ranges[0]
        self.var_y, self.start_y, self.end_y = self.ranges[1]
        self._color = color
        # if label is not None:
        #     self._update_latex_and_str_labels()

        if self.is_interactive and self.adaptive:
            raise NotImplementedError("Interactive plot with `adaptive=True` "
                "is not supported.")

        # # Check whether the depth is greater than 4 or less than 0.
        # if depth > 4:
        #     depth = 4
        # elif depth < 0:
        #     depth = 0
        # self.depth = 4 + depth
        self._post_init()

    @param.depends("depth", on_init=True, watch=True)
    def _update_actual_depth(self):
        self._actual_depth = self.depth + 4

    @property
    def expr(self):
        if self.adaptive:
            return self._adaptive_expr
        return self._non_adaptive_expr

    @expr.setter
    def expr(self, value):
        kwargs = self._preprocess_expr(value, {})
        self.param.update(kwargs)

    def _preprocess_expr(self, expr, kwargs):
        self._block_lambda_functions(expr)
        # these are needed for adaptive evaluation
        expr, has_relational = self._has_relational(sympify(expr))
        kwargs["_adaptive_expr"] = expr
        kwargs["_has_relational"] = has_relational
        kwargs["_label_str"] = str(expr)
        kwargs["_label_latex"] = latex(expr)

        adaptive = kwargs.get("adaptive", self.param.adaptive.default)
        if isinstance(expr, (BooleanFunction, Ne)) and (not self.adaptive):
            kwargs["adaptive"] = adaptive = True
            msg = "contains Boolean functions. "
            if isinstance(expr, Ne):
                msg = "is an unequality. "
            warnings.warn(f"The provided expression {msg}"
                "In order to plot the expression, the algorithm "
                "automatically switched to an adaptive sampling.",
                stacklevel=1)

        if isinstance(expr, BooleanFunction):
            # NOTE: at this stage, I'll be using the adaptive algorithm.
            # However, I need to set the expression for the uniform meshing
            # algorithm too, in case the adaptive algorithm fails to evaluate.
            kwargs["_non_adaptive_expr"] = expr
            kwargs["_is_equality"] = False
        else:
            # these are needed for uniform meshing evaluation
            expr, is_equality = self._preprocess_meshgrid_expression(
                expr, adaptive)
            kwargs["_non_adaptive_expr"] = expr
            kwargs["_is_equality"] = is_equality
        return kwargs

    # @expr.setter
    # def expr(self, expr):
    #     self._block_lambda_functions(expr)
    #     # these are needed for adaptive evaluation
    #     expr, has_equality = self._has_equality(sympify(expr))
    #     self._adaptive_expr = expr
    #     self.has_equality = has_equality
    #     self._label_str = str(expr)
    #     self._label_latex = latex(expr)

    #     if isinstance(expr, (BooleanFunction, Ne)) and (not self.adaptive):
    #         self.adaptive = True
    #         msg = "contains Boolean functions. "
    #         if isinstance(expr, Ne):
    #             msg = "is an unequality. "
    #         warnings.warn(f"The provided expression {msg}"
    #             "In order to plot the expression, the algorithm "
    #             "automatically switched to an adaptive sampling.",
    #             stacklevel=1)

    #     if isinstance(expr, BooleanFunction):
    #         self._non_adaptive_expr = None
    #         self._is_equality = False
    #     else:
    #         # these are needed for uniform meshing evaluation
    #         expr, is_equality = self._preprocess_meshgrid_expression(
    #             expr, self.adaptive)
    #         self._non_adaptive_expr = expr
    #         self._is_equality = is_equality

    @property
    def line_color(self):
        return self._color

    @line_color.setter
    def line_color(self, v):
        self._color = v

    color = line_color

    def _has_relational(self, expr):
        # Represents whether the expression contains an Equality, GreaterThan
        # or LessThan
        has_relational = False

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
                has_relational = True
        elif not isinstance(expr, Relational):
            expr = Equality(expr, 0)
            has_relational = True
        elif isinstance(expr, (Equality, GreaterThan, LessThan)):
            has_relational = True

        return expr, has_relational

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
        """
        Returns numerical data.

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
        except (AttributeError, TypeError) as err:
            # XXX: AttributeError("'list' object has no attribute 'is_real'")
            # That needs fixing somehow - we shouldn't be catching
            # AttributeError here.
            warnings.warn(
                "Adaptive meshing could not be applied to the"
                " expression, thus uniform meshing will be used."
                "The following is the error for debuggin purposes.\n"
                f"{err.__class__.__name__}: {err}")
            self.adaptive = False

        return data

    def _get_raster_interval(self, func):
        """Uses interval math to adaptively mesh and obtain the plot"""
        np = import_module('numpy')

        k = self._actual_depth
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
        if self._has_relational:
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

        xarray, yarray, z_grid = self.evaluator._evaluate()
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
        """
        Return the label to be used to display the expression.

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
            return self._label_str
        if (
            (self._label_str == str(self._adaptive_expr)) or
            ("Eq(%s, 0)" % self._label_str == str(self._adaptive_expr))
        ):
            return self._get_wrapped_label(self._label_latex, wrapper)
        return self._label_latex


class Implicit3DSeries(SurfaceBaseSeries):
    is_implicit = True
    _N = 60

    rendering_kw = param.Dict(default={}, doc="""
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the surface.
        Here are some useful links for the supported plotting libraries:

        * Plotly:
          https://plotly.com/python/3d-isosurface-plots/
        * K3D-Jupyter: look at the documentation of k3d.marching_cubes.
        """)

    def __init__(self, expr, range_x, range_y, range_z, label="", **kwargs):
        # kwargs.setdefault("n", self._N)
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self.ranges = [range_x, range_y, range_z]
        self.evaluator = GridEvaluator(series=self)
        self.var_x, self.start_x, self.end_x = self.ranges[0]
        self.var_y, self.start_y, self.end_y = self.ranges[1]
        self.var_z, self.start_z, self.end_z = self.ranges[2]
        if isinstance(self.expr, Plane):
            self.expr = self.expr.equation(self.var_x, self.var_y, self.var_z)
        self._set_surface_label(label)

    def __str__(self):
        var_x, start_x, end_x = self.ranges[0]
        var_y, start_y, end_y = self.ranges[1]
        var_z, start_z, end_z = self.ranges[2]
        return (
            "implicit surface series: %s for %s over %s and %s over %s"
            " and %s over %s") % (
                str(self.expr),
                str(var_x), str((float(start_x), float(end_x))),
                str(var_y), str((float(start_y), float(end_y))),
                str(var_z), str((float(start_z), float(end_z)))
            )

    def get_data(self):
        """
        Evaluate the expression over the provided domain. The backend will
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

        results = self.evaluator._evaluate()
        for i, r in enumerate(results):
            re_v, im_v = np.real(r), np.imag(r)
            re_v[np.invert(np.isclose(im_v, np.zeros_like(im_v)))] = np.nan
            results[i] = re_v

        return self._apply_transform(*results)

    def _apply_transform(self, *args):
        t = self._get_transform_helper()
        x, y, z, f = args
        return t(x, self.tx), t(y, self.ty), t(z, self.tz), f


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
        # super().__init__(**kwargs)
        # kwargs.setdefault("n", self._N)
        self._block_lambda_functions(plane)
        color_func = kwargs.get("color_func", None)
        if (color_func is not None) and (not callable(color_func)):
            raise ValueError(
                "`PlaneSeries`'s `color_func` attribute must be callable."
                " Here is its documentation:\n\n"
                f"{self.param.color_func.doc}")
        super().__init__(**kwargs)
        self.plane = sympify(plane)
        self.expr = self.plane
        if not isinstance(self.plane, Plane):
            raise TypeError(
                "`plane` must be an instance of sympy.geometry.Plane")
        self.x_range = sympify(x_range)
        self.y_range = sympify(y_range)
        self.z_range = sympify(z_range)
        self.ranges = [self.x_range, self.y_range, self.z_range]
        self._set_surface_label(label)
        if self.params and not self.plane.free_symbols:
            self.params = dict()
            with param.edit_constant(self):
                self._is_interactive = False

    def __str__(self):
        return self._str_helper(
            "plane series: %s over %s, %s, %s" % (
                self.plane, self.x_range, self.y_range, self.z_range))

    def get_data(self):
        np = import_module('numpy')

        x, y, z = symbols("x, y, z")
        plane = self.plane.subs(self.params)
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
                xscale=self.scales[0],
                yscale=self.scales[1]
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
                xscale=self.scales[0],
                yscale=self.scales[1]
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

            # plane has distance to origin as length of projection of
            # p1 onto normal vector
            proj_p2nv = nv.dot(p1)

            s = SurfaceOver2DRangeSeries(
                proj_p2nv,
                (x, *self.x_range[1:]),
                (y, *self.z_range[1:]),
                "",
                n1=self.n[0],
                n2=self.n[2],
                xscale=self.scales[0],
                yscale=self.scales[1]
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
                xscale=self.scales[0],
                yscale=self.scales[1]
            )
            xx, yy, zz = s.get_data()
            if (len(fs) > 1) and self._use_nan:
                idx = np.logical_or(zz < self.z_range[1], zz > self.z_range[2])
                zz[idx] = np.nan
        return self._apply_transform(xx, yy, zz)

    def eval_color_func(self, *coords):
        if self.color_func is None:
            return None
        nargs = arity(self.color_func)
        if nargs == 3:
            color = self.color_func(*coords)
        else:
            _raise_color_func_error(self, nargs)
        return _correct_shape(color, coords[0])


class GeometrySeries(_TzParameter, Line2DBaseSeries):
    """
    Represents an entity from the sympy.geometry module.
    Depending on the geometry entity, this class can either represents a
    point, a line, or a parametric line
    """

    is_geometry = True

    def __new__(cls, *args, **kwargs):
        if isinstance(args[0], Plane):
            return PlaneSeries(*args, **kwargs)
        elif isinstance(args[0], Curve):
            new_cls = (
                Parametric2DLineSeries
                if len(args[0].functions) == 2
                else Parametric3DLineSeries
            )
            return new_cls(*args[0].functions, args[0].limits, **kwargs)
        return object.__new__(cls)

    def __init__(self, expr, _range=None, label="", **kwargs):
        if not isinstance(expr, GeometryEntity):
            raise ValueError(
                "`expr` must be a geomtric entity.\n"
                + "Received: type(expr) = {}\n".format(type(expr))
                + "Expr: {}".format(expr)
            )

        kwargs["expr"] = expr
        kwargs["_label_str"] = str(expr) if label is None else label
        kwargs["_label_latex"] = latex(expr) if label is None else label
        super().__init__(**kwargs)
        r = expr.free_symbols.difference(set(self.params.keys()))
        if len(r) > 0:
            raise ValueError(
                "Too many free symbols. Please, specify the values of the "
                f"following symbols with the `params` dictionary: {r}"
            )

        if isinstance(expr, (LinearEntity3D, Point3D)):
            self.is_2Dline = False
            self.is_3Dline = True
            self.is_parametric = False
            if isinstance(expr, Point3D):
                self.is_point = True
        elif isinstance(expr, LinearEntity2D):
            self.is_2Dline = True
        elif isinstance(expr, (Polygon, Circle, Ellipse)):
            self.is_2Dline = not self.is_filled
        elif isinstance(expr, Point2D):
            self.is_point = True
            self.is_2Dline = True
            self.poles_locations = []

    def get_data(self):
        np = import_module('numpy')

        expr = self.expr.subs(self.params)
        if isinstance(expr, Point3D):
            return self._apply_transform(
                np.array([expr.x], dtype=float),
                np.array([expr.y], dtype=float),
                np.array([expr.z], dtype=float)
            )
        elif isinstance(expr, Point2D):
            return self._apply_transform(
                np.array([expr.x], dtype=float),
                np.array([expr.y], dtype=float)
            )
        elif isinstance(expr, Polygon):
            x = [float(v.x) for v in expr.vertices]
            y = [float(v.y) for v in expr.vertices]
            x.append(x[0])
            y.append(y[0])
            return self._apply_transform(np.array(x), np.array(y))
        elif isinstance(expr, Circle):
            cx, cy = float(expr.center[0]), float(expr.center[1])
            r = float(expr.radius)
            t = np.linspace(0, 2 * np.pi, self.n[0])
            x, y = cx + r * np.cos(t), cy + r * np.sin(t)
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            return self._apply_transform(x, y)
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
            return self._apply_transform(x, y)
        elif isinstance(expr, LinearEntity3D):
            p1, p2 = expr.points
            x = np.array([p1.x, p2.x], dtype=float)
            y = np.array([p1.y, p2.y], dtype=float)
            z = np.array([p1.z, p2.z], dtype=float)
            return self._apply_transform(x, y, z)
        elif isinstance(expr, (Segment, Ray)):
            p1, p2 = expr.points
            x = np.array([p1.x, p2.x])
            y = np.array([p1.y, p2.y])
            return self._apply_transform(x.astype(float), y.astype(float))
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
            return self._apply_transform(x.astype(float), y.astype(float))

    def __str__(self):
        return self._str_helper("geometry entity: %s" % str(self.expr))


class GenericDataSeries(BaseSeries):
    """
    Represents generic numerical data.

    Notes
    =====

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

    is_filled = param.Parameter()

    def __init__(self, tp, *args, **kwargs):
        kwargs_without_fill = kwargs.copy()
        kwargs_without_fill.pop("fill", None)
        super().__init__(**kwargs_without_fill)
        self.type = tp
        self.args = args
        self.rendering_kw = kwargs

    def get_data(self):
        return self.args

class HVLineSeries(BaseSeries):
    """
    Represent an horizontal or vertical line series.
    In Matplotlib, this will be rendered by axhline or axvline.
    """
    is_horizontal = param.Boolean(default=True, doc="""
        If True, the series represents and horizontal line. Otherwise,
        it represents a vertical line.""")

    def __init__(self, v, horizontal, label="", **kwargs):
        expr = sympify(v)
        kwargs["expr"] = expr
        kwargs["is_horizontal"] = horizontal
        kwargs["_label_str"] = str(expr) if label is None else label
        kwargs["_label_latex"] = latex(expr) if label is None else label
        super().__init__(**kwargs)

    def get_data(self):
        location = self.expr
        if self.is_interactive:
            location = self.expr.subs(self.params)
        return float(location)

    def __str__(self):
        pre = "horizontal" if self.is_horizontal else "vertical"
        post = "y = " if self.is_horizontal else "x = "
        return self._str_helper(pre + " line at " + post + str(self.expr))


# LineOver1DRangeSeries.__doc__ = "ASD"