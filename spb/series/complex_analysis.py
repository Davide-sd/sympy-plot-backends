import math
import numpy as np
import param
from numbers import Number
from inspect import signature
from spb.wegert import wegert
from spb.defaults import cfg
from spb.utils import (
    _get_free_symbols, unwrap, extract_solution, tf_to_control, prange
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
    _NMixin,
    ComplexGridEvaluator
)
from spb.series.base import (
    BaseSeries,
    _CastToInteger,
    _get_wrapper_for_expr
)
from spb.series.series_2d_3d import (
    LineOver1DRangeSeries,
    Line2DBaseSeries,
    SurfaceBaseSeries,
    SurfaceOver2DRangeSeries,
    Parametric3DLineSeries
)


class AbsArgLineSeries(LineOver1DRangeSeries):
    """
    Represents the absolute value of a complex function colored by its
    argument over a complex range (a + I*b, c + I * b).
    Note that the imaginary part of the start and end must be the same.
    """

    is_parametric = True
    is_complex = True
    tz = param.Callable(doc="""
        Numerical transformation function to be applied to the numerical values
        of the phase.""")

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_cm", True)
        super().__init__(*args, **kwargs)

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

    def _apply_transform(self, *args):
        t = self._get_transform_helper()
        x, y, a = args
        return t(x, self.tx), t(y, self.ty), t(a, self.tz)


class ComplexPointSeries(Line2DBaseSeries):
    """
    Representation for a line in the complex plane consisting of
    list of points.
    """

    numbers = param.Parameter(doc="""
        Complex numbers, or a list of complex numbers.""")
    color_func = param.Callable(default=None, doc="""
        A color function to be applied to the numerical data. It can be:

        * None: no color function.
        * callable: a function accepting two arguments (the real and imaginary
          parts of the complex coordinates) and returning numerical data.
        """)

    def __init__(self, numbers, label="", **kwargs):
        if isinstance(numbers, (list, tuple)):
            numbers = Tuple(*numbers)
        elif isinstance(numbers, Expr):
            numbers = Tuple(numbers)
        self._block_lambda_functions(*numbers)
        super().__init__(numbers=numbers, label=label, **kwargs)

        self.is_point = kwargs.get("scatter", kwargs.get("is_point", True))
        if self.use_cm and self.color_func:
            self.is_parametric = True

    @property
    def expr(self):
        return self.numbers

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')
        points = [complex(p.evalf(subs=self.params)) for p in self.numbers]
        points = np.array(points)
        r, i = np.real(points), np.imag(points)
        if self.use_cm and callable(self.color_func):
            return r, i, self.color_func(r, i)
        return r, i

    def __str__(self):
        return self._str_helper("complex points: %s" % self.numbers)


class ComplexSurfaceBaseSeries(SurfaceBaseSeries):
    """
    Represent a complex function to be shown on a 2D contour or 3D surface.
    """

    is_complex = True
    _N = 300

    expr = param.Parameter(doc="""
        The expression representing the complex function to be plotted.""")
    range_c = param.ClassSelector(class_=(tuple, Tuple, prange), doc="""
        A 3-tuple `(symb, min, max)` denoting the range of the complex
        variable. Default values: `min=-10-10j` and `max=10+10j`.""")
    is_filled = param.Boolean(True, doc="""
        If True, used filled contours. Otherwise, use line contours.""")
    show_clabels = param.Boolean(True, doc="""
        If True, used filled contours. Otherwise, use line contours.""")
    xscale = param.Selector(
        default="linear", objects=["linear", "log"], doc="""
        Discretization strategy along the x-direction (real part).
        Related parameters: ``n1``.""")
    yscale = param.Selector(
        default="linear", objects=["linear", "log"], doc="""
        Discretization strategy along the y-direction (imaginary part).
        Related parameters: ``n12``.""")
    n1 = _CastToInteger(default=100, doc="""
        Number of discretization points along the x-axis (real part) to be
        used in the evaluation. Related parameters: ``xscale``.""")
    n2 = _CastToInteger(default=100, doc="""
        Number of discretization points along the y-axis (imaginary part)
        to be used in the evaluation. Related parameters: ``yscale``.""")


    def __init__(self, expr, range_c, label="", **kwargs):
        _return = kwargs.pop("return", None)
        kwargs.setdefault("show_clabels", kwargs.pop("clabels", True))
        kwargs["expr"] = expr if callable(expr) else sympify(expr)
        kwargs["range_c"] = range_c
        super().__init__(**kwargs)
        self.ranges = [range_c]
        self.evaluator = ComplexGridEvaluator(series=self)
        self._label_str = str(expr) if label is None else label
        self._label_latex = latex(expr) if label is None else label
        # determines what data to return on the z-axis
        self._return = _return

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
            )
        )


class ComplexSurfaceSeries(
    ComplexSurfaceBaseSeries
):
    """
    Represents a 3D surface or contour plot of a complex function over
    the complex plane.
    """

    is_3Dsurface = True
    is_contour = False
    is_domain_coloring = False
    _allowed_keys = ["threed", "is_filled", "clabels"]

    # TODO: I have defined them in ComplexSurfaceBaseSeries.
    # Which one do I remove?
    is_filled = param.Boolean(True, doc="""
        If True, used filled contours. Otherwise, use line contours.""")

    show_clabels = param.Boolean(True, doc="""
        If True, used filled contours. Otherwise, use line contours.""")

    def __init__(self, expr, r, label="", **kwargs):
        threed = kwargs.pop("threed", False)
        kwargs.setdefault("is_filled", kwargs.pop("fill", True))
        super().__init__(expr, r, label, **kwargs)

        if isinstance(self, ComplexSurfaceSeries):
            self._block_lambda_functions(self.expr)

        if not threed:
            self.is_contour = True
            self.is_3Dsurface = False
            self.use_cm = True

        self._post_init()

    def _create_discretized_domain(self):
        return ComplexSurfaceBaseSeries._create_discretized_domain(self)

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

        domain, z = self.evaluator._evaluate(False)
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
            raise ValueError(
                "`_return` not recognized. Received: %s" % self._return)

        return self._apply_transform(np.real(domain), np.imag(domain), z)

    def eval_color_func(self, *coords):
        return SurfaceOver2DRangeSeries.eval_color_func(self, *coords)

class ComplexDomainColoringBaseSeries(param.Parameterized):
    coloring = param.Selector(
        default="a", objects=[
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
            "k", "k+log", "l", "m", "n", "o"
        ], doc="""
        Choose between different domain coloring options. Default to ``"a"``.
        Refer to [Wegert]_ for more information.

        - ``"a"``: standard domain coloring showing the argument of the
          complex function.
        - ``"b"``: enhanced domain coloring showing iso-modulus and iso-phase
          lines.
        - ``"c"``: enhanced domain coloring showing iso-modulus lines.
        - ``"d"``: enhanced domain coloring showing iso-phase lines.
        - ``"e"``: alternating black and white stripes corresponding to
          modulus.
        - ``"f"``: alternating black and white stripes corresponding to
          phase.
        - ``"g"``: alternating black and white stripes corresponding to
          real part.
        - ``"h"``: alternating black and white stripes corresponding to
          imaginary part.
        - ``"i"``: cartesian chessboard on the complex points space. The
          result will hide zeros.
        - ``"j"``: polar Chessboard on the complex points space. The result
          will show conformality.
        - ``"k"``: black and white magnitude of the complex function.
          Zeros are black, poles are white.
        - ``"k+log"``: same as ``"k"`` but apply a base 10 logarithm to the
          magnitude, which improves the visibility of zeros of functions with
          steep poles.
        - ``"l"``:enhanced domain coloring showing iso-modulus and iso-phase
          lines, blended with the magnitude: white regions indicates greater
          magnitudes. Can be used to distinguish poles from zeros.
        - ``"m"``: enhanced domain coloring showing iso-modulus lines, blended
          with the magnitude: white regions indicates greater magnitudes.
          Can be used to distinguish poles from zeros.
        - ``"n"``: enhanced domain coloring showing iso-phase lines, blended
          with the magnitude: white regions indicates greater magnitudes.
          Can be used to distinguish poles from zeros.
        - ``"o"``: enhanced domain coloring showing iso-phase lines, blended
          with the magnitude: white regions indicates greater magnitudes.
          Can be used to distinguish poles from zeros.

        The user can also provide a callable, ``f(w)``, where ``w`` is an
        [n x m] Numpy array (provided by the plotting module) containing
        the results (complex numbers) of the evaluation of the complex
        function. The callable should return:

        - img : ndarray [n x m x 3]
            An array of RGB colors (0 <= R,G,B <= 255)
        - colorscale : ndarray [N x 3] or None
            An array with N RGB colors, (0 <= R,G,B <= 255).
            If ``colorscale=None``, no color bar will be shown on the plot.
        """)

    phaseres = param.Integer(20, bounds=(1, 100), doc="""
        It controls the number of iso-phase and/or iso-modulus lines
        in domain coloring plots.""")

    cmap = param.ClassSelector(class_=(str, list, tuple), doc="""
        Specify the colormap to be used on enhanced domain coloring plots
        (both images and 3d plots). Default to ``"hsv"``. Can be any colormap
        from matplotlib or colorcet.""")

    blevel = param.Number(0.75, bounds=(0, 1), doc="""
        Controls the black level of ehanced domain coloring plots.
        It must be `0 (black) <= blevel <= 1 (white)`.""")

    phaseoffset = param.Number(0, bounds=[0, 2*math.pi], doc="""
        Controls the phase offset of the colormap in domain coloring plots.""")


class ComplexDomainColoringSeries(
    ComplexSurfaceBaseSeries,
    ComplexDomainColoringBaseSeries
):
    """
    Represents a 2D/3D domain coloring plot of a complex function over
    the complex plane.
    """
    is_3Dsurface = False
    is_domain_coloring = True
    _allowed_keys = [
        "threed", "coloring", "phaseres", "cmap", "blevel", "phaseoffset",
        "colorbar", "at_infinity", "riemann_mask", "annotate"
    ]

    at_infinity = param.Boolean(False, doc="""
        If False the visualization will be centered about the complex point
        zero. Otherwise, it will be centered at infinity.""")

    annotate = param.Boolean(True, doc="""
        Turn on/off the annotations on the 2D projections of the Riemann
        sphere. Default to True (annotations are visible). They can only
        be visible when ``riemann_mask=True``.""")

    riemann_mask = param.Boolean(False, doc="""
        Turn on/off the unit disk mask representing the Riemann sphere
        on the 2D projections. Default to True (mask is active).""")

    def __init__(self, expr, r, label="", **kwargs):
        threed = kwargs.pop("threed", False)
        if threed:
            kwargs.setdefault("use_cm", True)
        super().__init__(expr, r, label, **kwargs)

        if threed:
            self.is_3Dsurface = True
        #     self.use_cm = kwargs.get("use_cm", True)

        # apply the transformation z -> 1/z in order to study the behavior
        # of the function at z=infinity
        # self.at_infinity = kwargs.get("at_infinity", False)
        if self.at_infinity:
            if callable(self.expr):
                raise ValueError(
                    "``at_infinity=True`` is only supported for symbolic "
                    "expressions. Instead, a callable was provided.")
            z = self.ranges[0][0]
            tmp = self.expr.subs(z, 1 / z)
            if self._label_str == str(self.expr):
                # adjust labels to prevent the wrong one to be seen on colorbar
                self._label_str = str(tmp)
                self._label_latex = latex(tmp)
            self.expr = tmp

        # # domain coloring mode
        # self._init_domain_coloring_kw(**kwargs)

        # self.annotate = kwargs.get("annotate", True)
        # self.riemann_mask = kwargs.get("riemann_mask", False)
        self._post_init()

    @param.depends("expr", watch=True)
    def _update_evaluator(self):
        if self.evaluator is not None:
            self.evaluator.set_expressions()

    # def _init_domain_coloring_kw(self, **kwargs):
    #     self.coloring = kwargs.get("coloring", "a")
    #     if isinstance(self.coloring, str):
    #         self.coloring = self.coloring.lower()
    #     elif not callable(self.coloring):
    #         raise TypeError(
    #             "`coloring` must be a character from 'a' to 'j' or "
    #             "a callable.")
    #     self.phaseres = kwargs.get("phaseres", 20)
    #     self.cmap = kwargs.get("cmap", None)
    #     self.blevel = float(kwargs.get("blevel", 0.75))
    #     if self.blevel < 0:
    #         warnings.warn(
    #             "It must be 0 <= blevel <= 1. Automatically "
    #             "setting blevel = 0.")
    #         self.blevel = 0
    #     if self.blevel > 1:
    #         warnings.warn(
    #             "It must be 0 <= blevel <= 1. Automatically "
    #             "setting blevel = 1.")
    #         self.blevel = 1
    #     self.phaseoffset = float(kwargs.get("phaseoffset", 0))

    def _create_discretized_domain(self):
        return ComplexSurfaceBaseSeries._create_discretized_domain(self)

    def _domain_coloring(self, domain, w):
        if isinstance(self.coloring, str):
            self.coloring = self.coloring.lower()
            return wegert(
                self.coloring, w, self.phaseres, self.cmap,
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

        domain, z = self.evaluator._evaluate(False)
        return self._apply_transform(
            np.real(domain), np.imag(domain),
            np.absolute(z), np.angle(z),
            *self._domain_coloring(domain, z),
        )


class ComplexParametric3DLineSeries(Parametric3DLineSeries):
    """
    Represent a mesh/wireframe line of a complex surface series.
    """

    def __init__(self, *args, **kwargs):
        _return = kwargs.pop("return", None)
        super().__init__(*args, **kwargs)
        # determines what data to return on the z-axis
        self._return = _return

    def _adaptive_sampling(self):
        raise NotImplementedError

    def _uniform_sampling(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')

        results = self.evaluator._evaluate()
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
            raise ValueError(
                "`_return` not recognized. Received: %s" % self._return)

        return [*results[1:], results[0]]


class RiemannSphereSeries(
    ComplexDomainColoringBaseSeries,
    BaseSeries,
    _NMixin
):
    is_complex = True
    is_domain_coloring = True
    is_3Dsurface = True
    _N = 150
    _allowed_keys = [
        "cmap", "coloring", "blevel", "phaseres", "phaseoffset"]

    # n = param.List([100, 100, 100], item_type=Number, bounds=(3, 3), doc="""
    #     Number of discretization points along the x, y, z directions,
    #     respectively. It can easily be set with ``n=number``, which will
    #     set ``number`` for each element of the list.
    #     For surface, contour, 2d vector field plots it can be set with
    #     ``n=[num1, num2]``. For 3D implicit plots it can be set with
    #     ``n=[num1, num2, num3]``.

    #     Alternatively, ``n1=num1, n2=num2, n3=num3`` can be indipendently
    #     set in order to modify the respective element of the ``n`` list.""")
    expr = param.Parameter(doc="""
        The expression representing the complex function to be plotted.""")
    range_t = param.ClassSelector(class_=(tuple, Tuple, prange), doc="""
        A 3-tuple `(symb, min, max)` denoting the range of the polar angle
        theta, usually ranging from [0, pi/2] for half hemisphere.
        Default values: `min=-10` and `max=10`.""")
    range_p = param.ClassSelector(class_=(tuple, Tuple, prange), doc="""
        A 3-tuple `(symb, min, max)` denoting the range of the azimuthal angle
        phi, usually ranging from [0, 2*pi].
        Default values: `min=-10` and `max=10`.""")
    tx = param.Callable(doc="""
        Numerical transformation function to be applied to the data on the
        x-axis.""")
    ty = param.Callable(doc="""
        Numerical transformation function to be applied to the data on the
        y-axis.""")
    tz = param.Callable(doc="""
        Numerical transformation function to be applied to the data on the
        z-axis.""")
    n1 = _CastToInteger(default=150, doc="""
        Number of discretization points along the polar angle to be used
        in the evaluation.""")
    n2 = _CastToInteger(default=600, doc="""
        Number of discretization points along the azimuthal angle to be used
        in the evaluation.""")


    def __init__(self, expr, range_t, range_p, **kwargs):
        self._block_lambda_functions(expr)
        kwargs["expr"] = expr
        kwargs["range_t"] = range_t
        kwargs["range_p"] = range_p
        kwargs["use_cm"] = True
        super().__init__(**kwargs)
        if len(expr.free_symbols) > 1:
            # NOTE: considering how computationally heavy this series is,
            # it is rather unuseful to allow interactive-widgets plot.
            raise ValueError(
                "Complex function can only have one free symbol. "
                "Received free symbols: %s" % f.free_symbols)
        # NOTE: we can easily create a sphere with a single data series.
        # However, K3DBackend is unable to properly visualize it, and it
        # would require a few hours of work to apply the necessary edits.
        # Instead, I'm going to create two sphere caps (Northen and Southern
        # hemispheres, respectively), hence the need for ranges :D
        if self.n1 == self.n2:
            self.n2 = 4 * self.n1

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
        ts, te = [float(t) for t in self.range_t[1:]]
        ps, pe = [float(t) for t in self.range_p[1:]]
        t, p = np.mgrid[ts:te:self.n1*1j, ps:pe:self.n2*1j]
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
        return self._apply_transform(X, Y, Z, np.angle(w), img, cs)

    def _apply_transform(self, *args):
        t = self._get_transform_helper()
        x, y, _abs, _arg, img, colors = args
        return (
            t(x, self.tx), t(y, self.ty), t(_abs, self.tz),
            _arg, img, colors
        )
