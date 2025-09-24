import math
import param
from spb.wegert import wegert
from sympy import (
    latex, Tuple, symbols, sympify, Expr, lambdify, im, re
)
from sympy.external import import_module
import warnings
from spb.doc_utils.ipython import modify_parameterized_doc
from spb.series.evaluator import (
    _NMixin,
    ComplexGridEvaluator
)
from spb.series.base import (
    BaseSeries,
    _RangeTuple,
    _CastToInteger,
    _CastToFloat,
    _get_wrapper_for_expr,
    _check_misspelled_series_kwargs
)
from spb.series.series_2d_3d import (
    LineOver1DRangeSeries,
    Line2DBaseSeries,
    SurfaceBaseSeries,
    SurfaceOver2DRangeSeries,
    Parametric3DLineSeries
)


@modify_parameterized_doc()
class AbsArgLineSeries(LineOver1DRangeSeries):
    """
    Represents the absolute value of a complex function colored by its
    argument over a complex range (a + I*b, c + I * b).
    Note that the imaginary part of the start and end must be the same.

    Init signature: AbsArgLineSeries(expr, range_x, label, **kwargs)
    """
    _exclude_params_from_doc = ["color_func", "line_color", "steps"]

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
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')
        x, result = self.evaluator.evaluate()
        _re, _im = np.real(result), np.imag(result)
        _abs = np.sqrt(_re**2 + _im**2)
        _angle = np.arctan2(_im, _re)
        return x, _abs, _angle

    def get_data(self):
        """
        Return coordinates for plotting the line.

        Returns
        =======
        x : np.ndarray
            x-coordinates
        abs: np.ndarray
            Absolute value of the complex function.
        arg : np.ndarray
            Argument of the complex function.
        """
        return super().get_data()

    def _apply_transform(self, *args):
        t = self._get_transform_helper()
        x, y, a = args
        return t(x, self.tx), t(y, self.ty), t(a, self.tz)


@modify_parameterized_doc()
class ComplexPointSeries(Line2DBaseSeries):
    """
    Representation for a line in the complex plane consisting of
    list of points.
    """
    _exclude_params_from_doc = ["steps", "unwrap"]

    numbers = param.Parameter(doc="""
        Complex number, or a list of complex numbers.""")
    color_func = param.Callable(default=None, doc="""
        A color function to be applied to the numerical data. It can be:

        * None: no color function.
        * callable: a function accepting two arguments, the real and imaginary
          parts of the complex coordinates, and returning numerical data.
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

    expr = param.Parameter(doc="""
        The expression representing the complex function to be plotted.""")
    range_c = _RangeTuple(doc="""
        A 3-tuple `(symb, min, max)` denoting the range of the complex
        variable. Default values: `min=-10-10j` and `max=10+10j`.""")
    xscale = param.Selector(
        default="linear", objects=["linear", "log"], doc="""
        Discretization strategy along the x-direction (real part).
        Related parameters: ``n1``.""")
    yscale = param.Selector(
        default="linear", objects=["linear", "log"], doc="""
        Discretization strategy along the y-direction (imaginary part).
        Related parameters: ``n12``.""")
    n1 = _CastToInteger(default=300, bounds=(2, None), doc="""
        Number of discretization points along the x-axis (real part) to be
        used in the evaluation. Related parameters: ``xscale``.""")
    n2 = _CastToInteger(default=300, bounds=(2, None), doc="""
        Number of discretization points along the y-axis (imaginary part)
        to be used in the evaluation. Related parameters: ``yscale``.""")

    def __init__(self, expr, range_c, label="", **kwargs):
        _return = kwargs.pop("return", None)
        kwargs["expr"] = expr if callable(expr) else sympify(expr)
        kwargs["range_c"] = range_c
        kwargs["_range_names"] = ["range_c"]
        kwargs.setdefault("evaluator", ComplexGridEvaluator(series=self))
        super().__init__(**kwargs)
        self.evaluator.set_expressions()
        self._label_str = str(expr) if label is None else label
        self._label_latex = latex(expr) if label is None else label
        # determines what data to return on the z-axis
        self._return = _return

    @param.depends("n1", "n2", watch=True)
    def _update_discretized_domain(self):
        self.evaluator._create_discretized_domain()

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


@modify_parameterized_doc()
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

    color_func = param.Parameter(default=None, doc="""
        Define a custom color mapping to be used when ``use_cm=True``.
        It can either be:

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
    is_filled = param.Boolean(True, doc="""
        If True, used filled contours. Otherwise, use line contours.
        Relatated parameters: ``show_clabels``.""")
    show_clabels = param.Boolean(True, doc="""
        Toggle the label's visibility of contour lines. It only works when
        ``is_filled=False``. Note that some backend might not implement
        this feature. Relatated parameters: ``is_filled``.""")

    def __init__(self, expr, range_c, label="", **kwargs):
        threed = kwargs.pop("threed", False)
        kwargs.setdefault("is_filled", kwargs.pop("fill", True))
        kwargs.setdefault("show_clabels", kwargs.pop("clabels", True))
        super().__init__(expr, range_c, label, **kwargs)

        self._block_lambda_functions(self.expr)

        if not threed:
            self.is_contour = True
            self.is_3Dsurface = False
            self.use_cm = True

        self._post_init()

    def _create_discretized_domain(self):
        return ComplexSurfaceBaseSeries._create_discretized_domain(self)

    def get_data(self):
        """
        Return arrays of coordinates for plotting.

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

        domain, z = self.evaluator.evaluate(False)
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

    def _eval_color_func_helper(self, *coords):
        return SurfaceOver2DRangeSeries._eval_color_func_helper(
            self, *coords)


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
    phaseres = _CastToInteger(20, bounds=(1, 100), doc="""
        It controls the number of iso-phase and/or iso-modulus lines
        in domain coloring plots.""", label="Phase Resolution")
    cmap = param.ClassSelector(class_=(str, list, tuple), doc="""
        Specify the colormap to be used on enhanced domain coloring plots
        (both images and 3d plots). Default to ``"hsv"``. Can be any colormap
        from matplotlib or colorcet.""")
    blevel = _CastToFloat(0.75, bounds=(0, 1), doc="""
        Controls the black level of ehanced domain coloring plots.
        It must be `0 (black) <= blevel <= 1 (white)`.""",
        step=0.005, label="Black Level")
    phaseoffset = _CastToFloat(0, bounds=[0, 2*math.pi], doc="""
        Controls the phase offset of the colormap in domain coloring plots.""",
        step=0.05, label="Phase Offset [rad]")


@modify_parameterized_doc()
class ComplexDomainColoringSeries(
    ComplexSurfaceBaseSeries,
    ComplexDomainColoringBaseSeries
):
    """
    Represents a 2D/3D domain coloring plot of a complex function over
    the complex plane.
    """
    _exclude_params_from_doc = ["use_cm"]
    # NOTE: you might be tempted to include `riemann_mask` and `annotate`
    # in this list. Don't do it! It's a rabbit hole.
    # `domain_coloring` and `riemann_mask_2d` must be different from each
    # other. Better usability, better documentation, etc.
    _interactive_app_controls = [
        "n1", "n2", "coloring", "phaseres", "phaseoffset", "blevel"
    ]
    is_3Dsurface = False
    is_domain_coloring = True

    at_infinity = param.Boolean(False, doc="""
        If False the visualization will be centered about the complex point
        zero. Otherwise, it will be centered at infinity.""")
    annotate = param.Boolean(True, doc="""
        Turn on/off the annotations on the 2D projections of the Riemann
        sphere. They can only be visible when ``riemann_mask=True``.""")
    riemann_mask = param.Boolean(False, doc="""
        Turn on/off the unit disk mask representing the Riemann sphere
        on the 2D projections.""")
    rendering_kw = param.Dict(default={}, doc="""
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the line.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
        * Plotly: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Image.html
        * Bokeh: https://docs.bokeh.org/en/latest/docs/reference/models/glyphs/image_rgba.html#bokeh.models.ImageRGBA
        """)

    def __init__(self, expr, range_c, label="", **kwargs):
        threed = kwargs.pop("threed", False)
        if threed:
            kwargs.setdefault("use_cm", True)
        super().__init__(expr, range_c, label, **kwargs)

        if threed:
            self.is_3Dsurface = True

        # apply the transformation z -> 1/z in order to study the behavior
        # of the function at z=infinity
        if self.at_infinity:
            if callable(self.expr):
                raise ValueError(
                    "``at_infinity=True`` is only supported for symbolic "
                    "expressions. Instead, a callable was provided.")
            z = self.range_c[0]
            tmp = self.expr.subs(z, 1 / z)
            if self._label_str == str(self.expr):
                # adjust labels to prevent the wrong one to be seen on colorbar
                self._label_str = str(tmp)
                self._label_latex = latex(tmp)
            self.expr = tmp

        self._post_init()

    @param.depends("expr", watch=True)
    def _update_lambdifier(self):
        if self.evaluator is not None:
            self.evaluator.set_expressions()

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
        """
        Return arrays of coordinates for plotting.

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

        domain, z = self.evaluator.evaluate(False)
        return self._apply_transform(
            np.real(domain), np.imag(domain),
            np.absolute(z), np.angle(z),
            *self._domain_coloring(domain, z),
        )


@modify_parameterized_doc()
class ComplexParametric3DLineSeries(Parametric3DLineSeries):
    """
    Represent a mesh/wireframe line of a complex surface series.
    """

    # NOTE: why do I need this class? After all, I already have
    # Parametric3DLineSeries... Because I need to know what quantity to
    # return. This class is mainly used to draw wireframe lines of complex
    # surface series. A complex function can be plotted into the real or
    # imaginary components, or its absolute value or argument.
    def __init__(self, expr_x, expr_y, expr_z, range_p, label="", **kwargs):
        _return = kwargs.pop("return", None)
        super().__init__(expr_x, expr_y, expr_z, range_p, label, **kwargs)
        # determines what data to return on the z-axis
        self._return = _return

    def _uniform_sampling(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')

        results = self.evaluator.evaluate()
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


@modify_parameterized_doc()
class RiemannSphereSeries(
    ComplexDomainColoringBaseSeries,
    BaseSeries,
    _NMixin
):
    _exclude_params_from_doc = ["use_cm"]
    is_complex = True
    is_domain_coloring = True
    is_3Dsurface = True

    expr = param.Parameter(doc="""
        The expression representing the complex function to be plotted.""")
    range_t = _RangeTuple(doc="""
        A 3-tuple `(symb, min, max)` denoting the range of the polar angle
        theta, usually ranging from [0, pi/2] for half hemisphere.
        Default values: `min=-10` and `max=10`.""")
    range_p = _RangeTuple(doc="""
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
    n1 = _CastToInteger(default=150, bounds=(2, None), doc="""
        Number of discretization points along the polar angle to be used
        in the evaluation.""")
    n2 = _CastToInteger(default=600, bounds=(2, None), doc="""
        Number of discretization points along the azimuthal angle to be used
        in the evaluation.""")

    def __init__(self, expr, range_t, range_p, **kwargs):
        self._block_lambda_functions(expr)
        _check_misspelled_series_kwargs(self, **kwargs)
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
        """
        Return arrays of coordinates for plotting.

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
