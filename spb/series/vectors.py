import math
import numpy as np
import param
from numbers import Number
from inspect import signature
from spb.wegert import wegert
from spb.defaults import cfg
from spb.utils import (
    _get_free_symbols, unwrap, extract_solution, tf_to_control
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
    GridEvaluator,
    SliceVectorGridEvaluator,
    _GridEvaluationParameters,
    _correct_shape
)
from spb.series.base import BaseSeries, _CastToInteger
from spb.series.series_2d_3d import PlaneSeries, SurfaceOver2DRangeSeries


class VectorBase(_GridEvaluationParameters, BaseSeries):
    """Represent a vector field."""

    is_vector = True
    is_slice = False
    _allowed_keys = [
        "streamlines", "quiver_kw", "stream_kw", "normalize"]

    expr = param.Parameter()
    is_streamlines = param.Boolean(False, doc="""
        If True shows the streamlines, otherwise shows the vector field.""")
    normalize = param.Boolean(False, doc="""
        If True, draw arrows with the same length. Note that normalization
        is achieved at the backend side, which allows to get same length
        arrows, but colored with the actual magnitude.
        If normalization would be applied on the series get_data(), the
        coloring by magnitude would not be applicable at the backend.
        """)
    tx = param.Callable(doc="""
        Numerical transformation function to be applied to the data on the
        x-axis.""")
    ty = param.Callable(doc="""
        Numerical transformation function to be applied to the data on the
        y-axis.""")

    def __init__(self, exprs, ranges, label, **kwargs):
        kwargs.setdefault("use_cm", True)
        if kwargs.get("use_cm") is None:
            kwargs["use_cm"] = False
        if "streamlines" in kwargs:
            kwargs["is_streamlines"] = kwargs.pop("streamlines")
        super().__init__(**kwargs)
        self.expr = tuple([e if callable(e) else sympify(e) for e in exprs])
        self.ranges = list(ranges)
        self.evaluator = GridEvaluator(series=self)
        self._label_str = str(exprs) if label is None else label
        self._label_latex = latex(exprs) if label is None else label

        # if the expressions are lambda functions and no label has been
        # provided, then its better to do the following in order to avoid
        # suprises on the backend
        if any(callable(e) for e in self.expr):
            if self._label_str == str(self.expr):
                self._label_str = "Magnitude"

        # NOTE: when plotting vector fields it might be useful to repeat the
        # plot command switching between quivers and streamlines.
        # Usually, plotting libraries expose different functions for quivers
        # and streamlines, accepting different keyword arguments.
        # The choice to implement separates stream_kw and quiver_kw allows
        # this quick switch.
        rendering_kw = self._enforce_dict_on_rendering_kw(
            kwargs.get("rendering_kw", {}))
        other_kw = "stream_kw" if self.is_streamlines else "quiver_kw"
        self.rendering_kw = kwargs.get(other_kw, rendering_kw)
        self._post_init()

    def get_label(self, use_latex=False, wrapper="$%s$"):
        if use_latex:
            expr = self.expr
            if self._label_str != str(expr):
                return self._label_latex
            return self._get_wrapped_label(self._label_latex, wrapper)
        return self._label_str

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

        results = self.evaluator._evaluate()
        for i, r in enumerate(results):
            re_v, im_v = np.real(r), np.imag(r)
            re_v[np.invert(np.isclose(im_v, np.zeros_like(im_v)))] = np.nan
            results[i] = re_v

        return self._apply_transform(*results)

    def _apply_transform(self, *args):
        t = self._get_transform_helper()
        if self.is_2Dvector:
            x, y, u, v = args
            return (
                t(x, self.tx), t(y, self.ty),
                t(u, self.tx), t(v, self.ty)
            )
        else:
            x, y, z, u, v, w = args
            return (
                t(x, self.tx), t(y, self.ty), t(z, self.tz),
                t(u, self.tx), t(v, self.ty), t(w, self.tz)
            )

    def eval_color_func(self, *coords):
        color = self.evaluator.eval_color_func(*coords)
        if color is not None:
            return color

        nargs = arity(self.evaluator.color_func)
        if (
            (self.is_3Dvector and (nargs == 6))
            or (self.is_2Dvector and (nargs == 4))
        ):
            color = self.evaluator.color_func(*coords)
        else:
            _raise_color_func_error(self, nargs)
        return _correct_shape(color, coords[0])


class Vector2DSeries(VectorBase):
    """Represents a 2D vector field."""

    is_2Dvector = True
    # default number of discretization points
    _N = 25
    _allowed_keys = ["scalar"]
    rendering_kw = param.Dict(default={}, doc="""
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the quivers or
        streamlines.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib:

          - quivers: https://matplotlib.org/stable/api/quiver_api.html#module-matplotlib.quiver
          - streamlines: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.streamplot.html#matplotlib.axes.Axes.streamplot

        * Plotly:

          - 2D quivers: https://plotly.com/python/quiver-plots/
          - 2D streamlines: https://plotly.com/python/streamline-plots/
        """)
    xscale = param.Selector(
        default="linear", objects=["linear", "log"], doc="""
        Discretization strategy along the x-direction.
        Related parameters: ``n1``.""")
    yscale = param.Selector(
        default="linear", objects=["linear", "log"], doc="""
        Discretization strategy along the y-direction.
        Related parameters: ``n2``.""")
    n1 = _CastToInteger(default=100, doc="""
        Number of discretization points along the x-axis to be used in the
        evaluation. Related parameters: ``xscale``.""")
    n2 = _CastToInteger(default=100, doc="""
        Number of discretization points along the y-axis to be used in the
        evaluation. Related parameters: ``yscale``.""")


    def __init__(self, u, v, range1, range2, label="", **kwargs):
        # if "scalar" not in kwargs.keys():
        #     use_cm = False
        # elif (not kwargs["scalar"]) or (kwargs["scalar"] is None):
        #     use_cm = True
        # else:
        #     use_cm = False
        # kwargs.setdefault("use_cm", )
        super().__init__((u, v), (range1, range2), label, **kwargs)

        # self.use_cm = kwargs.get("use_cm", use_cm)

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
    rendering_kw = param.Dict(default={}, doc="""
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the quivers or
        streamlines.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib:

          - quivers: https://matplotlib.org/stable/api/quiver_api.html#module-matplotlib.quiver
          - streamlines: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.streamplot.html#matplotlib.axes.Axes.streamplot

        * Plotly:

          - 3D cones: https://plotly.com/python/cone-plot/
          - 3D streamlines: https://plotly.com/python/streamtube-plot/

        * K3D-Jupyter:

          - for quiver plots, the keys can be:

            * scale: a float number acting as a scale multiplier.
              Default to 1.
            * pivot: indicates the part of the arrow that is anchored to the
              X, Y, Z grid. It can be ``"tail", "mid", "middle", "tip"``.
              Defaul to ``"mid"``.
            * color: set a solid color by specifying an integer color, or
              a colormap by specifying one of k3d's colormaps.
              If this key is not provided, a default color or colormap is used,
              depending on the value of ``use_cm``.

          - for streamline plots, refers to k3d.line documentation.
        """)
    xscale = param.Selector(
        default="linear", objects=["linear", "log"], doc="""
        Discretization strategy along the x-direction.
        Related parameters: ``n1``.""")
    yscale = param.Selector(
        default="linear", objects=["linear", "log"], doc="""
        Discretization strategy along the y-direction.
        Related parameters: ``n2``.""")
    zscale = param.Selector(
        default="linear", objects=["linear", "log"], doc="""
        Discretization strategy along the z-direction.
        Related parameters: ``n3``.""")
    tz = param.Callable(doc="""
        Numerical transformation function to be applied to the data on the
        z-axis.""")
    n1 = _CastToInteger(default=100, doc="""
        Number of discretization points along the x-axis to be used in the
        evaluation. Related parameters: ``xscale``.""")
    n2 = _CastToInteger(default=100, doc="""
        Number of discretization points along the y-axis to be used in the
        evaluation. Related parameters: ``yscale``.""")
    n3 = _CastToInteger(default=100, doc="""
        Number of discretization points along the z-axis to be used in the
        evaluation. Related parameters: ``zscale``.""")



    def __init__(self, u, v, z, range1, range2, range3, label="", **kwargs):
        super().__init__((u, v, z), (range1, range2, range3), label, **kwargs)
        if self.is_streamlines and isinstance(self.color_func, Expr):
            raise TypeError(
                "Vector3DSeries with streamlines can't use "
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

    def __init__(
        self, slice_surf, u, v, w, range_x, range_y, range_z,
        label="", **kwargs
    ):
        plane_kwargs = kwargs.copy()
        plane_kwargs.pop("normalize", None)
        self.slice_surf_series = _build_slice_series(
            slice_surf, [range_x, range_y, range_z], **plane_kwargs)
        super().__init__(u, v, w, range_x, range_y, range_z, label, **kwargs)
        self.evaluator = SliceVectorGridEvaluator(series=self)

    def __str__(self):
        return "sliced " + super().__str__() + " at {}".format(
            self.slice_surf_series)


class ListTupleArray(param.ClassSelector):
    """Parameter accepting values of type List, Tuple or np.array, with
    an optionally specified length by setting the ``bounds`` value.
    """

    __slots__ = ['bounds']

    def __init__(self, is_instance=True, bounds=None, **params):
        self.bounds = bounds
        params["class_"] = (list, tuple, np.ndarray, Tuple)
        params["is_instance"] = is_instance
        super().__init__(**params)

    def _validate(self, val):
        super()._validate(val)
        self._validate_class_(val, self.class_, self.is_instance)
        self._validate_bounds(val, self.bounds)

    def _validate_bounds(self, val, bounds):
        "Checks that the list is of the right length and has the right contents."
        if bounds is None or (val is None and self.allow_None):
            return
        min_length, max_length = bounds
        l = len(val)
        class_name = self.__class__.__name__
        if min_length is not None and max_length is not None:
            if not (min_length <= l <= max_length):
                raise ValueError(
                    f"{class_name} parameter length must be between "
                    f"{min_length} and {max_length} (inclusive), not {l}."
                )
        elif min_length is not None:
            if not min_length <= l:
                raise ValueError(
                    f"{class_name} length must be at "
                    f"least {min_length}, not {l}."
                )
        elif max_length is not None:
            if not l <= max_length:
                raise ValueError(
                    f"{class_name} length must be at "
                    f"most {max_length}, not {l}."
                )

class Arrow2DSeries(BaseSeries):
    """Represent an arrow in a 2D space.
    """

    is_2Dvector = True
    _allowed_keys = ["normalize"]

    start = ListTupleArray(bounds=(2, 2), doc="""
        Coordinates of the start position, (x, y).""")
    direction = ListTupleArray(bounds=(2, 2), doc="""
        Componenents of the direction vector, (u, v).""")
    rendering_kw = param.Dict(default={}, doc="""
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the arrows.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib:
          https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html
        * Plotly:
          https://plotly.com/python/reference/layout/annotations/
        * Bokeh:
          https://docs.bokeh.org/en/latest/docs/reference/models/annotations.html#bokeh.models.Arrow
        """)
    tx = param.Callable(doc="""
        Numerical transformation function to be applied to the data on the
        x-axis.""")
    ty = param.Callable(doc="""
        Numerical transformation function to be applied to the data on the
        y-axis.""")

    def __init__(self, start, direction, label="", **kwargs):
        self._block_lambda_functions(start, direction)
        expr_in = lambda _list: [
            isinstance(t, Expr) and (not t.is_number) for t in _list
        ]
        any_expr = any(expr_in(start) + expr_in(direction))
        if any_expr:
            start = Tuple(*start)
            direction = Tuple(*direction)
        else:
            start = np.array(start, dtype=np.float64)
            direction = np.array(direction, dtype=np.float64)

        kwargs["start"] = start
        kwargs["direction"] = direction
        kwargs["expr"] = (start, direction)

        if not label:
            # label: (from) -> (to)
            kwargs["_label_str"] = (
                "({}) -> ({})".format(
                    ", ".join([str(t) for t in start]),
                    ", ".join([str(u + v) for u, v in zip(
                        start, direction)])
                )
            )
            kwargs["_label_latex"] = (
                r"\left({}\right) \rightarrow \left({}\right)".format(
                    ", ".join([latex(t) for t in start]),
                    ", ".join([latex(u + v) for u, v in zip(
                        start, direction)])
                )
            )
        else:
            kwargs["label"] = label

        super().__init__(**kwargs)

        if any_expr and (not self.params):
            raise ValueError(
                "Some or all elements of the provided coordinates "
                "are symbolic expressions, but the ``params`` dictionary "
                "was not provided: those elements can't be evaluated."
            )

        if not any(isinstance(t, np.ndarray) for t in [self.start, self.direction]):
            self._check_fs()

    def __str__(self):
        pre = "3D " if self.is_3D else "2D "
        cast = lambda t: t if isinstance(t, Expr) else float(t)
        start = tuple(cast(t) for t in self.start)
        end = tuple(cast(s + d) for s, d in zip(start, self.direction))
        return self._str_helper(
            pre + f"arrow from {start} to {end}"
        )

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
        return self._get_wrapped_label(self._label_latex, wrapper)

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        x1, y1, z1 [optional] : float
            Coordinates of the start position.
        x2, y2, z2 [optional] : float
            Coordinates of the end position.
        """
        np = import_module('numpy')
        start, direction = self.start, self.direction

        if not self.is_interactive:
            start, direction = [
                np.array(t, dtype=float) for t in [start, direction]
            ]
        else:
            start = np.array(
                [t.evalf(subs=self.params) for t in start], dtype=float)
            direction = np.array(
                [t.evalf(subs=self.params) for t in direction], dtype=float)

        end = start + direction
        return self._apply_transform(*start, *end)

    def _apply_transform(self, *args):
        t = self._get_transform_helper()
        if self.is_2Dvector:
            x1, y1, x2, y2 = args
            return (
                t(x1, self.tx), t(y1, self.ty),
                t(x2, self.tx), t(y2, self.ty)
            )
        else:
            x1, y1, z1, x2, y2, z2 = args
            return (
                t(x1, self.tx), t(y1, self.ty), t(z1, self.tz),
                t(x2, self.tx), t(y2, self.ty), t(z2, self.tz)
            )


class Arrow3DSeries(Arrow2DSeries):
    """Represent an arrow in a 3D space.
    """
    is_3D = True
    is_2Dvector = False
    is_3Dvector = True

    start = ListTupleArray(bounds=(3, 3), doc="""
        Coordinates of the start position, (x, y, z).""")
    direction = ListTupleArray(bounds=(3, 3), doc="""
        Componenents of the direction vector, (u, v, w).""")
    rendering_kw = param.Dict(default={}, doc="""
        A dictionary of keyword arguments to be passed to the renderers
        in order to further customize the appearance of the arrows.
        Here are some useful links for the supported plotting libraries:

        * Matplotlib:
          https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html
        * K3D-Jupyter:
          Look at the documentation of k3d.vectors.
        """)
    tz = param.Callable(doc="""
        Numerical transformation function to be applied to the data on the
        z-axis.""")

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        x, y, z : float
            Coordinates of the start position.
        u, v, w : float
            Coordinates of the end position.
        """
        return super().get_data()
