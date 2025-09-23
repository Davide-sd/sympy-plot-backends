import param
from sympy import (
    latex, Tuple, arity, symbols, sympify, Expr, Plane
)
from sympy.external import import_module
from spb.doc_utils.ipython import modify_parameterized_doc
from spb.series.evaluator import (
    GridEvaluator,
    SliceVectorGridEvaluator,
    _GridEvaluationParameters,
    _correct_shape
)
from spb.series.base import (
    BaseSeries,
    _RangeTuple,
    _CastToInteger,
    _check_misspelled_series_kwargs
)
from spb.series.series_2d_3d import PlaneSeries, SurfaceOver2DRangeSeries


class VectorBase(_GridEvaluationParameters, BaseSeries):
    """Represent a vector field."""

    is_vector = True
    is_slice = False

    _expr = param.Parameter(doc="""
        Holds a tuple of symbolic expressions representing the
        vector field.""")
    is_streamlines = param.Boolean(False, doc="""
        If True shows the streamlines, otherwise visualize the vector field
        with quivers.""")
    normalize = param.Boolean(False, doc="""
        If True, the vector field will be normalized, resulting in quivers
        having the same length. If ``use_cm=True``, the backend will color
        the quivers by the (pre-normalized) vector field's magnitude.
        Note: only quivers will be affected by this option.
        """)
    tx = param.Callable(doc="""
        Numerical transformation function to be applied to the data on the
        x-axis.""")
    ty = param.Callable(doc="""
        Numerical transformation function to be applied to the data on the
        y-axis.""")

    def __init__(self, exprs, ranges, label, **kwargs):
        _check_misspelled_series_kwargs(
            self, additional_keys=["scalar", "streamlines"], **kwargs)
        kwargs.setdefault("use_cm", True)
        if kwargs.get("use_cm") is None:
            kwargs["use_cm"] = False
        if "streamlines" in kwargs:
            kwargs["is_streamlines"] = kwargs.pop("streamlines")
        kwargs["_expr"] = exprs
        kwargs.setdefault("evaluator", GridEvaluator(series=self))
        super().__init__(**kwargs)
        self.evaluator.set_expressions()
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

    @property
    def expr(self):
        return self._expr

    def get_label(self, use_latex=False, wrapper="$%s$"):
        if use_latex:
            expr = self.expr
            if self._label_str != str(expr):
                return self._label_latex
            return self._get_wrapped_label(self._label_latex, wrapper)
        return self._label_str

    def get_data(self):
        """Return arrays of coordinates for plotting."""
        np = import_module('numpy')

        results = self.evaluator.evaluate()
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

    def _eval_color_func_helper(self, *coords):
        color_func = self.evaluator.request_color_func(self.modules)
        nargs = arity(color_func)
        if (
            (self.is_3Dvector and (nargs == 6))
            or (self.is_2Dvector and (nargs == 4))
        ):
            color = color_func(*coords)
        else:
            _raise_color_func_error(self, nargs)
        return _correct_shape(color, coords[0])


@modify_parameterized_doc()
class Vector2DSeries(VectorBase):
    """Represents a 2D vector field."""

    _interactive_app_controls = [
        "n1", "n2", "xscale", "yscale", "only_integers"
    ]
    is_2Dvector = True

    u = param.Parameter(doc="""
        The components of the vector field along the x-axis. It can be a:

        * A symbolic expression.
        * A numerical function of two variables.
        """)
    v = param.Parameter(doc="""
        The components of the vector field along the y-axis. It can be a:

        * A symbolic expression.
        * A numerical function of two variables.
        """)
    range_x = _RangeTuple(doc="""
        A 3-tuple `(symb, min, max)` denoting the range of the x variable.
        Default values: `min=-10` and `max=10`.""")
    range_y = _RangeTuple(doc="""
        A 3-tuple `(symb, min, max)` denoting the range of the y variable.
        Default values: `min=-10` and `max=10`.""")
    color_func = param.Parameter(default=None, doc="""
        Define the quiver/streamlines color mapping when ``use_cm=True``.
        It can either be:

        * A numerical function supporting vectorization. The arity must be:
          ``f(x, y, u, v)``. Further, ``scalar=False`` must be set in order
          to hide the contour plot so that a colormap is applied to
          quivers/streamlines.
        * A symbolic expression having at most as many free symbols as
          ``u, v``. This only works for quivers plot.
        * None: the default value, which will map colors according to the
          magnitude of the vector field.
        """)
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
    n1 = _CastToInteger(default=25, doc="""
        Number of discretization points along the x-axis to be used in the
        evaluation. Related parameters: ``xscale``.""")
    n2 = _CastToInteger(default=25, doc="""
        Number of discretization points along the y-axis to be used in the
        evaluation. Related parameters: ``yscale``.""")

    def __init__(self, u, v, range_x, range_y, label="", **kwargs):
        u = u if callable(u) else sympify(u)
        v = v if callable(v) else sympify(v)
        kwargs["u"] = u
        kwargs["v"] = v
        kwargs["range_x"] = range_x
        kwargs["range_y"] = range_y
        kwargs["_range_names"] = ["range_x", "range_y"]
        super().__init__((u, v), (range_x, range_y), label, **kwargs)

    @param.depends("n1", "n2", "xscale", "yscale", "only_integers", watch=True)
    def _update_discretized_domain(self):
        self.evaluator._create_discretized_domain()

    def get_data(self):
        """
        Return arrays of coordinates for plotting.

        Returns
        =======
        mesh_x : np.ndarray [n2 x n1]
            Discretized x-domain.
        mesh_y : np.ndarray [n2 x n1]
            Discretized y-domain.
        u : np.ndarray [n2 x n1]
            First component of the vector field.
        v : np.ndarray [n2 x n1]
            Second component of the vector field.
        """
        return super().get_data()

    def __str__(self):
        ranges = []
        f = lambda t: t if len(t.free_symbols) > 0 else float(t)
        for r in self.ranges:
            ranges.append((r[0], f(r[1]), f(r[2])))
        return self._str_helper(
            "2D vector series: [%s, %s] over %s, %s" % (
                *self.expr, *ranges))


@modify_parameterized_doc()
class Vector3DSeries(VectorBase):
    """Represents a 3D vector field."""

    _interactive_app_controls = [
        "n1", "n2", "n3", "xscale", "yscale", "zscale", "only_integers"
    ]
    is_3D = True
    is_3Dvector = True

    u = param.Parameter(doc="""
        The components of the vector field along the x-axis. It can be a:

        * A symbolic expression.
        * A numerical function of three variables.
        """)
    v = param.Parameter(doc="""
        The components of the vector field along the y-axis. It can be a:

        * A symbolic expression.
        * A numerical function of three variables.
        """)
    w = param.Parameter(doc="""
        The components of the vector field along the z-axis. It can be a:

        * A symbolic expression.
        * A numerical function of three variables.
        """)
    range_x = _RangeTuple(doc="""
        A 3-tuple `(symb, min, max)` denoting the range of the x variable.
        Default values: `min=-10` and `max=10`.""")
    range_y = _RangeTuple(doc="""
        A 3-tuple `(symb, min, max)` denoting the range of the y variable.
        Default values: `min=-10` and `max=10`.""")
    range_z = _RangeTuple(doc="""
        A 3-tuple `(symb, min, max)` denoting the range of the z variable.
        Default values: `min=-10` and `max=10`.""")
    color_func = param.Parameter(default=None, doc="""
        Define the quiver/streamlines color mapping when ``use_cm=True``.
        It can either be:

        * A numerical function supporting vectorization. The arity must be
          ``f(x, y, z, u, v, w)``.
        * A symbolic expression having at most as many free symbols as
          ``u, v, w``. This only works for quivers plot.
        * None: the default value, which will map colors according to the
          magnitude of the vector.
        """)
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
    n1 = _CastToInteger(default=10, doc="""
        Number of discretization points along the x-axis to be used in the
        evaluation. Related parameters: ``xscale``.""")
    n2 = _CastToInteger(default=10, doc="""
        Number of discretization points along the y-axis to be used in the
        evaluation. Related parameters: ``yscale``.""")
    n3 = _CastToInteger(default=10, doc="""
        Number of discretization points along the z-axis to be used in the
        evaluation. Related parameters: ``zscale``.""")

    def __init__(self, u, v, w, range_x, range_y, range_z, label="", **kwargs):
        u = u if callable(u) else sympify(u)
        v = v if callable(v) else sympify(v)
        w = w if callable(w) else sympify(w)
        kwargs["u"] = u
        kwargs["v"] = v
        kwargs["w"] = w
        kwargs["range_x"] = range_x
        kwargs["range_y"] = range_y
        kwargs["range_z"] = range_z
        kwargs["_range_names"] = ["range_x", "range_y", "range_z"]
        super().__init__((u, v, w), (range_x, range_y, range_z), label, **kwargs)
        if self.is_streamlines and isinstance(self.color_func, Expr):
            raise TypeError(
                "Vector3DSeries with streamlines can't use "
                "symbolic `color_func`.")

    @param.depends(
        "n1", "n2", "n3", "xscale", "yscale", "zscale", "only_integers",
        watch=True
    )
    def _update_discretized_domain(self):
        self.evaluator._create_discretized_domain()

    def get_data(self):
        """
        Return arrays of coordinates for plotting.

        Returns
        =======
        mesh_x : np.ndarray [n2 x n1]
            Discretized x-domain.
        mesh_y : np.ndarray [n2 x n1]
            Discretized y-domain.
        mesh_z : np.ndarray [n2 x n1]
            Discretized z-domain.
        u : np.ndarray [n2 x n1]
            First component of the vector field.
        v : np.ndarray [n2 x n1]
            Second component of the vector field.
        w : np.ndarray [n2 x n1]
            Third component of the vector field.
        """
        return super().get_data()

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
        int(kwargs.get("n1", Vector3DSeries.n1)),
        int(kwargs.get("n2", Vector3DSeries.n2)),
        int(kwargs.get("n3", Vector3DSeries.n3))]
    discr_symbols = [r[0] for r in ranges]
    idx = [discr_symbols.index(s) for s in [r[0] for r in new_ranges]]
    kwargs2 = kwargs.copy()
    kwargs2.pop("n3", None)
    kwargs2["n1"] = n[idx[0]]
    kwargs2["n2"] = n[idx[1]]

    return SurfaceOver2DRangeSeries(slice_surf, *new_ranges, **kwargs2)


@modify_parameterized_doc()
class SliceVector3DSeries(Vector3DSeries):
    """Represents a 3D vector field plotted over a slice. The slice can be
    a Plane or a surface.
    """
    is_slice = True
    slice_surf_series = param.ClassSelector(class_=BaseSeries, doc="""
        Represent the slice over which quivers will be plotted.
        The value of this parameter can be an instance of BaseSeries,
        a Plane or a symbolic expression. In the latter cases, it will
        be pre-processed in order to generate an appropriate data series.""")

    def __init__(
        self, slice_surf_series, u, v, w, range_x, range_y, range_z,
        label="", **kwargs
    ):
        slice_surf_kwargs = kwargs.copy()
        slice_surf_kwargs.pop("normalize", None)
        kwargs["slice_surf_series"] = _build_slice_series(
            slice_surf_series,
            [range_x, range_y, range_z],
            **slice_surf_kwargs)
        kwargs.setdefault("evaluator", SliceVectorGridEvaluator(series=self))
        super().__init__(
            u, v, w, range_x, range_y, range_z, label, **kwargs)

    # TODO: in order to implement the behaviour of `app=True`, more thinking
    # needs to be done in order to account which parameter between
    # n1, n2, n3 goes to slice_surf_series
    @param.depends("params",watch=True)
    def _update_discretized_domain(self):
        self.evaluator._update_discretized_domain()

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
        np = import_module("numpy")
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


@modify_parameterized_doc()
class Arrow2DSeries(BaseSeries):
    """Represent an arrow in a 2D space.
    """
    _exclude_params_from_doc = ["colorbar", "use_cm"]

    is_2Dvector = True

    _expr = param.Parameter()
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
        np = import_module("numpy")
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
        kwargs["_expr"] = (start, direction)

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

    @property
    def expr(self):
        return self._expr

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
        """
        Return arrays of coordinates for plotting.

        Returns
        =======
        x1, y1 : float
            Coordinates of the start position.
        x2, y2 : float
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


@modify_parameterized_doc()
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
        """
        Return arrays of coordinates for plotting.

        Returns
        =======
        x1, y1, z1 : float
            Coordinates of the start position.
        x2, y2, z2 : float
            Coordinates of the end position.
        """
        return super().get_data()
