import param
from spb.defaults import cfg
from spb.utils import _get_free_symbols, _check_misspelled_kwargs
from sympy import latex, Tuple, symbols, sympify, Expr, Symbol
from sympy.external import import_module
import warnings
from param.parameterized import Undefined
import typing


def _check_misspelled_series_kwargs(series, **kwargs):
    plot_function = kwargs.pop("plot_function", False)

    if plot_function:
        from spb.backends.base_backend import Plot
        plot_params = list(Plot.param) + [
            "show", "backend", "imodule", "threed", "process_piecewise",
            "animation", "servable", "template", "ncols", "layout",
            "markers", "rectangles", "annotations"
        ]
        for k in plot_params:
            kwargs.pop(k, None)

    _check_misspelled_kwargs(series, exclude_keys=["n"], **kwargs)


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


def _raise_color_func_error(series, nargs):
    if not isinstance(series, BaseSeries):
        return
    if series.color_func is None:
        return

    class_ = type(series).__name__
    raise ValueError(
        f"Error while processing the `color_func` of {class_}:"
        f" wrong number of arguments ({nargs}).\n"
        " Here is the documentation of the `color_func` attribute:\n\n"
        f"{series.param.color_func.doc}"
    )


class _ParametersDict(param.Dict):
    """As of `param 2.2.0, there is no mechanism to preprocess the value
    of an attribute just before it is set. This class allows to do just that.

    https://discourse.holoviz.org/t/what-is-the-best-way-to-make-custom-validation-and-transform-before-validation/3369
    """

    def __set__(self, obj, val):
        """Preprocess the parameters provided to the series.

        NOTE: at this point, if the data series is being instantiated,
        val is a dictionary with the following form:
            {
                symb1: (1, 0, 10, "label"),
                symb2: FloatSlider(value=2, min=0, max=5),
                (symb3, symb4): RangeSlider(...),
            }
        On the other hand, if the data series is being updated with new
        data from the widgets, val has this form:
            {
                symb1: val1,
                symb2: val2,
                (symb3, symb4): (val3, val4),
            }

        Here I unpack (symb3, symb4) so that self.param.keys()
        contains only symbols. This is what val is going to look
        after the preprocessing:
            {
                symb1: (1, 0, 10, "label"),
                symb2: FloatSlider(value=2, min=0, max=5),
                symb3: RangeSlider(...),
                symb4: RangeSlider(...),
            }
        Or if numeric values are provided:
            {
                symb1: val1,
                symb2: val2,
                symb3: val3,
                symb4: val4,
            }
        """
        # NOTE: Data series are unable to extract numerical values
        # from widgets. This step is done by iplot(). Before executing
        # the get_data() method, be sure to provide a ``params``
        # dictionary mapping symbols to numeric values.

        if any(isinstance(t, (list, tuple)) for t in val.keys()):
            new_params = {}
            for k, v in val.items():
                if isinstance(k, (list, tuple)):
                    # we are dealing with a multivalued widget
                    if isinstance(v, (list, tuple)):
                        # this is executed when params is updated with new
                        # numerical data from the widget
                        for symb, num in zip(k, v):
                            new_params[symb] = num
                    else:
                        # this is executed at data series instantiation
                        for symb in k:
                            new_params[symb] = v
                else:
                    new_params[k] = v
            val = new_params

        super().__set__( obj, val)


class _CastToInteger(param.Integer):
    """
    ``n1, n2, n3`` (number of discretization points) should be integer
    for np.linspace to work properly, but can receive float numbers.
    For example, 1e04.
    """

    def __set__(self, obj, val):
        super().__set__(obj, int(val))


class _RangeTuple(param.ClassSelector):
    """
    Represent a range for some variable. It must be a 3-elements tuple,
    `(symbol, min_val, max_val)`.
    """

    @typing.overload
    def __init__(
        self,
        default=None, *, is_instance=True,
        allow_None=False, doc=None, label=None, precedence=None, instantiate=True,
        constant=False, readonly=False, pickle_default_value=True, per_instance=True,
        allow_refs=False, nested_refs=False
    ):
        ...

    def __init__(self, default=Undefined, **params):
        super().__init__(default=default, class_=(tuple, Tuple), **params)

    def _validate(self, val):
        super()._validate(val)
        if val is not None:
            if len(val) != 3:
                raise ValueError(
                    f"Parameter '{self.name}' must be a 3-elements tuple."
                    f" Instead, {len(val)} elements were provided."
                )

    def __set__(self, obj, val):
        if (val is not None) and isinstance(val[0], str):
            val = (Symbol(val[0]), *val[1:])
        super().__set__(obj, sympify(val))


class BaseSeries(param.Parameterized):
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

    is_grid = False
    # Represents grids like s-grid, z-grid, n-grid, ...

    #####################
    # Instance Attributes
    #####################

    # NOTE: some data series should not be shown on the legend, for example
    # wireframe lines on 3D plots.
    show_in_legend = param.Boolean(True, doc="""
        Toggle the visibility of the data series on the legend.""")
    colorbar = param.Boolean(True, doc="""
        Toggle the visibility of the colorbar associated to the current data
        series. Note that a colorbar is only visible if ``use_cm=True`` and
        ``color_func`` is not None.""")
    use_cm = param.Boolean(False, doc="""
        Toggle the use of a colormap. By default, some series might use a
        colormap to display the necessary data. Setting this attribute
        to False will inform the associated renderer to use solid color.
        Related parameters: ``color_func``.""")
    # TODO: can I remove _label_str and only keep label?
    # NOTE: By default the data series stores two labels: one for the
    # string representation of the symbolic expression, the other for the
    # latex representation. The plotting library will then decide which one
    # is best to be shown. If the user set this parameter, both labels will
    # receive the same value. To retrieve one or the other representation,
    # call the ``get_label`` method of the data series.
    label = param.String("", doc="""
        Set the label associated to this series, which will be eventually
        shown on the legend or colorbar.""")
    rendering_kw = param.Dict(doc="""
        Keyword arguments to be passed to the renderers of the selected
        plotting library in order to further customize the appearance of this
        data series.""")
    # TODO: can the code be modified so that series.params ALWAYS returns
    # a dictionary mapping symbols to numerical values? This requires
    # the extraction of values during series instantiation.
    params = _ParametersDict({}, doc="""
        A dictionary mapping symbols to parameters. If provided, this
        dictionary enables the interactive-widgets plot.

        When calling a plotting function, the parameter can be specified with:

        * a widget from the ``ipywidgets`` module.
        * a widget from the ``panel`` module.
        * a tuple of the form:
           `(default, min, max, N, tick_format, label, spacing)`,
           which will instantiate a
           :py:class:`ipywidgets.widgets.widget_float.FloatSlider` or
           a :py:class:`ipywidgets.widgets.widget_float.FloatLogSlider`,
           depending on the spacing strategy. In particular:

           - default, min, max : float
                Default value, minimum value and maximum value of the slider,
                respectively. Must be finite numbers. The order of these 3
                numbers is not important: the module will figure it out
                which is what.
           - N : int, optional
                Number of steps of the slider.
           - tick_format : str or None, optional
                Provide a formatter for the tick value of the slider.
                Default to ``".2f"``.
           - label: str, optional
                Custom text associated to the slider.
           - spacing : str, optional
                Specify the discretization spacing. Default to ``"linear"``,
                can be changed to ``"log"``.

        Notes:

        1. parameters cannot be linked together (ie, one parameter
           cannot depend on another one).
        2. If a widget returns multiple numerical values (like
           :py:class:`panel.widgets.slider.RangeSlider` or
           :py:class:`ipywidgets.widgets.widget_float.FloatRangeSlider`),
           then a corresponding number of symbols must be provided.

        Here follows a couple of examples. If ``imodule="panel"``:

        .. code-block:: python

            import panel as pn
            params = {
                a: (1, 0, 5), # slider from 0 to 5, with default value of 1
                b: pn.widgets.FloatSlider(value=1, start=0, end=5), # same slider as above
                (c, d): pn.widgets.RangeSlider(value=(-1, 1), start=-3, end=3, step=0.1)
            }

        Or with ``imodule="ipywidgets"``:

        .. code-block:: python

            import ipywidgets as w
            params = {
                a: (1, 0, 5), # slider from 0 to 5, with default value of 1
                b: w.FloatSlider(value=1, min=0, max=5), # same slider as above
                (c, d): w.FloatRangeSlider(value=(-1, 1), min=-3, max=3, step=0.1)
            }

        When instantiating a data series directly, ``params`` must be a
        dictionary mapping symbols to numerical values.

        Let ``series`` be any data series. Then ``series.params`` returns a
        dictionary mapping symbols to numerical values.
        """)
    _label_str = param.String("", doc="""Contains str representation.""")
    _label_latex = param.String("", doc="""Contains latex representation.""")
    _is_interactive = param.Boolean(False, constant=True, doc="""
        Verify if this data series is interactive or not. Each data series
        expect one (or more) symbols to be specified as a discretization
        variable (ie, the ranges of the data series). However, the symbolic
        expressions may contain more symbols than what is expected by the
        ranges. In that case, the additional symbols are considered parameters,
        which will receive numerical values from interactive widgets.
        If this parameter is True, then the ``params`` attributes contains
        a non-empty dictionary.""")
    # TODO: I probably don't need this if I can better implement ``params``
    # in the first place. See TODO on ``params``.
    _original_params = param.Dict({}, doc="""
        This stores a copy of the ``params`` dictionary, just as it was
        provided by the user during a plotting function call. It is used
        by spb.interactive to keep track of multi-values widgets, which
        allows the mapping of symbols to the appropriate numerical values.""")
    _parametric_ranges = param.Boolean(False, doc="""
        Whether the series contains any parametric range, which is a range
        depending on symbols contained in ``params.keys()``.""")
    _range_names = param.List(default=[], item_type=str, doc="""
        List of parameter names refering to ranges. This parameter allows to
        quickly retrieve all ranges associated to a particular data series.""")

    def __repr__(self):
        if cfg["use_repr"] is False:
            return object.__repr__(self)
        return super().__repr__()

    def _enforce_dict_on_rendering_kw(self, rendering_kw):
        return {} if rendering_kw is None else rendering_kw

    @param.depends("_label_str", watch=True)
    def _update_label(self):
        # NOTE: this implements back-compatibility with sympy.plotting
        self.label = self._label_str

    @param.depends("label", watch=True)
    def _update_latex_and_str_labels(self):
        # this is triggered when someone changes the label after instantiating
        # the plot, like p[0].label = "something"
        self._label_latex = self.label
        self._label_str = self.label

    def __init__(self, *args, **kwargs):
        # allow the user to specify the number of discretization points
        # using different keyword arguments
        kwargs = _set_discretization_points(kwargs.copy(), type(self))

        # user (or plotting function) may still provide None to rendering_kw.
        # here we prevent this event from raising errors.
        # This helps to maintain back-compatibility with the graphics module.
        rendering_kw = kwargs.get("rendering_kw", None)
        if rendering_kw is None:
            kwargs["rendering_kw"] = {}

        # if user provides a label, overrides both the string and latex
        # representations
        label = kwargs.get("label", None)
        if label:
            kwargs["_label_str"] = kwargs["_label_latex"] = label

        _params = kwargs.setdefault("params", {})
        # this is used by spb.interactive to keep track of multi-values widgets
        kwargs.setdefault("_original_params", kwargs.get("params", {}))

        # remove keyword arguments that are not parameters of this series
        kwargs = {
            k: v for k, v in kwargs.items()
            if k in self._get_list_of_allowed_params()
        }

        super().__init__(*args, **kwargs)

        if len(_params) > 0:
            with param.edit_constant(self):
                self._is_interactive = True

            numbers_or_expressions = set().union(*[nv[1:] for nv in self.ranges])
            fs = set().union(*[e.free_symbols for e in numbers_or_expressions])
            if len(fs) > 0:
                self._parametric_ranges = True

    def _post_init(self):
        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        if any(callable(e) for e in exprs) and self.params:
            raise TypeError(
                "`params` was provided, hence an interactive plot "
                "is expected. However, interactive plots do not support "
                "user-provided numerical functions.")

        # if the expressions is a lambda function and no label has been
        # provided, then its better to do the following in order to avoid
        # suprises on the backend
        if any(callable(e) for e in exprs):
            if self._label_str == str(self.expr):
                self.label = ""

        self._check_fs()

    @classmethod
    def _get_list_of_allowed_params(cls):
        # also allows n1, n2, n3. they will be removed later on inside
        # _set_discretization_points
        return list(cls.param) + [
            "nb_of_points",
            "nb_of_points_x", "nb_of_points_y",
            "nb_of_points_u", "nb_of_points_v"
        ]

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
        if hasattr(self, "color_func"):
            fs = fs.union(_get_free_symbols(self.color_func))
        fs = fs.difference(params.keys())
        if ranges is not None:
            fs = fs.difference([r[0] for r in ranges])

        if len(fs) > 0:
            if (ranges is not None) and len(ranges) > 0:
                erl = f"Expressions: {exprs}\n"
                if (
                    hasattr(self, "color_func")
                    and isinstance(self.color_func, Expr)
                ):
                    erl += f"color_func: {self.color_func}\n"
                erl += f"Ranges: {ranges}\nLabel: {label}\n"
            else:
                erl = "Expressions: %s\nLabel: %s\n" % (exprs, label)
            raise ValueError(
                "Incompatible expression and parameters.\n%s"
                "params: %s\n"
                "Specify what these symbols represent: %s\n"
                "Are they ranges or parameters?" % (erl, params, fs)
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
            remaining_fs = fs.difference(params.keys())
            if len(remaining_fs) > 0:
                raise ValueError(
                    "Unkown symbols found in plotting range: %s. " % (r,) +
                    "Are the following parameters? %s" % remaining_fs)

    @property
    def is_3D(self):
        flags3D = [self.is_3Dline, self.is_3Dsurface, self.is_3Dvector]
        return any(flags3D)

    @property
    def is_line(self):
        flagslines = [self.is_2Dline, self.is_3Dline]
        return any(flagslines)

    @property
    def is_interactive(self):
        return self._is_interactive

    def _line_surface_color(self, prop, val):
        """This method enables back-compatibility with old sympy.plotting"""
        # NOTE: color_func is set inside the init method of the series.
        # If line_color/surface_color is not a callable, then color_func will
        # be set to None.
        prop = prop[1:] # remove underscore
        if callable(val) or isinstance(val, Expr):
            prop_val = None
            cf_val = val
        else:
            prop_val = val
            cf_val = None

        # prevents the triggering of events, which would cause recursion error
        with param.discard_events(self):
            setattr(self, prop, prop_val)
            if val is not None:
                # avoid resetting color_func when user writes line_color=None
                self.color_func = cf_val

    @property
    def scales(self):
        # get scale function
        gs = lambda k: getattr(self, k) if hasattr(self, k) else "linear"
        return [gs("xscale"), gs("yscale"), gs("zscale")]

    def eval_color_func(self, *args):
        """
        Evaluate the color function. Depending on the data series, either the
        data series itself or the backend will eventually execute this function
        to generate the appropriate coloring value.

        Parameters
        ----------
        args : tuple
            Arguments to be passed to the coloring function. Can be numerical
            coordinates or parameters or both. Read the documentation of each
            data series `color_func` attribute to find out what the
            arguments should be.

        Returns
        -------
        color : np.ndarray or float
            Results of the numerical evaluation of the ``color_func``
            attribute.
        """
        if hasattr(self, "evaluator") and (self.evaluator is not None):
            color = self.evaluator.eval_color_func(*args)
            if color is not None:
                return color
        if hasattr(self, "_eval_color_func_helper"):
            return self._eval_color_func_helper(*args)
        raise NotImplementedError

    def get_data(self):
        """Compute and returns the numerical data.

        The number of arrays returned by this method depends on the
        specific instance. Let ``s`` be an instance of ``BaseSeries``.
        Make sure to read ``help(s.get_data)`` to understand what it returns.
        """
        raise NotImplementedError

    def _get_wrapped_label(self, label, wrapper):
        """Given a latex representation of an expression, wrap it inside
        some characters. Matplotlib needs "$%s%$", K3D-Jupyter needs "%s".
        """
        return wrapper % label

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
        if self._label_str == str(self.expr):
            return self._get_wrapped_label(self._label_latex, wrapper)
        return self._label_latex

    @property
    def ranges(self):
        """
        Return a list of up to three 3-elements tuples, each one having the
        form (symbol, min, max), representing the ranges of numerical values
        used by each of the specified symbols.
        """
        return [getattr(self, k) for k in self._range_names]

    @ranges.setter
    def ranges(self, values):
        for k, v in zip(self._range_names, values):
            setattr(self, k, v)

    def _apply_transform(self, *args):
        """Apply transformations to the results of numerical evaluation.

        Parameters
        ==========
        args : tuple
            Results of the numerical evaluation.

        Returns
        =======
        transformed_args : tuple
            Tuple containing the transformed results.
        """
        raise NotImplementedError

    def _get_transform_helper(self):
        t = lambda x, transform: x if transform is None else transform(x)
        return t

    def _str_helper(self, s):
        pre, post = "", ""
        if self.is_interactive:
            pre = "interactive "
            post = " and parameters " + str(tuple(self.params.keys()))
        return pre + s + post


def _set_discretization_points(kwargs, Series):
    """Allow the use of the keyword arguments n, n1 and n2 (and n3) to
    specify the number of discretization points in two (or three) directions.

    Parameters
    ==========

    kwargs : dict

    Series : BaseSeries
        The type of the series, which indicates the kind of plot we are
        trying to create.

    Returns
    =======

    kwargs : dict
    """
    deprecated_keywords = {
        "nb_of_points": "n1",
        "nb_of_points_x": "n1",
        "nb_of_points_y": "n2",
        "nb_of_points_u": "n1",
        "nb_of_points_v": "n2",
        "points": "n"
    }
    for k, v in deprecated_keywords.items():
        if k in kwargs.keys():
            kwargs[v] = kwargs.pop(k)

    n = [None] * 3
    provided_n = kwargs.pop("n", None)
    if provided_n is not None:
        if hasattr(provided_n, "__iter__"):
            for i in range(min(len(provided_n), 3)):
                n[i] = int(provided_n[i])
        else:
            n = [int(provided_n)] * 3

    if n[0] is not None:
        kwargs.setdefault("n1", n[0])
    if n[1] is not None:
        kwargs.setdefault("n2", n[1])
    if n[2] is not None:
        kwargs.setdefault("n3", n[2])
    return kwargs

