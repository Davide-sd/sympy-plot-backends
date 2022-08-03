from spb.defaults import TWO_D_B, THREE_D_B, cfg
from spb.ccomplex.complex import _build_series as _build_complex_series
from spb.functions import _set_labels
from spb.series import InteractiveSeries, _set_discretization_points
from spb.vectors import _preprocess, _build_series as _build_vector_series
from spb.utils import _plot_sympify, _unpack_args_extended, _validate_kwargs
from sympy import latex, Tuple
from sympy.external import import_module
import warnings

param = import_module(
    'param',
    min_module_version='1.11.0')
pn = import_module(
    'panel',
    min_module_version='0.12.0')

pn.extension("plotly", sizing_mode="stretch_width")


class MyList(param.ObjectSelector):
    """Represent a list of numbers discretizing a log-spaced slider.
    This parameter will be rendered by pn.widgets.DiscreteSlider
    """
    pass


# explicitely ask panel to use DiscreteSlider when it encounters a
# MyList object
pn.Param.mapping[MyList] = pn.widgets.DiscreteSlider


# Define a few CSS rules that are going to overwrite the template's ones.
# They are only going to be used when the interactive application will be
# served to a new browser window.
_CUSTOM_CSS = """
#header {padding: 0}
.title {
    font-size: 1em;
    font-weight: bold;
    padding-left: 10px;
}
"""

_CUSTOM_CSS_NO_HEADER = """
#header {display: none}
"""


class DynamicParam(param.Parameterized):
    """Dynamically add parameters based on the user-provided dictionary.
    Also, generate the lambda functions to be evaluated at a later stage.
    """

    # NOTE: why DynamicParam is a child class of param.Parameterized?
    # param is a full-python library, doesn't depend on anything else.
    # In theory, by using a parameterized class it should be possible to
    # create an InteractivePlotGUI class targeting a specific GUI.
    # At this moment, InteractivePlot is built on top of 'panel', so it only
    # works inside a Jupyter Notebook. Maybe it's possible to use PyQt or Tk.

    # Each one of the dynamically added parameters (widgets) will execute a
    # function that modify this parameter, which in turns will trigger an
    # overall update.
    check_val = param.Integer(default=0)

    def _tuple_to_dict(self, k, v):
        """The user can provide a variable length tuple/list containing:

        (default, min, max, N [optional], tick_format [optional],
            label [optional], spacing [optional])

        where:
            default : float
                Default value of the slider
            min : float
                Minimum value of the slider.
            max : float
                Maximum value of the slider.
            N : int
                Number of increments in the slider.
                (start - end) / N represents the step increment. Default to 40.
                Set N=-1 to have unit step increments.
            tick_format : bokeh.models.formatters.TickFormatter or None
                Default to None. Provide a formatter for the tick value of the
                slider.
            label : str
                Label of the slider. Default to None. If None, the string or
                latex representation will be used. See use_latex for more
                information.
            spacing : str
                Discretization spacing. Can be "linear" or "log".
                Default to "linear".
        """
        np = import_module('numpy')

        if not hasattr(v, "__iter__"):
            raise TypeError(
                "Provide a tuple or list for the parameter {}".format(k))

        if len(v) >= 5:
            # remove tick_format, as it won't be used for the creation of the
            # parameter. Its value has already been stored.
            v = list(v)
            v.pop(4)

        N = 40
        defaults_keys = ["default", "softbounds", "step", "label", "type"]
        defaults_values = [
            1,
            0,
            2,
            N,
            "$%s$" % latex(k) if self._use_latex else str(k),
            "linear",
        ]
        values = defaults_values.copy()
        values[: len(v)] = v
        # set the step increment for the slider
        _min, _max = float(values[1]), float(values[2])
        if values[3] > 0:
            N = int(values[3])
            values[3] = (_max - _min) / N
        else:
            values[3] = 1

        if values[-1] == "log":
            # In case of a logarithm slider, we need to instantiate the
            # custom parameter MyList.

            # # divide the range in N steps evenly spaced in a log scale
            options = np.geomspace(_min, _max, N)
            # the provided default value may not be in the computed options.
            # If that's the case, I chose the closest value
            default = values[0]
            if default not in options:
                default = min(options, key=lambda x: abs(x - default))
            return MyList(default=default, objects=list(options), label=values[4])

        # combine _min, _max into softbounds tuple
        values = [
            float(values[0]),
            (_min, _max),
            *values[3:]
        ]
        return {k: v for k, v in zip(defaults_keys, values)}

    def __init__(self, *args, name="", params=None, **kwargs):
        bokeh = import_module(
            'bokeh',
            import_kwargs={'fromlist': ['models']},
            min_module_version='2.3.0')
        TickFormatter = bokeh.models.formatters.TickFormatter

        # use latex on control labels and legends
        self._use_latex = kwargs.pop("use_latex", True)
        self._name = name

        # remove the previous class attributes added by the previous instances
        cls_name = type(self).__name__
        prev_params = [k for k in type(self).__dict__.keys() if "dyn_param_" in k]
        for p in prev_params:
            delattr(type(self), p)

        # this must be present in order to assure correct behaviour
        super().__init__(name=name, **kwargs)
        if not params:
            raise ValueError("`params` must be provided.")
        self._original_params = params

        # The following dictionary will be used to create the appropriate
        # lambda function arguments:
        #    key: the provided symbol
        #    val: name of the associated parameter
        self.mapping = {}

        # NOTE: unfortunately, parameters from the param library do not
        # provide a keyword argument to set a formatter for the slider's tick
        # value. As a workaround, the user can provide a formatter for each
        # parameter, which will be stored in the following dictionary and
        # later used in the instantiation of the widgets.
        self.formatters = {}

        # create and attach the params to the class
        for i, (k, v) in enumerate(params.items()):
            # store the formatter
            formatter = None
            if isinstance(v, (list, tuple)) and (len(v) >= 5):
                if (v[4] is not None) and (not isinstance(v[4], TickFormatter)):
                    raise TypeError(
                        "To format the tick value of the widget associated " +
                        "to the symbol {}, an instance of ".format(k) +
                        "bokeh.models.formatters.TickFormatter is expected. " +
                        "Instead, an instance of {} was given.".format(
                            type(v[4])))
                formatter = v[4]
            self.formatters[k] = formatter

            if not isinstance(v, param.parameterized.Parameter):
                v = self._tuple_to_dict(k, v)
                # at this stage, v could be a dictionary representing a number,
                # or a MyList parameter, representing a log slider
                if not isinstance(v, param.parameterized.Parameter):
                    v.pop("type", None)
                    v = param.Number(**v)

            param_name = "dyn_param_{}".format(i)
            # TODO: using a private method: not the smartest thing to do
            self.param._add_parameter(param_name, v)
            self.param.watch(self._increment_val, param_name)
            self.mapping[k] = param_name

    def _increment_val(self, *depends):
        self.check_val += 1

    def read_parameters(self):
        # TODO: check if param is still available, otherwise raise error
        readout = dict()
        for k, v in self.mapping.items():
            readout[k] = getattr(self, v)
        return readout

    @param.depends("check_val", watch=True)
    def update(self):
        params = self.read_parameters()
        # NOTE: in case _backend is not an attribute, it means that this
        # class has been instantiated by create_widgets
        if hasattr(self, "_backend"):
            self._backend._update_interactive(params)
            self._action_post_update()


def _new_class(cls, **kwargs):
    "Creates a new class which overrides parameter defaults."
    return type(type(cls).__name__, (cls,), kwargs)


class PanelLayout:
    """Mixin class to group together the layout functionalities related to
    the library panel.
    """

    def __init__(self, layout, ncols, throttled=False, servable=False, custom_css="", pane_kw=None):
        """
        Parameters
        ==========
            layout : str
                The layout for the controls/plot. Possible values:
                    'tb': controls in the top bar.
                    'bb': controls in the bottom bar.
                    'sbl': controls in the left side bar.
                    'sbr': controls in the right side bar.
                Default layout to 'tb'.

            ncols : int
                Number of columns to lay out the widgets. Default to 2.

            throttled : boolean, optional
                Default to False. If True the recompute will be done at
                mouse-up event on sliders. If False, every slider tick will
                force a recompute.

            servable : boolean, optional
                Default to False. If True the application will be served on
                a new browser window and a template will be applied to it.

            custom_css : str, optional
                This functionality is not yet fully implemented, please don't
                use it.

            pane_kw : dict, optional
                A dictionary of keyword arguments that are going to be passed
                to the pane containing the chart.
        """
        # NOTE: More often than not, the numerical evaluation is going to be
        # resource-intensive. By default, panel's sliders will force a
        # recompute at every step change. As a consequence, the user
        # experience will be laggy. To solve this problem, the update must
        # be triggered on mouse-up event, which is set using throttled=True.
        #
        # https://panel.holoviz.org/reference/panes/Param.html#disabling-continuous-updates-for-slider-widgets

        layouts = ["tb", "bb", "sbl", "sbr"]
        layout = layout.lower()
        if layout not in layouts:
            warnings.warn(
                "`layout` must be one of the following: {}\n".format(layouts)
                + "Falling back to layout='tb'."
            )
            layout = "tb"
        self._layout = layout
        self._ncols = ncols
        self._throttled = throttled
        self._servable = servable
        self._custom_css = custom_css
        self._pane_kw = pane_kw

        # NOTE: here I create a temporary panel.Param object in order to
        # reuse the code from the pn.Param.widget method, which returns the
        # correct widget associated to a given parameter's type.
        # Alternatively, I would need to copy parts of that method in order
        # to deal with the different representations of param.Integer and
        # param.Number depending if the bounds are None values.
        # Note that I'm only interested in the widget type: panel is then
        # going to recreate the widgets and setting the proper throttled
        # value. This is definitely not an optimal procedure, as we are
        # creating the "same" widget two times, but it works :)
        tmp_panel = pn.Param(self)
        widgets = dict()
        for k, v in self.mapping.items():
            widgets[v] = {"type": type(tmp_panel.widget(v))}
            t = getattr(self.param, v)
            if isinstance(t, param.Number):
                widgets[v]["throttled"] = throttled
                if self.formatters[k] is not None:
                    widgets[v]["format"] = self.formatters[k]

        self.controls = pn.Param(
            self,
            parameters=list(self.mapping.values()),
            widgets=widgets,
            default_layout=_new_class(pn.GridBox, ncols=ncols),
            show_name=False,
            sizing_mode="stretch_width",
        )

    def _init_pane(self):
        """Here we wrap the figure exposed by the backend with a Pane, which
        allows to set useful properties.
        """
        # NOTE: If the following import statement was located at the
        # beginning of the file, there would be a circular import.
        from spb import KB, MB

        default_kw = dict()
        if isinstance(self._backend, KB):
            # TODO: for some reason, panel is going to set width=0
            # if K3D-Jupyter is used.
            # Temporary workaround: create a Pane with a default width.
            # Long term solution: create a PR on panel to create a K3DPane
            # so that panel will automatically deal with K3D, in the same
            # way it does with Bokeh, Plotly, Matplotlib, ...
            default_kw["width"] = 800
        elif isinstance(self._backend, MB):
            # since we are using Jupyter and interactivity, it is useful to
            # activate ipympl interactive frame, as well as setting a lower
            # dpi resolution of the matplotlib image
            default_kw["dpi"] = 96
            # NOTE: the following must be set to False in order for the
            # example outputs to become visible on Sphinx.
            default_kw["interactive"] = False

        merge = self._backend.merge
        kw = merge({}, default_kw, self._pane_kw)
        self.pane = pn.pane.Pane(self.fig, **kw)

    def layout_controls(self):
        return self.controls

    def _action_post_update(self):
        # NOTE: If the following import statement was located at the
        # beginning of the file, there would be a circular import.
        from spb import KB

        if not isinstance(self._backend, KB):
            # KB exhibits a strange behavior when executing the following
            # lines. For the moment, do not execute them with KB
            self.pane.param.trigger("object")
            self.pane.object = self.fig

    def show(self):
        self._init_pane()

        if self._layout == "tb":
            content = pn.Column(self.layout_controls, self.pane)
        elif self._layout == "bb":
            content = pn.Column(self.pane, self.layout_controls)
        elif self._layout == "sbl":
            content = pn.Row(pn.Column(self.layout_controls, css_classes=["iplot-controls"], width=250, sizing_mode="fixed"), pn.Column(self.pane), width_policy="max")
        elif self._layout == "sbr":
            content = pn.Row(pn.Column(self.pane), pn.Column(self.layout_controls, css_classes=["iplot-controls"]))

        if not self._servable:
            return content

        css = _CUSTOM_CSS + self._custom_css
        if len(self._name.strip()) == 0:
            css = _CUSTOM_CSS_NO_HEADER + self._custom_css


        # theme = pn.template.vanilla.VanillaDarkTheme if cfg["interactive"]["theme"] == "dark" else pn.template.vanilla.VanillaDefaultTheme
        # vanilla = pn.template.VanillaTemplate(title=self._name, theme=theme)
        # vanilla.main.append(content)
        # vanilla.config.raw_css.append(css)

        theme = pn.template.bootstrap.BootstrapDarkTheme if cfg["interactive"]["theme"] == "dark" else pn.template.bootstrap.BootstrapDefaultTheme
        vanilla = pn.template.BootstrapTemplate(title=self._name, theme=theme)
        vanilla.main.append(content)
        vanilla.config.raw_css.append(css)

        return vanilla.servable().show()


def create_series(*args, **kwargs):
    """
    Create interactive data series, ie. series whose numerical data depends
    on the expression and all its parameters.

    Typical usage examples are in the followings:

    - Create a single interactive series:
        `create_series([expr, range], **kwargs)`
    - Create multiple interactive series:
        `create_series([(expr1, range1), (expr2, range2)], **kwargs)`

    The correct series type is instantiated only if all ranges are specified.
    So, to create an interactive line series, one range must be specified.
    To create an interactive surface series, two ranges must be provided, and
    so on.

    Notes
    =====

    The keyword arguments to be provided depends on the interested data series
    to be created. For example, if we are trying to plot a line, then the
    same keyword argument associated to the ``plot()`` function can be used.
    Similarly, if we are trying to create vector-related interactive series,
    the same keyword arguments associated to ``plot_vector()`` can be used.
    And so on.

    However, interactive data series do not support adaptive algorithms.
    Hence, adaptive-related keyword arguments will not be used.

    Parameters
    ==========

    args : list/tuple

        A list or tuple of the form ``[(expr1, range1), ...]``, where:

        expr : Expr
            Expression (or expressions) representing the function to evaluate.

        range: (symbol, min, max)
            A 3-tuple (or multiple 3-tuple) denoting the range of the
            variable. For the function to work properly, all ranges must be
            provided.

    params : dict
        A dictionary mapping symbols to numerical values. If not specified,
        ``iplot`` should be provided instead.

    iplot : InteractivePlot, optional
        An existing instance of ``InteractivePlot`` from which the parameters
        will be extracted. If both ``params`` and ``iplot`` are provided, then
        ``iplot`` has the precedence.

    is_complex : boolean, optional
        Default to False. If True, it directs the internal algorithm to
        create all the necessary series to create a complex plot (for example,
        one for the real part, one for the imaginary part, ...).

    is_polar : boolean, optional
        Default to False. If True:

        * for a 2D line plot requests the backend to use a polar chart.
        * for a 3D surface (or contour) requests a polar discretization.
          In this case, the first range represents the radius, the second one
          represents the angle.

    is_vector : boolean, optional
        Default to False. If True, it directs the internal algorithm to
        create all the necessary series to create a vector plot (for example,
        plotting the magnitude of the vector field as a contour plot, ...).

    n1, n2, n3 : int, optional
        Number of discretization points in the 3 directions.

    n : int, optional
        Set the same number of discretization points on all directions.


    Returns
    =======

    s : list
        A list of interactive data series.


    Examples
    ========

    >>> from sympy import (symbols, pi, sin, cos, exp, Plane,
    ...     Matrix, gamma, I, sqrt, Abs)
    >>> from spb.interactive import create_series
    >>> u, v, x, y, z = symbols('u, v, x:z')

    2D line interactive series:

    >>> s = create_series((u * sqrt(x), (x, -5, 5)), params={u: 1})
    >>> print(len(s), type(s[0]))
    (1, spb.series.LineInteractiveSeries)

    2D parametric line interactive series:

    >>> s = create_series((u * cos(x), u * sin(x), (x, -5, 5)), params={u: 1})
    >>> len(s), type(s[0])
    (1, spb.series.Parametric2DLineInteractiveSeries)

    Multiple 2D lines interactive series:

    >>> s = create_series(
    ...     (u * sqrt(x), (x, -5, 5)),
    ...     (cos(u * x), (x, -3, 3)),
    ...     params={u: 1})
    >>> len(s), type(s[0]), type(s[1])
    (2, spb.series.LineInteractiveSeries, spb.series.LineInteractiveSeries)

    Surface interactive series:

    >>> s = create_series((cos(x**2 + u * y**2), (x, -5, 5), (y, -5, 5)),
    ...     params={u: 1}, threed=True)
    >>> len(s), type(s[0])
    (1, spb.series.SurfaceInteractiveSeries)

    Contour interactive series:

    >>> s = create_series((cos(x**2 + u * y**2), (x, -5, 5), (y, -5, 5)),
    ...     params={u: 1}, threed=False)
    >>> len(s), type(s[0])
    (1, spb.series.ContourInteractiveSeries)

    Interactive series of the absolute value of a complex function colored
    by its argument over a real domain. Note that we are passing
    ``is_complex=True``:

    >>> s = create_series((u * sqrt(x), (x, -5, 5)), params={u: 1},
    ...     is_complex=True)
    >>> len(s), type(s[0])
    (1, spb.series.AbsArgLineInteractiveSeries)

    Real and imaginary parts of a complex function over a real domain:

    >>> s = create_series((u * sqrt(x), (x, -5, 5)), params={u: 1},
    ...     is_complex=True, absarg=False, real=True, imag=True)
    >>> len(s), type(s[0]), type(s[1])
    (2, spb.series.LineInteractiveSeries, spb.series.LineInteractiveSeries)

    Complex domain coloring interactive series:

    >>> s = create_series((u * sqrt(x), (x, -5-5j, 5+5j)), params={u: 1},
    ...     is_complex=True)
    >>> len(s), type(s[0])

    Real and imaginary parts of a complex function over a complex domain:

    >>> s = create_series((u * sqrt(x), (x, -5-5j, 5+5j)), params={u: 1},
    ...     is_complex=True, threed=True, absarg=False, real=True, imag=True)
    >>> len(s), type(s[0]), type(s[1])
    (2, spb.series.ComplexSurfaceInteractiveSeries, spb.series.ComplexSurfaceInteractiveSeries)

    2D vector interactive series (only quivers):

    >>> from sympy.vector import CoordSys3D
    >>> N = CoordSys3D("N")
    >>> i, j, k = N.base_vectors()
    >>> x, y, z = N.base_scalars()
    >>> a, b, c = symbols("a:c")
    >>> v1 = -a * sin(y) * i + b * cos(x) * j
    >>> s = create_series((v1, (x, -5, 5), (y, -4, 4)),
    ...     params={a: 2, b: 3}, is_vector=False)
    >>> len(s), type(s[0])
    (1, spb.series.Vector2DInteractiveSeries)

    2D vector interactive series (contour + quivers):

    >>> s = create_series((v1, (x, -5, 5), (y, -4, 4)),
    ...     params={a: 2, b: 3}, is_vector=True)
    >>> len(s), type(s[0]), type(s[1])
    (2, spb.series.ContourInteractiveSeries, spb.series.Vector2DInteractiveSeries)

    Sliced 3D vector (single slice):

    >>> from sympy import Plane
    >>> v3 = a * z * i + b * y * j + c * x * k
    >>> s = create_series((v3, (x, -5, 5), (y, -4, 4), (z, -6, 6)),
    ...     params={a: 2, b: 3, c: 1}, slice=Plane((1, 2, 3), (1, 0, 0)))
    >>> len(s), type(s[0])
    (1, spb.series.SliceVector3DInteractiveSeries)

    Geometry interactive series:

    >>> from sympy import Circle
    >>> s = create_series((Circle((0, 0), u), ), params={u: 1})
    >>> len(s), type(s[0])
    (1, spb.series.GeometryInteractiveSeries)

    See also
    ========

    iplot, create_widgets

    """
    args = list(map(_plot_sympify, args))

    iplot_obj = kwargs.pop("iplot", None)
    if iplot_obj is not None:
        # read the parameters to generate the initial numerical data for
        # the interactive series
        kwargs["params"] = iplot_obj.read_parameters()

    kwargs = _set_discretization_points(kwargs, InteractiveSeries)
    _slice = kwargs.get("slice", None)
    is_complex = kwargs.get("is_complex", False)
    is_vector = kwargs.get("is_vector", False)
    series = []
    if is_complex:
        new_args = []
        for a in args:
            exprs, ranges, label, rkw = _unpack_args_extended(
                *a, matrices=False, fill_ranges=False
            )
            new_args.append(Tuple(exprs[0], *ranges, label, rkw, sympify=False))
        series = _build_complex_series(*new_args, interactive=True, **kwargs)
    elif is_vector:
        args = _preprocess(*args, matrices=False, fill_ranges=False)
        series = _build_vector_series(*args, interactive=True, **kwargs)
    else:
        for a in args:
            # with interactive-parametric plots, vectors could have more
            # free symbols than the number of dimensions. We set
            # fill_ranges=False in order to not fill ranges, otherwise
            # ranges will be created also for parameters. This means
            # the user must provided all the necessary ranges.
            exprs, ranges, label, rkw = _unpack_args_extended(
                *a, matrices=True, fill_ranges=False
            )
            kwargs["rendering_kw"] = rkw
            if isinstance(_slice, (tuple, list)):
                # Sliced 3D vector field: each slice creates a
                # unique series
                kwargs2 = kwargs.copy()
                kwargs2.pop("slice")
                for s in _slice:
                    kwargs2["slice"] = s
                    series.append(
                        InteractiveSeries(exprs, ranges, label, **kwargs2)
                    )
            else:
                series.append(InteractiveSeries(exprs, ranges, label, **kwargs))
    return series


class InteractivePlot(DynamicParam, PanelLayout):

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, name="", params=None, **kwargs):
        """
        Parameters
        ==========
            args : tuple
                The usual plot arguments
            name : str
                Name of the interactive application
            params : dict
                In the keys there will be the symbols, in the values there
                will be parameters to create the slider associated to
                a symbol.
            kwargs : dict
                Usual keyword arguments to be used by the backends and series.
        """
        original_kwargs = kwargs.copy()

        layout = kwargs.pop("layout", "tb")
        ncols = kwargs.pop("ncols", 2)
        throttled = kwargs.pop("throttled", cfg["interactive"]["throttled"])
        servable = kwargs.pop("servable", cfg["interactive"]["servable"])
        use_latex = kwargs.pop("use_latex", cfg["interactive"]["use_latex"])
        pane_kw = kwargs.pop("pane_kw", dict())
        # NOTE: do not document these arguments yet, they might change in the
        # future.
        custom_css = kwargs.pop("custom_css", "")

        self._name = name
        super().__init__(*args, name=self._name, params=params, use_latex=use_latex)
        PanelLayout.__init__(self, layout, ncols, throttled, servable, custom_css, pane_kw)

        # create the series and apply the global labels and rendering keywords
        labels = kwargs.pop("label", [])
        rendering_kw = kwargs.pop("rendering_kw", None)
        series = create_series(*args, iplot=self, **kwargs)
        _set_labels(series, labels, rendering_kw)

        if kwargs.get("is_complex", False):
            from spb.ccomplex.complex import _set_axis_labels
            _set_axis_labels(series, kwargs)
        is_3D = all([s.is_3D for s in series])
        # create the plot
        Backend = kwargs.pop("backend", THREE_D_B if is_3D else TWO_D_B)
        kwargs["is_iplot"] = True
        self._backend = Backend(*series, **kwargs)
        _validate_kwargs(self._backend, **original_kwargs)

    @property
    def fig(self):
        """Return the plot object"""
        return self._backend.fig

    @property
    def backend(self):
        """Return the backend"""
        return self._backend

    def save(self, *args, **kwargs):
        """Save the current figure.
        This is a wrapper to the backend's `save` function. Refer to the
        backend's documentation to learn more about arguments and keyword
        arguments.
        """
        self._backend.save(*args, **kwargs)

    def __add__(self, other):
        return self._do_sum(other)

    def __radd__(self, other):
        return other._do_sum(self)

    def _do_sum(self, other):
        """Differently from Plot.extend, this method creates a new plot object,
        which uses the series of both plots and merges the _kwargs dictionary
        of `self` with the one of `other`.
        """
        from spb.backends.base_backend import Plot
        mergedeep = import_module('mergedeep')
        merge = mergedeep.merge

        if not isinstance(other, (Plot, InteractivePlot)):
            raise TypeError(
                "Both sides of the `+` operator must be instances of the "
                "InteractivePlot or Plot class.\n"
                "Received: {} + {}".format(type(self), type(other)))

        series = self._backend.series
        if isinstance(other, Plot):
            series.extend(other.series)
        else:
            series.extend(other._backend.series)

        # check that the interactive series uses the same parameters
        symbols = []
        for s in series:
            if s.is_interactive:
                symbols.append(list(s.params.keys()))
        if not all(t == symbols[0] for t in symbols):
            raise ValueError(
                "The same parameters must be used when summing up multiple "
                "interactive plots.")

        backend_kw = self._backend._copy_kwargs()
        panel_kw = {
            "backend": type(self._backend),
            "layout": self._layout,
            "ncols": self._ncols,
            "throttled": self._throttled,
            "use_latex": self._use_latex,
            "params": self._original_params,
            "show": False
        }

        new_iplot = type(self)(**merge({}, backend_kw, panel_kw))
        new_iplot._backend.series.extend(series)
        return new_iplot


def iplot(*args, show=True, **kwargs):
    """Create an interactive application containing widgets and charts in order
    to study symbolic expressions.

    Note that this function is already integrated with many of the usual
    plotting functions, hence it is recommended to use those instead of
    ``iplot`` because their documentation is more specific.

    However, the following documentation explains in details the main features
    exposed by the interactive module, which might not be included on the
    documentation of those other functions.

    Parameters
    ==========

    args : tuples
        Each tuple represents an expression. Depending on the type of
        expression, the tuple should have the following forms:

        1. line:
           `(expr, range, label [optional])`
        2. parametric line:
           `(expr1, expr2, expr3 [optional], range, label [optional])`
        3. surface:
           `(expr, range1, range2, label [optional])`
        4. parametric surface:
           `(expr1, expr2, expr3, range1, range2, label [optional])`

        The label is always optional, whereas the ranges must always be
        specified. The ranges will create the discretized domain.

    params : dict
        A dictionary mapping the symbols to a parameter. The parameter can be:

        1. an instance of `param.parameterized.Parameter`.
        2. a tuple of the form:
           `(default, min, max, N, tick_format, label, spacing)`
           where:

           - default, min, max : float
                Default value, minimum value and maximum value of the slider,
                respectively. Must be finite numbers.
           - N : int, optional
                Number of steps of the slider.
           - tick_format : TickFormatter or None, optional
                Provide a formatter for the tick value of the slider. If None,
                `panel` will automatically apply a default formatter.
                Alternatively, an instance of
                `bokeh.models.formatters.TickFormatter` can be used.
                Default to None.
           - label: str, optional
                Custom text associated to the slider.
           - spacing : str, optional
                Specify the discretization spacing. Default to `"linear"`, can
                be changed to `"log"`.

        Note that the parameters cannot be linked together (ie, one parameter
        cannot depend on another one).

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. The value depends on the backend
        being used. Read that backend's documentation to find out the
        possible values.

    backend : Plot, optional
        The backend to be used to generate the plot. It must be a subclass of
        `spb.backends.base_backend.Plot`. If not provided, the module will
        use the default backend.

    color_func : callable, optional
        A numerical function which defines the 2D line color or the surface
        color:

        - For 2D plots, a function of two variables is needed: x, y (the
          points computed by the internal algorithm).
        - For 3D surfaces, a function of three variables is needed: x, y, z
          (the points computed by the internal algorithm).
        - For 3D parametric surfaces, a function of five variables is needed:
          x, y, z, u, v (the points computed by the internal algorithm and the
          parameters).

        For surface plots, the coloring is applied only if ``use_cm=True``.

    label : list/tuple, optional
        The labels to be shown in the legend. If not provided, the string
        representation of `expr` will be used. The number of labels must be
        equal to the number of series generated by the plotting function.

    layout : str, optional
        The layout for the controls/plot. Possible values:

        - `'tb'`: controls in the top bar.
        - `'bb'`: controls in the bottom bar.
        - `'sbl'`: controls in the left side bar.
        - `'sbr'`: controls in the right side bar.

        Default layout to `'tb'`. Note that side bar layouts may not
        work well with some backends.

    ncols : int, optional
        Number of columns to lay out the widgets. Default to 2.

    is_complex : boolean, optional
        Default to False. If True, it directs the internal algorithm to
        create all the necessary series to create a complex plot (for example,
        one for the real part, one for the imaginary part).

    is_vector : boolean, optional
        Default to False. If True, it directs the internal algorithm to
        create all the necessary series to create a vector plot (for example,
        plotting the magnitude of the vector field as a contour plot).

    name : str, optional
        The name to be shown on top of the interactive application, when
        served on a new browser window. Refer to ``servable`` to learn more.
        Default to an empty string.

    pane_kw : dict, optional
        A dictionary of keyword/values which is passed to the pane containing
        the chart in order to further customize the output (read the Notes
        section to understand how the interactive plot is built).
        The following web pages shows the available options:

        * Refer to [#fn2]_ for ``MatplotlibBackend``. Two interesting options
          are:

          * ``interactive``: wheter to activate the ipympl interactive backend.
          * ``dpi``: set the dots per inch of the output png. Default to 96.

        * Refer to [#fn3]_ for ``PlotlyBackend``.

    rendering_kw : dict or list of dicts, optional
        A dictionary of keywords/values which is passed to the backend's
        function to customize the appearance of lines, surfaces, etc.
        Refer to the plotting library (backend) manual for more informations.
        If a list of dictionaries is provided, the number of dictionaries must
        be equal to the number of series generated by the plotting function.

    servable : bool, optional
        Default to False, which will show the interactive application on the
        output cell of a Jupyter Notebook. If True, the application will be
        served on a new browser window.

    show : bool, optional
        Default to True.
        If True, it will return an object that will be rendered on the
        output cell of a Jupyter Notebook. If False, it returns an instance
        of `InteractivePlot`, which can later be be shown by calling the
        `show()` method.

    use_latex : bool, optional
        Default to True.
        If True, the latex representation of the symbols will be used in the
        labels of the parameter-controls. If False, the string
        representation will be used instead.

    detect_poles : boolean
        Chose whether to detect and correctly plot poles in line plots.
        Defaulto to `False`. To improve detection, increase the number of
        discretization points `n` and/or change the value of `eps`.

    eps : float
        An arbitrary small value used by the `detect_poles` algorithm.
        Default value to 0.1. Before changing this value, it is recommended to
        increase the number of discretization points.

    n1, n2, n3 : int, optional
        Set the number of discretization points in the three directions,
        respectively.

    n : int, optional
        Set the number of discretization points on all directions.
        It overrides `n1, n2, n3`.

    nc : int, optional
        Number of discretization points for the contour plot when
        `is_vector=True`.

    throttled : boolean, optional
        Default to False. If True the recompute will be done at mouse-up event
        on sliders. If False, every slider tick will force a recompute.

    title : str, optional
        Title of the plot.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    zlabel : str, optional
        Label for the z-axis.

    xlim : (float, float), optional
        Denotes the x-axis limits, `(min, max)`.

    ylim : (float, float), optional
        Denotes the y-axis limits, `(min, max)`.

    zlim : (float, float), optional
        Denotes the z-axis limits, `(min, max)`.


    Examples
    ========

    .. jupyter-execute::

       from sympy import (symbols, sqrt, cos, exp, sin, pi, re, im,
           Matrix, Plane, Polygon, I, log)
       from spb.interactive import iplot
       from spb.backends.matplotlib import MB
       from spb.backends.bokeh import BB
       from spb.backends.plotly import PB
       x, y, z = symbols("x, y, z")

    Surface plot between -10 <= x, y <= 10 with a damping parameter varying
    from 0 to 1, with a default value of 0.15, discretized with 100 points
    on both directions. Note the use of `threed=True` to specify a 3D plot.
    If `threed=False`, a contour plot will be generated.

    .. jupyter-execute::

       r = sqrt(x**2 + y**2)
       d = symbols('d')
       expr = 10 * cos(r) * exp(-r * d)
       iplot(
           (expr, (x, -10, 10), (y, -10, 10)),
           params = { d: (0.15, 0, 1) },
           title = "My Title",
           xlabel = "x axis",
           ylabel = "y axis",
           zlabel = "z axis",
           backend = PB,
           n = 50,
           threed = True
       )

    A line plot of the magnitude of a transfer function, illustrating the use
    of multiple expressions and:

    1. some expression may not use all the parameters.
    2. custom labeling of the expressions.
    3. custom rendering of the expressions.
    4. custom number of steps in the slider.
    5. custom format of the value shown on the slider. This might be useful to
       correctly visualize very small or very big numbers.
    6. custom labeling of the parameter-sliders.

    .. jupyter-execute::

       from bokeh.models.formatters import PrintfTickFormatter
       formatter = PrintfTickFormatter(format="%.3f")
       kp, t, z, o = symbols("k_P, tau, zeta, omega")
       G = kp / (I**2 * t**2 * o**2 + 2 * z * t * o * I + 1)
       mod = lambda x: 20 * log(sqrt(re(x)**2 + im(x)**2), 10)
       iplot(
           (mod(G.subs(z, 0)), (o, 0.1, 100), "G(z=0)", {"line_dash": "dotted"}),
           (mod(G.subs(z, 1)), (o, 0.1, 100), "G(z=1)", {"line_dash": "dotted"}),
           (mod(G), (o, 0.1, 100), "G"),
           params = {
               kp: (1, 0, 3),
               t: (1, 0, 3),
               z: (0.2, 0, 1, 200, formatter, "z")
           },
           backend = BB,
           n = 2000,
           xscale = "log",
           xlabel = "Frequency, omega, [rad/s]",
           ylabel = "Magnitude [dB]",
           use_latex = False
       )

    A line plot with a parameter representing an angle in radians, but
    showing the value in degrees on its label:

    .. jupyter-execute::

       from bokeh.models.formatters import FuncTickFormatter
       # Javascript code is passed to `code=`
       formatter = FuncTickFormatter(code="return (180./3.1415926 * tick).toFixed(2)")

       iplot(
           (1 + x * sin(t), (x, -5, 5)),
           params = {
               t: (0, -2 * pi, 2 * pi, 100, formatter, "theta [deg]")
           },
           backend = BB,
           ylim = (-3, 4),
           use_latex = False
       )

    A 3D slice-vector plot. Note: whenever we want to create parametric vector
    plots, we should set `is_vector=True`.

    .. jupyter-execute::

       a, b = symbols("a, b")
       iplot(
           (Matrix([z * a, y * b, x]), (x, -5, 5), (y, -5, 5), (z, -5, 5)),
           params = {
               a: (1, 0, 5),
               b: (1, 0, 5)
           },
           backend = PB,
           is_vector = True,
           n = 10,
           slice = Plane((0, 0, 0), (0, 1, 0)),
           quiver_kw = {"sizeref": 8},
           use_latex = False
       )

    A parametric complex domain coloring plot. Note: whenever we want to create
    parametric complex plots, we must set `is_complex=True`.

    .. jupyter-execute::

       iplot(
           ((z**2 + 1) / (x * (z**2 - 1)), (z, -4 - 2 * I, 4 + 2 * I)),
           params = {
               x: (1, -2, 2)
           },
           backend = MB,
           is_complex = True,
           coloring = "b",
           grid = False,
           use_latex = False
       )

    A parametric plot of a symbolic polygon. Note the use of `param` to create
    an integer slider.

    .. jupyter-execute::

       import param
       a, b, c, d = symbols('a:d')
       iplot(
           (Polygon((a, b), c, n=d), ),
           params = {
               a: (0, -2, 2),
               b: (0, -2, 2),
               c: (1, 0, 5),
               d: param.Integer(3, softbounds=(3, 10), label="n"),
           },
           backend = BB,
           is_filled = False,
           aspect = "equal",
           use_latex = False
       )

    Combine together `InteractivePlot` and ``Plot`` instances. The same
    parameters dictionary must be used for every ``iplot`` command. Note:

    1. the first plot dictates the labels, title and wheter to show the legend
       or not.
    2. Instances of ``Plot`` class must be place on the right side of the `+`
       sign.
    3. `show=False` has been set in order for ``iplot`` to return an instance
       of ``InteractivePlot``, which supports addition.
    4. Once we are done playing with parameters, we can access the backend
       with ``p.backend``. Then, we can use the ``p.backend.fig`` attribute
       to retrieve the figure, or ``p.backend.save()`` to save the figure.

    .. jupyter-execute::

       from spb.functions import plot
       u = symbols("u")
       params = {
           u: (1, 0, 2)
       }
       p1 = iplot(
           (cos(u * x), (x, -5, 5)),
           params = params,
           backend = MB,
           xlabel = "x1",
           ylabel = "y1",
           title = "title 1",
           legend = True,
           show = False,
           use_latex = False
       )
       p2 = iplot(
           (sin(u * x), (x, -5, 5)),
           params = params,
           backend = MB,
           xlabel = "x2",
           ylabel = "y2",
           title = "title 2",
           show = False
       )
       p3 = plot(sin(x)*cos(x), (x, -5, 5), dict(marker="^"), backend=MB,
           adaptive=False, n=50,
           is_point=True, is_filled=True, show=False)
       p = p1 + p2 + p3
       p.show()

    Serves the interactive plot to a separate browser window. Note that
    ``K3DBackend`` is not supported for this operation mode.

    .. code-block:: python

       from spb.backends.bokeh import BB
       from bokeh.models.formatters import PrintfTickFormatter
       formatter = PrintfTickFormatter(format='%.4f')

       p1, p2, t, r, c = symbols("p1, p2, t, r, c")
       phi = - (r * t + p1 * sin(c * r * t) + p2 * sin(2 * c * r * t))
       phip = phi.diff(t)
       r1 = phip / (1 + phip)

       iplot(
           (r1, (t, 0, 2*pi)),
           params = {
               p1: (0.035, -0.035, 0.035, 50, formatter),
               p2: (0.005, -0.02, 0.02, 50, formatter),
               r: (2, 2, 5, 3),  # another way to create an integer slider
               c: (3, 1, 5, 4)
           },
           is_polar = True,
           use_latex = False,
           backend = BB,
           aspect = "equal",
           n = 5000,
           layout = "sbl",
           ncols = 1,
           servable = True,
           name = "Non Circular Planetary Drive - Ring Profile"
       )

    Notes
    =====

    1. This function is specifically designed to work within Jupyter Notebook.
       However, it is also possible to use it from a regular Python console,
       by executing: ``iplot(..., servable=True)``, which will create a server
       process loading the interactive plot on the browser.
       However, ``K3DBackend`` is not supported in this mode of operation.

    2. The interactive application generated by ``iplot`` consists of two main
       containers:

       * a pane containing the widgets.
       * a pane containing the chart. We can further customize this container
         by setting the ``pane_kw`` dictionary. Please, read its documentation
         to understand the availabel options.

    3. Some examples use an instance of ``PrintfTickFormatter`` to format the
       value shown by a slider. This class is exposed by Bokeh, but can be
       used by ``iplot`` with any backend. Refer to [#fn1]_ for more
       information about tick formatting.

    4. It has been observed that Dark Reader (or other night-mode-enabling
       browser extensions) might interfere with the correct behaviour of
       the output of  ``iplot``. Please, consider adding ``localhost`` to the
       exclusion list of such browser extensions.

    5. Say we are creating two different interactive plots and capturing
       their output on two variables, using ``show=False``. For example,
       ``p1 = iplot(..., show=False)`` and ``p2 = iplot(..., show=False)``.
       Then, running ``p1.show()`` on the screen will result in an error.
       This is standard behaviour that can't be changed, as `panel's`
       parameters are class attributes that gets deleted each time a new
       instance is created.

    6. ``MatplotlibBackend`` can be used, but the resulting figure is just a
       PNG image without any interactive frame. Thus, data exploration is not
       great. Therefore, the use of ``PlotlyBackend`` or ``BokehBackend`` is
       encouraged.

    7. Once this module has been loaded, there could be problems with all
       other plotting functions when using ``BokehBackend``, namely the
       figure won't show up in the output cell. If that is the case, we need
       to turn off  automatic updates on panning by setting
       ``update_event=False`` in the function call.

    8. When ``BokehBackend`` is used:

       * the user-defined theme won't be applied.
       * rendering of gradient lines is slow.
       * color bars might not update their ranges.

    9. Once this module has been loaded and ``iplot`` has been executed, if
       the kernel of the notebook needs to be restarted, the safest procedure
       is the following:

       * save the current notebook.
       * close the notebook and Jupyter server.
       * restart Jupyter server and open the notebook.
       * reload the cells.

       Failing to follow this procedure might results in the notebook to
       become  unresponsive once the module has been reloaded, with several
       errors appearing on the output cell.


    References
    ==========

    .. [#fn1] https://docs.bokeh.org/en/latest/docs/user_guide/styling.html#tick-label-formats
    .. [#fn2] https://panel.holoviz.org/reference/panes/Matplotlib.html
    .. [#fn3] https://panel.holoviz.org/reference/panes/Plotly.html


    See also
    ========

    create_series, create_widgets

    """
    i = InteractivePlot(*args, **kwargs)
    if show:
        return i.show()
    return i


def create_widgets(params, **kwargs):
    """ Create panel's widgets starting from parameters.

    Parameters
    ==========

    params : dict
        A dictionary mapping the symbols to a parameter. The parameter can be:

        1. an instance of `param.parameterized.Parameter`. Refer to [#fn5]_
           for a list of available parameters.
        2. a tuple of the form:
           `(default, min, max, N, tick_format, label, spacing)`
           where:

           - default, min, max : float
                Default value, minimum value and maximum value of the slider,
                respectively. Must be finite numbers.
           - N : int, optional
                Number of steps of the slider.
           - tick_format : TickFormatter or None, optional
                Provide a formatter for the tick value of the slider. If None,
                `panel` will automatically apply a default formatter.
                Alternatively, an instance of
                `bokeh.models.formatters.TickFormatter` can be used.
                Default to None.
           - label: str, optional
                Custom text associated to the slider.
           - spacing : str, optional
                Specify the discretization spacing. Default to `"linear"`,
                can be changed to `"log"`.

        Note that the parameters cannot be linked together (ie, one parameter
        cannot depend on another one).

    use_latex : bool, optional
        Default to True.
        If True, the latex representation of the symbols will be used in the
        labels of the parameter-controls. If False, the string representation
        will be used instead.


    Returns
    =======

    widgets : dict
        A dictionary mapping the symbols from `params` to the appropriate
        widget.


    Examples
    ========

    .. jupyter-execute::

       from sympy.abc import x, y, z
       from spb.interactive import create_widgets
       import param
       from bokeh.models.formatters import PrintfTickFormatter
       formatter = PrintfTickFormatter(format="%.4f")
       r = create_widgets({
           x: (0.035, -0.035, 0.035, 100, formatter),
           y: (200, 1, 1000, 10, None, "test", "log"),
           z: param.Integer(3, softbounds=(3, 10), label="n")
       })


    References
    ==========

    .. [#fn5] https://panel.holoviz.org/user_guide/Param.html


    See also
    ========

    iplot, create_series

    """
    dp = DynamicParam(params=params, **kwargs)
    tmp_panel = pn.Param(dp)

    results = dict()
    for k, v in dp.mapping.items():
        results[k] = tmp_panel.widget(v)
        if dp.formatters[k] is not None:
            results[k].format = dp.formatters[k]
    return results
