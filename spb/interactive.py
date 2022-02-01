from sympy.printing.latex import latex
from sympy.core.containers import Tuple
from sympy.external import import_module
from spb.series import (
    InteractiveSeries,
    _set_discretization_points,
)
from spb.ccomplex.complex import _build_series as _build_complex_series
from spb.vectors import _preprocess, _build_series as _build_vector_series
from spb.utils import _plot_sympify, _unpack_args
from spb.defaults import TWO_D_B, THREE_D_B, cfg
import warnings

param = import_module(
    'param',
    min_module_version='1.11.0')
pn = import_module(
    'panel',
    min_module_version='0.12.0')

pn.extension("plotly")


class MyList(param.ObjectSelector):
    """Represent a list of numbers discretizing a log-spaced slider.
    This parameter will be rendered by pn.widgets.DiscreteSlider
    """
    pass


# explicitely ask panel to use DiscreteSlider when it encounters a
# MyList object
pn.Param._mapping[MyList] = pn.widgets.DiscreteSlider


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
        _min, _max = values[1], values[2]
        if values[3] > 0:
            N = values[3]
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
            values[0],
            (values[1], values[2]),
            *values[3:]
        ]
        return {k: v for k, v in zip(defaults_keys, values)}

    def __init__(self, *args, name="", params=None, **kwargs):
        bokeh = import_module(
            'bokeh',
            import_kwargs={'fromlist': ['models']},
            min_module_version='2.3.0')
        TickFormatter = bokeh.models.formatters.TickFormatter

        # remove the previous class attributes added by the previous instances
        cls_name = type(self).__name__
        setattr(type(self), "_" + cls_name + "__params", dict())
        prev_params = [k for k in type(self).__dict__.keys() if "dyn_param_" in k]
        for p in prev_params:
            delattr(type(self), p)

        # use latex on control labels and legends
        self._use_latex = kwargs.pop("use_latex", True)

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

            # TODO: using a private method: not the smartest thing to do
            self.param._add_parameter("dyn_param_{}".format(i), v)
            self.mapping[k] = "dyn_param_{}".format(i)

    def read_parameters(self):
        readout = dict()
        for k, v in self.mapping.items():
            readout[k] = getattr(self, v)
        return readout


def _new_class(cls, **kwargs):
    "Creates a new class which overrides parameter defaults."
    return type(type(cls).__name__, (cls,), kwargs)


class PanelLayout:
    """Mixin class to group together the layout functionalities related to
    the library panel.
    """

    def __init__(self, layout, ncols, throttled=False):
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
            widgets=widgets,
            default_layout=_new_class(pn.GridBox, ncols=ncols),
            show_name=False,
            sizing_mode="stretch_width",
        )

    def layout_controls(self):
        return self.controls

    @pn.depends("controls")
    def view(self):
        params = self.read_parameters()
        self._backend._update_interactive(params)
        # TODO:
        # 1. for some reason, panel is going to set width=0 if K3D-Jupyter.
        # Temporary workaround: create a Pane with a default width.
        # Long term solution: create a PR on panel to create a K3DPane so that
        # panel will automatically deal with K3D, in the same way it does with
        # Bokeh, Plotly, Matplotlib, ...
        # 2. If the following import statement was located at the beginning of
        # the file, there would be a circular import.
        from spb.backends.k3d import KB

        if isinstance(self._backend, KB):
            return pn.pane.Pane(self._backend.fig, width=800)
        else:
            return self.fig

    def show(self):
        if self._layout == "tb":
            return pn.Column(self.layout_controls, self.view)
        elif self._layout == "bb":
            return pn.Column(self.view, self.layout_controls)
        elif self._layout == "sbl":
            return pn.Row(self.layout_controls, self.view)
        elif self._layout == "sbr":
            return pn.Row(self.view, self.layout_controls)


class InteractivePlot(DynamicParam, PanelLayout):
    """Contains all the logic to create parametric-interactive plots."""

    # NOTE: why isn't Plot a parent class for InteractivePlot?
    # If that was the case, we would need to create multiple subclasses of
    # InteractivePlot, each one targeting a different backend.
    # Instead, we keep the backend (the actual plot) as an instance attribute.

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, name="", params=None, **kwargs):
        """
        Parameters
        ==========
            args : tuple
                The usual plot arguments
            name : str
                Unused parameter
            params : dict
                In the keys there will be the symbols, in the values there
                will be parameters to create the slider associated to
                a symbol.
            kwargs : dict
                Usual keyword arguments to be used by the backends and series.
        """

        layout = kwargs.pop("layout", "tb")
        ncols = kwargs.pop("ncols", 2)
        throttled = kwargs.pop("throttled", cfg["interactive"]["throttled"])
        use_latex = kwargs.pop("use_latex", cfg["interactive"]["use_latex"])

        args = list(map(_plot_sympify, args))
        super().__init__(*args, name=name, params=params, use_latex=use_latex)
        PanelLayout.__init__(self, layout, ncols, throttled)

        # create the series
        series = self._create_series(*args, **kwargs)
        is_3D = all([s.is_3D for s in series])
        # create the plot
        Backend = kwargs.pop("backend", THREE_D_B if is_3D else TWO_D_B)
        kwargs["is_iplot"] = True
        self._backend = Backend(*series, **kwargs)

    def _create_series(self, *args, **kwargs):
        # read the parameters to generate the initial numerical data for
        # the interactive series
        kwargs["params"] = self.read_parameters()
        kwargs = _set_discretization_points(kwargs, InteractiveSeries)
        _slice = kwargs.get("slice", None)
        is_complex = kwargs.get("is_complex", False)
        is_vector = kwargs.get("is_vector", False)
        series = []
        if is_complex:
            new_args = []
            for a in args:
                exprs, ranges, label = _unpack_args(
                    *a, matrices=False, fill_ranges=False
                )
                new_args.append(Tuple(exprs[0], *ranges, label, sympify=False))
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
                exprs, ranges, label = _unpack_args(
                    *a, matrices=True, fill_ranges=False
                )
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
    """
    Create interactive plots of symbolic expressions.

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

    layout : str, optional
        The layout for the controls/plot. Possible values:

        - `'tb'`: controls in the top bar.
        - `'bb'`: controls in the bottom bar.
        - `'sbl'`: controls in the left side bar.
        - `'sbr'`: controls in the right side bar.

        Default layout to `'tb'`. Note that side bar layouts may not
        work well with some backends, and with `MatplotlibBackend` the widgets
        are always going to be displayed below the figure.

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

    polar : boolean, optional
        Default to False. If True, generate a polar plot of a curve with
        radius `expr` as a function of the range

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
    3. custom number of steps in the slider.
    4. custom format of the value shown on the slider. This might be useful to
       correctly visualize very small or very big numbers.
    5. custom labeling of the parameter-sliders.

    .. jupyter-execute::

       from bokeh.models.formatters import PrintfTickFormatter
       formatter = PrintfTickFormatter(format="%.3f")
       kp, t, z, o = symbols("k_P, tau, zeta, omega")
       G = kp / (I**2 * t**2 * o**2 + 2 * z * t * o * I + 1)
       mod = lambda x: 20 * log(sqrt(re(x)**2 + im(x)**2), 10)
       iplot(
           (mod(G.subs(z, 0)), (o, 0.1, 100), "G(z=0)"),
           (mod(G.subs(z, 1)), (o, 0.1, 100), "G(z=1)"),
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

    .. code-block:: python

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
           show = False
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
       p3 = plot(sin(x)*cos(x), (x, -5, 5), backend=MB,
           adaptive=False, n=50,
           is_point=True, is_filled=True,
           line_kw=dict(marker="^"), show=False)
       p = p1 + p2 + p3
       p.show()

    Serves the interactive plot on a separate browser window. Note:

    1. only ``BokehBackend`` and ``PlotlyBackend`` are supported for this
       operation mode.
    2. the output of ``iplot`` is captures into a variable.
    3. `show=True` has been set in order for ``iplot`` to return a `panel`
       object.

    .. code-block:: python

       from spb.backends.bokeh import BB
       from bokeh.models.formatters import PrintfTickFormatter
       formatter = PrintfTickFormatter(format='%.4f')

       p1, p2, t, r, c = symbols("p1, p2, t, r, c")
       phi = - (r * t + p1 * sin(c * r * t) + p2 * sin(2 * c * r * t))
       phip = phi.diff(t)
       r1 = phip / (1 + phip)

       t = iplot(
           (r1, (t, 0, 2*pi)),
           params = {
               p1: (0.035, -0.035, 0.035, 50, formatter),
               p2: (0.005, -0.02, 0.02, 50, formatter),
               r: (2, 2, 5, 3),
               c: (3, 1, 5, 4)
           },
           polar = True,
           use_latex = False,
           backend = BB,
           aspect = "equal",
           n = 5000,
           layout = "sbl",
           ncols = 1,
           show = True
       )
       t.show()

    Notes
    =====

    1. This function is specifically designed to work within Jupyter Notebook.
       However, it is also possible to use it from a regular Python,
       interpreter but only with ``BokehBackend`` and ``PlotlyBackend``.
       In such cases, it must be called with ``iplot(..., backend=BB).show()``, which will create a server process loading the interactive plot on
       the browser.

    2. Some examples use an instance of ``PrintfTickFormatter`` to format the
       value shown by a slider. This class is exposed by Bokeh, but can be
       used by ``iplot`` with any backend. Refer to [#fn1]_ for more
       information about tick formatting.

    3. The duality of the keyword argument ``show``:

       * If True, the function returns a `panel` object that will be rendered
         on the output cell of a Jupyter Notebook.
       * If False, it returns an instance of `InteractivePlot`.

       Let's focus on the syntax ``t = iplot(..., show=True, backend=BB)``, as
       shown in the last example.
       Here, the variable `t` captures the `panel` object, thus nothing
       will be rendered on the output cell. We can use this variable to serve
       the interactive plot through a server process on a separate browser
       window, by calling ``t.show()``. In doing so, the overall interactive
       plot is not subjected to the width limitation of a classical Jupyter
       Notebook. It is possible to play with the following keyword arguments
       to further customize the look and take advantage of the full page:
       `size, ncols, layout`.
       As stated before, only ``BokehBackend`` and ``PlotlyBackend`` are
       supported in this mode of operation.

    4. Say we are creating two different interactive plots and capturing
       their output on two variables, using ``show=False``. For example,
       ``p1 = iplot(..., show=False)`` and ``p2 = iplot(..., show=False)``.
       Then, running ``p1.show()`` on the screen will result in an error.
       This is standard behaviour that can't be changed, as `panel's`
       parameters are class attributes that gets deleted each time a new
       instance is created.

    5. ``MatplotlibBackend`` can be used, but the resulting figure is just a
       PNG image without any interactive frame. Thus, data exploration is not
       great. Therefore, the use of ``PlotlyBackend`` or ``BokehBackend`` is
       encouraged.

    6. Once this module has been loaded, there could be problems with all
       other plotting functions when using ``BokehBackend``, namely the
       figure won't show up in the output cell. If that is the case, we need
       to turn off  automatic updates on panning by setting
       ``update_event=False`` in the function call.

    7. When ``BokehBackend`` is used:

       * the user-defined theme won't be applied.
       * rendering of gradient lines is slow.
       * color bars might not update their ranges.

    8. Once this module has been loaded and ``iplot`` has been executed, if
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

        1. an instance of `param.parameterized.Parameter`. Refer to [#fn2]_
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

    .. [#fn2] https://panel.holoviz.org/user_guide/Param.html

    """
    dp = DynamicParam(params=params, **kwargs)
    tmp_panel = pn.Param(dp)

    results = dict()
    for k, v in dp.mapping.items():
        results[k] = tmp_panel.widget(v)
        if dp.formatters[k] is not None:
            results[k].format = dp.formatters[k]
    return results
