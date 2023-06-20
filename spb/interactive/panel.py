from spb.defaults import TWO_D_B, THREE_D_B, cfg
from spb.utils import _validate_kwargs
from spb.interactive import _tuple_to_dict, IPlot
from spb.interactive.bootstrap_spb import SymPyBootstrapTemplate
from sympy import latex, Tuple
from sympy.external import import_module
import warnings

param = import_module(
    'param',
    min_module_version='1.11.0',
    warn_not_installed=True)
pn = import_module(
    'panel',
    min_module_version='0.12.0',
    warn_not_installed=True)

pn.extension("plotly", sizing_mode="stretch_width")


class MyList(param.ObjectSelector):
    """Represent a list of numbers discretizing a log-spaced slider.
    This parameter will be rendered by pn.widgets.DiscreteSlider
    """
    pass


# explicitely ask panel to use DiscreteSlider when it encounters a
# MyList object
pn.Param.mapping[MyList] = pn.widgets.DiscreteSlider


class DynamicParam(param.Parameterized):
    """Dynamically add parameters based on the user-provided dictionary.
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

    def _tuple_to_dict_panel(self, k, v):
        np = import_module('numpy')

        d = _tuple_to_dict(k, v, self._use_latex, "$$%s$$")
        values = list(d.values())

        if values[-1] == "log":
            # In case of a logarithm slider, we need to instantiate the
            # custom parameter MyList.

            N = 40 if len(v) <=3 else int(v[3])
            _min, _max = values[1:3]
            # # divide the range in N steps evenly spaced in a log scale
            options = np.geomspace(_min, _max, N)
            # the provided default value may not be in the computed options.
            # If that's the case, I chose the closest value
            default = values[0]
            if default not in options:
                default = min(options, key=lambda x: abs(x - default))
            return MyList(default=default, objects=list(options), label=values[4])

        defaults_keys = ["default", "softbounds", "step", "label", "type"]
        values = [values[0], tuple(values[1:3]), *values[3:]]
        return {k: v for k, v in zip(defaults_keys, values)}

    def __init__(self, *args, name="", params=None, **kwargs):
        bokeh = import_module(
            'bokeh',
            import_kwargs={'fromlist': ['models']},
            min_module_version='2.3.0',
            warn_not_installed=True)
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
                if len(v) >= 5:
                    # remove tick_format, as it won't be used for the creation
                    # of the parameter. Its value has already been stored.
                    v = list(v)
                    v.pop(4)
                v = self._tuple_to_dict_panel(k, v)
                # at this stage, v could be a dictionary representing a number,
                # or a MyList parameter, representing a log slider
                if not isinstance(v, param.parameterized.Parameter):
                    v.pop("type", None)
                    v = param.Number(**v)

            param_name = "dyn_param_{}".format(i)
            self.param.add_parameter(param_name, v)
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
            self._backend.update_interactive(params)
            self._action_post_update()


def _new_class(cls, **kwargs):
    "Creates a new class which overrides parameter defaults."
    return type(type(cls).__name__, (cls,), kwargs)


class PanelLayout:
    """Mixin class to group together the layout functionalities related to
    the library panel.
    """

    def __init__(self, layout, ncols, throttled=False, servable=False,
        custom_css="", pane_kw=None, template=None):
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

            template : optional
                Specify the template to be used to build the interactive
                application. It can be one of the following options:

                * None: the default template will be used.
                * dictionary of keyword arguments to customize the default
                  template.
                * an instance of pn.template.base.BasicTemplate
                * a subclass of pn.template.base.BasicTemplate
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
        self._template = template

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
        widgets = {}
        for k, v in self.mapping.items():
            widgets[v] = {"type": type(tmp_panel.widget(v))}
            t = getattr(self.param, v)
            if isinstance(t, param.Number):
                if self.formatters[k] is not None:
                    widgets[v]["format"] = self.formatters[k]
        
        # turn on/off throttling according to:
        # https://panel.holoviz.org/how_to/performance/throttling.html
        pn.config.throttled = throttled

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
        # NOTE: If the following import statement was located at the
        # beginning of the file, there would be a circular import.
        from spb import PB
        if isinstance(self._backend, PB):
            # NOTE: while pn.pane.Plotly can receive an instance of go.Figure,
            # the update will be extremely slow, because after each trace
            # is updated it will trigger an update event on the javascript
            # side. Instead, by providing the following dictionary, first the
            # traces are updated, then the pane creates the figure.
            # TODO: can the backend be modified by adding data and layout
            # attributes, avoiding the creation of the figure? The figure could
            # be created inside the fig getter.
            self.pane = pn.pane.Plotly(self.fig.to_dict(), **kw)
        else:
            self.pane = pn.pane.panel(self.fig, **kw)

    def layout_controls(self):
        return self.controls

    def _action_post_update(self):
        # NOTE: If the following import statement was located at the
        # beginning of the file, there would be a circular import.
        from spb import KB, PB

        if not isinstance(self._backend, KB):
            # KB exhibits a strange behavior when executing the following
            # lines. For the moment, do not execute them with KB
            self.pane.param.trigger("object")
            # self.pane.object = self.fig
            if not isinstance(self._backend, PB):
                self.pane.object = self.fig
            else:
                # NOTE: sadly, there is a bug with Panel and Plotly: if the
                # user modifies the layout of the chart (for example zoom or
                # rotate the view), the next update will reset them.
                # https://github.com/holoviz/panel/issues/1801
                self.pane.object = self.fig.to_dict()

    def show(self):
        self._init_pane()

        if not self._servable:
            if self._layout == "tb":
                content = pn.Column(self.layout_controls, self.pane)
            elif self._layout == "bb":
                content = pn.Column(self.pane, self.layout_controls)
            elif self._layout == "sbl":
                content = pn.Row(
                    pn.Column(self.layout_controls),
                    pn.Column(self.pane), width_policy="max")
            elif self._layout == "sbr":
                content = pn.Row(
                    pn.Column(self.pane),
                    pn.Column(self.layout_controls))

            return content

        return self._create_template(True)

    def _create_template(self, show=False):
        """Instantiate a template, populate it and serves it.

        Parameters
        ==========

        show : boolean
            If True, the template will be served on a new browser window.
            Otherwise, just return the template: ``show=False`` is used
            by the documentation to visualize servable applications.
        """
        if not show:
            self._init_pane()

        # pn.theme was introduced with panel 1.0.0, before there was
        # pn.template.theme
        submodule = pn.theme if hasattr(pn, "theme") else pn.template.theme
        theme = submodule.DarkTheme
        if cfg["interactive"]["theme"] != "dark":
            theme = submodule.DefaultTheme
        default_template_kw = dict(title=self._name, theme=theme)

        if (self._template is None) or isinstance(self._template, dict):
            merge = self._backend.merge
            kw = self._template if isinstance(self._template, dict) else {}
            kw = merge(default_template_kw, kw)
            kw["sidebar_location"] = self._layout
            if len(self._name.strip()) == 0:
                kw.setdefault("show_header", False)
            template = SymPyBootstrapTemplate(**kw)
        elif isinstance(self._template, pn.template.base.BasicTemplate):
            template = self._template
        elif (isinstance(self._template, type) and
            issubclass(self._template, pn.template.base.BasicTemplate)):
            template = self._template(**default_template_kw)
        else:
            raise TypeError("`template` not recognized. It can either be a "
                "dictionary of keyword arguments to be passed to the default "
                "template, an instance of pn.template.base.BasicTemplate "
                "or a subclass of pn.template.base.BasicTemplate. Received: "
                "type(template) = %s" % type(self._template))

        template.main.append(self.pane)
        template.sidebar.append(self.layout_controls)

        if show:
            return template.servable().show()
        return template


class InteractivePlot(DynamicParam, PanelLayout, IPlot):

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *series, name="", params=None, **kwargs):
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
        custom_css = kwargs.pop("custom_css", "")
        template = kwargs.pop("template", None)

        self._name = name
        super().__init__(name=self._name, params=params, use_latex=use_latex)
        PanelLayout.__init__(self, layout, ncols, throttled, servable,
            custom_css, pane_kw, template)

        # assure that each series has the correct values associated
        # to parameters
        series = list(series)
        for s in series:
            s.params = self.read_parameters()

        is_3D = all([s.is_3D for s in series])
        Backend = kwargs.pop("backend", THREE_D_B if is_3D else TWO_D_B)
        kwargs["is_iplot"] = True
        kwargs["imodule"] = "panel"
        self._backend = Backend(*series, **kwargs)
        _validate_kwargs(self._backend, **original_kwargs)

    @property
    def pane_kw(self):
        """Return the keyword arguments used to customize the wrapper to the
        plot.
        """
        return self._pane_kw

    def _get_iplot_kw(self):
        return {
            "backend": type(self._backend),
            "layout": self._layout,
            "template": self._template,
            "ncols": self._ncols,
            "throttled": self._throttled,
            "use_latex": self._use_latex,
            "params": self._original_params,
            "pane_kw": self._pane_kw
        }


def iplot(*series, show=True, **kwargs):
    """Create an interactive application containing widgets and charts in order
    to study symbolic expressions, using Holoviz's Panel for the user interace.

    Note: this function is already integrated with many of the usual
    plotting functions: since their documentation is more specific, it is
    highly recommended to use those instead.

    However, the following documentation explains in details the main features
    exposed by the interactive module, which might not be included on the
    documentation of those other functions.

    Parameters
    ==========

    series : BaseSeries
        Instances of BaseSeries, representing the symbolic expression to be
        plotted.

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
                Specify the discretization spacing. Default to ``"linear"``,
                can be changed to ``"log"``.

        Note that the parameters cannot be linked together (ie, one parameter
        cannot depend on another one).

    layout : str, optional
        The layout for the controls/plot. Possible values:

        - ``'tb'``: controls in the top bar.
        - ``'bb'``: controls in the bottom bar.
        - ``'sbl'``: controls in the left side bar.
        - ``'sbr'``: controls in the right side bar.

        If ``servable=False`` (plot shown inside Jupyter Notebook), then
        the default value is ``'tb'``. If ``servable=True`` (plot shown on a
        new browser window) then the default value is ``'sbl'``.
        Note that side bar layouts may not work well with some backends.

    ncols : int, optional
        Number of columns to lay out the widgets. Default to 2.

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

    servable : bool, optional
        Default to False, which will show the interactive application on the
        output cell of a Jupyter Notebook. If True, the application will be
        served on a new browser window.

    show : bool, optional
        Default to True.
        If True, it will return an object that will be rendered on the
        output cell of a Jupyter Notebook. If False, it returns an instance
        of ``InteractivePlot``, which can later be be shown by calling the
        `show()` method.

    template : optional
        Specify the template to be used to build the interactive application
        when ``servable=True``. It can be one of the following options:

        * None: the default template will be used.
        * dictionary of keyword arguments to customize the default template.
          Among the options:

          * ``full_width`` (boolean): use the full width of the browser page.
            Default to True.
          * ``sidebar_width`` (str): CSS value of the width of the sidebar
            in pixel or %. Applicable only when ``layout='sbl'`` or
            ``layout='sbr'``.
          * ``show_header`` (boolean): wheter to show the header of the
            application. Default to True.

        * an instance of ``pn.template.base.BasicTemplate``
        * a subclass of ``pn.template.base.BasicTemplate``

    throttled : boolean, optional
        Default to False. If True the recompute will be done at mouse-up event
        on sliders. If False, every slider tick will force a recompute.

    use_latex : bool, optional
        Default to True.
        If True, the latex representation of the symbols will be used in the
        labels of the parameter-controls. If False, the string
        representation will be used instead.


    Examples
    ========

    NOTE: the following examples use the ordinary plotting function because
    ``iplot`` is already integrated with them.

    Surface plot between -10 <= x, y <= 10 with a damping parameter varying
    from 0 to 1, with a default value of 0.15, discretized with 50 points
    on both directions.

    .. panel-screenshot::

       from sympy import (symbols, sqrt, cos, exp, sin, pi, re, im,
           Matrix, Plane, Polygon, I, log)
       from spb import *
       x, y, z = symbols("x, y, z")
       r = sqrt(x**2 + y**2)
       d = symbols('d')
       expr = 10 * cos(r) * exp(-r * d)
       plot3d(
           (expr, (x, -10, 10), (y, -10, 10)),
           params = { d: (0.15, 0, 1) },
           title = "My Title",
           xlabel = "x axis",
           ylabel = "y axis",
           zlabel = "z axis",
           backend = PB,
           n = 51,
           use_cm = True,
           use_latex=False,
           wireframe = True, wf_n1=15, wf_n2=15,
           wf_rendering_kw={"line_color": "#003428", "line_width": 0.75}
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

    .. panel-screenshot::

       from sympy import (symbols, sqrt, cos, exp, sin, pi, re, im,
           Matrix, Plane, Polygon, I, log)
       from spb import *
       from bokeh.models.formatters import PrintfTickFormatter
       formatter = PrintfTickFormatter(format="%.3f")
       kp, t, z, o = symbols("k_P, tau, zeta, omega")
       G = kp / (I**2 * t**2 * o**2 + 2 * z * t * o * I + 1)
       mod = lambda x: 20 * log(sqrt(re(x)**2 + im(x)**2), 10)
       plot(
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
           use_latex = False,
       )

    A line plot illustrating the Fouries series approximation of a saw tooth
    wave and:

    1. custom format of the value shown on the slider.
    2. creation of an integer spinner widget. This is achieved by setting
       ``None`` as one of the bounds of the integer parameter.

    .. panel-screenshot::

       from sympy import *
       from spb import *
       import param
       from bokeh.models.formatters import PrintfTickFormatter

       x, T, n, m = symbols("x, T, n, m")
       sawtooth = frac(x / T)
       # Fourier Series of a sawtooth wave
       fs = S(1) / 2 - (1 / pi) * Sum(sin(2 * n * pi * x / T) / n, (n, 1, m))

       formatter = PrintfTickFormatter(format="%.3f")
       plot(
           (sawtooth, (x, 0, 10), "f", {"line_dash": "dotted"}),
           (fs, (x, 0, 10), "approx"),
           params = {
               T: (4, 0, 10, 80, formatter),
               m: param.Integer(4, bounds=(1, None), label="Sum up to n ")
           },
           xlabel = "x",
           ylabel = "y",
           backend = BB,
           use_latex = False
       )

    A line plot with a parameter representing an angle in radians, but
    showing the value in degrees on its label:

    .. panel-screenshot::
       :small-size: 800, 570

       from sympy import sin, pi, symbols
       from spb import *
       from bokeh.models.formatters import FuncTickFormatter
       # Javascript code is passed to `code=`
       formatter = FuncTickFormatter(code="return (180./3.1415926 * tick).toFixed(2)")
       x, t = symbols("x, t")

       plot(
           (1 + x * sin(t), (x, -5, 5)),
           params = {
               t: (1, -2 * pi, 2 * pi, 100, formatter, "theta [deg]")
           },
           backend = MB,
           xlabel = "x", ylabel = "y",
           ylim = (-3, 4),
           use_latex = False,
       )

    Combine together ``InteractivePlot`` and ``Plot`` instances. The same
    parameters dictionary must be used for every interactive plot command.
    Note:

    1. the first plot dictates the labels, title and wheter to show the legend
       or not.
    2. Instances of ``Plot`` class must be place on the right side of the ``+``
       sign.
    3. ``show=False`` has been set in order for ``iplot`` to return an
       instance of ``InteractivePlot``, which supports addition.
    4. Once we are done playing with parameters, we can access the backend
       with ``p.backend``. Then, we can use the ``p.backend.fig`` attribute
       to retrieve the figure, or ``p.backend.save()`` to save the figure.

    .. panel-screenshot::
       :small-size: 800, 570

       from sympy import sin, cos, symbols
       from spb import *
       x, u = symbols("x, u")
       params = {
           u: (1, 0, 2)
       }
       p1 = plot(
           (cos(u * x), (x, -5, 5)),
           params = params,
           backend = MB,
           xlabel = "x1",
           ylabel = "y1",
           title = "title 1",
           legend = True,
           show = False,
           use_latex = False,
           imodule="panel"
       )
       p2 = plot(
           (sin(u * x), (x, -5, 5)),
           params = params,
           backend = MB,
           xlabel = "x2",
           ylabel = "y2",
           title = "title 2",
           show = False,
           imodule="panel"
       )
       p3 = plot(sin(x)*cos(x), (x, -5, 5), dict(marker="^"), backend=MB,
           adaptive=False, n=50,
           is_point=True, is_filled=True, show=False)
       p = p1 + p2 + p3
       p.show()

    Serves the interactive plot to a separate browser window. Note that
    ``K3DBackend`` is not supported for this operation mode. Also note the
    two ways to create a integer sliders.

    .. panel-screenshot::
       :small-size: 800, 500

       from sympy import *
       from spb import *
       import param
       from bokeh.models.formatters import PrintfTickFormatter
       formatter = PrintfTickFormatter(format='%.4f')

       p1, p2, t, r, c = symbols("p1, p2, t, r, c")
       phi = - (r * t + p1 * sin(c * r * t) + p2 * sin(2 * c * r * t))
       phip = phi.diff(t)
       r1 = phip / (1 + phip)

       plot_polar(
           (r1, (t, 0, 2*pi)),
           params = {
               p1: (0.035, -0.035, 0.035, 50, formatter),
               p2: (0.005, -0.02, 0.02, 50, formatter),
               # integer parameter created with param
               r: param.Integer(2, softbounds=(2, 5), label="r"),
               # integer parameter created with usual syntax
               c: (3, 1, 5, 4)
           },
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
       It is also possible to use it from a regular Python console,
       by executing: ``iplot(..., servable=True)``, which will create a server
       process loading the interactive plot on the browser.
       However, ``K3DBackend`` is not supported in this mode of operation.

    2. The interactive application consists of two main containers:

       * a pane containing the widgets.
       * a pane containing the chart. We can further customize this container
         by setting the ``pane_kw`` dictionary. Please, read its documentation
         to understand the available options.

    3. Some examples use an instance of ``PrintfTickFormatter`` to format the
       value shown by a slider. This class is exposed by Bokeh, but can be
       used in interactive plots with any backend. Refer to [#fn1]_ for more
       information about tick formatting.

    4. It has been observed that Dark Reader (or other night-mode-enabling
       browser extensions) might interfere with the correct behaviour of
       the output of  interactive plots. Please, consider adding ``localhost``
       to the exclusion list of such browser extensions.

    5. Say we are creating two different interactive plots and capturing
       their output on two variables, using ``show=False``. For example,
       ``p1 = plot(..., params={a:(...), b:(...), ...}, show=False)`` and
       ``p2 = plot(..., params={a:(...), b:(...), ...}, show=False)``.
       Then, running ``p1.show()`` on the screen will result in an error.
       This is standard behaviour that can't be changed, as `panel's`
       parameters are class attributes that gets deleted each time a new
       instance is created.

    6. ``MatplotlibBackend`` can be used, but the resulting figure is just a
       PNG image without any interactive frame. Thus, data exploration is not
       great. Therefore, the use of ``PlotlyBackend`` or ``BokehBackend`` is
       encouraged.

    7. When ``BokehBackend`` is used:

       * rendering of gradient lines is slow.
       * color bars might not update their ranges.

    8. Once this module has been loaded and executed, the safest procedure
       to restart Jupyter Notebook's kernel is the following:

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

    create_widgets

    """
    i = InteractivePlot(*series, **kwargs)
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

    .. code-block:: python

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
