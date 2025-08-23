"""
Implements interactive-widgets plotting with Holoviz Panel using
`pn.bind`, which binds a function or method to the values of widgets.
"""

from spb.defaults import TWO_D_B, THREE_D_B, cfg
from spb.utils import _check_misspelled_kwargs
from spb.interactive import _tuple_to_dict, IPlot
from spb.interactive.bootstrap_spb import SymPyBootstrapTemplate
from spb.plotgrid import PlotGrid
from spb.utils import _aggregate_parameters
from sympy import latex
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

pn.extension("mathjax", "plotly", sizing_mode="stretch_width")


def _dict_to_slider(d):
    if d["type"] == "linear":
        return pn.widgets.FloatSlider(
            start=d["min"], end=d["max"], value=d["value"],
            step=d["step"], name=d["description"], format=d["formatter"]
        )
    else:
        np = import_module("numpy")
        _min, _max, step, value = d["min"], d["max"], d["step"], d["value"]
        N = int((_max - _min) / step)
        # divide the range in N steps evenly spaced in a log scale
        options = np.geomspace(_min, _max, N)
        # the provided default value may not be in the computed options.
        # If that's the case, I chose the closest value
        if value not in options:
            value = min(options, key=lambda x: abs(x - value))

        kwargs = dict(
            options=options.tolist(), value=value, name=d["description"])
        if d["formatter"]:
            kwargs["formatter"] = d["formatter"]
        return pn.widgets.DiscreteSlider(**kwargs)


class DynamicParam(param.Parameterized):
    """This class is used to convert a parameter from the ``param`` module to
    a widget from ``panel``.

    Examples
    ========

    >>> import param
    >>> import panel as pn
    >>> p = param.Number(default=0, bounds=(0, 5))
    >>> dyn_param = DynamicParam(p)
    >>> tmp_panel = pn.Param(dyn_param)
    >>> widget = tmp_panel.widget("dyn_param_0")
    >>> type(widget)
    panel.widgets.slider.FloatSlider

    """
    def __init__(self, current_param, **kwargs):
        # remove the previous class attributes added by the previous instances
        prev_params = [k for k in type(self).__dict__.keys() if "dyn_param_" in k]
        for p in prev_params:
            delattr(type(self), p)

        # this must be present in order to assure correct behaviour
        super().__init__(name="", **kwargs)

        self.param.add_parameter("dyn_param_0", current_param)


class PanelCommon(IPlot):
    """Common code for interactive applications with Holoviz Panel.
    """

    @property
    def pane_kw(self):
        """Return the keyword arguments used to customize the wrapper to the
        plot.
        """
        return self._pane_kw

    def _init_pane(self):
        """Here we wrap the figure exposed by the backend with a Pane, which
        allows to set useful properties.
        """
        # NOTE: If the following import statement was located at the
        # beginning of the file, there would be a circular import.
        from spb import KB, MB, BB, PB

        default_kw = {}
        if isinstance(self.backend, PB):
            pane_func = pn.pane.Plotly
        elif (
            isinstance(self.backend, MB) or        # vanilla MB
            (
                hasattr(self.backend, "is_matplotlib_fig") and
                self.backend.is_matplotlib_fig     # plotgrid with all MBs
            )
        ):
            # since we are using Jupyter and interactivity, it is useful to
            # activate ipympl interactive frame, as well as setting a lower
            # dpi resolution of the matplotlib image
            default_kw["dpi"] = 96
            # NOTE: the following must be set to False in order for the
            # example outputs to become visible on Sphinx.
            default_kw["interactive"] = False
            pane_func = pn.pane.Matplotlib
        elif isinstance(self.backend, BB):
            pane_func = pn.pane.Bokeh
        elif isinstance(self.backend, KB):
            # TODO: for some reason, panel is going to set width=0
            # if K3D-Jupyter is used.
            # Temporary workaround: create a Pane with a default width.
            # Long term solution: create a PR on panel to create a K3DPane
            # so that panel will automatically deal with K3D, in the same
            # way it does with Bokeh, Plotly, Matplotlib, ...
            default_kw["width"] = 800
            pane_func = pn.pane.panel
        else:
            # here we are dealing with plotgrid of BB/PB/or mixed backend...
            # but not with plotgrids of MB
            self._init_pane_for_plotgrid()
            return
        kw = self.merge({}, default_kw, self._pane_kw)
        self.pane = pane_func(self._binding, **kw)

    @property
    def layout_controls(self):
        """Return the controls used by the interactive application"""
        raise NotImplementedError

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
            kw = self._template if isinstance(self._template, dict) else {}
            kw = self.merge(default_template_kw, kw)
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

        self._populate_template(template)

        if show:
            return template.servable().show()
        return template


class InteractivePlot(PanelCommon):

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
        mergedeep = import_module('mergedeep')
        self.merge = mergedeep.merge

        layout = kwargs.pop("layout", "tb").lower()
        available_layouts = ["tb", "bb", "sbl", "sbr"]
        if layout not in available_layouts:
            warnings.warn(
                "`layout` must be one of the following: %s\n"
                "Falling back to layout='tb'." % available_layouts
            )
            layout = "tb"
        self._layout = layout
        self._ncols = kwargs.pop("ncols", 2)
        self._throttled = kwargs.pop("throttled", cfg["interactive"]["throttled"])
        self._servable = kwargs.pop("servable", cfg["interactive"]["servable"])
        self._use_latex = kwargs.pop("use_latex", cfg["interactive"]["use_latex"])
        self._pane_kw = kwargs.pop("pane_kw", dict())
        self._custom_css = kwargs.pop("custom_css", "")
        self._template = kwargs.pop("template", None)
        self._name = name

        params = _aggregate_parameters(params, series)
        self._original_params = params

        # The following dictionary will be used to create the appropriate
        # lambda function arguments:
        #    key: the provided symbol
        #    val: widget
        self.mapping = create_widgets(params, self._use_latex)

        plotgrid = kwargs.get("plotgrid", None)
        if plotgrid:
            self.backend = plotgrid
            self._binding = pn.bind(self._update, *self._widgets_for_binding())
        else:
            # assure that each series has the correct values associated
            # to parameters
            series = list(series)
            for s in series:
                if s.is_interactive:
                    s.params = self.read_parameters()

            is_3D = all([s.is_3D for s in series])
            Backend = kwargs.pop("backend", THREE_D_B if is_3D else TWO_D_B)
            kwargs["is_iplot"] = True
            kwargs["imodule"] = "panel"
            self.backend = Backend(*series, **kwargs)
            # _check_misspelled_kwargs(self.backend, **original_kwargs)

            from spb import PB
            if Backend is PB:
                self._binding = pn.bind(
                    self._update_plotly, *self._widgets_for_binding())
            else:
                self._binding = pn.bind(
                    self._update, *self._widgets_for_binding())

    def _widgets_for_binding(self):
        """Select the appropriate things to return for the `pn.bind` function.
        """
        widgets = list(self.mapping.values())
        if self._throttled:
            def is_panel_slider(t):
                if not isinstance(t, pn.widgets.base.Widget):
                    return False
                if "Slider" in type(t).__name__:
                    return True
                return False
            widgets = [
                w if not is_panel_slider(w) else w.param.value_throttled
                for w in widgets
            ]
        return widgets

    def read_parameters(self):
        return {symb: widget.value for symb, widget in self.mapping.items()}

    def _update(self, *values):
        d = {symb: v for symb, v in zip(list(self.mapping.keys()), values)}
        self.backend.update_interactive(d)
        return self.fig

    def _update_plotly(self, *values):
        d = {symb: v for symb, v in zip(list(self.mapping.keys()), values)}
        self.backend.update_interactive(d)
        # NOTE: while pn.pane.Plotly can receive an instance of go.Figure,
        # the update will be extremely slow, because after each trace
        # is updated with new data, it will trigger an update event on the
        # javascript side. Instead, by providing the following dictionary,
        # first the traces are updated, then the pane creates the figure
        # (with only one single javascript update).
        # TODO: can the backend be modified by adding data and layout
        # attributes, avoiding the creation of the figure? The figure could
        # be created inside the fig getter.
        return self.fig.to_dict()

    def _get_iplot_kw(self):
        return {
            "backend": type(self.backend),
            "layout": self._layout,
            "template": self._template,
            "ncols": self._ncols,
            "throttled": self._throttled,
            "use_latex": self._use_latex,
            "params": self._original_params,
            "pane_kw": self._pane_kw
        }

    def _init_pane_for_plotgrid(self):
        # First, set the necessary data to create bindings for each subplot
        self.backend.pre_set_bindings(
            list(self.mapping.keys()),
            self._widgets_for_binding()
        )
        # Then, create the pn.GridSpec figure
        self.pane = self.backend.fig

    def _populate_template(self, template):
        template.main.append(self.pane)
        template.sidebar.append(self.layout_controls)

    @property
    def layout_controls(self):
        widgets = list(self.mapping.values())
        return pn.GridBox(*widgets, ncols=self._ncols)


def iplot(*series, show=True, **kwargs):
    """Create an interactive application containing widgets and charts in order
    to study symbolic expressions, using Holoviz's Panel for the user interace.

    This function is already integrated with many of the usual
    plotting functions: since their documentation is more specific, it is
    highly recommended to use those instead.

    However, the following documentation explains in details the main features
    exposed by the interactive module, which might not be included on the
    documentation of those other functions.

    Parameters
    ==========

    series : BaseSeries
        Instances of :py:class:`spb.series.BaseSeries`, representing the
        symbolic expression to be plotted.

    params : dict
        A dictionary mapping the symbols to a parameter. The parameter can be:

        1. An instance of :py:class:`panel.widgets.base.Widget`, something like
           :py:class:`panel.widgets.FloatSlider`.
        2. An instance of :py:class:`param.parameterized.Parameter`.
        3. A tuple with the form:
           `(default, min, max, N, tick_format, label, spacing)`,
           which will instantiate a :py:class:`panel.widgets.FloatSlider` or
           a :py:class:`panel.widgets.DiscreteSlider`, depending on the
           spacing strategy. In particular:

           - default, min, max : float
                Default value, minimum value and maximum value of the slider,
                respectively. Must be finite numbers. The order of these 3
                numbers is not important: the module will figure it out
                which is what.
           - N : int, optional
                Number of steps of the slider.
           - tick_format : TickFormatter or None, optional
                Provide a formatter for the tick value of the slider. If None,
                `panel` will automatically apply a default formatter.
                Alternatively, an instance of
                :py:class:`bokeh.models.formatters.TickFormatter` can be used.
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

        * If Matplotlib is used, the figure is wrapped by
          :py:class:`panel.pane.plot.Matplotlib`. Two interesting options are:

          * ``interactive``: wheter to activate the ipympl interactive backend.
          * ``dpi``: set the dots per inch of the output png. Default to 96.

        * If Plotly is used, the figure is wrapped by
          :py:class:`panel.pane.plotly.Plotly`.

        * If Bokeh is used, the figure is wrapped by
          :py:class:`panel.pane.plot.Bokeh`.

    servable : bool, optional
        Default to False, which will show the interactive application on the
        output cell of a Jupyter Notebook. If True, the application will be
        served on a new browser window.

    show : bool, optional
        Default to True.
        If True, it will return an object that will be rendered on the
        output cell of a Jupyter Notebook. If False, it returns an instance
        of ``InteractivePlot``, which can later be be shown by calling the
        ``show()`` method.

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

        * an instance of :py:class:`panel.template.base.BasicTemplate`.
        * a subclass of :py:class:`panel.template.base.BasicTemplate`.

    title : str or tuple
        The title to be shown on top of the figure. To specify a parametric
        title, write a tuple of the form:``(title_str, param_symbol1, ...)``,
        where:

        * ``title_str`` must be a formatted string, for example:
          ``"test = {:.2f}"``.
        * ``param_symbol1, ...`` must be a symbol or a symbolic expression
          whose free symbols are contained in the ``params`` dictionary.

    throttled : boolean, optional
        Default to False. If True the recompute will be done at mouse-up event
        on sliders. If False, every slider tick will force a recompute.

    use_latex : bool, optional
        Default to True.
        If True, the latex representation of the symbols will be used in the
        labels of the parameter-controls. If False, the string
        representation will be used instead.


    See also
    ========

    create_widgets


    Notes
    =====

    1. This function is specifically designed to work within Jupyter Notebook.
       It is also possible to use it from a regular Python console,
       by executing: ``iplot(..., servable=True)``, which will create a server
       process loading the interactive plot on the browser.
       However, :py:class:`spb.backends.k3d.K3DBackend` is not supported
       in this mode of operation.

    2. The interactive application consists of two main containers:

       * a pane containing the widgets.
       * a pane containing the chart, which can be further customize
         by setting the ``pane_kw`` dictionary. Please, read its documentation
         to understand the available options.

    3. Some examples use an instance of
       :py:class:`bokeh.models.PrintfTickFormatter` to format the
       value shown by a slider. This class is exposed by Bokeh, but can be
       used in interactive plots with any backend.

    4. It has been observed that Dark Reader (or other night-mode-enabling
       browser extensions) might interfere with the correct behaviour of
       the output of  interactive plots. Please, consider adding ``localhost``
       to the exclusion list of such browser extensions.

    5. :py:class:`spb.backends.matplotlib.MatplotlibBackend` can be used,
       but the resulting figure is just a PNG image without any interactive
       frame. Thus, data exploration is not great. Therefore, the use of
       :py:class:`spb.backends.plotly.PlotlyBackend` or
       :py:class:`spb.backends.bokeh.BokehBackend` is encouraged.

    6. When ``BokehBackend`` is used:

       * rendering of gradient lines is slow.
       * color bars might not update their ranges.


    Examples
    ========

    NOTE: the following examples use the ordinary plotting functions because
    ``iplot`` is already integrated with them.

    Surface plot between -10 <= x, y <= 10 discretized with 50 points
    on both directions, with a damping parameter varying from 0 to 1, and a
    default value of 0.15:

    .. panel-screenshot::

       from sympy import *
       from spb import *
       x, y, z = symbols("x, y, z")
       r = sqrt(x**2 + y**2)
       d = symbols('d')
       expr = 10 * cos(r) * exp(-r * d)
       graphics(
           surface(
               expr, (x, -10, 10), (y, -10, 10), label="z-range",
               params={d: (0.15, 0, 1)}, n=51, use_cm=True,
               wireframe = True, wf_n1=15, wf_n2=15,
               wf_rendering_kw={"line_color": "#003428", "line_width": 0.75}
           ),
           title = "My Title",
           xlabel = "x axis",
           ylabel = "y axis",
           zlabel = "z axis",
           backend = PB
       )

    A line plot of the magnitude of a transfer function, illustrating the use
    of multiple expressions and:

    1. some expression may not use all the parameters.
    2. custom labeling of the expressions.
    3. custom rendering of the expressions.
    4. different ways to create sliders.
    5. custom format of the value shown on the slider. This might be useful to
       correctly visualize very small or very big numbers.
    6. custom labeling of the sliders.

    .. panel-screenshot::

       from sympy import (symbols, sqrt, cos, exp, sin, pi, re, im,
           Matrix, Plane, Polygon, I, log)
       from spb import *
       from bokeh.models.formatters import PrintfTickFormatter
       import panel as pn
       import param
       formatter = PrintfTickFormatter(format="%.3f")
       kp, t, xi, o = symbols("k_P, tau, xi, omega")
       G = kp / (I**2 * t**2 * o**2 + 2 * xi * t * o * I + 1)
       mod = lambda x: 20 * log(sqrt(re(x)**2 + im(x)**2), 10)
       plot(
           (mod(G.subs(xi, 0)), (o, 0.1, 100), "G(xi=0)", {"line_dash": "dotted"}),
           (mod(G.subs(xi, 1)), (o, 0.1, 100), "G(xi=1)", {"line_dash": "dotted"}),
           (mod(G), (o, 0.1, 100), "G"),
           params = {
               kp: (1, 0, 3),
               t: param.Number(default=1, bounds=(0, 3), label="Time constant"),
               xi: pn.widgets.FloatSlider(value=0.2, start=0, end=1,
                   step=0.005, format=formatter, name="Damping ratio")
           },
           backend = BB,
           n = 2000,
           xscale = "log",
           xlabel = "Frequency, omega, [rad/s]",
           ylabel = "Magnitude [dB]",
       )

    A line plot illustrating the Fouries series approximation of a saw tooth
    wave and:

    1. custom format of the value shown on the slider.
    2. creation of an integer spinner widget.

    .. panel-screenshot::

       from sympy import *
       from spb import *
       import panel as pn
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
               m: pn.widgets.IntInput(value=4, start=1, name="Sum up to n ")
           },
           xlabel = "x",
           ylabel = "y",
           backend = BB
       )

    A line plot with a parameter representing an angle in radians, but
    showing the value in degrees on its label:

    .. panel-screenshot::
       :small-size: 800, 570

       from sympy import sin, pi, symbols
       from spb import *
       from bokeh.models.formatters import CustomJSTickFormatter
       # Javascript code is passed to `code=`
       formatter = CustomJSTickFormatter(code="return (180./3.1415926 * tick).toFixed(2)")
       x, t = symbols("x, t")

       plot(
           (1 + x * sin(t), (x, -5, 5)),
           params = {
               t: (1, -2 * pi, 2 * pi, 100, formatter, "theta [deg]")
           },
           backend = MB,
           xlabel = "x", ylabel = "y",
           ylim = (-3, 4)
       )

    Combine together interactive and non interactive plots:

    .. panel-screenshot::
       :small-size: 800, 570

       from sympy import sin, cos, symbols
       from spb import *
       x, u = symbols("x, u")
       params = {
           u: (1, 0, 2)
       }
       graphics(
           line(cos(u * x), (x, -5, 5), params=params),
           line(sin(u * x), (x, -5, 5), params=params),
           line(
               sin(x)*cos(x), (x, -5, 5),
               rendering_kw={"marker": "^", "linestyle": ":"}, n=50),
       )

    Serves the interactive plot to a separate browser window. Note that
    :py:class:`spb.backends.k3d.K3DBackend` is not supported for this
    operation mode. Also note the two ways to create a integer sliders.

    .. panel-screenshot::
       :small-size: 800, 500

       from sympy import *
       from spb import *
       import param
       import panel as pn
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
               # integer parameter created with widgets
               c: pn.widgets.IntSlider(value=3, start=1, end=5, name="c")
           },
           backend = BB,
           aspect = "equal",
           n = 5000,
           layout = "sbl",
           ncols = 1,
           servable = True,
           name = "Non Circular Planetary Drive - Ring Profile"
       )

    """
    i = InteractivePlot(*series, **kwargs)
    if show:
        return i.show()
    return i


def create_widgets(params, use_latex=True, **kwargs):
    """ Create panel's widgets starting from parameters.

    Parameters
    ==========

    params : dict
        A dictionary mapping the symbols to a parameter. The parameter can be:

        1. an instance of :py:class:`param.parameterized.Parameter`.
           Refer to [#fn5]_ for a list of available parameters.
        2. A tuple with the form:
           `(default, min, max, N, tick_format, label, spacing)`,
           which will instantiate a :py:class:`panel.widgets.FloatSlider` or
           a :py:class:`panel.widgets.DiscreteSlider`, depending on the
           spacing strategy. In particular:

           - default, min, max : float
                Default value, minimum value and maximum value of the slider,
                respectively. Must be finite numbers.
           - N : int, optional
                Number of steps of the slider.
           - tick_format : TickFormatter or None, optional
                Provide a formatter for the tick value of the slider. If None,
                `panel` will automatically apply a default formatter.
                Alternatively, an instance of
                :py:class:`bokeh.models.formatters.TickFormatter` can be used.
                Default to None.
           - label: str, optional
                Custom text associated to the slider.
           - spacing : str, optional
                Specify the discretization spacing. Default to ``"linear"``,
                can be changed to ``"log"``.

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
           y: (200, 1, 1000, 10, "test", "log"),
           z: param.Integer(3, softbounds=(3, 10), label="n")
       })


    References
    ==========

    .. [#fn5] https://panel.holoviz.org/user_guide/Param.html


    See also
    ========

    iplot

    """
    results = dict()
    for symb, v in params.items():
        if isinstance(v, (pn.widgets.base.Widget)):
            if hasattr(v, "name") and len(v.name) == 0:
                # show the symbol if no label was set to the widget
                wrapper = "$$%s$$" if use_latex else "%s"
                func = latex if use_latex else str
                v.name = wrapper % func(symb)
            results[symb] = v
        elif isinstance(v, param.parameterized.Parameter):
            dyn_param = DynamicParam(v)
            tmp_panel = pn.Param(dyn_param)
            results[symb] = tmp_panel.widget("dyn_param_0")
        elif isinstance(v, (list, tuple)):
            d = _tuple_to_dict(symb, v, use_latex, "$$%s$$")
            results[symb] = _dict_to_slider(d)
        else:
            raise TypeError(
                "Parameter type not recognized. Expected list/tuple/"
                "param.Parameter/pn.widgets. Received: %s "
                "of type %s" % (v, type(v))
            )
    return results
