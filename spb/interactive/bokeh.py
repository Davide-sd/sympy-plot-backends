"""
Implements interactive-widgets plotting with Bokeh.
"""

from spb.defaults import TWO_D_B, THREE_D_B, cfg
from spb.utils import _validate_kwargs, get_environment
from spb.interactive import _tuple_to_dict, IPlot
from spb.interactive.bootstrap_spb import SymPyBootstrapTemplate
from spb.plotgrid import PlotGrid
from sympy.external import import_module
import warnings

# param = import_module(
#     'param',
#     min_module_version='1.11.0',
#     warn_not_installed=True)
bokeh = import_module(
    'bokeh',
    import_kwargs={
        'fromlist': [
            'models', 'events', 'plotting', 'io',
            'palettes', 'embed', 'resources'
        ]
    },
    warn_not_installed=True,
    min_module_version='2.3.0'
)


def _dict_to_slider(d):
    if d["type"] == "linear":
        kwargs = dict(
            start=d["min"], end=d["max"], value=d["value"],
            step=d["step"], title=d["description"]
        )
        if d["formatter"]:
            kwargs["format"] = d["formatter"]
        return bokeh.models.Slider(**kwargs)
    else:
        raise NotImplementedError("Bokeh doesn't support logarithmic sliders.")
        # np = import_module("numpy")
        # _min, _max, step, value = d["min"], d["max"], d["step"], d["value"]
        # N = int((_max - _min) / step)
        # # divide the range in N steps evenly spaced in a log scale
        # options = np.geomspace(_min, _max, N)
        # # the provided default value may not be in the computed options.
        # # If that's the case, I chose the closest value
        # if value not in options:
        #     value = min(options, key=lambda x: abs(x - value))

        # kwargs = dict(
        #     options=options.tolist(), value=value, name=d["description"])
        # if d["formatter"]:
        #     kwargs["formatter"] = d["formatter"]
        # return pn.widgets.DiscreteSlider(**kwargs)


class InteractivePlot(IPlot):

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

        if params is None:
            params = {}
        if len(params) == 0:
            # this is the case when an interactive widget plot is build with
            # the `graphics` interface: need to construct the params dict
            # by looping over the series
            for s in series:
                if s.is_interactive:
                    params.update(s.params)
        self._original_params = params

        # The following dictionary will be used to create the appropriate
        # lambda function arguments:
        #    key: the provided symbol
        #    val: widget
        self.mapping = create_widgets(params, self._use_latex)
        for widget in self.mapping.values():
            if hasattr(widget, "value"):
                widget.on_change('value', self._update)
            else:
                widget.on_change('active', self._update)

        plotgrid = kwargs.get("plotgrid", None)
        if plotgrid:
            self.backend = plotgrid
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
            _validate_kwargs(self.backend, **original_kwargs)

        self._run_in_notebook = False
        if get_environment() == 0:
            self._run_in_notebook = True
            bokeh.io.output_notebook(hide_banner=True)

    def read_parameters(self):
        readouts = {}
        for symb, widget in self.mapping.items():
            if hasattr(widget, "value"):
                readouts[symb] = widget.value
            else:
                readouts[symb] = widget.active
        return readouts

    def _update(self, attr, old, new):
        d = self.read_parameters()
        self.backend.update_interactive(d)

    @property
    def pane_kw(self):
        """Return the keyword arguments used to customize the wrapper to the
        plot.
        """
        return self._pane_kw

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

    @property
    def layout_controls(self):
        n = self._ncols
        widgets = list(self.mapping.values())
        rows = [widgets[i:i+n] for i in range(0, len(widgets), n)]
        rows = [bokeh.layouts.row(*r) for r in rows]
        return bokeh.layouts.column(*rows)

    def _launch_server(self, doc):
        """ By launching a server application, we can use Python callbacks
        associated to events.
        """
        doc.theme = cfg["bokeh"]["theme"]
        doc.add_root(self._pane)

    def show(self):
        fig = self.backend.fig
        if self._layout == "tb":
            # NOTE: white because on Jupyter with dark theme, control's labels
            # are unreadable...
            content = bokeh.layouts.column(self.layout_controls, fig,
                background="white")
        elif self._layout == "bb":
            content = bokeh.layouts.column(fig, self.layout_controls,
                background="white")
        elif self._layout == "sbl":
            content = bokeh.layouts.row(self.layout_controls, fig,
                background="white")
        elif self._layout == "sbr":
            content = bokeh.layouts.row(fig, self.layout_controls,
                background="white")
        self._pane = content

        if self._run_in_notebook:
            return bokeh.plotting.show(self._launch_server)
        else:
            # NOTE:
            # 1. From: https://docs.bokeh.org/en/latest/docs/user_guide/server/library.html
            #    In particular: https://github.com/bokeh/bokeh/tree/3.4.0/examples/server/api/standalone_embed.py
            # 2. Watch out for memory leaks on Firefox.
            # 3. Use Control+C to stop the server process
            from bokeh.server.server import Server
            server = Server(self._launch_server, num_procs=1)
            server.start()
            server.io_loop.add_callback(server.show, "/")
            server.io_loop.start()


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

        * Refer to [#fn2]_ for
          :py:class:`spb.backends.matplotlib.MatplotlibBackend`.
          Two interesting options are:

          * ``interactive``: wheter to activate the ipympl interactive backend.
          * ``dpi``: set the dots per inch of the output png. Default to 96.

        * Refer to [#fn3]_ for :py:class:`spb.backends.plotly.PlotlyBackend`.

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


    Examples
    ========

    NOTE: the following examples use the ordinary plotting function because
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
    2. creation of an integer spinner widget. This is achieved by setting
       ``None`` as one of the bounds of the integer parameter.

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

    6. :py:class:`spb.backends.matplotlib.MatplotlibBackend` can be used,
       but the resulting figure is just a PNG image without any interactive
       frame. Thus, data exploration is not great. Therefore, the use of
       :py:class:`spb.backends.plotly.PlotlyBackend` or
       :py:class:`spb.backends.bokeh.BokehBackend` is encouraged.

    7. When :py:class:`spb.backends.bokeh.BokehBackend` is used:

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

    iplot, create_series

    """
    results = dict()
    for symb, v in params.items():
        if isinstance(v, (list, tuple)):
            d = _tuple_to_dict(symb, v, use_latex, "$$%s$$")
            results[symb] = _dict_to_slider(d)
        elif isinstance(v, bokeh.models.widgets.widget.Widget):
            results[symb] = v
        else:
            raise TypeError(
                "Parameter type not recognized. Expected list/tuple/"
                "param.Parameter/pn.widgets. Received: %s "
                "of type %s" % (v, type(v))
            )
    return results
