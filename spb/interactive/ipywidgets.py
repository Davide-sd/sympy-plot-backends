import ipywidgets
from sympy import latex
from sympy.external import import_module
from spb.defaults import TWO_D_B, THREE_D_B
from spb.interactive import _tuple_to_dict, IPlot
from spb.utils import _aggregate_parameters
from spb import BB, MB, PlotGrid
from IPython.display import clear_output
import warnings


def _build_widgets(params, use_latex=True):
    widgets = []
    for s, v in params.items():
        if hasattr(v, "__iter__") and (not isinstance(v, str)):
            d = _tuple_to_dict(s, v, use_latex=use_latex)
            formatter = d.pop("formatter")
            if formatter and not isinstance(formatter, str):
                warnings.warn(
                    "`ipywidgets` requires ``formatter`` to be of type str.\n"
                    "Received: %s\nFor example, try '.5f'" % type(formatter)
                )
            if d.pop("type") == "linear":
                d2 = d.copy()
                if formatter and isinstance(formatter, str):
                    d2["readout_format"] = formatter
                widgets.append(ipywidgets.FloatSlider(**d2))
            else:
                widgets.append(ipywidgets.FloatLogSlider(**d))
        elif isinstance(v, ipywidgets.Widget):
            if hasattr(v, "description") and len(v.description) == 0:
                # show the symbol if no label was set to the widget
                wrapper = "$%s$" if use_latex else "%s"
                func = latex if use_latex else str
                v.description = wrapper % func(s)
            widgets.append(v)
        else:
            raise ValueError(
                "Cannot build a widget with the provided input: %s" % v)
    return widgets


def _build_grid_layout(widgets, ncols):
    np = import_module('numpy')

    if ncols <= 0:
        raise ValueError(
            "The number of columns must be greater or equal than 1.")

    nrows = int(np.ceil(len(widgets) / ncols))
    grid = ipywidgets.GridspecLayout(nrows, ncols)
    c = 0
    for i in range(ncols):
        for j in range(nrows):
            if c < len(widgets):
                grid[j, i] = widgets[c]
                c += 1
    return grid


class InteractivePlot(IPlot):
    def __init__(self, *series, **kwargs):
        params = kwargs.pop("params", {})
        params = _aggregate_parameters(params, series)
        self._original_params = params
        self._use_latex = kwargs.pop("use_latex", True)
        self._ncols = kwargs.get("ncols", 2)
        self._layout = kwargs.get("layout", "tb")

        self._widgets = _build_widgets(params, self._use_latex)
        self._grid_widgets = _build_grid_layout(self._widgets, self._ncols)

        # map symbols to widgets
        self._params_widgets = {
            k: v for k, v in zip(params.keys(), self._widgets)}

        plotgrid = kwargs.get("plotgrid", None)

        if plotgrid:
            self.backend = plotgrid
        else:
            # assure that each series has the correct values associated
            # to parameters
            for s in series:
                if s.is_interactive:
                    s.params = {
                        k: v.value for k, v in self._params_widgets.items()}

            is_3D = all([s.is_3D for s in series])
            Backend = kwargs.pop("backend", THREE_D_B if is_3D else TWO_D_B)
            kwargs["is_iplot"] = True
            kwargs["imodule"] = "ipywidgets"
            kwargs["use_latex"] = self._use_latex
            self.backend = Backend(*series, **kwargs)

    def _get_iplot_kw(self):
        return {
            "backend": type(self.backend),
            "layout": self._layout,
            "ncols": self._ncols,
            "use_latex": self._use_latex,
            "params": self._original_params,
        }

    def _update(self, change):
        # bind widgets state to this update function
        self.backend.update_interactive(
            {k: v.value for k, v in self._params_widgets.items()})
        if isinstance(self.backend, BB):
            bokeh = import_module(
                'bokeh',
                import_kwargs={'fromlist': ['io']},
                warn_not_installed=True,
                min_module_version='2.3.0')
            with self._output_figure:
                clear_output(True) # NOTE: this is the cause of flickering
                bokeh.io.show(self.backend.fig)

    def show(self):
        for w in self._widgets:
            w.observe(self._update, "value")

        # create the output figure
        if (isinstance(self.backend, MB) or
            (isinstance(self.backend, PlotGrid) and self.backend.is_matplotlib_fig)):
            # without plt.ioff, picture will show up twice. Morover, there
            # won't be any update
            self.backend.plt.ioff()
            if isinstance(self.backend, PlotGrid):
                if not self.backend.imagegrid:
                    self.backend.fig.tight_layout()
            self._output_figure = ipywidgets.Box([self.backend.fig.canvas])
        elif isinstance(self.backend, BB):
            if self.backend.update_event:
                warnings.warn(
                    "You are trying to generate an interactive plot with "
                    "Bokeh using `update_event=True`. This mode of operation "
                    "is not supported. However, setting "
                    "`imodule='panel', servable=True` "
                    "with BokehBackend works just fine."
                )
            self._output_figure = ipywidgets.Output()
            bokeh = import_module(
                'bokeh',
                import_kwargs={'fromlist': ['io']},
                warn_not_installed=True,
                min_module_version='2.3.0')
            with self._output_figure:
                bokeh.io.show(self.backend.fig)
        else:
            self._output_figure = self.backend.fig

        if (isinstance(self.backend, MB) or
            (isinstance(self.backend, PlotGrid) and self.backend.is_matplotlib_fig)):
            # turn back interactive behavior with plt.ion, so that picture
            # will be updated.
            self.backend.plt.ion() # without it there won't be any update

        if self._layout == "tb":
            return ipywidgets.VBox([self._grid_widgets, self._output_figure])
        elif self._layout == "bb":
            return ipywidgets.VBox([self._output_figure, self._grid_widgets])
        elif self._layout == "sbl":
            return ipywidgets.HBox([self._grid_widgets, self._output_figure])
        return ipywidgets.HBox([self._output_figure, self._grid_widgets])


def iplot(*series, show=True, **kwargs):
    """Create an interactive application containing widgets and charts in order
    to study symbolic expressions, using ipywidgets.

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

        1. a widget.
        2. a tuple of the form:
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

        Note that the parameters cannot be linked together (ie, one parameter
        cannot depend on another one).

    layout : str, optional
        The layout for the controls/plot. Possible values:

        - ``'tb'``: controls in the top bar.
        - ``'bb'``: controls in the bottom bar.
        - ``'sbl'``: controls in the left side bar.
        - ``'sbr'``: controls in the right side bar.

        The default value is ``'tb'``.

    ncols : int, optional
        Number of columns to lay out the widgets. Default to 2.

    show : bool, optional
        Default to True.
        If True, it will return an object that will be rendered on the
        output cell of a Jupyter Notebook. If False, it returns an instance
        of ``InteractivePlot``, which can later be shown by calling the
        ``show()`` method.

    title : str or tuple
        The title to be shown on top of the figure. To specify a parametric
        title, write a tuple of the form:``(title_str, param_symbol1, ...)``,
        where:

        * ``title_str`` must be a formatted string, for example:
          ``"test = {:.2f}"``.
        * ``param_symbol1, ...`` must be a symbol or a symbolic expression
          whose free symbols are contained in the ``params`` dictionary.

    use_latex : bool, optional
        Default to True.
        If True, the latex representation of the symbols will be used in the
        labels of the parameter-controls. If False, the string
        representation will be used instead.


    Notes
    =====

    1. This function is specifically designed to work within Jupyter Notebook
       and requires the
       `ipywidgets module <https://ipywidgets.readthedocs.io>`_ .

    2. To update Matplotlib plots, the ``%matplotlib widget`` command must be
       executed at the top of the Jupyter Notebook. It requires the
       installation of the
       `ipympl module <https://github.com/matplotlib/ipympl>`_ .


    Examples
    ========

    NOTE: the following examples use the ordinary plotting functions because
    ``iplot`` is already integrated with them.

    Surface plot between -10 <= x, y <= 10 discretized with 50 points
    on both directions, with a damping parameter varying from 0 to 1, and a
    default value of 0.15:

    .. code-block::

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
           backend = PB,
           use_latex=False
       )

    A line plot illustrating how to specify widgets. In particular:

    1. the parameter ``d`` will be rendered as a slider, with a custom
       formatter showing 3 decimal places.
    2. the parameter ``n`` is a spinner.
    3. the parameter ``phi`` will be rendered as a slider: note the custom
       number of steps and the custom label.
    4. when using Matplotlib, the ``%matplotlib widget`` must be executed at
       the top of the notebook.

    .. code-block:: python

       %matplotlib widget
       from sympy import *
       from spb import *
       import ipywidgets

       x, phi, n, d = symbols("x, phi, n, d")
       plot(
           cos(x * n - phi) * exp(-abs(x) * d), (x, -5*pi, 5*pi),
           params={
               d: (0.1, 0, 1, ".3f"),
               n: ipywidgets.BoundedIntText(value=2, min=1, max=10,
                   description="$n$"),
               phi: (0, 0, 2*pi, 50, "$\\phi$ [rad]")
           },
           ylim=(-1.25, 1.25))

    A line plot illustrating the Fouries series approximation of a saw tooth
    wave and:

    1. custom number of steps and label in the slider.
    2. creation of an integer spinner widget.

    .. code-block:: python

       from sympy import *
       from spb import *
       import ipywidgets

       x, T, n, m = symbols("x, T, n, m")
       sawtooth = frac(x / T)
       # Fourier Series of a sawtooth wave
       fs = S(1) / 2 - (1 / pi) * Sum(sin(2 * n * pi * x / T) / n, (n, 1, m))

       plot(
           (sawtooth, (x, 0, 10), "f", {"line_dash": "dotted"}),
           (fs, (x, 0, 10), "approx"),
           params = {
               T: (4, 0, 10, 80, "Period, T"),
               m: ipywidgets.BoundedIntText(value=4, min=1, max=100,
                   description="Sum up to n ")
           },
           xlabel = "x",
           ylabel = "y",
           backend = BB
       )

    A line plot of the magnitude of a transfer function, illustrating the use
    of multiple expressions and:

    1. some expression may not use all the parameters.
    2. custom labeling of the expressions.
    3. custom rendering of the expressions.
    4. custom number of steps in the slider.
    5. custom labeling of the parameter-sliders.

    .. code-block:: python

       from sympy import (symbols, sqrt, cos, exp, sin, pi, re, im,
           Matrix, Plane, Polygon, I, log)
       from spb import *
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
               z: (0.2, 0, 1, 200, "z")
           },
           backend = BB,
           n = 2000,
           xscale = "log",
           xlabel = "Frequency, omega, [rad/s]",
           ylabel = "Magnitude [dB]",
       )

    A polar line plot. Note:

    1. when using Matplotlib, the ``%matplotlib widget`` must be executed at
       the top of the notebook.
    2. the two ways to create a integer sliders.

    .. code-block:: python

       %matplotlib widget
       from sympy import *
       from spb import *
       import ipywidgets

       p1, p2, t, r, c = symbols("p1, p2, t, r, c")
       phi = - (r * t + p1 * sin(c * r * t) + p2 * sin(2 * c * r * t))
       phip = phi.diff(t)
       r1 = phip / (1 + phip)

       plot_polar(
           (r1, (t, 0, 2*pi)),
           params = {
               p1: (0.035, -0.035, 0.035, 50),
               p2: (0.005, -0.02, 0.02, 50),
               # integer parameter created with ipywidgets
               r: ipywidgets.BoundedIntText(value=2, min=2, max=5,
                   description="r"),
               # integer parameter created with usual syntax
               c: (3, 1, 5, 4)
           },
           backend = MB,
           aspect = "equal",
           n = 5000,
           name = "Non Circular Planetary Drive - Ring Profile"
       )

    Combine together interactive and non interactive plots:

    .. code-block:: python

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

    """
    i = InteractivePlot(*series, **kwargs)
    if show:
        return i.show()
    return i
