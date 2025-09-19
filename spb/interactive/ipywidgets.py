import ipywidgets
from sympy import latex
from sympy.external import import_module
from spb.defaults import cfg, TWO_D_B, THREE_D_B
from spb.doc_utils.docstrings import _PARAMS
from spb.doc_utils.ipython import (
    modify_parameterized_doc,
    modify_graphics_series_doc
)
from spb.interactive import _tuple_to_dict, IPlot
from spb.utils import _aggregate_parameters
from spb import BB, MB, PlotGrid
from IPython.display import clear_output
import warnings
import param
import math
from collections import defaultdict


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

    nrows = int(np.ceil(len(widgets) / ncols))
    grid = ipywidgets.GridspecLayout(nrows, ncols)
    c = 0
    for i in range(ncols):
        for j in range(nrows):
            if c < len(widgets):
                grid[j, i] = widgets[c]
                c += 1
    return grid


def _get_widget_from_param_module(obj: param.Parameterized, p_name: str):
    """
    Attempt to build a widget with `ipywidgets` from the specified parameter.

    Parameters
    ----------
    obj : param.Parameterized
        The object containing the parameters
    p_name : str
        The name of the parameter to consider.
    """
    parameter = obj.param[p_name]
    default_val = getattr(obj, p_name)

    if isinstance(parameter, param.Boolean):
        widget = ipywidgets.Checkbox(
            value=default_val,
            description=parameter.label
        )

    elif isinstance(parameter, param.Number):
        bounds = parameter.bounds
        ibounds = parameter.inclusive_bounds
        softbounds = parameter.softbounds
        lb, hb = bounds if (bounds is not None) else (None, None)
        ilb, ihb = ibounds if (ibounds is not None) else (True, True)
        slb, shb = softbounds if (softbounds is not None) else (None, None)

        is_spinner_l = True if (lb is None) and (slb is None) else False
        is_spinner_h = True if (hb is None) and (shb is None) else False
        is_spinner = is_spinner_l or is_spinner_h
        is_integer = isinstance(parameter, param.Integer)
        if is_integer:
            type_ = ipywidgets.IntSlider if not is_spinner else ipywidgets.IntText
        else:
            type_ = ipywidgets.FloatSlider if not is_spinner else ipywidgets.FloatText

        kw = dict(
            value=default_val,
            description=parameter.label,
            step=parameter.step if parameter.step is not None else 1,
        )
        if any(t is not None for t in [lb, slb]):
            _min = slb if slb is not None else lb
            step = 1 if is_integer else kw["step"]
            if all(t is not None for t in [lb, slb]):
                _min = max(lb, slb)
            _min += (0 if ilb else step)
            kw["min"] = _min
        if any(t is not None for t in [hb, shb]):
            _max = shb if shb is not None else hb
            step = 1 if is_integer else kw["step"]
            if all(t is not None for t in [hb, shb]):
                _max = min(hb, shb)
            _max -= (0 if ihb else step)
            kw["max"] = _max
        if is_spinner and (("min" in kw) or ("max" in kw)):
            # NOTE:
            # currently, ipywidgets doesn't support half-bounded spinners.
            # So, if an integer parameter has bounds=(None, something) or
            # bounds=(something, None), then an ipywidgets.BoundedIntText
            # will be created, but with a wrong bound. The None will be
            # replaced with some number.
            if is_integer:
                type_ = ipywidgets.BoundedIntText
            else:
                type_ = ipywidgets.BoundedFloatText
            k = 1000
            if "min" not in kw:
                kw["min"] = -k * abs(kw["max"])
            if "max" not in kw:
                kw["max"] = k * abs(kw["min"])
        widget = type_(**kw)

    elif isinstance(parameter, param.Range):
        v1, v2 = default_val
        bounds = parameter.bounds
        ibounds = parameter.inclusive_bounds
        softbounds = parameter.softbounds
        lb, hb = bounds if (bounds is not None) else (None, None)
        ilb, ihb = ibounds if (ibounds is not None) else (True, True)
        slb, shb = softbounds if (softbounds is not None) else (None, None)

        step = parameter.step if parameter.step is not None else 1
        is_int = lambda t: (t is None) or isinstance(t, int)
        is_integer = all(is_int(t) for t in [lb, hb, slb, shb, v1, v2, step])
        type_ = ipywidgets.IntRangeSlider if is_integer else ipywidgets.FloatRangeSlider

        start = min(v1, slb if slb else (lb if lb else v1))
        end = max(v2, shb if shb else (hb if hb else v2))
        if math.isclose(start, end):
            start -= 10*step
            end += 10*step

        if (parameter.step is None) and (step > abs(start - end)):
            # NOTE: here it is guaranteed that start != end
            # hence, step > 0
            step = abs(start - end) / 10

        if not ilb:
            start += step
        if not ihb:
            end -= step

        kw = dict(
            value=[v1, v2],
            min=start,
            max=end,
            description=parameter.label,
            step=step,
        )
        widget = type_(**kw)

    elif isinstance(parameter, param.Selector):
        widget = ipywidgets.Dropdown(
            description=parameter.label,
            options=list(parameter.objects.values()),
            value=default_val
        )

    else:
        raise NotImplementedError(
            f"`{type(parameter).__name__}` has not been implemented yet.")

    return widget



@modify_parameterized_doc()
class InteractivePlot(IPlot):
    def __init__(self, *series, **kwargs):

        params = kwargs.pop("params", {})
        params = _aggregate_parameters(params, series)

        kwargs.setdefault("ncols", 2)
        kwargs.setdefault("use_latex", cfg["interactive"]["use_latex"])
        kwargs.setdefault("layout", "tb")
        kwargs.setdefault("_original_params", params)

        # remove keyword arguments that are not parameters of this backend
        kwargs_for_init = {k: v for k, v in kwargs.items() if k in list(self.param)}

        super().__init__(**kwargs_for_init)

        # map symbols to widgets
        self._params_widgets = {
            k: v for k, v in zip(
                params.keys(),
                _build_widgets(params, self.use_latex)
        )}
        # additional widgets coming from data series in order to customize
        # the data generation process
        self._additional_widgets = {}

        # bind the update function
        for w in self._params_widgets.values():
            w.observe(self._update, "value")

        plotgrid = kwargs.get("plotgrid", None)

        if plotgrid:
            self.backend = plotgrid
            self._grid_widgets = _build_grid_layout(
                list(self._params_widgets.values()), self.ncols)
        else:
            additional_widgets = defaultdict(list)
            for i, s in enumerate(series):
                if s.is_interactive:
                    # assure that each series has the correct values
                    # associated to parameters
                    s.params = {
                        k: v.value for k, v in self._params_widgets.items()}
                    if hasattr(s, "_interactive_app_controls"):
                        name = f"{type(s).__name__}-{i}"
                        for k in s._interactive_app_controls:
                            w = _get_widget_from_param_module(s, k)
                            # bind the update function
                            w.observe(self._update, "value")
                            additional_widgets[name].append(w)

            self._additional_widgets = additional_widgets
            accordions = [
                ipywidgets.Accordion(
                    children=[ipywidgets.VBox(v)], titles=[k])
                    for k, v in additional_widgets.items()
            ]
            self._grid_widgets = _build_grid_layout(
                list(self._params_widgets.values()) + accordions,
                self.ncols
            )

            is_3D = all([s.is_3D for s in series])
            Backend = kwargs.pop("backend", THREE_D_B if is_3D else TWO_D_B)
            kwargs["is_iplot"] = True
            kwargs["imodule"] = "ipywidgets"
            kwargs["use_latex"] = self.use_latex
            self.backend = Backend(*series, **kwargs)

    def _get_iplot_kw(self):
        params = {}
        # copy all parameters into the dictionary
        for k in list(self.param):
            params[k] = getattr(self, k)
        return params

    def _update(self, change):
        # update the data series that shows additional widgets
        for name, widgets in self._additional_widgets.items():
            idx = int(name.split("-")[1])
            series = self.backend.series[idx]
            keys = series._interactive_app_controls
            d = {k: widgets[j].value for j, k in enumerate(keys)}
            series.param.update(d)

        # generate new numerical data and update the renderers
        self.backend.update_interactive(
            {k: v.value for k, v in self._params_widgets.items()})

        if isinstance(self.backend, BB):
            bokeh = import_module(
                'bokeh',
                import_kwargs={'fromlist': ['io']},
                warn_not_installed=True,
                min_module_version='2.3.0')
            if hasattr(self, "_output_figure"):
                # NOTE: during testing, this attribute doesn't exist because
                # it is created when `show` is executed, which never happens
                # in tests.
                with self._output_figure:
                    clear_output(True) # NOTE: this is the cause of flickering
                    bokeh.io.show(self.backend.fig)

    def show(self):
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

        if self.layout == "tb":
            return ipywidgets.VBox([self._grid_widgets, self._output_figure])
        elif self.layout == "bb":
            return ipywidgets.VBox([self._output_figure, self._grid_widgets])
        elif self.layout == "sbl":
            return ipywidgets.HBox([self._grid_widgets, self._output_figure])
        return ipywidgets.HBox([self._output_figure, self._grid_widgets])


@modify_graphics_series_doc(InteractivePlot, replace={"params": _PARAMS})
def iplot(*series, show=True, **kwargs):
    """
    Create an interactive application containing widgets and charts in order
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
