import ipywidgets
from sympy import latex
from sympy.external import import_module
from spb.defaults import TWO_D_B, THREE_D_B
from spb.interactive import _tuple_to_dict
from spb import BB, MB
import plotly.graph_objects as go
from bokeh.io import push_notebook
from IPython.display import clear_output


def _build_widgets(params, use_latex=True):
    widgets = []
    for s, v in params.items():
        if hasattr(v, "__iter__") and (not isinstance(v, str)):
            d = _tuple_to_dict(s, v, use_latex=use_latex)
            if d.pop("type") == "linear":
                widgets.append(ipywidgets.FloatSlider(**d))
            else:
                widgets.append(ipywidgets.FloatLogSlider(**d))
        elif isinstance(v, ipywidgets.Widget):
            widgets.append(v)
        else:
            raise ValueError("Cannot build a widget with the provided input:"
                "%s" % v)
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


def iplot(*series, **kwargs):
    """Create an interactive application containing widgets and charts in order
    to study symbolic expressions, using ipywidgets.

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

        1. a widget.
        2. a tuple of the form:
           `(default, min, max, N, tick_format, label, spacing)`
           where:

           - default, min, max : float
                Default value, minimum value and maximum value of the slider,
                respectively. Must be finite numbers.
           - N : int, optional
                Number of steps of the slider.
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
        of `InteractivePlot`, which can later be be shown by calling the
        `show()` method.

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

    .. code-block::

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

    A line plot illustrating how to specify widgets. In particular:

    1. the parameter ``d`` will be rendered as a slider.
    2. the parameter ``n`` is a spinner.
    3. the parameter ``phi`` will be rendered as a slider: note the custom
       number of steps and the custom label.

    .. code-block:: python

       %matplotlib widget
       from sympy import *
       from spb import *
       import ipywidgets

       x, phi, n, d = symbols("x, phi, n, d")
       plot(
           cos(x * n - phi) * exp(-abs(x) * d), (x, -5*pi, 5*pi),
           params={
               d: (0.1, 0, 1),
               n: ipywidgets.BoundedIntText(value=2, min=1, max=10, description="$n$"),
               phi: (0, 0, 2*pi, 50, "$\phi$ [rad]")
           },
           ylim=(-1.25, 1.25))

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
           use_latex = True,
       )

    Notes
    =====

    1. This function is specifically designed to work within Jupyter Notebook
       and requires the ``ipywidgets`` module[#fna]_ .

    2. To update Matplotlib plots, the ``%matplotlib widget`` command must be
       executed at the top of the Jupyter Notebook. It requires the
       installation of the ``ipympl`` module[#fnb]_ .

    3. Differently from Holoviz's Panel, combining together different
       interactive plots is not supported with ipywidgets.

    4. Differently from Holoviz's Panel, combining together different
       interactive plots is not supported with ipywidgets.


    References
    ==========

    .. [#fna] https://ipywidgets.readthedocs.io
    .. [#fnb] https://github.com/matplotlib/ipympl

    """
    params = kwargs.pop("params", dict())
    use_latex = kwargs.get("use_latex", True)
    ncols = kwargs.get("ncols", 2)
    layout = kwargs.get("layout", "tb")

    widgets = _build_widgets(params, use_latex)
    grid_widgets = _build_grid_layout(widgets, ncols)

    # map symbols to widgets
    params_widgets = {k: v for k, v in zip(params.keys(), widgets)}
    # assure that each series has the correct values associated
    # to parameters
    for s in series:
        s.params = {k: v.value for k, v in params_widgets.items()}

    is_3D = all([s.is_3D for s in series])
    Backend = kwargs.pop("backend", THREE_D_B if is_3D else TWO_D_B)
    kwargs["is_iplot"] = True
    kwargs["imodule"] = "ipywidgets"
    backend = Backend(*series, **kwargs)

    if isinstance(backend, MB):
        backend.plt.ioff() # without it there won't be any update
        output_figure = ipywidgets.Box([backend.fig.canvas])
    elif isinstance(backend, BB):
        output_figure = ipywidgets.Output()
        from bokeh.io import show
        with output_figure:
            show(backend.fig)
    else:
        output_figure = backend.fig

    def update(change):
        backend._update_interactive(
            {k: v.value for k, v in params_widgets.items()})
        if isinstance(backend, BB):
            with output_figure:
                clear_output(True)
                show(backend.fig)

    for w in widgets:
        w.observe(update, "value")

    if isinstance(backend, MB):
        backend.plt.ion() # without it there won't be any update

    if layout == "tb":
        return ipywidgets.VBox([grid_widgets, output_figure])
    elif layout == "bb":
        return ipywidgets.VBox([output_figure, grid_widgets])
    elif layout == "sbl":
        return ipywidgets.HBox([grid_widgets, output_figure])
    return ipywidgets.HBox([output_figure, grid_widgets])
