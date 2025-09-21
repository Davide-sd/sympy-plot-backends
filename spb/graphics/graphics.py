import param
from sympy import latex, Symbol
from spb.backends.base_backend import Plot, PlotAttributes
from spb.defaults import TWO_D_B, THREE_D_B
from spb.doc_utils.ipython import modify_graphics_doc
from spb.interactive import create_interactive_plot
from spb.series import (
    LineOver1DRangeSeries, Parametric3DLineSeries,
    SurfaceOver2DRangeSeries, ContourSeries,
    ImplicitSeries, Implicit3DSeries, BaseSeries
)
from spb.utils import _instantiate_backend, _check_misspelled_kwargs


# NOTE: why is `graphics` a subclass of param.ParameterizedFunction?
# because it automatically gets the init signature according to
# `PlotAttributes`, which makes it more reliable and easier to update:
# no worries of forgetting to document some parameter.
@modify_graphics_doc(priority=["args"])
class graphics(PlotAttributes, param.ParameterizedFunction):
    """
    Plots a collection of data series.

    Parameters
    ==========

    args :
        Instances of ``BaseSeries`` or lists of instances of ``BaseSeries``.
    app : boolean
        Default to False. If True, shows interactive widgets useful to
        customize the numerical data computation.
        Related parameters: ``imodule``.

    Returns
    =======

    p : Plot or InteractivePlot
        This function returns:

        * an instance of ``InteractivePlot`` if any of the data series is
          interactive (``params`` has been set), or if ``app=True``.
        * an instance of ``Plot`` otherwise.

    Examples
    ========

    Combining together multiple data series of the same type, enabling
    auto-update on pan:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy import *
        >>> from spb import *
        >>> x = symbols("x")
        >>> graphics(
        ...     line(cos(x), label="a"),
        ...     line(sin(x), (x, -pi, pi), label="b"),
        ...     line(log(x), rendering_kw={"linestyle": "--"}),
        ...     title="My title", ylabel="y", update_event=True
        ... )
        Plot object containing:
        [0]: cartesian line: cos(x) for x over (-10.0, 10.0)
        [1]: cartesian line: sin(x) for x over (-3.141592653589793, 3.141592653589793)
        [2]: cartesian line: log(x) for x over (-10.0, 10.0)


    Combining together multiple data series of the different types:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy import *
        >>> from spb import *
        >>> x = symbols("x")
        >>> graphics(
        ...     line((cos(x)+1)/2, (x, -pi, pi), label="a"),
        ...     line(-(cos(x)+1)/2, (x, -pi, pi), label="b"),
        ...     line_parametric_2d(cos(x), sin(x), (x, 0, 2*pi), label="c", use_cm=False),
        ...     title="My title", ylabel="y", aspect="equal"
        ... )
        Plot object containing:
        [0]: cartesian line: cos(x)/2 + 1/2 for x over (-3.141592653589793, 3.141592653589793)
        [1]: cartesian line: -cos(x)/2 - 1/2 for x over (-3.141592653589793, 3.141592653589793)
        [2]: parametric cartesian line: (cos(x), sin(x)) for x over (0.0, 6.283185307179586)


    Set tick labels to be some multiple of `pi`:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> x, y = symbols("x, y")
        >>> expr = 5 * (cos(x) - 0.2 * sin(y))**2 + 5 * (-0.2 * cos(x) + sin(y))**2
        >>> graphics(
        ...     contour(expr, (x, 0, 2 * pi), (y, 0, 2 * pi), fill=False),
        ...     x_ticks_formatter=multiples_of_pi_over_4(),
        ...     y_ticks_formatter=multiples_of_pi_over_3()
        ... )
        Plot object containing:
        [0]: contour: 5*(-0.2*sin(y) + cos(x))**2 + 5*(sin(y) - 0.2*cos(x))**2 for x over (0, 2*pi) and y over (0, 2*pi)


    Use ``hooks`` to further customize the figure before it is shown on the
    screen, for example applying custom tick labels to a colorbar:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> def colorbar_ticks_formatter(plot_object):
        ...     fig, ax = plot_object.fig, plot_object.ax
        ...     cax = fig.axes[1]
        ...     formatter = multiples_of_pi()
        ...     cax.yaxis.set_major_locator(formatter.MB_major_locator())
        ...     cax.yaxis.set_major_formatter(formatter.MB_func_formatter())

        >>> u = symbols("u")
        >>> graphics(
        ...     line_parametric_2d(
        ...         2 * cos(u) + 5 * cos(2 * u / 3),
        ...         2 * sin(u) - 5 * sin(2 * u / 3),
        ...         (u, 0, 6 * pi)
        ...     ),
        ...     hooks=[colorbar_ticks_formatter]
        ... )
        Plot object containing:
        [0]: parametric cartesian line: (5*cos(2*u/3) + 2*cos(u), -5*sin(2*u/3) + 2*sin(u)) for u over (0, 6*pi)


    Plot over an existing figure. Note that:

    * If an existing Matplotlib's figure is available, users can specify one
      of the following keyword arguments:

        * ``fig=`` to provide the existing figure. The module will then plot
          the symbolic expressions over the first Matplotlib's axes.
        * ``ax=`` to provide the Matplotlib's axes over which symbolic
          expressions will be plotted. This is useful if users have a figure
          with multiple subplots.
    * If an existing Bokeh/Plotly/K3D's figure is available, user should
      pass the following keyword arguments: ``fig=`` for the existing figure
      and ``backend=`` to specify which backend should be used.
    * This module will override axis labels, title, and grid.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy import symbols, cos, pi
        >>> from spb import *
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> # plot some numerical data
        >>> fig, ax = plt.subplots()
        >>> xx = np.linspace(-np.pi, np.pi, 20)
        >>> yy = np.cos(xx)
        >>> noise = (np.random.random_sample(len(xx)) - 0.5) / 5
        >>> yy = yy * (1+noise)
        >>> ax.scatter(xx, yy, marker="*", color="m")  # doctest: +SKIP
        >>> # plot a symbolic expression
        >>> x = symbols("x")
        >>> graphics(
        ...     line(cos(x), (x, -pi, pi), rendering_kw={"ls": "--", "lw": 0.8}),
        ...     ax=ax, update_event=True)
        Plot object containing:
        [0]: cartesian line: cos(x) for x over (-3.141592653589793, 3.141592653589793)


    Interactive-widget plot combining together data series of different types:

    .. panel-screenshot::

        from sympy import *
        from spb import *
        import k3d
        a, b, s, e, t = symbols("a, b, s, e, t")
        c = 2 * sqrt(a * b)
        r = a + b
        params = {
            a: (1.5, 0, 2),
            b: (1, 0, 2),
            s: (0, 0, 2),
            e: (2, 0, 2)
        }
        graphics(
            surface_revolution(
                (r * cos(t), r * sin(t)), (t, 0, pi),
                params=params, n=50, parallel_axis="x", show_curve=False,
                rendering_kw={"color":0x353535},
                force_real_eval=True
            ),
            line_parametric_3d(
                a * cos(t) + b * cos(3 * t),
                a * sin(t) - b * sin(3 * t),
                c * sin(2 * t), prange(t, s*pi, e*pi),
                rendering_kw={"color_map": k3d.matplotlib_color_maps.Summer},
                params=params
            ),
            backend=KB
        )

    Interactive widget plot, showing widgets related to data series
    that allows to easily customize the data generation process:

    .. panel-screenshot::
        :small-size: 1000, 550

        from sympy import *
        from spb import *
        z = symbols("z")

        graphics(
            domain_coloring(sin(z), (z, -2-2j, 2+2j), coloring="b"),
            backend=MB,
            grid=False,
            layout="sbl",
            ncols=1,
            template={"sidebar_width": "30%"},
            app=True
        )

    See Also
    ========

    plotgrid
    """

    def __call__(self, *args, **params):
        p = param.ParamOverrides(self, {})

        series = []
        for a in args:
            if (isinstance(a, (list, tuple)) and
                all(isinstance(s, BaseSeries) for s in a)):
                series.extend(a)
            elif isinstance(a, BaseSeries):
                series.append(a)
            else:
                raise TypeError(
                    "Only instances of ``BaseSeries`` or lists of "
                    "instances of ``BaseSeries`` are supported. Received: "
                    f"{type(a)}")

        is_3D = any(s.is_3D for s in series)
        params.setdefault("backend", TWO_D_B if is_3D else THREE_D_B)

        # allow data series to show their UI controls
        app = params.pop("app", False)

        # TODO: this can be done without the params of this class, using instead
        # the params of Plot
        keys_to_be_aware_of = [
            "process_piecewise", "backend", "show", "fig", "ax",
            # this enables animations
            "animation",
            # these enable interactive widgets plotting
            "pane_kw", "ncols", "layout", "template", "servable",
            "plot_function"
        ]
        # remove keyword arguments that are not parameters of this backend
        keys_to_maintain = list(Plot.param) + keys_to_be_aware_of

        if not params.get("plot_function", False):
            _check_misspelled_kwargs(
                self, additional_keys=keys_to_be_aware_of, **params)
        params = {k: v for k, v in params.items() if k in keys_to_maintain}

        # set the appropriate transformation on 2D line series if polar axis
        # are requested
        if params.get("polar_axis", False):
            for s in series:
                if s.is_2Dline:
                    s.is_polar = True

        # set axis labels
        if all(isinstance(s, LineOver1DRangeSeries) for s in series):
            fs = set([s.ranges[0][0] for s in series])
            if len(fs) == 1:
                x = fs.pop()
                fx = lambda use_latex: x.name if not use_latex else latex(x)
                wrap = lambda use_latex: "f(%s)" if not use_latex else r"f\left(%s\right)"
                fy = lambda use_latex: wrap(use_latex) % fx(use_latex)
                params.setdefault("xlabel", fx)
                params.setdefault("ylabel", fy)
        elif (
            all(isinstance(s, (ContourSeries, SurfaceOver2DRangeSeries))
                for s in series) or
            (all(isinstance(s, (SurfaceOver2DRangeSeries, Parametric3DLineSeries))
                for s in series) and all(s.label == "__k__" for s in series
                if isinstance(s, Parametric3DLineSeries)))
            ):
            free_x = set([
                s.ranges[0][0] for s in series
                if isinstance(s, (ContourSeries, SurfaceOver2DRangeSeries))])
            free_y = set([
                s.ranges[1][0] for s in series
                if isinstance(s, (ContourSeries, SurfaceOver2DRangeSeries))])
            if all(len(t) == 1 for t in [free_x, free_y]):
                x = free_x.pop() if free_x else Symbol("x")
                y = free_y.pop() if free_y else Symbol("y")
                fx = lambda use_latex: x.name if not use_latex else latex(x)
                fy = lambda use_latex: y.name if not use_latex else latex(y)
                wrap = lambda use_latex: "f(%s, %s)" if not use_latex else r"f\left(%s, %s\right)"
                fz = lambda use_latex: wrap(use_latex) % (fx(use_latex), fy(use_latex))
                params.setdefault("xlabel", fx)
                params.setdefault("ylabel", fy)
                params.setdefault("zlabel", fz)
        elif all(isinstance(s, Implicit3DSeries) for s in series):
            free_x = set([s.ranges[0][0] for s in series])
            free_y = set([s.ranges[1][0] for s in series])
            free_z = set([s.ranges[2][0] for s in series])
            if all(len(t) == 1 for t in [free_x, free_y, free_z]):
                fx = lambda use_latex: free_x.pop().name if not use_latex else latex(free_x.pop())
                fy = lambda use_latex: free_y.pop().name if not use_latex else latex(free_y.pop())
                fz = lambda use_latex: free_z.pop().name if not use_latex else latex(free_z.pop())
                params.setdefault("xlabel", fx)
                params.setdefault("ylabel", fy)
                params.setdefault("zlabel", fz)
        elif all(isinstance(s, ImplicitSeries) for s in series):
            free_x = set([s.ranges[0][0] for s in series])
            free_y = set([s.ranges[1][0] for s in series])
            if all(len(t) == 1 for t in [free_x, free_y]):
                fx = lambda use_latex: free_x.pop().name if not use_latex else latex(free_x.pop())
                fy = lambda use_latex: free_y.pop().name if not use_latex else latex(free_y.pop())
                params.setdefault("xlabel", fx)
                params.setdefault("ylabel", fy)

        from spb.backends.matplotlib.matplotlib import MB
        if not issubclass(params.get("backend"), MB):
            params.pop("ax", None)

        if any(s.is_interactive for s in series) or app:
            return create_interactive_plot(*series, **params)

        Backend = params.pop("backend", TWO_D_B if is_3D else THREE_D_B)
        return _instantiate_backend(Backend, *series, **params)

