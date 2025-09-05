import param
from sympy import latex, Symbol
from spb.backends.base_backend import Plot, PlotAttributes
from spb.defaults import TWO_D_B, THREE_D_B
from spb.interactive import create_interactive_plot
from spb.series import (
    LineOver1DRangeSeries, Parametric3DLineSeries,
    SurfaceOver2DRangeSeries, ContourSeries,
    ImplicitSeries, Implicit3DSeries, BaseSeries
)
from spb.utils import _instantiate_backend, _check_misspelled_kwargs


# def graphics(
#     *args, aspect=None, axis_center=None, is_polar=None, legend=None,
#     show=True, size=None, title=None, xlabel=None, ylabel=None, zlabel=None,
#     xlim=None, ylim=None, zlim=None, fig=None, ax=None,
#     update_event=None, **kwargs
# ):
#     """Plots a collection of data series.

#     Parameters
#     ==========

#     *args :
#         Instances of ``BaseSeries`` or lists of instances of ``BaseSeries``.
#     ax : matplotlib.axes.Axes
#         An existing Matplotlib's Axes over which the symbolic expressions will
#         be plotted.
#     aspect : (float, float) or str, optional
#         Set the aspect ratio of the plot. The value depends on the backend
#         being used. Read that backend's documentation to find out the
#         possible values.
#     axis_center : (float, float), optional
#         Tuple of two floats denoting the coordinates of the center or
#         {'center', 'auto'}. Only available with ``MatplotlibBackend``.
#     backend : Plot, optional
#         A subclass of ``Plot``, which will perform the rendering.
#         Default to ``MatplotlibBackend``.
#     fig :
#         An existing figure. Be sure to also specify the proper ``backend=``.
#     is_polar : boolean, optional
#         Default to False. If True, requests the backend to use a 2D polar
#         chart, if implemented.
#     legend : bool, optional
#         Show/hide the legend. Default to None (the backend determines when
#         it is appropriate to show it).
#     show : bool, optional
#         The default value is set to ``True``. Set show to ``False`` and
#         the function will not display the plot. The returned instance of
#         the ``Plot`` class can then be used to save or display the plot
#         by calling the ``save()`` and ``show()`` methods respectively.
#     size : (float, float), optional
#         A tuple in the form (width, height) to specify the size of
#         the overall figure. The default value is set to ``None``, meaning
#         the size will be set by the backend.
#     title : str, optional
#         Title of the plot.
#     update_event : bool
#         If True, enable auto-update on panning. Default to False.
#         Some backend may not implement this feature.
#     use_latex : boolean, optional
#         Turn on/off the rendering of latex labels. If the backend doesn't
#         support latex, it will render the string representations instead.
#     xlabel, ylabel, zlabel : str, optional
#         Labels for the x-axis, y-axis or z-axis, respectively.
#     xscale, yscale, zscale : 'linear' or 'log', optional
#         Sets the scaling of the x-axis, y-axis, z-axis, respectively.
#         Default to ``'linear'``. Some backend might not implement this
#         feature.
#     xlim, ylim, zlim : (float, float), optional
#         Denotes the x-axis/y-axis/z-axis limits, respectively, ``(min, max)``.
#     **kwargs :
#         Refer to the documentation of a backend class in order to find
#         more available keyword arguments.

#     Returns
#     =======

#     p : Plot or InteractivePlot
#         If any of the data series is interactive (``params`` has been set)
#         then an instance of ``InteractivePlot`` is returned, otherwise an
#         instance of the ``Plot`` class is returned.

#     Examples
#     ========

#     Combining together multiple data series of the same type, enabling
#     auto-update on pan:

#     .. plot::
#         :context: close-figs
#         :format: doctest
#         :include-source: True

#         >>> from sympy import *
#         >>> from spb import *
#         >>> x = symbols("x")
#         >>> graphics(
#         ...     line(cos(x), label="a"),
#         ...     line(sin(x), (x, -pi, pi), label="b"),
#         ...     line(log(x), rendering_kw={"linestyle": "--"}),
#         ...     title="My title", ylabel="y", update_event=True
#         ... )
#         Plot object containing:
#         [0]: cartesian line: cos(x) for x over (-10.0, 10.0)
#         [1]: cartesian line: sin(x) for x over (-3.141592653589793, 3.141592653589793)
#         [2]: cartesian line: log(x) for x over (-10.0, 10.0)


#     Combining together multiple data series of the different types:

#     .. plot::
#         :context: close-figs
#         :format: doctest
#         :include-source: True

#         >>> from sympy import *
#         >>> from spb import *
#         >>> x = symbols("x")
#         >>> graphics(
#         ...     line((cos(x)+1)/2, (x, -pi, pi), label="a"),
#         ...     line(-(cos(x)+1)/2, (x, -pi, pi), label="b"),
#         ...     line_parametric_2d(cos(x), sin(x), (x, 0, 2*pi), label="c", use_cm=False),
#         ...     title="My title", ylabel="y", aspect="equal"
#         ... )
#         Plot object containing:
#         [0]: cartesian line: cos(x)/2 + 1/2 for x over (-3.141592653589793, 3.141592653589793)
#         [1]: cartesian line: -cos(x)/2 - 1/2 for x over (-3.141592653589793, 3.141592653589793)
#         [2]: parametric cartesian line: (cos(x), sin(x)) for x over (0.0, 6.283185307179586)


#     Plot over an existing figure. Note that:

#     * If an existing Matplotlib's figure is available, users can specify one
#         of the following keyword arguments:

#         * ``fig=`` to provide the existing figure. The module will then plot the
#         symbolic expressions over the first Matplotlib's axes.
#         * ``ax=`` to provide the Matplotlib's axes over which symbolic
#         expressions will be plotted. This is useful if users have a figure with
#         multiple subplots.
#     * If an existing Bokeh/Plotly/K3D's figure is available, user should
#         pass the following keyword arguments: ``fig=`` for the existing figure
#         and ``backend=`` to specify which backend should be used.
#     * This module will override axis labels, title, and grid.

#     .. plot::
#         :context: close-figs
#         :format: doctest
#         :include-source: True

#         >>> from sympy import symbols, cos, pi
#         >>> from spb import *
#         >>> import numpy as np
#         >>> import matplotlib.pyplot as plt
#         >>> # plot some numerical data
#         >>> fig, ax = plt.subplots()
#         >>> xx = np.linspace(-np.pi, np.pi, 20)
#         >>> yy = np.cos(xx)
#         >>> noise = (np.random.random_sample(len(xx)) - 0.5) / 5
#         >>> yy = yy * (1+noise)
#         >>> ax.scatter(xx, yy, marker="*", color="m")  # doctest: +SKIP
#         >>> # plot a symbolic expression
#         >>> x = symbols("x")
#         >>> graphics(
#         ...     line(cos(x), (x, -pi, pi), rendering_kw={"ls": "--", "lw": 0.8}),
#         ...     ax=ax, update_event=True)
#         Plot object containing:
#         [0]: cartesian line: cos(x) for x over (-3.141592653589793, 3.141592653589793)


#     Interactive-widget plot combining together data series of different types:

#     .. panel-screenshot::

#         from sympy import *
#         from spb import *
#         import k3d
#         a, b, s, e, t = symbols("a, b, s, e, t")
#         c = 2 * sqrt(a * b)
#         r = a + b
#         params = {
#             a: (1.5, 0, 2),
#             b: (1, 0, 2),
#             s: (0, 0, 2),
#             e: (2, 0, 2)
#         }
#         graphics(
#             surface_revolution(
#                 (r * cos(t), r * sin(t)), (t, 0, pi),
#                 params=params, n=50, parallel_axis="x", show_curve=False,
#                 rendering_kw={"color":0x353535},
#                 force_real_eval=True
#             ),
#             line_parametric_3d(
#                 a * cos(t) + b * cos(3 * t),
#                 a * sin(t) - b * sin(3 * t),
#                 c * sin(2 * t), prange(t, s*pi, e*pi),
#                 rendering_kw={"color_map": k3d.matplotlib_color_maps.Summer},
#                 params=params
#             ),
#             backend=KB
#         )

#     See Also
#     ========

#     plotgrid

#     """
#     series = []
#     for a in args:
#         if (isinstance(a, (list, tuple)) and
#             all(isinstance(s, BaseSeries) for s in a)):
#             series.extend(a)
#         elif isinstance(a, BaseSeries):
#             series.append(a)
#         else:
#             raise TypeError(
#                 "Only instances of ``BaseSeries`` or lists of "
#                 "instances of ``BaseSeries`` are supported. Received: "
#                 f"{type(a)}")

#     # set the appropriate transformation on 2D line series if polar axis
#     # are requested
#     if kwargs.get("polar_axis", False):
#         for s in series:
#             if s.is_2Dline:
#                 s.is_polar = True

#     # set axis labels
#     if all(isinstance(s, LineOver1DRangeSeries) for s in series):
#         fs = set([s.ranges[0][0] for s in series])
#         if len(fs) == 1:
#             x = fs.pop()
#             fx = lambda use_latex: x.name if not use_latex else latex(x)
#             wrap = lambda use_latex: "f(%s)" if not use_latex else r"f\left(%s\right)"
#             fy = lambda use_latex: wrap(use_latex) % fx(use_latex)
#             kwargs.setdefault("xlabel", fx)
#             kwargs.setdefault("ylabel", fy)
#     elif (
#         all(isinstance(s, (ContourSeries, SurfaceOver2DRangeSeries))
#             for s in series) or
#         (all(isinstance(s, (SurfaceOver2DRangeSeries, Parametric3DLineSeries))
#             for s in series) and all(s.label == "__k__" for s in series
#             if isinstance(s, Parametric3DLineSeries)))
#         ):
#         free_x = set([
#             s.ranges[0][0] for s in series
#             if isinstance(s, (ContourSeries, SurfaceOver2DRangeSeries))])
#         free_y = set([
#             s.ranges[1][0] for s in series
#             if isinstance(s, (ContourSeries, SurfaceOver2DRangeSeries))])
#         if all(len(t) == 1 for t in [free_x, free_y]):
#             x = free_x.pop() if free_x else Symbol("x")
#             y = free_y.pop() if free_y else Symbol("y")
#             fx = lambda use_latex: x.name if not use_latex else latex(x)
#             fy = lambda use_latex: y.name if not use_latex else latex(y)
#             wrap = lambda use_latex: "f(%s, %s)" if not use_latex else r"f\left(%s, %s\right)"
#             fz = lambda use_latex: wrap(use_latex) % (fx(use_latex), fy(use_latex))
#             kwargs.setdefault("xlabel", fx)
#             kwargs.setdefault("ylabel", fy)
#             kwargs.setdefault("zlabel", fz)
#     elif all(isinstance(s, Implicit3DSeries) for s in series):
#         free_x = set([s.ranges[0][0] for s in series])
#         free_y = set([s.ranges[1][0] for s in series])
#         free_z = set([s.ranges[2][0] for s in series])
#         if all(len(t) == 1 for t in [free_x, free_y, free_z]):
#             fx = lambda use_latex: free_x.pop().name if not use_latex else latex(free_x.pop())
#             fy = lambda use_latex: free_y.pop().name if not use_latex else latex(free_y.pop())
#             fz = lambda use_latex: free_z.pop().name if not use_latex else latex(free_z.pop())
#             kwargs.setdefault("xlabel", fx)
#             kwargs.setdefault("ylabel", fy)
#             kwargs.setdefault("zlabel", fz)
#     elif all(isinstance(s, ImplicitSeries) for s in series):
#         free_x = set([s.ranges[0][0] for s in series])
#         free_y = set([s.ranges[1][0] for s in series])
#         if all(len(t) == 1 for t in [free_x, free_y]):
#             fx = lambda use_latex: free_x.pop().name if not use_latex else latex(free_x.pop())
#             fy = lambda use_latex: free_y.pop().name if not use_latex else latex(free_y.pop())
#             kwargs.setdefault("xlabel", fx)
#             kwargs.setdefault("ylabel", fy)

#     if xlabel:
#         kwargs["xlabel"] = xlabel
#     if ylabel:
#         kwargs["ylabel"] = ylabel
#     if zlabel:
#         kwargs["zlabel"] = zlabel

#     if any(s.is_interactive for s in series):
#         return create_interactive_plot(
#             *series,
#             aspect=aspect, axis_center=axis_center, is_polar=is_polar,
#             legend=legend, show=show, size=size, title=title,
#             xlim=xlim, ylim=ylim, zlim=zlim, ax=ax, fig=fig,
#             update_event=update_event, **kwargs)

#     is_3D = any(s.is_3D for s in series)
#     Backend = kwargs.pop("backend", TWO_D_B if is_3D else THREE_D_B)
#     return _instantiate_backend(
#         Backend, *series,
#         aspect=aspect, axis_center=axis_center,
#         is_polar=is_polar, legend=legend, show=show, size=size,
#         title=title, xlim=xlim, ylim=ylim, zlim=zlim, ax=ax, fig=fig,
#         update_event=update_event, **kwargs)





# class graphics(param.ParameterizedFunction):
#     ax = param.Parameter(doc="""
#         An existing Matplotlib's Axes over which the symbolic
#         expressions will be plotted.""")
#     aspect = param.ClassSelector(
#         class_=(tuple, list, str), doc="""
#         Set the aspect ratio of the plot. The value depends on the backend
#         being used. Read that backend's documentation to find out the
#         possible values.""")
#     axis_center = param.ClassSelector(
#         class_=(tuple, list, str), doc="""
#         Tuple of two floats denoting the coordinates of the center or
#         {'center', 'auto'}. Only available with ``MatplotlibBackend``.""")
#     backend = param.ClassSelector(
#         class_=Plot, doc="""
#         A subclass of ``Plot``, which will perform the rendering.
#         Default to ``MatplotlibBackend``.""")
#     fig = param.Parameter(
#         default=None, doc="""
#         An existing figure. Be sure to also specify the proper ``backend=``.""")
#     is_polar = param.Boolean(
#         default=False, doc="""
#         If True, requests the backend to use a 2D polar chart, if implemented.
#         If False, use cartesian axis.""")
#     legend = param.Boolean(
#         default=None, doc="""
#         Show/hide the legend. Default to None (the backend determines when
#         it is appropriate to show it).""")
#     show = param.Boolean(
#         default=True, doc="""
#         Set show to ``False`` and the function will not display the plot.
#         The returned instance of the ``Plot`` class can then be used to save
#         or display the plot by calling the ``save()`` and ``show()`` methods,
#         respectively.""")
#     size = param.Tuple(
#         default=None, length=2, doc="""
#         A tuple in the form (width, height) to specify the size of
#         the overall figure. The default value is set to ``None``, meaning
#         the size will be set by the backend.""")
#     title = param.String(default="", doc="Title of the plot.")
#     update_event = param.Boolean(
#         default=False, doc="""
#         If True, enable auto-update on panning. Some backend may not implement
#         this feature.""")
#     use_latex = param.Boolean(
#         default=True, doc="""
#         Turn on/off the rendering of latex labels. If the backend doesn't
#         support latex, it will render the string representations instead.""")
#     xlabel = param.String(
#         default="", doc="Label for the x-axis")
#     ylabel = param.String(
#         default="", doc="Label for the y-axis")
#     zlabel = param.String(
#         default="", doc="Label for the z-axis")
#     xscale = param.Selector(
#         default="linear", objects=["linear", "log"], doc="""
#         Sets the scaling of the x-axis. Some backend might not implement this
#         the 'log' scale.""")
#     yscale = param.Selector(
#         default="linear", objects=["linear", "log"], doc="""
#         Sets the scaling of the y-axis. Some backend might not implement this
#         the 'log' scale.""")
#     zscale = param.Selector(
#         default="linear", objects=["linear", "log"], doc="""
#         Sets the scaling of the z-axis. Some backend might not implement this
#         the 'log' scale.""")
#     xlim = param.Tuple(
#         default=None, length=2, doc="""
#         Denotes the x-axis limits.""")
#     ylim = param.Tuple(
#         default=None, length=2, doc="""
#         Denotes the y-axis limits.""")
#     zlim = param.Tuple(
#         default=None, length=2, doc="""
#         Denotes the z-axis limits.""")

#     def __call__(self, *args, **params):
#         series = []
#         for a in args:
#             if (isinstance(a, (list, tuple)) and
#                 all(isinstance(s, BaseSeries) for s in a)):
#                 series.extend(a)
#             elif isinstance(a, BaseSeries):
#                 series.append(a)
#             else:
#                 raise TypeError(
#                     "Only instances of ``BaseSeries`` or lists of "
#                     "instances of ``BaseSeries`` are supported. Received: "
#                     f"{type(a)}")

#         is_3D = any(s.is_3D for s in series)
#         params.setdefault("backend", TWO_D_B if is_3D else THREE_D_B)

#         # TODO: this can be done without the params of this class, using instead
#         # the params of Plot
#         # remove keyword arguments that are not parameters of this backend
#         keys_to_maintain = list(self.param) + ["process_piecewise", "imodule", "polar_axis"]
#         keys_to_maintain = list(Plot.param) + ["process_piecewise", "backend"]


#         print("graphics params", list(self.param))
#         print("Plot params", list(Plot.param))
#         params = {k: v for k, v in params.items() if k in keys_to_maintain}

#         p = param.ParamOverrides(self, params)

#         # set the appropriate transformation on 2D line series if polar axis
#         # are requested
#         if params.get("polar_axis", False):
#             for s in series:
#                 if s.is_2Dline:
#                     s.is_polar = True

#         # set axis labels
#         if all(isinstance(s, LineOver1DRangeSeries) for s in series):
#             fs = set([s.ranges[0][0] for s in series])
#             if len(fs) == 1:
#                 x = fs.pop()
#                 fx = lambda use_latex: x.name if not use_latex else latex(x)
#                 wrap = lambda use_latex: "f(%s)" if not use_latex else r"f\left(%s\right)"
#                 fy = lambda use_latex: wrap(use_latex) % fx(use_latex)
#                 params.setdefault("xlabel", fx)
#                 params.setdefault("ylabel", fy)
#         elif (
#             all(isinstance(s, (ContourSeries, SurfaceOver2DRangeSeries))
#                 for s in series) or
#             (all(isinstance(s, (SurfaceOver2DRangeSeries, Parametric3DLineSeries))
#                 for s in series) and all(s.label == "__k__" for s in series
#                 if isinstance(s, Parametric3DLineSeries)))
#             ):
#             free_x = set([
#                 s.ranges[0][0] for s in series
#                 if isinstance(s, (ContourSeries, SurfaceOver2DRangeSeries))])
#             free_y = set([
#                 s.ranges[1][0] for s in series
#                 if isinstance(s, (ContourSeries, SurfaceOver2DRangeSeries))])
#             if all(len(t) == 1 for t in [free_x, free_y]):
#                 x = free_x.pop() if free_x else Symbol("x")
#                 y = free_y.pop() if free_y else Symbol("y")
#                 fx = lambda use_latex: x.name if not use_latex else latex(x)
#                 fy = lambda use_latex: y.name if not use_latex else latex(y)
#                 wrap = lambda use_latex: "f(%s, %s)" if not use_latex else r"f\left(%s, %s\right)"
#                 fz = lambda use_latex: wrap(use_latex) % (fx(use_latex), fy(use_latex))
#                 params.setdefault("xlabel", fx)
#                 params.setdefault("ylabel", fy)
#                 params.setdefault("zlabel", fz)
#         elif all(isinstance(s, Implicit3DSeries) for s in series):
#             free_x = set([s.ranges[0][0] for s in series])
#             free_y = set([s.ranges[1][0] for s in series])
#             free_z = set([s.ranges[2][0] for s in series])
#             if all(len(t) == 1 for t in [free_x, free_y, free_z]):
#                 fx = lambda use_latex: free_x.pop().name if not use_latex else latex(free_x.pop())
#                 fy = lambda use_latex: free_y.pop().name if not use_latex else latex(free_y.pop())
#                 fz = lambda use_latex: free_z.pop().name if not use_latex else latex(free_z.pop())
#                 params.setdefault("xlabel", fx)
#                 params.setdefault("ylabel", fy)
#                 params.setdefault("zlabel", fz)
#         elif all(isinstance(s, ImplicitSeries) for s in series):
#             free_x = set([s.ranges[0][0] for s in series])
#             free_y = set([s.ranges[1][0] for s in series])
#             if all(len(t) == 1 for t in [free_x, free_y]):
#                 fx = lambda use_latex: free_x.pop().name if not use_latex else latex(free_x.pop())
#                 fy = lambda use_latex: free_y.pop().name if not use_latex else latex(free_y.pop())
#                 params.setdefault("xlabel", fx)
#                 params.setdefault("ylabel", fy)

#         if p.xlabel:
#             params["xlabel"] = p.xlabel
#         if p.ylabel:
#             params["ylabel"] = p.ylabel
#         if p.zlabel:
#             params["zlabel"] = p.zlabel

#         print("graphics before", params)

#         # copy all parameters into the dictionary
#         for k in list(p.param):
#             if (k != "name") and (k not in params):
#                 params[k] = getattr(p, k)

#         # TODO: can this be done better? Like, not using _library?
#         if params.get("backend")._library != "matplotlib":
#             params.pop("ax")

#         print("graphics after", params)

#         if any(s.is_interactive for s in series):
#             return create_interactive_plot(*series, **params)

#         Backend = params.pop("backend", TWO_D_B if is_3D else THREE_D_B)
#         return _instantiate_backend(Backend, *series, **params)





class graphics(PlotAttributes, param.ParameterizedFunction):
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

        # print("graphics", params)

        # TODO: this can be done without the params of this class, using instead
        # the params of Plot
        keys_to_be_aware_of = [
            "process_piecewise", "backend", "show", "fig", "ax",
            # this enables animations
            "animation",
            # these enable interactive widgets plotting
            "pane_kw", "ncols", "layout", "template",
            "plot_function"
        ]
        # remove keyword arguments that are not parameters of this backend
        keys_to_maintain = list(Plot.param) + keys_to_be_aware_of

        # possible_mispelled_keys = [
        #     k for k in params in k not in keys_to_maintain
        # ]
        if not params.get("plot_function", False):
            _check_misspelled_kwargs(
                self, additional_keys=keys_to_be_aware_of, **params)
        params = {k: v for k, v in params.items() if k in keys_to_maintain}

        print("graphics params", params)
        # p = param.ParamOverrides(self, params)
        # print("graphics", params)

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

        # if p.xlabel:
        #     params["xlabel"] = p.xlabel
        # if p.ylabel:
        #     params["ylabel"] = p.ylabel
        # if p.zlabel:
        #     params["zlabel"] = p.zlabel

        # print("graphics before", params)

        # # copy all parameters into the dictionary
        # for k in list(p.param):
        #     if (k != "name") and (k not in params):
        #         params[k] = getattr(p, k)

        from spb.backends.matplotlib.matplotlib import MB
        if not issubclass(params.get("backend"), MB):
            params.pop("ax", None)

        # print("graphics after", params)

        if any(s.is_interactive for s in series):
            return create_interactive_plot(*series, **params)

        Backend = params.pop("backend", TWO_D_B if is_3D else THREE_D_B)
        return _instantiate_backend(Backend, *series, **params)

