from spb.defaults import cfg
from sympy.external import import_module
from spb.backends.base_backend import Plot
from spb.backends.matplotlib import MB
from spb.backends.plotly import PB
from spb.backends.bokeh import BB
from spb.interactive import IPlot, create_interactive_plot
from sympy.utilities.exceptions import sympy_deprecation_warning
from IPython.display import clear_output


# NOTE: the code in this module, particularly the one about interactive widget
# plot, is ugly and probably difficult to comprehend. Turns out that it is
# extremely difficult to get ipywidgets (and in much less extent, panel) to
# work with different plotting libraries...


def _nrows_ncols(nr, nc, nplots):
    """Define the correct number of rows and/or columns based on the number
    of plots to be shown.
    """
    np = import_module('numpy')

    if (nc <= 0) and (nr <= 0):
        nc = 1
        nr = nplots
    elif nr <= 0:
        nr = int(np.ceil(nplots / nc))
    elif nc <= 0:
        nc = int(np.ceil(nplots / nr))
    elif nr == 1:
        nc = nplots
    elif nr * nc < nplots:
        nr += 1
        return _nrows_ncols(nr, nc, nplots)
    return nr, nc


def _are_all_plots_instances_of(plots, Backend):
    """Verify that plots (or interactive plots) are produces with the
    specified backend.
    """
    return all(isinstance(t, Backend)  or
        (isinstance(t, IPlot) and isinstance(t.backend, Backend))
        for t in plots)


def _create_mpl_figure(mapping, imagegrid=False, size=None, is_iplot_panel=False):
    matplotlib = import_module(
        'matplotlib',
        import_kwargs={'fromlist': ['pyplot', 'gridspec']},
        min_module_version='1.1.0',
        catch=(RuntimeError,))
    mpl_toolkits = import_module(
        'mpl_toolkits',
        import_kwargs={'fromlist': ['axes_grid1']},
        catch=(RuntimeError,))
    plt = matplotlib.pyplot

    def get_fig_panes_plots(fig):
        panes_plots = {}
        if is_iplot_panel:
            pn = import_module(
                'panel',
                min_module_version='0.12.0')
            pane = pn.pane.Matplotlib(fig, dpi=96)
            panes_plots[pane] = fig
            return pane, panes_plots
        return fig, panes_plots

    kw = {} if not size else {"figsize": size}
    if is_iplot_panel:
        fig = matplotlib.figure.Figure(**kw)
    else:
        fig = plt.figure(**kw)

    new_plots = []
    panes_plots = {}
    if imagegrid:
        gs =list(mapping.keys())[0].get_gridspec()
        grid = mpl_toolkits.axes_grid1.ImageGrid(
            fig, 111,
            nrows_ncols=(gs.nrows, gs.ncols),
            axes_pad=0.15,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="7%",
            cbar_pad=0.15,
        )
        for (_, p), ax in zip(mapping.items(), grid):
            if isinstance(p, IPlot):
                p = p.backend
            # cpa: current plot attributes
            cpa = p._copy_kwargs()
            cpa["fig"] = fig
            cpa["ax"] = ax
            cpa["imagegrid"] = True
            p = MB(*p.series, **cpa)
            p.draw()
            new_plots.append(p)
        fig, panes_plots = get_fig_panes_plots(fig)
        return fig, new_plots, panes_plots

    for spec, p in mapping.items():
        if isinstance(p, IPlot):
            p = p.backend
        kw = {"projection": "3d"} if (len(p.series) > 0 and
            p.series[0].is_3D) else ({"projection": "polar"} if p.polar_axis
            else {})
        cur_ax = fig.add_subplot(spec, **kw)
        # cpa: current plot attributes
        cpa = p._copy_kwargs()
        cpa["fig"] = fig
        cpa["ax"] = cur_ax
        p = MB(*p.series, **cpa)
        p.draw()
        new_plots.append(p)

    fig.tight_layout()
    fig, panes_plots = get_fig_panes_plots(fig)
    return fig, new_plots, panes_plots


def _create_panel_figure(mapping, panel_kw):
    pn = import_module(
        'panel',
        min_module_version='0.12.0')
    pn.extension("plotly")

    panes_plots = {}
    fig = pn.GridSpec(**panel_kw)
    for spec, p in mapping.items():
        rs = spec.rowspan
        cs = spec.colspan
        if isinstance(p, IPlot):
            # a panel's `pane` must receive a figure of some kind, not another
            # panel's object, otherwise there will be performance penalty,
            # especially noticeable with Plotly and Bokeh
            p = p.backend
        _fig = p.fig
        if isinstance(p, PB):
            pane = pn.pane.Plotly(_fig.to_dict())
            fig[slice(rs.start, rs.stop), slice(cs.start, cs.stop)] = pane
        else:
            pane = pn.pane.panel(_fig)
            fig[slice(rs.start, rs.stop), slice(cs.start, cs.stop)] = pane
        panes_plots[pane] = p
    return fig, panes_plots


def _create_ipywidgets_figure(mapping, panel_kw):
    ipy = import_module('ipywidgets')
    plotly = import_module(
        'plotly',
        import_kwargs={'fromlist': ['graph_objects']},
        warn_not_installed=True,
        min_module_version='5.0.0')
    go = plotly.graph_objects

    fig = ipy.GridspecLayout(**panel_kw)
    bokeh_outputs_plots = []
    for spec, p in mapping.items():
        rs = spec.rowspan
        cs = spec.colspan
        plot_fig = p.fig
        if isinstance(p, PB):
            # ipywidgets requires Plotly's FigureWidget
            plot_fig = go.FigureWidget(p.fig.to_dict())
        elif _are_all_plots_instances_of([p], BB):
            bokeh = import_module(
                'bokeh',
                import_kwargs={'fromlist': ['io']},
                warn_not_installed=True,
                min_module_version='2.3.0')
            # let's assume cfg["bokeh"]["height"] is an integer
            min_height = str(cfg["bokeh"]["height"]) + "px"
            new_fig = ipy.Output(layout=ipy.Layout(
                height='auto', min_height=min_height, width='100%', max_width="100%"))
            with new_fig:
                bokeh.io.show(plot_fig)
            bokeh_outputs_plots.append((new_fig, p))
            plot_fig = new_fig
        fig[slice(rs.start, rs.stop), slice(cs.start, cs.stop)] = ipy.Box([plot_fig])
    return fig, bokeh_outputs_plots


def _check_gs(gs):
    """Helper function to verify the provided GridSpec.
    """
    if not isinstance(gs, dict):
        raise TypeError("`gs` must be a dictionary.")

    matplotlib = import_module(
        'matplotlib',
        import_kwargs={'fromlist': ['pyplot', 'gridspec']},
        min_module_version='1.1.0',
        catch=(RuntimeError,))

    SubplotSpec = matplotlib.gridspec.SubplotSpec
    if not isinstance(list(gs.keys())[0], SubplotSpec):
        raise ValueError(
            "Keys of `gs` must be of elements of type "
            "matplotlib.gridspec.SubplotSpec. Use "
            "matplotlib.gridspec.GridSpec to create them.")


def _get_all_parameters(plots):
    """Loop over the provided plots and extract the original parameters.
    """
    all_parameters, all_plots = {}, []
    for plot in plots:
        if isinstance(plot, IPlot):
            all_plots.append(plot.backend)
            # all_plots.append(plot)
            all_parameters.update(plot._original_params)
        else:
            all_plots.append(plot)

    return all_parameters, all_plots


def _get_plots_imodule(plots):
    """Verify that all plots uses the same interactive module, and return it.
    """
    imodules = set()
    for plot in plots:
        if isinstance(plot, IPlot):
            imodules.add(plot.backend.imodule)
        else:
            imodules.add(plot.imodule)

    if None in imodules:
        imodules.remove(None)

    if len(imodules) > 1:
        raise ValueError(
            "The provided interactive plots uses different interactive "
            "modules. This is not supported. Please, only chose one "
            "interactive module for all plots.\n"
            f"Received interactive modules: {imodules}")
    return imodules.pop() if len(imodules) > 0 else None


def _check_imodules(plots_imodule, plotgrid_imodule):
    def raise_error(plots_imodule, plotgrid_imodule):
        raise ValueError(
            "The interactive module used by `plotgrid` is different from "
            "the interactive module used by the plots. This is not supported. "
            "Please, only chose one interactive module. Received:\n"
            f"plotgrid imodule={plotgrid_imodule}\n"
            f"plots imodule={plots_imodule}"
        )
    default_imodule = cfg["interactive"]["module"]
    if (plots_imodule is None) and (plotgrid_imodule is None):
        pass
    elif plots_imodule is None:
        if plotgrid_imodule != default_imodule:
            raise_error(default_imodule, plotgrid_imodule)
    elif plotgrid_imodule is None:
        if plots_imodule != default_imodule:
            raise_error(plots_imodule, default_imodule)
    elif plotgrid_imodule != plots_imodule:
        raise_error(plots_imodule, plotgrid_imodule)


def plotgrid(*args, **kwargs):
    """Combine multiple plots into a grid-like layout.
    This function has two modes of operation, depending on the input arguments.
    Make sure to read the examples to fully understand them.

    Parameters
    ==========

    args : sequence
        A sequence of aldready created plots. This, in combination with
        ``nr`` and ``nc`` represents the first mode of operation, where a
        basic grid with (nc * nr) subplots will be created.

    nr, nc : int, optional
        Number of rows and columns.
        By default, ``nc = 1`` and ``nr = -1``: this will create as many rows
        as necessary, and a single column.
        By setting ``nr = 1`` a grid with a single row and as many columns as
        necessary will be created.

    gs : dict, optional
        A dictionary mapping Matplotlib's ``GridSpec`` objects to the plots.
        The keys represent the cells of the layout. Each cell will host the
        associated plot.
        This represents the second mode of operation, as it allows to create
        more complicated layouts.

    imagegrid : boolean, optional
        Requests Matplotlib's ``ImageGrid`` axes [#fn2]_ to be used. This is
        best suited for plots with equal aspect ratio sharing a common
        colorbar. Default to False.

    panel_kw : dict, optional
        A dictionary of keyword arguments to be passed to panel's ``GridSpec``
        for further customization. Default to
        ``dict(sizing_mode="stretch_width")``. Refer to [#fn1]_ for more
        information.

    show : boolean, optional
        It applies only to Matplotlib figures. Default to True.

    Returns
    =======

    fig : ``plt.Figure`` or ``pn.GridSpec``
        If all input plots are instances of ``MatplotlibBackend``, than a
        Matplotlib ``Figure`` will be returned. Otherwise an instance of
        Holoviz Panel's ``GridSpec`` will be returned.


    Examples
    ========

    First mode of operation with instances of ``MatplotlibBackend``:

    .. plot::
       :include-source: True
       :context: reset

       from sympy import symbols, sin, cos, tan, exp, sqrt, Matrix, gamma, I
       from spb import *

       x, y, z = symbols("x, y, z")
       p1 = plot(sin(x), backend=MB, title="sin(x)", show=False)
       p2 = plot(tan(x), backend=MB, adaptive=False, detect_poles=True,
            title="tan(x)", show=False)
       p3 = plot(exp(-x), backend=MB, title="exp(-x)", show=False)
       plotgrid(p1, p2, p3)

    When plots represents images with equal aspect ratio and common
    colorbar, set ``imagegrid=True``:

    .. plot::
       :include-source: True
       :context: reset

       from sympy import symbols, sin, cos, pi, I
       from spb import *
       z = symbols("z")
       options = dict(coloring="b", show=False, grid=False)
       p1 = plot_complex(sin(z), (z, -pi-pi*I, pi+pi*I), **options)
       p2 = plot_complex(cos(z), (z, -pi-pi*I, pi+pi*I), **options)
       plotgrid(p1, p2, nr=1, imagegrid=True)

    Second mode of operation, using Matplotlib ``GridSpec``:

    .. plot::
       :include-source: True
       :context: reset

       from sympy import *
       from spb import *
       from matplotlib.gridspec import GridSpec
       x, y, z = symbols("x, y, z")
       p1 = plot(sin(x), cos(x), adaptive=False, show=False)
       expr = Tuple(1, sin(x**2 + y**2))
       p2 = plot_vector(expr, (x, -2, 2), (y, -2, 2),
            streamlines=True, scalar=False, use_cm=False,
            title=r"$\\vec{F}(x, y) = %s$" % latex(expr),
            xlabel="x", ylabel="y", show=False)
       p3 = plot_complex(gamma(z), (z, -3-3*I, 3+3*I), title=r"$\gamma(z)$",
            grid=False, show=False)

       gs = GridSpec(3, 4)
       mapping = {
           gs[2, :]: p1,
           gs[0:2, 0:2]: p2,
           gs[0:2, 2:]: p3,
       }
       plotgrid(gs=mapping)

    Interactive-widget plotgrid with first mode of operation, illustrating:

    * ``plotgrid`` accepts interactive plots.
    * the use of the ``prange`` class (parametric range).
    * the same interactive module, ``imodule``, must be used on the plots as
      well as on the plotgrid. Here, ``imodule="panel"`` has been used, but
      users can change it to ``imodule="ipywidgets"``, provided that
      ``%matplotlib widget`` is executed first.

    .. panel-screenshot::
       :small-size: 800, 675

       from sympy import *
       from spb import *
       from sympy.abc import a, b, c, d, x
       imodule = "panel"
       options = dict(
           imodule=imodule, show=False, use_latex=False, params={
               a: (1, 0, 2),
               b: (5, 0, 10),
               c: (0, 0, 2*pi),
               d: (10, 1, 20)
           })

       p1 = plot(sin(x*a + c) * exp(-abs(x) / b), prange(x, -d, d), **options)
       p2 = plot(cos(x*a + c) * exp(-abs(x) / b), (x, -10, 10), **options)
       plotgrid(p1, p2, imodule=imodule)

    References
    ==========

    .. [#fn1] https://panel.holoviz.org/reference/layouts/GridSpec.html
    .. [#fn2] https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axes_grid1.axes_grid.ImageGrid.html

    """

    nr = kwargs.get("nr", -1)
    nc = kwargs.get("nc", 1)
    nr, nc = _nrows_ncols(nr, nc, len(args))
    show = kwargs.pop("show", True)
    gs = kwargs.get("gs", None)

    all_parameters = {}
    # TODO: remove new_args
    new_args = []
    if len(args) > 0:
        plots_imodule = _get_plots_imodule(args)
        all_parameters, new_args = _get_all_parameters(args)
    elif gs:
        _check_gs(gs)
        plots = list(gs.values())
        plots_imodule = _get_plots_imodule(plots)
        all_parameters, new_args = _get_all_parameters(plots)
    else:
        plots_imodule = None

    _check_imodules(plots_imodule, kwargs.get("imodule", None))

    is_iplot = len(all_parameters) > 0
    p = PlotGrid(nr, nc, *args, show=False, is_iplot=is_iplot, **kwargs)
    if is_iplot:
        kwargs["plotgrid"] = p
        kwargs["params"] = all_parameters
        kwargs["show"] = show
        return create_interactive_plot(**kwargs)

    if not show:
        return p
    if p.is_matplotlib_fig:
        p.show()
        return p
    return p.show()


class PlotGrid:
    """Implement the logic to create a grid of plots. Refer to ``plotgrid``
    about examples.
    """
    _panel_row_height = 350

    def __init__(self, nrows, ncolumns, *args, **kwargs):
        self.matplotlib = import_module(
            'matplotlib',
            import_kwargs={'fromlist': ['pyplot', 'gridspec']},
            min_module_version='1.1.0',
            catch=(RuntimeError,))
        self.plt = self.matplotlib.pyplot

        self.nrows = nrows
        self.ncolumns = ncolumns
        self.args = args
        self.size = kwargs.get("size", None)
        # requests Matplotlib's ImageGrid axis to be used
        self.imagegrid = kwargs.get("imagegrid", False)
        self._fig = None
        # If args are all instances of MB, than new plots will be created.
        # All of them will share the same figure, but uses a different axes.
        # Need to store the new plots in order to update them, in case of
        # interactive widget plot.
        self._new_plots = []
        # the following is used when imodule="ipywidgets". It maps bokeh
        # outputs to plots, so that bokeh outputs can be reconstructed after
        # the plots have updated their data.
        self._bokeh_outputs_plots = []
        # the following is used when imodule="panel". It maps plots to panes,
        # so that panel can update what is shown on the pane after the plots
        # have updated their data.
        self._panes_plots = {}
        self._is_iplot = kwargs.get("is_iplot", False)
        self._imodule = kwargs.get("imodule", cfg["interactive"]["module"])

        # validate GridSpec, if provided
        self.gs = kwargs.get("gs", None)
        if self.gs:
            _check_gs(self.gs)
            self.is_matplotlib_fig = _are_all_plots_instances_of(
                self.gs.values(), MB)
            self.is_bokeh_fig = _are_all_plots_instances_of(
                self.gs.values(), BB)
        else:
            self.is_matplotlib_fig = _are_all_plots_instances_of(args, MB)
            self.is_bokeh_fig = _are_all_plots_instances_of(args, BB)

        self.panel_kw = kwargs.get("panel_kw", dict())

        if kwargs.get("show", True):
            self.show()

    @property
    def backend(self):
        # TODO: follow sympy doc procedure to create this deprecation
        sympy_deprecation_warning(
            f"`backend` is deprecated. Use `fig` instead.",
            deprecated_since_version="1.12",
            active_deprecations_target='---')

    @property
    def _series(self):
        # TODO: follow sympy doc procedure to create this deprecation
        sympy_deprecation_warning(
            f"`_series` is deprecated.",
            deprecated_since_version="1.12",
            active_deprecations_target='---')

    def close(self):
        """Close the current plot, if it is a Matplotlib figure."""
        self.plt.close(self.fig)

    @property
    def fig(self):
        if self._fig is None:
            self._create_figure()
        return self._fig

    def save(self, path, **kwargs):
        """Save the current plot at the specified location.

        Refer to:

        * [#fn10]_ to visualize all the available keyword arguments when
          saving a Matplotlib figure.
        * [#fn11]_ to visualize all the available keyword arguments when
          saving a Holoviz's Panel object.

        References
        ==========
        .. [#fn10] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
        .. [#fn11] https://panel.holoviz.org/api/panel.viewable.html#panel.viewable.Viewable.save
        """
        if self.is_matplotlib_fig:
            self.fig.savefig(path, **kwargs)
        else:
            self.fig.save(path, **kwargs)

    def _create_figure(self, **kwargs):
        GridSpec = self.matplotlib.gridspec.GridSpec
        gs = self.gs
        is_iplot_panel = self._is_iplot and (self._imodule == "panel")

        if (gs is None) and (len(self.args) == 0):
            self._fig = self.plt.figure()

        elif (gs is None):
            ### First mode of operation
            nr, nc = self.nrows, self.ncolumns
            gs = GridSpec(nr, nc)
            mapping = {}
            c = 0
            for i in range(nr):
                for j in range(nc):
                    if c < len(self.args):
                        mapping[gs[i, j]] = self.args[c]
                    c += 1

            if self.is_matplotlib_fig:
                self._fig, self._new_plots, self._panes_plots = _create_mpl_figure(
                    mapping, self.imagegrid, self.size, is_iplot_panel)
            else:
                size = self.size

                # NOTE: assumimg all plots are of the same backend
                self._new_plots = self.args

                if self._imodule == "panel":
                    self.panel_kw.setdefault("width", 800 if not size else size[0])
                    self.panel_kw.setdefault("height",
                        nr * self._panel_row_height if not size else size[1])
                    self._fig, self._panes_plots = _create_panel_figure(mapping, self.panel_kw)
                else:
                    get_size = lambda t: str(t) + "px" if isinstance(t, int) else t
                    self.panel_kw.setdefault("width", "800px" if not size else get_size(size[0]))
                    if not self.is_bokeh_fig:
                        # NOTE: this doesn't work well with bokeh
                        self.panel_kw.setdefault("height",
                            str(nr * self._panel_row_height) + "px" if not size else get_size(size[1]))
                    self.panel_kw["n_rows"] = nr
                    self.panel_kw["n_columns"] = nc
                    self._fig, self._bokeh_outputs_plots = _create_ipywidgets_figure(
                        mapping, self.panel_kw)

        else:
            ### Second mode of operation
            if self.is_matplotlib_fig:
                self._fig, self._new_plots, self._panes_plots = _create_mpl_figure(
                    gs, self.imagegrid, self.size, is_iplot_panel)
            else:
                for plot in gs.values():
                    if isinstance(plot, IPlot):
                        self._new_plots.append(plot.backend)
                    else:
                        self._new_plots.append(plot)

                if self._imodule == "panel":
                    self._fig, self._panes_plots = _create_panel_figure(gs, self.panel_kw)
                else:
                    first_element = list(gs.keys())[0]
                    mpl_gs = first_element.get_gridspec()
                    self.panel_kw = {
                        "n_rows": mpl_gs.nrows, "n_columns": mpl_gs.ncols}
                    self._fig, self._bokeh_outputs_plots = _create_ipywidgets_figure(
                        gs, self.panel_kw)

    def _action_post_update(self):
        """With Holoviz's Panel, plots are contained into `panes`: they are
        ultimately responsible to update what is shown on the screen.
        This method is executed by the interactive widget plot after all
        subplots have updated their data.
        """
        for pane, plot in self._panes_plots.items():
            pane.param.trigger("object")
            if isinstance(plot, PB):
                pane.object = plot.fig.to_dict()
            elif isinstance(plot, self.matplotlib.figure.Figure):
                pane.object = plot
            else:
                pane.object = plot.fig

    def update_interactive(self, params):
        """Implement the logic to update the data generated by
        interactive-widget plots.

        Parameters
        ==========

        params : dict
            Map parameter-symbols to numeric values.
        """
        for p in self._new_plots:
            if isinstance(p, IPlot):
                p.backend.update_interactive(params)
            else:
                p.update_interactive(params)

        # update bokeh panes if ipywidgets was used to create
        # this visualization
        bokeh = import_module(
            'bokeh',
            import_kwargs={'fromlist': ['io']},
            warn_not_installed=True,
            min_module_version='2.3.0')
        for (bokeh_output, plot) in self._bokeh_outputs_plots:
            with bokeh_output:
                clear_output(True)
                bokeh.io.show(plot.fig)

    def show(self, **kwargs):
        """Display the current plot.

        Parameters
        ==========

        **kwargs : dict
            Keyword arguments to be passed to plt.show() if a Matplotlib
            figure is created.
        """
        if (self._fig is None) or self.is_matplotlib_fig:
            self._create_figure()

        if self.is_matplotlib_fig:
            if not self.imagegrid:
                self._fig.tight_layout()
            self.plt.show(**kwargs)
        else:
            # holoviz's panel object must be shown on an interactive cell
            return self.fig
