import os
from spb.defaults import cfg
from spb.doc_utils.ipython import modify_parameterized_doc
from spb.backends.base_backend import Plot
from spb.backends.utils import tick_formatter_multiples_of
from spb.backends.bokeh.renderers import (
    Line2DRenderer, Vector2DRenderer, ComplexRenderer, ContourRenderer,
    GeometryRenderer, GenericRenderer, HVLineRenderer, Arrow2DRenderer,
    ZGridLineRenderer, SGridLineRenderer, NGridLineRenderer,
    MCirclesRenderer, PoleZeroRenderer, RootLocusRenderer, NyquistRenderer,
    NicholsLineRenderer
)
from spb.series import (
    LineOver1DRangeSeries, List2DSeries, Parametric2DLineSeries,
    ColoredLineOver1DRangeSeries, AbsArgLineSeries, ComplexPointSeries,
    Vector2DSeries, ComplexDomainColoringSeries, ContourSeries,
    Geometry2DSeries, GenericDataSeries, HVLineSeries, Arrow2DSeries,
    ZGridLineSeries, SGridLineSeries, NGridLineSeries, NicholsLineSeries,
    MCirclesSeries, PoleZeroSeries, PoleZeroWithSympySeries,
    SystemResponseSeries, ColoredSystemResponseSeries, RootLocusSeries,
    NyquistLineSeries, HLineSeries, VLineSeries
)
from spb.utils import get_environment
from sympy.external import import_module


@modify_parameterized_doc()
class BokehBackend(Plot):
    """
    A backend for plotting SymPy's symbolic expressions using Bokeh.
    This implementation only supports 2D plots.

    Notes
    =====

    By providing ``update_event=True`` to any plot function, this backend
    binds pan/zoom events in order to automatically compute new data as the
    user interact with the plot.

    When executing this mode of operation inside:

    * Jupyter Notebook/Lab: no problem has been encountered (with
      Firefox/Chrome).
    * A standard Python interpreter:

      * No problem has been encountered with Chrome.
      * Memory leaks has been observed with Firefox. Watch out your system
        monitor!


    See also
    ========

    Plot, MatplotlibBackend, PlotlyBackend, K3DBackend
    """

    renderers_map = {
        LineOver1DRangeSeries: Line2DRenderer,
        List2DSeries: Line2DRenderer,
        Parametric2DLineSeries: Line2DRenderer,
        ColoredLineOver1DRangeSeries: Line2DRenderer,
        AbsArgLineSeries: Line2DRenderer,
        ComplexPointSeries: Line2DRenderer,
        Vector2DSeries: Vector2DRenderer,
        ComplexDomainColoringSeries: ComplexRenderer,
        ContourSeries: ContourRenderer,
        Geometry2DSeries: GeometryRenderer,
        GenericDataSeries: GenericRenderer,
        HVLineSeries: HVLineRenderer,
        HLineSeries: HVLineRenderer,
        VLineSeries: HVLineRenderer,
        Arrow2DSeries: Arrow2DRenderer,
        RootLocusSeries: RootLocusRenderer,
        SGridLineSeries: SGridLineRenderer,
        ZGridLineSeries: ZGridLineRenderer,
        SystemResponseSeries: Line2DRenderer,
        ColoredSystemResponseSeries: Line2DRenderer,
        PoleZeroSeries: PoleZeroRenderer,
        PoleZeroWithSympySeries: PoleZeroRenderer,
        NGridLineSeries: NGridLineRenderer,
        NicholsLineSeries: NicholsLineRenderer,
        MCirclesSeries: MCirclesRenderer,
        NyquistLineSeries: NyquistRenderer,
    }

    def __init__(self, *args, **kwargs):
        self.np = import_module('numpy')
        self.bokeh = import_module(
            'bokeh',
            import_kwargs={
                'fromlist': [
                    'models', 'events', 'plotting', 'io',
                    'palettes', 'embed', 'resources', 'server'
                ]
            },
            warn_not_installed=True,
            min_module_version='2.3.0'
        )
        bp = self.bokeh.palettes
        cc = import_module(
            'colorcet',
            min_module_version='3.0.0')
        matplotlib = import_module(
            'matplotlib',
            import_kwargs={'fromlist': ['pyplot', 'cm']},
            min_module_version='1.1.0',
            catch=(RuntimeError,))
        cm = matplotlib.cm

        kwargs["_library"] = "bokeh"
        kwargs.setdefault("colorloop", bp.Category10[10])
        kwargs.setdefault("colormaps", [
            cc.bmy, "aggrnyl", cc.kbc, cc.bjy, "plotly3"])
        kwargs.setdefault("cyclic_colormaps", [
            cm.hsv, cm.twilight, cc.cyclic_mygbm_30_95_c78_s25
        ])

        kwargs.setdefault("use_latex", cfg["bokeh"]["use_latex"])
        kwargs.setdefault("theme", cfg["bokeh"]["theme"])
        kwargs.setdefault("grid", cfg["bokeh"]["grid"])
        kwargs.setdefault("minor_grid", cfg["bokeh"]["show_minor_grid"])
        kwargs.setdefault("update_event", cfg["bokeh"]["update_event"])

        # _init_cyclers needs to know if an existing figure was provided
        self._use_existing_figure = "fig" in kwargs

        super().__init__(*args, **kwargs)

        self._init_cyclers()

        if self.polar_axis:
            raise ValueError("BokehBackend doesn't support polar axis.")

        self._set_labels()
        self._set_title()

        self._run_in_notebook = False
        if get_environment() == 0:
            self._run_in_notebook = True
            self.bokeh.io.output_notebook(hide_banner=True)

        if (
            (len([s for s in self._series if s.is_2Dline]) > 10) and
            (not type(self).colorloop) and
            not ("process_piecewise" in kwargs.keys())
        ):
            # add colors if needed
            self.colorloop = bp.Category20[20]

        self._handles = dict()
        sizing_mode = cfg["bokeh"]["sizing_mode"]
        title, xlabel, ylabel, zlabel = self._get_title_and_labels()
        kw = dict(
            title=title,
            x_axis_label=xlabel if xlabel else "x",
            y_axis_label=ylabel if ylabel else "y",
            sizing_mode="fixed" if self.size else sizing_mode,
            width=int(self.size[0]) if self.size else cfg["bokeh"]["width"],
            height=int(self.size[1]) if self.size else cfg["bokeh"]["height"],
            tools="pan,wheel_zoom,box_zoom,reset,save",
            match_aspect=True if self.aspect == "equal" else False,
        )
        if self.xlim:
            kw["x_range"] = self.xlim
        if self.ylim:
            kw["y_range"] = self.ylim
        if self.xscale:
            kw["x_axis_type"] = self.xscale
        if self.yscale:
            kw["y_axis_type"] = self.yscale
        if self._fig is None:
            self._fig = self.bokeh.plotting.figure(**kw)
        self._fig.axis.visible = self.axis

        show_major_grid = True if self.grid else False
        show_minor_grid = True if self.minor_grid else False
        grid_lines_kw = {}
        if isinstance(self.grid, dict):
            grid_lines_kw = self.merge(grid_lines_kw, self.grid)
        if show_minor_grid:
            grid_lines_kw["minor_grid_line_alpha"] = cfg["bokeh"]["minor_grid_line_alpha"]
            grid_lines_kw["minor_grid_line_color"] = self._fig.grid.grid_line_color[0]
            grid_lines_kw["minor_grid_line_dash"] = cfg["bokeh"]["minor_grid_line_dash"]
        if isinstance(self.minor_grid, dict):
            grid_lines_kw = self.merge({}, grid_lines_kw, self.minor_grid)

        self._fig.grid.visible = show_major_grid
        for k, v in grid_lines_kw.items():
            setattr(self._fig.grid, k, v)

        if self.invert_x_axis:
            self._fig.x_range.flipped = True

        if self.x_ticks_formatter:
            if isinstance(self.x_ticks_formatter, tick_formatter_multiples_of):
                self._fig.xaxis.ticker = self.x_ticks_formatter.BB_ticker()
                self._fig.xaxis.formatter = self.x_ticks_formatter.BB_formatter()
        if self.y_ticks_formatter:
            if isinstance(self.y_ticks_formatter, tick_formatter_multiples_of):
                self._fig.yaxis.ticker = self.y_ticks_formatter.BB_ticker()
                self._fig.yaxis.formatter = self.y_ticks_formatter.BB_formatter()

        if self.update_event:
            self._fig.on_event(self.bokeh.events.RangesUpdate, self._ranges_update)

        self._create_renderers()

    def _ranges_update(self, event):
        xlim = (event.x0, event.x1)
        ylim = (event.y0, event.y1)
        params = self._update_series_ranges(xlim, ylim)
        self.update_interactive(params)

    def _init_cyclers(self):
        start_index_cl, start_index_cm = None, None
        if self._use_existing_figure:
            fig = self._use_existing_figure if self._fig is None else self._fig
            # attempt to determine how many lines are plotted
            # on the user-provided figure
            start_index_cl = len(fig.renderers)
        super()._init_cyclers(start_index_cl, 0)

    @property
    def fig(self):
        """Returns the figure."""
        if (
            (len(self.renderers) > 0) and
            (
                (self.renderers[0] and len(self.renderers[0].handles) == 0)
                or (self.renderers[0] is None)
            )
        ):
            # if the backend was created without showing it
            self.draw()
        return self._fig

    def draw(self):
        """
        Loop over data renderers, generates numerical data and add it to
        the figure. Note that this method doesn't show the plot.
        """
        self._process_renderers()
        self._execute_hooks()

    def _set_piecewise_color(self, s, color):
        """Set the color to the given series"""
        if "color" not in s.rendering_kw:
            # only set the color if the user didn't do that already
            s.rendering_kw["color"] = color
            if s.is_scatter and (not s.is_filled):
                s.rendering_kw["fill_color"] = "white"

    @staticmethod
    def _do_sum_kwargs(p1, p2):
        return p1._copy_kwargs()

    def _process_renderers(self):
        self._init_cyclers()

        if not self._use_existing_figure:
            # If this instance visualizes only symbolic expressions,
            # I want to clear axes so that each time `.show()` is called there
            # won't be repeated handles.
            # On the other hand, if the current axes is provided by the user,
            # we don't want to erase its content.

            # Must clear both the renderers as well as the
            # colorbars which are added to the right side.
            self._fig.renderers = []
            self._fig.right = []

        xlims, ylims = [], []
        for r, s in zip(self.renderers, self.series):
            self._check_supported_series(r, s)
            r.draw()
            if hasattr(r, "xlims"):
                xlims.extend(r.xlims)
                ylims.extend(r.ylims)

        if (len(xlims) > 0) and (self.xlim is None):
            # this is used in order to properly visualized some *GridSeries
            np = self.np
            xlims = np.array(xlims)
            xlim = (np.nanmin(xlims[:, 0]), np.nanmax(xlims[:, 1]))
            self._fig.x_range = self.bokeh.models.Range1d(*xlim)

        if (len(ylims) > 0) and (self.ylim is None):
            # this is used in order to properly visualized some *GridSeries
            np = self.np
            ylims = np.array(ylims)
            ylim = (np.nanmin(ylims[:, 0]), np.nanmax(ylims[:, 1]))
            self._fig.y_range = self.bokeh.models.Range1d(*ylim)

        if len(self._fig.legend) > 0:
            # hide default legend
            self._fig.legend.visible = False
            # add a new legend only showing the appropriate items
            legend_items = []
            end = 0
            if self._use_existing_figure:
                legend_items = self._fig.legend.items
                # keep existing legend entries if we are dealing with a
                # user-provided figure
                end = len(legend_items) - len(self.series)
                legend_items = legend_items[:end]
            for s, r in zip(self.series, self.renderers):
                if (
                    s.show_in_legend and
                    (s.is_2Dline or s.is_geometry) and
                    (not s.use_cm)
                ):
                    if hasattr(r.handles[0][0], "__iter__"):
                        bokeh_renderer = r.handles[0][0][0]
                    else:
                        bokeh_renderer = r.handles[0][0]
                    legend_items.append(
                        self.bokeh.models.LegendItem(
                            label=s.get_label(self.use_latex),
                            renderers=[bokeh_renderer])
                    )
            if self.legend and (len(legend_items) > 0):
                legend = self.bokeh.models.Legend(items=legend_items)
                # interactive legend
                legend.click_policy = "hide"
                self._fig.add_layout(legend, "right")

    def _get_img(self, img):
        np = import_module('numpy')
        new_img = np.zeros(img.shape[:2], dtype=np.uint32)
        pixel = new_img.view(dtype=np.uint8).reshape((*img.shape[:2], 4))
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                pixel[j, i] = [*img[j, i], 255]
        return new_img

    def _get_segments(self, x, y, *others):
        # MultiLine works with line segments, not with line points! :|
        xs = [x[i - 1 : i + 1] for i in range(1, len(x))]
        ys = [y[i - 1 : i + 1] for i in range(1, len(y))]
        # let n be the number of points. Then, the number of segments
        # will be (n - 1). Therefore, we remove one parameter. If n is
        # sufficiently high, there shouldn't be any noticeable problem in
        # the visualization.
        others = list(others)
        for i, o in enumerate(others):
            others[i] = o[:-1]
        return [xs, ys, *others]

    def _create_gradient_line(
        self, x_key, y_key, p_key, source, colormap, name, line_kw,
        is_scatter=False
    ):
        param = source[p_key]
        color_mapper = self.bokeh.models.LinearColorMapper(
            palette=colormap, low=min(param), high=max(param))
        data_source = self.bokeh.models.ColumnDataSource(source)

        lkw = dict(
            line_width=2,
            name=name,
            line_color={"field": p_key, "transform": color_mapper},
        )
        kw = self.merge({}, lkw, line_kw)
        if not is_scatter:
            glyph = self.bokeh.models.MultiLine(xs=x_key, ys=y_key, **kw)
        else:
            glyph = self.bokeh.models.Scatter(x=x_key, y=y_key, **kw)
        colorbar = self.bokeh.models.ColorBar(
            color_mapper=color_mapper, title=name, width=8)
        return data_source, glyph, colorbar, kw

    def update_interactive(self, params):
        """
        Implement the logic to update the data generated by
        interactive-widget plots.

        Parameters
        ==========

        params : dict
            Map parameter-symbols to numeric values.
        """
        if len(self.renderers) > 0 and len(self.renderers[0].handles) == 0:
            self.draw()

        xlims, ylims = [], []
        for r in self.renderers:
            if (
                r.series.is_interactive
                or hasattr(r.series, "_interactive_app_controls")
            ):
                r.update(params)
                if hasattr(r, "_xlims"):
                    xlims.extend(r._xlims)
                    ylims.extend(r._ylims)

        if (len(xlims) > 0) and (self.xlim is None):
            # this is used in order to properly visualized some *GridSeries
            np = self.np
            xlims = np.array(xlims)
            xlim = (np.nanmin(xlims[:, 0]), np.nanmax(xlims[:, 1]))
            self._fig.x_range.update(start=xlim[0], end=xlim[1])

        if (len(ylims) > 0) and (self.ylim is None):
            # this is used in order to properly visualized some *GridSeries
            np = self.np
            ylims = np.array(ylims)
            ylim = (np.nanmin(ylims[:, 0]), np.nanmax(ylims[:, 1]))
            self._fig.y_range.update(start=ylim[0], end=ylim[1])

        self._set_axes_texts()
        self._execute_hooks()

    def _set_axes_texts(self):
        title, xlabel, ylabel, zlabel = self._get_title_and_labels()
        self._fig.title = title
        self._fig.xaxis.axis_label = xlabel
        self._fig.yaxis.axis_label = ylabel

    def save(self, path, **kwargs):
        """
        Export the plot to a static picture or to an interactive html file.

        Refer to [#fn3]_ and [#fn4]_ to visualize all the available keyword
        arguments.

        Notes
        =====

        1. In order to export static pictures, the user also need to install
           the packages listed in [#fn5]_.
        2. When exporting a fully portable html file, by default the necessary
           Javascript libraries will be loaded with a CDN. This creates the
           smallest file size possible, but it requires an internet connection
           in order to view/load the file and its dependencies.

        References
        ==========
        .. [#fn3] https://docs.bokeh.org/en/latest/docs/user_guide/export.html
        .. [#fn4] https://docs.bokeh.org/en/latest/docs/user_guide/embed.html
        .. [#fn5] https://docs.bokeh.org/en/latest/docs/reference/io.html#module-bokeh.io.export

        """
        merge = self.merge

        ext = os.path.splitext(path)[1]
        if ext.lower() in [".htm", ".html"]:
            CDN = self.bokeh.resources.CDN
            file_html = self.bokeh.embed.file_html
            skw = dict(resources=CDN, title="Bokeh Plot")
            html = file_html(self.fig, **merge(skw, kwargs))
            with open(path, 'w') as f:
                f.write(html)
        elif ext == ".svg":
            self._fig.output_backend = "svg"
            self.bokeh.io.export_svg(self.fig, filename=path)
        else:
            if ext == "":
                path += ".png"
            self._fig.output_backend = "canvas"
            self.bokeh.io.export_png(self._fig, filename=path)

    def _launch_server(self, doc):
        """ By launching a server application, we can use Python callbacks
        associated to events.
        """
        doc.theme = self.theme
        doc.add_root(self.fig)

    def show(self):
        """Visualize the plot on the screen."""
        if len(self._fig.renderers) != len(self.series):
            self.draw()

        if self.update_event:
            if self._run_in_notebook:
                self.bokeh.plotting.show(self._launch_server)
            else:
                # NOTE:
                # 1. From: https://docs.bokeh.org/en/latest/docs/user_guide/server/library.html
                #    In particular: https://github.com/bokeh/bokeh/tree/3.4.0/examples/server/api/standalone_embed.py
                # 2. TODO: Only works for one plot, then python needs to be
                #    closed and reopened.
                # 3. Use Control+C to stop the server process
                # 4. Watch out for memory leaks on Firefox.
                from bokeh.server.server import Server
                server = Server(self._launch_server, num_procs=1)
                server.start()
                server.io_loop.add_callback(server.show, "/")
                server.io_loop.start()
        else:
            # launch a static figure
            curdoc = self.bokeh.io.curdoc
            curdoc().theme = self.theme
            self.bokeh.plotting.show(self._fig)

    def _get_quivers_data(self, xs, ys, u, v, **quiver_kw):
        """Compute the segments coordinates to plot quivers.

        Parameters
        ==========
        xs : np.ndarray
            A 2D numpy array representing the discretization in the
            x-coordinate
        ys : np.ndarray
            A 2D numpy array representing the discretization in the
            y-coordinate
        u : np.ndarray
            A 2D numpy array representing the x-component of the vector
        v : np.ndarray
            A 2D numpy array representing the x-component of the vector
        kwargs : dict, optional
            An optional

        Returns
        =======
        data: dict
            A dictionary suitable to create a data source to be used with
            Bokeh's Segment.
        quiver_kw : dict
            A dictionary containing keywords to customize the appearance
            of Bokeh's Segment glyph
        """
        np = import_module('numpy')
        scale = quiver_kw.pop("scale", 1.0)
        pivot = quiver_kw.pop("pivot", "mid")
        arrow_heads = quiver_kw.pop("arrow_heads", True)

        xs, ys, u, v = [t.flatten() for t in [xs, ys, u, v]]

        magnitude = np.sqrt(u ** 2 + v ** 2)
        rads = np.arctan2(v, u)
        lens = magnitude / max(magnitude) * scale

        # Compute segments and arrowheads
        # Compute offset depending on pivot option
        xoffsets = np.cos(rads) * lens / 2.0
        yoffsets = np.sin(rads) * lens / 2.0
        if pivot == "mid":
            nxoff, pxoff = xoffsets, xoffsets
            nyoff, pyoff = yoffsets, yoffsets
        elif pivot == "tip":
            nxoff, pxoff = 0, xoffsets * 2
            nyoff, pyoff = 0, yoffsets * 2
        elif pivot == "tail":
            nxoff, pxoff = xoffsets * 2, 0
            nyoff, pyoff = yoffsets * 2, 0
        else:
            raise ValueError(
                "`pivot` must be one of ['mid', 'tip', 'tail']")
        x0s, x1s = (xs + nxoff, xs - pxoff)
        y0s, y1s = (ys + nyoff, ys - pyoff)

        if arrow_heads:
            arrow_len = lens / 4.0
            xa1s = x0s - np.cos(rads + np.pi / 4) * arrow_len
            ya1s = y0s - np.sin(rads + np.pi / 4) * arrow_len
            xa2s = x0s - np.cos(rads - np.pi / 4) * arrow_len
            ya2s = y0s - np.sin(rads - np.pi / 4) * arrow_len
            x0s = np.tile(x0s, 3)
            x1s = np.concatenate([x1s, xa1s, xa2s])
            y0s = np.tile(y0s, 3)
            y1s = np.concatenate([y1s, ya1s, ya2s])

        data = {
            "x0": x0s,
            "x1": x1s,
            "y0": y0s,
            "y1": y1s,
        }

        return data, quiver_kw


BB = BokehBackend
