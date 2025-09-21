import param
import itertools
import os
from spb.defaults import cfg
from spb.doc_utils.ipython import modify_parameterized_doc
from spb.backends.base_backend import Plot
from spb.backends.utils import tick_formatter_multiples_of
from spb.backends.plotly.renderers import (
    Line2DRenderer, Line3DRenderer, Vector2DRenderer, Vector3DRenderer,
    ComplexRenderer, ContourRenderer, SurfaceRenderer, Implicit3DRenderer,
    GeometryRenderer, GenericRenderer, HVLineRenderer, Arrow2DRenderer
)
from spb.series import (
    LineOver1DRangeSeries, List2DSeries, Parametric2DLineSeries,
    ColoredLineOver1DRangeSeries, AbsArgLineSeries, ComplexPointSeries,
    Parametric3DLineSeries, ComplexParametric3DLineSeries,
    List3DSeries, Vector2DSeries, Vector3DSeries, SliceVector3DSeries,
    RiemannSphereSeries, Implicit3DSeries,
    ComplexDomainColoringSeries, ComplexSurfaceSeries,
    ContourSeries, SurfaceOver2DRangeSeries, ParametricSurfaceSeries,
    PlaneSeries, Geometry2DSeries, Geometry3DSeries, GenericDataSeries,
    HVLineSeries, Arrow2DSeries, HLineSeries, VLineSeries
)
from sympy.external import import_module
import warnings


@modify_parameterized_doc()
class PlotlyBackend(Plot):
    """
    A backend for plotting SymPy's symbolic expressions using Plotly.

    Notes
    =====

    A few bugs related to Plotly might prevent the correct visualization:

    * with 2D domain coloring, the vertical axis is reversed, with negative
      values on the top and positive values on the bottom.
    * with 3D complex plots: when hovering a point, the tooltip will display
      wrong information for the argument and the phase.
      https://github.com/plotly/plotly.js/issues/5003
      Hopefully, this bug will be fixed upstream.

    See also
    ========

    Plot, MatplotlibBackend, BokehBackend, K3DBackend
    """

    wireframe_color = "#000000"

    scattergl_threshold = 2000
    # color bar spacing
    _cbs = 0.15
    # color bar scale down factor
    _cbsdf = 0.75

    renderers_map = {
        LineOver1DRangeSeries: Line2DRenderer,
        List2DSeries: Line2DRenderer,
        Parametric2DLineSeries: Line2DRenderer,
        ColoredLineOver1DRangeSeries: Line2DRenderer,
        AbsArgLineSeries: Line2DRenderer,
        ComplexPointSeries: Line2DRenderer,
        Parametric3DLineSeries: Line3DRenderer,
        ComplexParametric3DLineSeries: Line3DRenderer,
        List3DSeries: Line3DRenderer,
        Vector2DSeries: Vector2DRenderer,
        Vector3DSeries: Vector3DRenderer,
        SliceVector3DSeries: Vector3DRenderer,
        Implicit3DSeries: Implicit3DRenderer,
        ComplexDomainColoringSeries: ComplexRenderer,
        ComplexSurfaceSeries: ComplexRenderer,
        RiemannSphereSeries: ComplexRenderer,
        ContourSeries: ContourRenderer,
        SurfaceOver2DRangeSeries: SurfaceRenderer,
        ParametricSurfaceSeries: SurfaceRenderer,
        PlaneSeries: SurfaceRenderer,
        Geometry2DSeries: GeometryRenderer,
        Geometry3DSeries: GeometryRenderer,
        GenericDataSeries: GenericRenderer,
        HVLineSeries: HVLineRenderer,
        HLineSeries: HVLineRenderer,
        VLineSeries: HVLineRenderer,
        Arrow2DSeries: Arrow2DRenderer
    }

    quivers_colors = param.ClassSelector(default=[], class_=(list, tuple), doc="""
        List of colors for rendering quivers.""")

    def __init__(self, *series, **kwargs):
        self.np = import_module('numpy')
        self.plotly = import_module(
            'plotly',
            import_kwargs={'fromlist': ['graph_objects', 'figure_factory']},
            warn_not_installed=True,
            min_module_version='5.0.0')
        self.go = self.plotly.graph_objects
        self.create_quiver = self.plotly.figure_factory.create_quiver
        self.create_streamline = self.plotly.figure_factory.create_streamline

        kwargs["_library"] = "plotly"
        # The following colors corresponds to the discret color map
        # px.colors.qualitative.Plotly.
        kwargs.setdefault("colorloop", [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"])
        kwargs.setdefault("colormaps", [
            "aggrnyl", "plotly3", "reds_r", "ice", "inferno",
            "deep_r", "turbid_r", "gnbu_r", "geyser_r", "oranges_r"])
        kwargs.setdefault("cyclic_colormaps", [
            "phase", "twilight", "hsv", "icefire"])
        # TODO: here I selected black and white, but they are not visible
        # with dark or light theme respectively... Need a better selection
        # of colors. Although, they are placed in the middle of the loop,
        # so they are unlikely going to be used.
        kwargs.setdefault("quivers_colors", [
            "magenta", "crimson", "darkorange", "dodgerblue", "wheat",
            "slategrey", "white", "black", "darkred", "indigo"])
        kwargs.setdefault("update_event", cfg["plotly"]["update_event"])
        kwargs.setdefault("use_latex", cfg["plotly"]["use_latex"])
        kwargs.setdefault("grid", cfg["plotly"]["grid"])
        kwargs.setdefault("minor_grid", cfg["plotly"]["show_minor_grid"])
        kwargs.setdefault("theme", cfg["plotly"]["theme"])

        # _init_cyclers needs to know if an existing figure was provided
        self._use_existing_figure = "fig" in kwargs

        super().__init__(*series, **kwargs)

        if (self.update_event and any(isinstance(s, Vector2DSeries) for
            s in series)):
            warnings.warn(
                "You are trying to use `update_event=True` with a 2D quiver "
                "plot. This is likely going to cause a render-loop. You might "
                "need to interrupt the kernel."
            )

        self._init_cyclers()

        if not self._use_existing_figure:
            if (
                (self.is_iplot and (self.imodule == "ipywidgets"))
                or self.update_event
            ):
                self._fig = self.go.FigureWidget()
            else:
                self._fig = self.go.Figure()

        # NOTE: Plotly 3D currently doesn't support latex labels
        # https://github.com/plotly/plotly.js/issues/608
        self._set_labels()
        self._set_title()

        if (
            (len([s for s in self._series if s.is_2Dline]) > 10) and
            (not type(self).colorloop) and
            not ("process_piecewise" in kwargs.keys())
        ):
            # add colors if needed
            # this corresponds to px.colors.qualitative.Light24
            self.colorloop = [
                "#FD3216", "#00FE35", "#6A76FC", "#FED4C4", "#FE00CE",
                "#0DF9FF", "#F6F926", "#FF9616", "#479B55", "#EEA6FB",
                "#DC587D", "#D626FF", "#6E899C", "#00B5F7", "#B68E00",
                "#C9FBE5", "#FF0092", "#22FFA7", "#E3EE9E", "#86CE00",
                "#BC7196", "#7E7DCD", "#FC6955", "#E48F72"
            ]

        self._colorbar_counter = 0
        self._scale_down_colorbar = (
            self.legend and
            any(s.use_cm for s in self.series) and
            any((not s.use_cm) for s in self.series)
        )
        self._show_2D_vectors = any(s.is_2Dvector for s in self.series)
        self._create_renderers()
        self._n_annotations = 0

        if self.update_event:
            self._fig.layout.on_change(
                lambda obj, xrange, yrange: self._update_axis_limits(xrange, yrange),
                ('xaxis', 'range'), ('yaxis', 'range'))

    def _update_axis_limits(self, *limits):
        """Update the ranges of data series in order to implement pan/zoom
        update events.

        Parameters
        ==========
        limits : iterable
            Tuples of (min, max) values.
        """

        params = self._update_series_ranges(*limits)
        self.update_interactive(params)

    @property
    def fig(self):
        """Returns the figure."""
        if len(self.renderers) > 0 and len(self.renderers[0].handles) == 0:
            # if the backend was created without showing it
            self.draw()
        return self._fig

    def draw(self):
        """ Loop over data renderers, generates numerical data and add it to
        the figure. Note that this method doesn't show the plot.
        """
        self._process_renderers()
        self._update_layout()
        self._execute_hooks()

    process_series = draw

    def _set_piecewise_color(self, s, color):
        """Set the color to the given series"""
        if "line_color" not in s.rendering_kw:
            # only set the color if the user didn't do that already
            s.rendering_kw["line_color"] = color
            if not s.is_filled:
                s.rendering_kw["marker"] = dict(
                    color="#E5ECF6",
                    line=dict(color=color))

    @staticmethod
    def _do_sum_kwargs(p1, p2):
        return p1._copy_kwargs()

    def _init_cyclers(self):
        start_index_cl, start_index_cm = None, None
        if self._use_existing_figure:
            # attempt to determine how many lines or surfaces are plotted
            # on the user-provided figure

            # assume user plotted 3d surfaces using solid colors
            count_meshes = sum([
                isinstance(c, self.go.Surface) for c in self._fig.data])
            count_lines = sum([
                isinstance(c, self.go.Scatter) for c in self._fig.data])
            start_index_cl = count_lines + count_meshes
        super()._init_cyclers(start_index_cl, 0)
        tb = type(self)
        quivers_colors = (
            self.quivers_colors if not tb.quivers_colors
            else tb.quivers_colors
        )
        self._qc = itertools.cycle(quivers_colors)

    def _create_colorbar(self, label, sc=False):
        """This method reduces code repetition.

        Parameters
        ==========
            label : str
                Name to display besides the color bar
            sc : boolean
                Scale Down the color bar to make room for the legend.
                Default to False
        """
        k = self._colorbar_counter
        self._colorbar_counter += 1
        return dict(
            x=1 + self._cbs * k,
            title=dict(
                text=label,
                side="right"
            ),
            # scale down the color bar to make room for legend
            len=(
                self._cbsdf if (sc and (self.legend or (self.legend is None)))
                else 1
            ),
            yanchor="bottom",
            y=0,
        )

    def _solid_colorscale(self, s):
        # create a solid color to be used when s.use_cm=False
        col = s.line_color
        if col is None:
            col = next(self._cl)
        return [[0, col], [1, col]]

    def _process_renderers(self):
        self._init_cyclers()
        if not self._use_existing_figure:
            # If this instance visualizes only symbolic expressions,
            # I want to clear axes so that each time `.show()` is called there
            # won't be repeated handles.
            # On the other hand, if the current axes is provided by the user,
            # we don't want to erase its content.
            self._fig.data = []

        for r, s in zip(self.renderers, self.series):
            self._check_supported_series(r, s)
            r.draw()

    def update_interactive(self, params):
        """
        Implement the logic to update the data generated by
        interactive-widget plots.

        Parameters
        ==========

        params : dict
            Map parameter-symbols to numeric values.
        """
        # Because InteractivePlot doesn't call the show method, the following
        # line of code will add the numerical data (if not already present).
        if len(self.renderers) > 0 and len(self.renderers[0].handles) == 0:
            self.draw()

        if self.imodule == "ipywidgets":
            with self._fig.batch_update():
                self._update_interactive_helper(params)
        else:
            self._update_interactive_helper(params)

        self._set_axes_texts()
        self._execute_hooks()

    def _update_interactive_helper(self, params):
        for r in self.renderers:
            if (
                r.series.is_interactive
                or hasattr(r.series, "_interactive_app_controls")
            ):
                r.update(params)

    def _get_data_limits_for_custom_tickers(self):
        _min = lambda t: min(t) if len(t) > 0 else 0
        _max = lambda t: max(t) if len(t) > 0 else 0
        x_min, x_max = [], []
        y_min, y_max = [], []
        for s in self.series:
            if isinstance(s, (LineOver1DRangeSeries, SurfaceOver2DRangeSeries)):
                x_min.append(s.ranges[0][1].subs(s.params))
                x_max.append(s.ranges[0][2].subs(s.params))
            if isinstance(s, SurfaceOver2DRangeSeries):
                y_min.append(s.ranges[1][1].subs(s.params))
                y_max.append(s.ranges[1][2].subs(s.params))
        x_min, y_min = float(_min(x_min)), float(_min(y_min))
        x_max, y_max = float(_max(x_max)), float(_max(y_max))
        return x_min, x_max, y_min, y_max

    def _update_layout(self):
        title, xlabel, ylabel, zlabel = self._get_title_and_labels()
        show_major_grid = True if self.grid else False
        show_minor_grid = True if self.minor_grid else False
        major_grid_line_kw = {}
        minor_grid_line_kw = {}
        if isinstance(self.grid, dict):
            major_grid_line_kw = self.grid
        if isinstance(self.minor_grid, dict):
            minor_grid_line_kw = self.minor_grid
        minor_grid_line_kw_x = minor_grid_line_kw.copy()
        minor_grid_line_kw_y = minor_grid_line_kw.copy()

        # if necessary, apply custom tick formatting
        x_tickvals, x_ticktext = None, None
        y_tickvals, y_ticktext = None, None
        polar_angular_dtick = 30
        is_formatter = lambda t: isinstance(t, tick_formatter_multiples_of)
        if any(is_formatter(t) for t in [
            self.x_ticks_formatter, self.y_ticks_formatter]
        ):
            x_min, x_max, y_min, y_max = self._get_data_limits_for_custom_tickers()
        if is_formatter(self.x_ticks_formatter):
            if not self.np.isclose(x_min, x_max):
                x_tickvals, x_ticktext = self.x_ticks_formatter.PB_ticks(
                    x_min, x_max)
            q = self.x_ticks_formatter.quantity
            n = self.x_ticks_formatter.n
            n_minor = self.x_ticks_formatter.n_minor
            minor_grid_line_kw_x["dtick"] = (q / n) / (n_minor + 1)
            polar_angular_dtick = q / n
        if is_formatter(self.y_ticks_formatter):
            if not self.np.isclose(y_min, y_max):
                y_tickvals, y_ticktext = self.y_ticks_formatter.PB_ticks(
                    y_min, y_max)
            q = self.y_ticks_formatter.quantity
            n = self.y_ticks_formatter.n
            n_minor = self.y_ticks_formatter.n_minor
            minor_grid_line_kw_y["dtick"] = (q / n) / (n_minor + 1)

        self._fig.update_layout(
            template=self.theme,
            width=None if not self.size else self.size[0],
            height=None if not self.size else self.size[1],
            title=r"<b>%s</b>" % ("" if not title else title),
            title_x=0.5,
            xaxis=dict(
                title="" if not xlabel else xlabel,
                range=None if not self.xlim else self.xlim,
                type=self.xscale,
                showgrid=show_major_grid,  # thin lines in the background
                zeroline=show_major_grid,  # thick line at x=0
                constrain="domain",
                visible=self.axis,
                autorange=None if not self.invert_x_axis else "reversed",
                tickvals=x_tickvals,
                ticktext=x_ticktext,
                **major_grid_line_kw
            ),
            yaxis=dict(
                title="" if not ylabel else ylabel,
                range=None if not self.ylim else self.ylim,
                type=self.yscale,
                showgrid=show_major_grid,  # thin lines in the background
                zeroline=show_major_grid,  # thick line at x=0
                scaleanchor="x" if self.aspect == "equal" else None,
                visible=self.axis,
                tickvals=y_tickvals,
                ticktext=y_ticktext,
                **major_grid_line_kw
            ),
            polar=dict(
                angularaxis=dict(
                    direction='counterclockwise',
                    rotation=0,
                    thetaunit="radians" if is_formatter(self.x_ticks_formatter) else None,
                    dtick=polar_angular_dtick,
                ),
                radialaxis=dict(
                    range=None if not self.ylim else self.ylim
                ),
                sector=None if not self.xlim else self.xlim
            ),
            margin=dict(
                t=50,
                l=0,
                b=0,
                r=40
            ),
            showlegend=True if self.legend else False,
            scene=dict(
                xaxis=dict(
                    title="" if not xlabel else xlabel,
                    range=None if not self.xlim else self.xlim,
                    type=self.xscale,
                    showgrid=show_major_grid,  # thin lines in the background
                    zeroline=show_major_grid,  # thick line at x=0
                    visible=show_major_grid,  # numbers below,
                    tickvals=x_tickvals,
                    ticktext=x_ticktext,
                ),
                yaxis=dict(
                    title="" if not ylabel else ylabel,
                    range=None if not self.ylim else self.ylim,
                    type=self.yscale,
                    showgrid=show_major_grid,  # thin lines in the background
                    zeroline=show_major_grid,  # thick line at x=0
                    visible=show_major_grid,  # numbers below,
                    tickvals=y_tickvals,
                    ticktext=y_ticktext,
                ),
                zaxis=dict(
                    title="" if not zlabel else zlabel,
                    range=None if not self.zlim else self.zlim,
                    type=self.zscale,
                    showgrid=show_major_grid,  # thin lines in the background
                    zeroline=show_major_grid,  # thick line at x=0
                    visible=show_major_grid,  # numbers below
                ),
                aspectmode=(
                    "manual" if isinstance(self.aspect, dict)
                    else (self.aspect if self.aspect != "equal" else "auto")
                ),
                aspectratio=(
                    self.aspect if isinstance(self.aspect, dict) else None
                ),
                camera=self.camera
            ),
        )
        self._fig.update_xaxes(minor=dict(
            showgrid=show_minor_grid, **minor_grid_line_kw_x))
        self._fig.update_yaxes(minor=dict(
            showgrid=show_minor_grid, **minor_grid_line_kw_y))

    def _set_axes_texts(self):
        title, xlabel, ylabel, zlabel = self._get_title_and_labels()
        self._fig.update_layout(
            title=r"<b>%s</b>" % ("" if not title else title),
            xaxis=dict(
                title="" if not xlabel else xlabel,
            ),
            yaxis=dict(
                title="" if not ylabel else ylabel,
            ),
            scene=dict(
                xaxis=dict(
                    title="" if not xlabel else xlabel,
                ),
                yaxis=dict(
                    title="" if not ylabel else ylabel,
                ),
                zaxis=dict(
                    title="" if not zlabel else zlabel,
                ),
            ),
        )

    def show(self):
        """
        Visualize the plot on the screen.
        """
        if len(self.renderers) > 0 and len(self.renderers[0].handles) == 0:
            self.draw()
        self._fig.show()

    def save(self, path, **kwargs):
        """
        Export the plot to a static picture or to an interactive html file.

        Refer to [#fn11]_ and [#fn12]_ to visualize all the available keyword
        arguments.

        Notes
        =====
        In order to export static pictures, the user also need to install the
        packages listed in [#fn11]_.

        References
        ==========
        .. [#fn11] https://plotly.com/python/static-image-export/
        .. [#fn12] https://plotly.com/python/interactive-html-export/

        """
        if len(self.renderers) > 0 and len(self.renderers[0].handles) == 0:
            self.draw()

        ext = os.path.splitext(path)[1]
        if ext.lower() in [".htm", ".html"]:
            self.fig.write_html(path, **kwargs)
        else:
            self._fig.write_image(path, **kwargs)


PB = PlotlyBackend
