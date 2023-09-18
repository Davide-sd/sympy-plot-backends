import itertools
import os
from spb.defaults import cfg
from spb.backends.base_backend import Plot
from spb.backends.matplotlib.renderers import (
    Line2DRenderer, Line3DRenderer, Vector2DRenderer, Vector3DRenderer,
    ComplexRenderer, ContourRenderer, SurfaceRenderer, Implicit3DRenderer,
    GeometryRenderer, GenericRenderer, HVLineRenderer
)
from spb.series import (
    LineOver1DRangeSeries, List2DSeries, Parametric2DLineSeries,
    ColoredLineOver1DRangeSeries, AbsArgLineSeries, ComplexPointSeries,
    Parametric3DLineSeries, ComplexParametric3DLineSeries,
    List3DSeries, Vector2DSeries, Vector3DSeries, SliceVector3DSeries,
    RiemannSphereSeries, Implicit3DSeries,
    ComplexDomainColoringSeries, ComplexSurfaceSeries,
    ContourSeries, SurfaceOver2DRangeSeries, ParametricSurfaceSeries,
    PlaneSeries, GeometrySeries, GenericDataSeries,
    HVLineSeries
)
from sympy.external import import_module


class PlotlyBackend(Plot):
    """
    A backend for plotting SymPy's symbolic expressions using Plotly.

    Parameters
    ==========

    aspect : str, optional
        Set the aspect ratio of the plot. Default to ``"auto"``.
        Possible values:

        - ``"equal"``: sets equal spacing on the axis of a 2D plot.
        - For 3D plots:

          * ``"cube"``: fix the ratio to be a cube
          * ``"data"``: draw axes in proportion to the proportion of their
            ranges
          * ``"auto"``: automatically produce something that is well
            proportioned using 'data' as the default.
          * manually set the aspect ratio by providing a dictionary.
            For example: ``dict(x=1, y=1, z=2)`` forces the z-axis to appear
            twice as big as the other two.

    camera : dict, optional
        A dictionary of keyword arguments that will be passed to the layout's
        ``scene_camera`` in order to set the 3D camera orientation.
        Refer to [#fn18]_ for more information.

    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to Matplotlib's plot
        functions to customize the appearance of lines, surfaces, images,
        contours, quivers, streamlines...
        To learn more about customization:

        * Refer to [#fn1]_ and [#fn2]_ to customize contour plots.
        * Refer to [#fn3]_ and [#fn4]_ to customize line plots.
        * Refer to [#fn7]_ to customize surface plots.
        * Refer to [#fn14]_ to customize implicit surface plots.
        * Refer to [#fn5]_ to customize 2D quiver plots. Default to:
          ``dict( scale = 0.075 )``.
        * Refer to [#fn6]_ to customize 3D cone plots. Default to:
          ``dict( sizemode = "absolute", sizeref = 40 )``.
        * Refer to [#fn8]_ to customize 2D streamlines plots. Defaul to:
          ``dict( arrow_scale = 0.15 )``.
        * Refer to [#fn9]_ to customize 3D streamlines plots. Defaul to:
          ``dict( sizeref = 0.3 )``.

    axis : boolean, optional
        Turns on/off the axis visibility (and associated tick labels).
        Default to True (axis are visible).

    theme : str, optional
        Set the theme. Default to ``"plotly_dark"``. Find more Plotly themes at
        [#fn10]_ .

    use_cm : boolean, optional
        If True, apply a color map to the meshes/surface. If False, solid
        colors will be used instead. Default to True.

    annotations : list, optional
        A list of dictionaries specifying the type the markers required.
        The keys in the dictionary should be equivalent to the arguments
        of the Plotly's `graph_objects.Scatter` class. Refer to [#fn15]_
        for more information.
        This feature is experimental. It might get removed in the future.

    markers : list, optional
        A list of dictionaries specifying the type the markers required.
        The keys in the dictionary should be equivalent to the arguments
        of the Plotly's `graph_objects.Scatter` class. Refer to [#fn3]_
        for more information.
        This feature is experimental. It might get removed in the future.

    rectangles : list, optional
        A list of dictionaries specifying the dimensions of the
        rectangles to be plotted. The keys in the dictionary should be
        equivalent to the arguments of the Plotly's
        `graph_objects.Figure.add_shape` function. Refer to [#fn16]_
        for more information.
        This feature is experimental. It might get removed in the future.

    fill : dict, optional
        A list of dictionaries specifying the type the markers required.
        The keys in the dictionary should be equivalent to the arguments
        of the Plotly's `graph_objects.Scatter` class. Refer to [#fn17]_
        for more information.
        This feature is experimental. It might get removed in the future.

    References
    ==========
    .. [#fn1] https://plotly.com/python/contour-plots/
    .. [#fn2] https://plotly.com/python/builtin-colorscales/
    .. [#fn3] https://plotly.com/python/line-and-scatter/
    .. [#fn4] https://plotly.com/python/3d-scatter-plots/
    .. [#fn5] https://plotly.com/python/quiver-plots/
    .. [#fn6] https://plotly.com/python/cone-plot/
    .. [#fn7] https://plotly.com/python/3d-surface-plots/
    .. [#fn8] https://plotly.com/python/streamline-plots/
    .. [#fn9] https://plotly.com/python/streamtube-plot/
    .. [#fn10] https://plotly.com/python/templates/
    .. [#fn13] https://github.com/plotly/plotly.js/issues/5003
    .. [#fn14] https://plotly.com/python/3d-isosurface-plots/
    .. [#fn15] https://plotly.com/python/text-and-annotations/
    .. [#fn16] https://plotly.com/python/shapes/
    .. [#fn17] https://plotly.com/python/filled-area-plots/
    .. [#fn18] https://plotly.com/python/3d-camera-controls/


    Notes
    =====

    A few bugs related to Plotly might prevent the correct visualization:

    * with 2D domain coloring, the vertical axis is reversed, with negative
      values on the top and positive values on the bottom.
    * with 3D complex plots: when hovering a point, the tooltip will display
      wrong information for the argument and the phase. Hopefully, this bug
      [#fn13]_ will be fixed upstream.

    See also
    ========

    Plot, MatplotlibBackend, BokehBackend, K3DBackend
    """

    _library = "plotly"
    _allowed_keys = Plot._allowed_keys + [
        "markers", "annotations", "fill", "rectangles", "camera"]

    colorloop = []
    colormaps = []
    cyclic_colormaps = []
    quivers_colors = []
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
        GeometrySeries: GeometryRenderer,
        GenericDataSeries: GenericRenderer,
        HVLineSeries: HVLineRenderer
    }

    pole_line_kw = {"line": dict(color='black', dash='dot', width=1)}

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

        # The following colors corresponds to the discret color map
        # px.colors.qualitative.Plotly.
        self.colorloop = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
        self.colormaps = [
            "aggrnyl", "plotly3", "reds_r", "ice", "inferno",
            "deep_r", "turbid_r", "gnbu_r", "geyser_r", "oranges_r"]
        self.cyclic_colormaps = ["phase", "twilight", "hsv", "icefire"]
        # TODO: here I selected black and white, but they are not visible
        # with dark or light theme respectively... Need a better selection
        # of colors. Although, they are placed in the middle of the loop,
        # so they are unlikely going to be used.
        self.quivers_colors = [
            "magenta", "crimson", "darkorange", "dodgerblue", "wheat",
            "slategrey", "white", "black", "darkred", "indigo"]

        self._init_cyclers()
        super().__init__(*series, **kwargs)

        # NOTE: Plotly 3D currently doesn't support latex labels
        # https://github.com/plotly/plotly.js/issues/608
        self._use_latex = kwargs.get("use_latex", cfg["plotly"]["use_latex"])
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

        self._theme = kwargs.get("theme", cfg["plotly"]["theme"])
        self.grid = kwargs.get("grid", cfg["plotly"]["grid"])
        if self.is_iplot and (self.imodule == "ipywidgets"):
            self._fig = self.go.FigureWidget()
        else:
            self._fig = self.go.Figure()
        self._colorbar_counter = 0

        self._scale_down_colorbar = (
            self.legend and
            any(s.use_cm for s in self.series) and
            any((not s.use_cm) for s in self.series)
        )
        self._show_2D_vectors = any(s.is_2Dvector for s in self.series)
        self._create_renderers()

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
        kw = p1._copy_kwargs()
        kw["theme"] = p1._theme
        return kw

    def _init_cyclers(self):
        super()._init_cyclers()
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
            title=label,
            titleside="right",
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
        self._fig.data = []

        for r, s in zip(self.renderers, self.series):
            self._check_supported_series(r, s)
            r.draw()

    def update_interactive(self, params):
        """Implement the logic to update the data generated by
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

    def _update_interactive_helper(self, params):
        for r in self.renderers:
            if r.series.is_interactive:
                r.update(params)

    def _update_layout(self):
        title, xlabel, ylabel, zlabel = self._get_title_and_labels()
        self._fig.update_layout(
            template=self._theme,
            width=None if not self.size else self.size[0],
            height=None if not self.size else self.size[1],
            title=r"<b>%s</b>" % ("" if not title else title),
            title_x=0.5,
            xaxis=dict(
                title="" if not xlabel else xlabel,
                range=None if not self.xlim else self.xlim,
                type=self.xscale,
                showgrid=self.grid,  # thin lines in the background
                zeroline=self.grid,  # thick line at x=0
                constrain="domain",
                visible=self.axis,
                autorange=None if not self._invert_x_axis else "reversed"
            ),
            yaxis=dict(
                title="" if not ylabel else ylabel,
                range=None if not self.ylim else self.ylim,
                type=self.yscale,
                showgrid=self.grid,  # thin lines in the background
                zeroline=self.grid,  # thick line at x=0
                scaleanchor="x" if self.aspect == "equal" else None,
                visible=self.axis
            ),
            polar=dict(
                angularaxis={'direction': 'counterclockwise', 'rotation': 0},
                radialaxis={'range': None if not self.ylim else self.ylim},
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
                    showgrid=self.grid,  # thin lines in the background
                    zeroline=self.grid,  # thick line at x=0
                    visible=self.grid,  # numbers below
                ),
                yaxis=dict(
                    title="" if not ylabel else ylabel,
                    range=None if not self.ylim else self.ylim,
                    type=self.yscale,
                    showgrid=self.grid,  # thin lines in the background
                    zeroline=self.grid,  # thick line at x=0
                    visible=self.grid,  # numbers below
                ),
                zaxis=dict(
                    title="" if not zlabel else zlabel,
                    range=None if not self.zlim else self.zlim,
                    type=self.zscale,
                    showgrid=self.grid,  # thin lines in the background
                    zeroline=self.grid,  # thick line at x=0
                    visible=self.grid,  # numbers below
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
        """Visualize the plot on the screen."""
        if len(self.renderers) > 0 and len(self.renderers[0].handles) == 0:
            self.draw()
        self._fig.show()

    def save(self, path, **kwargs):
        """ Export the plot to a static picture or to an interactive html file.

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
