import itertools
from spb.defaults import cfg
from spb.backends.base_backend import Plot
from spb.backends.matplotlib.renderers import (
    Line2DRenderer, Line3DRenderer, Vector2DRenderer, Vector3DRenderer,
    Implicit2DRenderer, ComplexRenderer, ContourRenderer, SurfaceRenderer,
    GeometryRenderer, GenericRenderer, HVLineRenderer,
    NyquistRenderer, NicholsRenderer, Arrow2DRendererFancyArrowPatch
)
from spb.series import (
    LineOver1DRangeSeries, List2DSeries, Parametric2DLineSeries,
    ColoredLineOver1DRangeSeries, AbsArgLineSeries, ComplexPointSeries,
    Parametric3DLineSeries, ComplexParametric3DLineSeries,
    List3DSeries, Vector2DSeries, Vector3DSeries, SliceVector3DSeries,
    ImplicitSeries, RiemannSphereSeries,
    ComplexDomainColoringSeries, ComplexSurfaceSeries,
    ContourSeries, SurfaceOver2DRangeSeries, ParametricSurfaceSeries,
    PlaneSeries, GeometrySeries, GenericDataSeries,
    HVLineSeries, NyquistLineSeries, NicholsLineSeries,
    Arrow2DSeries
)
from sympy.external import import_module
from packaging import version

# Global variable
# Set to False when running tests / doctests so that the plots don't show.
_show = True


def unset_show():
    """
    Disable show(). For use in the tests.
    """
    global _show
    _show = False


class MatplotlibBackend(Plot):
    """
    A backend for plotting SymPy's symbolic expressions using Matplotlib.

    Parameters
    ==========

    aspect : (float, float) or str, optional
        Set the aspect ratio of a 2D plot. Possible values:

        * ``"auto"``: Matplotlib will fit the plot in the vibile area.
        * ``"equal"``: sets equal spacing.
        * tuple containing 2 float numbers, from which the aspect ratio is
          computed. This only works for 2D plots.

    axis_center : (float, float) or str or None, optional
        Set the location of the intersection between the horizontal and
        vertical axis in a 2D plot. It can be:

        * ``None``: traditional layout, with the horizontal axis fixed on the
          bottom and the vertical axis fixed on the left. This is the default
          value.
        * a tuple ``(x, y)`` specifying the exact intersection point.
        * ``'center'``: center of the current plot area.
        * ``'auto'``: the intersection point is automatically computed.

    camera : dict, optional
        A dictionary of keyword arguments that will be passed to the
        ``Axes3D.view_init`` method. Refer to [#fn9]_ for more information.

    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to Matplotlib's plot
        functions to customize the appearance of lines, surfaces, images,
        contours, quivers, streamlines...
        To learn more about customization:

        * Refer to [#fn1]_ to customize contour plots.
        * Refer to [#fn2]_ to customize image plots.
        * Refer to [#fn3]_ to customize solid line plots.
        * Refer to [#fn4]_ to customize colormap-based line plots.
        * Refer to [#fn5]_ to customize quiver plots.
        * Refer to [#fn6]_ to customize surface plots.
        * Refer to [#fn7]_ to customize stramline plots.
        * Refer to [#fn8]_ to customize 3D scatter plots.
        * Refer to [#fn11]_ to customize 2D arrows.

    axis : boolean, optional
        Turns on/off the axis visibility (and associated tick labels).
        Default to True (axis are visible).

    use_cm : boolean, optional
        If True, apply a color map to the mesh/surface or parametric lines.
        If False, solid colors will be used instead. Default to True.

    annotations : list, optional
        A list of dictionaries specifying the type of annotation
        required. The keys in the dictionary should be equivalent
        to the arguments of the `matplotlib.axes.Axes.annotate` method.
        This feature is experimental. It might get removed in the future.

    markers : list, optional
        A list of dictionaries specifying the type the markers required.
        The keys in the dictionary should be equivalent to the arguments
        of the `matplotlib.pyplot.plot()` function along with the marker
        related keyworded arguments.
        This feature is experimental. It might get removed in the future.

    rectangles : list, optional
        A list of dictionaries specifying the dimensions of the
        rectangles to be plotted. The keys in the dictionary should be
        equivalent to the arguments of the `matplotlib.patches.Rectangle`
        class.
        This feature is experimental. It might get removed in the future.

    fill : dict, optional
        A dictionary specifying the type of color filling required in
        the plot. The keys in the dictionary should be equivalent to the
        arguments of the `matplotlib.axes.Axes.fill_between` method.
        This feature is experimental. It might get removed in the future.


    References
    ==========
    .. [#fn1] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
    .. [#fn2] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    .. [#fn3] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    .. [#fn4] https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection
    .. [#fn5] https://matplotlib.org/stable/api/quiver_api.html#module-matplotlib.quiver
    .. [#fn6] https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface
    .. [#fn7] https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.streamplot.html#matplotlib.axes.Axes.streamplot
    .. [#fn8] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    .. [#fn9] https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D.view_init
    .. [#fn11] https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html

    See also
    ========

    Plot, PlotlyBackend, BokehBackend, K3DBackend
    """

    _library = "matplotlib"
    _allowed_keys = Plot._allowed_keys + [
        "markers", "annotations", "fill", "rectangles", "camera"]

    wireframe_color = "k"
    colormaps = []
    cyclic_colormaps = []

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
        ImplicitSeries: Implicit2DRenderer,
        ComplexDomainColoringSeries: ComplexRenderer,
        ComplexSurfaceSeries: ComplexRenderer,
        RiemannSphereSeries: ComplexRenderer,
        ContourSeries: ContourRenderer,
        SurfaceOver2DRangeSeries: SurfaceRenderer,
        ParametricSurfaceSeries: SurfaceRenderer,
        PlaneSeries: SurfaceRenderer,
        GeometrySeries: GeometryRenderer,
        GenericDataSeries: GenericRenderer,
        HVLineSeries: HVLineRenderer,
        NyquistLineSeries: NyquistRenderer,
        NicholsLineSeries: NicholsRenderer,
        Arrow2DSeries: Arrow2DRendererFancyArrowPatch
    }

    pole_line_kw = {"color": "k", "linestyle": ":"}

    def __init__(self, *args, **kwargs):
        self.matplotlib = import_module(
            'matplotlib',
            import_kwargs={
                'fromlist': [
                    'pyplot', 'cm', 'collections', 'colors', 'patches'
                ]
            },
            warn_not_installed=True,
            min_module_version='1.1.0',
            catch=(RuntimeError,))
        self.mpl_toolkits = import_module(
            'mpl_toolkits', # noqa
            import_kwargs={'fromlist': ['mplot3d']},
            catch=(RuntimeError,))
        self.np = import_module('numpy')
        self.plt = self.matplotlib.pyplot
        self.cm = cm = self.matplotlib.cm
        self.LineCollection = self.matplotlib.collections.LineCollection
        self.ListedColormap = self.matplotlib.colors.ListedColormap
        self.Line2D = self.matplotlib.lines.Line2D
        self.Rectangle = self.matplotlib.patches.Rectangle
        self.Normalize = self.matplotlib.colors.Normalize
        self.Line3DCollection = self.mpl_toolkits.mplot3d.art3d.Line3DCollection
        self.Path3DCollection = self.mpl_toolkits.mplot3d.art3d.Path3DCollection
        self.Axes3D = self.mpl_toolkits.mplot3d.Axes3D

        # set default colors
        self.colormaps = [
            cm.viridis, cm.autumn, cm.winter, cm.plasma, cm.jet,
            cm.gnuplot, cm.brg, cm.coolwarm, cm.cool, cm.summer]
        self.cyclic_colormaps = [cm.twilight, cm.hsv]
        # load default colorloop
        self.colorloop = self.plt.rcParams['axes.prop_cycle'].by_key()["color"]

        self._init_cyclers()
        super().__init__(*args, **kwargs)

        # set labels
        self._use_latex = kwargs.get(
            "use_latex", cfg["matplotlib"]["use_latex"])
        self._set_labels()
        self._set_title()

        if (
            (len([s for s in self._series if s.is_2Dline]) > 10) and
            (not type(self).colorloop) and
            not ("process_piecewise" in kwargs.keys())
        ):
            # add colors if needed
            self.colorloop = cm.tab20.colors

        # plotgrid() can provide its figure and axes to be populated with
        # the data from the series.
        self._plotgrid_fig = kwargs.pop("fig", None)
        self._plotgrid_ax = kwargs.pop("ax", None)

        if self.axis_center is None:
            self.axis_center = cfg["matplotlib"]["axis_center"]
        self.grid = kwargs.get("grid", cfg["matplotlib"]["grid"])
        self._show_minor_grid = kwargs.get(
            "show_minor_grid", cfg["matplotlib"]["show_minor_grid"])

        self._legend_handles = []

        # when using plotgrid, set imagegrid=True to require matplotlib to
        # use ImageGrid, which is suited to create equal aspect ratio axes
        # sharing colorbar
        self._imagegrid = kwargs.get("imagegrid", False)

        self._create_renderers()

    def _set_piecewise_color(self, s, color):
        """Set the color to the given series"""
        if "color" not in s.rendering_kw:
            # only set the color if the user didn't do that already
            s.rendering_kw["color"] = color

    @staticmethod
    def _do_sum_kwargs(p1, p2):
        return p1._copy_kwargs()

    def _init_cyclers(self):
        super()._init_cyclers()
        np = import_module('numpy')

        # For flexibily, spb.backends.utils.convert_colormap returns numpy
        # ndarrays whenever plotly/colorcet/k3d color map are given. Here we
        # create ListedColormap that can be used by Matplotlib
        def process_iterator(it, colormaps):
            cm = []
            for i in range(len(colormaps)):
                c = next(it)
                cm.append(
                    c if not isinstance(c, np.ndarray)
                    else self.ListedColormap(c)
                )
            return itertools.cycle(cm)

        self._cm = process_iterator(self._cm, self.colormaps)
        self._cyccm = process_iterator(self._cyccm, self.cyclic_colormaps)

    def _create_figure(self):
        if self._plotgrid_fig is not None:
            self._fig = self._plotgrid_fig
            self._ax = self._plotgrid_ax
        else:
            if self.is_iplot and (self.imodule == "panel"):
                self._fig = self.matplotlib.figure.Figure(figsize=self.size)
            else:
                self._fig = self.plt.figure(figsize=self.size)

            is_3D = [s.is_3D for s in self.series]
            if any(is_3D) and (not all(is_3D)):
                # allow sum of 3D plots with contour plots
                if not all(s.is_3D or s.is_contour for s in self.series):
                    raise ValueError(
                        "MatplotlibBackend can not mix 2D and 3D.")

            kwargs = {}
            if any(is_3D):
                kwargs["projection"] = "3d"
            elif (
                self.polar_axis and
                any(s.is_2Dline or s.is_contour for s in self.series)
            ):
                kwargs["projection"] = "polar"
            self._ax = self._fig.add_subplot(1, 1, 1, **kwargs)

    def _create_ax_if_not_available(self):
        if (not hasattr(self, "_ax")):
            # if the backend was created without showing it
            self.draw()

    @property
    def fig(self):
        """Returns the figure."""
        self._create_ax_if_not_available()
        return self._fig

    @property
    def ax(self):
        """Returns the axis used for the plot.

        Notes
        =====
        To get the axis of a colorbar, index ``p.fig.axes`` where ``p`` is a
        plot object. ``p.fig.axes[0]`` corresponds to ``p.ax``.
        """
        self._create_ax_if_not_available()
        return self._ax

    def _process_renderers(self):
        # XXX Workaround for matplotlib issue
        # https://github.com/matplotlib/matplotlib/issues/17130
        xlims, ylims, zlims = [], [], []

        self._ax.cla()
        self._init_cyclers()
        self._legend_handles = []

        for r, s in zip(self.renderers, self.series):
            self._check_supported_series(r, s)
            r.draw()
            xlims.extend(r._xlims)
            ylims.extend(r._ylims)
            zlims.extend(r._zlims)

        # Set global options.
        # TODO The 3D stuff
        # XXX The order of those is important.
        if self.xscale and not isinstance(self._ax, self.Axes3D):
            self._ax.set_xscale(self.xscale)
        if self.yscale and not isinstance(self._ax, self.Axes3D):
            self._ax.set_yscale(self.yscale)
        if self.axis_center:
            val = self.axis_center
            if isinstance(self._ax, self.Axes3D):
                pass
            elif val == "center":
                self._ax.spines["left"].set_position("center")
                self._ax.spines["bottom"].set_position("center")
                self._ax.yaxis.set_ticks_position("left")
                self._ax.xaxis.set_ticks_position("bottom")
                self._ax.spines["right"].set_visible(False)
                self._ax.spines["top"].set_visible(False)
            elif val == "auto":
                xl, xh = self._ax.get_xlim()
                yl, yh = self._ax.get_ylim()
                pos_left = ("data", 0) if xl * xh <= 0 else "center"
                pos_bottom = ("data", 0) if yl * yh <= 0 else "center"
                self._ax.spines["left"].set_position(pos_left)
                self._ax.spines["bottom"].set_position(pos_bottom)
                self._ax.yaxis.set_ticks_position("left")
                self._ax.xaxis.set_ticks_position("bottom")
                self._ax.spines["right"].set_visible(False)
                self._ax.spines["top"].set_visible(False)
            else:
                self._ax.spines["left"].set_position(("data", val[0]))
                self._ax.spines["bottom"].set_position(("data", val[1]))
                self._ax.yaxis.set_ticks_position("left")
                self._ax.xaxis.set_ticks_position("bottom")
                self._ax.spines["right"].set_visible(False)
                self._ax.spines["top"].set_visible(False)
        if not self.axis:
            self._ax.axis(False)
        if self.grid:
            if isinstance(self._ax, self.Axes3D):
                self._ax.grid()
            else:
                self._ax.grid(
                    visible=True, which='major', linestyle='-',
                    linewidth=0.75, color='0.75')
                self._ax.grid(
                    visible=True, which='minor', linestyle='--',
                    linewidth=0.6, color='0.825')
                if self._show_minor_grid:
                    self._ax.minorticks_on()
        if self.legend:
            if len(self._legend_handles) > 0:
                handles, _ = self._ax.get_legend_handles_labels()
                self._ax.legend(
                    handles=self._legend_handles + handles, loc="best")
            else:
                handles, _ = self._ax.get_legend_handles_labels()
                # Show the legend only if there are legend entries.
                # For example, if we are plotting only parametric expressions,
                # there will be only colorbars, no legend entries.
                if len(handles) > 0:
                    self._ax.legend(loc="best")
        if isinstance(self._ax, self.Axes3D):
            if self.camera is not None:
                self._ax.view_init(**self.camera)
        self._set_axes_texts()
        self._set_lims(xlims, ylims, zlims)
        self._set_aspect()

    def _set_axes_texts(self):
        title, xlabel, ylabel, zlabel = self._get_title_and_labels()

        if title:
            self._ax.set_title(title)
        if xlabel:
            self._ax.set_xlabel(
                xlabel, position=(1, 0) if self.axis_center else (0.5, 0)
            )
        if ylabel:
            self._ax.set_ylabel(
                ylabel, position=(0, 1) if self.axis_center else (0, 0.5)
            )
        if isinstance(self._ax, self.Axes3D):
            if zlabel:
                self._ax.set_zlabel(zlabel, position=(0, 1))

    def update_interactive(self, params):
        """Implement the logic to update the data generated by
        interactive-widget plots.

        Parameters
        ==========

        params : dict
            Map parameter (symbols) to numeric values.
        """
        # Because InteractivePlot doesn't call the show method, the following
        # line of code will add the numerical data (if not already present).
        if len(self.renderers) > 0 and len(self.renderers[0].handles) == 0:
            self.draw()

        xlims, ylims, zlims = [], [], []
        for r in self.renderers:
            if r.series.is_interactive:
                r.update(params)
            xlims.extend(r._xlims)
            ylims.extend(r._ylims)
            zlims.extend(r._zlims)

        self._set_axes_texts()

        # Update the plot limits according to the new data
        if not isinstance(self._ax, self.Axes3D):
            # https://stackoverflow.com/questions/10984085/automatically-rescale-ylim-and-xlim-in-matplotlib
            # recompute the ax.dataLim
            self._ax.relim()
            # update ax.viewLim using the new dataLim
            self._ax.autoscale_view()

        self._set_lims(xlims, ylims, zlims)

    def _set_aspect(self):
        aspect = self.aspect
        current_version = version.parse(self.matplotlib.__version__)
        v_3_6_0 = version.parse("3.6.0")
        if isinstance(aspect, str):
            if (aspect == "equal") and (current_version < v_3_6_0):
                if any(s.is_3D for s in self.series):
                    # plot_vector uses aspect="equal" by default. Older
                    # matplotlib versions do not support equal 3D axis.
                    aspect = "auto"
        elif hasattr(aspect, "__iter__"):
            aspect = float(aspect[1]) / aspect[0]
        else:
            aspect = "auto"
        self._ax.set_aspect(aspect)

    def _set_lims(self, xlims, ylims, zlims):
        np = self.np
        if not isinstance(self._ax, self.Axes3D):
            self._ax.autoscale_view(
                scalex=self._ax.get_autoscalex_on(),
                scaley=self._ax.get_autoscaley_on()
            )

            # HACK: in order to make interactive contour plots to scale to
            # the appropriate range
            if xlims and (
                any(s.is_contour for s in self.series)
                or any(s.is_vector and (not s.is_3D) for s in self.series)
                or any(s.is_2Dline and s.is_parametric for s in self.series)
            ):
                xlims = np.array(xlims)
                xlim = (np.nanmin(xlims[:, 0]), np.nanmax(xlims[:, 1]))
                self._ax.set_xlim(xlim)
            if ylims and (
                any(s.is_contour for s in self.series)
                or any(s.is_vector and (not s.is_3D) for s in self.series)
                or any(s.is_2Dline and s.is_parametric for s in self.series)
            ):
                ylims = np.array(ylims)
                ylim = (np.nanmin(ylims[:, 0]), np.nanmax(ylims[:, 1]))
                self._ax.set_ylim(ylim)
        else:
            # XXX Workaround for matplotlib issue
            # https://github.com/matplotlib/matplotlib/issues/17130
            if xlims:
                xlims = np.array(xlims)
                xlim = (np.nanmin(xlims[:, 0]), np.nanmax(xlims[:, 1]))
                self._ax.set_xlim(xlim)
            else:
                self._ax.set_xlim([0, 1])

            if ylims:
                ylims = np.array(ylims)
                ylim = (np.nanmin(ylims[:, 0]), np.nanmax(ylims[:, 1]))
                self._ax.set_ylim(ylim)
            else:
                self._ax.set_ylim([0, 1])

            if zlims:
                zlims = np.array(zlims)
                zlim = [np.nanmin(zlims[:, 0]), np.nanmax(zlims[:, 1])]
                if np.isnan(zlim[0]):
                    zlim[0] = -10
                    if not np.isnan(zlim[1]):
                        zlim[0] = zlim[1] - 10
                if np.isnan(zlim[1]):
                    zlim[1] = 10
                zlim = (-10 if np.isnan(z) else z for z in zlim)
                self._ax.set_zlim(zlim)
            else:
                self._ax.set_zlim([0, 1])

        if self._invert_x_axis:
            self._ax.invert_xaxis()

        # xlim and ylim should always be set at last so that plot limits
        # doesn't get altered during the process.
        if self.xlim:
            self._ax.set_xlim(self.xlim)
        if self.ylim:
            self._ax.set_ylim(self.ylim)
        if self.zlim:
            self._ax.set_zlim(self.zlim)

    def _add_colorbar(self, c, label, show_cb, norm=None, cmap=None):
        """Add a colorbar for the specificied collection

        Parameters
        ==========

        c : collection
        label : str
        show_cb : boolean
        """
        # design choice: instead of showing a legend entry (which
        # would require to work with proxy artists and custom
        # classes in order to create a gradient line), just show a
        # colorbar with the name of the expression on the side.
        if show_cb:
            if norm is None:
                cb = self._fig.colorbar(c, ax=self._ax)
            else:
                mappable = self.cm.ScalarMappable(cmap=cmap, norm=norm)
                cb = self._fig.colorbar(mappable, ax=self._ax)
            cb.set_label(label, rotation=90)
            return True
        return False

    def _update_colorbar(self, cax, cmap, label, param=None, norm=None):
        """Update a Matplotlib colorbar. The name is misleading, because
        updating a colorbar is non-trivial. Here, I create a new colorbar
        which will be placed on the same colorbar axis as the original.
        """
        np = self.np
        cax.clear()
        if norm is None:
            norm = self.Normalize(vmin=np.amin(param), vmax=np.amax(param))
        mappable = self.cm.ScalarMappable(cmap=cmap, norm=norm)
        self._fig.colorbar(
            mappable, orientation="vertical", label=label, cax=cax)

    def get_segments(self, x, y, z=None):
        """
        Convert two list of coordinates to a list of segments to be used
        with Matplotlib's LineCollection.

        Parameters
        ==========
        x: list
            List of x-coordinates

        y: list
            List of y-coordinates

        z: list (optional)
            List of z-coordinates for a 3D line.
        """
        np = self.np
        if z is not None:
            dim = 3
            points = (x, y, z)
        else:
            dim = 2
            points = (x, y)
        points = np.ma.array(points).T.reshape(-1, 1, dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def draw(self):
        """ Loop over the renderers, generates numerical data and add it to
        the figure. Note that this method doesn't show the plot.
        """
        # create the figure from scratch every time, otherwise if the plot was
        # previously shown, it would not be possible to show it again. This
        # behaviour is specific to Matplotlib
        self._create_figure()
        self._process_renderers()

    process_series = draw

    def show(self, **kwargs):
        """Display the current plot.

        Parameters
        ==========

        **kwargs : dict
            Keyword arguments to be passed to plt.show().
        """
        self.draw()
        if _show:
            self._fig.tight_layout()
            self.plt.show(**kwargs)
        else:
            self.close()

    def save(self, path, **kwargs):
        """Save the current plot at the specified location.

        Refer to [#fn10]_ to visualize all the available keyword arguments.

        References
        ==========

        .. [#fn10] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html

        """
        if self._fig is None:
            self.draw()
        self._fig.savefig(path, **kwargs)

    def close(self):
        """Close the current plot."""
        self.plt.close(self._fig)


MB = MatplotlibBackend
