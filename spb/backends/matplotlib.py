import itertools
from spb.defaults import cfg
from spb.backends.base_backend import Plot
from spb.backends.utils import compute_streamtubes
from sympy import latex
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


def _matplotlib_list(interval_list):
    """
    Returns lists for matplotlib `fill` command from a list of bounding
    rectangular intervals
    """
    xlist = []
    ylist = []
    if len(interval_list):
        for intervals in interval_list:
            intervalx = intervals[0]
            intervaly = intervals[1]
            xlist.extend(
                [intervalx.start, intervalx.start, intervalx.end, intervalx.end, None]
            )
            ylist.extend(
                [intervaly.start, intervaly.end, intervaly.end, intervaly.start, None]
            )
    else:
        # XXX Ugly hack. Matplotlib does not accept empty lists for `fill`
        xlist.extend([None, None, None, None])
        ylist.extend([None, None, None, None])
    return xlist, ylist


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

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.matplotlib = import_module(
            'matplotlib',
            import_kwargs={'fromlist': ['pyplot', 'cm', 'collections', 'colors']},
            warn_not_installed=True,
            min_module_version='1.1.0',
            catch=(RuntimeError,))
        self.plt = self.matplotlib.pyplot
        self.cm = cm = self.matplotlib.cm
        self.LineCollection = self.matplotlib.collections.LineCollection
        self.ListedColormap = self.matplotlib.colors.ListedColormap
        self.Line2D = self.matplotlib.lines.Line2D
        self.Rectangle = self.matplotlib.patches.Rectangle
        self.Normalize = self.matplotlib.colors.Normalize

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
        self._use_latex = kwargs.get("use_latex", cfg["matplotlib"]["use_latex"])
        self._set_labels()

        if ((len([s for s in self._series if s.is_2Dline]) > 10) and
            (not type(self).colorloop) and
            not ("process_piecewise" in kwargs.keys())):
            # add colors if needed
            self.colorloop = cm.tab20.colors

        # plotgrid() can provide its figure and axes to be populated with
        # the data from the series.
        self._plotgrid_fig = kwargs.pop("fig", None)
        self._plotgrid_ax = kwargs.pop("ax", None)

        if self.axis_center is None:
            self.axis_center = cfg["matplotlib"]["axis_center"]
        self.grid = kwargs.get("grid", cfg["matplotlib"]["grid"])
        self._show_minor_grid = kwargs.get("show_minor_grid", cfg["matplotlib"]["show_minor_grid"])

        self._handles = dict()
        self._legend_handles = []

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
                cm.append(c if not isinstance(c, np.ndarray) else self.ListedColormap(c))
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

            kwargs = dict()
            if any(is_3D):
                kwargs["projection"] = "3d"
            elif (self.polar_axis and
                any(s.is_2Dline or s.is_contour for s in self.series)):
                kwargs["projection"] = "polar"
            self._ax = self._fig.add_subplot(1, 1, 1, **kwargs)

    def _create_ax_if_not_available(self):
        if (not hasattr(self, "_ax")):
            # if the backend was created without showing it
            self.process_series()

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

    @staticmethod
    def get_segments(x, y, z=None):
        """
        Convert two list of coordinates to a list of segments to be used
        with Matplotlib's LineCollection.

        Parameters
        ==========
            x: list
                List of x-coordinates

            y: list
                List of y-coordinates

            z: list
                List of z-coordinates for a 3D line.
        """
        np = import_module('numpy')
        if z is not None:
            dim = 3
            points = (x, y, z)
        else:
            dim = 2
            points = (x, y)
        points = np.ma.array(points).T.reshape(-1, 1, dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def _add_colorbar(self, c, label, use_cm, override=False, norm=None, cmap=None):
        """Add a colorbar for the specificied collection

        Parameters
        ==========

        c : collection

        label : str

        override : boolean
            For parametric plots the colorbar acts like a legend. Hence,
            when legend=False we don't display the colorbar. However,
            for contour plots the colorbar is essential to understand it.
            Hence, to show it we set override=True.
            Default to False.
        """
        np = import_module('numpy')

        # design choice: instead of showing a legend entry (which
        # would require to work with proxy artists and custom
        # classes in order to create a gradient line), just show a
        # colorbar with the name of the expression on the side.
        if (self.legend and use_cm) or override:
            if norm is None:
                cb = self._fig.colorbar(c, ax=self._ax)
            else:
                mappable = self.cm.ScalarMappable(cmap=cmap, norm=norm)
                cb = self._fig.colorbar(mappable)
            cb.set_label(label, rotation=90)
            return True
        return False

    def _add_handle(self, i, h, kw=None, *args):
        """self._handle is a dictionary which will be used with iplot.
        In particular:
            key: integer corresponding to the i-th series.
            value: a list of two elements:
                1. handle of the object created by Matplotlib commands
                2. optionally, keyword arguments used to create the handle.
                    Some object can't be updated, hence we need to reconstruct
                    it from scratch at every update.
                3. anything else needed to reconstruct the object.
        """
        self._handles[i] = [h if not isinstance(h, (list, tuple)) else h[0], kw, *args]

    def _process_series(self, series):
        np = import_module('numpy')
        mpl_toolkits = import_module(
            'mpl_toolkits', # noqa
            import_kwargs={'fromlist': ['mplot3d']},
            catch=(RuntimeError,))
        Line3DCollection = mpl_toolkits.mplot3d.art3d.Line3DCollection
        merge = self.merge

        # XXX Workaround for matplotlib issue
        # https://github.com/matplotlib/matplotlib/issues/17130
        xlims, ylims, zlims = [], [], []

        self._ax.cla()
        self._init_cyclers()
        self._legend_handles = []

        for i, s in enumerate(series):
            kw = None

            if s.is_2Dline:
                if s.is_parametric:
                    x, y, param = s.get_data()
                else:
                    x, y = s.get_data()

                if s.is_parametric and s.use_cm:
                    colormap = (
                        next(self._cyccm)
                        if self._use_cyclic_cm(param, s.is_complex)
                        else next(self._cm)
                    )
                    if not s.is_point:
                        lkw = dict(array=param, cmap=colormap)
                        kw = merge({}, lkw, s.rendering_kw)
                        segments = self.get_segments(x, y)
                        c = self.LineCollection(segments, **kw)
                        self._ax.add_collection(c)
                    else:
                        lkw = dict(c=param, cmap=colormap)
                        kw = merge({}, lkw, s.rendering_kw)
                        c = self._ax.scatter(x, y, **kw)

                    is_cb_added = self._add_colorbar(c, s.get_label(self._use_latex), s.use_cm)
                    self._add_handle(i, c, kw, is_cb_added, self._fig.axes[-1])
                else:
                    color = next(self._cl) if s.line_color is None else s.line_color
                    lkw = dict(label=s.get_label(self._use_latex), color=color)
                    if s.is_point:
                        lkw["marker"] = "o"
                        lkw["linestyle"] = "None"
                        if not s.is_filled:
                            lkw["markerfacecolor"] = (1, 1, 1)
                    kw = merge({}, lkw, s.rendering_kw)
                    l = self._ax.plot(x, y, **kw)
                    self._add_handle(i, l)

            elif s.is_contour:
                x, y, z = s.get_data()
                ckw = dict(cmap=next(self._cm))
                if any(s.is_vector and (not s.is_streamlines) for s in self.series):
                    # NOTE:
                    # When plotting and updating a vector plot containing both
                    # a contour series and a quiver series, because it's not
                    # possible to update contour objects (we can only remove
                    # and recreating them), the quiver series which is usually
                    # after the contour plot (in terms of rendering order) will
                    # be moved on top, resulting in the contour to hide the
                    # quivers. Setting zorder appears to fix the problem.
                    ckw["zorder"] = 0
                kw = merge({}, ckw, s.rendering_kw)
                func = self._ax.contourf if s.is_filled else self._ax.contour
                c = func(x, y, z, **kw)
                clabel = None
                if s.is_filled:
                    self._add_colorbar(c, s.get_label(self._use_latex),
                        s.use_cm, True)
                else:
                    if s.show_clabels:
                        clabel = self._ax.clabel(c)
                self._add_handle(i, c, kw, self._fig.axes[-1], clabel)

            elif s.is_3Dline:
                if s.is_parametric:
                    x, y, z, param = s.get_data()
                else:
                    x, y, z = s.get_data()
                    param = np.ones_like(x)
                lkw = dict()

                if not s.is_point:
                    if s.use_cm:
                        segments = self.get_segments(x, y, z)
                        lkw["cmap"] = next(self._cm)
                        lkw["array"] = param
                        kw = merge({}, lkw, s.rendering_kw)
                        c = Line3DCollection(segments, **kw)
                        self._ax.add_collection(c)
                        self._add_colorbar(c, s.get_label(self._use_latex), s.use_cm)
                        self._add_handle(i, c, kw, self._fig.axes[-1])
                    else:
                        lkw["label"] = s.get_label(self._use_latex)
                        kw = merge({}, lkw, s.rendering_kw,
                            ({} if s.line_color is None
                            else {"color": s.line_color}) if s.show_in_legend
                            else {"color": self.wireframe_color})
                        l = self._ax.plot(x, y, z, **kw)
                        self._add_handle(i, l)
                else:
                    if s.use_cm:
                        lkw["cmap"] = next(self._cm)
                        lkw["c"] = param
                    else:
                        # lkw["c"] = param
                        lkw["color"] = next(self._cl) if s.line_color is None else s.line_color

                    if not s.is_filled:
                        lkw["facecolors"] = "none"

                    lkw["alpha"] = 1
                    kw = merge({}, lkw, s.rendering_kw)
                    l = self._ax.scatter(x, y, z, **kw)
                    if s.use_cm:
                        self._add_colorbar(l, s.get_label(self._use_latex), s.use_cm)
                        self._add_handle(i, l, kw, self._fig.axes[-1])
                    else:
                        self._add_handle(i, l)
                xlims.append((np.amin(x), np.amax(x)))
                ylims.append((np.amin(y), np.amax(y)))
                zlims.append((np.amin(z), np.amax(z)))

            elif (s.is_3Dsurface and (not s.is_domain_coloring) and (not s.is_implicit)):
                if not s.is_parametric:
                    x, y, z = self.series[i].get_data()
                    facecolors = s.eval_color_func(x, y, z)
                else:
                    x, y, z, u, v = self.series[i].get_data()
                    facecolors = s.eval_color_func(x, y, z, u, v)
                skw = dict(rstride=1, cstride=1, linewidth=0.1)
                norm, cmap = None, None
                if s.use_cm:
                    vmin = s.rendering_kw.get("vmin", np.amin(facecolors))
                    vmax = s.rendering_kw.get("vmax", np.amax(facecolors))
                    norm = self.Normalize(vmin=vmin, vmax=vmax)
                    cmap = next(self._cm)
                    skw["cmap"] = cmap
                else:
                    skw["color"] = next(self._cl) if s.surface_color is None else s.surface_color
                    proxy_artist = self.Rectangle((0, 0), 1, 1,
                        color=skw["color"], label=s.get_label(self._use_latex))
                    self._legend_handles.append(proxy_artist)

                kw = merge({}, skw, s.rendering_kw)
                if s.use_cm:
                    # facecolors must be computed here because s.rendering_kw
                    # might have its own cmap
                    cmap = kw["cmap"]
                    if isinstance(cmap, str):
                        cmap = self.cm.get_cmap(cmap)
                    kw["facecolors"] = cmap(norm(facecolors))
                c = self._ax.plot_surface(x, y, z, **kw)
                is_cb_added = self._add_colorbar(c, s.get_label(self._use_latex), s.use_cm, norm=norm, cmap=cmap)
                self._add_handle(i, c, kw, is_cb_added, self._fig.axes[-1])
                xlims.append((np.amin(x), np.amax(x)))
                ylims.append((np.amin(y), np.amax(y)))
                zlims.append((np.amin(z), np.amax(z)))

            elif s.is_implicit and not s.is_3Dsurface:
                points = s.get_data()
                color = next(self._cl) if (s.color is None) or isinstance(s.color, bool) else s.color
                if len(points) == 2:
                    # interval math plotting
                    x, y = _matplotlib_list(points[0])
                    fkw = {"color": color, "edgecolor": "None"}
                    kw = merge({}, fkw, s.rendering_kw)
                    c = self._ax.fill(x, y, **kw)
                    self._add_handle(i, c, kw)
                    proxy_artist = self.Rectangle((0, 0), 1, 1,
                        color=kw["color"], label=s.get_label(self._use_latex))
                else:
                    # use contourf or contour depending on whether it is
                    # an inequality or equality.
                    xarray, yarray, zarray, plot_type = points
                    if plot_type == "contour":
                        colormap = self.ListedColormap([color, color])
                        ckw = dict(cmap=colormap)
                        kw = merge({}, ckw, s.rendering_kw)
                        c = self._ax.contour(xarray, yarray, zarray, [0.0],
                            **kw)
                        proxy_artist = self.Line2D([], [],
                            color=color, label=s.get_label(self._use_latex))
                    else:
                        colormap = self.ListedColormap(["#ffffff00", color])
                        ckw = dict(cmap=colormap, levels=[-1e-15, 0, 1e-15],
                            extend="both")
                        kw = merge({}, ckw, s.rendering_kw)
                        c = self._ax.contourf(xarray, yarray, zarray, **kw)
                        proxy_artist = self.Rectangle((0, 0), 1, 1,
                            color=color, label=s.get_label(self._use_latex))
                    self._add_handle(i, c, kw)

                if s.show_in_legend:
                    self._legend_handles.append(proxy_artist)

            elif s.is_vector:
                if s.is_2Dvector:
                    xx, yy, uu, vv = s.get_data()
                    mag = np.sqrt(uu ** 2 + vv ** 2)
                    uu0, vv0 = [t.copy() for t in [uu, vv]]
                    if s.normalize:
                        uu, vv = [t / mag for t in [uu, vv]]
                    if s.is_streamlines:
                        skw = dict()
                        if (not s.use_quiver_solid_color) and s.use_cm:
                            color_val = mag
                            if s.color_func is not None:
                                color_val = s.eval_color_func(xx, yy, uu0, vv0)
                            skw["cmap"] = next(self._cm)
                            skw["color"] = color_val
                            kw = merge({}, skw, s.rendering_kw)
                            sp = self._ax.streamplot(xx, yy, uu, vv, **kw)
                            is_cb_added = self._add_colorbar(
                                sp.lines, s.get_label(self._use_latex), s.use_cm)
                        else:
                            skw["color"] = next(self._cl)
                            kw = merge({}, skw, s.rendering_kw)
                            sp = self._ax.streamplot(xx, yy, uu, vv, **kw)
                            is_cb_added = False
                        self._add_handle(i, sp, kw, is_cb_added,
                            self._fig.axes[-1])
                    else:
                        qkw = dict()
                        if any(s.is_contour for s in self.series):
                            # NOTE:
                            # When plotting and updating a vector plot
                            # containing both a contour series and a quiver
                            # series, because it's not possible to update
                            # contour objects (we can only remove and
                            # recreating them), the quiver series which is
                            # usually after the contour plot (in terms of
                            # rendering order) will be moved on top, resulting
                            # in the contour to hide the quivers. Setting
                            # zorder appears to fix the problem.
                            qkw["zorder"] = 1
                        if (not s.use_quiver_solid_color) and s.use_cm:
                            # don't use color map if a scalar field is
                            # visible or if use_cm=False
                            color_val = mag
                            if s.color_func is not None:
                                color_val = s.eval_color_func(xx, yy, uu0, vv0)
                            qkw["cmap"] = next(self._cm)
                            kw = merge({}, qkw, s.rendering_kw)
                            q = self._ax.quiver(xx, yy, uu, vv, color_val, **kw)
                            is_cb_added = self._add_colorbar(
                                q, s.get_label(self._use_latex), s.use_cm)
                        else:
                            is_cb_added = False
                            qkw["color"] = next(self._cl)
                            kw = merge({}, qkw, s.rendering_kw)
                            q = self._ax.quiver(xx, yy, uu, vv, **kw)
                        self._add_handle(i, q, kw, is_cb_added,
                            self._fig.axes[-1])
                else:
                    xx, yy, zz, uu, vv, ww = s.get_data()
                    mag = np.sqrt(uu ** 2 + vv ** 2 + ww ** 2)

                    uu0, vv0, zz0 = [t.copy() for t in [uu, vv, ww]]
                    if s.normalize:
                        uu, vv, ww = [t / mag for t in [uu, vv, ww]]

                    if s.is_streamlines:
                        vertices, color_val = compute_streamtubes(
                            xx, yy, zz, uu, vv, ww, s.rendering_kw,
                            s.color_func)

                        lkw = dict()
                        stream_kw = s.rendering_kw.copy()
                        # remove rendering-unrelated keywords
                        for k in ["starts", "max_prop", "npoints", "radius"]:
                            if k in stream_kw.keys():
                                stream_kw.pop(k)

                        if s.use_cm:
                            segments = self.get_segments(
                                vertices[:, 0], vertices[:, 1], vertices[:, 2])
                            lkw["cmap"] = next(self._cm)
                            lkw["array"] = color_val
                            kw = merge({}, lkw, stream_kw)
                            c = Line3DCollection(segments, **kw)
                            self._ax.add_collection(c)
                            self._add_colorbar(c, s.get_label(self._use_latex), s.use_cm)
                            self._add_handle(i, c)
                        else:
                            lkw["label"] = s.get_label(self._use_latex)
                            kw = merge({}, lkw, stream_kw)
                            l = self._ax.plot(vertices[:, 0], vertices[:, 1],
                                vertices[:, 2], **kw)
                            self._add_handle(i, l)

                        xlims.append((np.amin(xx), np.amax(xx)))
                        ylims.append((np.amin(yy), np.amax(yy)))
                        zlims.append((np.amin(zz), np.amax(zz)))
                    else:
                        qkw = dict()
                        if s.use_cm:
                            # NOTE: each quiver is composed of 3 lines: the
                            # stem and two segments for the head. I could set
                            # the colors keyword argument in order to apply
                            # the same color to the entire quiver, like this:
                            # [c1, c2, ..., cn, c1, c1, c2, c2, ... cn, cn]
                            # However, it doesn't appear to work reliably, so
                            # I'll keep things simpler.
                            color_val = mag
                            if s.color_func is not None:
                                color_val = s.eval_color_func(
                                    xx, yy, zz, uu0, vv0, zz0)
                            qkw["cmap"] = next(self._cm)
                            qkw["array"] = color_val.flatten()
                            kw = merge({}, qkw, s.rendering_kw)
                            q = self._ax.quiver(xx, yy, zz, uu, vv, ww, **kw)
                            is_cb_added = self._add_colorbar(
                                q, s.get_label(self._use_latex), s.use_cm)
                        else:
                            qkw["color"] = next(self._cl)
                            kw = merge({}, qkw, s.rendering_kw)
                            q = self._ax.quiver(xx, yy, zz, uu, vv, ww, **kw)
                            is_cb_added = False
                        self._add_handle(i, q, kw, is_cb_added, self._fig.axes[-1])
                    xlims.append((np.amin(xx), np.amax(xx)))
                    ylims.append((np.amin(yy), np.amax(yy)))
                    zlims.append((np.nanmin(zz), np.nanmax(zz)))

            elif s.is_complex:
                if not s.is_3Dsurface:
                    x, y, _, _, img, colors = s.get_data()
                    ikw = dict(
                        extent=[np.amin(x), np.amax(x), np.amin(y), np.amax(y)],
                        interpolation="nearest",
                        origin="lower",
                    )
                    kw = merge({}, ikw, s.rendering_kw)
                    image = self._ax.imshow(img, **kw)
                    self._add_handle(i, image, kw)

                    # chroma/phase-colorbar
                    if colors is not None:
                        colors = colors / 255.0

                        colormap = self.ListedColormap(colors)
                        norm = self.Normalize(vmin=-np.pi, vmax=np.pi)
                        cb2 = self._fig.colorbar(
                            self.cm.ScalarMappable(norm=norm, cmap=colormap),
                            orientation="vertical",
                            label="Argument",
                            ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                            ax=self._ax,
                        )
                        cb2.ax.set_yticklabels(
                            [r"-$\pi$", r"-$\pi / 2$", "0", r"$\pi / 2$", r"$\pi$"]
                        )
                else:
                    x, y, mag, arg, facecolors, colorscale = s.get_data()

                    skw = dict(rstride=1, cstride=1, linewidth=0.1)
                    if s.use_cm:
                        skw["facecolors"] = facecolors / 255
                    else:
                        skw["color"] = next(self._cl) if s.surface_color is None else s.surface_color
                    kw = merge({}, skw, s.rendering_kw)
                    c = self._ax.plot_surface(x, y, mag, **kw)

                    if s.use_cm and (colorscale is not None):
                        if len(colorscale.shape) == 3:
                            colorscale = colorscale.reshape((-1, 3))
                        else:
                            colorscale = colorscale / 255.0

                        # this colorbar is essential to understand the plot.
                        # Always show it, except when use_cm=False
                        norm = self.Normalize(vmin=-np.pi, vmax=np.pi)
                        mappable = self.cm.ScalarMappable(
                            cmap=self.ListedColormap(colorscale), norm=norm
                        )
                        cb = self._fig.colorbar(
                            mappable,
                            orientation="vertical",
                            label="Argument",
                            ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                            ax=self._ax,
                        )
                        cb.ax.set_yticklabels(
                            [r"-$\pi$", r"-$\pi / 2$", "0", r"$\pi / 2$", r"$\pi$"]
                        )
                    self._add_handle(i, c, kw)
                    xlims.append((np.amin(x), np.amax(x)))
                    ylims.append((np.amin(y), np.amax(y)))
                    zlims.append((np.amin(mag), np.amax(mag)))

            elif s.is_geometry:
                x, y = s.get_data()
                color = next(self._cl)
                fkw = dict(facecolor=color, fill=s.is_filled, edgecolor=color)
                kw = merge({}, fkw, s.rendering_kw)
                c = self._ax.fill(x, y, **kw)
                self._add_handle(i, c, kw)

            elif s.is_generic:
                if s.type == "markers":
                    kw = merge({}, {"color": next(self._cl)}, s.rendering_kw)
                    self._ax.plot(*s.args, **kw)
                elif s.type == "annotations":
                    self._ax.annotate(*s.args, **s.rendering_kw)
                elif s.type == "fill":
                    kw = merge({}, {"color": next(self._cl)}, s.rendering_kw)
                    self._ax.fill_between(*s.args, **kw)
                elif s.type == "rectangles":
                    kw = merge({}, {"color": next(self._cl)}, s.rendering_kw)
                    self._ax.add_patch(
                        self.matplotlib.patches.Rectangle(*s.args, **kw))

            else:
                raise NotImplementedError(
                    "{} is not supported by {}\n".format(type(s), type(self).__name__)
                )

        Axes3D = mpl_toolkits.mplot3d.Axes3D

        # Set global options.
        # TODO The 3D stuff
        # XXX The order of those is important.
        if self.xscale and not isinstance(self._ax, Axes3D):
            self._ax.set_xscale(self.xscale)
        if self.yscale and not isinstance(self._ax, Axes3D):
            self._ax.set_yscale(self.yscale)
        if self.axis_center:
            val = self.axis_center
            if isinstance(self._ax, Axes3D):
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
        if self.grid:
            if isinstance(self._ax, Axes3D):
                self._ax.grid()
            else:
                self._ax.grid(visible=True, which='major', linestyle='-',
                    linewidth=0.75, color='0.75')
                self._ax.grid(visible=True, which='minor', linestyle='--',
                    linewidth=0.6, color='0.825')
                if self._show_minor_grid:
                    self._ax.minorticks_on()
        if self.legend:
            if len(self._legend_handles) > 0:
                self._ax.legend(handles=self._legend_handles, loc="best")
            else:
                handles, _ = self._ax.get_legend_handles_labels()
                # Show the legend only if there are legend entries.
                # For example, if we are plotting only parametric expressions,
                # there will be only colorbars, no legend entries.
                if len(handles) > 0:
                    self._ax.legend(loc="best")
        if self.title:
            self._ax.set_title(self.title)
        if self.xlabel:
            self._ax.set_xlabel(
                self.xlabel, position=(1, 0) if self.axis_center else (0.5, 0)
            )
        if self.ylabel:
            self._ax.set_ylabel(
                self.ylabel, position=(0, 1) if self.axis_center else (0, 0.5)
            )
        if isinstance(self._ax, Axes3D):
            if self.zlabel:
                self._ax.set_zlabel(self.zlabel, position=(0, 1))
            if self.camera is not None:
                self._ax.view_init(**self.camera)

        self._set_lims(xlims, ylims, zlims)
        self._set_aspect()

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

    def _get_plotting_func_name(self, t):
        if t == "markers":
            return "scatter"
        elif t == "annotations":
            return "annotate"
        elif t == "fills":
            return "fill"
        elif t == "rectangles":
            return "rect"
        raise ValueError("%s is not supported by MatplotlibBackend" % t)

    def _set_lims(self, xlims, ylims, zlims):
        np = import_module('numpy')
        mpl_toolkits = import_module(
            'mpl_toolkits', # noqa
            import_kwargs={'fromlist': ['mplot3d']},
            catch=(RuntimeError,))
        Axes3D = mpl_toolkits.mplot3d.Axes3D
        if not isinstance(self._ax, Axes3D):
            self._ax.autoscale_view(
                scalex=self._ax.get_autoscalex_on(), scaley=self._ax.get_autoscaley_on()
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

        # xlim and ylim should always be set at last so that plot limits
        # doesn't get altered during the process.
        if self.xlim:
            self._ax.set_xlim(self.xlim)
        if self.ylim:
            self._ax.set_ylim(self.ylim)
        if self.zlim:
            self._ax.set_zlim(self.zlim)

    def _update_colorbar(self, cax, cmap, label, param=None, norm=None):
        """This method reduces code repetition.
        The name is misleading: here we create a new colorbar which will be
        placed on the same colorbar axis as the original.
        """
        np = import_module('numpy')
        cax.clear()
        if norm is None:
            norm = self.Normalize(vmin=np.amin(param), vmax=np.amax(param))
        mappable = self.cm.ScalarMappable(cmap=cmap, norm=norm)
        self._fig.colorbar(mappable, orientation="vertical", label=label, cax=cax)

    def update_interactive(self, params):
        """Implement the logic to update the data generated by
        interactive-widget plots.

        Parameters
        ==========

        params : dict
            Map parameter-symbols to numeric values.
        """
        np = import_module('numpy')
        mpl_toolkits = import_module(
            'mpl_toolkits', # noqa
            import_kwargs={'fromlist': ['mplot3d']},
            catch=(RuntimeError,))
        Line3DCollection = mpl_toolkits.mplot3d.art3d.Line3DCollection
        Path3DCollection = mpl_toolkits.mplot3d.art3d.Path3DCollection

        # iplot doesn't call the show method. The following line of
        # code will add the numerical data (if not already present).
        if len(self._handles) == 0:
            self.process_series()

        xlims, ylims, zlims = [], [], []
        for i, s in enumerate(self.series):
            if s.is_interactive:
                self.series[i].params = params
                if s.is_2Dline:
                    if s.is_parametric and s.use_cm:
                        x, y, param = self.series[i].get_data()
                        kw, is_cb_added, cax = self._handles[i][1:]

                        if not s.is_point:
                            segments = self.get_segments(x, y)
                            self._handles[i][0].set_segments(segments)
                        else:
                            self._handles[i][0].set_offsets(np.c_[x,y])

                        self._handles[i][0].set_array(param)
                        self._handles[i][0].set_clim(
                            vmin=min(param), vmax=max(param))

                        if is_cb_added:
                            norm = self.Normalize(vmin=np.amin(param), vmax=np.amax(param))
                            self._update_colorbar(cax, kw["cmap"], s.get_label(self._use_latex), norm=norm)
                        xlims.append((np.amin(x), np.amax(x)))
                        ylims.append((np.amin(y), np.amax(y)))
                    else:
                        if s.is_parametric:
                            x, y, param = self.series[i].get_data()
                        else:
                            x, y = self.series[i].get_data()
                        # TODO: Point2D are updated but not visible.
                        self._handles[i][0].set_data(x, y)

                elif s.is_3Dline:
                    if s.is_parametric:
                        x, y, z, param = self.series[i].get_data()
                    else:
                        x, y, z = self.series[i].get_data()

                    if isinstance(self._handles[i][0], Line3DCollection):
                        # gradient lines
                        segments = self.get_segments(x, y, z)
                        self._handles[i][0].set_segments(segments)
                    elif isinstance(self._handles[i][0], Path3DCollection):
                        # 3D points
                        self._handles[i][0]._offsets3d = (x, y, z)
                    else:
                        if hasattr(self._handles[i][0], "set_data_3d"):
                            # solid lines
                            self._handles[i][0].set_data_3d(x, y, z)
                        else:
                            # scatter
                            self._handles[i][0].set_offset(np.c_[x, y, z])

                    if s.is_parametric and s.use_cm:
                        self._handles[i][0].set_array(param)
                        kw, cax = self._handles[i][1:]
                        self._update_colorbar(cax, kw["cmap"], s.get_label(self._use_latex), param=param)
                    xlims.append((np.amin(x), np.amax(x)))
                    ylims.append((np.amin(y), np.amax(y)))
                    zlims.append((np.amin(z), np.amax(z)))

                elif s.is_contour and (not s.is_complex):
                    x, y, z = self.series[i].get_data()
                    kw, cax, clabels = self._handles[i][1:]
                    for c in self._handles[i][0].collections:
                        c.remove()
                    if (not s.is_filled) and s.show_clabels:
                        for cl in clabels:
                            cl.remove()
                    func = self._ax.contourf if s.is_filled else self._ax.contour
                    self._handles[i][0] = func(x, y, z, **kw)
                    if s.is_filled:
                        self._update_colorbar(cax, kw["cmap"], s.get_label(self._use_latex), param=z)
                    else:
                        if s.show_clabels:
                            clabels = self._ax.clabel(self._handles[i][0])
                            self._handles[i][-1] = clabels
                    xlims.append((np.amin(x), np.amax(x)))
                    ylims.append((np.amin(y), np.amax(y)))

                elif s.is_3Dsurface and (not s.is_domain_coloring) and (not s.is_implicit):
                    if not s.is_parametric:
                        x, y, z = self.series[i].get_data()
                        facecolors = s.eval_color_func(x, y, z)
                    else:
                        x, y, z, u, v = self.series[i].get_data()
                        facecolors = s.eval_color_func(x, y, z, u, v)
                    # TODO: by setting the keyword arguments, somehow the
                    # update becomes really really slow.
                    kw, is_cb_added, cax = self._handles[i][1:]

                    if is_cb_added:
                        # TODO: if use_cm=True and a single 3D expression is
                        # shown with legend=False, this won't get executed.
                        # In widget plots, the surface will never change color.
                        vmin = s.rendering_kw.get("vmin", np.amin(facecolors))
                        vmax = s.rendering_kw.get("vmax", np.amax(facecolors))
                        norm = self.Normalize(vmin=vmin, vmax=vmax)
                        cmap = kw["cmap"]
                        if isinstance(cmap, str):
                            cmap = self.cm.get_cmap(cmap)
                        kw["facecolors"] = cmap(norm(facecolors))
                    self._handles[i][0].remove()
                    self._handles[i][0] = self._ax.plot_surface(
                        x, y, z, **kw)

                    if is_cb_added:
                        self._update_colorbar(cax, kw["cmap"], s.get_label(self._use_latex), norm=norm)
                    xlims.append((np.amin(x), np.amax(x)))
                    ylims.append((np.amin(y), np.amax(y)))
                    zlims.append((np.amin(z), np.amax(z)))

                elif s.is_implicit and not s.is_3Dsurface:
                    points = s.get_data()
                    if len(points) == 2:
                        raise NotImplementedError
                    else:
                        for c in self._handles[i][0].collections:
                            c.remove()
                        xx, yy, zz, plot_type = points
                        kw = self._handles[i][1]
                        if plot_type == "contour":
                            self._handles[i][0] = self._ax.contour(
                                xx, yy, zz, [0.0], **kw
                            )
                        else:
                            self._handles[i][0] = self._ax.contourf(
                                xx, yy, zz, **kw)
                        xlims.append((np.amin(xx), np.amax(xx)))
                        ylims.append((np.amin(yy), np.amax(yy)))

                elif s.is_vector and s.is_3D:
                    if s.is_streamlines:
                        raise NotImplementedError

                    xx, yy, zz, uu, vv, ww = self.series[i].get_data()
                    kw, is_cb_added, cax = self._handles[i][1:]
                    mag = np.sqrt(uu ** 2 + vv ** 2 + ww ** 2)
                    uu0, vv0, ww0 = [t.copy() for t in [uu, vv, ww]]
                    if s.normalize:
                        uu, vv, ww = [t / mag for t in [uu, vv, ww]]
                    self._handles[i][0].remove()

                    if "array" in kw.keys():
                        color_val = mag
                        if s.color_func is not None:
                            color_val = s.eval_color_func(xx, yy, zz, uu0, vv0, ww0)
                        kw["array"] = color_val.flatten()

                    self._handles[i][0] = self._ax.quiver(xx, yy, zz, uu, vv, ww, **kw)

                    if is_cb_added:
                        self._update_colorbar(cax, kw["cmap"], s.get_label(self._use_latex), param=mag)
                    xlims.append((np.amin(xx), np.amax(xx)))
                    ylims.append((np.amin(yy), np.amax(yy)))
                    zlims.append((np.nanmin(zz), np.nanmax(zz)))

                elif s.is_vector:
                    xx, yy, uu, vv = self.series[i].get_data()
                    mag = np.sqrt(uu ** 2 + vv ** 2)
                    uu0, vv0 = [t.copy() for t in [uu, vv]]
                    if s.normalize:
                        uu, vv = [t / mag for t in [uu, vv]]
                    if s.is_streamlines:
                        raise NotImplementedError

                        # Streamlines are composed by lines and arrows.
                        # Arrows belongs to a PatchCollection. Currently,
                        # there is no way to remove a PatchCollection....
                        kw = self._handles[i][1]
                        self._handles[i][0].lines.remove()
                        self._handles[i][0].arrows.remove()
                        self._handles[i][0] = self._ax.streamplot(xx, yy, uu, vv, **kw)
                    else:
                        kw, is_cb_added, cax = self._handles[i][1:]
                        color_val = mag
                        if s.color_func is not None:
                            color_val = s.eval_color_func(xx, yy, uu0, vv0)

                        if is_cb_added:
                            self._handles[i][0].set_UVC(uu, vv, color_val)
                            self._update_colorbar(cax, kw["cmap"], s.get_label(self._use_latex), color_val)
                        else:
                            self._handles[i][0].set_UVC(uu, vv)
                        self._handles[i][0].set_offsets(np.c_[xx.flatten(), yy.flatten()])
                    xlims.append((np.amin(xx), np.amax(xx)))
                    ylims.append((np.amin(yy), np.amax(yy)))

                elif s.is_complex:
                    if not s.is_3Dsurface:
                        x, y, _, _, img, colors = s.get_data()
                        self._handles[i][0].set_data(img)
                        self._handles[i][0].set_extent((x.min(), x.max(), y.min(), y.max()))
                    else:
                        x, y, mag, arg, facecolors, colorscale = s.get_data()
                        self._handles[i][0].remove()
                        kw = self._handles[i][1]
                        if s.use_cm:
                            kw["facecolors"] = facecolors / 255
                        self._handles[i][0] = self._ax.plot_surface(x, y, mag, **kw)
                        xlims.append((np.amin(x), np.amax(x)))
                        ylims.append((np.amin(y), np.amax(y)))
                        zlims.append((np.amin(mag), np.amax(mag)))

                elif s.is_geometry and not (s.is_2Dline):
                    x, y = self.series[i].get_data()
                    self._handles[i][0].remove()
                    self._handles[i][0] = self._ax.fill(x, y, **self._handles[i][1])[0]

        # Update the plot limits according to the new data
        Axes3D = mpl_toolkits.mplot3d.Axes3D
        if not isinstance(self._ax, Axes3D):
            # https://stackoverflow.com/questions/10984085/automatically-rescale-ylim-and-xlim-in-matplotlib
            # recompute the ax.dataLim
            self._ax.relim()
            # update ax.viewLim using the new dataLim
            self._ax.autoscale_view()
        else:
            pass

        self._set_lims(xlims, ylims, zlims)

    def process_series(self):
        """ Loop over data series, generates numerical data and add it to the
        figure.
        """
        # create the figure from scratch every time, otherwise if the plot was
        # previously shown, it would not be possible to show it again. This
        # behaviour is specific to Matplotlib
        self._create_figure()
        self._process_series(self.series)

    def show(self, **kwargs):
        """Display the current plot.

        Parameters
        ==========

        **kwargs : dict
            Keyword arguments to be passed to plt.show().
        """
        self.process_series()
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
            self.process_series()
        self._fig.savefig(path, **kwargs)

    def close(self):
        """Close the current plot."""
        self.plt.close(self._fig)


MB = MatplotlibBackend
