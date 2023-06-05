import itertools
import os
from spb.defaults import cfg
from spb.backends.base_backend import Plot
from spb.backends.utils import get_seeds_points
from spb.series import GenericDataSeries
from sympy.external import import_module
import warnings


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

    def __init__(self, *args, **kwargs):
        plotly = import_module(
            'plotly',
            import_kwargs={'fromlist': ['graph_objects', 'figure_factory']},
            warn_not_installed=True,
            min_module_version='5.0.0')
        go = plotly.graph_objects

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
        super().__init__(*args, **kwargs)

        # NOTE: Plotly 3D currently doesn't support latex labels
        # https://github.com/plotly/plotly.js/issues/608
        self._use_latex = kwargs.get("use_latex", cfg["plotly"]["use_latex"])
        self._set_labels()

        if ((len([s for s in self._series if s.is_2Dline]) > 10) and
            (not type(self).colorloop) and
            not ("process_piecewise" in kwargs.keys())):
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
            self._fig = go.FigureWidget()
        else:
            self._fig = go.Figure()
        self._colorbar_counter = 0

        if self.aouc:
            # assumption: there is only one data series being plotted.
            at_infinity = self.series[0].at_infinity
            sign = 1 if not at_infinity else -1
            labels = ["1", "i", "-i"]
            labels = ["<b>%s</b>" % t for t in labels]

            new_series = [
                GenericDataSeries("markers",
                    x=[sign, 0, 0], y=[0, 1, -1], mode="markers+text",
                    text=labels, marker=dict(color="#E5ECF6", size=8,
                        line=dict(width=2, color="black")),
                    textposition=[
                        "top right" if not at_infinity else "top left",
                        "bottom center", "top center"],
                    textfont=dict(size=15), showlegend=False),
                GenericDataSeries("markers",
                    x=[0], y=[0], mode="markers+text",
                    text="<b>inf</b>" if at_infinity else "<b>0</b>",
                    marker=dict(color="#E5ECF6", size=8,
                        line=dict(width=2, color="black")) if at_infinity
                        else dict(size=8, color="black"),
                    textposition="top right", textfont=dict(size=15),
                    showlegend=False),
            ]
            self._series = self._series + new_series

    @property
    def fig(self):
        """Returns the figure."""
        if len(self.series) != len(self._fig.data):
            # if the backend was created without showing it
            self.process_series()
        return self._fig

    def process_series(self):
        """ Loop over data series, generates numerical data and add it to the
        figure.
        """
        # this is necessary in order for the series to be added even if
        # show=False
        self._process_series(self._series)
        self._update_layout()

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
        quivers_colors = self.quivers_colors if not tb.quivers_colors else tb.quivers_colors
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
            len=self._cbsdf if (sc and (self.legend or (self.legend is None))) else 1,
            yanchor="bottom",
            y=0,
        )

    def _solid_colorscale(self, s):
        # create a solid color to be used when s.use_cm=False
        col = s.line_color
        if col is None:
            col = next(self._cl)
        return [[0, col], [1, col]]

    def _scatter_class(self, go, n, polar=False):
        if not polar:
            return go.Scatter if n < self.scattergl_threshold else go.Scattergl
        return go.Scatterpolar if n < self.scattergl_threshold else go.Scatterpolargl

    def _process_series(self, series):
        np = import_module('numpy')
        plotly = import_module(
            'plotly',
            import_kwargs={'fromlist': ['graph_objects', 'figure_factory']},
            min_module_version='5.0.0')
        go = plotly.graph_objects
        create_quiver = plotly.figure_factory.create_quiver
        create_streamline = plotly.figure_factory.create_streamline
        merge = self.merge
        self._init_cyclers()

        mix_3Dsurfaces_3Dlines = (any(s.is_3Dsurface for s in series) and
            any(s.is_3Dline and s.show_in_legend for s in series))
        show_2D_vectors = any(s.is_2Dvector for s in series)

        self._fig.data = []

        count = 0
        for ii, s in enumerate(series):
            kw = None

            if s.is_2Dline:
                if s.is_parametric:
                    x, y, param = s.get_data()
                    # hides/show the colormap depending on s.use_cm
                    mode = "lines+markers" if not s.is_point else "markers"
                    if (not s.is_point) and (not s.use_cm):
                        mode = "lines"
                    if s.get_label(False) != "__k__":
                        color = next(self._cl) if s.line_color is None else s.line_color
                    else:
                        color = "black"
                    # hover template
                    ht = (
                        "x: %{x}<br />y: %{y}<br />u: %{customdata}"
                        if not s.is_complex
                        else "x: %{x}<br />y: %{y}<br />Arg: %{customdata}"
                    )
                    if s.is_polar:
                        ht = "r: %{r}<br />θ: %{theta}<br />u: %{customdata}"

                    lkw = dict(
                        name=s.get_label(self._use_latex),
                        line_color=color,
                        mode=mode,
                        customdata=param,
                        hovertemplate=ht,
                        showlegend=s.show_in_legend,
                    )
                    if s.use_cm:
                        lkw["marker"] = dict(
                            color=param,
                            colorscale=(
                                next(self._cyccm)
                                if self._use_cyclic_cm(param, s.is_complex)
                                else next(self._cm)
                            ),
                            size=6,
                            showscale=s.use_cm and s.colorbar,
                        )
                        if lkw["marker"]["showscale"]:
                            # only add a colorbar if required.

                            # TODO: when plotting many (14 or more) parametric
                            # expressions, each one requiring a colorbar, it might
                            # happens that the horizontal space required by all
                            # colorbars is greater than the available figure width.
                            # That raises a strange error.
                            lkw["marker"]["colorbar"] = self._create_colorbar(s.get_label(self._use_latex), True)

                    kw = merge({}, lkw, s.rendering_kw)

                    if s.is_polar:
                        kw.setdefault("thetaunit", "radians")
                        cls = self._scatter_class(go, len(x), True)
                        self._fig.add_trace(cls(r=y, theta=x, **kw))
                    else:
                        cls = self._scatter_class(go, len(x))
                        self._fig.add_trace(cls(x=x, y=y, **kw))
                else:
                    x, y = s.get_data()
                    color = next(self._cl) if s.line_color is None else s.line_color
                    lkw = dict(
                        name=s.get_label(self._use_latex),
                        mode="lines" if not s.is_point else "markers",
                        line_color=color,
                        showlegend=s.show_in_legend
                    )
                    if s.is_point:
                        lkw["marker"] = dict(size=8)
                        if not s.is_filled:
                            lkw["marker"] = dict(
                                color="#E5ECF6",
                                size=8,
                                line=dict(
                                    width=2,
                                    color=color
                                )
                            )
                    kw = merge({}, lkw, s.rendering_kw)
                    if s.is_polar:
                        kw.setdefault("thetaunit", "radians")
                        cls = self._scatter_class(go, len(x), True)
                        self._fig.add_trace(cls(r=y, theta=x, **kw))
                    else:
                        cls = self._scatter_class(go, len(x))
                        self._fig.add_trace(cls(x=x, y=y, **kw))
            elif s.is_3Dline:
                # NOTE: As a design choice, I decided to show the legend entry
                # as well as the colorbar (if use_cm=True). Even though the
                # legend entry shows the wrong color (black line), it is useful
                # in order to hide/show a specific series whenever we are
                # plotting multiple series.
                if s.is_parametric:
                    x, y, z, param = s.get_data()
                else:
                    x, y, z = s.get_data()
                    param = np.ones_like(x)

                if not s.is_point:
                    lkw = dict(
                        name=s.get_label(self._use_latex),
                        mode="lines",
                        showlegend=s.show_in_legend,
                    )
                    if s.use_cm:
                        # only add a colorbar if required.

                        # TODO: when plotting many (14 or more) parametric
                        # expressions, each one requiring a colorbar, it might
                        # happens that the horizontal space required by all
                        # colorbars is greater than the available figure width.
                        # That raises a strange error.
                        lkw["line"] = dict(
                            width = 4,
                            colorbar = self._create_colorbar(s.get_label(self._use_latex), True),
                            colorscale = (
                                next(self._cm) if s.use_cm
                                else self._solid_colorscale(s)
                            ),
                            color = param,
                            showscale = s.colorbar,
                        )
                    else:
                        lkw["line"] = dict(
                            width = 4,
                            color = (
                                (next(self._cl) if s.line_color is None
                                else s.line_color) if (s.show_in_legend or s.get_label(self._use_latex) != "__k__")
                                else self.wireframe_color)
                        )
                else:
                    color = next(self._cl) if s.line_color is None else s.line_color
                    lkw = dict(
                        name=s.get_label(self._use_latex),
                        mode="markers",
                        showlegend=s.show_in_legend)

                    lkw["marker"] = dict(
                        color=color if not s.use_cm else param,
                        size=8,
                        colorscale=next(self._cm) if s.use_cm else None,
                        showscale = s.use_cm and s.colorbar,
                    )
                    if s.use_cm:
                        lkw["marker"]["colorbar"] = self._create_colorbar(s.get_label(self._use_latex), True)

                    if not s.is_filled:
                        # TODO: how to show a colorscale if is_point=True
                        # and is_filled=False?
                        lkw["marker"] = dict(
                            color="#E5ECF6",
                            line=dict(
                                width=2,
                                color=lkw["marker"]["color"],
                                colorscale=lkw["marker"]["colorscale"],
                            )
                        )

                kw = merge({}, lkw, s.rendering_kw)
                self._fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **kw))

            elif s.is_3Dsurface and (not s.is_domain_coloring) and (not s.is_implicit):
                if not s.is_parametric:
                    xx, yy, zz = s.get_data()
                    surfacecolor = s.eval_color_func(xx, yy, zz)
                else:
                    xx, yy, zz, uu, vv = s.get_data()
                    surfacecolor = s.eval_color_func(xx, yy, zz, uu, vv)

                # create a solid color to be used when s.use_cm=False
                col = next(self._cl) if s.surface_color is None else s.surface_color
                colorscale = [[0, col], [1, col]]
                colormap = next(self._cm)
                skw = dict(
                    name=s.get_label(self._use_latex),
                    showscale=s.use_cm and s.colorbar,
                    showlegend=(not s.use_cm) and s.show_in_legend,
                    colorbar=self._create_colorbar(s.get_label(self._use_latex), mix_3Dsurfaces_3Dlines),
                    colorscale=colormap if s.use_cm else colorscale,
                    surfacecolor=surfacecolor,
                    cmin=surfacecolor.min(),
                    cmax=surfacecolor.max()
                )

                kw = merge({}, skw, s.rendering_kw)
                self._fig.add_trace(go.Surface(x=xx, y=yy, z=zz, **kw))

                count += 1

            elif s.is_3Dsurface and s.is_implicit:
                xx, yy, zz, rr = s.get_data()
                # create a solid color
                col = next(self._cl)
                colorscale = [[0, col], [1, col]]
                skw = dict(
                    isomin=0,
                    isomax=0,
                    showscale=False,
                    colorscale=colorscale
                )
                kw = merge({}, skw, s.rendering_kw)
                self._fig.add_trace(go.Isosurface(
                    x=xx.flatten(),
                    y=yy.flatten(),
                    z=zz.flatten(),
                    value=rr.flatten(), **kw
                ))
                count += 1


            elif s.is_contour and (not s.is_complex):
                if s.is_polar:
                    raise NotImplementedError()
                xx, yy, zz = s.get_data()
                xx = xx[0, :]
                yy = yy[:, 0]
                ckw = dict(
                    contours=dict(
                        coloring=None if s.is_filled else "lines",
                        showlabels=True if (not s.is_filled) and s.show_clabels else False,
                    ),
                    colorscale=next(self._cm),
                    colorbar=self._create_colorbar(s.get_label(self._use_latex), show_2D_vectors),
                    showscale=s.is_filled and s.colorbar,
                    zmin=zz.min(), zmax=zz.max()
                )
                kw = merge({}, ckw, s.rendering_kw)
                self._fig.add_trace(go.Contour(x=xx, y=yy, z=zz, **kw))
                count += 1

            elif s.is_vector:
                if s.color_func is not None:
                    warnings.warn("PlotlyBackend doesn't support custom "
                        "coloring of 2D/3D quivers or streamlines plots. "
                        "`color_func` will not be used.")
                if s.is_2Dvector:
                    xx, yy, uu, vv = s.get_data()
                    if s.normalize:
                        mag = np.sqrt(uu**2 + vv**2 )
                        uu, vv = [t / mag for t in [uu, vv]]
                    # NOTE: currently, it is not possible to create
                    # quivers/streamlines with a color scale:
                    # https://community.plotly.com/t/how-to-make-python-quiver-with-colorscale/41028
                    if s.is_streamlines:
                        skw = dict(
                            line_color=next(self._qc), arrow_scale=0.15, name=s.get_label(self._use_latex)
                        )
                        kw = merge({}, skw, s.rendering_kw)
                        stream = create_streamline(
                            xx[0, :], yy[:, 0], uu, vv, **kw)
                        self._fig.add_trace(stream.data[0])
                    else:
                        qkw = dict(line_color=next(self._qc), scale=0.075, name=s.get_label(self._use_latex))
                        kw = merge({}, qkw, s.rendering_kw)
                        quiver = create_quiver(xx, yy, uu, vv, **kw)
                        self._fig.add_trace(quiver.data[0])
                else:
                    xx, yy, zz, uu, vv, ww = s.get_data()
                    if s.is_streamlines:
                        stream_kw = s.rendering_kw.copy()
                        seeds_points = get_seeds_points(
                            xx, yy, zz, uu, vv, ww, to_numpy=True, **stream_kw)

                        skw = dict(
                            colorscale=(
                                next(self._cm)
                                if s.use_cm
                                else self._solid_colorscale(s)
                            ),
                            sizeref=0.3,
                            showscale=s.use_cm and s.colorbar,
                            colorbar=self._create_colorbar(s.get_label(self._use_latex)),
                            starts=dict(
                                x=seeds_points[:, 0],
                                y=seeds_points[:, 1],
                                z=seeds_points[:, 2],
                            ),
                        )

                        # remove rendering-unrelated keywords
                        for _k in ["starts", "max_prop", "npoints", "radius"]:
                            if _k in stream_kw.keys():
                                stream_kw.pop(_k)

                        kw = merge({}, skw, stream_kw)

                        self._fig.add_trace(
                            go.Streamtube(
                                x=xx.flatten(),
                                y=yy.flatten(),
                                z=zz.flatten(),
                                u=uu.flatten(),
                                v=vv.flatten(),
                                w=ww.flatten(),
                                **kw))
                    else:
                        mag = np.sqrt(uu**2 + vv**2 + ww**2)
                        if s.normalize:
                            # NOTE/TODO: as of Plotly 5.9.0, it is impossible
                            # to set the color of cones. Hence, by applying the
                            # normalization, all cones will have the same
                            # color.
                            uu, vv, ww = [t / mag for t in [uu, vv, ww]]
                        qkw = dict(
                            showscale=s.colorbar,
                            colorscale=next(self._cm),
                            sizemode="absolute",
                            sizeref=40,
                            colorbar=self._create_colorbar(s.get_label(self._use_latex)),
                            cmin=mag.min(),
                            cmax=mag.max(),
                        )
                        kw = merge({}, qkw, s.rendering_kw)
                        self._fig.add_trace(
                            go.Cone(
                                x=xx.flatten(),
                                y=yy.flatten(),
                                z=zz.flatten(),
                                u=uu.flatten(),
                                v=vv.flatten(),
                                w=ww.flatten(),
                                **kw))
                count += 1

            elif s.is_complex:
                if not s.is_3Dsurface:
                    x, y, mag, angle, img, colors = s.get_data()
                    xmin, xmax = x.min(), x.max()
                    ymin, ymax = y.min(), y.max()

                    if s.at_infinity:
                        mag, angle, img = [np.flip(np.flip(t, axis=0),
                            axis=1) for t in [mag, angle, img]]

                    self._fig.add_trace(
                        go.Image(
                            x0=xmin,
                            y0=ymin,
                            dx=(xmax - xmin) / s.n[0],
                            dy=(ymax - ymin) / s.n[1],
                            z=img,
                            name=s.get_label(self._use_latex),
                            customdata=np.dstack([mag, angle]),
                            hovertemplate=(
                                "x: %{x}<br />y: %{y}<br />RGB: %{z}"
                                + "<br />Abs: %{customdata[0]}<br />Arg: %{customdata[1]}"
                            ),
                        )
                    )

                    if (colors is not None) and s.colorbar:
                        # chroma/phase-colorbar
                        self._fig.add_trace(
                            go.Scatter(
                                x=[xmin, xmax],
                                y=[ymin, ymax],
                                showlegend=False,
                                mode="markers",
                                marker=dict(
                                    opacity=0,
                                    colorscale=[
                                        "rgb(%s, %s, %s)" % tuple(c) for c in colors
                                    ],
                                    color=[-np.pi, np.pi],
                                    colorbar=dict(
                                        tickvals=[
                                            -np.pi,
                                            -np.pi / 2,
                                            0,
                                            np.pi / 2,
                                            np.pi,
                                        ],
                                        ticktext=[
                                            "-&#x3C0;",
                                            "-&#x3C0; / 2",
                                            "0",
                                            "&#x3C0; / 2",
                                            "&#x3C0;",
                                        ],
                                        x=1 + 0.1 * count,
                                        title="Argument" if s.get_label(False) == str(s.expr) else s.get_label(self._use_latex),
                                        titleside="right",
                                    ),
                                    showscale=True,
                                ),
                            )
                        )

                    count += 1
                else:
                    xx, yy, mag, angle, colors, colorscale = s.get_data()
                    if s.coloring != "a":
                        warnings.warn(
                            "The visualization could be wrong becaue Plotly "
                            + "doesn't support custom coloring over surfaces. "
                            + "The surface color will show the "
                            + "argument of the complex function."
                        )
                    # create a solid color to be used when s.use_cm=False
                    col = next(self._cl)
                    if s.use_cm:
                        if colorscale is None:
                            colorscale = "gray"
                        else:
                            tmp = []
                            locations = list(range(0, len(colorscale)))
                            locations = [t / (len(colorscale) - 1) for t in locations]
                            for loc, c in zip(locations, colorscale):
                                tmp.append([loc, "rgb" + str(tuple(c))])
                            colorscale = tmp
                            # to avoid jumps in the colormap, first and last colors
                            # must be the same.
                            colorscale[-1][1] = colorscale[0][1]
                    else:
                        colorscale = [[0, col], [1, col]]
                    colormap = next(self._cyccm)
                    skw = dict(
                        name=s.get_label(self._use_latex),
                        showscale=s.colorbar,
                        colorbar=dict(
                            x=1 + 0.1 * count,
                            title="Argument",
                            titleside="right",
                            tickvals=[
                                -np.pi,
                                -np.pi / 2,
                                0,
                                np.pi / 2,
                                np.pi,
                            ],
                            ticktext=[
                                "-&#x3C0;",
                                "-&#x3C0; / 2",
                                "0",
                                "&#x3C0; / 2",
                                "&#x3C0;",
                            ]
                        ),
                        cmin=-np.pi,
                        cmax=np.pi,
                        colorscale=colorscale,
                        surfacecolor=angle,
                        customdata=angle,
                        hovertemplate="x: %{x}<br />y: %{y}<br />Abs: %{z}<br />Arg: %{customdata}",
                    )

                    kw = merge({}, skw, s.rendering_kw)
                    self._fig.add_trace(go.Surface(x=xx, y=yy, z=mag, **kw))

                    count += 1

            elif s.is_geometry:
                x, y = s.get_data()
                lkw = dict(
                    name=s.get_label(self._use_latex), mode="lines", fill="toself", line_color=next(self._cl)
                )
                kw = merge({}, lkw, s.rendering_kw)
                self._fig.add_trace(go.Scatter(x=x, y=y, **kw))

            elif s.is_generic:
                if s.type == "markers":
                    kw = merge({}, {"line_color": next(self._cl)}, s.rendering_kw)
                    self._fig.add_trace(go.Scatter(*s.args, **kw))
                elif s.type == "annotations":
                    kw = merge({}, {
                        "line_color": next(self._cl),
                        "mode": "text",
                        }, s.rendering_kw)
                    self._fig.add_trace(go.Scatter(*s.args, **kw))
                elif s.type == "fill":
                    kw = merge({}, {
                        "line_color": next(self._cl),
                        "fill": "tozeroy"
                        }, s.rendering_kw)
                    self._fig.add_trace(go.Scatter(*s.args, **kw))
                elif s.type == "rectangles":
                    self._fig.add_shape(*s.args, **s.rendering_kw)

            else:
                raise NotImplementedError(
                    "{} is not supported by {}".format(type(s), type(self).__name__)
                )

    def update_interactive(self, params):
        """Implement the logic to update the data generated by
        interactive-widget plots.

        Parameters
        ==========

        params : dict
            Map parameter-symbols to numeric values.
        """
        if self.imodule == "ipywidgets":
            with self._fig.batch_update():
                self._update_interactive_helper(params)
        else:
            self._update_interactive_helper(params)

    def _update_interactive_helper(self, params):
        np = import_module('numpy')
        plotly = import_module(
            'plotly',
            import_kwargs={'fromlist': ['graph_objects', 'figure_factory']},
            min_module_version='5.0.0')
        create_quiver = plotly.figure_factory.create_quiver
        merge = self.merge

        for i, s in enumerate(self.series):
            if s.is_interactive:
                self.series[i].params = params
                if s.is_2Dline and s.is_parametric:
                    x, y, param = self.series[i].get_data()
                    self.fig.data[i]["x"] = x
                    self.fig.data[i]["y"] = y
                    self.fig.data[i]["marker"]["color"] = param
                    self.fig.data[i]["customdata"] = param

                elif s.is_2Dline:
                    x, y = self.series[i].get_data()
                    if not s.is_polar:
                        self.fig.data[i]["x"] = x
                        self.fig.data[i]["y"] = y
                    else:
                        self.fig.data[i]["r"] = y
                        self.fig.data[i]["theta"] = x

                elif s.is_3Dline:
                    if s.is_parametric:
                        x, y, z, param = self.series[i].get_data()
                    else:
                        x, y, z = self.series[i].get_data()
                        param = np.zeros_like(x)
                    self.fig.data[i]["x"] = x
                    self.fig.data[i]["y"] = y
                    self.fig.data[i]["z"] = z
                    if s.use_cm:
                        self.fig.data[i]["line"]["color"] = param

                elif s.is_3Dsurface and (not s.is_domain_coloring) and (not s.is_implicit):
                    if not s.is_parametric:
                        x, y, z = s.get_data()
                        surfacecolor = s.eval_color_func(x, y, z)
                    else:
                        x, y, z, u, v = s.get_data()
                        surfacecolor = s.eval_color_func(x, y, z, u, v)
                    self.fig.data[i]["x"] = x
                    self.fig.data[i]["y"] = y
                    _min, _max = surfacecolor.min(), surfacecolor.max()
                    self.fig.data[i]["z"] = z
                    self.fig.data[i]["surfacecolor"] = surfacecolor
                    self.fig.data[i]["cmin"] = _min
                    self.fig.data[i]["cmax"] = _max

                elif s.is_contour and (not s.is_complex):
                    xx, yy, zz = s.get_data()
                    self.fig.data[i]["x"] = xx[0, :]
                    self.fig.data[i]["y"] = yy[:, 0]
                    self.fig.data[i]["z"] = zz
                    self.fig.data[i]["zmin"] = zz.min()
                    self.fig.data[i]["zmax"] = zz.max()

                elif s.is_vector and s.is_3D:
                    if s.is_streamlines:
                        raise NotImplementedError
                    x, y, z, u, v, w = self.series[i].get_data()
                    mag = np.sqrt(u**2 + v**2 + w**2)
                    if s.normalize:
                            # NOTE/TODO: as of Plotly 5.9.0, it is impossible
                            # to set the color of cones. Hence, by applying the
                            # normalization, all cones will have the same
                            # color.
                            u, v, w = [t / mag for t in [u, v, w]]
                    self.fig.data[i]["x"] = x.flatten()
                    self.fig.data[i]["y"] = y.flatten()
                    self.fig.data[i]["z"] = z.flatten()
                    self.fig.data[i]["u"] = u.flatten()
                    self.fig.data[i]["v"] = v.flatten()
                    self.fig.data[i]["w"] = w.flatten()
                    self.fig.data[i]["cmin"] = cmin=mag.min()
                    self.fig.data[i]["cmax"] = cmin=mag.max()

                elif s.is_vector:
                    x, y, u, v = self.series[i].get_data()
                    if s.normalize:
                        mag = np.sqrt(u**2 + v**2 )
                        u, v = [t / mag for t in [u, v]]
                    if s.is_streamlines:
                        # TODO: iplot doesn't work with 2D streamlines.
                        raise NotImplementedError
                    else:
                        qkw = dict(
                            line_color=self.quivers_colors[i], scale=0.075, name=s.get_label(self._use_latex)
                        )
                        kw = merge({}, qkw, s.rendering_kw)
                        quivers = create_quiver(x, y, u, v, **kw)
                        data = quivers.data[0]
                    self.fig.data[i]["x"] = data["x"]
                    self.fig.data[i]["y"] = data["y"]

                elif s.is_complex:
                    if not s.is_3Dsurface:
                        # TODO: for some unkown reason, domain_coloring and
                        # interactive plot don't like each other...
                        raise NotImplementedError
                    else:
                        xx, yy, mag, angle, colors, colorscale = s.get_data()
                        self.fig.data[i]["z"] = mag
                        self.fig.data[i]["surfacecolor"] = angle
                        self.fig.data[i]["customdata"] = angle
                        m, M = min(angle.flatten()), max(angle.flatten())
                        # show pi symbols on the colorbar if the range is
                        # close enough to [-pi, pi]
                        if (abs(m + np.pi) < 1e-02) and (abs(M - np.pi) < 1e-02):
                            self.fig.data[i]["colorbar"]["tickvals"] = [
                                m,
                                -np.pi / 2,
                                0,
                                np.pi / 2,
                                M,
                            ]
                            self.fig.data[i]["colorbar"]["ticktext"] = [
                                "-&#x3C0;",
                                "-&#x3C0; / 2",
                                "0",
                                "&#x3C0; / 2",
                                "&#x3C0;",
                            ]

                elif s.is_geometry and not (s.is_2Dline):
                    x, y = self.series[i].get_data()
                    self.fig.data[i]["x"] = x
                    self.fig.data[i]["y"] = y

    def _update_layout(self):
        self._fig.update_layout(
            template=self._theme,
            width=None if not self.size else self.size[0],
            height=None if not self.size else self.size[1],
            title=r"<b>%s</b>" % ("" if not self.title else self.title),
            title_x=0.5,
            xaxis=dict(
                title="" if not self.xlabel else self.xlabel,
                range=None if not self.xlim else self.xlim,
                type=self.xscale,
                showgrid=self.grid,  # thin lines in the background
                zeroline=self.grid,  # thick line at x=0
                constrain="domain",
                visible=self.axis
            ),
            yaxis=dict(
                title="" if not self.ylabel else self.ylabel,
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
                    title="" if not self.xlabel else self.xlabel,
                    range=None if not self.xlim else self.xlim,
                    type=self.xscale,
                    showgrid=self.grid,  # thin lines in the background
                    zeroline=self.grid,  # thick line at x=0
                    visible=self.grid,  # numbers below
                ),
                yaxis=dict(
                    title="" if not self.ylabel else self.ylabel,
                    range=None if not self.ylim else self.ylim,
                    type=self.yscale,
                    showgrid=self.grid,  # thin lines in the background
                    zeroline=self.grid,  # thick line at x=0
                    visible=self.grid,  # numbers below
                ),
                zaxis=dict(
                    title="" if not self.zlabel else self.zlabel,
                    range=None if not self.zlim else self.zlim,
                    type=self.zscale,
                    showgrid=self.grid,  # thin lines in the background
                    zeroline=self.grid,  # thick line at x=0
                    visible=self.grid,  # numbers below
                ),
                aspectmode=("manual" if isinstance(self.aspect, dict) else (self.aspect if self.aspect != "equal" else "auto")),
                aspectratio=self.aspect if isinstance(self.aspect, dict) else None,
                camera=self.camera
            ),
        )

    def show(self):
        """Visualize the plot on the screen."""
        if len(self._fig.data) != len(self.series):
            self.process_series()
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
        if (len(self.series) > 0) and (len(self.fig.data) == 0):
            self.process_series()

        ext = os.path.splitext(path)[1]
        if ext.lower() in [".htm", ".html"]:
            self.fig.write_html(path, **kwargs)
        else:
            self._fig.write_image(path, **kwargs)


PB = PlotlyBackend
