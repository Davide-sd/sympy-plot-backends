from spb.defaults import cfg
from spb.backends.base_backend import Plot
from spb.backends.utils import get_seeds_points
from sympy.external import import_module
import itertools
import warnings
import os


class PlotlyBackend(Plot):
    """
    A backend for plotting SymPy's symbolic expressions using Plotly.

    Parameters
    ==========

    aspect : str, optional
        Set the aspect ratio of the plot. Default to `"auto"`.
        Possible values:

        - `"equal"`: sets equal spacing on the axis of a 2D plot.
        - `"cube"`, `"auto"` for 3D plots.

    contour_kw : dict, optional
        A dictionary of keywords/values which is passed to Plotly's contour
        function to customize the appearance.
        Refer to [#fn1]_ and [#fn2]_  to learn more about customization.

    line_kw : dict, optional
        A dictionary of keywords/values which is passed to Plotly's scatter
        functions to customize the appearance.
        Refer to [#fn3]_ and [#fn4]_ to learn more about customization.

    quiver_kw : dict, optional
        A dictionary of keywords/values which is passed to Plotly's quivers
        function to customize the appearance.

        - For 2D vector fields, default to: `dict( scale = 0.075 )`
          Refer to [#fn5]_ for more options.
        - For 3D vector fields, default to: `dict( sizemode = "absolute", sizeref = 40 )`
          Refer [#fn6]_ for more options.

    surface_kw : dict, optional
        A dictionary of keywords/values which is passed to Plotly's
        Surface function to customize the appearance.
        Refer to [#fn7]_ for more options.

    stream_kw : dict, optional
        A dictionary of keywords/values which is passed to Plotly's
        streamlines function to customize the appearance.

        - For 2D vector fields, defaul to: `dict( arrow_scale = 0.15 )`
          Refer to [#fn8]_ for more options.
        - For 3D vector fields, default to: `dict( sizeref = 0.3 )`
          Refer to [#fn9]_ for more options.

    theme : str, optional
        Set the theme. Default to `"plotly_dark"`. Find more Plotly themes at
        [#fn10]_ .

    use_cm : boolean, optional
        If True, apply a color map to the meshes/surface. If False, solid
        colors will be used instead. Default to True.

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


    See also
    ========

    Plot, MatplotlibBackend, BokehBackend, K3DBackend
    """

    _library = "plotly"

    # The following colors corresponds to the discret color map
    # px.colors.qualitative.Plotly.
    colorloop = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
    # a selection of color maps to be used in 3D surfaces, contour plots.
    colormaps = [
        "aggrnyl", "plotly3", "reds_r", "ice", "inferno",
        "deep_r", "turbid_r", "gnbu_r", "geyser_r", "oranges_r"]
    # to be used in complex-parametric plots
    cyclic_colormaps = ["phase", "twilight", "hsv", "icefire"]

    # TODO: here I selected black and white, but they are not visible with dark
    # or light theme respectively... Need a better selection of colors.
    # Although, they are placed in the middle of the loop, so they are unlikely
    # going to be used.
    quivers_colors = [
        "magenta", "crimson", "darkorange", "dodgerblue", "wheat",
        "slategrey", "white", "black", "darkred", "indigo"]

    # color bar spacing
    _cbs = 0.15
    # color bar scale down factor
    _cbsdf = 0.75

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        plotly = import_module(
            'plotly',
            import_kwargs={'fromlist': ['graph_objects', 'figure_factory']},
            min_module_version='5.0.0')
        go = plotly.graph_objects

        self._init_cyclers()
        super().__init__(*args, **kwargs)
        self._theme = kwargs.get("theme", cfg["plotly"]["theme"])

        # add colors if needed
        if ((len([s for s in self._series if s.is_2Dline]) > 10) and
            (len(self.colorloop) <= 10)):
            # this corresponds to px.colors.qualitative.Light24
            self.colorloop = [
                "#FD3216", "#00FE35", "#6A76FC", "#FED4C4", "#FE00CE",
                "#0DF9FF", "#F6F926", "#FF9616", "#479B55", "#EEA6FB",
                "#DC587D", "#D626FF", "#6E899C", "#00B5F7", "#B68E00",
                "#C9FBE5", "#FF0092", "#22FFA7", "#E3EE9E", "#86CE00",
                "#BC7196", "#7E7DCD", "#FC6955", "#E48F72"
            ]

        self._fig = go.Figure()
        # this is necessary in order for the series to be added even if
        # show=False
        self._process_series(self._series)
        self._update_layout()
    
    def _set_piecewise_color(self, s, color):
        """Set the color to the given series"""
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
        self._qc = itertools.cycle(self.quivers_colors)

    def _create_colorbar(self, k, label, sc=False):
        """This method reduces code repetition.

        Parameters
        ==========
            k : int
                index of the current color bar
            label : str
                Name to display besides the color bar
            sc : boolean
                Scale Down the color bar to make room for the legend.
                Default to False
        """
        return dict(
            x=1 + self._cbs * k,
            title=label,
            titleside="right",
            # scale down the color bar to make room for legend
            len=self._cbsdf if (sc and self.legend) else 1,
            yanchor="bottom",
            y=0,
        )

    def _solid_colorscale(self):
        # create a solid color to be used when self._use_cm=False
        col = next(self._cl)
        return [[0, col], [1, col]]

    def _process_series(self, series):
        np = import_module('numpy')
        plotly = import_module(
            'plotly',
            import_kwargs={'fromlist': ['graph_objects', 'figure_factory']},
            min_module_version='5.0.0')
        go = plotly.graph_objects
        create_quiver = plotly.figure_factory.create_quiver
        create_streamline = plotly.figure_factory.create_streamline
        mergedeep = import_module('mergedeep')
        merge = mergedeep.merge
        self._init_cyclers()

        # if legend=True and both 3d lines and surfaces are shown, then hide the
        # surfaces color bars and only shows line labels in the legend.
        # TODO: can I show both colorbars and legends by scaling down the color
        # bars?
        show_3D_colorscales = True
        show_2D_vectors = False
        for s in series:
            if s.is_3Dline:
                show_3D_colorscales = False
            if s.is_2Dvector:
                show_2D_vectors = True

        self._fig.data = []

        count = 0
        for ii, s in enumerate(series):
            kw = None

            if s.is_2Dline:
                if s.is_parametric:
                    x, y, param = s.get_data()
                    # hides/show the colormap depending on self._use_cm
                    mode = "lines+markers" if not s.is_point else "markers"
                    if (not s.is_point) and (not self._use_cm):
                        mode = "lines"
                    lkw = dict(
                        name=s.label,
                        line_color=next(self._cl),
                        mode=mode,
                        marker=dict(
                            color=param,
                            colorscale=(
                                next(self._cyccm)
                                if self._use_cyclic_cm(param, s.is_complex)
                                else next(self._cm)
                            ),
                            size=6,
                            showscale=self.legend and self._use_cm,
                            colorbar=self._create_colorbar(ii, s.label, True),
                        ),
                        customdata=param,
                        hovertemplate=(
                            "x: %{x}<br />y: %{y}<br />u: %{customdata}"
                            if not s.is_complex
                            else "x: %{x}<br />y: %{y}<br />Arg: %{customdata}"
                        ),
                    )
                    kw = merge({}, lkw, s.rendering_kw)
                    self._fig.add_trace(go.Scatter(x=x, y=y, **kw))
                else:
                    x, y = s.get_data()
                    color = next(self._cl)
                    lkw = dict(
                        name=s.label,
                        mode="lines" if not s.is_point else "markers",
                        line_color=color
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
                    self._fig.add_trace(go.Scatter(x=x, y=y, **kw))
            elif s.is_3Dline:
                # NOTE: As a design choice, I decided to show the legend entry
                # as well as the colorbar (if use_cm=True). Even though the
                # legend entry shows the wrong color (black line), it is useful
                # in order to hide/show a specific series whenever we are
                # plotting multiple series.
                x, y, z, param = s.get_data()
                if not s.is_point:
                    lkw = dict(
                        name=s.label,
                        mode="lines",
                        line=dict(
                            width=4,
                            colorscale=(
                                next(self._cm)
                                if self._use_cm
                                else self._solid_colorscale()
                            ),
                            color=param,
                            showscale=self.legend and self._use_cm,
                            colorbar=self._create_colorbar(ii, s.label, True),
                        ),
                    )
                else:
                    lkw = dict(
                        name=s.label,
                        mode="markers",
                        line_color=next(self._cl))
                kw = merge({}, lkw, s.rendering_kw)
                self._fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **kw))

            elif (s.is_3Dsurface and not s.is_domain_coloring):
                xx, yy, zz = s.get_data()

                # create a solid color to be used when self._use_cm=False
                col = next(self._cl)
                colorscale = [[0, col], [1, col]]
                colormap = next(self._cm)
                skw = dict(
                    name=s.label,
                    showscale=self.legend and show_3D_colorscales,
                    colorbar=self._create_colorbar(ii, s.label),
                    colorscale=colormap if self._use_cm else colorscale,
                )

                kw = merge({}, skw, s.rendering_kw)
                self._fig.add_trace(go.Surface(x=xx, y=yy, z=zz, **kw))

                count += 1

            elif s.is_contour and (not s.is_complex):
                xx, yy, zz = s.get_data()
                xx = xx[0, :]
                yy = yy[:, 0]
                ckw = dict(
                    contours=dict(
                        coloring=None,
                        showlabels=False,
                    ),
                    colorscale=next(self._cm),
                    colorbar=self._create_colorbar(ii, s.label, show_2D_vectors),
                )
                kw = merge({}, ckw, s.rendering_kw)
                self._fig.add_trace(go.Contour(x=xx, y=yy, z=zz, **kw))
                count += 1

            elif s.is_vector:
                if s.is_2Dvector:
                    xx, yy, uu, vv = s.get_data()
                    # NOTE: currently, it is not possible to create
                    # quivers/streamlines with a color scale:
                    # https://community.plotly.com/t/how-to-make-python-quiver-with-colorscale/41028
                    if s.is_streamlines:
                        skw = dict(
                            line_color=next(self._qc), arrow_scale=0.15, name=s.label
                        )
                        kw = merge({}, skw, s.rendering_kw)
                        stream = create_streamline(
                            xx[0, :], yy[:, 0], uu, vv, **kw)
                        self._fig.add_trace(stream.data[0])
                    else:
                        qkw = dict(line_color=next(self._qc), scale=0.075, name=s.label)
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
                                if self._use_cm
                                else self._solid_colorscale()
                            ),
                            sizeref=0.3,
                            showscale=self.legend and self._use_cm,
                            colorbar=self._create_colorbar(ii, s.label),
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
                        qkw = dict(
                            showscale=(not s.is_slice) or self.legend,
                            colorscale=next(self._cm),
                            sizemode="absolute",
                            sizeref=40,
                            colorbar=self._create_colorbar(ii, s.label),
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

                    self._fig.add_trace(
                        go.Image(
                            x0=xmin,
                            y0=ymin,
                            dx=(xmax - xmin) / s.n1,
                            dy=(ymax - ymin) / s.n2,
                            z=img,
                            name=s.label,
                            customdata=np.dstack([mag, angle]),
                            hovertemplate=(
                                "x: %{x}<br />y: %{y}<br />RGB: %{z}"
                                + "<br />Abs: %{customdata[0]}<br />Arg: %{customdata[1]}"
                            ),
                        )
                    )

                    if colors is not None:
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
                                        title="Argument",
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
                            "Plotly doesn't support custom coloring "
                            + "over surfaces. The surface color will show the "
                            + "argument of the complex function."
                        )
                    # create a solid color to be used when self._use_cm=False
                    col = next(self._cl)
                    if self._use_cm:
                        tmp = []
                        locations = list(range(0, len(colorscale)))
                        locations = [t / (len(colorscale) - 1) for t in locations]
                        for loc, c in zip(locations, colorscale):
                            tmp.append([loc, "rgb" + str(tuple(c))])
                        colorscale = tmp
                        # print(colorscale)
                    else:
                        colorscale = [[0, col], [1, col]]
                    colormap = next(self._cyccm)
                    skw = dict(
                        name=s.label,
                        showscale=True,
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
                    name=s.label, mode="lines", fill="toself", line_color=next(self._cl)
                )
                kw = merge({}, lkw, s.rendering_kw)
                self._fig.add_trace(go.Scatter(x=x, y=y, **kw))

            else:
                raise NotImplementedError(
                    "{} is not supported by {}".format(type(s), type(self).__name__)
                )

    def _update_interactive(self, params):
        np = import_module('numpy')
        plotly = import_module(
            'plotly',
            import_kwargs={'fromlist': ['graph_objects', 'figure_factory']},
            min_module_version='5.0.0')
        create_quiver = plotly.figure_factory.create_quiver
        mergedeep = import_module('mergedeep')
        merge = mergedeep.merge

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
                    if s.is_geometry:
                        self.fig.data[i]["x"] = x
                    self.fig.data[i]["y"] = y

                elif s.is_3Dline:
                    x, y, z, param = s.get_data()
                    self.fig.data[i]["x"] = x
                    self.fig.data[i]["y"] = y
                    self.fig.data[i]["z"] = z
                    self.fig.data[i]["line"]["color"] = param

                elif s.is_3Dsurface and s.is_parametric:
                    x, y, z = self.series[i].get_data()
                    self.fig.data[i]["x"] = x
                    self.fig.data[i]["y"] = y
                    self.fig.data[i]["z"] = z

                elif s.is_3Dsurface and (not s.is_domain_coloring):
                    x, y, z = self.series[i].get_data()
                    self.fig.data[i]["z"] = z

                elif s.is_contour and (not s.is_complex):
                    _, _, zz = s.get_data()
                    self.fig.data[i]["z"] = zz

                elif s.is_vector and s.is_3D:
                    if s.is_streamlines:
                        raise NotImplementedError
                    _, _, _, u, v, w = self.series[i].get_data()
                    self.fig.data[i]["u"] = u.flatten()
                    self.fig.data[i]["v"] = v.flatten()
                    self.fig.data[i]["w"] = w.flatten()

                elif s.is_vector:
                    x, y, u, v = self.series[i].get_data()
                    if s.is_streamlines:
                        # TODO: iplot doesn't work with 2D streamlines.
                        raise NotImplementedError
                    else:
                        qkw = dict(
                            line_color=self.quivers_colors[i], scale=0.075, name=s.label
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
            ),
            yaxis=dict(
                title="" if not self.ylabel else self.ylabel,
                range=None if not self.ylim else self.ylim,
                type=self.yscale,
                showgrid=self.grid,  # thin lines in the background
                zeroline=self.grid,  # thick line at x=0
                scaleanchor="x" if self.aspect == "equal" else None,
            ),
            margin=dict(
                t=50,
                l=0,
                b=0,
            ),
            showlegend=self.legend,
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
                aspectmode=(self.aspect if self.aspect != "equal" else "auto"),
            ),
        )

    def show(self):
        """Visualize the plot on the screen."""
        if len(self._fig.data) != len(self.series):
            self._process_series(self.series)
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
            self._process_series(self._series)
            self._update_layout()

        ext = os.path.splitext(path)[1]
        if ext.lower() in [".html", ".html"]:
            self.fig.write_html(path, **kwargs)
        else:
            self._fig.write_image(path, **kwargs)


PB = PlotlyBackend
