from spb.defaults import plotly_theme
from spb.backends.base_backend import Plot
from spb.utils import get_seeds_points
import plotly.graph_objects as go
from plotly.figure_factory import create_quiver, create_streamline
from mergedeep import merge
import itertools

"""
TODO:
1. iplot support for 2D/3D streamlines.
2. colorbar title on the side, vertically aligned
"""

class PlotlyBackend(Plot):
    """ A backend for plotting SymPy's symbolic expressions using Plotly.

    Keyword Arguments
    =================

        aspect : str
            Default to "auto". Possible values:
            "equal": sets equal spacing on the axis of a 2D plot.
            "cube", "auto" for 3D plots.
        
        contour_kw : dict
            A dictionary of keywords/values which is passed to Plotly's contour
            function to customize the appearance.
            Refer to the following web pages to learn more about customization:
            https://plotly.com/python/contour-plots/
            https://plotly.com/python/builtin-colorscales/
        
        line_kw : dict
            A dictionary of keywords/values which is passed to Plotly's scatter
            functions to customize the appearance.
            Refer to the following web pages to learn more about customization:
            https://plotly.com/python/line-and-scatter/
            https://plotly.com/python/3d-scatter-plots/
        
        quiver_kw : dict
            A dictionary of keywords/values which is passed to Plotly's quivers
            function to customize the appearance.
            For 2D vector fields, default to:
                ``dict( scale = 0.075 )``
                Refer to this documentation page:
                https://plotly.com/python/quiver-plots/
            For 3D vector fields, default to:
                ``dict( sizemode = "absolute", sizeref = 40 )``
                Refer to this documentation page:
                https://plotly.com/python/cone-plot/

        surface_kw : dict
            A dictionary of keywords/values which is passed to Plotly's
            Surface function to customize the appearance.
            Refer to this documentation page:
            https://plotly.com/python/3d-surface-plots/

        stream_kw : dict
            A dictionary of keywords/values which is passed to Plotly's
            streamlines function to customize the appearance.
            For 2D vector fields, defaul to: 
                ``dict( arrow_scale = 0.15 )``
                Refer to this documentation page:
                https://plotly.com/python/streamline-plots/
            For 3D vector fields, default to:
                ```dict(
                        sizeref = 0.3,
                        starts = dict(
                            x = seeds_points[:, 0],
                            y = seeds_points[:, 1],
                            z = seeds_points[:, 2]
                    ))
                ```
            where `seeds_points` are the starting points of the streamlines.
            Refer to this documentation page:
            https://plotly.com/python/streamtube-plot/

        theme : str
            Set the theme. Default to "plotly_dark". Find more Plotly themes at
            the following page:
            https://plotly.com/python/templates/
        
        use_cm : boolean
            If True, apply a color map to the meshes/surface. If False, solid
            colors will be used instead. Default to True.
        
        wireframe : boolean
            Visualize the wireframe lines on surfaces. Default to False.
            Note that it may have a negative impact on the performances.

    Export
    ======

    In order to export the plots you will need to install the packages listed
    in the following page:
    https://plotly.com/python/static-image-export/
    """
    # The following colors corresponds to the discret color map 
    # px.colors.qualitative.Plotly. Thei are here in order to avoid the 
    # following statement: import plotly.express as px
    # which happens to slow down the loading phase.
    colorloop = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    # a selection of color maps to be used in 3D surfaces, contour plots.
    # NOTE: if you change any of these, make sure to also change 
    # wireframe_colors.
    colormaps = [
        'aggrnyl', 'plotly3', 'reds_r', 'ice', 'inferno', 
        'deep_r', 'turbid_r', 'gnbu_r', 'geyser_r', 'oranges_r' 
    ]
    # a selection of solid colors for wireframe lines that may look good with
    # the above colormaps
    wireframe_colors = [
        "#0071c3", "#af67d9", "#e64b17", "#1378cd", "#be5466",
        "#6f969e", "#aa692c", "#60ccc0", "#f2a45d", "#f2a45d"
    ]
    # a few solid color that offers medium to good contrast against Plotly's
    # default colorscale for contour plots
    # TODO: here I selected black and white, but they are not visible with dark
    # or light theme respectively... Need a better selection of colors. 
    # Although, they are placed in the middle of the loop, so they are unlikely
    # going to be used.
    quivers_colors = [
        "magenta", "crimson", "darkorange", "dodgerblue", "wheat", 
        "slategrey", "white", "black", "darkred", "indigo"
        # "cyan", "greenyellow", "grey", "darkred", "white",
        # "black", "orange", "silver", "darkcyan", "magenta"
    ]

    def __new__(cls, *args, **kwargs):
        # Since Plot has its __new__ method, this will prevent infinite
        # recursion
        return object.__new__(cls)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fig = go.Figure()
        # this is necessary in order for the series to be added even if
        # show=False
        self._process_series(self._series)
        self._update_layout()
    
    def _init_cyclers(self):
        self._cl = itertools.cycle(self.colorloop)
        self._cm = itertools.cycle(self.colormaps)
        self._wfcm = itertools.cycle(self.wireframe_colors)
        self._qc = itertools.cycle(self.quivers_colors)
    
    def _process_series(self, series):
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
        
        for ii, s in enumerate(series):
            if s.is_2Dline:
                x, y = s.get_data()
                line_kw = self._kwargs.get("line_kw", dict())
                if s.is_parametric:
                    u = s.discretized_var
                    # TODO: show the colorbar with the u parameter
                    lkw = dict(
                        line_color = next(self._cl),
                        mode = "lines+markers",
                        marker = dict(
                            color = u,
                            colorscale = next(self._cm),
                            size = 6
                        ))
                    self._fig.add_trace(
                        go.Scatter(
                            x = x, y = y, name = s.label,
                            **merge({}, lkw, line_kw)
                        )
                    )
                else:
                    lkw = dict(
                        mode = "lines",
                        line_color = next(self._cl)
                    )
                    self._fig.add_trace(
                        go.Scatter(x = x, y = y, name = s.label, 
                            **merge({}, lkw, line_kw))
                    )
            elif s.is_3Dline:
                x, y, z = s.get_data()
                u = s.discretized_var
                lkw = dict(
                    mode = "lines",
                    line = dict(
                        width = 4,
                        colorscale = next(self._cm),
                        color = u,
                        showscale = True,
                        colorbar = dict(
                            x = 1 + 0.1 * ii,
                            title = s.label,
                            titleside = 'right',
                        ),
                    )
                )
                line_kw = self._kwargs.get("line_kw", dict())
                self._fig.add_trace(
                    go.Scatter3d(
                        x = x, y = y, z = z,
                        name = s.label,
                        **merge({}, lkw, line_kw)
                    )
                )
            elif s.is_3Dsurface:
                xx, yy, zz = s.get_data()
                # create a solid color to be used when self._use_cm=False
                col = next(self._cl)
                colorscale = [
                    [0, col],
                    [1, col]
                ]
                skw = dict(
                    showscale = self.legend and show_3D_colorscales,
                    colorbar = dict(
                        x = 1 + 0.1 * ii,
                        title = s.label,
                        titleside = 'right',
                    ),
                    colorscale = next(self._cm) if self._use_cm else colorscale
                )
                surface_kw = self._kwargs.get("surface_kw", dict())
                self._fig.add_trace(
                    go.Surface(
                        x = xx, y = yy, z = zz, name = s.label,
                        **merge({}, skw, surface_kw)
                    )
                )
                
                if self._kwargs.get("wireframe", False):
                    line_marker = dict(
                        color = next(self._wfcm),
                        width = 2
                    )
                    for i, j, k in zip(xx, yy, zz):
                        self._fig.add_trace(
                            go.Scatter3d(
                                x = i, y = j, z = k,
                                mode = 'lines',
                                line = line_marker,
                                showlegend = False
                            )
                        )
                    for i, j, k in zip(xx.T, yy.T, zz.T):
                        self._fig.add_trace(
                            go.Scatter3d(
                                x = i, y = j, z = k,
                                mode = 'lines',
                                line = line_marker,
                                showlegend = False
                            )
                        )
            elif s.is_contour:
                xx, yy, zz = s.get_data()
                xx = xx[0, :]
                yy = yy[:, 0]
                # default values
                ckw = dict(
                    contours = dict(
                        coloring = None,
                        showlabels = False,
                    ),
                    colorscale = next(self._cm),
                    colorbar = dict(
                        title = s.label,
                        titleside = 'right',
                        # scale down the color bar to make room for legend
                        len = 0.75 if (show_2D_vectors and self.legend) else 1,
                        yanchor = "bottom", y=0,
                        x = 1 + 0.1 * ii,
                    )
                )
                # user-provided values
                contour_kw = self._kwargs.get("contour_kw", dict())
                self._fig.add_trace(go.Contour(x = xx, y = yy, z = zz,
                        **merge({}, ckw, contour_kw)))
            elif s.is_implicit:
                points = s.get_data()
                if len(points) == 2:
                    # interval math plotting
                    x, y, pixels = self._get_pixels(s, points[0])
                    ckw = dict(
                        colorscale = [
                            [0, "rgba(0,0,0,0)"],
                            [1, next(self._cl)],
                        ],
                        showscale = False,
                        name = s.label
                    )
                    contour_kw = self._kwargs.get("contour_kw", dict())
                    self._fig.add_trace(go.Heatmap(x = x, y = y, z = pixels,
                        **merge({}, ckw, contour_kw)))
                else:
                    x, y, z, plot_type = points
                    zf = z.flatten()
                    m, M = min(zf), max(zf)
                    col = next(self._cl)
                    # default values
                    ckw = dict(
                        contours = dict(
                            coloring = "none" if plot_type == "contour" else None,
                            showlabels = False,
                            type = "levels" if plot_type == "contour" else "constraint",
                            operation = '<', value = [(m + M) / 2]
                        ),
                        colorscale = [
                            [0, "rgba(0,0,0,0)"],
                            [1, next(self._cl)],
                        ],
                        fillcolor = col,
                        showscale = True,
                        name = s.label,
                        line = dict( color = col )
                    )
                    contour_kw = self._kwargs.get("contour_kw", dict())
                    self._fig.add_trace(go.Contour(x = x, y = y, z = z,
                            **merge({}, ckw, contour_kw)))
            elif s.is_vector:
                if s.is_2Dvector:
                    xx, yy, uu, vv = s.get_data()
                    streamlines = self._kwargs.get("streamlines", False)
                    if streamlines:
                        # NOTE: currently, it is not possible to create streamlines with
                        # a color scale: https://community.plotly.com/t/how-to-make-python-quiver-with-colorscale/41028
                        # default values
                        skw = dict( line_color = next(self._qc), arrow_scale = 0.15,
                            name = s.label)
                        # user-provided values
                        stream_kw = self._kwargs.get("stream_kw", dict())
                        stream = create_streamline(xx[0, :], yy[:, 0], uu, vv,
                            **merge({}, skw, stream_kw))
                        self._fig.add_trace(stream.data[0])
                    else:
                        # NOTE: currently, it is not possible to create quivers with
                        # a color scale: https://community.plotly.com/t/how-to-make-python-quiver-with-colorscale/41028
                        # default values
                        qkw = dict( line_color = next(self._qc), scale = 0.075,
                            name = s.label )
                        # user-provided values
                        quiver_kw = self._kwargs.get("quiver_kw", dict())
                        quiver = create_quiver(xx, yy, uu, vv, 
                            **merge({}, qkw, quiver_kw)) # merge two dictionaries
                        self._fig.add_trace(quiver.data[0])
                else:
                    xx, yy, zz, uu, vv, ww = s.get_data()
                    streamlines = self._kwargs.get("streamlines", False)
                    if streamlines:
                        seeds_points = get_seeds_points(xx, yy, zz, uu, vv, ww)

                        # default values
                        skw = dict( colorscale = next(self._cm), sizeref = 0.3,
                                colorbar = dict(
                                    title = s.label,
                                    titleside = 'right',
                                    x = 1 + 0.1 * ii,
                                ),
                                starts = dict(
                                    x = seeds_points[:, 0],
                                    y = seeds_points[:, 1],
                                    z = seeds_points[:, 2],
                                ))
                        # user-provided values
                        stream_kw = self._kwargs.get("stream_kw", dict())
                        self._fig.add_trace(
                            go.Streamtube( x = xx.flatten(), y = yy.flatten(),
                                z = zz.flatten(), u = uu.flatten(),
                                v = vv.flatten(), w = ww.flatten(),
                                **merge({}, skw, stream_kw))
                        )
                    else:
                        # default values
                        qkw = dict(
                            colorscale = next(self._cm),
                            sizemode = "absolute", 
                            sizeref = 40,
                            colorbar = dict(
                                x = 1 + 0.1 * ii,
                                title = s.label,
                                titleside = 'right',
                            ))
                        # user-provided values
                        quiver_kw = self._kwargs.get("quiver_kw", dict())
                        self._fig.add_trace(
                            go.Cone( x = xx.flatten(), y = yy.flatten(),
                                z = zz.flatten(), u = uu.flatten(),
                                v = vv.flatten(), w = ww.flatten(),
                                **merge({}, qkw, quiver_kw))
                        )
            else:
                raise NotImplementedError(
                    "{} is not supported by {}".format(type(s), type(self).__name__)
                )
    
    def _update_interactive(self, params):
        for i, s in enumerate(self.series):
            if s.is_interactive:
                self.series[i].update_data(params)
                if s.is_2Dline and s.is_parametric:
                    x, y = self.series[i].get_data()
                    self.fig.data[i]["x"] = x
                    self.fig.data[i]["y"] = y
                elif s.is_2Dline and (not s.is_parametric):
                    x, y = self.series[i].get_data()
                    self.fig.data[i]["y"] = y
                elif s.is_3Dline or (s.is_3Dsurface and s.is_parametric):
                    x, y, z = self.series[i].get_data()
                    self.fig.data[i]["x"] = x
                    self.fig.data[i]["y"] = y
                    self.fig.data[i]["z"] = z
                elif s.is_3Dsurface and (not s.is_parametric):
                    x, y, z = self.series[i].get_data()
                    self.fig.data[i]["z"] = z
                elif s.is_vector and s.is_3D:
                    streamlines = self._kwargs.get("streamlines", False)
                    if streamlines:
                        raise NotImplementedError
                    _, _, _, u, v, w = self.series[i].get_data()
                    self.fig.data[i]["u"] = u.flatten()
                    self.fig.data[i]["v"] = v.flatten()
                    self.fig.data[i]["w"] = w.flatten()
                elif s.is_vector:
                    x, y, u, v = self.series[i].get_data()
                    streamlines = self._kwargs.get("streamlines", False)
                    if streamlines:
                        streams = create_streamline(x[0, :], y[:, 0], u, v)
                        data = streams.data[0]
                        # TODO: iplot doesn't work with 2D streamlines. Why?
                        # Is it possible that the sequential update of x and y
                        # is the cause of the error? Since at every update,
                        # len(x) = len(y) but those are different from before.
                        raise NotImplementedError
                    else:
                        # default values
                        qkw = dict( line_color = self.quivers_colors[i],
                            scale = 0.075, name = s.label )
                        # user-provided values
                        quiver_kw = self._kwargs.get("quiver_kw", dict())
                        quivers = create_quiver(x, y, u, v,
                            **merge({}, qkw, quiver_kw))
                        data = quivers.data[0]
                    self.fig.data[i]["x"] = data["x"]
                    self.fig.data[i]["y"] = data["y"]


    def _update_layout(self):
        self._fig.update_layout(
            template = self._kwargs.get("theme", plotly_theme),
            width = None if not self.size else self.size[0],
            height = None if not self.size else self.size[1],
            title = r"<b>%s</b>" % ("" if not self.title else self.title),
            title_x = 0.5,
            xaxis = dict(
                title = "" if not self.xlabel else self.xlabel,
                range = None if not self.xlim else self.xlim,
                type = self.xscale,
                showgrid = self.axis, # thin lines in the background
                zeroline = self.axis, # thick line at x=0
                visible = self.axis,  # numbers below
                constrain = 'domain'
            ),
            yaxis = dict(
                title = "" if not self.ylabel else self.ylabel,
                range = None if not self.ylim else self.ylim,
                type = self.yscale,
                showgrid = self.axis, # thin lines in the background
                zeroline = self.axis, # thick line at x=0
                visible = self.axis,  # numbers below,
                scaleanchor = "x" if self.aspect == "equal" else None
            ),
            margin = dict(
                t = 50,
                l = 0,
                b = 0,
            ),
            showlegend = self.legend,
            scene = dict(
                xaxis = dict(
                    title = "" if not self.xlabel else self.xlabel,
                    range = None if not self.xlim else self.xlim,
                    type = self.xscale,
                    showgrid = self.axis, # thin lines in the background
                    zeroline = self.axis, # thick line at x=0
                    visible = self.axis,  # numbers below
                ),
                yaxis = dict(
                    title = "" if not self.ylabel else self.ylabel,
                    range = None if not self.ylim else self.ylim,
                    type = self.yscale,
                    showgrid = self.axis, # thin lines in the background
                    zeroline = self.axis, # thick line at x=0
                    visible = self.axis,  # numbers below
                ),
                zaxis = dict(
                    title = "" if not self.zlabel else self.zlabel,
                    range = None if not self.zlim else self.zlim,
                    type = self.zscale,
                    showgrid = self.axis, # thin lines in the background
                    zeroline = self.axis, # thick line at x=0
                    visible = self.axis,  # numbers below
                ),
                aspectmode = (self.aspect if self.aspect != "equal"
                                else "auto")
            )
        )
    
    def show(self):
        self._fig.show()

    def save(self, path, **kwargs):
        self._process_series(self._series)
        self._update_layout()
        self._fig.write_image(path)

PB = PlotlyBackend