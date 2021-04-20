from spb.base_backend import MyBaseBackend
import plotly.graph_objects as go

class PlotlyBackend(MyBaseBackend):
    """ A backend for plotting SymPy's symbolic expressions using Plotly.

    Keyword Arguments
    =================

        theme : str
            Set the theme. Default to "plotly_dark". Find more Plotly themes at
            the following page:
            https://plotly.com/python/templates/
        
        wireframe : boolean
            Visualize the wireframe lines on surfaces. Default to False.
            Note that it may have a negative impact on the performances.

    Export
    ======

    In order to export the plots you will need to install the packages listed
    in the following page:
    https://plotly.com/python/static-image-export/
    """
    colormaps = [
        'aggrnyl', 'plotly3', 'reds_r', 'ice', 'inferno', 
        'deep_r', 'turbid_r', 'gnbu_r', 'geyser_r', 'oranges_r' 
    ]
    wireframe_colors = [
        "#0071c3", "#af67d9", "#e64b17", "#1378cd", "#be5466",
        "#6f969e", "#aa692c", "#60ccc0", "#f2a45d", "#f2a45d"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fig = go.Figure()
    
    def _process_series(self, series):
        cm = iter(self.colormaps)
        wfcm = iter(self.wireframe_colors)

        # if legend=True and both 3d lines and surfaces are shown, then hide the
        # surfaces color bars and only shows line labels in the legend.
        show_3D_colorscales = True
        for s in series:
            if s.is_3Dline:
                show_3D_colorscales = False
        
        for i, s in enumerate(series):
            if s.is_2Dline:
                x, y = s.get_data()
                if s.is_parametric:
                    length = self._line_length(x, y, 
                        start = s.start, end = s.end)
                    self._fig.add_trace(
                        go.Scatter(
                            x = x, y = y, name = s.label,
                            mode = "lines+markers",
                            marker = dict(
                                color = length,
                                colorscale = next(cm),
                                size = 6
                            )
                        )
                    )
                else:
                    self._fig.add_trace(go.Scatter(x = x, y = y, name = s.label))
            elif s.is_3Dline:
                x, y, z = s.get_data()
                if s.is_parametric:
                    length = self._line_length(x, y, z, start = s.start,
                        end = s.end)
                    self._fig.add_trace(
                        go.Scatter3d(
                            x = x, y = y, z = z,
                            name = s.label, mode = "lines+markers",
                            line = dict(width = 4),
                            marker = dict(
                                color = length,
                                colorscale = next(cm),
                                size = 4
                            )
                        )
                    )
                else:
                    self._fig.add_trace(
                    go.Scatter3d(
                        x = x, y = y, z = z,
                        name = s.label, mode = "lines",
                        line = dict(width = 4)
                    )
                )
            elif s.is_3Dsurface:
                xx, yy, zz = s.get_data()
                self._fig.add_trace(
                    go.Surface(
                        x = xx, y = yy, z = zz,
                        name = s.label,
                        showscale = self.legend and show_3D_colorscales,
                        colorbar = dict(
                            x = 1 + 0.1 * i
                        ),
                        colorscale = next(cm)
                    )
                )
                
                if self._kwargs.get("wireframe", False):
                    line_marker = dict(
                        color = next(wfcm),
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
                self._fig.add_trace(go.Contour(x = xx, y = yy, z = zz))
            else:
                raise NotImplementedError
        
    def _update_layout(self):
        self._fig.update_layout(
            template = self._kwargs.get("theme", "plotly_dark"),
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
            ),
            yaxis = dict(
                title = "" if not self.ylabel else self.ylabel,
                range = None if not self.ylim else self.ylim,
                type = self.yscale,
                showgrid = self.axis, # thin lines in the background
                zeroline = self.axis, # thick line at x=0
                visible = self.axis,  # numbers below
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
                aspectmode = "cube"
            )
        )
    
    def show(self):
        self._process_series(self._series)
        self._update_layout()
        self._fig.show()

    def save(self, path):
        self._process_series(self._series)
        self._update_layout()
        self._fig.write_image(path)
