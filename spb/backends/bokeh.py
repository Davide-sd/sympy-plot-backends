from spb.defaults import bokeh_theme
from spb.backends.base_backend import Plot
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import bokeh.palettes as bp
from bokeh.io import curdoc
from bokeh.models import (
    LinearColorMapper, ColumnDataSource, MultiLine, ColorBar
)
from bokeh.io import export_png, export_svg
import itertools
import colorcet as cc
import os
import numpy as np
from mergedeep import merge

class BokehBackend(Plot):
    """ A backend for plotting SymPy's symbolic expressions using Bokeh.
    Note: this implementation only implements 2D plots.

    Keyword Arguments
    =================

        colorbar_kw : dict
            A dictionary with keyword arguments to customize the colorbar.

        theme : str
            Set the theme. Default to "dark_minimal". Find more Bokeh themes at
            the following page:
            https://docs.bokeh.org/en/latest/docs/reference/themes.html

    Export
    ======

    In order to export the plots you will need to install the packages listed
    in the following page:
    https://docs.bokeh.org/en/latest/docs/user_guide/export.html

    At the time of writing this backend, geckodriver is not available to pip.
    Do a quick search on the web to find the appropriate installer.
    """
    
    colormaps = [
        cc.fire, cc.isolum, cc.rainbow, cc.blues, cc.bmy, cc.colorwheel, cc.bgy
    ]
    # TODO: better selection of discrete color maps for contour plots
    contour_colormaps = [
        bp.Plasma10, bp.Blues9, bp.Greys10
    ]

    def __new__(cls, *args, **kwargs):
        # Since Plot has its __new__ method, this will prevent infinite
        # recursion
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._get_mode() == 0:
            output_notebook(
                hide_banner=True
            )
        
        self._colors = itertools.cycle(bp.Category10[10])
        self._cm = itertools.cycle(self.colormaps)
        self._ccm = itertools.cycle(self.contour_colormaps)

        curdoc().theme = kwargs.get("theme", bokeh_theme)
        TOOLTIPS = [
            ("x", "$x"),
            ("y", "$y")
        ]
        self._fig = figure(
            title = self.title,
            x_axis_label = self.xlabel if self.xlabel else "x",
            y_axis_label = self.ylabel if self.ylabel else "y",
            sizing_mode = "fixed" if self.size else "stretch_width",
            width = int(self.size[0]) if self.size else 500,
            height = int(self.size[1]) if self.size else 400,
            x_axis_type = self.xscale,
            y_axis_type = self.yscale,
            x_range = self.xlim,
            y_range = self.ylim,
            tools = "pan,wheel_zoom,box_zoom,reset,hover,save",
            tooltips = TOOLTIPS,
            match_aspect = True if self.aspect_ratio == "equal" else False
        )
        self._fig.axis.visible = self.axis
        self._fig.grid.visible = self.axis

    def _process_series(self, series):
        # clear figure
        self._fig.renderers = []

        for i, s in enumerate(series):
            if s.is_2Dline:
                x, y = s.get_data()
                # Bokeh is not able to deal with None values. Need to replace
                # them with np.nan
                y = [t if (t is not None) else np.nan for t in y]
                if s.is_parametric:
                    u = s.discretized_var
                    ds, line, cb = self._create_gradient_line(x, y, u,
                            next(self._cm), s.label)
                    self._fig.add_glyph(ds, line)
                    self._fig.add_layout(cb, "right")
                else:
                    self._fig.line(x, y, legend_label=s.label,
                                line_width=2, color=next(self._colors))
            elif s.is_contour:
                x, y, z = s.get_data()
                x = x.flatten()
                y = y.flatten()
                zz = z.flatten()
                minx, miny, minz = min(x), min(y), min(zz)
                maxx, maxy, maxz = max(x), max(y), max(zz)

                cm = next(self._ccm)
                self._fig.image(image=[z], x=minx, y=miny,
                        dw=abs(maxx- minx), dh=abs(maxy- miny),
                        palette=cm)
                
                colormapper = LinearColorMapper(palette=cm, low=minz, high=maxz)
                # default options
                cbkw = dict(width = 8)
                # user defined options
                colorbar_kw = self._kwargs.get("colorbar_kw", dict())
                colorbar = ColorBar(color_mapper=colormapper, title=s.label,
                    **merge({}, cbkw, colorbar_kw))
                self._fig.add_layout(colorbar, 'right')
            else:
                raise ValueError(
                    "Bokeh only support 2D plots."
                )

        if len(self._fig.legend) > 0:
            self._fig.legend.visible = self.legend
            # interactive legend
            self._fig.legend.click_policy = "hide"
            self._fig.add_layout(self._fig.legend[0], 'right')
    
    def _get_segments(self, x, y, u):
        # MultiLine works with line segments, not with line points! :|
        xs = [x[i-1:i+1] for i in range(1, len(x))]
        ys = [y[i-1:i+1] for i in range(1, len(y))]
        # TODO: let n be the number of points. Then, the number of segments will
        # be (n - 1). Therefore, we remove one parameter. If n is sufficiently
        # high, there shouldn't be any noticeable problem in the visualization.
        us = u[:-1]
        return xs, ys, us

    def _create_gradient_line(self, x, y, u, colormap, name):
        xs, ys, us = self._get_segments(x, y, u)

        color_mapper = LinearColorMapper(palette = colormap, 
            low = min(u), high = max(u))
        data_source = ColumnDataSource(dict(xs = xs, ys = ys, us = us))

        glyph = MultiLine(xs="xs", ys="ys", 
                    line_color={'field': 'us', 'transform': color_mapper}, 
                    line_width=2, name=name)
        # default options
        cbkw = dict(width = 8)
        # user defined options
        colorbar_kw = self._kwargs.get("colorbar_kw", dict())
        colorbar = ColorBar(color_mapper=color_mapper, title=name,
            **merge({}, cbkw, colorbar_kw))
        return data_source, glyph, colorbar

    def _update_interactive(self, params):
        rend = self.fig.renderers
        for i, s in enumerate(self.series):
            if s.is_interactive:
                self.series[i].update_data(params)
                
                if s.is_2Dline and (not s.is_parametric):
                    x, y = self.series[i].get_data()
                    rend[i].data_source.data.update({'x': x, 'y': y})
                elif s.is_2Dline and s.is_parametric:
                    x, y = self.series[i].get_data()
                    u = s.discretized_var
                    xs, ys, us = self._get_segments(x, y, u)
                    rend[i].data_source.data.update({'xs': xs, 'ys': ys, 'us': us})

    def save(self, path, **kwargs):
        self._process_series(self._series)
        ext = os.path.splitext(path)[1]
        if ext == ".svg":
            self._fig.output_backend = "svg"
            export_svg(self.fig, filename=path)
        else:
            if ext == "":
                path += ".png"
            self._fig.output_backend = "canvas"
            export_png(self._fig, filename=path)
    
    def show(self):
        self._process_series(self._series)
        show(self._fig)

BB = BokehBackend