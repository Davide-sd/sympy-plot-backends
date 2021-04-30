from spb.backends.base_backend import Plot
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.palettes import Category10
from bokeh.io import curdoc
from bokeh.models import LinearColorMapper, ColumnDataSource
from bokeh.io import export_png, export_svg
import itertools
import colorcet
import os
import numpy as np

# TODO:
# 1. list of colormaps to loop over for parametric plots
# 

class BokehBackend(Plot):
    """ A backend for plotting SymPy's symbolic expressions using Bokeh.
    Note: this implementation only implements 2D plots.

    Keyword Arguments
    =================

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

    def __new__(cls, *args, **kwargs):
        # Since Plot has its __new__ method, this will prevent infinite
        # recursion
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._get_mode() == 0:
            output_notebook()
        
        # infinity cycler over 10 colors
        self._colors = itertools.cycle(Category10[10])
            
        curdoc().theme = kwargs.get("theme", "dark_minimal")
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
            tooltips = TOOLTIPS
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
                    l = self._line_length(x, y, start=s.start, end=s.end)
                    self._fig.line(x, y, legend_label=s.label,
                                  line_width=2, color=next(self._colors))
                    color_mapper = LinearColorMapper(palette=colorcet.rainbow, 
                        low=min(l), high=max(l))
                    
                    data_source = ColumnDataSource({'x': x , 'y': y, 'l' : l})
                    self._fig.scatter(x='x', y='y', source=data_source,
                                color={'field': 'l', 'transform': color_mapper})
                else:
                    self._fig.line(x, y, legend_label=s.label,
                                line_width=2, color=next(self._colors))
            else:
                raise ValueError(
                    "Bokeh only support 2D plots."
                )

        self._fig.legend.visible = self.legend
        # interactive legend
        self._fig.legend.click_policy = "hide"
        self._fig.add_layout(self._fig.legend[0], 'right')
    
    def _update_interactive(self, params):
        for i, s in enumerate(self.series):
            if s.is_interactive:
                self.series[i].update_data(params)
                
                if s.is_2Dline and s.is_parametric:
                    x, y = self.series[i].get_data()
                    self.fig.renderers[i].data_source.data.update({'x': x, 'y': y})
                    self.fig.renderers[i + 1].data_source.data.update({'x': x, 'y': y})
                if s.is_2Dline and (not s.is_parametric):
                    x, y = self.series[i].get_data()
                    self.fig.renderers[i].data_source.data.update({'y': y})

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