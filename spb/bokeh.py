from base_backend import MyBaseBackend
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.palettes import Category10
from bokeh.io import curdoc
from bokeh.models import LinearColorMapper, ColumnDataSource
from bokeh.io import export_png, export_svg
import itertools
import colorcet
import os

# TODO:
# 1. list of colormaps to loop over for parametric plots
# 

class BokehBackend(MyBaseBackend):
    """ A backend for plotting SymPy's symbolic expressions using Bokeh.
    Note: this implementation only implements 2D plots.

    Note: in order to export plots you will need to install the packages listed
    in the following page:

    https://docs.bokeh.org/en/latest/docs/user_guide/export.html

    At the time of writing this backend, geckodriver is not available to pip.
    Do a quick search on the web to find the appropriate installer.
    """
    def __init__(self, parent):
        super().__init__(parent)

        if self._get_mode() == 0:
            output_notebook()
            
        curdoc().theme = "dark_minimal"
        TOOLTIPS = [
            ("x", "$x"),
            ("y", "$y")
        ]
        self.fig = figure(
            title = parent.title,
            x_axis_label = parent.xlabel if parent.xlabel else "x",
            y_axis_label = parent.ylabel if parent.ylabel else "y",
            sizing_mode = "fixed" if parent.size else "stretch_width",
            width = int(parent.size[0]) if parent.size else 500,
            height = int(parent.size[1]) if parent.size else 400,
            x_axis_type = parent.xscale,
            y_axis_type = parent.yscale,
            x_range = parent.xlim,
            y_range = parent.ylim,
            tools = "pan,wheel_zoom,box_zoom,reset,hover",
            tooltips = TOOLTIPS
        )
        self.fig.axis.visible = parent.axis
        self.fig.grid.visible = parent.axis

    def _process_series(self, series):
        colors = itertools.cycle(Category10[10])

        for i, s in enumerate(series):
            if s.is_2Dline:
                if s.is_parametric:
                    x, y = s.get_data(False)
                    l = self._line_length(x, y, start=s.start, end=s.end)
                    self.fig.line(*s.get_data(False), legend_label=s.label,
                                  line_width=2, color=next(colors))
                    color_mapper = LinearColorMapper(palette=colorcet.rainbow, 
                        low=min(l), high=max(l))
                    
                    data_source = ColumnDataSource({'x': x , 'y': y, 'l' : l})
                    self.fig.scatter(x='x', y='y', source=data_source,
                                color={'field': 'l', 'transform': color_mapper})
                else:
                    self.fig.line(*s.get_data(False), legend_label=s.label,
                                line_width=2, color=next(colors))
            else:
                raise ValueError(
                    "Bokeh only support 2D plots."
                )

        self.fig.legend.visible = self.parent.legend
        self.fig.add_layout(self.fig.legend[0], 'right')

    def save(self, path):
        self._process_series(self.parent._series)
        ext = os.path.splitext(path)[1]
        if ext == ".svg":
            self.fig.output_backend = "svg"
            export_svg(self.fig, filename=path)
        else:
            if ext == "":
                path += ".png"
            self.fig.output_backend = "canvas"
            export_png(self.fig, filename=path)
    
    def show(self):
        self._process_series(self.parent._series)
        show(self.fig)
