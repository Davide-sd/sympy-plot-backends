import param
import numpy as np
import panel as pn
from sympy import latex, Tuple
from spb.backends.bokeh import BB
from spb.backends.plotly import PB
from spb.backends.k3d import KB
from spb.backends.base_backend import Plot
from spb.series import InteractiveSeries
import plotly.graph_objects as go
import k3d
from matplotlib.tri import Triangulation
from spb.utils import get_lambda, _plot_sympify, _is_range
from spb.defaults import I_B
pn.extension("plotly")


"""
TODO:
    1. Make each interactive class to use the backend to initially create the
        plot... Probably need to create List3DSeries, ....
    2. Move bg_color, fg_color, ... of backends into defaults.py and update the
        tutorials accordingly.
"""

class DynamicParam(param.Parameterized):
    """ Dynamically add parameters based on the user-provided dictionary.
    Also, generate the lambda functions to be evaluated at a later stage.
    """
    
    def _tuple_to_dict(self, k, v):
        """ The user can provide a variable length tuple/list containing:
            (default, softbounds, N, label, type (lin, log))
        where:
            default : float
                Default value of the slider
            softbounds : tuple
                Tuple of two float (or integer) numbers: (start, end).
            N : int
                Number of increments in the slider. (start - end) / N represents
                the step increment. Default to 40. Set N=-1 to have unit step increments.
            label : str
                Label of the slider. Default to None. If None, the string or
                latex representation will be used. See use_latex for more information.
            type : str
                Can be "linear" or "log". Default to "linear".
        """
        defaults_keys = ["default", "softbounds", "step", "label", "type"]
        defaults_values = [1, (0, 2), 40, "$%s$" % latex(k) if self.use_latex else str(k), "linear"]
        values = defaults_values.copy()
        values[:len(v)] = v
        # set the step increment for the slider
        if values[2] > 0:
            values[2] = (values[1][1] - values[1][0]) / values[2]
        else:
            values[2] = 1
        return {k: v for k, v in zip (defaults_keys, values)}
    
    def __init__(self, *args, name="", parameters=None, **kwargs):
        # use latex on control labels and legends
        self.use_latex = kwargs.pop("use_latex", True)
        
        # this must be present in order to assure correct behaviour
        super().__init__(name=name, **kwargs)
        if not parameters:
            raise ValueError("`parameters` must be provided.")
        
        # The following dictionary will be used to create the appropriate
        # lambda function arguments:
        #    key: the provided symbol
        #    val: name of the associated parameter
        self.mapping = {}
        
        # create and attach the params to the class
        for i, (k, v) in enumerate(parameters.items()):
            if not isinstance(v, param.parameterized.Parameter):
                v = self._tuple_to_dict(k, v)
                # TODO: modify this to implement log slider
                v.pop("type", None)
                v = param.Number(**v)
            
            # TODO: using a private method: not the smartest thing to do
            self.param._add_parameter("dyn_param_{}".format(i), v)
            self.mapping[k] = "dyn_param_{}".format(i)
    
    def read_parameters(self):
        readout = dict()
        for k, v in self.mapping.items():
            readout[k] = getattr(self, v)
        return readout
        
    def layout_controls(self):
        # split the controls in two columns
        params = sorted(list(self.mapping.values()))
        n = int(len(params) / 2)
        left = pn.panel(self, parameters=params[:n])
        right = pn.panel(self, parameters=params[n:])
        return pn.Row(left, right)
        
class InteractivePlot(DynamicParam):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, name="", parameters=None, fig_kw=None, **kwargs):
        args = list(map(_plot_sympify, args))
        super().__init__(*args, name="", parameters=parameters, **kwargs)
        
        # create the series
        series = self._create_series(*args, **fig_kw)
        # create the plot
        Backend = fig_kw.pop("backend", I_B)
        self._backend = Backend(*series, **fig_kw)
        
        # read the parameters and generate the initial numerical data for
        # the interactive series
        args = self.read_parameters()
        for s in self._backend.series:
            if s.is_interactive:
                s.update_data(args)
        # add the series to the plot
        self._backend._process_series(self._backend._series)
        
    
    def _create_series(self, *args, **kwargs):
        # TODO: would be nice to analyze the arguments/parameters and decide
        # which kind of series we should build.
        series = []
        for a in args:
            series.append(InteractiveSeries(*a, fs_params=self.mapping, **kwargs))
        return series
    
    def fig(self):
        """ Return the plot object
        """
        return self._backend.fig
            
    def view(self):
        params = self.read_parameters()
        self._backend._update_interactive(params)
        if isinstance(self._backend, KB):
            return pn.pane.Pane(self._backend.fig, width=800)
        else:
            return self.fig
    
    def show(self):
        return pn.Column(self.layout_controls, self.view)

# class BokehInteractive(DynamicParam, BB):
#     def __init__(self, *args, name="", parameters=None, fig_kw=None, **kwargs):
#         super().__init__(*args, name="", parameters=parameters, **kwargs)
#         BB.__init__(self, **fig_kw)
#         self.data_added = False
    
    
#     def update(self):
#         for i in range(len(self._functions)):
#             f, args, ranges, label = self._functions[i]
#             args = self.read_parameters(args)
            
#             if len(ranges) > 1:
#                 raise ValueError(
#                         "BokehInteractive only support 2D plots, but the " +
#                         "provided expressions uses multiple ranges.")
#             x = list(ranges.values())[0]
#             y = f(*args)
#             if not self.data_added:
#                 self.fig.line(y, x, legend_label=label,
#                               color=next(self._colors), line_width=2)
#             else:
#                 self.fig.renderers[i].data_source.data.update({'x': y, 'y': x})
        
#         if not self.data_added:
#             self._fig.legend.visible = self.legend
#             # interactive legend
#             self._fig.legend.click_policy = "hide"
#             self._fig.add_layout(self._fig.legend[0], 'right')
#             self.data_added = True
            

#     def view(self):
#         self.update()
#         return pn.pane.Bokeh(self.fig)
    
#     def show(self):
#         return pn.Column(self.layout_controls, self.view)


# class PlotlyInteractive(DynamicParam, PB):
#     def __init__(self, *args, name="", parameters=None, fig_kw=None, **kwargs):
#         super().__init__(*args, name="", parameters=parameters, **kwargs)
#         PB.__init__(self, **fig_kw)
#         self.data_added = False
    
    
#     def update(self):
#         for i in range(len(self._functions)):
#             f, args, ranges, label = self._functions[i]
#             args = self.read_parameters(args)
            
#             if len(ranges) > 2:
#                 raise ValueError(
#                     "PlotlyInteractive only support 2D and 3D plots, but the " +
#                     "provided expressions uses {} ranges.".format(len(ranges)))
#             if len(ranges) == 1:
#                 x = list(ranges.values())[0]
#                 y = f(*args)
#                 if not self.data_added:
#                     self.fig.add_trace(go.Scatter(x = y, y = x, name = label))
#                 else:
#                     self.fig.data[i]["x"] = y
#             else:
#                 x, y = ranges.values()
#                 z = f(*args)
#                 if not self.data_added:
#                     self._fig.add_trace(
#                         go.Surface(
#                             x = x, y = y, z = z,
#                             name = label,
#                             colorscale = next(self._cm)
#                         )
#                     )
#                 else:
#                     self.fig.data[i]["z"] = z
#         self.data_added = True
            

#     def view(self):
#         self.update()
#         return pn.pane.Plotly(self.fig)
    
#     def show(self):
#         return pn.Column(self.layout_controls, self.view)


# class K3DInteractive(DynamicParam, KB):
#     def __init__(self, *args, name="", parameters=None, fig_kw=None, **kwargs):
#         super().__init__(*args, name="", parameters=parameters, **kwargs)
#         KB.__init__(self, **fig_kw)
#         self.data_added = False
#         self.renderers = []
#         self._populate_plot()
        
    
#     def _populate_plot(self):
#         for i in range(len(self._functions)):
#             f, args, ranges, label = self._functions[i]
#             args = self.read_parameters(args)
            
#             if (len(ranges) == 1) or (len(ranges) > 2):
#                 raise ValueError(
#                     "K3DInteractive only support 3D plots, but the " +
#                     "provided expressions uses {} ranges.".format(len(ranges)))
            
#             x, y = ranges.values()
#             z = f(*args)
#             x = x.flatten()
#             y = y.flatten()
#             z = z.flatten()
#             vertices = np.vstack([x, y, z])
#             indices = Triangulation(x, y).triangles.astype(np.uint32)
#             a = dict(
#                 name = label if self._kwargs.get("show_label", False) else None,
#                 side = "double",
#                 flat_shading = False,
#                 wireframe = self._kwargs.get("wireframe", False),
#                 color = self._convert_to_int(next(self._iter_colorloop)),
#             )
#             if self._use_cm:
#                 a["color_map"] = next(self._iter_colormaps)
#                 a["attribute"] = z
#             surf = k3d.mesh(vertices.T, indices, **a)
#             self.renderers.append(surf)
#             self._fig += surf
            
    
#     def update(self):
#         for i in range(len(self._functions)):
#             f, args, ranges, label = self._functions[i]
#             args = self.read_parameters(args)
#             x, y = ranges.values()
#             z = f(*args)   
#             x = x.flatten()
#             y = y.flatten()
#             z = z.flatten()
#             vertices = np.vstack([x, y, z])
#             self.renderers[i].vertices= vertices.T

#             ## TODO: This doesn't work, why?
#             # self.renderers[i].vertices[:, 2] = z
            

#     def view(self):
#         self.update()
#         return pn.pane.Pane(self.fig, width=800)
    
#     def show(self):
#         return pn.Column(self.layout_controls, self.view)