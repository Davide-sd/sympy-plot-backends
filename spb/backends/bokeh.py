from spb.defaults import bokeh_theme
from spb.backends.base_backend import Plot
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import bokeh.palettes as bp
from bokeh.io import curdoc
from bokeh.models import LinearColorMapper, ColumnDataSource, ColorBar
from bokeh.io import export_png, export_svg
import itertools
import colorcet as cc
import os
import numpy as np
from mergedeep import merge
from typing import Any, List, Tuple
import holoviews as hv

# The following function comes from
# https://docs.bokeh.org/en/latest/docs/gallery/quiver.html
def streamlines(x: np.ndarray, y, u, v, density: float = 1) -> Tuple[List[Any], List[Any]]:
    ''' Return streamlines of a vector flow.

    * x and y are 1d arrays defining an *evenly spaced* grid.
    * u and v are 2d arrays (shape [y,x]) giving velocities.
    * density controls the closeness of the streamlines.

    '''

    ## Set up some constants - size of the grid used.
    NGX = len(x)
    NGY = len(y)

    ## Constants used to convert between grid index coords and user coords.
    DX = x[1]-x[0]
    DY = y[1]-y[0]
    XOFF = x[0]
    YOFF = y[0]

    ## Now rescale velocity onto axes-coordinates
    u = u / (x[-1]-x[0])
    v = v / (y[-1]-y[0])
    speed = np.sqrt(u*u+v*v)
    ## s (path length) will now be in axes-coordinates, but we must
    ## rescale u for integrations.
    u *= NGX
    v *= NGY
    ## Now u and v in grid-coordinates.

    NBX = int(30*density)
    NBY = int(30*density)

    blank = np.zeros((NBY,NBX))

    bx_spacing = NGX/float(NBX-1)
    by_spacing = NGY/float(NBY-1)

    def blank_pos(xi, yi):
        return int((xi / bx_spacing) + 0.5), \
               int((yi / by_spacing) + 0.5)

    def value_at(a, xi, yi):
        if type(xi) == np.ndarray:
            x = xi.astype(np.int)
            y = yi.astype(np.int)
        else:
            x = np.int(xi)
            y = np.int(yi)
        a00 = a[y,x]
        a01 = a[y,x+1]
        a10 = a[y+1,x]
        a11 = a[y+1,x+1]
        xt = xi - x
        yt = yi - y
        a0 = a00*(1-xt) + a01*xt
        a1 = a10*(1-xt) + a11*xt
        return a0*(1-yt) + a1*yt

    def rk4_integrate(x0, y0):
        ## This function does RK4 forward and back trajectories from
        ## the initial conditions, with the odd 'blank array'
        ## termination conditions. TODO tidy the integration loops.

        def f(xi, yi):
            dt_ds = 1./value_at(speed, xi, yi)
            ui = value_at(u, xi, yi)
            vi = value_at(v, xi, yi)
            return ui*dt_ds, vi*dt_ds

        def g(xi, yi):
            dt_ds = 1./value_at(speed, xi, yi)
            ui = value_at(u, xi, yi)
            vi = value_at(v, xi, yi)
            return -ui*dt_ds, -vi*dt_ds

        check = lambda xi, yi: xi>=0 and xi<NGX-1 and yi>=0 and yi<NGY-1

        bx_changes = []
        by_changes = []

        ## Integrator function
        def rk4(x0, y0, f):
            ds = 0.01 #min(1./NGX, 1./NGY, 0.01)
            stotal = 0
            xi = x0
            yi = y0
            xb, yb = blank_pos(xi, yi)
            xf_traj = []
            yf_traj = []
            while check(xi, yi):
                # Time step. First save the point.
                xf_traj.append(xi)
                yf_traj.append(yi)
                # Next, advance one using RK4
                try:
                    k1x, k1y = f(xi, yi)
                    k2x, k2y = f(xi + .5*ds*k1x, yi + .5*ds*k1y)
                    k3x, k3y = f(xi + .5*ds*k2x, yi + .5*ds*k2y)
                    k4x, k4y = f(xi + ds*k3x, yi + ds*k3y)
                except IndexError:
                    # Out of the domain on one of the intermediate steps
                    break
                xi += ds*(k1x+2*k2x+2*k3x+k4x) / 6.
                yi += ds*(k1y+2*k2y+2*k3y+k4y) / 6.
                # Final position might be out of the domain
                if not check(xi, yi): break
                stotal += ds
                # Next, if s gets to thres, check blank.
                new_xb, new_yb = blank_pos(xi, yi)
                if new_xb != xb or new_yb != yb:
                    # New square, so check and colour. Quit if required.
                    if blank[new_yb,new_xb] == 0:
                        blank[new_yb,new_xb] = 1
                        bx_changes.append(new_xb)
                        by_changes.append(new_yb)
                        xb = new_xb
                        yb = new_yb
                    else:
                        break
                if stotal > 2:
                    break
            return stotal, xf_traj, yf_traj

        integrator = rk4

        sf, xf_traj, yf_traj = integrator(x0, y0, f)
        sb, xb_traj, yb_traj = integrator(x0, y0, g)
        stotal = sf + sb
        x_traj = xb_traj[::-1] + xf_traj[1:]
        y_traj = yb_traj[::-1] + yf_traj[1:]

        ## Tests to check length of traj. Remember, s in units of axes.
        if len(x_traj) < 1: return None
        if stotal > .2:
            initxb, inityb = blank_pos(x0, y0)
            blank[inityb, initxb] = 1
            return x_traj, y_traj
        else:
            for xb, yb in zip(bx_changes, by_changes):
                blank[yb, xb] = 0
            return None

    ## A quick function for integrating trajectories if blank==0.
    trajectories = []
    def traj(xb, yb):
        if xb < 0 or xb >= NBX or yb < 0 or yb >= NBY:
            return
        if blank[yb, xb] == 0:
            t = rk4_integrate(xb*bx_spacing, yb*by_spacing)
            if t is not None:
                trajectories.append(t)

    ## Now we build up the trajectory set. I've found it best to look
    ## for blank==0 along the edges first, and work inwards.
    for indent in range((max(NBX,NBY))//2):
        for xi in range(max(NBX,NBY)-2*indent):
            traj(xi+indent, indent)
            traj(xi+indent, NBY-1-indent)
            traj(indent, xi+indent)
            traj(NBX-1-indent, xi+indent)

    xs = [np.array(t[0])*DX+XOFF for t in trajectories]
    ys = [np.array(t[1])*DY+YOFF for t in trajectories]

    return xs, ys


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
        # hv.extension('bokeh')
        
        # infinity cycler over 10 colors
        self._cl = itertools.cycle(bp.Category10[10])
        # self._cm = itertools.cycle(self.colormaps)
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
                    l = self._line_length(x, y, start=s.start, end=s.end)
                    self._fig.line(x, y, legend_label=s.label,
                                  line_width=2, color=next(self._cl))
                    color_mapper = LinearColorMapper(palette=colorcet.rainbow, 
                        low=min(l), high=max(l))
                    
                    data_source = ColumnDataSource({'x': x , 'y': y, 'l' : l})
                    self._fig.scatter(x='x', y='y', source=data_source,
                                color={'field': 'l', 'transform': color_mapper})
                else:
                    self._fig.line(x, y, legend_label=s.label,
                                line_width=2, color=next(self._cl))
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
            elif s.is_vector and s.is_2D:
                x, y, u, v = s.get_data()
                # The following comes from:
                # https://docs.bokeh.org/en/latest/docs/gallery/quiver.html
                speed = np.sqrt(u**2 + v**2)
                angle = np.arctan(v / u)
                # print("dc", x.shape)
                # x0 = x[::2, ::2].flatten()
                # y0 = y[::2, ::2].flatten()
                # length = speed[::2, ::2].flatten()/5
                # angle = theta[::2, ::2].flatten()
                # x1 = x0 + length * np.cos(angle)
                # y1 = y0 + length * np.sin(angle)
                # cm = np.array(["#C7E9B4", "#7FCDBB", "#41B6C4", "#1D91C0", "#225EA8", "#0C2C84"])
                # ix = ((length-length.min())/(length.max()-length.min())*5).astype('int')
                # colors = cm[ix]
                # self._fig.segment(x0, y0, x1, y1, color=colors, line_width=2)

                vectorfield = hv.VectorField([x, y, angle, speed])
                p = hv.render(vectorfield, backend='bokeh')
                # ds = p.renderers[0].data_source.data
                # print("dioporco", ds["x0"].shape)
                # self._fig.segment(x, y, angle, speed, color="red", line_width=2)
                # self._fig.segment(ds["x0"], ds["y0"], ds["x1"], ds["y1"], color="#FF0000", line_width=2)
                self._fig.renderers.append(p.renderers[0])
            else:
                raise ValueError(
                    "Bokeh only support 2D plots."
                )

        if len(self._fig.legend) > 0:
            self._fig.legend.visible = self.legend
            # interactive legend
            self._fig.legend.click_policy = "hide"
            self._fig.add_layout(self._fig.legend[0], 'right')
    
    def _update_interactive(self, params):
        # Parametric lines are rendered with two lines:
        # 1. the solid one
        # 2. the gradient dots
        # Hence, need to keep track of how many parametric lines we encounter.
        pc = 0
        rend = self.fig.renderers
        for i, s in enumerate(self.series):
            if s.is_interactive:
                self.series[i].update_data(params)
                
                if s.is_2Dline and s.is_parametric:
                    x, y = self.series[i].get_data()
                    rend[i + pc].data_source.data.update({'x': x, 'y': y})
                    rend[i + pc + 1].data_source.data.update({'x': x, 'y': y})
                    pc += 1
                if s.is_2Dline and (not s.is_parametric):
                    x, y = self.series[i].get_data()
                    rend[i + pc].data_source.data.update({'y': y})

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


# class VectorFieldPlot(ColorbarPlot):

#     arrow_heads = param.Boolean(default=True, doc="""
#         Whether or not to draw arrow heads.""")

#     magnitude = param.ClassSelector(class_=(basestring, dim), doc="""
#         Dimension or dimension value transform that declares the magnitude
#         of each vector. Magnitude is expected to be scaled between 0-1,
#         by default the magnitudes are rescaled relative to the minimum
#         distance between vectors, this can be disabled with the
#         rescale_lengths option.""")

#     padding = param.ClassSelector(default=0.05, class_=(int, float, tuple))

#     pivot = param.ObjectSelector(default='mid', objects=['mid', 'tip', 'tail'],
#                                  doc="""
#         The point around which the arrows should pivot valid options
#         include 'mid', 'tip' and 'tail'.""")

#     rescale_lengths = param.Boolean(default=True, doc="""
#         Whether the lengths will be rescaled to take into account the
#         smallest non-zero distance between two vectors.""")

#     # Deprecated parameters

#     color_index = param.ClassSelector(default=None, class_=(basestring, int),
#                                       allow_None=True, doc="""
#         Deprecated in favor of dimension value transform on color option,
#         e.g. `color=dim('Magnitude')`.
#         """)

#     size_index = param.ClassSelector(default=None, class_=(basestring, int),
#                                      allow_None=True, doc="""
#         Deprecated in favor of the magnitude option, e.g.
#         `magnitude=dim('Magnitude')`.
#         """)

#     normalize_lengths = param.Boolean(default=True, doc="""
#         Deprecated in favor of rescaling length using dimension value
#         transforms using the magnitude option, e.g.
#         `dim('Magnitude').norm()`.""")

#     selection_display = BokehOverlaySelectionDisplay()

#     style_opts = base_properties + line_properties + ['scale', 'cmap']

#     _nonvectorized_styles = base_properties + ['scale', 'cmap']

#     _plot_methods = dict(single='segment')

#     def _get_lengths(self, element, ranges):
#         size_dim = element.get_dimension(self.size_index)
#         mag_dim = self.magnitude
#         if size_dim and mag_dim:
#             self.param.warning(
#                 "Cannot declare style mapping for 'magnitude' option "
#                 "and declare a size_index; ignoring the size_index.")
#         elif size_dim:
#             mag_dim = size_dim
#         elif isinstance(mag_dim, basestring):
#             mag_dim = element.get_dimension(mag_dim)

#         (x0, x1), (y0, y1) = (element.range(i) for i in range(2))
#         if mag_dim:
#             if isinstance(mag_dim, dim):
#                 magnitudes = mag_dim.apply(element, flat=True)
#             else:
#                 magnitudes = element.dimension_values(mag_dim)
#                 _, max_magnitude = ranges[dimension_name(mag_dim)]['combined']
#                 if self.normalize_lengths and max_magnitude != 0:
#                     magnitudes = magnitudes / max_magnitude
#             if self.rescale_lengths:
#                 base_dist = get_min_distance(element)
#                 magnitudes *= base_dist
#         else:
#             magnitudes = np.ones(len(element))
#             if self.rescale_lengths:
#                 base_dist = get_min_distance(element)
#                 magnitudes *= base_dist

#         return magnitudes

#     def _glyph_properties(self, *args):
#         properties = super(VectorFieldPlot, self)._glyph_properties(*args)
#         properties.pop('scale', None)
#         return properties


#     def get_data(self, element, ranges, style):
#         input_scale = style.pop('scale', 1.0)

#         # Get x, y, angle, magnitude and color data
#         rads = element.dimension_values(2)
#         if self.invert_axes:
#             xidx, yidx = (1, 0)
#             rads = np.pi/2 - rads
#         else:
#             xidx, yidx = (0, 1)
#         lens = self._get_lengths(element, ranges)/input_scale
#         cdim = element.get_dimension(self.color_index)
#         cdata, cmapping = self._get_color_data(element, ranges, style,
#                                                name='line_color')

#         # Compute segments and arrowheads
#         xs = element.dimension_values(xidx)
#         ys = element.dimension_values(yidx)

#         # Compute offset depending on pivot option
#         xoffsets = np.cos(rads)*lens/2.
#         yoffsets = np.sin(rads)*lens/2.
#         if self.pivot == 'mid':
#             nxoff, pxoff = xoffsets, xoffsets
#             nyoff, pyoff = yoffsets, yoffsets
#         elif self.pivot == 'tip':
#             nxoff, pxoff = 0, xoffsets*2
#             nyoff, pyoff = 0, yoffsets*2
#         elif self.pivot == 'tail':
#             nxoff, pxoff = xoffsets*2, 0
#             nyoff, pyoff = yoffsets*2, 0
#         x0s, x1s = (xs + nxoff, xs - pxoff)
#         y0s, y1s = (ys + nyoff, ys - pyoff)

#         color = None
#         if self.arrow_heads:
#             arrow_len = (lens/4.)
#             xa1s = x0s - np.cos(rads+np.pi/4)*arrow_len
#             ya1s = y0s - np.sin(rads+np.pi/4)*arrow_len
#             xa2s = x0s - np.cos(rads-np.pi/4)*arrow_len
#             ya2s = y0s - np.sin(rads-np.pi/4)*arrow_len
#             x0s = np.tile(x0s, 3)
#             x1s = np.concatenate([x1s, xa1s, xa2s])
#             y0s = np.tile(y0s, 3)
#             y1s = np.concatenate([y1s, ya1s, ya2s])
#             if cdim and cdim.name in cdata:
#                 color = np.tile(cdata[cdim.name], 3)
#         elif cdim:
#             color = cdata.get(cdim.name)

#         data = {'x0': x0s, 'x1': x1s, 'y0': y0s, 'y1': y1s}
#         mapping = dict(x0='x0', x1='x1', y0='y0', y1='y1')
#         if cdim and color is not None:
#             data[cdim.name] = color
#             mapping.update(cmapping)

#         return (data, mapping, style)
