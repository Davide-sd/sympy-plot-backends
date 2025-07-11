from spb.backends.base_renderer import Renderer
from spb.backends.bokeh.renderers.polezero import _get_source_tooltips
from sympy.external import import_module


def _get_x_y_coords(roots, gains):
    """Create one single line, where branches are separated by NaNs.
    this makes it easier to deal with widget's update, because with
    interactive widgets plot the number of roots/poles can vary.
    """
    np = import_module("numpy")
    x, y, g = [], [], []
    for col in roots.T:
        x.append(np.real(col))
        x.append(np.array([np.nan]))
        y.append(np.imag(col))
        y.append(np.array([np.nan]))
        g.append(gains)
        g.append(np.array([np.nan]))
    x = np.concatenate(x)
    y = np.concatenate(y)
    g = np.concatenate(g)
    return x, y, g


def _draw_root_locus_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    roots, gains = data

    color = next(p._cl) if s.line_color is None else s.line_color
    lkw = dict(
        line_color=color, line_width=2, legend_label=s.get_label(p.use_latex)
    )
    if s.is_point:
        lkw["marker"] = "o"

    kw = p.merge({}, lkw, s.rendering_kw)
    x, y, gains = _get_x_y_coords(roots, gains)
    source, tooltips = _get_source_tooltips(p, x, y)
    # add gains to tooltips
    source["k"] = gains
    tooltips.insert(2, ("K", "@k"))

    line = p._fig.line("x", "y", source=source, **kw)
    p._fig.add_tools(p.bokeh.models.HoverTool(
        tooltips=tooltips,
        renderers=[line]
    ))
    p._fig.add_tools(p.bokeh.models.TapTool())

    zrk = p.merge({},
        dict(marker="o", line_color=color, line_width=2,
            fill_color="white", size=7),
        s._zeros_rk)
    prk =  p.merge({},
        dict(marker="x", line_color=color, line_width=2, size=10),
        s._poles_rk)

    hz = p._fig.scatter(
        "x", "y", source={"x": s.zeros.real, "y": s.zeros.imag}, **zrk)
    hp = p._fig.scatter(
        "x", "y", source={"x": s.poles.real, "y": s.poles.imag}, **prk)

    # bind scatter's visibility to the line's visibility
    callback = p.bokeh.models.CustomJS(
        args=dict(line=line, hz=hz, hp=hp),
        code="""
        hz.visible = line.visible;
        hp.visible = line.visible;
        """
    )
    line.js_on_change("visible", callback)

    return [line, hz, hp]


def _update_root_locus_helper(renderer, data, handles):
    p, s = renderer.plot, renderer.series
    np = p.np
    roots, gains = data

    x, y, gains = _get_x_y_coords(roots, gains)
    source, _ = _get_source_tooltips(p, x, y)
    source["k"] = gains
    handles[0].data_source.data.update(source)
    handles[1].data_source.data.update({"x": s.zeros.real, "y": s.zeros.imag})
    handles[2].data_source.data.update({"x": s.poles.real, "y": s.poles.imag})


class RootLocusRenderer(Renderer):
    draw_update_map = {
        _draw_root_locus_helper: _update_root_locus_helper
    }

    def draw(self):
        data = self.series.get_data()
        self._set_axis_limits([self.series.xlim, self.series.ylim])
        for draw_method in self.draw_update_map.keys():
            self.handles.append(
                draw_method(self, data))

    def update(self, params):
        self.series.params = params
        data = self.series.get_data()
        self._set_axis_limits([self.series.xlim, self.series.ylim])
        for update_method, handle in zip(
            self.draw_update_map.values(), self.handles
        ):
            update_method(self, data, handle)

    def _set_axis_limits(self, data):
        np = self.plot.np
        # NOTE: axis limits cannot be NaN or Inf
        self._xlims = [[np.nanmin(data[0]), np.nanmax(data[0])]]
        self._ylims = [[np.nanmin(data[1]), np.nanmax(data[1])]]
