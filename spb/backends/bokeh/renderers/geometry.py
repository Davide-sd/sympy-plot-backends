from spb.backends.base_renderer import Renderer
from spb.backends.bokeh.renderers.line2d import (
    _draw_line2d_helper, _update_line2d_helper
)


def _draw_geometry_helper(renderer, data):
    p, s = renderer.plot, renderer.series

    x, y = data
    color = next(p._cl)
    pkw = dict(alpha=0.5, line_width=2, line_color=color, fill_color=color)
    kw = p.merge({}, pkw, s.rendering_kw)

    alpha = kw["alpha"]

    # need a scatter in order to show tooltips
    source = {"x": x, "y": y}
    tooltips = [("x", "@x"), ("y", "@y")]
    scatter = p._fig.scatter(
        "x", "y", source=source, color=color, size=4, alpha=alpha)
    p._fig.add_tools(p.bokeh.models.HoverTool(
        tooltips=tooltips,
        renderers=[scatter]
    ))
    handle = p._fig.patch(x, y, **kw)

    return [handle, scatter]


def _update_geometry_helper(renderer, data, handle):
    x, y = data
    source = {"x": x, "y": y}
    geometry, scatter = handle
    geometry.data_source.data.update(source)
    scatter.data_source.data.update(source)


class GeometryRenderer(Renderer):
    draw_update_map = {
        _draw_geometry_helper: _update_geometry_helper
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.series.is_2Dline:
            self.draw_update_map = {
                _draw_line2d_helper: _update_line2d_helper
            }
