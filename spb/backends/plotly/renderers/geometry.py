from spb.backends.base_renderer import Renderer
from spb.backends.plotly.renderers.line2d import _draw_line2d_helper, _update_line2d_helper
from spb.backends.plotly.renderers.line3d import _draw_line3d_helper, _update_line3d_helper


def _draw_geometry_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    x, y = data
    lkw = dict(
        name=s.get_label(p._use_latex), mode="lines",
        fill="toself", line_color=next(p._cl)
    )
    kw = p.merge({}, lkw, s.rendering_kw)
    handle = p.go.Scatter(x=x, y=y, **kw)
    p._fig.add_trace(handle)
    return handle


def _update_geometry_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    x, y = data
    handle["x"] = x
    handle["y"] = y


class GeometryRenderer(Renderer):
    draw_update_map = {
        _draw_geometry_helper: _update_geometry_helper
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.series.is_3Dline:
            self.draw_update_map = {
                _draw_line3d_helper: _update_line3d_helper
            }
        elif self.series.is_2Dline:
            self.draw_update_map = {
                _draw_line2d_helper: _update_line2d_helper
            }
