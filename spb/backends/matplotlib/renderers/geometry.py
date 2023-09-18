from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.backends.matplotlib.renderers.line2d import (
    _draw_line2d_helper, _update_line2d_helper
)
from spb.backends.matplotlib.renderers.line3d import (
    _draw_line3d_helper, _update_line3d_helper
)


def _draw_geometry_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    x, y = data
    color = next(p._cl)
    fkw = dict(facecolor=color, fill=s.is_filled, edgecolor=color)
    kw = p.merge({}, fkw, s.rendering_kw)
    c = p._ax.fill(x, y, **kw)
    proxy_artist = p.Rectangle(
        (0, 0), 1, 1,
        color=color, label=s.get_label(p._use_latex)
    )
    if s.show_in_legend:
        p._legend_handles.append(proxy_artist)
    return [c, kw]


def _update_geometry_helper(renderer, data, handle):
    p = renderer.plot
    x, y = data
    # NOTE: ax.fill sometimes returns a list of polygons, sometime one polygon.
    if not isinstance(handle[0], (list, tuple)):
        handle[0] = [handle[0]]
    for h in handle[0]:
        h.remove()
    handle[0] = p._ax.fill(x, y, **handle[1])[0]


class GeometryRenderer(MatplotlibRenderer):
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
