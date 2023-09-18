from spb.backends.base_renderer import Renderer
from spb.backends.k3d.renderers.line3d import (
    _draw_line3d_helper, _update_line3d_helper
)


class GeometryRenderer(Renderer):
    draw_update_map = {
        _draw_line3d_helper: _update_line3d_helper
    }
