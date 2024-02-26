import numpy as np
from spb.backends.base_renderer import Renderer


def _draw_arrow_3d(renderer, data):
    p, s = renderer.plot, renderer.series
    xx, yy, zz, uu, vv, ww = data
    uu -= xx
    vv -= yy
    ww -= zz

    color = p._rgb_to_int([int(255*t) for t in next(p._cl)])
    pkw = dict(
        origin_color=color,
        head_color=color
    )
    kw = p.merge({}, pkw, s.rendering_kw)
    arrow = p.k3d.vectors([xx, yy, zz], [uu, vv, ww], **kw)
    p._fig += arrow

    return [arrow]


def _update_arrow3d(renderer, data, handle):
    p = renderer.plot
    xx, yy, zz, uu, vv, ww = data
    uu -= xx
    vv -= yy
    ww -= zz
    handle[0].origins = np.array([xx, yy, zz])
    handle[0].vectors = np.array([uu, vv, ww])


class Arrow3DRenderer(Renderer):
    draw_update_map = {
        _draw_arrow_3d: _update_arrow3d
    }
