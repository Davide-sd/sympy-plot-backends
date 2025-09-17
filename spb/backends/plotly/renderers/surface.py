from spb.backends.base_renderer import Renderer
from spb.backends.utils import _get_cmin_cmax, _returns_z_coord
import inspect
from sympy import Expr


def _draw_surface_helper(renderer, data):
    p, s = renderer.plot, renderer.series

    if not s.is_parametric:
        xx, yy, zz = data
        surfacecolor = s.eval_color_func(xx, yy, zz)
    else:
        xx, yy, zz, uu, vv = data
        surfacecolor = s.eval_color_func(xx, yy, zz, uu, vv)

    # create a solid color to be used when s.use_cm=False
    col = next(p._cl) if s.surface_color is None else s.surface_color
    colorscale = [[0, col], [1, col]]
    colormap = next(p._cm)
    cmin, cmax = _get_cmin_cmax(surfacecolor, p, s)
    skw = dict(
        name=s.get_label(p.use_latex),
        showscale=s.use_cm and s.colorbar,
        showlegend=(not s.use_cm) and s.show_in_legend,
        colorbar=p._create_colorbar(
            s.get_label(p.use_latex), p._scale_down_colorbar),
        colorscale=colormap if s.use_cm else colorscale,
        surfacecolor=surfacecolor,
        cmin=cmin,
        cmax=cmax
    )

    kw = p.merge({}, skw, s.rendering_kw)
    handle = p.go.Surface(x=xx, y=yy, z=zz, **kw)
    p._fig.add_trace(handle)
    return len(p._fig.data) - 1


def _update_surface_helper(renderer, data, idx):
    p, s = renderer.plot, renderer.series
    handle = p.fig.data[idx]

    if not s.is_parametric:
        x, y, z = data
        surfacecolor = s.eval_color_func(x, y, z)
    else:
        x, y, z, u, v = data
        surfacecolor = s.eval_color_func(x, y, z, u, v)
    handle["x"] = x
    handle["y"] = y
    cmin, cmax = _get_cmin_cmax(surfacecolor, p, s)
    handle["z"] = z
    handle["surfacecolor"] = surfacecolor
    handle["cmin"] = cmin
    handle["cmax"] = cmax


class SurfaceRenderer(Renderer):
    draw_update_map = {
        _draw_surface_helper: _update_surface_helper
    }
