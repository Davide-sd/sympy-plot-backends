from spb.backends.base_renderer import Renderer
from spb.utils import get_vertices_indices
from spb.series import PlaneSeries

def _draw_surface_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    mlab = p.mlab
    
    if s.is_parametric:
        x, y, z, u, v = data
        attribute = s.eval_color_func(x, y, z, u, v)
    else:
        x, y, z = data
        attribute = s.eval_color_func(x, y, z)
    a = dict(
        color=None if s.use_cm else (
            next(p._cl) if s.surface_color is None
            else s.surface_color),
        colormap=next(p._cm),
        extent=p._get_extent(x, y, z)
    )
    kw = p.merge({}, a, s.rendering_kw)
    p._add_figure_to_kwargs(kw)
    colorbar_kw = kw.pop("colorbar_kw", dict())
    if not "scalars" in kw.keys():
        kw["scalars"] = attribute
    obj = mlab.mesh(x, y, z, **kw)
    p._add_colorbar(s, obj, colorbar_kw, kw.get("color", None))
    return obj


def _update_surface_helper(renderer, data, handle):
    raise NotImplementedError
    

class SurfaceRenderer(Renderer):
    draw_update_map = {
        _draw_surface_helper: _update_surface_helper
    }
