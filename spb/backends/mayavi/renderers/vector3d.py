from spb.backends.base_renderer import Renderer
from spb.backends.utils import get_seeds_points, compute_streamtubes
import warnings


def _draw_vector3d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    mlab = p.mlab
    
    x, y, z, u, v, w = data
    a = dict(
        color=None if s.use_cm else (
            next(p._cl) if s.line_color is None
            else s.line_color),
        colormap=next(p._cm),
        extent=p._get_extent(x, y, z)
    )
    # remove unused keys
    unused_keys = ["starts", "npoints", "max_prop", "radius"]
    provided_keys = []
    for k in unused_keys:
        if k in s.rendering_kw.keys():
            provided_keys.append(k)
            s.rendering_kw.pop(k)
    if len(provided_keys) > 0:
        warnings.warn(
            "The following `stream_kw` keyword arguments are not "
            "used by Mayavi: " + str(provided_keys))

    kw = p.merge({}, a, s.rendering_kw)
    p._add_figure_to_kwargs(kw)
    colorbar_kw = kw.pop("colorbar_kw", dict())
    func = mlab.flow if s.is_streamlines else mlab.quiver3d
    obj = func(x, y, z, u, v, w, **kw)
    p._add_colorbar(s, obj, colorbar_kw, kw.get("color", None))
    return obj


def _update_vector3d_helper(renderer, data, handle):
    raise NotImplementedError
    

class Vector3DRenderer(Renderer):
    draw_update_map = {
        _draw_vector3d_helper: _update_vector3d_helper
    }
