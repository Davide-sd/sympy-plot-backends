from spb.backends.base_renderer import Renderer


def _draw_implicit3d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    mlab = p.mlab
    
    x, y, z, r = data
    a = dict(
        color=None if s.use_cm else (
            next(p._cl) if s.surface_color is None
            else s.surface_color),
        colormap=next(p._cm),
        contours=[0],
        # NOTE: can't use extent here, as the actual surface
        # dimension is computed by Mayavi and we can't access it
        # at this time
        # extent=self._get_extent(x, y, z)
    )
    kw = p.merge({}, a, s.rendering_kw)
    p._add_figure_to_kwargs(kw)
    colorbar_kw = kw.pop("colorbar_kw", dict())
    obj = mlab.contour3d(x, y, z, r, **kw)
    p._add_colorbar(s, obj, colorbar_kw, kw.get("color", None))
    return obj


def _update_implicit3d_helper(renderer, data, handle):
    raise NotImplementedError


class Implicit3DRenderer(Renderer):
    draw_update_map = {
        _draw_implicit3d_helper: _update_implicit3d_helper
    }
