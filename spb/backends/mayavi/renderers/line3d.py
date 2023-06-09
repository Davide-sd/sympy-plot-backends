from spb.backends.base_renderer import Renderer


def _draw_line3d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    mlab = p.mlab
    
    x, y, z, u = data
    a = dict(
        color=(None if s.use_cm else (
            (next(p._cl) if s.line_color is None
            else s.line_color)) if s.show_in_legend
            else p.wireframe_color),
        colormap=next(p._cm),
        extent=p._get_extent(x, y, z)
    )
    if not s.is_point:
        a["tube_radius"] = 0.05
    kw = p.merge({}, a, s.rendering_kw)
    p._add_figure_to_kwargs(kw)
    colorbar_kw = kw.pop("colorbar_kw", dict())

    if s.is_point:
        if s.use_cm:
            obj = mlab.points3d(x, y, z, u, **kw)
        else:
            obj = mlab.points3d(x, y, z, **kw)
    else:
        obj = mlab.plot3d(x, y, z, u, **kw)

    p._add_colorbar(s, obj, colorbar_kw, kw.get("color", None))
    return obj


def _update_line3d_helper(renderer, data, handle):
    raise NotImplementedError
    

class Line3DRenderer(Renderer):
    draw_update_map = {
        _draw_line3d_helper: _update_line3d_helper
    }
