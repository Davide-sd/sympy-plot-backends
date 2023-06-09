from spb.backends.base_renderer import Renderer
from spb.utils import get_vertices_indices
from spb.series import PlaneSeries

def _draw_surface_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    Triangulation = p.matplotlib.tri.Triangulation

    if s.is_parametric:
        x, y, z, u, v = data
        vertices, indices = get_vertices_indices(x, y, z)
        vertices = vertices.astype(np.float32)
        attribute = s.eval_color_func(vertices[:, 0], vertices[:, 1], vertices[:, 2], u.flatten().astype(np.float32), v.flatten().astype(np.float32))
    else:
        x, y, z = data
        if isinstance(s, PlaneSeries):
            # avoid triangulation errors when plotting vertical
            # planes
            vertices, indices = get_vertices_indices(x, y, z)
        else:
            x = x.flatten()
            y = y.flatten()
            z = z.flatten()
            vertices = np.vstack([x, y, z]).T.astype(np.float32)
            indices = Triangulation(x, y).triangles.astype(np.uint32)
        attribute = s.eval_color_func(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    a = dict(
        name=s.get_label(p._use_latex, "%s") if p._show_label else None,
        side="double",
        flat_shading=False,
        wireframe=False,
        color=p._convert_to_int(next(p._cl)) if s.surface_color is None else s.surface_color,
        colorLegend=p.legend or s.use_cm,
    )
    if s.use_cm:
        a["color_map"] = next(p._cm)
        a["attribute"] = attribute
        # NOTE: color_range must contains elements of type float.
        # If np.float32 is provided, mgspack will fail to serialize
        # it, hence no html export, hence no screenshots on
        # documentation.
        a["color_range"] = [float(attribute.min()), float(attribute.max())]

    kw = p.merge({}, a, s.rendering_kw)
    surf = p.k3d.mesh(vertices, indices, **kw)
    p._fig += surf

    return surf


def _update_surface_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    np = p.np

    if s.is_parametric:
        x, y, z, u, v = data
        x, y, z, u, v = [t.flatten().astype(np.float32) for t in [x, y, z, u, v]]
        attribute = s.eval_color_func(x, y, z, u, v)
    else:
        x, y, z = data
        x, y, z = [t.flatten().astype(np.float32) for t in [x, y, z]]
        attribute = s.eval_color_func(x, y, z)

    vertices = np.vstack([x, y, z]).astype(np.float32)
    handle.vertices = vertices.T
    if s.use_cm:
        handle.attribute = attribute
        handle.color_range = [float(attribute.min()), float(attribute.max())]
    p._high_aspect_ratio(x, y, z)
    

class SurfaceRenderer(Renderer):
    draw_update_map = {
        _draw_surface_helper: _update_surface_helper
    }
