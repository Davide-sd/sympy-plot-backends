from spb.backends.base_renderer import Renderer
from spb.backends.utils import _get_cmin_cmax
from spb.utils import get_vertices_indices
from spb.series import PlaneSeries


def _draw_surface_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    Triangulation = p.matplotlib.tri.Triangulation

    if s.is_parametric:
        x, y, z, u, v = data
        attribute = s.eval_color_func(x, y, z, u, v).astype(np.float32).flatten()
        vertices, indices = get_vertices_indices(x, y, z)
        vertices = vertices.astype(np.float32)
    else:
        x, y, z = data
        attribute = s.eval_color_func(x, y, z).astype(np.float32).flatten()
        if isinstance(s, PlaneSeries):
            # avoid triangulation errors when plotting vertical
            # planes
            vertices, indices = get_vertices_indices(x, y, z)
            vertices = vertices.astype(np.float32)
            indices = np.asarray(indices, dtype=np.uint32)
        else:
            x = x.flatten()
            y = y.flatten()
            z = z.flatten()
            vertices = np.vstack([x, y, z]).T.astype(np.float32)
            indices = Triangulation(x, y).triangles.astype(np.uint32)

    a = dict(
        name=s.get_label(p.use_latex, "%s") if p.show_label else None,
        side="double",
        flat_shading=False,
        wireframe=False,
        color=(
            p._convert_to_int(next(p._cl)) if s.surface_color is None
            else s.surface_color
        ),
        colorLegend=p.legend or s.use_cm,
    )
    if s.use_cm:
        cmin, cmax = _get_cmin_cmax(attribute, p, s)
        a["color_map"] = next(p._cm)
        a["attribute"] = attribute
        # NOTE: color_range must contains elements of type float.
        # If np.float32 is provided, mgspack will fail to serialize
        # it, hence no html export, hence no screenshots on
        # documentation.
        a["color_range"] = [float(cmin), float(cmax)]

    kw = p.merge({}, a, s.rendering_kw)
    surf = p.k3d.mesh(vertices, indices, **kw)
    p._fig += surf

    return surf


def _update_surface_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    np = p.np
    update_discr = (
        (s.n != renderer.previous_n)
        or (s.only_integers != renderer.previous_only_integers)
    )

    if s.is_parametric:
        x, y, z, u, v = data
        attribute = s.eval_color_func(x, y, z, u, v).astype(np.float32).flatten()
        if update_discr:
            vertices, indices = get_vertices_indices(x, y, z)
            vertices = vertices.astype(np.float32)
        else:
            x, y, z, u, v = [
                t.flatten().astype(np.float32) for t in [x, y, z, u, v]
            ]
            vertices = np.vstack([x, y, z]).T.astype(np.float32)
    else:
        x, y, z = data
        x, y, z = [t.flatten().astype(np.float32) for t in [x, y, z]]
        attribute = s.eval_color_func(x, y, z).astype(np.float32).flatten()
        if update_discr:
            Triangulation = p.matplotlib.tri.Triangulation
            vertices = np.vstack([x, y, z]).T.astype(np.float32)
            indices = Triangulation(x, y).triangles.astype(np.uint32)
        else:
            vertices = np.vstack([x, y, z]).astype(np.float32).T

    handle.vertices = vertices
    if update_discr:
        handle.indices = indices
        renderer.previous_n = s.n
        renderer.previous_only_integers = s.only_integers

    if s.use_cm:
        cmin, cmax = _get_cmin_cmax(attribute, p, s)
        handle.attribute = attribute
        handle.color_range = [float(cmin), float(cmax)]
    p._high_aspect_ratio(x, y, z)


class SurfaceRenderer(Renderer):
    draw_update_map = {
        _draw_surface_helper: _update_surface_helper
    }

    def __init__(self, plot, s):
        super().__init__(plot, s)
        # previous numbers of discretization points
        self.previous_n = s.n
        self.previous_only_integers = s.only_integers
