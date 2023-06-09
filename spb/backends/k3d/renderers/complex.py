from spb.backends.base_renderer import Renderer
from spb.backends.k3d.renderers.surface import _draw_surface_helper, _update_surface_helper
import warnings


def _draw_domain_coloring_helper(renderer, data):
    raise NotImplementedError


def _update_domain_coloring_helper(renderer, data, handle):
    raise NotImplementedError


def _draw_analytic_landscape_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    Triangulation = p.matplotlib.tri.Triangulation

    x, y, mag, arg, colors, colorscale = data
    x, y, z = [t.flatten() for t in [x, y, mag]]
    vertices = np.vstack([x, y, z]).T.astype(np.float32)
    indices = Triangulation(x, y).triangles.astype(np.uint32)
    p._high_aspect_ratio(x, y, z)

    a = dict(
        name=s.get_label(p._use_latex, "%s") if p._show_label else None,
        side="double",
        flat_shading=False,
        wireframe=False,
        color=p._convert_to_int(next(p._cl)),
        colorLegend=p.legend or s.use_cm,
    )
    if s.use_cm:
        colors = colors.reshape((-1, 3))
        a["colors"] = [p._rgb_to_int(c) for c in colors]

        if colorscale is None:
            # grayscale colormap
            r = [0, 0, 0, 0, 1, 1, 1, 1]
        else:
            r = []
            loc = np.linspace(0, 1, colorscale.shape[0])
            colorscale = colorscale / 255
            for l, c in zip(loc, colorscale):
                r.append(l)
                r += list(c)

        a["color_map"] = r
        a["color_range"] = [-np.pi, np.pi]
    kw = p.merge({}, a, s.rendering_kw)
    surf = p.k3d.mesh(vertices, indices, **kw)
    p._fig += surf
    
    return surf


def _update_analytic_landscape_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    np = p.np

    x, y, mag, _, colors, _ = data
    x, y, z = [t.flatten().astype(np.float32) for t in [x, y, mag]]
    vertices = np.vstack([x, y, z]).astype(np.float32)
    handle.vertices = vertices.T
    if s.use_cm:
        colors = colors.reshape((-1, 3))
        colors = [p._rgb_to_int(c) for c in colors]
        handle.colors = colors
    p._high_aspect_ratio(x, y, z)


class ComplexRenderer(Renderer):
    draw_update_map = {
        _draw_analytic_landscape_helper: _update_analytic_landscape_helper
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.series.is_domain_coloring:
            self.draw_update_map = {
                _draw_surface_helper: _update_surface_helper
            }
        else:
            if not self.series.is_3Dsurface:
                self.draw_update_map = {
                    _draw_domain_coloring_helper: _update_domain_coloring_helper
                }
