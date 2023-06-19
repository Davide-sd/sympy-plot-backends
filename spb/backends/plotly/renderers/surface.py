from spb.backends.base_renderer import Renderer


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
    skw = dict(
        name=s.get_label(p._use_latex),
        showscale=s.use_cm and s.colorbar,
        showlegend=(not s.use_cm) and s.show_in_legend,
        colorbar=p._create_colorbar(
            s.get_label(p._use_latex), p._scale_down_colorbar),
        colorscale=colormap if s.use_cm else colorscale,
        surfacecolor=surfacecolor,
        cmin=surfacecolor.min(),
        cmax=surfacecolor.max()
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
    _min, _max = surfacecolor.min(), surfacecolor.max()
    handle["z"] = z
    handle["surfacecolor"] = surfacecolor
    handle["cmin"] = _min
    handle["cmax"] = _max


class SurfaceRenderer(Renderer):
    draw_update_map = {
        _draw_surface_helper: _update_surface_helper
    }
