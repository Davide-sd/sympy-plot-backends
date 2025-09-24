from spb.backends.base_renderer import Renderer
import warnings


def _draw_contour_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    handle = []

    if s.is_polar:
        raise NotImplementedError()
    x, y, z = data

    # NOTE: at the time of writing this, Bokeh doesn't support
    # levels=int number.
    if "levels" in s.rendering_kw.keys():
        levels = s.rendering_kw["levels"]
    else:
        levels = p.np.linspace(z.min(), z.max(), 10)

    if (not s.is_filled) and s.show_clabels:
        warnings.warn("BokehBackend doesn't currently support contour labels.")

    cm = next(p._cm)
    ckw = dict(fill_color=cm, line_color=cm, levels=levels)
    kw = p.merge({}, ckw, s.rendering_kw)

    h = p._fig.contour(x, y, z, **kw)
    handle.extend([h, levels])
    p._fig.add_tools(p.bokeh.models.HoverTool(
        tooltips=[("x", "@x"), ("y", "@y"), ("z", "@z")],
        renderers=[handle[0]]
    ))

    if s.colorbar:
        colorbar = h.construct_color_bar(title=s.get_label(p.use_latex))
        p._fig.add_layout(colorbar, "right")
        handle.append(colorbar)

    return handle


def _update_contour_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    x, y, z = data
    countour_handle, levels, cb_handle = handle
    levels = p.np.linspace(z.min(), z.max(), len(levels))

    contour_data = p.bokeh.plotting.contour.contour_data(x, y, z, levels)
    handle[0].set_data(contour_data)
    if s.colorbar:
        # NOTE: as of Bokeh 3.4.1, there is a bug that prevents ticks to
        # be updated.
        handle[2].update(levels=list(levels))


class ContourRenderer(Renderer):
    draw_update_map = {
        _draw_contour_helper: _update_contour_helper
    }
