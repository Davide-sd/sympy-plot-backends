from spb.backends.base_renderer import Renderer


def _draw_hvline_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    handle = []

    rkw = {"line_color": "#000000"}
    if s.label is not None:
        rkw["name"] = s.label
    rkw["dimension"] = "width" if s.is_horizontal else "height"
    rkw = p.merge(rkw, s.rendering_kw)

    handle = p.bokeh.models.Span(location=data, **rkw)
    p._fig.add_layout(handle)
    return handle


def _update_hvline_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    np = p.np
    handle.location = data


class HVLineRenderer(Renderer):
    draw_update_map = {
        _draw_hvline_helper: _update_hvline_helper
    }
