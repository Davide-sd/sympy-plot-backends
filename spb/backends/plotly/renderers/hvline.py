from spb.backends.base_renderer import Renderer


def _draw_hvline_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    rkw = {"line_color": "black", "line_width": 1}
    if s.label is not None:
        rkw["name"] = s.label
    rkw = p.merge(rkw, s.rendering_kw)
    if s.is_horizontal:
        p._fig.add_hline(y=data, **rkw)
    else:
        p._fig.add_vline(x=data, **rkw)
    return len(p._fig.layout.shapes) - 1


def _update_hvline_helper(renderer, data, idx):
    p, s = renderer.plot, renderer.series
    if s.is_horizontal:
        p._fig.layout.shapes[idx]["y0"] = data
        p._fig.layout.shapes[idx]["y1"] = data
    else:
        p._fig.layout.shapes[idx]["x0"] = data
        p._fig.layout.shapes[idx]["x1"] = data


class HVLineRenderer(Renderer):
    draw_update_map = {
        _draw_hvline_helper: _update_hvline_helper
    }
