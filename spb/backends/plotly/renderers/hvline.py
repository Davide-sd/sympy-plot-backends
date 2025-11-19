from spb.backends.base_renderer import Renderer


def _draw_hvline_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    rkw = {"line_color": "black", "line_width": 1}
    if s.label:
        rkw["name"] = s.get_label(p.use_latex)
    rkw = p.merge(rkw, s.rendering_kw)
    if s.is_horizontal:
        p._fig.add_hline(y=data, **rkw)
    else:
        p._fig.add_vline(x=data, **rkw)
    indices = [len(p._fig.layout.shapes) - 1]
    if s.label and s.show_in_legend:
        # add a fake renderer to show the label on the legend
        p._fig.add_trace(p.go.Scatter(
            x=[None], y=[None],
            mode="lines",
            showlegend=True,
            **rkw
        ))
        indices.append(len(p._fig.data) - 1)
    return indices


def _update_hvline_helper(renderer, data, indices):
    p, s = renderer.plot, renderer.series
    idx = indices[0]
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
