from spb.backends.base_renderer import Renderer


def _draw_hvline_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    handle = []

    rkw = {"line_color": "#000000"}
    if s.label is not None:
        rkw["name"] = s.label
    rkw["dimension"] = "width" if s.is_horizontal else "height"
    rkw = p.merge(rkw, s.rendering_kw)

    handles = [
        p.bokeh.models.Span(location=data, **rkw)
    ]
    p._fig.add_layout(handles[0])

    if s.label and s.show_in_legend:
        rkw.pop("name")
        rkw.pop("dimension")
        # add a fake renderer to show the label on the legend
        fake_line = p._fig.line(
            [0], [0], legend_label=s.get_label(p.use_latex), **rkw)
        handles.append(fake_line)

        fake_line.js_on_change(
            "visible",
            p.bokeh.models.CustomJS(
                args=dict(s=handles[0], r=fake_line),
                code="s.visible = r.visible;"
            )
        )

    return handles


def _update_hvline_helper(renderer, data, handles):
    handles[0].location = data


class HVLineRenderer(Renderer):
    draw_update_map = {
        _draw_hvline_helper: _update_hvline_helper
    }
