from spb.backends.base_renderer import Renderer


def _draw_arrow_2d(renderer, data):
    p, s = renderer.plot, renderer.series
    x1, y1, x2, y2 = data

    color = next(p._cl)

    # need a scatter in order to show tooltips
    source = {
        "xs": [x1, x2],
        "ys": [y1, y2]
    }
    tooltips=[("x", "@xs"), ("y", "@ys")]
    scatter = p._fig.scatter("xs", "ys", source=source, color=color, size=1)
    p._fig.add_tools(p.bokeh.models.HoverTool(
        tooltips=tooltips,
        renderers=[scatter]
    ))

    akw = dict(
        end=p.bokeh.models.VeeHead(
            size=10, fill_color=color, line_width=0
        ),
        line_color=color,
        x_start=x1,
        y_start=y1,
        x_end=x2,
        y_end=y2
    )
    kw = p.merge({}, akw, s.rendering_kw)
    arrow = p.bokeh.models.Arrow(**kw)
    p._fig.add_layout(arrow)

    return [arrow, scatter]


def _update_arrow_2d(renderer, data, handle):
    x1, y1, x2, y2 = data
    arrow, line = handle

    arrow.x_start = x1
    arrow.y_start = y1
    arrow.x_end = x2
    arrow.y_end = y2

    source = {
        "xs": [x1, x2],
        "ys": [y1, y2]
    }
    line.data_source.data.update(source)


class Arrow2DRenderer(Renderer):
    draw_update_map = {
        _draw_arrow_2d: _update_arrow_2d
    }
