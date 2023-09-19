from spb.backends.base_renderer import Renderer


def _draw_arrow_2d(renderer, data):
    p, s = renderer.plot, renderer.series
    x, y, u, v = data

    color = next(p._cl)
    akw = dict(
        end=p.bokeh.models.VeeHead(
            size=10, fill_color=color, line_width=0
        ),
        line_color=color,
        x_start=x,
        y_start=y,
        x_end=u,
        y_end=v
    )
    kw = p.merge({}, akw, s.rendering_kw)
    arrow = p.bokeh.models.Arrow(**kw)
    p._fig.add_layout(arrow)

    return [arrow]


def _update_arrow_2d(renderer, data, handle):
    x, y, u, v = data
    arrow = handle[0]

    arrow.x_start = x
    arrow.y_start = y
    arrow.x_end = u
    arrow.y_end = v


class Arrow2DRenderer(Renderer):
    draw_update_map = {
        _draw_arrow_2d: _update_arrow_2d
    }
