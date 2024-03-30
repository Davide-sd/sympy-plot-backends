from spb.backends.base_renderer import Renderer


def fdac(x):
    """Return the first digit after comma of a number.
    """
    return abs(int((x - int(x)) * 10))


def _draw_mcircles_helper(renderer, data):
    p, s = renderer.plot, renderer.series

    rkw = p.mcircles_line_kw
    rkw = p.merge({}, rkw, s.rendering_kw)

    # find the best y-location for 0-dB label
    max_y = -10
    for _, _, y in data:
        my = max(y)
        if my > max_y:
            max_y = my

    handles = []
    for (mdb, x, y) in data:
        label = str(int(mdb)) if fdac(mdb) == 0 else str(mdb)
        if len(x) > 1:
            source = {"x": x, "y": y}
            h1 = p._fig.line(x="x", y="y", source=source, **rkw)
            xtext_pos = (x.max() + x.min()) / 2
            ytext_pos = y.max()
            h2 = p.bokeh.models.Label(
                x=xtext_pos, y=ytext_pos, text=label + " dB",
                text_font_size='12px', text_baseline="middle",
                text_align="center", text_color='#000000'
            )
            p._fig.add_layout(h2)
        else:
            h1 = p.bokeh.models.Span(location=x[0], dimension="height", **rkw)
            p._fig.add_layout(h1)
            h2 = p.bokeh.models.Label(
                x=x[0], y=max_y, text=label + " dB",
                text_font_size='12px', text_baseline="middle",
                text_align="center", text_color='#000000'
            )
            p._fig.add_layout(h2)
        handles.append([h1, h2])

    if s.show_minus_one:
        # Mark the -1 point
        minus_one = p._fig.scatter(
            [-1], [0], marker="+", line_color="red", size=8)

    return handles


def _update_mcircles_helper(renderer, data, handles):
    # bokeh makes it difficult to implement it, because the transition from
    # negative to positive magnitude [dB] corresponds to a vertical line.
    # AFAIK, it's not possible to remove a renderer from a bokeh figure...
    raise NotImplementedError


class MCirclesRenderer(Renderer):
    draw_update_map = {
        _draw_mcircles_helper: _update_mcircles_helper
    }
