from spb.backends.base_renderer import Renderer


def _draw_arrow_2d(renderer, data):
    p, s = renderer.plot, renderer.series
    x, y, u, v = data

    akw = dict(
        x=x,
        y=y,
        xref="x", yref="y",
        text="" if not s.show_in_legend else s.get_label(p._use_latex),
        showarrow=True,
        axref="x", ayref='y',
        ax=u,
        ay=v,
        arrowhead=3,
        arrowwidth=1.5,
        arrowcolor=next(p._cl),
        arrowside="start",
    )
    kw = p.merge({}, akw, s.rendering_kw)
    arrow = p.go.layout.Annotation(kw)
    p._fig.add_annotation(arrow)
    # plotly internally recreates the annotation, so `arrow` is not a
    # reference. We need to use indeces to update the proper annotation.
    handle = [p._n_annotations]
    p._n_annotations += 1

    return handle


def _update_arrow_2d(renderer, data, handle):
    d = {k: v for k, v in zip(["x", "y", "ax", "ay"], data)}
    arrow_index = handle[0]
    renderer.plot.fig.layout.annotations[arrow_index].update(d)


class Arrow2DRenderer(Renderer):
    draw_update_map = {
        _draw_arrow_2d: _update_arrow_2d
    }
