from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer


def _draw_pole_zero_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    x, y = data
    color = next(p._cl) if s.line_color is None else s.line_color
    label = s.get_label(p._use_latex)
    is_pole = s.return_poles
    user_color = s.pole_color if is_pole else s.zero_color

    lkw = dict(
        label=label if s.show_in_legend else "_nolegend_",
        linestyle="none",
        color=user_color if user_color else color,
        marker="x" if is_pole else "o",
        markersize=s.pole_markersize if is_pole else s.zero_markersize
    )
    if not s.is_filled:
        lkw["markerfacecolor"] = (1, 1, 1)
    kw = p.merge({}, lkw, s.rendering_kw)
    l, = p._ax.plot(x, y, **kw)

    return [l]


def _update_pole_zero_helper(renderer, data, handles):
    p, s = renderer.plot, renderer.series
    x, y = data

    # TODO: Point2D are updated but not visible.
    handles[0].set_data(x, y)


class PoleZeroRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_pole_zero_helper: _update_pole_zero_helper
    }
