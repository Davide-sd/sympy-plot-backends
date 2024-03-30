from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.backends.matplotlib.renderers.nyquist import _draw_arrows_helper
from sympy.external import import_module


def _draw_nichols_helper(renderer, data):
    mpl = import_module("matplotlib")
    p, s = renderer.plot, renderer.series
    _, ol_phase, ol_mag, _, _ = data

    color = next(p._cl) if s.line_color is None else s.line_color
    lkw = dict(
        label=s.get_label(p._use_latex) if s.show_in_legend else "_nolegend_",
        color=color
    )
    if s.is_point:
        lkw["marker"] = "o"
        lkw["linestyle"] = "None"
        if not s.is_filled:
            lkw["markerfacecolor"] = (1, 1, 1)
    kw = p.merge({}, lkw, s.rendering_kw)
    line, = p._ax.plot(ol_phase, ol_mag, **kw)

    # Set the arrow style
    arrow_style = mpl.patches.ArrowStyle('simple', head_width=6, head_length=6)
    arrows = _draw_arrows_helper(
        p._ax, line,
        s.arrow_locs,
        arrowstyle=arrow_style,
        dir=1
    )

    return [line, arrows]


def _update_nichols_helper(renderer, data, handles):
    mpl = import_module("matplotlib")
    p, s = renderer.plot, renderer.series
    _, ol_phase, ol_mag, _, _ = data
    line, arrows = handles

    line.set_data(ol_phase, ol_mag)
    for a in arrows:
        a.remove()
    arrows.clear()
    # Set the arrow style
    arrow_style = mpl.patches.ArrowStyle('simple', head_width=6, head_length=6)
    new_arrows = _draw_arrows_helper(
        p._ax, line,
        s.arrow_locs,
        arrowstyle=arrow_style,
        dir=1
    )
    arrows.extend(new_arrows)


class NicholsLineRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_nichols_helper: _update_nichols_helper
    }
