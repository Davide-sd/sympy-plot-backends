from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.backends.matplotlib.renderers.vector2d import (
    _draw_vector2d_helper, _update_vector2d_helper
)

class Arrow2DRendererQuivers(MatplotlibRenderer):
    draw_update_map = {
        _draw_vector2d_helper: _update_vector2d_helper
    }


def _draw_arrow_2d(renderer, data):
    p, s = renderer.plot, renderer.series
    xx, yy, uu, vv = data
    mpatches = p.matplotlib.patches

    arrowstyle = mpatches.ArrowStyle(
        "fancy", head_width=2, head_length=3)
    pkw = dict(
        mutation_scale=2,
        arrowstyle=arrowstyle,
        shrinkA=0, shrinkB=0,
        color=next(p._cl)
    )
    kw = p.merge({}, pkw, s.rendering_kw)
    arrow = mpatches.FancyArrowPatch((xx, yy), (uu, vv), **kw)
    p._ax.add_patch(arrow)

    if s.show_in_legend:
        proxy_artist = p.Line2D(
            [], [],
            color=kw["color"], label=s.get_label(p._use_latex)
        )
        p._legend_handles.append(proxy_artist)
    return [arrow]


def _update_arrow2d(renderer, data, handle):
    xx, yy, uu, vv = data
    arrow = handle[0]
    arrow.set_positions((xx, yy), (uu, vv))


class Arrow2DRendererFancyArrowPatch(MatplotlibRenderer):
    draw_update_map = {
        _draw_arrow_2d: _update_arrow2d
    }
