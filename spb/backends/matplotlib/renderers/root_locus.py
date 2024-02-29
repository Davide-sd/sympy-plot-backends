from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
import numpy as np


def _draw_root_locus_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    roots, gains = data

    color = next(p._cl) if s.line_color is None else s.line_color
    lkw = dict(
        color=color
    )
    if s.is_point:
        lkw["marker"] = "o"
        lkw["linestyle"] = "None"
        if not s.is_filled:
            lkw["markerfacecolor"] = (1, 1, 1)

    handles = []
    kw = p.merge({}, lkw, s.rendering_kw)
    for i, col in enumerate(roots.T):
        handles.append(
            p._ax.plot(p.np.real(col), p.np.imag(col), **kw)[0])

    if s.show_in_legend:
        proxy_artist = p.Line2D(
            [], [], color=color, label=s.get_label(p._use_latex))
        p._legend_handles.append(proxy_artist)

    zrk = p.merge({},
        dict(marker="o", color=color, fc=(0, 0, 0, 0), lw=1.25),
        s._zeros_rk)
    prk =  p.merge({},
        dict(marker="x", color=color),
        s._poles_rk)
    p._ax.scatter(s.zeros.real, s.zeros.imag, **zrk)
    p._ax.scatter(s.poles.real, s.poles.imag, **prk)

    return [handles]


def _update_root_locus_helper(renderer, data, handles):
    pass


class RootLocusRenderer(MatplotlibRenderer):
    _sal = True

    draw_update_map = {
        _draw_root_locus_helper: _update_root_locus_helper
    }

    def draw(self):
        data = self.series.get_data()

        self._set_axis_limits([self.series.xlim, self.series.ylim])

        for draw_method in self.draw_update_map.keys():
            self.handles.append(
                draw_method(self, data))
