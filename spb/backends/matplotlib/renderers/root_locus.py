from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from sympy.external import import_module


def _get_x_y_coords(roots):
    """Create one single line, where branches are separated by NaNs.
    this makes it easier to deal with widget's update, because with
    interactive widgets plot the number of roots/poles can vary.
    """
    np = import_module("numpy")
    x, y = [], []
    for col in roots.T:
        x.append(np.real(col))
        x.append(np.array([np.nan]))
        y.append(np.imag(col))
        y.append(np.array([np.nan]))
    x = np.concatenate(x)
    y = np.concatenate(y)
    return x, y


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
    x, y = _get_x_y_coords(roots)
    handles.append(p._ax.plot(x, y, **kw)[0])

    if s.show_in_legend:
        proxy_artist = p.Line2D(
            [], [], color=color, label=s.get_label(p.use_latex))
        p._legend_handles.append(proxy_artist)

    zrk = p.merge(
        {},
        dict(marker="o", color=color, fc=(0, 0, 0, 0), lw=1.25),
        s._zeros_rk)
    prk = p.merge(
        {},
        dict(marker="x", color=color),
        s._poles_rk)
    hz = p._ax.scatter(s.zeros.real, s.zeros.imag, **zrk)
    hp = p._ax.scatter(s.poles.real, s.poles.imag, **prk)

    return handles + [hz, hp]


def _update_root_locus_helper(renderer, data, handles):
    p, s = renderer.plot, renderer.series
    np = p.np
    roots, gains = data

    x, y = _get_x_y_coords(roots)
    handles[0].set_data(x, y)

    handles[-2].set_offsets(np.c_[s.zeros.real, s.zeros.imag])
    handles[-1].set_offsets(np.c_[s.poles.real, s.poles.imag])


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

    def update(self, params):
        self.series.params = params
        data = self.series.get_data()
        self._set_axis_limits([self.series.xlim, self.series.ylim])
        for update_method, handle in zip(
            self.draw_update_map.values(), self.handles
        ):
            update_method(self, data, handle)
