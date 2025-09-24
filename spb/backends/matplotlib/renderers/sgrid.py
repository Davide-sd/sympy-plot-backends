from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.series import SGridLineSeries, RootLocusSeries
from sympy.external import import_module


def _text_position_limits(r, p, s):
    """Computes the limits for text-annotations.

    Parameters
    ==========
    r : Renderer
    p : BaseBackend
    s : SGridLineSeries
    """
    xlim = p.xlim if p.xlim else s.xlim
    ylim = p.ylim if p.ylim else s.ylim

    if (xlim is None) and (ylim is None):
        xlim = r.default_xlim
        ylim = r.default_ylim
    xtext_pos_lim = xlim[0] + (xlim[1] - xlim[0]) * 0.0 if xlim else -1
    ytext_pos_lim = ylim[1] - (ylim[1] - ylim[0]) * 0.03 if ylim else 1
    return xlim, ylim, xtext_pos_lim, ytext_pos_lim


def _draw_sgrid_helper(renderer, data):
    p, s = renderer.plot, renderer.series

    if s.auto:
        return []

    np = p.np
    xi_dict, wn_dict, y_tp, x_ts = data

    lkw = {"color": '0.75', "linestyle": '--', "linewidth": 0.75, "zorder": 0}
    kw = p.merge({}, lkw, s.rendering_kw)

    xlim, ylim, xtext_pos_lim, ytext_pos_lim = _text_position_limits(
        renderer, p, s)

    if s.show_control_axis:
        p._ax.axhline(0, **kw)
        p._ax.axvline(0, **kw)

    # damping ratio lines
    xi_handles = []
    for (x, a, yp), v in xi_dict.items():
        h1, = p._ax.plot(v["x"], v["y"], **kw)
        h2, h3 = None, None
        if not np.isclose(x, 1):
            h2, = p._ax.plot(v["x"], -v["y"], **kw)
        if yp < 0 and not np.isclose(x, 1):
            xtext_pos = 1/yp * ylim[1]
            ytext_pos = yp * xtext_pos_lim
            if np.abs(xtext_pos) > np.abs(xtext_pos_lim):
                xtext_pos = xtext_pos_lim
            else:
                ytext_pos = ytext_pos_lim
            h3 = p._ax.annotate(v["label"], textcoords='data',
                xy=[xtext_pos, ytext_pos], fontsize=8)
        xi_handles.append([h1, h2, h3])

    # natural frequency lines
    wn_handles = []
    for k, v in wn_dict.items():
        h1, = p._ax.plot(v["x"], v["y"], **kw)
        h2 = p._ax.annotate(v["label"], textcoords='data',
            xy=[v["lx"], v["ly"]], fontsize=8, horizontalalignment="center")
        wn_handles.append([h1, h2])

    # peak time lines
    tp_handles = []
    for y in y_tp:
        h = p._ax.axhline(y, **kw)
        tp_handles.append(h)

    # settling time lines
    ts_handles = []
    for x in x_ts:
        h = p._ax.axvline(x, **kw)
        ts_handles.append(h)

    return [xi_handles, wn_handles, tp_handles, ts_handles]


def _update_sgrid_helper(renderer, data, handles):
    p, s = renderer.plot, renderer.series

    if s.auto:
        return

    np = p.np
    xi_dict, wn_dict, y_tp, x_ts = data
    xi_handles, wn_handles, tp_handles, ts_handles = handles

    xlim, ylim, xtext_pos_lim, ytext_pos_lim = _text_position_limits(
        renderer, p, s)

    # damping ratio lines
    for (k, v), handles in zip(xi_dict.items(), xi_handles):
        h1, h2, h3 = handles
        h1.set_data(v["x"], v["y"])
        if h2:
            h2.set_data(v["x"], -v["y"])
        if h3:
            x, a, yp = k
            if yp < 0 and not np.isclose(x, 1):
                xtext_pos = 1/yp * ylim[1]
                ytext_pos = yp * xtext_pos_lim
                if np.abs(xtext_pos) > np.abs(xtext_pos_lim):
                    xtext_pos = xtext_pos_lim
                else:
                    ytext_pos = ytext_pos_lim
                h3.set_position([xtext_pos, ytext_pos])
                h3.set_text(v["label"])

    # natural frequency lines
    y_offset = 0 if ylim is None else 0.015 * abs(ylim[1] - ylim[0])
    for v, handles in zip(wn_dict.values(), wn_handles):
        h1, h2 = handles
        h1.set_data(v["x"], v["y"])
        h2.set_position([v["lx"], y_offset])
        h2.set_text(v["label"])

    # peak time lines
    for y, h in zip(y_tp, tp_handles):
        h.set_ydata([y, y])

    # settling time lines
    for x, h in zip(x_ts, ts_handles):
        h.set_xdata([x, x])


def _find_data_axis_limits(ax):
    """Loop over the lines in order to find their minimum and
    maximum coordinates.
    """
    np = import_module("numpy")
    xlims, ylims = [], []
    for line in ax.lines:
        x, y = line.get_data()
        xlims.append([np.nanmin(x), np.nanmax(x)])
        ylims.append([np.nanmin(y), np.nanmax(y)])
    for line in ax.collections:
        seg = line._get_segments()
        xmin, ymin = np.min(np.vstack(seg), axis=0)
        xmax, ymax = np.max(np.vstack(seg), axis=0)
        xlims.append([xmin, xmax])
        ylims.append([ymin, ymax])
    return xlims, ylims


def _modify_axis_limits(lim, margin_factor_lower, margin_factor_upper):
    """Modify the provided axis limits and add appropriate margins.
    """
    np = import_module("numpy")
    min_t, max_t = lim
    # this offset allows to have a little bit of empty space on the
    # LHP of root locus plot
    offset = 0.25
    min_t = min_t - offset if np.isclose(min_t, 0) else min_t
    max_t = max_t + offset if np.isclose(max_t, 0) else max_t
    # provide a little bit of margin
    delta = abs(max_t - min_t)
    lim = [
        min_t - delta * margin_factor_lower,
        max_t + delta * margin_factor_upper
    ]
    if np.isclose(*lim):
        # prevent axis limits to be the same
        lim[0] -= 1
        lim[1] += 1
    return lim


class SGridLineRenderer(MatplotlibRenderer):
    default_xlim = [-11, 1]
    default_ylim = [-5, 5]
    _sal = True
    draw_update_map = {
        _draw_sgrid_helper: _update_sgrid_helper
    }

    def _set_axis_limits_before_compute_data(self):
        np = import_module("numpy")
        # loop over the data already present on the plot and find
        # appropriate axis limits
        root_locus_series = [
            s for s in self.plot.series if isinstance(s, RootLocusSeries)]
        if len(root_locus_series) > 0:
            xlims, ylims = [], []
            for s in root_locus_series:
                xlims.append(s.xlim)
                ylims.append(s.ylim)
        else:
            xlims, ylims = _find_data_axis_limits(self.plot.ax)

        if len(xlims) > 0:
            xlims = np.array(xlims)
            ylims = np.array(ylims)
            xlim = (np.nanmin(xlims[:, 0]), np.nanmax(xlims[:, 1]))
            ylim = (np.nanmin(ylims[:, 0]), np.nanmax(ylims[:, 1]))
        else:
            xlim = self.plot.xlim if self.plot.xlim else self.default_xlim
            ylim = self.plot.ylim if self.plot.ylim else self.default_ylim

        xlim = _modify_axis_limits(xlim, 0.15, 0.05)
        ylim = _modify_axis_limits(ylim, 0.05, 0.05)
        self.series.set_axis_limits(xlim, ylim)

    def draw(self):
        if (self.series.xlim is None) and (self.series.ylim is None):
            # if user didn't set xlim/ylim in sgrid() function call
            self._set_axis_limits_before_compute_data()

        data = self.series.get_data()
        self._set_axis_limits([self.series.xlim, self.series.ylim])
        for draw_method in self.draw_update_map.keys():
            self.handles.append(
                draw_method(self, data))

    def update(self, params):
        self.series.params = params
        self._set_axis_limits_before_compute_data()
        data = self.series.get_data()
        self._set_axis_limits([self.series.xlim, self.series.ylim])
        for update_method, handle in zip(
            self.draw_update_map.values(), self.handles
        ):
            update_method(self, data, handle)
