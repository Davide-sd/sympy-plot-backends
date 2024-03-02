from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.series import SGridLineSeries
import numpy as np

def _get_axis_limits(p, s):
    """There could be multiple sgrid series on the plot. For example, one for
    the actual sgrid, another one to indicate a particular damping ratio...
    In the latter case, it is very likely that xlim/ylim are None. Need to
    find the problem axis limits in order to place the annotation.
    Priority is given to the plot object axis limits. If not set, look for
    already existing sgrid series, select the one associated to root locus
    series and get their axis limits.

    Parameters
    ==========
    p : BaseBackend
    s : BaseSeries
    """
    xlim, ylim = s._get_axis_limits()

    if p.xlim and p.ylim:
        xlim = p.xlim if p.xlim else xlim
        ylim = p.ylim if p.ylim else ylim
    if (xlim is None) and (ylim is None):
        other_sgrid_series = [t for t in p.series
            if (isinstance(t, SGridLineSeries) and (t is not s)
            and len(t.associated_rl_series) > 0)
        ]
        if len(other_sgrid_series) > 0:
            xlim, ylim = other_sgrid_series[0]._get_axis_limits()
        else:
            xlim = [-11, 1]
            ylim = [-5, 5]
    xtext_pos_lim = xlim[0] + (xlim[1] - xlim[0]) * 0.0 if xlim else -1
    ytext_pos_lim = ylim[1] - (ylim[1] - ylim[0]) * 0.03 if ylim else 1
    return xlim, ylim, xtext_pos_lim, ytext_pos_lim


def _draw_sgrid_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    xi_dict, wn_dict, y_tp, x_ts = data

    lkw = p.grid_line_kw
    kw = p.merge({}, lkw, s.rendering_kw)

    xlim, ylim, xtext_pos_lim, ytext_pos_lim = _get_axis_limits(p, s)

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
    xi_dict, wn_dict, y_tp, x_ts = data
    xi_handles, wn_handles, tp_handles, ts_handles = handles

    xlim, ylim, xtext_pos_lim, ytext_pos_lim = _get_axis_limits(p, s)

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


class SGridLineRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_sgrid_helper: _update_sgrid_helper
    }
