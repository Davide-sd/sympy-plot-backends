from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.series import SGridLineSeries
import numpy as np


def _draw_sgrid_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    xi, wn = data
    angles = np.pi - np.arccos(xi)
    y_over_x = np.tan(angles)
    radius = max(1000, max(wn)) if len(wn) > 0 else 1000

    lkw = p.grid_line_kw
    kw = p.merge({}, lkw, s.rendering_kw)

    xlim, ylim = s._get_axis_limits()
    # There could be multiple sgrid series on the plot. For example, one for
    # the actual sgrid, another one to indicate a particular damping ratio...
    # In the latter case, it is very likely that xlim/ylim are None. Need to
    # find the problem axis limits in order to place the annotation.
    # Priority is given to the plot object axis limits. If not set, look for
    # already existing sgrid series, select the one associated to root locus
    # series and get their axis limits.
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

    if s.show_control_axis:
        p._ax.axhline(0, **kw)
        p._ax.axvline(0, **kw)

    for x, a, yp in zip(xi, angles, y_over_x):
        xcoords = np.array([0, radius * np.cos(a)])
        ycoords = np.array([0, radius * np.sin(a)])
        p._ax.plot(xcoords, ycoords, **kw)
        if not np.isclose(x, 1):
            p._ax.plot(xcoords, -ycoords, **kw)
        if yp < 0 and (ylim is not None) and not np.isclose(x, 1):
            xtext_pos = 1/yp * ylim[1]
            ytext_pos = yp * xtext_pos_lim
            if np.abs(xtext_pos) > np.abs(xtext_pos_lim):
                xtext_pos = xtext_pos_lim
            else:
                ytext_pos = ytext_pos_lim
            an = "%.2f" % x
            p._ax.annotate(an, textcoords='data', xy=[xtext_pos, ytext_pos],
                        fontsize=8)

    t = np.linspace(np.pi/2, 3*np.pi/2, 100)
    y_offset = 0 if s.ylim is None else 0.015 * abs(s.ylim[1] - s.ylim[0])
    for w in wn:
        p._ax.plot(w * np.cos(t), w * np.sin(t), **kw)
        an = "%.2f" % w
        p._ax.annotate(an, textcoords='data', xy=[-w, y_offset], fontsize=8,
            horizontalalignment="center")

    return []


def _update_sgrid_helper(renderer, data, handles):
    p, s = renderer.plot, renderer.series


class SGridLineRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_sgrid_helper: _update_sgrid_helper
    }
