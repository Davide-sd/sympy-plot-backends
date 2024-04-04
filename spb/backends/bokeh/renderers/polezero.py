from spb.backends.base_renderer import Renderer
from spb.series import SGridLineSeries, ZGridLineSeries
from sympy.external import import_module


def _get_source_tooltips(plot, x, y):
    """Compute the appropriate source/tooltips combination depending if the
    plot is showing an s-grid, z-grid or nothing.

    Parameters
    ----------
    plot : BB
    x : np.ndarray
        Real part of the roots
    y : np.ndarray
        Imaginary part of the roots

    Returns
    -------
    source : dict
    tooltips : list
    """
    np = import_module("numpy")
    is_sgrid = any(isinstance(t, SGridLineSeries) for t in plot.series)
    is_zgrid = any(isinstance(t, ZGridLineSeries) for t in plot.series)

    if is_sgrid:
        tp = np.nan * np.ones_like(y)
        tp[y != 0] = np.pi / np.abs(y[y != 0])
        ts = np.nan * np.ones_like(x)
        ts[x < 0] = 4 / np.abs(x[x < 0])
        wn = np.sqrt(x**2 + y**2)
        xi = 4 / ts / wn
        xi[x == 0] = 0
        source = {"x": x, "y": y, "tp": tp, "ts": ts, "wn": wn, "xi": xi}
        tooltips = [
            ("Real", "@x"), ("Imag", "@y"), ("wn", "@wn"), ("xi", "@xi"),
            ("Tp [s]", "@tp"), ("Ts [s]", "@ts")
        ]
    elif is_zgrid:
        mag = np.sqrt(x**2 + y**2)
        angles = np.arctan2(np.abs(y), x)

        ts_t = np.nan * np.ones_like(x)
        ts_t[mag < 1] = -4 / np.log(mag[mag < 1])
        tp_t = np.pi / angles

        # https://www.eng.mu.edu/nagurka/LuntzNagurkaKurfess_Explicit%20Parameter%20Dependency%20in%20Digital%20Control_1993.pdf
        wnt = np.nan * np.ones_like(x)
        idx = mag < 1
        wnt[idx] = np.sqrt(angles[idx]**2 + np.log(mag[idx])**2)
        # TODO: find a way to set NaN when the root is located in the negative
        # real axis, close to horizontal axis, where xi shouldn't exist.
        xi = -np.log(mag) / wnt

        # find the first z-grid series and extract the sampling time
        series = [s for s in plot.series if isinstance(s, ZGridLineSeries)][0]
        T = series.sampling_period
        if T:
            source = {
                "x": x, "y": y, "ts": ts_t * T, "tp": tp_t * T,
                "wn": wnt * T, "xi": xi
            }
            tooltips = [
                ("Real", "@x"), ("Imag", "@y"),
                ("wn [rad/s]", "@wn"), ("xi", "@xi"),
                ("Ts [s]", "@ts"), ("Tp [s]", "@tp")
            ]
        else:
            source = {
                "x": x, "y": y, "ts": ts_t, "tp": tp_t, "wn": wnt / np.pi, "xi": xi
            }
            tooltips = [
                ("Real", "@x"), ("Imag", "@y"),
                ("wn", "@wn π/T"), ("xi", "@xi"),
                ("Ts/T", "@ts"), ("Tp/T", "@tp")
            ]
    else:
        source = {"x": x, "y": y}
        tooltips = [("Real", "@x"), ("Imag", "@y")]

    return source, tooltips


def _draw_pole_zero_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    x, y = data
    color = next(p._cl) if s.line_color is None else s.line_color
    label = s.get_label(p._use_latex)
    is_pole = s.return_poles
    user_color = s.pole_color if is_pole else s.zero_color

    lkw = dict(
        legend_label=label,
        line_color=user_color if user_color else color,
        line_width=2,
        fill_color=user_color if user_color else color,
        marker="x" if is_pole else "o",
        size=s.pole_markersize if is_pole else s.zero_markersize
    )
    if not s.is_filled:
        lkw["fill_color"] = "white"
    kw = p.merge({}, lkw, s.rendering_kw)

    source, tooltips = _get_source_tooltips(p, x, y)
    l = p._fig.scatter("x", "y", source=source, **kw)

    p._fig.add_tools(p.bokeh.models.HoverTool(
        tooltips=tooltips,
        renderers=[l]
    ))

    return [l]


def _update_pole_zero_helper(renderer, data, handles):
    p, s = renderer.plot, renderer.series
    x, y = data
    source = {"x": x, "y": y}
    handles[0].data_source.data.update(source)


class PoleZeroRenderer(Renderer):
    draw_update_map = {
        _draw_pole_zero_helper: _update_pole_zero_helper
    }
