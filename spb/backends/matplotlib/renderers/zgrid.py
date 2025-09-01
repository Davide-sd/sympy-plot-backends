from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer


def _draw_zgrid_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    xi_dict, wn_dict, tp_dict, ts_dict = data

    lkw = {"color": '0.75', "linestyle": '--', "linewidth": 0.75}
    kw = p.merge({}, lkw, s.rendering_kw)

    # damping ratio lines
    xi_handles = []
    for k, v in xi_dict.items():
        h1, = p._ax.plot(v["x1"], v["y1"], **kw)
        h2, = p._ax.plot(v["x2"], v["y2"], **kw)
        h3 = p._ax.annotate(
            v["label"], xy=(v["lx"], v["ly"]),
            xytext=(v["lx"], v["ly"]), size=7,
            horizontalalignment="center",
            verticalalignment="center")
        xi_handles.append([h1, h2, h3])

    # natural frequency lines
    wn_handles = []
    for k, v in wn_dict.items():
        h1, = p._ax.plot(v["x"], v["y"], **kw)
        h2 = p._ax.annotate(
            v["label"](p.use_latex), xy=(v["lx"], v["ly"]),
            xytext=(v["lx"], v["ly"]), size=9,
            horizontalalignment="center",
            verticalalignment="center")
        wn_handles.append([h1, h2])

    # peak time lines
    tp_handles = []
    for k, v in tp_dict.items():
        h1, = p._ax.plot(v["x"], v["y"], **kw)
        h2 = p._ax.annotate(
            v["label"], xy=(v["lx"], v["ly"]),
            xytext=(v["lx"], v["ly"]), size=7,
            horizontalalignment="center",
            verticalalignment="center")
        tp_handles.append([h1, h2])

    # settling time lines
    ts_handles = []
    for k, v in ts_dict.items():
        h1, = p._ax.plot(v["x"], v["y"], **kw)
        h2 = p._ax.annotate(
            v["label"], xy=(v["lx"], v["ly"]),
            xytext=(v["lx"], v["ly"]), size=7,
            horizontalalignment="center",
            verticalalignment="center")
        ts_handles.append([h1, h2])

    if s.show_control_axis:
        p._ax.axhline(0, **kw)
        p._ax.axvline(0, **kw)

    return [xi_handles, wn_handles, tp_handles, ts_handles]


def _update_zgrid_helper(renderer, data, handles):
    p = renderer.plot
    xi_handles, wn_handles, tp_handles, ts_handles = handles
    xi_dict, wn_dict, tp_dict, ts_dict = data

    for v, handles in zip(xi_dict.values(), xi_handles):
        h1, h2, h3 = handles
        h1.set_data(v["x1"], v["y1"])
        h2.set_data(v["x2"], v["y2"])
        h3.set_position([v["lx"], v["ly"]])
        h3.set_text(v["label"])

    for v, handles in zip(wn_dict.values(), wn_handles):
        h1, h2 = handles
        h1.set_data(v["x"], v["y"])
        h2.set_position([v["lx"], v["ly"]])
        h2.set_text(v["label"](p.use_latex))

    for v, handles in zip(tp_dict.values(), tp_handles):
        h1, h2 = handles
        h1.set_data(v["x"], v["y"])
        h2.set_position([v["lx"], v["ly"]])
        h2.set_text(v["label"])

    for v, handles in zip(ts_dict.values(), ts_handles):
        h1, h2 = handles
        h1.set_data(v["x"], v["y"])
        h2.set_position([v["lx"], v["ly"]])
        h2.set_text(v["label"])


class ZGridLineRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_zgrid_helper: _update_zgrid_helper
    }
