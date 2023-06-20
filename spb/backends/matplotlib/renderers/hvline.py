from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer


def _draw_hvline_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    rkw = {"color": "k"}
    if s.label is not None:
        rkw["label"] = s.label
    rkw = p.merge(rkw, s.rendering_kw)
    if s.is_horizontal:
        handle = p._ax.axhline(data, **rkw)
    else:
        handle = p._ax.axvline(data, **rkw)
    return handle


def _update_hvline_helper(renderer, data, handle):
    s = renderer.series
    method = handle.set_ydata if s.is_horizontal else handle.set_xdata
    method([data, data])


class HVLineRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_hvline_helper: _update_hvline_helper
    }
