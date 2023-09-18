from spb.backends.base_renderer import Renderer
import warnings


def _draw_vector2d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    if s.color_func is not None:
        warnings.warn(
            "PlotlyBackend doesn't support custom "
            "coloring of 2D/3D quivers or streamlines plots. "
            "`color_func` will not be used.")

    xx, yy, uu, vv = data
    if s.normalize:
        mag = np.sqrt(uu**2 + vv**2)
        uu, vv = [t / mag for t in [uu, vv]]
    # NOTE: currently, it is not possible to create
    # quivers/streamlines with a color scale:
    # https://community.plotly.com/t/how-to-make-python-quiver-with-colorscale/41028
    col = next(p._qc)
    if s.is_streamlines:
        skw = dict(
            line_color=col, arrow_scale=0.15,
            name=s.get_label(p._use_latex)
        )
        kw = p.merge({}, skw, s.rendering_kw)
        stream = p.create_streamline(
            xx[0, :], yy[:, 0], uu, vv, **kw)
        handle = stream.data[0]
    else:
        qkw = dict(
            line_color=col, scale=0.075,
            name=s.get_label(p._use_latex)
        )
        kw = p.merge({}, qkw, s.rendering_kw)
        quiver = p.create_quiver(xx, yy, uu, vv, **kw)
        handle = quiver.data[0]
    p._fig.add_trace(handle)
    return [len(p._fig.data) - 1, col]


def _update_vector2d_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    np = p.np
    x, y, u, v = data
    idx, quivers_col = handle
    old_quivers = p.fig.data[idx]

    if s.normalize:
        mag = np.sqrt(u**2 + v**2)
        u, v = [t / mag for t in [u, v]]
    if s.is_streamlines:
        # TODO: iplot doesn't work with 2D streamlines.
        raise NotImplementedError
    else:
        qkw = dict(
            line_color=quivers_col, scale=0.075,
            name=s.get_label(p._use_latex)
        )
        kw = p.merge({}, qkw, s.rendering_kw)
        new_quivers = p.create_quiver(x, y, u, v, **kw)
        data = new_quivers.data[0]
    old_quivers["x"] = data["x"]
    old_quivers["y"] = data["y"]


class Vector2DRenderer(Renderer):
    draw_update_map = {
        _draw_vector2d_helper: _update_vector2d_helper
    }
