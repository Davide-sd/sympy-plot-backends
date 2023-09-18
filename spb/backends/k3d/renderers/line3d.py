from spb.backends.base_renderer import Renderer


def _draw_line3d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np

    if s.is_2Dline:
        raise NotImplementedError

    if s.is_point:
        if s.is_parametric:
            x, y, z, param = data
        else:
            x, y, z = data
            param = np.zeros_like(x)
        positions = np.vstack([x, y, z]).T.astype(np.float32)
        a = dict(
            point_size=0.2, color=p._convert_to_int(next(p._cl)))
        if s.use_cm:
            a["color_map"] = next(p._cm)
            a["attribute"] = param
        kw = p.merge({}, a, s.rendering_kw)
        plt_points = p.k3d.points(positions=positions, **kw)
        plt_points.shader = "mesh"
        handle = plt_points
        p._fig += plt_points

    else:
        if s.is_parametric:
            x, y, z, param = data
        else:
            x, y, z = data
            param = np.zeros_like(x)
        vertices = np.vstack([x, y, z]).T.astype(np.float32)
        # keyword arguments for the line object
        a = dict(
            width=0.1 if s.show_in_legend else 0.001,
            name=p._get_series_label(s, "%s") if p._show_label else None,
            color=(
                p.wireframe_color if not s.show_in_legend
                else (
                    p._convert_to_int(next(p._cl)) if s.line_color is None
                    else s.line_color
                )
            ),
            shader="mesh",
        )
        if s.use_cm:
            a["attribute"] = (param.astype(np.float32),)
            a["color_map"] = next(p._cm)
            a["color_range"] = [float(param.min()), float(param.max())]
        kw = p.merge({}, a, s.rendering_kw)
        line = p.k3d.line(vertices, **kw)
        handle = line
        p._fig += line

    return handle


def _update_line3d_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    np = p.np

    if s.is_3Dline and s.is_point:
        if s.is_parametric:
            x, y, z, _ = data
        else:
            x, y, z = data
        positions = np.vstack([x, y, z]).T.astype(np.float32)
        handle.positions = positions

    else:
        x, y, z, _ = data
        vertices = np.vstack([x, y, z]).T.astype(np.float32)
        handle.vertices = vertices


class Line3DRenderer(Renderer):
    draw_update_map = {
        _draw_line3d_helper: _update_line3d_helper
    }
