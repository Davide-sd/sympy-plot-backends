from spb.backends.base_renderer import Renderer
from spb.backends.utils import compute_streamtubes


def _draw_vector3d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    handle = []

    if s.is_streamlines:
        xx, yy, zz, uu, vv, ww = data
        vertices, color_val = compute_streamtubes(
            xx, yy, zz, uu, vv, ww, s.rendering_kw, s.color_func)

        stream_kw = s.rendering_kw.copy()
        skw = dict(width=0.1, shader="mesh")
        if s.use_cm:
            skw["color_map"] = stream_kw.get("color", next(p._cm))
            skw["color_range"] = [
                float(np.nanmin(color_val)), float(np.nanmax(color_val))
            ]
            skw["attribute"] = color_val
        else:
            col = stream_kw.pop("color", next(p._cl))
            if not isinstance(col, int):
                col = p._convert_to_int(col)
            stream_kw["color"] = col

        kw = p.merge({}, skw, stream_kw)
        streamlines = p.k3d.line(
            vertices.astype(np.float32), **kw)
        handle.append(streamlines)
        p._fig += streamlines
    else:
        xx, yy, zz, uu, vv, ww = data
        qkw = dict(scale=1)
        qkw = p.merge(qkw, s.rendering_kw)
        quiver_kw = s.rendering_kw
        if s.use_cm:
            colormap = quiver_kw.get("color", next(p._cm))
        else:
            colormap = None
            col = quiver_kw.get("color", next(p._cl))
            if not isinstance(col, int):
                col = p._convert_to_int(col)
            solid_color = col * np.ones(xx.size)

        origins, vectors, colors = p._build_k3d_vector_data(
            xx, yy, zz, uu, vv, ww, qkw, colormap, s.normalize, s)
        if colors is None:
            colors = solid_color
        vec_colors = p._create_vector_colors(colors)

        pivot = quiver_kw.get("pivot", "mid")
        if pivot not in p._qp_offset.keys():
            raise ValueError(
                "`pivot` must be one of the following values: "
                "{}".format(list(p._qp_offset.keys())))

        vec_kw = qkw.copy()
        kw_to_remove = ["scale", "color", "pivot"]
        for k in kw_to_remove:
            if k in vec_kw.keys():
                vec_kw.pop(k)
        vec_kw["origins"] = origins - vectors * p._qp_offset[pivot]
        vec_kw["vectors"] = vectors
        vec_kw["colors"] = vec_colors

        vec = p.k3d.vectors(**vec_kw)
        p._fig += vec
        handle = [vec, qkw, colormap]

    return handle


def _update_vector3d_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series

    if s.is_streamlines:
        raise NotImplementedError

    xx, yy, zz, uu, vv, ww = data
    quivers, qkw, colormap = handle
    origins, vectors, colors = p._build_k3d_vector_data(
        xx, yy, zz, uu, vv, ww, qkw, colormap, s.normalize, s)
    if colors is not None:
        vec_colors = p._create_vector_colors(colors)
        quivers.colors = vec_colors

    pivot = s.rendering_kw.get("pivot", "mid")
    quivers.origins = origins - vectors * p._qp_offset[pivot]
    quivers.vectors = vectors


class Vector3DRenderer(Renderer):
    draw_update_map = {
        _draw_vector3d_helper: _update_vector3d_helper
    }
