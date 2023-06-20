from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.backends.utils import compute_streamtubes


def _draw_vector3d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    xx, yy, zz, uu, vv, ww = data
    mag = p.np.sqrt(uu ** 2 + vv ** 2 + ww ** 2)

    uu0, vv0, zz0 = [t.copy() for t in [uu, vv, ww]]
    if s.normalize:
        uu, vv, ww = [t / mag for t in [uu, vv, ww]]

    if s.is_streamlines:
        vertices, color_val = compute_streamtubes(
            xx, yy, zz, uu, vv, ww, s.rendering_kw,
            s.color_func)

        lkw = dict()
        stream_kw = s.rendering_kw.copy()
        # remove rendering-unrelated keywords
        for k in ["starts", "max_prop", "npoints", "radius"]:
            if k in stream_kw.keys():
                stream_kw.pop(k)

        if s.use_cm:
            segments = p.get_segments(
                vertices[:, 0], vertices[:, 1], vertices[:, 2])
            lkw["cmap"] = next(p._cm)
            lkw["array"] = color_val
            kw = p.merge({}, lkw, stream_kw)
            streamlines = p.Line3DCollection(segments, **kw)
            p._ax.add_collection(streamlines)
            p._add_colorbar(
                streamlines, s.get_label(p._use_latex), s.use_cm and s.colorbar)

        else:
            lkw["label"] = s.get_label(p._use_latex)
            kw = p.merge({}, lkw, stream_kw)
            streamlines = p._ax.plot(vertices[:, 0], vertices[:, 1],
                vertices[:, 2], **kw)

        handle = [streamlines]
    else:
        qkw = dict()
        if s.use_cm:
            # NOTE: each quiver is composed of 3 lines: the
            # stem and two segments for the head. I could set
            # the colors keyword argument in order to apply
            # the same color to the entire quiver, like this:
            # [c1, c2, ..., cn, c1, c1, c2, c2, ... cn, cn]
            # However, it doesn't appear to work reliably, so
            # I'll keep things simpler.
            color_val = mag
            if s.color_func is not None:
                color_val = s.eval_color_func(
                    xx, yy, zz, uu0, vv0, zz0)
            qkw["cmap"] = next(p._cm)
            qkw["array"] = color_val.flatten()
            kw = p.merge({}, qkw, s.rendering_kw)
            q = p._ax.quiver(xx, yy, zz, uu, vv, ww, **kw)
            is_cb_added = p._add_colorbar(
                q, s.get_label(p._use_latex), s.use_cm and s.colorbar)
        else:
            qkw["color"] = next(p._cl)
            kw = p.merge({}, qkw, s.rendering_kw)
            q = p._ax.quiver(xx, yy, zz, uu, vv, ww, **kw)
            is_cb_added = False
        handle = [q, kw, is_cb_added, p._fig.axes[-1]]
    return handle


def _update_vector3d_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    if s.is_streamlines:
        raise NotImplementedError

    xx, yy, zz, uu, vv, ww = data
    quivers, kw, is_cb_added, cax = handle
    mag = p.np.sqrt(uu ** 2 + vv ** 2 + ww ** 2)
    uu0, vv0, ww0 = [t.copy() for t in [uu, vv, ww]]
    if s.normalize:
        uu, vv, ww = [t / mag for t in [uu, vv, ww]]
    quivers.remove()

    if "array" in kw.keys():
        color_val = mag
        if s.color_func is not None:
            color_val = s.eval_color_func(xx, yy, zz, uu0, vv0, ww0)
        kw["array"] = color_val.flatten()

    handle[0] = p._ax.quiver(xx, yy, zz, uu, vv, ww, **kw)

    if is_cb_added:
        p._update_colorbar(cax, kw["cmap"], s.get_label(p._use_latex), param=mag)


class Vector3DRenderer(MatplotlibRenderer):
    _sal = True
    draw_update_map = {
        _draw_vector3d_helper: _update_vector3d_helper
    }
