from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer


def _draw_line3d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    if s.is_parametric:
        x, y, z, param = data
    else:
        x, y, z = data
        param = p.np.ones_like(x)
    lkw = dict()

    if not s.is_point:
        if s.use_cm:
            segments = p.get_segments(x, y, z)
            lkw["cmap"] = next(p._cm)
            lkw["array"] = param
            kw = p.merge({}, lkw, s.rendering_kw)
            c = p.Line3DCollection(segments, **kw)
            p._ax.add_collection(c)
            p._add_colorbar(
                c, s.get_label(p.use_latex), s.use_cm and s.colorbar)
            handle = (c, kw, p.fig.axes[-1])
        else:
            slabel = s.get_label(p.use_latex)
            lkw["label"] = slabel if s.show_in_legend else "_nolegend_"
            lkw["color"] = (
                (
                    next(p._cl) if s.line_color is None
                    else s.line_color
                ) if (s.show_in_legend or not s._is_wireframe_line)
                else p.wireframe_color
            )
            kw = p.merge({}, lkw, s.rendering_kw)
            l = p._ax.plot(x, y, z, **kw)
            handle = l
    else:
        if s.use_cm:
            lkw["cmap"] = next(p._cm)
            lkw["c"] = param
        else:
            lkw["label"] = s.get_label(p.use_latex)
            color = next(p._cl) if s.line_color is None else s.line_color
            lkw["color"] = color

        if not s.is_filled:
            lkw["facecolors"] = "none"

        lkw["alpha"] = 1
        kw = p.merge({}, lkw, s.rendering_kw)
        l = p._ax.scatter(x, y, z, **kw)
        if s.use_cm:
            p._add_colorbar(
                l, s.get_label(p.use_latex), s.use_cm and s.colorbar)
            handle = [l, kw, p.fig.axes[-1]]
        else:
            handle = [l]
    return handle


def _update_line3d_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    if s.is_parametric:
        x, y, z, param = data
    else:
        x, y, z = data

    line = handle[0]
    if isinstance(line, p.Line3DCollection):
        # gradient lines
        segments = p.get_segments(x, y, z)
        line.set_segments(segments)
    elif isinstance(line, p.Path3DCollection):
        # 3D points
        line._offsets3d = (x, y, z)
    else:
        if hasattr(line, "set_data_3d"):
            # solid lines
            line.set_data_3d(x, y, z)
        else:
            # scatter
            line.set_offset(p.np.c_[x, y, z])

    if s.is_parametric and s.use_cm:
        line.set_array(param)
        kw, cax = handle[1:]
        p._update_colorbar(
            cax, kw["cmap"], s.get_label(p.use_latex), param=param)


class Line3DRenderer(MatplotlibRenderer):
    _sal = True
    draw_update_map = {
        _draw_line3d_helper: _update_line3d_helper
    }
