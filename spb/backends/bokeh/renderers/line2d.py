from spb.backends.base_renderer import Renderer


def _draw_line2d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    handle = []

    if s.is_parametric and s.use_cm:
        x, y, param = data
        colormap = (
            next(p._cyccm)
            if p._use_cyclic_cm(param, s.is_complex)
            else next(p._cm)
        )
        ds, line, cb, kw = p._create_gradient_line(
            x, y, param, colormap, s.get_label(p._use_latex),
            s.rendering_kw, s.is_point)
        h = p._fig.add_glyph(ds, line)
        handle.append(h)
        if s.colorbar:
            handle.append(cb)
            p._fig.add_layout(cb, "right")
    else:
        if s.is_parametric:
            x, y, param = data
            source = {"xs": x, "ys": y, "us": param}
        else:
            x, y = data
            source = {
                "xs": x if not s.is_polar else y * np.cos(x),
                "ys": y if not s.is_polar else y * np.sin(x)
            }

        if s.get_label(False) != "__k__":
            color = next(p._cl) if s.line_color is None else s.line_color
        else:
            color = "#000000"
        lkw = dict(
            line_width=2, color=color,
            legend_label=s.get_label(p._use_latex)
        )
        if not s.is_point:
            kw = p.merge({}, lkw, s.rendering_kw)
            handle = [p._fig.line("xs", "ys", source=source, **kw)]
        else:
            lkw["size"] = 8
            lkw["marker"] = "circle"
            if not s.is_filled:
                lkw["fill_color"] = "white"
            kw = p.merge({}, lkw, s.rendering_kw)
            handle = [p._fig.scatter("xs", "ys", source=source, **kw)]

    # add vertical lines at discontinuities
    vlines = []
    for x_loc in s.poles_locations:
        vl = p.bokeh.models.Span(
            location=float(x_loc), dimension="height", **p.pole_line_kw)
        p._fig.add_layout(vl)
        vlines.append(vl)
    return [handle, vlines]


def _update_line2d_helper(renderer, data, handles):
    p, s = renderer.plot, renderer.series
    np = p.np
    handle, vlines = handles

    if s.is_2Dline and s.is_parametric and s.use_cm:
        x, y, param = data
        if not s.is_point:
            xs, ys, us = p._get_segments(x, y, param)
        else:
            xs, ys, us = x, y, param
        handle[0].data_source.data.update({"xs": xs, "ys": ys, "us": us})
        if len(handle) > 1:
            cb = handle[1]
            cb.color_mapper.update(low=min(us), high=max(us))

    elif s.is_2Dline:
        if s.is_parametric:
            x, y, param = data
            source = {"xs": x, "ys": y, "us": param}
        else:
            x, y = data
            source = {
                "xs": x if not s.is_polar else y * np.cos(x),
                "ys": y if not s.is_polar else y * np.sin(x)
            }
        handle[0].data_source.data.update(source)

    # update vertical lines
    if len(vlines) != len(s.poles_locations):
        # instead of removing elements from p._fig.center, let's add new
        # lines or hide the ones that are not needed
        if len(vlines) < len(s.poles_locations):
            for i in range(len(s.poles_locations) - len(vlines)):
                vl = p.bokeh.models.Span(
                    location=0, dimension="height", **p.pole_line_kw)
                p._fig.add_layout(vl)
                handles[1].append(vl)

        vlines = handles[1]
        for vl, x_loc in zip(vlines, s.poles_locations):
            vl.location = float(x_loc)

        # hide the unnecessary ones
        for vl in vlines[len(s.poles_locations):]:
            vl.visible = False

    elif len(vlines) > 0:
        for vl, x_loc in zip(vlines, s.poles_locations):
            vl.location = float(x_loc)


class Line2DRenderer(Renderer):
    draw_update_map = {
        _draw_line2d_helper: _update_line2d_helper
    }
