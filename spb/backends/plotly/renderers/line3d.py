from spb.backends.base_renderer import Renderer


def _draw_line3d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np

    # NOTE: As a design choice, I decided to show the legend entry
    # as well as the colorbar (if use_cm=True). Even though the
    # legend entry shows the wrong color (black line), it is useful
    # in order to hide/show a specific series whenever we are
    # plotting multiple series.
    if s.is_parametric:
        x, y, z, param = data
    else:
        x, y, z = data
        param = np.ones_like(x)

    if not s.is_point:
        lkw = dict(
            name=s.get_label(p.use_latex),
            mode="lines",
            showlegend=s.show_in_legend,
        )
        if s.use_cm:
            # only add a colorbar if required.

            # TODO: when plotting many (14 or more) parametric
            # expressions, each one requiring a colorbar, it might
            # happens that the horizontal space required by all
            # colorbars is greater than the available figure width.
            # That raises a strange error.
            lkw["line"] = dict(
                width=4,
                colorbar=p._create_colorbar(
                    s.get_label(p.use_latex), p._scale_down_colorbar),
                colorscale=(
                    next(p._cm) if s.use_cm
                    else p._solid_colorscale(s)
                ),
                color=param,
                showscale=s.colorbar,
            )
        else:
            lkw["line"] = dict(
                width=4,
                color=(
                    (
                        next(p._cl) if s.line_color is None
                        else s.line_color
                    ) if (
                        s.show_in_legend or
                        not s._is_wireframe_line
                    )
                    else p.wireframe_color
                )
            )
    else:
        color = next(p._cl) if s.line_color is None else s.line_color
        lkw = dict(
            name=s.get_label(p.use_latex),
            mode="markers",
            showlegend=s.show_in_legend)

        lkw["marker"] = dict(
            color=color if not s.use_cm else param,
            size=8,
            colorscale=next(p._cm) if s.use_cm else None,
            showscale=s.use_cm and s.colorbar,
        )
        if s.use_cm:
            lkw["marker"]["colorbar"] = p._create_colorbar(
                s.get_label(p.use_latex), p._scale_down_colorbar)

        if not s.is_filled:
            # TODO: how to show a colorscale if is_point=True
            # and is_filled=False?
            lkw["marker"] = dict(
                color="#E5ECF6",
                line=dict(
                    width=2,
                    color=lkw["marker"]["color"],
                    colorscale=lkw["marker"]["colorscale"],
                )
            )

    kw = p.merge({}, lkw, s.rendering_kw)
    handle = p.go.Scatter3d(x=x, y=y, z=z, **kw)
    p._fig.add_trace(handle)
    return len(p._fig.data) - 1


def _update_line3d_helper(renderer, data, idx):
    p, s = renderer.plot, renderer.series
    np = p.np
    handle = p.fig.data[idx]

    if s.is_parametric:
        x, y, z, param = data
    else:
        x, y, z = data
        param = np.zeros_like(x)
    handle["x"] = x
    handle["y"] = y
    handle["z"] = z
    if s.use_cm:
        handle["line"]["color"] = param


class Line3DRenderer(Renderer):
    draw_update_map = {
        _draw_line3d_helper: _update_line3d_helper
    }
