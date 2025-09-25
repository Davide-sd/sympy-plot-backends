from spb.backends.base_renderer import Renderer


def _scatter_class(plot, n, polar=False):
    go = plot.go
    if not polar:
        return go.Scatter if n < plot.scattergl_threshold else go.Scattergl
    return (
        go.Scatterpolar if n < plot.scattergl_threshold
        else go.Scatterpolargl
    )


def _draw_line2d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    if s.is_parametric:
        x, y, param = data
        # hides/show the colormap depending on s.use_cm
        mode = "lines+markers" if not s.is_scatter else "markers"
        if (not s.is_scatter) and (not s.use_cm):
            mode = "lines"
        if s.get_label(False) != "__k__":
            color = next(p._cl) if s.line_color is None else s.line_color
        else:
            color = "black"
        # hover template
        ht = (
            "x: %{x}<br />y: %{y}<br />u: %{customdata}"
            if not s.is_complex
            else "x: %{x}<br />y: %{y}<br />Arg: %{customdata}"
        )
        if hasattr(s, "is_polar") and s.is_polar:
            ht = "r: %{r}<br />θ: %{theta}<br />u: %{customdata}"

        lkw = dict(
            name=s.get_label(p.use_latex),
            line_color=color,
            mode=mode,
            customdata=param,
            hovertemplate=ht,
            showlegend=s.show_in_legend,
        )
        if s.use_cm:
            lkw["marker"] = dict(
                color=param,
                colorscale=(
                    next(p._cyccm)
                    if p._use_cyclic_cm(param, s.is_complex)
                    else next(p._cm)
                ),
                size=6,
                showscale=s.use_cm and s.colorbar,
            )
            if lkw["marker"]["showscale"]:
                # only add a colorbar if required.

                # TODO: when plotting many (14 or more) parametric
                # expressions, each one requiring a colorbar, it might
                # happens that the horizontal space required by all
                # colorbars is greater than the available figure width.
                # That raises a strange error.
                lkw["marker"]["colorbar"] = p._create_colorbar(
                    s.get_label(p.use_latex), True)

        kw = p.merge({}, lkw, s.rendering_kw)

        if hasattr(s, "is_polar") and s.is_polar:
            kw.setdefault("thetaunit", "radians")
            cls = _scatter_class(p, len(x), True)
            handle = cls(r=y, theta=x, **kw)
        else:
            cls = _scatter_class(p, len(x))
            handle = cls(x=x, y=y, **kw)
    else:
        x, y = data
        color = next(p._cl) if s.line_color is None else s.line_color
        lkw = dict(
            name=s.get_label(p.use_latex),
            mode="lines" if not s.is_scatter else "markers",
            line_color=color,
            showlegend=s.show_in_legend
        )
        if s.is_scatter:
            lkw["marker"] = dict(size=8)
            if not s.is_filled:
                lkw["marker"] = dict(
                    color="#E5ECF6",
                    size=8,
                    line=dict(
                        width=2,
                        color=color
                    )
                )
        kw = p.merge({}, lkw, s.rendering_kw)
        if hasattr(s, "is_polar") and s.is_polar:
            kw.setdefault("thetaunit", "radians")
            cls = _scatter_class(p, len(x), True)
            handle = cls(r=y, theta=x, **kw)
        else:
            cls = _scatter_class(p, len(x))
            handle = cls(x=x, y=y, **kw)

    p._fig.add_trace(handle)

    # add vertical lines at discontinuities
    hvlines = []
    if hasattr(s, "poles_locations"):
        pole_line_kw = p.merge(
            {"line": dict(color='black', dash='dot', width=1)},
            s.poles_rendering_kw
        )
        for x_loc in s.poles_locations:
            p._fig.add_vline(float(x_loc), **pole_line_kw)
        n = len(p._fig.layout["shapes"])
        m = len(s.poles_locations)
        hvlines = list(range(n - m, n))

    # NOTE: as of Plotly 5.12.0, the figure appears to create a copy of
    # `handle`, which cannot be used to update the figure. Hence, need to keep
    # track of the index of traces
    return [len(p._fig.data) - 1, hvlines]


def _update_line2d_helper(renderer, data, idxs):
    p, s = renderer.plot, renderer.series
    handle = p.fig.data[idxs[0]]
    vlines_idx = idxs[1]

    if s.is_2Dline and s.is_parametric:
        x, y, param = data
        handle["x"] = x
        handle["y"] = y
        handle["marker"]["color"] = param
        handle["customdata"] = param
    else:
        x, y = data
        if hasattr(s, "is_polar") and s.is_polar:
            handle["r"] = y
            handle["theta"] = x
        else:
            handle["x"] = x
            handle["y"] = y

    # update vertical lines
    if hasattr(s, "poles_locations"):
        if len(vlines_idx) != len(s.poles_locations):
            # TODO: highly unreliable! It doesn't work.
            shapes = list(p._fig.layout.shapes)
            p._fig.layout.shapes = []
            for idx in reversed(vlines_idx):
                shapes.pop(idx)
            for shape in shapes:
                p._fig.add_shape(shape)

            pole_line_kw = p.merge(
                {"line": dict(color='black', dash='dot', width=1)},
                s.poles_rendering_kw
            )
            for x_loc in s.poles_locations:
                # TODO: weirdly, add_vline refuses to work here...
                # p._fig.add_vline(x=float(x_loc), **p.pole_line_kw)
                p._fig.add_shape(
                    type="line", x0=float(x_loc), x1=float(x_loc),
                    y0=0, y1=1, xref="x", yref="y domain",
                    **pole_line_kw
                )
            n = len(p._fig.layout["shapes"])
            m = len(s.poles_locations)
            idxs[1] = list(range(n - m, n))
        elif len(vlines_idx) > 0:
            for idx, x_loc in zip(vlines_idx, s.poles_locations):
                p._fig.layout.shapes[idx]["x0"] = float(x_loc)
                p._fig.layout.shapes[idx]["x1"] = float(x_loc)


class Line2DRenderer(Renderer):
    draw_update_map = {
        _draw_line2d_helper: _update_line2d_helper
    }
