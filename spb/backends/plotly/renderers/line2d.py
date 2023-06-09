from spb.backends.base_renderer import Renderer


def _scatter_class(plot, n, polar=False):
    go = plot.go
    if not polar:
        return go.Scatter if n < plot.scattergl_threshold else go.Scattergl
    return go.Scatterpolar if n < plot.scattergl_threshold else go.Scatterpolargl


def _draw_line2d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    if s.is_parametric:
        x, y, param = data
        # hides/show the colormap depending on s.use_cm
        mode = "lines+markers" if not s.is_point else "markers"
        if (not s.is_point) and (not s.use_cm):
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
        if s.is_polar:
            ht = "r: %{r}<br />θ: %{theta}<br />u: %{customdata}"

        lkw = dict(
            name=s.get_label(p._use_latex),
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
                    s.get_label(p._use_latex), True)

        kw = p.merge({}, lkw, s.rendering_kw)

        if s.is_polar:
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
            name=s.get_label(p._use_latex),
            mode="lines" if not s.is_point else "markers",
            line_color=color,
            showlegend=s.show_in_legend
        )
        if s.is_point:
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
        if s.is_polar:
            kw.setdefault("thetaunit", "radians")
            cls = _scatter_class(p, len(x), True)
            handle = cls(r=y, theta=x, **kw)
        else:
            cls = _scatter_class(p, len(x))
            handle = cls(x=x, y=y, **kw)
    
    p._fig.add_trace(handle)
    return handle


def _update_line2d_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series

    if s.is_2Dline and s.is_parametric:
        x, y, param = data
        handle["x"] = x
        handle["y"] = y
        handle["marker"]["color"] = param
        handle["customdata"] = param
    else:
        x, y = data
        if not s.is_polar:
            handle["x"] = x
            handle["y"] = y
        else:
            handle["r"] = y
            handle["theta"] = x


class Line2DRenderer(Renderer):
    draw_update_map = {
        _draw_line2d_helper: _update_line2d_helper
    }
