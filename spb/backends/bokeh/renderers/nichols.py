from spb.backends.base_renderer import Renderer
from spb.backends.bokeh.renderers.nyquist import _compute_arrows_position


def _draw_nichols_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    handles = []
    omega, ol_phase, ol_mag, cl_phase, cl_mag = data
    tooltips = [
        ("OL Phase [deg]", "@ol_p"), ("OL Mag [dB]", "@ol_m"),
        ("CL Phase [deg]", "@cl_p"), ("CL Mag [dB]", "@cl_m"),
        ("Freq [rad/s]", "@o")
    ]

    if s.use_cm:
        colormap = (
            next(p._cyccm)
            if p._use_cyclic_cm(omega, s.is_complex)
            else next(p._cm)
        )

        if not s.is_point:
            ol_phase, ol_mag, omega, cl_phase, cl_mag = p._get_segments(
                ol_phase, ol_mag, omega,
                cl_phase, cl_mag
            )

        source = {
            "ol_p": ol_phase, "ol_m": ol_mag,
            "cl_p": cl_phase, "cl_m": cl_mag, "o": omega
        }

        ds, line, cb, kw = p._create_gradient_line(
            "ol_p", "ol_m", "o", source,
            colormap, s.get_label(p._use_latex),
            s.rendering_kw, s.is_point)
        h = p._fig.add_glyph(ds, line)
        handles.append(h)

        if s.colorbar:
            handles.append(cb)
            p._fig.add_layout(cb, "right")
    else:
        source = {
            "ol_p": ol_phase, "ol_m": ol_mag,
            "cl_p": cl_phase, "cl_m": cl_mag, "o": omega
        }
        color = next(p._cl) if s.line_color is None else s.line_color
        lkw = dict(
            line_width=2, color=color,
            legend_label=s.get_label(p._use_latex)
        )
        arrows = None
        if not s.is_point:
            kw = p.merge({}, lkw, s.rendering_kw)
            handles.append(p._fig.line("ol_p", "ol_m", source=source, **kw))

            # Set the arrow style
            arrow_style = p.bokeh.models.VeeHead(
                    line_color=color, fill_color=color, size=10)
            source = _compute_arrows_position(
                ol_phase, ol_mag, arrow_locs=s.arrow_locs, dir=1)
            arrows = p.bokeh.models.Arrow(
                source=p.bokeh.models.ColumnDataSource(data=source),
                line_color=color, end=arrow_style)
            p._fig.add_layout(arrows)
            handles.append(arrows)

            # bind all handles' visibility to the primary_line's visibility
            callback = p.bokeh.models.CustomJS(
                args=dict(line=handles[0], arrows=arrows),
                code="""
                if (arrows !== null) {
                    arrows.visible = line.visible;
                }
                """
            )
            handles[0].js_on_change("visible", callback)
        else:
            lkw["size"] = 8
            lkw["marker"] = "circle"
            if not s.is_filled:
                lkw["fill_color"] = "white"
            kw = p.merge({}, lkw, s.rendering_kw)
            handles.append(p._fig.scatter("ol_p", "ol_m", source=source, **kw))

    p._fig.add_tools(p.bokeh.models.HoverTool(
        tooltips=tooltips,
        renderers=[handles[0]]
    ))

    return handles


def _update_nichols_helper(renderer, data, handles):
    p, s = renderer.plot, renderer.series
    np = p.np
    omega, ol_phase, ol_mag, cl_phase, cl_mag = data

    if s.use_cm:
        if not s.is_point:
            ol_phase, ol_mag, omega, cl_phase, cl_mag = p._get_segments(
                ol_phase, ol_mag, omega,
                cl_phase, cl_mag
            )
        source = {
            "ol_p": ol_phase, "ol_m": ol_mag,
            "cl_p": cl_phase, "cl_m": cl_mag, "o": omega
        }
        handles[0].data_source.data.update(source)
        if len(handles) > 1:
            cb = handles[1]
            cb.color_mapper.update(low=min(omega), high=max(omega))

    else:
        source = {
            "ol_p": ol_phase, "ol_m": ol_mag,
            "cl_p": cl_phase, "cl_m": cl_mag, "o": omega
        }
        handles[0].data_source.data.update(source)
        if len(handles) > 1:
            source = _compute_arrows_position(
                ol_phase, ol_mag, arrow_locs=s.arrow_locs, dir=1)
            handles[1].source.data.update(source)


class NicholsLineRenderer(Renderer):
    draw_update_map = {
        _draw_nichols_helper: _update_nichols_helper
    }
