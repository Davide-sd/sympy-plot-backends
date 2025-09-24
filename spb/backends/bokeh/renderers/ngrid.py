from spb.backends.bokeh.renderers.sgrid import (
    SGridLineRenderer, _text_position_limits, _find_data_axis_limits,
    _modify_axis_limits
)
from sympy.external import import_module


def _draw_ngrid_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    m_mag, m_phase, n_mag, n_phase, phase_offsets = data

    lkw = {"line_color": "#aaa", "line_dash": "dotted"}
    kw = p.merge({}, lkw, s.rendering_kw)

    xlim, ylim, xtext_pos_lim, ytext_pos_lim = _text_position_limits(
        renderer, p, s)

    cl_mag_lines = []
    cl_phase_lines = []
    cl_mag_labels = []
    cl_phase_labels = []

    for idx, phase_offset in enumerate(phase_offsets):
        if s.show_cl_mags:
            x = (m_phase + phase_offset).T
            y = m_mag.T
            for _x, _y in zip(x, y):
                source = {"x": _x, "y": _y}
                cl_mag_lines.append(
                    p._fig.line("x", "y", source=source, **kw)
                )

        if s.show_cl_phases:
            x = (n_phase + phase_offset).T
            y = n_mag.T
            for _x, _y in zip(x, y):
                source = {"x": _x, "y": _y}
                cl_phase_lines.append(
                    p._fig.line("x", "y", source=source, **kw)
                )

        if idx == len(phase_offsets) - 1:
            if s.show_cl_mags:
                # Add magnitude labels
                for x, y, m in zip(
                    m_phase[:][-1] + phase_offset,
                    m_mag[:][-1],
                    s.cl_mags
                ):
                    align = 'right' if m < 0.0 else 'left'
                    label = '%s dB' % (m if m != 0 else 0)
                    h = p.bokeh.models.Label(
                        x=x, y=y, text=label, text_font_size='12px',
                        text_align=align, text_color='gray'
                    )
                    p._fig.add_layout(h)
                    cl_mag_labels.append(h)
            if s.show_cl_phases:
                # phase labels
                if s.label_cl_phases:
                    for x, y, pp in zip(
                        n_phase[:][0] + phase_offset,
                        n_mag[:][0],
                        s.cl_phases
                    ):
                        if pp > -175:
                            align = 'right'
                        elif pp > -185:
                            align = 'center'
                        else:
                            align = 'left'
                        h = p.bokeh.models.Label(
                            x=x, y=y, text=f'{round(pp)}\N{DEGREE SIGN}',
                            text_font_size='12px', text_baseline="bottom",
                            text_align=align, text_color='gray'
                        )
                        p._fig.add_layout(h)
                        cl_phase_labels.append(h)

    xm = phase_offsets - 180
    markers = p._fig.scatter(
        x=xm, y=np.zeros_like(xm), marker="+", color="red", size=10)

    return [
        cl_mag_lines, cl_phase_lines, cl_mag_labels, cl_phase_labels, markers
    ]


def _update_ngrid_helper(renderer, data, handles):
    # Too time-consuming to implement. Set xlim/ylim on the plot call and live
    # with it :)
    pass


class NGridLineRenderer(SGridLineRenderer):
    default_xlim = [-360, 0]
    # From NGridLineSeries.get_data(), we see that key_cl_mags goes from -40dB
    # up to 12dB. If needed, more values are added at the bottom. Here, I set
    # -60dB so that there will be a magnitude line, which in turn it is used
    # to set the y-position of phase labels (if label_cl_phases=True).
    default_ylim = [-60, 55]

    def _set_axis_limits_before_compute_data(self):
        np = import_module("numpy")
        # loop over the data already present on the plot and find
        # appropriate axis limits
        xlims, ylims = _find_data_axis_limits(self.plot._fig, "ol_p", "ol_m")

        if len(xlims) > 0:
            xlims = np.array(xlims)
            ylims = np.array(ylims)
            xlim = (np.nanmin(xlims[:, 0]), np.nanmax(xlims[:, 1]))
            ylim = (np.nanmin(ylims[:, 0]), np.nanmax(ylims[:, 1]))
        else:
            xlim = self.plot.xlim if self.plot.xlim else self.default_xlim
            ylim = self.plot.ylim if self.plot.ylim else self.default_ylim

        # need to do this otherwise grid labels would be difficult to place.
        non_grid_series = [s for s in self.plot.series if not s.is_grid]
        if len(non_grid_series) > 0:
            xlim = list(xlim)
            xlim[0] = xlim[0] - (xlim[0] % 360)
            xlim[1] = xlim[1] + (360 - (xlim[1] % 360))
        ylim = _modify_axis_limits(ylim, 0.05, 0.05)
        self.series.set_axis_limits(xlim, ylim)

    draw_update_map = {
        _draw_ngrid_helper: _update_ngrid_helper
    }
