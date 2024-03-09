from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.backends.matplotlib.renderers.sgrid import (
    SGridLineRenderer, _text_position_limits)
from spb.series import NGridLineSeries


def _draw_ngrid_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    ax = p._ax
    m_mag, m_phase, n_mag, n_phase, phase_offsets = data

    lkw = p.ngrid_line_kw
    kw = p.merge({}, lkw, s.rendering_kw)

    xlim, ylim, xtext_pos_lim, ytext_pos_lim = _text_position_limits(
        renderer, p, s)

    cl_mag_lines = []
    cl_phase_lines = []
    cl_mag_labels = []
    cl_phase_labels = []

    for idx, phase_offset in enumerate(phase_offsets):
        if s.show_cl_mags:
            cl_mag_lines.append(
                ax.plot(
                    m_phase + phase_offset,
                    m_mag,
                    **kw)[0]
                )
        if s.show_cl_phases:
            cl_phase_lines.append(
                ax.plot(
                    n_phase + phase_offset, n_mag,
                    **kw)[0]
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
                    cl_mag_labels.append(
                        ax.text(
                            x, y, label, size='small', ha=align,
                            color='gray', clip_on=True
                        )
                    )
            if s.show_cl_phases:
                # phase labels
                if s.label_cl_phases:
                    for x, y, p in zip(
                        n_phase[:][0] + phase_offset,
                        n_mag[:][0],
                        s.cl_phases
                    ):
                        if p > -175:
                            align = 'right'
                        elif p > -185:
                            align = 'center'
                        else:
                            align = 'left'
                        cl_phase_labels.append(
                            ax.text(
                                x, y, f'{round(p)}\N{DEGREE SIGN}',
                                size='small',
                                ha=align,
                                va='bottom',
                                color='gray',
                                clip_on=True)
                            )

    xm = phase_offsets - 180
    markers, = ax.plot(
        xm, np.zeros_like(xm), linestyle="none", marker="+", color="r")

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
    default_ylim = [-60, 50]

    draw_update_map = {
        _draw_ngrid_helper: _update_ngrid_helper
    }
