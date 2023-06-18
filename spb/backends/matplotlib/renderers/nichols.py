from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.backends.matplotlib.renderers.line2d import (
    _draw_line2d_helper, _update_line2d_helper
)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# NOTE: most of the following code comes from the `python-control` package,
# specifically the nichols.py module.

def _inner_extents(ax, xlim=None, ylim=None):
    # needs to loop over lines (mainly for interactive plots)
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    for l in ax.lines:
        x, y = l.get_data()
        xm, xM = np.nanmin(x), np.nanmax(x)
        ym, yM = np.nanmin(y), np.nanmax(y)
        if xm < xmin: xmin = xm
        if ym < ymin: ymin = ym
        if xM > xmax: xmax = xM
        if yM > ymax: ymax = yM

    if all(not np.isinf(t) for t in [xmin, ymin, xmax, ymax]):
        if xlim:
            xmin, xmax = xlim
        if ylim:
            ymin, ymax = ylim
        return xmin, ymin, xmax, ymax

    # intersection of data and view extents
    # if intersection empty, return view extents
    _inner = matplotlib.transforms.Bbox.intersection(ax.viewLim, ax.dataLim)
    if _inner is None:
        return ax.ViewLim.extents
    else:
        return _inner.extents


def nichols_grid(cl_mags=None, cl_phases=None, line_style='dotted', ax=None,
                 label_cl_phases=True, xlim=None, ylim=None):
    """Nichols chart grid

    Plots a Nichols chart grid on the current axis, or creates a new chart
    if no plot already exists.

    Parameters
    ----------
    cl_mags : array-like (dB), optional
        Array of closed-loop magnitudes defining the iso-gain lines on a
        custom Nichols chart.
    cl_phases : array-like (degrees), optional
        Array of closed-loop phases defining the iso-phase lines on a custom
        Nichols chart. Must be in the range -360 < cl_phases < 0
    line_style : string, optional
        :doc:`Matplotlib linestyle \
            <matplotlib:gallery/lines_bars_and_markers/linestyles>`
    ax : matplotlib.axes.Axes, optional
        Axes to add grid to.  If ``None``, use ``plt.gca()``.
    label_cl_phases: bool, optional
        If True, closed-loop phase lines will be labelled.
    xlim : (xmin, xmax)
        The extent of the x-axis.
    ylim : (ymin, ymax)
        The extent of the y-axis.

    Returns
    -------
    cl_mag_lines: list of `matplotlib.line.Line2D`
      The constant closed-loop gain contours
    cl_phase_lines: list of `matplotlib.line.Line2D`
      The constant closed-loop phase contours
    cl_mag_labels: list of `matplotlib.text.Text`
      mcontour labels; each entry corresponds to the respective entry
      in ``cl_mag_lines``
    cl_phase_labels: list of `matplotlib.text.Text`
      ncontour labels; each entry corresponds to the respective entry
      in ``cl_phase_lines``
    """
    if ax is None:
        ax = plt.gca()

    # Default chart size
    ol_phase_min = -359.99
    ol_phase_max = 0.0
    ol_mag_min = -40.0
    ol_mag_max = default_ol_mag_max = 50.0

    if ax.has_data():
        # Find extent of intersection the current dataset or view
        ol_phase_min, ol_mag_min, ol_phase_max, ol_mag_max = _inner_extents(ax, xlim, ylim)

    # M-circle magnitudes.
    if cl_mags is None:
        # Default chart magnitudes
        # The key set of magnitudes are always generated, since this
        # guarantees a recognizable Nichols chart grid.
        key_cl_mags = np.array([-40.0, -20.0, -12.0, -6.0, -3.0, -1.0, -0.5,
                                0.0, 0.25, 0.5, 1.0, 3.0, 6.0, 12.0])

        # Extend the range of magnitudes if necessary. The extended arange
        # will end up empty if no extension is required. Assumes that
        # closed-loop magnitudes are approximately aligned with open-loop
        # magnitudes beyond the value of np.min(key_cl_mags)
        cl_mag_step = -20.0  # dB
        extended_cl_mags = np.arange(np.min(key_cl_mags),
                                     ol_mag_min + cl_mag_step, cl_mag_step)
        cl_mags = np.concatenate((extended_cl_mags, key_cl_mags))

    # a minimum 360deg extent containing the phases
    phase_round_max = 360.0*np.ceil(ol_phase_max/360.0)
    phase_round_min = min(phase_round_max-360,
                          360.0*np.floor(ol_phase_min/360.0))

    # N-circle phases (should be in the range -360 to 0)
    if cl_phases is None:
        # aim for 9 lines, but always show (-360+eps, -180, -eps)
        # smallest spacing is 45, biggest is 180
        phase_span = phase_round_max - phase_round_min
        spacing = np.clip(round(phase_span / 8 / 45) * 45, 45, 180)
        key_cl_phases = np.array([-0.25, -359.75])
        other_cl_phases = np.arange(-spacing, -360.0, -spacing)
        cl_phases = np.unique(np.concatenate((key_cl_phases, other_cl_phases)))
    elif not ((-360 < np.min(cl_phases)) and (np.max(cl_phases) < 0.0)):
        raise ValueError('cl_phases must between -360 and 0, exclusive')

    # Find the M-contours
    m = m_circles(cl_mags, phase_min=np.min(cl_phases),
                  phase_max=np.max(cl_phases))
    m_mag = 20*np.log10(np.abs(m))
    m_phase = np.mod(np.degrees(np.angle(m)), -360.0)  # Unwrap

    # Find the N-contours
    n = n_circles(cl_phases, mag_min=np.min(cl_mags), mag_max=np.max(cl_mags))
    n_mag = 20*np.log10(np.abs(n))
    n_phase = np.mod(np.degrees(np.angle(n)), -360.0)  # Unwrap

    # Plot the contours behind other plot elements.
    # The "phase offset" is used to produce copies of the chart that cover
    # the entire range of the plotted data, starting from a base chart computed
    # over the range -360 < phase < 0. Given the range
    # the base chart is computed over, the phase offset should be 0
    # for -360 < ol_phase_min < 0.
    phase_offsets = 360 + np.arange(phase_round_min, phase_round_max, 360.0)

    cl_mag_lines = []
    cl_phase_lines = []
    cl_mag_labels = []
    cl_phase_labels = []

    for idx, phase_offset in enumerate(phase_offsets):
        # Draw M and N contours
        cl_mag_lines.extend(
            ax.plot(m_phase + phase_offset, m_mag, color='lightgray',
                    linestyle=line_style, zorder=0))
        cl_phase_lines.extend(
            ax.plot(n_phase + phase_offset, n_mag, color='lightgray',
                    linestyle=line_style, zorder=0))

        if idx == len(phase_offsets) - 1:
            # Add magnitude labels
            for x, y, m in zip(m_phase[:][-1] + phase_offset, m_mag[:][-1],
                            cl_mags):
                align = 'right' if m < 0.0 else 'left'
                label = '%s dB' % (m if m != 0 else 0)
                cl_mag_labels.append(
                    ax.text(x, y, label, size='small', ha=align,
                            color='gray', clip_on=True))

            # phase labels
            if label_cl_phases:
                for x, y, p in zip(n_phase[:][0] + phase_offset,
                                n_mag[:][0],
                                cl_phases):
                    if p > -175:
                        align = 'right'
                    elif p > -185:
                        align = 'center'
                    else:
                        align = 'left'
                    cl_phase_labels.append(
                        ax.text(x, y, f'{round(p)}\N{DEGREE SIGN}',
                                size='small',
                                ha=align,
                                va='bottom',
                                color='gray',
                                clip_on=True))

    xm = phase_offsets - 180
    markers, = ax.plot(xm, np.zeros_like(xm), linestyle="none", marker="+",
        color="r")

    # Fit axes to generated chart
    ax.axis([phase_round_min,
             phase_round_max,
             np.min(np.concatenate([cl_mags,[ol_mag_min]])),
             np.max([ol_mag_max, default_ol_mag_max])])

    return cl_mag_lines, cl_phase_lines, cl_mag_labels, cl_phase_labels, markers

#
# Utility functions
#
# This section of the code contains some utility functions for
# generating Nichols plots
#


def closed_loop_contours(Gcl_mags, Gcl_phases):
    """Contours of the function Gcl = Gol/(1+Gol), where
    Gol is an open-loop transfer function, and Gcl is a corresponding
    closed-loop transfer function.

    Parameters
    ----------
    Gcl_mags : array-like
        Array of magnitudes of the contours
    Gcl_phases : array-like
        Array of phases in radians of the contours

    Returns
    -------
    contours : complex array
        Array of complex numbers corresponding to the contours.
    """
    # Compute the contours in Gcl-space. Since we're given closed-loop
    # magnitudes and phases, this is just a case of converting them into
    # a complex number.
    Gcl = Gcl_mags*np.exp(1.j*Gcl_phases)

    # Invert Gcl = Gol/(1+Gol) to map the contours into the open-loop space
    return Gcl/(1.0 - Gcl)


def m_circles(mags, phase_min=-359.75, phase_max=-0.25):
    """Constant-magnitude contours of the function Gcl = Gol/(1+Gol), where
    Gol is an open-loop transfer function, and Gcl is a corresponding
    closed-loop transfer function.

    Parameters
    ----------
    mags : array-like
        Array of magnitudes in dB of the M-circles
    phase_min : degrees
        Minimum phase in degrees of the N-circles
    phase_max : degrees
        Maximum phase in degrees of the N-circles

    Returns
    -------
    contours : complex array
        Array of complex numbers corresponding to the contours.
    """
    # Convert magnitudes and phase range into a grid suitable for
    # building contours
    phases = np.radians(np.linspace(phase_min, phase_max, 2000))
    Gcl_mags, Gcl_phases = np.meshgrid(10.0**(mags/20.0), phases)
    return closed_loop_contours(Gcl_mags, Gcl_phases)


def n_circles(phases, mag_min=-40.0, mag_max=12.0):
    """Constant-phase contours of the function Gcl = Gol/(1+Gol), where
    Gol is an open-loop transfer function, and Gcl is a corresponding
    closed-loop transfer function.

    Parameters
    ----------
    phases : array-like
        Array of phases in degrees of the N-circles
    mag_min : dB
        Minimum magnitude in dB of the N-circles
    mag_max : dB
        Maximum magnitude in dB of the N-circles

    Returns
    -------
    contours : complex array
        Array of complex numbers corresponding to the contours.
    """
    # Convert phases and magnitude range into a grid suitable for
    # building contours
    mags = np.linspace(10**(mag_min/20.0), 10**(mag_max/20.0), 2000)
    Gcl_phases, Gcl_mags = np.meshgrid(np.radians(phases), mags)
    return closed_loop_contours(Gcl_mags, Gcl_phases)


def _new_horizontal_data_bounds(x):
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    return xmin // 360, xmax // 360


def _draw_nichols_grid(ax, label_cl_phases, cl_line_style, xlim, ylim):
    kw = {"label_cl_phases": label_cl_phases}
    if cl_line_style:
        kw["line_style"] = cl_line_style
    kw["xlim"] = xlim
    kw["ylim"] = ylim
    return nichols_grid(ax=ax, **kw)


def _draw_nichols_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    # verify if this is the fist series to be plotted
    is_it_the_first = p.series.index(s) == 0
    line_handle = _draw_line2d_helper(renderer, data)
    x_bounds = _new_horizontal_data_bounds(data[0])
    ng_handles = None
    # plot the Nichols grid only for the first series
    if s.ngrid and is_it_the_first:
        ng_handles = _draw_nichols_grid(
            p.ax, s.label_cl_phases, s.cl_line_style, p.xlim, p.ylim)
    return [line_handle, x_bounds, ng_handles]


def _update_nichols_helper(renderer, data, handles):
    line_handle = handles[0]
    x_bounds = handles[1]
    ng_handles = handles[-1]
    new_x_bounds = _new_horizontal_data_bounds(data[0])
    handles[1] = new_x_bounds
    _update_line2d_helper(renderer, data, line_handle)

    # update gridlines if new data bounds are different than before
    if any(s != t for s, t in zip(x_bounds, new_x_bounds)) and ng_handles:
        cl_mag_lines, cl_phase_lines, cl_mag_labels, cl_phase_labels, markers = ng_handles
        p, s = renderer.plot, renderer.series
        for l in cl_mag_lines:
            l.remove()
        for l in cl_phase_lines:
            l.remove()
        for l in cl_mag_labels:
            l.remove()
        for l in cl_phase_labels:
            l.remove()
        markers.remove()
        handles[-1] = _draw_nichols_grid(
            p.ax, s.label_cl_phases, s.cl_line_style, p.xlim, p.ylim)


class NicholsRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_nichols_helper: _update_nichols_helper
    }
