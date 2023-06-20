from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.utils import unwrap

import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.text as text
import warnings


# taken from matplotlib
def _get_label_width(ax, val, fs):
    """Return the width of the label, in pixels."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    return (
        text.Text(0, 0,
            val,
            figure=fig,
            size=fs,
            fontproperties=font_manager.FontProperties())
        .get_window_extent(renderer).width)


def _draw_m_circles(ax, xlim=None, ylim=None, x=None, y=None,
    clabels_to_top=True):
    # TODO: maybe it's easier and cleaner to plot lines instead of contours,
    # especially the labels positioning. And it would be faster too on
    # interactive updates.

    # get limits for computing m-circles
    axis_limits = {k: None for k in ["xmin", "xmax", "ymin", "ymax"]}
    if xlim:
        axis_limits["xmin"], axis_limits["xmax"] = xlim
    if ylim:
        axis_limits["ymin"], axis_limits["ymax"] = ylim

    mx, my = ax.margins()
    if axis_limits["xmin"] is None:
        if x is not None:
            _min = min(x.min(), -1) # there is a marker at (-1, 0)
            dx = x.max() - _min
            axis_limits["xmin"] = _min - mx * dx
            axis_limits["xmax"] = x.max() + mx * dx
        else:
            axis_limits["xmin"] = -1
            axis_limits["xmax"] = 1
    if axis_limits["ymin"] is None:
        if y is not None:
            y = np.concatenate([y, -np.flip(y)])
            dy = y.max() - y.min()
            axis_limits["ymin"] = y.min() - my * dy
            axis_limits["ymax"] = y.max() + my * dy
        else:
            axis_limits["ymin"] = -1
            axis_limits["ymax"] = 1

    dbs = [-20, -10, -6, -4, -2, 0, 2, 4, 6, 10, 20]
    magnitudes = [10**(t / 20) for t in dbs]
    magnitudes[5] = -0.5
    labels = [str(t) + " dB" for t in dbs]
    # M-circles when M != 0 dB
    f1 = lambda x, y, M: (x - M**2 / (1 - M**2))**2 + y**2 - (M / (1 - M**2))**2
    # M-circles when M == 0 dB
    f2 = lambda x, y, M: x + 0.5

    dx = axis_limits["xmax"] - axis_limits["xmin"]
    dy = axis_limits["ymax"] - axis_limits["ymin"]
    n1 = 200
    n2 = int(dy / dx * n1)
    x, y = np.mgrid[
        axis_limits["xmin"]:axis_limits["xmax"]:n1*1j,
        axis_limits["ymin"]:axis_limits["ymax"]:n2*1j]

    contours = []
    clabels = []
    for m, l in zip(magnitudes, labels):
        f = f1 if not np.isclose(m, -0.5) else f2
        c = ax.contour(x, y, f(x, y, m), levels=[0.0],
            linestyles=":", colors="darkgray",
            linewidths=1)
        contours.extend(c.collections)

        locations = False
        if clabels_to_top:
            # computes position for contour labels.
            # Circles intersecting the bounding box will show a label
            # somewhere near the top border.
            locations = []
            for coll in c.collections:
                for path in coll.get_paths():
                    vert = path.vertices.copy()
                    # exclude regions too close to top and bottom
                    idx1 = vert[:, 1] > axis_limits["ymax"] - my * dy
                    idx2 = vert[:, 1] < axis_limits["ymin"] + my * dy
                    vert[(idx1 | idx2)] = np.nan
                    # exclude regions too close to left and right
                    # TODO: it would be nice to use label size... but, how to
                    # convert from pixel size to data coordinates?
                    idx1 = vert[:, 0] > axis_limits["xmax"] - mx * dx
                    idx2 = vert[:, 0] < axis_limits["xmin"] +  mx * dx
                    vert[(idx1 | idx2)] = np.nan
                    if not np.isnan(vert).all():
                        idx = np.nanargmax(vert[:, 1])
                        locations.append(vert[idx, :])

        texts = ax.clabel(c, fontsize=9, colors="k", manual=locations)
        clabels.extend(texts)
        for t in texts:
            t.set_rotation(0)
            t.set_text(l)

    return contours, clabels


def _draw_arrows_helper(
        ax, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
        arrowstyle='-|>', dir=1):
    """
    Add arrows along the coordinates x, y at selected locations.

    Parameters:
    -----------
    ax: Axes object as returned by axes command (or gca)
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    dir: direction of the arrows. +1 along the line, -1 in opposite direction.

    Returns:
    --------
    arrows: list of arrows

    Based on https://stackoverflow.com/questions/26911898/
    Based on https://github.com/python-control/python-control/blob/main/control/freqplot.py
    """
    if not isinstance(line, mpl.lines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    # Compute the arc length along the curve
    s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    arrows = []
    for loc in arrow_locs:
        n = np.searchsorted(s, s[-1] * loc)

        # Figure out what direction to paint the arrow
        if dir == 1:
            arrow_tail = (x[n], y[n])
            arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        elif dir == -1:
            # Orient the arrow in the other direction on the segment
            arrow_tail = (x[n + 1], y[n + 1])
            arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        else:
            raise ValueError("unknown value for keyword 'dir'")

        p = mpl.patches.FancyArrowPatch(
            arrow_tail, arrow_head, lw=0, **arrow_kw)
        ax.add_patch(p)
        arrows.append(p)
    return arrows

#
# Function to compute Nyquist curve offsets
#
# This function computes a smoothly varying offset that starts and ends at
# zero at the ends of a scaled segment.
#
def _compute_curve_offset(resp, mask, max_offset):
    # from https://github.com/python-control/python-control/blob/main/control/freqplot.py

    # Compute the arc length along the curve
    s_curve = np.cumsum(
        np.sqrt(np.diff(resp.real) ** 2 + np.diff(resp.imag) ** 2))

    # Initialize the offset
    offset = np.zeros(resp.size)
    arclen = np.zeros(resp.size)

    # Walk through the response and keep track of each continous component
    i, nsegs = 0, 0
    while i < resp.size:
        # Skip the regular segment
        while i < resp.size and mask[i]:
            i += 1              # Increment the counter
            if i == resp.size:
                break
            # Keep track of the arclength
            arclen[i] = arclen[i-1] + np.abs(resp[i] - resp[i-1])

        nsegs += 0.5
        if i == resp.size:
            break

        # Save the starting offset of this segment
        seg_start = i

        # Walk through the scaled segment
        while i < resp.size and not mask[i]:
            i += 1
            if i == resp.size:  # See if we are done with this segment
                break
            # Keep track of the arclength
            arclen[i] = arclen[i-1] + np.abs(resp[i] - resp[i-1])

        nsegs += 0.5
        if i == resp.size:
            break

        # Save the ending offset of this segment
        seg_end = i

        # Now compute the scaling for this segment
        s_segment = arclen[seg_end-1] - arclen[seg_start]
        offset[seg_start:seg_end] = max_offset * s_segment/s_curve[-1] * \
            np.sin(np.pi * (arclen[seg_start:seg_end]
                            - arclen[seg_start])/s_segment)

    return offset


def _process_data_helper(data, max_curve_magnitude, max_curve_offset,
    encirclement_threshold, indent_direction, tf_poles, tf_cl_poles):
    resp = data[0] + 1j * data[1]
    splane_contour = data[2]

    # from https://github.com/python-control/python-control/blob/main/control/freqplot.py

    # Compute CW encirclements of -1 by integrating the (unwrapped) angle
    phase = -unwrap(np.angle(resp + 1))
    encirclements = np.sum(np.diff(phase)) / np.pi
    count = int(np.round(encirclements, 0))

    # Let the user know if the count might not make sense
    if abs(encirclements - count) > encirclement_threshold:
        warnings.warn(
            "number of encirclements was a non-integer value; this can"
            " happen is contour is not closed, possibly based on a"
            " frequency range that does not include zero.")

    #
    # Make sure that the enciriclements match the Nyquist criterion
    #
    # If the user specifies the frequency points to use, it is possible
    # to miss enciriclements, so we check here to make sure that the
    # Nyquist criterion is actually satisfied.
    #
    # Count the number of open/closed loop RHP poles
    if indent_direction == 'right':
        P = (tf_poles.real > 0).sum()
    else:
        P = (tf_poles.real >= 0).sum()
    Z = (tf_cl_poles.real >= 0).sum()


    # Check to make sure the results make sense; warn if not
    if Z != count + P:
        warnings.warn(
            "number of encirclements does not match Nyquist criterion;"
            " check frequency range and indent radius/direction",
            UserWarning, stacklevel=2)
    elif indent_direction == 'none' and any(sys.poles().real == 0):
        warnings.warn(
            "system has pure imaginary poles but indentation is"
            " turned off; results may be meaningless",
            RuntimeWarning, stacklevel=2)

    # Find the different portions of the curve (with scaled pts marked)
    reg_mask = np.logical_or(
        np.abs(resp) > max_curve_magnitude,
        splane_contour.real != 0)
    # reg_mask = np.logical_or(
    #     np.abs(resp.real) > max_curve_magnitude,
    #     np.abs(resp.imag) > max_curve_magnitude)

    scale_mask = ~reg_mask \
        & np.concatenate((~reg_mask[1:], ~reg_mask[-1:])) \
        & np.concatenate((~reg_mask[0:1], ~reg_mask[:-1]))

    # Rescale the points with large magnitude
    rescale = np.logical_and(
        reg_mask, abs(resp) > max_curve_magnitude)
    resp[rescale] *= max_curve_magnitude / abs(resp[rescale])

    # the regular portions of the curve
    x_reg = np.ma.masked_where(reg_mask, resp.real)
    y_reg = np.ma.masked_where(reg_mask, resp.imag)


    # Figure out how much to offset the curve: the offset goes from
    # zero at the start of the scaled section to max_curve_offset as
    # we move along the curve
    curve_offset = _compute_curve_offset(
        resp, scale_mask, max_curve_offset)

    # the scaled sections of the curve
    x_scl = np.ma.masked_where(scale_mask, resp.real)
    y_scl = np.ma.masked_where(scale_mask, resp.imag)

    # the primary curve (invisible) for setting arrows
    x_inv1, y_inv1 = resp.real.copy(), resp.imag.copy()
    x_inv1[reg_mask] *= (1 + curve_offset[reg_mask])
    y_inv1[reg_mask] *= (1 + curve_offset[reg_mask])

    # Add the arrows to the mirror image (on top of an invisible contour)
    x_inv2, y_inv2 = resp.real.copy(), resp.imag.copy()
    x_inv2[reg_mask] *= (1 - curve_offset[reg_mask])
    y_inv2[reg_mask] *= (1 - curve_offset[reg_mask])

    return x_reg, y_reg, x_scl, y_scl, x_inv1, y_inv1, x_inv2, y_inv2, curve_offset


def _create_line_style(plot_obj, user_provided_style, default_style, name, color):
    if user_provided_style:
        if isinstance(user_provided_style, str):
            skw = {"color": color, "linestyle": user_provided_style}
            style = [skw, skw]
        elif isinstance(user_provided_style, dict):
            pskw = plot_obj.merge({"color": color}, user_provided_style)
            style = [pskw] * 2
        elif (isinstance(user_provided_style, (tuple, list)) and
            len(user_provided_style) == 2 and
            all(isinstance(t, str) for t in user_provided_style)):
            pskw1 = {"color": color, "linestyle": user_provided_style[0]}
            pskw2 = {"color": color, "linestyle": user_provided_style[1]}
            style = [pskw1, pskw2]
        elif (isinstance(user_provided_style, (tuple, list)) and
            len(user_provided_style) == 2 and
            all(isinstance(t, dict) for t in user_provided_style)):
            pskw1 = plot_obj.merge({"color": color}, user_provided_style[0])
            pskw2 = plot_obj.merge({"color": color}, user_provided_style[1])
            style = [pskw1, pskw2]
        else:
            raise ValueError(f"`{name}` not valid. Read the documentation "
                "to learn how to set this keyword argument.")
    else:
        if user_provided_style is not False:
            pskw1 = {"color": color, "linestyle": default_style[0]}
            pskw2 = {"color": color, "linestyle": default_style[1]}
            style = [pskw1, pskw2]
        else:
            style = user_provided_style
    return style


def _draw_nyquist_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    color = next(p._cl)
    ax = p.ax
    np = p.np

    # from https://github.com/python-control/python-control/blob/main/control/freqplot.py

    m_handles = None
    if s.m_circles:
        m_handles = _draw_m_circles(p.ax, p.xlim, p.ylim,
            data[0], data[1], s.clabels_to_top)

    new_data = _process_data_helper(
        data, s.max_curve_magnitude, s.max_curve_offset,
        s.encirclement_threshold, s.indent_direction,
        s._poles, s._poles_cl)
    x_reg, y_reg, x_scl, y_scl, x_inv1, y_inv1, x_inv2, y_inv2, curve_offset = new_data

    primary_style = ['-', '-.']
    mirror_style = ['--', ':']
    primary_style = _create_line_style(
        p, s.primary_style, primary_style, "primary_style", color)
    mirror_style = _create_line_style(
        p, s.mirror_style, mirror_style, "mirror_style", color)
    primary_style[0]["label"] = s.get_label(p._use_latex)

    # Set the arrow style
    arrow_style = mpl.patches.ArrowStyle('simple', head_width=6, head_length=6)

    secondary_line, invisible_secondary_line = None, None
    scl_primary_line, scl_secondary_line = None, None

    primary_line, = ax.plot(x_reg, y_reg, **primary_style[0])

    if x_scl.count() >= 1 and y_scl.count() >= 1:
        scl_primary_line, = ax.plot(
            x_scl * (1 + curve_offset),
            y_scl * (1 + curve_offset),
            **primary_style[1])

    # Plot the primary curve (invisible) for setting arrows
    invisible_primary_line, = ax.plot(x_inv1, y_inv1, linestyle='None', color=color)

    # Add arrows
    arrows_handles = []
    arrows1 = _draw_arrows_helper(
        ax, invisible_primary_line, s.arrows_loc, arrowstyle=arrow_style, dir=1)
    arrows_handles.extend(arrows1)

    # Plot the mirror image
    if mirror_style is not False:
        # Plot the regular and scaled segments
        secondary_line, = ax.plot(x_reg, -y_reg, **mirror_style[0])
        if x_scl.count() >= 1 and y_scl.count() >= 1:
            scl_secondary_line, = ax.plot(
                x_scl * (1 - curve_offset),
                -y_scl * (1 - curve_offset),
                **mirror_style[1])

        # Add the arrows (on top of an invisible contour)
        invisible_secondary_line, = ax.plot(x_inv2, -y_inv2, linestyle='None', color=color)
        arrows2 = _draw_arrows_helper(
            ax, invisible_secondary_line, s.arrows_loc, arrowstyle=arrow_style, dir=-1)
        arrows_handles.extend(arrows2)

    # Mark the start of the curve
    start_marker_handle = None
    if s.start_marker:
        smkw = {"marker": "o", "markersize": 4, "color": color}
        if isinstance(s.start_marker, str):
            smkw["marker"] = s.start_marker
        elif isinstance(s.start_marker, dict):
            smkw = p.merge({}, s.start_marker)
        start_marker_handle, = ax.plot(x_reg[0], y_reg[0], **smkw)

    # Mark the -1 point
    ax.plot([-1], [0], 'r+')


    handles = [
        m_handles,
        primary_line, scl_primary_line, invisible_primary_line,
        secondary_line, scl_secondary_line, invisible_secondary_line,
        start_marker_handle, arrows_handles
    ]
    return handles


def _update_nyquist_helper(renderer, data, handles):
    # TODO: remove and recompute m-circles if prange is used.

    p, s = renderer.plot, renderer.series
    ax = p.ax
    primary_line, scl_primary_line, invisible_primary_line = handles[1:4]
    secondary_line, scl_secondary_line, invisible_secondary_line = handles[4:7]
    start_marker_handle = handles[-2]
    arrows_handles = handles[-1]
    arrow_style = mpl.patches.ArrowStyle('simple', head_width=6, head_length=6)

    new_data = _process_data_helper(
        data, s.max_curve_magnitude, s.max_curve_offset,
        s.encirclement_threshold, s.indent_direction,
        s._poles, s._poles_cl)
    x_reg, y_reg, x_scl, y_scl, x_inv1, y_inv1, x_inv2, y_inv2, curve_offset = new_data

    # remove previous arrows
    for a in arrows_handles:
        a.remove()
    arrows_handles.clear()

    primary_line.set_data(x_reg, y_reg)
    if scl_primary_line:
        scl_primary_line.set_data(
            x_scl * (1 + curve_offset),
            y_scl * (1 + curve_offset))
    invisible_primary_line.set_data(x_inv1, y_inv1)
    arrows1 = _draw_arrows_helper(
        ax, invisible_primary_line, s.arrows_loc, arrowstyle=arrow_style, dir=1)
    arrows_handles.extend(arrows1)

    if secondary_line:
        secondary_line.set_data(x_reg, -y_reg)
        if scl_secondary_line:
            scl_secondary_line.set_data(
                x_scl * (1 - curve_offset),
                -y_scl * (1 - curve_offset))
        invisible_secondary_line.set_data(x_inv2, -y_inv2)
        arrows2 = _draw_arrows_helper(
            ax, invisible_secondary_line, s.arrows_loc, arrowstyle=arrow_style, dir=-1)
        arrows_handles.extend(arrows2)

    if start_marker_handle:
        start_marker_handle.set_data([x_reg[0]], [y_reg[0]])


# TODO: sometimes it raises strange
# errors when parametric interactive plots are created...
# For example, tf = 1 / (s + p) with p=0.06, some line disappears...

class NyquistRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_nyquist_helper: _update_nyquist_helper
    }
