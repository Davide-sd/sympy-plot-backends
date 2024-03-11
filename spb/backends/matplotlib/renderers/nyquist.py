from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.utils import unwrap

import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.text as text
import warnings


def _draw_arrows_helper(
    ax, line, arrow_locs=[0.2, 0.4, 0.6, 0.8], arrowstyle='-|>', dir=1
):
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

    Notes
    -----

    Based on:
    https://stackoverflow.com/questions/26911898/

    Based on:
    https://github.com/python-control/python-control/blob/main/control/freqplot.py
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


def _create_line_style(
    plot_obj, user_provided_style, default_style, name, color
):
    if user_provided_style:
        if isinstance(user_provided_style, str):
            skw = {"color": color, "linestyle": user_provided_style}
            style = [skw, skw]
        elif isinstance(user_provided_style, dict):
            pskw = plot_obj.merge({"color": color}, user_provided_style)
            style = [pskw] * 2
        elif (
            isinstance(user_provided_style, (tuple, list)) and
            len(user_provided_style) == 2 and
            all(isinstance(t, str) for t in user_provided_style)
        ):
            pskw1 = {"color": color, "linestyle": user_provided_style[0]}
            pskw2 = {"color": color, "linestyle": user_provided_style[1]}
            style = [pskw1, pskw2]
        elif (
            isinstance(user_provided_style, (tuple, list)) and
            len(user_provided_style) == 2 and
            all(isinstance(t, dict) for t in user_provided_style)
        ):
            pskw1 = plot_obj.merge({"color": color}, user_provided_style[0])
            pskw2 = plot_obj.merge({"color": color}, user_provided_style[1])
            style = [pskw1, pskw2]
        else:
            raise ValueError(
                f"`{name}` not valid. Read the documentation "
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

    x_reg, y_reg, x_scl, y_scl, x_inv1, y_inv1, x_inv2, y_inv2, curve_offset = data

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
            **primary_style[1]
        )

    # Plot the primary curve (invisible) for setting arrows
    invisible_primary_line, = ax.plot(
        x_inv1, y_inv1, linestyle='None', color=color)

    # Add arrows
    arrows_handles = []
    arrows1 = _draw_arrows_helper(
        ax, invisible_primary_line,
        s.arrow_locs,
        arrowstyle=arrow_style,
        dir=1
    )
    arrows_handles.extend(arrows1)

    # Plot the mirror image
    if mirror_style is not False:
        # Plot the regular and scaled segments
        secondary_line, = ax.plot(x_reg, -y_reg, **mirror_style[0])
        if x_scl.count() >= 1 and y_scl.count() >= 1:
            scl_secondary_line, = ax.plot(
                x_scl * (1 - curve_offset),
                -y_scl * (1 - curve_offset),
                **mirror_style[1]
            )

        # Add the arrows (on top of an invisible contour)
        invisible_secondary_line, = ax.plot(
            x_inv2, -y_inv2, linestyle='None', color=color)
        arrows2 = _draw_arrows_helper(
            ax,
            invisible_secondary_line,
            s.arrow_locs,
            arrowstyle=arrow_style,
            dir=-1
        )
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
        primary_line, scl_primary_line, invisible_primary_line,
        secondary_line, scl_secondary_line, invisible_secondary_line,
        start_marker_handle, arrows_handles
    ]
    return handles


def _update_nyquist_helper(renderer, data, handles):
    p, s = renderer.plot, renderer.series
    ax = p.ax
    primary_line, scl_primary_line, invisible_primary_line = handles[:3]
    secondary_line, scl_secondary_line, invisible_secondary_line = handles[3:6]
    start_marker_handle = handles[-2]
    arrows_handles = handles[-1]
    arrow_style = mpl.patches.ArrowStyle('simple', head_width=6, head_length=6)

    x_reg, y_reg, x_scl, y_scl, x_inv1, y_inv1, x_inv2, y_inv2, curve_offset = data

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
        ax,
        invisible_primary_line,
        s.arrow_locs,
        arrowstyle=arrow_style,
        dir=1
    )
    arrows_handles.extend(arrows1)

    if secondary_line:
        secondary_line.set_data(x_reg, -y_reg)
        if scl_secondary_line:
            scl_secondary_line.set_data(
                x_scl * (1 - curve_offset),
                -y_scl * (1 - curve_offset))
        invisible_secondary_line.set_data(x_inv2, -y_inv2)
        arrows2 = _draw_arrows_helper(
            ax, invisible_secondary_line,
            s.arrow_locs,
            arrowstyle=arrow_style,
            dir=-1
        )
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
