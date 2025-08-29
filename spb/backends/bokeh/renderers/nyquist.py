from spb.backends.base_renderer import Renderer
from spb.series import MCirclesSeries
from spb.utils import unwrap
from sympy.external import import_module
import warnings


def _compute_arrows_position(
    x, y, arrow_locs=[0.2, 0.4, 0.6, 0.8], dir=1
):
    """
    Compute the position of arrows along the coordinates x, y at
    selected locations.

    Parameters:
    -----------
    x: x-coordinate
    y: y-coordinate
    arrow_locs: list of locations where to insert arrows, % of total length
    dir: direction of the arrows. +1 along the line, -1 in opposite direction.

    Returns:
    --------
    source: a dictionary containing the data for bokeh.models.Arrow

    Notes
    -----

    Based on:
    https://stackoverflow.com/questions/26911898/

    Based on:
    https://github.com/python-control/python-control/blob/main/control/freqplot.py
    """
    np = import_module("numpy")

    # Compute the arc length along the curve
    s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    arrows = []
    x_start, y_start, x_end, y_end = [], [], [], []
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

        x_start.append(arrow_tail[0])
        y_start.append(arrow_tail[1])
        x_end.append(arrow_head[0])
        y_end.append(arrow_head[1])

    source = {
        "x_start": x_start, "x_end": x_end, "y_start": y_start, "y_end": y_end
    }
    return source


def _create_line_style(
    plot_obj, user_provided_style, default_style, name, color
):
    if user_provided_style:
        if isinstance(user_provided_style, str):
            skw = {
                "line_color": color,
                "line_dash": user_provided_style,
                "line_width": 2
            }
            style = [skw, skw]
        elif isinstance(user_provided_style, dict):
            pskw = plot_obj.merge(
                {"line_color": color, "line_width": 2},
                user_provided_style)
            style = [pskw] * 2
        elif (
            isinstance(user_provided_style, (tuple, list)) and
            len(user_provided_style) == 2 and
            all(isinstance(t, str) for t in user_provided_style)
        ):
            pskw1 = {
                "line_color": color,
                "line_dash": user_provided_style[0],
                "line_width": 2
            }
            pskw2 = {
                "line_color": color,
                "line_dash": user_provided_style[1],
                "line_width": 2
            }
            style = [pskw1, pskw2]
        elif (
            isinstance(user_provided_style, (tuple, list)) and
            len(user_provided_style) == 2 and
            all(isinstance(t, dict) for t in user_provided_style)
        ):
            pskw1 = plot_obj.merge(
                {"line_color": color, "line_width": 2},
                user_provided_style[0])
            pskw2 = plot_obj.merge(
                {"line_color": color, "line_width": 2},
                user_provided_style[1])
            style = [pskw1, pskw2]
        else:
            raise ValueError(
                f"`{name}` not valid. Read the documentation "
                "to learn how to set this keyword argument.")
    else:
        if user_provided_style is not False:
            pskw1 = {
                "line_color": color,
                "line_dash": default_style[0],
                "line_width": 2
            }
            pskw2 = {
                "line_color": color,
                "line_dash": default_style[1],
                "line_width": 2
            }
            style = [pskw1, pskw2]
        else:
            style = user_provided_style
    return style


def _get_source(x, y, mcircles):
    np = import_module("numpy")
    mag = np.sqrt(x**2 + y**2)
    mdb = 20 * np.log10(mag)
    pha = np.arctan2(y, x)
    pha_deg = np.degrees(pha)
    source = {
        "x": x, "y": y, "m": mag, "mdb": mdb, "p": pha, "pd": pha_deg
    }
    if mcircles:
        cl_mag, cl_mag_db = _compute_cl_magnitudes(x, y)
        source["cl_m"] = cl_mag
        source["cl_mdb"] = cl_mag_db
    return source


def _compute_cl_magnitudes(x, y):
    """Compute closed-loop magnitudes starting from the coordinates of points
    in the complex plane.

    Notes
    -----
    Starting from M-Circles equations, sympy was used to generate the
    equations to compute closed-loop magnitudes:

    .. code-block:: python
       from sympy import *
       m = symbols("m", real=True, positive=True)
       x, y, t = symbols("x, y, t", real=True)
       e1 = x - m / (1 - m**2) * (m + cos(t))
       e2 = y - m / (1 - m**2) * sin(t)
       sol = solve([e1, e2], [m, t]) # IT TAKES A FEW MINUTES
       m_eq = [s[0] for s in sol]

    It creates 4 equations. For any arbitrary point, two equations produces
    NaN, one a positive value and another a negative value.
    I take the positive value, then adjust the Magnitude in dB.
    """
    np = import_module("numpy")
    m1 = -(np.sqrt(4*y**2 + np.sin(2*np.arctan((x**2 + x - y**2 - np.sqrt(x**4 + 2*x**3 + 2*x**2*y**2 + x**2 + 2*x*y**2 + y**4 + y**2))/(y*(2*x + 1))))**2) - np.sin(2*np.arctan((x**2 + x - y**2 - np.sqrt(x**4 + 2*x**3 + 2*x**2*y**2 + x**2 + 2*x*y**2 + y**4 + y**2))/(y*(2*x + 1)))))/(2*y)
    m2 = (np.sqrt(4*y**2 + np.sin(2*np.arctan((x**2 + x - y**2 - np.sqrt(x**4 + 2*x**3 + 2*x**2*y**2 + x**2 + 2*x*y**2 + y**4 + y**2))/(y*(2*x + 1))))**2) + np.sin(2*np.arctan((x**2 + x - y**2 - np.sqrt(x**4 + 2*x**3 + 2*x**2*y**2 + x**2 + 2*x*y**2 + y**4 + y**2))/(y*(2*x + 1)))))/(2*y)
    m3 = -(np.sqrt(4*y**2 + np.sin(2*np.arctan((x**2 + x - y**2 + np.sqrt(x**4 + 2*x**3 + 2*x**2*y**2 + x**2 + 2*x*y**2 + y**4 + y**2))/(y*(2*x + 1))))**2) - np.sin(2*np.arctan((x**2 + x - y**2 + np.sqrt(x**4 + 2*x**3 + 2*x**2*y**2 + x**2 + 2*x*y**2 + y**4 + y**2))/(y*(2*x + 1)))))/(2*y)
    m4 = (np.sqrt(4*y**2 + np.sin(2*np.arctan((x**2 + x - y**2 + np.sqrt(x**4 + 2*x**3 + 2*x**2*y**2 + x**2 + 2*x*y**2 + y**4 + y**2))/(y*(2*x + 1))))**2) + np.sin(2*np.arctan((x**2 + x - y**2 + np.sqrt(x**4 + 2*x**3 + 2*x**2*y**2 + x**2 + 2*x*y**2 + y**4 + y**2))/(y*(2*x + 1)))))/(2*y)
    m = np.abs(np.vstack([m1, m2, m3, m4]))
    m = np.nanmax(m, axis=0)
    mdb = 20 * np.log10(m)
    idx = x > -0.5
    mdb[idx] = -1 * mdb[idx]
    return m, mdb


def _draw_nyquist_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    color = next(p._cl)
    fig = p._fig

    x_reg, y_reg, x_scl, y_scl, x_inv1, y_inv1, x_inv2, y_inv2, curve_offset = data
    mcircles = any(isinstance(t, MCirclesSeries) for t in p.series)

    primary_style = ['solid', 'dashdot']
    mirror_style = ['dashed', 'dotted']
    primary_style = _create_line_style(
        p, s.primary_style, primary_style, "primary_style", color)
    mirror_style = _create_line_style(
        p, s.mirror_style, mirror_style, "mirror_style", color)
    primary_style[0]["legend_label"] = s.get_label(p.use_latex)

    # Set the arrow style
    arrow_style = p.bokeh.models.VeeHead(
            line_color=color, fill_color=color, size=10)

    secondary_line, invisible_secondary_line = None, None
    scl_primary_line, scl_secondary_line = None, None

    tooltips = [
        ("Real", "@x"), ("Imag", "@y"), ("OL Mag", "@m"), ("OL Mag [dB]", "@mdb"),
        ("OL Phase [rad]", "@p"), ("OL Phase [deg]", "@pd")
    ]
    if mcircles:
        tooltips += [("CL Mag", "@cl_m"), ("CL Mag [dB]", "@cl_mdb")]
    primary_line = fig.line(
        "x", "y", source=_get_source(x_reg, y_reg, mcircles),
        **primary_style[0])
    fig.add_tools(p.bokeh.models.HoverTool(
        tooltips=tooltips,
        renderers=[primary_line]
    ))

    if x_scl.count() >= 1 and y_scl.count() >= 1:
        scl_primary_line = fig.line(
            "x", "y", source=_get_source(
                x_scl * (1 + curve_offset),
                y_scl * (1 + curve_offset),
                mcircles
            ),
            **primary_style[1]
        )
        fig.add_tools(p.bokeh.models.HoverTool(
            tooltips=tooltips,
            renderers=[scl_primary_line]
        ))

    # Add arrows
    arrows_handles = []
    source = _compute_arrows_position(x_reg, y_reg, s._arrow_locs, dir=1)
    arrows1 = p.bokeh.models.Arrow(
        source=p.bokeh.models.ColumnDataSource(data=source),
        line_color=color, end=arrow_style)
    fig.add_layout(arrows1)
    arrows_handles.append(arrows1)
    arrows2 = None

    # Plot the mirror image
    if mirror_style is not False:
        # Plot the regular and scaled segments
        secondary_line = fig.line(
            "x", "y", source=_get_source(x_reg, -y_reg, mcircles),
            **mirror_style[0])
        fig.add_tools(p.bokeh.models.HoverTool(
            tooltips=tooltips,
            renderers=[secondary_line]
        ))
        if x_scl.count() >= 1 and y_scl.count() >= 1:
            scl_secondary_line = fig.line(
                "x", "y", source=_get_source(
                    x_scl * (1 - curve_offset),
                    -y_scl * (1 - curve_offset),
                    mcircles
                ),
                **mirror_style[1]
            )
            fig.add_tools(p.bokeh.models.HoverTool(
                tooltips=tooltips,
                renderers=[scl_secondary_line]
            ))

        # Add arrows
        source = _compute_arrows_position(x_reg, -y_reg, s._arrow_locs, dir=-1)
        arrows2 = p.bokeh.models.Arrow(
            source=p.bokeh.models.ColumnDataSource(data=source),
            line_color=color, end=arrow_style)
        fig.add_layout(arrows2)
        arrows_handles.append(arrows2)

    # Mark the start of the curve
    start_marker_handle = None
    if s.start_marker:
        smkw = {
            "marker": "o", "size": 8,
            "line_color": color, "fill_color": color
        }
        if isinstance(s.start_marker, str):
            smkw["marker"] = s.start_marker
        elif isinstance(s.start_marker, dict):
            smkw = p.merge({}, s.start_marker)
        start_marker_handle = fig.scatter(
            "x", "y", source={"x": [x_reg[0]], "y": [y_reg[0]]}, **smkw)

    # bind all handles' visibility to the primary_line's visibility
    callback = p.bokeh.models.CustomJS(
        args=dict(
            primary_line=primary_line, scl_primary_line=scl_primary_line,
            secondary_line=secondary_line, scl_secondary_line=scl_secondary_line,
            start_marker_handle=start_marker_handle,
            arrows1=arrows1, arrows2=arrows2
        ),
        code="""
        secondary_line.visible = primary_line.visible;
        arrows1.visible = primary_line.visible;
        if (scl_primary_line !== null) {
            scl_primary_line.visible = primary_line.visible;
        }
        if (scl_secondary_line !== null) {
            scl_secondary_line.visible = primary_line.visible;
        }
        if (arrows2 !== null) {
            arrows2.visible = primary_line.visible;
        }
        if (start_marker_handle !== null) {
            start_marker_handle.visible = primary_line.visible;
        }
        """
    )
    primary_line.js_on_change("visible", callback)

    handles = [
        primary_line, scl_primary_line,
        secondary_line, scl_secondary_line,
        start_marker_handle, arrows_handles
    ]
    return handles


def _update_nyquist_helper(renderer, data, handles):
    mpl = import_module("matplotlib")
    p, s = renderer.plot, renderer.series
    primary_line, scl_primary_line = handles[:2]
    secondary_line, scl_secondary_line = handles[2:4]
    start_marker_handle, arrows_handles = handles[4:]
    mcircles = any(isinstance(t, MCirclesSeries) for t in p.series)

    x_reg, y_reg, x_scl, y_scl, x_inv1, y_inv1, x_inv2, y_inv2, curve_offset = data


    primary_line.data_source.data.update(_get_source(x_reg, y_reg, mcircles))
    if scl_primary_line:
        scl_primary_line.data_source.data.update(_get_source(
            x_scl * (1 + curve_offset),
            y_scl * (1 + curve_offset),
            mcircles)
        )

    source = _compute_arrows_position(x_reg, y_reg, s._arrow_locs, dir=1)
    arrows_handles[0].source.data.update(source)

    if secondary_line:
        secondary_line.data_source.data.update(
            _get_source(x_reg, -y_reg, mcircles))
        if scl_secondary_line:
            scl_secondary_line.data_source.data.update(_get_source(
                x_scl * (1 - curve_offset),
                -y_scl * (1 - curve_offset),
                mcircles)
            )

        source = _compute_arrows_position(x_reg, -y_reg, s._arrow_locs, dir=1)
        arrows_handles[1].source.data.update(source)

    if start_marker_handle:
        start_marker_handle.data_source.data.update(
            {"x": [x_reg[0]], "y": [y_reg[0]]})


# TODO: sometimes it raises strange
# errors when parametric interactive plots are created...
# For example, tf = 1 / (s + p) with p=0.06, some line disappears...

class NyquistRenderer(Renderer):
    draw_update_map = {
        _draw_nyquist_helper: _update_nyquist_helper
    }
