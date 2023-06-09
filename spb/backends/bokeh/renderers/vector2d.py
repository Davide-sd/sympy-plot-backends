from spb.backends.base_renderer import Renderer
from spb.backends.utils import get_seeds_points
from sympy.external import import_module
import warnings


def compute_streamlines(x, y, u, v, density=1.0):
    """Return streamlines of a vector flow.

    * x and y are 1d arrays defining an *evenly spaced* grid.
    * u and v are 2d arrays (shape [y,x]) giving velocities.
    * density controls the closeness of the streamlines.

    Credit: https://docs.bokeh.org/en/latest/docs/gallery/quiver.html
    """
    np = import_module('numpy')

    ## Set up some constants - size of the grid used.
    NGX = len(x)
    NGY = len(y)

    ## Constants used to convert between grid index coords and user coords.
    DX = x[1] - x[0]
    DY = y[1] - y[0]
    XOFF = x[0]
    YOFF = y[0]

    ## Now rescale velocity onto axes-coordinates
    u = u / (x[-1] - x[0])
    v = v / (y[-1] - y[0])
    speed = np.sqrt(u * u + v * v)
    ## s (path length) will now be in axes-coordinates, but we must
    ## rescale u for integrations.
    u *= NGX
    v *= NGY
    ## Now u and v in grid-coordinates.

    NBX = int(30 * density)
    NBY = int(30 * density)

    blank = np.zeros((NBY, NBX))

    bx_spacing = NGX / float(NBX - 1)
    by_spacing = NGY / float(NBY - 1)

    def blank_pos(xi, yi):
        return int((xi / bx_spacing) + 0.5), int((yi / by_spacing) + 0.5)

    def value_at(a, xi, yi):
        if type(xi) == np.ndarray:
            x = xi.astype(int)
            y = yi.astype(int)
        else:
            x = int(xi)
            y = int(yi)
        a00 = a[y, x]
        a01 = a[y, x + 1]
        a10 = a[y + 1, x]
        a11 = a[y + 1, x + 1]
        xt = xi - x
        yt = yi - y
        a0 = a00 * (1 - xt) + a01 * xt
        a1 = a10 * (1 - xt) + a11 * xt
        return a0 * (1 - yt) + a1 * yt

    def rk4_integrate(x0, y0):
        ## This function does RK4 forward and back trajectories from
        ## the initial conditions, with the odd 'blank array'
        ## termination conditions. TODO tidy the integration loops.

        def f(xi, yi):
            dt_ds = 1.0 / value_at(speed, xi, yi)
            ui = value_at(u, xi, yi)
            vi = value_at(v, xi, yi)
            return ui * dt_ds, vi * dt_ds

        def g(xi, yi):
            dt_ds = 1.0 / value_at(speed, xi, yi)
            ui = value_at(u, xi, yi)
            vi = value_at(v, xi, yi)
            return -ui * dt_ds, -vi * dt_ds

        check = lambda xi, yi: xi >= 0 and xi < NGX - 1 and yi >= 0 and yi < NGY - 1

        bx_changes = []
        by_changes = []

        ## Integrator function
        def rk4(x0, y0, f):
            ds = 0.01  # min(1./NGX, 1./NGY, 0.01)
            stotal = 0
            xi = x0
            yi = y0
            xb, yb = blank_pos(xi, yi)
            xf_traj = []
            yf_traj = []
            while check(xi, yi):
                # Time step. First save the point.
                xf_traj.append(xi)
                yf_traj.append(yi)
                # Next, advance one using RK4
                try:
                    k1x, k1y = f(xi, yi)
                    k2x, k2y = f(xi + 0.5 * ds * k1x, yi + 0.5 * ds * k1y)
                    k3x, k3y = f(xi + 0.5 * ds * k2x, yi + 0.5 * ds * k2y)
                    k4x, k4y = f(xi + ds * k3x, yi + ds * k3y)
                except IndexError:
                    # Out of the domain on one of the intermediate steps
                    break
                xi += ds * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
                yi += ds * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
                # Final position might be out of the domain
                if not check(xi, yi):
                    break
                stotal += ds
                # Next, if s gets to thres, check blank.
                new_xb, new_yb = blank_pos(xi, yi)
                if new_xb != xb or new_yb != yb:
                    # New square, so check and colour. Quit if required.
                    if blank[new_yb, new_xb] == 0:
                        blank[new_yb, new_xb] = 1
                        bx_changes.append(new_xb)
                        by_changes.append(new_yb)
                        xb = new_xb
                        yb = new_yb
                    else:
                        break
                if stotal > 2:
                    break
            return stotal, xf_traj, yf_traj

        integrator = rk4

        sf, xf_traj, yf_traj = integrator(x0, y0, f)
        sb, xb_traj, yb_traj = integrator(x0, y0, g)
        stotal = sf + sb
        x_traj = xb_traj[::-1] + xf_traj[1:]
        y_traj = yb_traj[::-1] + yf_traj[1:]

        ## Tests to check length of traj. Remember, s in units of axes.
        if len(x_traj) < 1:
            return None
        if stotal > 0.2:
            initxb, inityb = blank_pos(x0, y0)
            blank[inityb, initxb] = 1
            return x_traj, y_traj
        else:
            for xb, yb in zip(bx_changes, by_changes):
                blank[yb, xb] = 0
            return None

    ## A quick function for integrating trajectories if blank==0.
    trajectories = []

    def traj(xb, yb):
        if xb < 0 or xb >= NBX or yb < 0 or yb >= NBY:
            return
        if blank[yb, xb] == 0:
            t = rk4_integrate(xb * bx_spacing, yb * by_spacing)
            if t is not None:
                trajectories.append(t)

    ## Now we build up the trajectory set. I've found it best to look
    ## for blank==0 along the edges first, and work inwards.
    for indent in range((max(NBX, NBY)) // 2):
        for xi in range(max(NBX, NBY) - 2 * indent):
            traj(xi + indent, indent)
            traj(xi + indent, NBY - 1 - indent)
            traj(indent, xi + indent)
            traj(NBX - 1 - indent, xi + indent)

    xs = [np.array(t[0]) * DX + XOFF for t in trajectories]
    ys = [np.array(t[1]) * DY + YOFF for t in trajectories]

    return xs, ys


def _draw_vector2d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    handle = []

    if s.is_streamlines:
        x, y, u, v = data
        sqk = dict(color=next(p._cl), line_width=2, line_alpha=0.8)
        stream_kw = s.rendering_kw.copy()
        density = stream_kw.pop("density", 2)
        kw = p.merge({}, sqk, stream_kw)
        xs, ys = compute_streamlines(
            x[0, :], y[:, 0], u, v, density=density)
        handle.append(p._fig.multi_line(xs, ys, **kw))
    else:
        x, y, u, v = data
        mag = np.sqrt(u**2 + v**2)
        u0, v0 = [t.copy() for t in [u, v]]
        if s.normalize:
            u, v = [t / mag for t in [u, v]]
        data, quiver_kw = p._get_quivers_data(x, y, u, v,
            **s.rendering_kw.copy())
        color_val = mag
        if s.color_func is not None:
            color_val = s.eval_color_func(x, y, u0, v0)
        color_val = np.tile(color_val.flatten(), 3)
        data["color_val"] = color_val

        color_mapper = p.bokeh.models.LinearColorMapper(
            palette=next(p._cm),
            low=min(color_val), high=max(color_val))
        # don't use color map if a scalar field is visible or if
        # use_cm=False
        line_color = (
            {"field": "color_val", "transform": color_mapper}
            if ((not s.use_quiver_solid_color) and s.use_cm)
            else next(p._cl)
        )
        source = p.bokeh.models.ColumnDataSource(data=data)
        qkw = dict(line_color=line_color, line_width=1,
            name=s.get_label(p._use_latex))
        kw = p.merge({}, qkw, quiver_kw)
        glyph = p.bokeh.models.Segment(
            x0="x0", y0="y0", x1="x1", y1="y1", **kw)
        handle.append(p._fig.add_glyph(source, glyph))
        if isinstance(line_color, dict) and s.colorbar:
            colorbar = p.bokeh.models.ColorBar(
                color_mapper=color_mapper, width=8,
                title=s.get_label(p._use_latex))
            p._fig.add_layout(colorbar, "right")
            handle.append(colorbar)
            
    return handle


def _update_vector2d_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    np = p.np
    
    x, y, u, v = data
    if s.is_streamlines:
        density = s.rendering_kw.copy().pop("density", 2)
        xs, ys = compute_streamlines(
            x[0, :], y[:, 0], u, v, density=density
        )
        handle[0].data_source.data.update({"xs": xs, "ys": ys})
    else:
        quiver_kw = s.rendering_kw.copy()
        mag = np.sqrt(u**2 + v**2)
        u0, v0 = [t.copy() for t in [u, v]]
        if s.normalize:
            u, v = [t / mag for t in [u, v]]
        data, quiver_kw = p._get_quivers_data(
            x, y, u, v, **quiver_kw
        )
        color_val = mag
        if s.color_func is not None:
            color_val = s.eval_color_func(x, y, u0, v0)
        color_val = np.tile(color_val.flatten(), 3)
        data["color_val"] = color_val
        handle[0].data_source.data.update(data)

        line_color = handle[0].glyph.line_color
        if (not s.use_quiver_solid_color) and s.use_cm:
            # update the colorbar
            cmap = line_color["transform"].palette
            color_val = data["color_val"]
            color_mapper = p.bokeh.models.LinearColorMapper(
                palette=cmap, low=min(color_val),
                high=max(color_val))
            line_color = quiver_kw.get(
                "line_color",
                {
                    "field": "color_val",
                    "transform": color_mapper
                },
            )
            handle[0].glyph.line_color = line_color
            if s.colorbar:
                cb = handle[1]
                cb.color_mapper.update(
                    low=min(color_val), high=max(color_val))



class Vector2DRenderer(Renderer):
    draw_update_map = {
        _draw_vector2d_helper: _update_vector2d_helper
    }
