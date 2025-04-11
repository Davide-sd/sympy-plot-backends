from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from packaging import version


def _matplotlib_list(interval_list):
    """
    Returns lists for matplotlib `fill` command from a list of bounding
    rectangular intervals
    """
    xlist = []
    ylist = []
    if len(interval_list):
        for intervals in interval_list:
            intervalx = intervals[0]
            intervaly = intervals[1]
            xlist.extend(
                [
                    intervalx.start, intervalx.start,
                    intervalx.end, intervalx.end, None
                ]
            )
            ylist.extend(
                [
                    intervaly.start, intervaly.end,
                    intervaly.end, intervaly.start, None
                ]
            )
    else:
        # XXX Ugly hack. Matplotlib does not accept empty lists for `fill`
        xlist.extend([None, None, None, None])
        ylist.extend([None, None, None, None])
    return xlist, ylist


def _draw_implicit2d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    color = (
        next(p._cl) if (s.color is None) or isinstance(s.color, bool)
        else s.color
    )
    if len(data) == 2:
        # interval math plotting
        x, y = _matplotlib_list(data[0])
        fkw = {"color": color, "edgecolor": "None"}
        kw = p.merge({}, fkw, s.rendering_kw)
        c = p._ax.fill(x, y, **kw)
        proxy_artist = p.Rectangle(
            (0, 0), 1, 1,
            color=kw["color"], label=s.get_label(p._use_latex)
        )
    else:
        # use contourf or contour depending on whether it is
        # an inequality or equality.
        xarray, yarray, zarray, plot_type = data
        if plot_type == "contour":
            colormap = p.ListedColormap([color, color])
            ckw = dict(cmap=colormap)
            kw = p.merge({}, ckw, s.rendering_kw)
            c = p._ax.contour(
                xarray, yarray, zarray, [0.0], **kw)
            proxy_artist = p.Line2D(
                [], [],
                color=color, label=s.get_label(p._use_latex)
            )
        else:
            colormap = p.ListedColormap(["#ffffff00", color])
            ckw = dict(
                cmap=colormap, levels=[-1e-15, 0, 1e-15],
                extend="both"
            )
            kw = p.merge({}, ckw, s.rendering_kw)
            c = p._ax.contourf(xarray, yarray, zarray, **kw)
            proxy_artist = p.Rectangle(
                (0, 0), 1, 1,
                color=color, label=s.get_label(p._use_latex)
            )

    if s.show_in_legend:
        p._legend_handles.append(proxy_artist)

    return [c, kw]


def _update_implicit2d_helper(renderer, data, handle):
    p = renderer.plot
    current_version = version.parse(p.matplotlib.__version__)
    v_3_10_0 = version.parse("3.10.0")

    if len(data) == 2:
        raise NotImplementedError
    else:
        if current_version < v_3_10_0:
            for c in handle[0].collections:
                c.remove()
        else:
            # NOTE: API changed with Matplotlib 3.10.0
            # https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.10.0.html#numdecs-parameter-and-attribute-of-loglocator
            handle[0].remove()
        xx, yy, zz, plot_type = data
        kw = handle[1]
        if plot_type == "contour":
            handle[0] = p._ax.contour(xx, yy, zz, [0.0], **kw)
        else:
            handle[0] = p._ax.contourf(xx, yy, zz, **kw)


class Implicit2DRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_implicit2d_helper: _update_implicit2d_helper
    }
