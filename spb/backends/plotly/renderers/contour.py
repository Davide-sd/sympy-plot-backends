from spb.backends.base_renderer import Renderer


def _draw_contour_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    if s.is_polar:
        raise NotImplementedError()
    xx, yy, zz = data
    xx = xx[0, :]
    yy = yy[:, 0]
    ckw = dict(
        contours=dict(
            coloring=None if s.is_filled else "lines",
            showlabels=True if (not s.is_filled) and s.show_clabels else False,
        ),
        colorscale=next(p._cm),
        colorbar=p._create_colorbar(
            s.get_label(p._use_latex), p._show_2D_vectors),
        showscale=s.is_filled and s.colorbar,
        zmin=zz.min(), zmax=zz.max()
    )
    kw = p.merge({}, ckw, s.rendering_kw)
    handle = p.go.Contour(x=xx, y=yy, z=zz, **kw)
    p._fig.add_trace(handle)
    return len(p._fig.data) - 1


def _update_contour_helper(renderer, data, idx):
    p = renderer.plot
    handle = p.fig.data[idx]

    xx, yy, zz = data
    handle["x"] = xx[0, :]
    handle["y"] = yy[:, 0]
    handle["z"] = zz
    handle["zmin"] = zz.min()
    handle["zmax"] = zz.max()


class ContourRenderer(Renderer):
    draw_update_map = {
        _draw_contour_helper: _update_contour_helper
    }
