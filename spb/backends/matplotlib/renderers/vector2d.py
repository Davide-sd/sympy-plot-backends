from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer


def _draw_vector2d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    xx, yy, uu, vv = data
    mag = p.np.sqrt(uu ** 2 + vv ** 2)
    uu0, vv0 = [t.copy() for t in [uu, vv]]
    if s.normalize:
        uu, vv = [t / mag for t in [uu, vv]]
    solid_color = None
    if s.is_streamlines:
        skw = dict()
        if s.use_cm:
            color_val = mag
            if s.color_func is not None:
                color_val = s.eval_color_func(xx, yy, uu0, vv0)
            skw["cmap"] = next(p._cm)
            skw["color"] = color_val
            kw = p.merge({}, skw, s.rendering_kw)
            sp = p._ax.streamplot(xx, yy, uu, vv, **kw)
            is_cb_added = p._add_colorbar(
                sp.lines, s.get_label(p.use_latex), s.use_cm and s.colorbar)
        else:
            skw["color"] = next(p._cl)
            kw = p.merge({}, skw, s.rendering_kw)
            solid_color = kw["color"]
            sp = p._ax.streamplot(xx, yy, uu, vv, **kw)
            is_cb_added = False
        handle = [sp, kw, is_cb_added, p.fig.axes[-1]]
    else:
        qkw = dict()
        if any(s.is_contour for s in p.series):
            # NOTE:
            # When plotting and updating a vector plot
            # containing both a contour series and a quiver
            # series, because it's not possible to update
            # contour objects (we can only remove and
            # recreating them), the quiver series which is
            # usually after the contour plot (in terms of
            # rendering order) will be moved on top, resulting
            # in the contour to hide the quivers. Setting
            # zorder appears to fix the problem.
            qkw["zorder"] = 1
        if s.use_cm:
            # don't use color map if a scalar field is
            # visible or if use_cm=False
            color_val = mag
            if s.color_func is not None:
                color_val = s.eval_color_func(xx, yy, uu0, vv0)
            qkw["cmap"] = next(p._cm)
            kw = p.merge({}, qkw, s.rendering_kw)
            q = p._ax.quiver(xx, yy, uu, vv, color_val, **kw)
            is_cb_added = p._add_colorbar(
                q, s.get_label(p.use_latex), s.use_cm and s.colorbar)
        else:
            is_cb_added = False
            qkw["color"] = next(p._cl)
            kw = p.merge({}, qkw, s.rendering_kw)
            solid_color = kw["color"]
            q = p._ax.quiver(xx, yy, uu, vv, **kw)
        handle = [q, kw, is_cb_added, p.fig.axes[-1]]

    if (not s.use_cm) and s.show_in_legend:
        # quivers are rendered with solid color: set up a legend handle
        proxy_artist = p.Line2D(
            [], [],
            color=solid_color, label=s.get_label(p.use_latex)
        )
        p._legend_handles.append(proxy_artist)

    return handle


def _update_vector2d_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    update_discr = (
        (s.n != renderer.previous_n)
        or (s.only_integers != renderer.previous_only_integers)
    )

    xx, yy, uu, vv = data
    mag = p.np.sqrt(uu ** 2 + vv ** 2)
    uu0, vv0 = [t.copy() for t in [uu, vv]]
    if s.normalize:
        uu, vv = [t / mag for t in [uu, vv]]
    if s.is_streamlines:
        raise NotImplementedError
    else:
        quivers, kw, is_cb_added, cax = handle
        color_val = mag
        if s.color_func is not None:
            color_val = s.eval_color_func(xx, yy, uu0, vv0)

        if update_discr:
            quivers.X = xx
            quivers.Y = yy
            quivers.N = xx.shape[0] * xx.shape[1]
            renderer.previous_n = s.n
            renderer.previous_only_integers = s.only_integers

        if is_cb_added:
            quivers.set_UVC(uu, vv, color_val)
            p._update_colorbar(
                cax,
                kw["cmap"],
                s.get_label(p.use_latex),
                color_val
            )
        else:
            quivers.set_UVC(uu, vv)
        quivers.set_offsets(p.np.c_[xx.flatten(), yy.flatten()])


class Vector2DRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_vector2d_helper: _update_vector2d_helper
    }

    def __init__(self, plot, s):
        super().__init__(plot, s)
        # previous numbers of discretization points
        self.previous_n = s.n
        self.previous_only_integers = s.only_integers
