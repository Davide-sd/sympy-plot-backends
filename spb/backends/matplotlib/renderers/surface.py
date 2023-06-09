from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer


def _draw_surface_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    if not s.is_parametric:
        x, y, z = data
        facecolors = s.eval_color_func(x, y, z)
    else:
        x, y, z, u, v = data
        facecolors = s.eval_color_func(x, y, z, u, v)
    skw = dict(rstride=1, cstride=1, linewidth=0.1)
    norm, cmap = None, None
    if s.use_cm:
        vmin = s.rendering_kw.get("vmin", p.np.amin(facecolors))
        vmax = s.rendering_kw.get("vmax", p.np.amax(facecolors))
        norm = p.Normalize(vmin=vmin, vmax=vmax)
        cmap = next(p._cm)
        skw["cmap"] = cmap
    else:
        skw["color"] = next(p._cl) if s.surface_color is None else s.surface_color
        proxy_artist = p.Rectangle((0, 0), 1, 1,
            color=skw["color"], label=s.get_label(p._use_latex))
        if s.show_in_legend:
            p._legend_handles.append(proxy_artist)

    kw = p.merge({}, skw, s.rendering_kw)
    if s.use_cm:
        # facecolors must be computed here because s.rendering_kw
        # might have its own cmap
        cmap = kw["cmap"]
        if isinstance(cmap, str):
            cmap = p.cm.get_cmap(cmap)
        kw["facecolors"] = cmap(norm(facecolors))
    c = p._ax.plot_surface(x, y, z, **kw)
    is_cb_added = p._add_colorbar(c, s.get_label(p._use_latex), s.use_cm and s.colorbar, norm=norm, cmap=cmap)
    return [c, kw, is_cb_added, p._fig.axes[-1]]


def _update_surface_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    if not s.is_parametric:
        x, y, z = data
        facecolors = s.eval_color_func(x, y, z)
    else:
        x, y, z, u, v = data
        facecolors = s.eval_color_func(x, y, z, u, v)
    # TODO: by setting the keyword arguments, somehow the
    # update becomes really really slow.
    kw, is_cb_added, cax = handle[1:]

    if is_cb_added or ("cmap" in kw.keys()):
        # TODO: if use_cm=True and a single 3D expression is
        # shown with legend=False, this won't get executed.
        # In widget plots, the surface will never change color.
        vmin = s.rendering_kw.get("vmin", p.np.amin(facecolors))
        vmax = s.rendering_kw.get("vmax", p.np.amax(facecolors))
        norm = p.Normalize(vmin=vmin, vmax=vmax)
        cmap = kw["cmap"]
        if isinstance(cmap, str):
            cmap = p.cm.get_cmap(cmap)
        kw["facecolors"] = cmap(norm(facecolors))
    handle[0].remove()
    handle[0] = p._ax.plot_surface(
        x, y, z, **kw)

    if is_cb_added:
        p._update_colorbar(cax, kw["cmap"], s.get_label(p._use_latex), norm=norm)


class SurfaceRenderer(MatplotlibRenderer):
    _sal = True
    draw_update_map = {
        _draw_surface_helper: _update_surface_helper
    }
