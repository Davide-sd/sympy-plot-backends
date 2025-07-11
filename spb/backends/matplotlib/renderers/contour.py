from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from sympy.external import import_module

def _draw_contour_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    x, y, z = data
    ckw = dict(cmap=next(p._cm))
    if any(s.is_vector and (not s.is_streamlines) for s in p.series):
        # NOTE:
        # When plotting and updating a vector plot containing both
        # a contour series and a quiver series, because it's not
        # possible to update contour objects (we can only remove
        # and recreating them), the quiver series which is usually
        # after the contour plot (in terms of rendering order) will
        # be moved on top, resulting in the contour to hide the
        # quivers. Setting zorder appears to fix the problem.
        ckw["zorder"] = 0
    kw = p.merge({}, ckw, s.rendering_kw)
    func = p._ax.contourf if s.is_filled else p._ax.contour
    c = func(x, y, z, **kw)
    clabel = None
    if s.is_filled:
        p._add_colorbar(c, s.get_label(p.use_latex), s.colorbar)
    else:
        if s.show_clabels:
            clabel = p._ax.clabel(c)
    return [c, kw, p.fig.axes[-1], clabel]


def _update_contour_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    x, y, z = data
    kw, cax, clabels = handle[1:]

    # TODO: remove this and update setup.py
    packaging = import_module("packaging")
    matplotlib = import_module("matplotlib")
    curr_mpl_ver = packaging.version.parse(matplotlib.__version__)
    mpl_3_8 = packaging.version.parse("3.8.0")
    if curr_mpl_ver >= mpl_3_8:
        handle[0].remove()
    else:
        for c in handle[0].collections:
            c.remove()
        if (not s.is_filled) and s.show_clabels:
            for cl in clabels:
                cl.remove()

    func = p._ax.contourf if s.is_filled else p._ax.contour
    handle[0] = func(x, y, z, **kw)

    if s.is_filled and s.colorbar:
        p._update_colorbar(cax, kw["cmap"], s.get_label(p.use_latex), param=z)
    else:
        if (not s.is_filled) and s.show_clabels:
            clabels = p._ax.clabel(handle[0])
            handle[-1] = clabels


class ContourRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_contour_helper: _update_contour_helper
    }
