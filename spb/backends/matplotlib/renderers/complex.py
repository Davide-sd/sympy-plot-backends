from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.backends.matplotlib.renderers.surface import _draw_surface_helper, _update_surface_helper
from spb.backends.matplotlib.renderers.contour import _draw_contour_helper, _update_contour_helper


def _draw_complex_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    if not s.is_3Dsurface:
        x, y, _, _, img, colors = data
        ikw = dict(
            interpolation="spline36",
            origin="lower",
        )
        if s.at_infinity:
            ikw["extent"] = [np.amax(x), np.amin(x), np.amin(y), np.amax(y)]
            img = np.flip(np.flip(img, axis=0), axis=1)
        else:
            ikw["extent"] = [np.amin(x), np.amax(x), np.amin(y), np.amax(y)]
        kw = p.merge({}, ikw, s.rendering_kw)
        image = p._ax.imshow(img, **kw)
        handle = [image, kw]

        # chroma/phase-colorbar
        if (colors is not None) and s.colorbar:
            colors = colors / 255.0

            colormap = p.ListedColormap(colors)
            norm = p.Normalize(vmin=-np.pi, vmax=np.pi)
            method = p._fig.colorbar if not p._imagegrid else p._ax.cax.colorbar
            cb2 = method(
                p.cm.ScalarMappable(norm=norm, cmap=colormap),
                # orientation="vertical",
                label="Argument" if s.get_label(False) == str(s.expr) else s.get_label(self._use_latex),
                ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                ax=p._ax,
            )
            cb2.ax.set_yticklabels(
                [r"-$\pi$", r"-$\pi / 2$", "0", r"$\pi / 2$", r"$\pi$"]
            )
    else:
        x, y, mag, arg, facecolors, colorscale = data

        skw = dict(rstride=1, cstride=1, linewidth=0.1)
        if s.use_cm:
            skw["facecolors"] = facecolors / 255
        else:
            skw["color"] = next(p._cl) if s.surface_color is None else s.surface_color
        kw = p.merge({}, skw, s.rendering_kw)
        c = p._ax.plot_surface(x, y, mag, **kw)

        if s.use_cm and (colorscale is not None) and s.colorbar:
            if len(colorscale.shape) == 3:
                colorscale = colorscale.reshape((-1, 3))
            else:
                colorscale = colorscale / 255.0

            norm = p.Normalize(vmin=-np.pi, vmax=np.pi)
            mappable = p.cm.ScalarMappable(
                cmap=p.ListedColormap(colorscale), norm=norm
            )
            cb = p._fig.colorbar(
                mappable,
                orientation="vertical",
                label="Argument",
                ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                ax=p._ax,
            )
            cb.ax.set_yticklabels(
                [r"-$\pi$", r"-$\pi / 2$", "0", r"$\pi / 2$", r"$\pi$"]
            )
        handle = [c, kw]
    return handle


def _update_complex_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    if not s.is_3Dsurface:
        x, y, _, _, img, colors = data
        handle[0].set_data(img)
        handle[0].set_extent((x.min(), x.max(), y.min(), y.max()))
    else:
        x, y, mag, arg, facecolors, colorscale = data
        handle[0].remove()
        kw = handle[1]
        if s.use_cm:
            kw["facecolors"] = facecolors / 255
        handle[0] = p._ax.plot_surface(x, y, mag, **kw)
    

class ComplexRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_complex_helper: _update_complex_helper
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.series.is_domain_coloring:
            if self.series.is_3Dsurface:
                self.draw_update_map = {
                    _draw_surface_helper: _update_surface_helper
                }
            else:
                self.draw_update_map = {
                    _draw_contour_helper: _update_contour_helper
                }
        if self.series.is_3Dsurface:
            self._sal = True
