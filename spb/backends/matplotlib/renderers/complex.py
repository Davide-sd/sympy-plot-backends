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
                label="Argument" if s.get_label(False) == str(s.expr) else s.get_label(p._use_latex),
                ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                ax=p._ax,
            )
            cb2.ax.set_yticklabels(
                [r"-$\pi$", r"-$\pi / 2$", "0", r"$\pi / 2$", r"$\pi$"]
            )

        if s.riemann_mask and s.annotate:
            pixel_offset = 15
            # assumption: there is only one data series being plotted.
            sign = -1 if s.at_infinity else 1

            # plot markers at (1, 0), (0, i), (0, -i)
            markers, = p._ax.plot([1, 0, 0], [0, 1, -1], linestyle="none",
                color="k", markersize=4, marker="o",
                markerfacecolor=(1, 1, 1), zorder=10)
            center, = p._ax.plot([0], [0], linestyle="none",
                color="k", markersize=4, marker="o", zorder=10,
                markerfacecolor=(1, 1, 1) if s.at_infinity else None)
            a_plus_1 = p._ax.annotate(
                text="1", xy=(1, 0), xytext=(pixel_offset * sign, 0),
                textcoords="offset pixels", ha="center", va="center")
            a_plus_i = p._ax.annotate(
                text="i", xy=(0, 1), xytext=(0, pixel_offset),
                textcoords="offset pixels", ha="center", va="center")
            a_minus_i = p._ax.annotate(
                text="-i", xy=(0, -1), xytext=(0, -pixel_offset),
                textcoords="offset pixels", ha="center", va="center")
            a_center = p._ax.annotate(
                text=r"$\infty$" if s.at_infinity else "0",
                xy=(0, 0), xytext=(0, -pixel_offset),
                textcoords="offset pixels", ha="center", va="center")

            handle += [markers, center, a_plus_1, a_plus_i, a_minus_i, a_center]
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
    np = p.np

    if not s.is_3Dsurface:
        x, y, _, _, img, colors = data
        if s.at_infinity:
            img = np.flip(np.flip(img, axis=0), axis=1)
        handle[0].set_data(img)
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
