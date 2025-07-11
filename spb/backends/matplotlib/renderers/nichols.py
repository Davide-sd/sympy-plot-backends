from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.backends.matplotlib.renderers.nyquist import _draw_arrows_helper
from sympy.external import import_module


def _draw_nichols_helper(renderer, data):
    mpl = import_module("matplotlib")
    p, s = renderer.plot, renderer.series
    omega, ol_phase, ol_mag, _, _ = data

    if s.use_cm:
        colormap = next(p._cm)
        if not s.is_point:
            lkw = dict(array=omega, cmap=colormap)
            kw = p.merge({}, lkw, s.rendering_kw)
            segments = p.get_segments(ol_phase, ol_mag)
            c = p.LineCollection(segments, **kw)
            p._ax.add_collection(c)
        else:
            lkw = dict(c=omega, cmap=colormap)
            kw = p.merge({}, lkw, s.rendering_kw)
            c = p._ax.scatter(ol_phase, ol_mag, **kw)
        is_cb_added = p._add_colorbar(
            c, s.get_label(p.use_latex), s.use_cm and s.colorbar
        )
        handles = (c, kw, is_cb_added, p.fig.axes[-1])

    else:
        color = next(p._cl) if s.line_color is None else s.line_color
        lkw = dict(
            label=s.get_label(p.use_latex) if s.show_in_legend else "_nolegend_",
            color=color
        )

        if s.is_point:
            lkw["marker"] = "o"
            lkw["linestyle"] = "None"
            if not s.is_filled:
                lkw["markerfacecolor"] = (1, 1, 1)
        kw = p.merge({}, lkw, s.rendering_kw)
        line, = p._ax.plot(ol_phase, ol_mag, **kw)

        # Set the arrow style
        arrow_style = mpl.patches.ArrowStyle('simple', head_width=6, head_length=6)
        arrows = _draw_arrows_helper(
            p._ax, line,
            s.arrow_locs,
            arrowstyle=arrow_style,
            dir=1
        )
        handles = [line, arrows]

    return handles


def _update_nichols_helper(renderer, data, handles):
    mpl = import_module("matplotlib")
    p, s = renderer.plot, renderer.series
    omega, ol_phase, ol_mag, _, _ = data

    if s.use_cm:
        line, kw, is_cb_added, cax = handles

        if not s.is_point:
            segments = p.get_segments(ol_phase, ol_mag)
            line.set_segments(segments)
        else:
            line.set_offsets(p.np.c_[ol_phase, ol_mag])

        line.set_array(omega)
        line.set_clim(vmin=min(omega), vmax=max(omega))

        if is_cb_added:
            norm = p.Normalize(vmin=p.np.amin(omega), vmax=p.np.amax(omega))
            p._update_colorbar(
                cax, kw["cmap"], s.get_label(p.use_latex), norm=norm)
    else:
        line, arrows = handles
        line.set_data(ol_phase, ol_mag)
        for a in arrows:
            a.remove()
        arrows.clear()
        # Set the arrow style
        arrow_style = mpl.patches.ArrowStyle('simple', head_width=6, head_length=6)
        new_arrows = _draw_arrows_helper(
            p._ax, line,
            s.arrow_locs,
            arrowstyle=arrow_style,
            dir=1
        )
        arrows.extend(new_arrows)


class NicholsLineRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_nichols_helper: _update_nichols_helper
    }
