from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer


def _draw_line2d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    if s.is_parametric:
        x, y, param = data
    else:
        x, y = data

    if s.is_parametric and s.use_cm:
        colormap = (
            next(p._cyccm)
            if p._use_cyclic_cm(param, s.is_complex)
            else next(p._cm)
        )
        if not s.is_scatter:
            lkw = dict(array=param, cmap=colormap)
            kw = p.merge({}, lkw, s.rendering_kw)
            segments = p.get_segments(x, y)
            c = p.LineCollection(segments, **kw)
            p._ax.add_collection(c)
        else:
            lkw = dict(c=param, cmap=colormap)
            kw = p.merge({}, lkw, s.rendering_kw)
            c = p._ax.scatter(x, y, **kw)

        colorbar = p._add_colorbar(
            c, s.get_label(p.use_latex), s.use_cm and s.colorbar
        )
        handle = (c, kw, colorbar, p.fig.axes[-1])
    else:
        if s.get_label(False) != "__k__":
            color = next(p._cl) if s.line_color is None else s.line_color
        else:
            color = p.wireframe_color

        lkw = dict(
            label=s.get_label(p.use_latex) if s.show_in_legend else "_nolegend_",
            color=color
        )
        if s.is_scatter:
            lkw["marker"] = "o"
            lkw["linestyle"] = "None"
            if not s.is_filled:
                lkw["markerfacecolor"] = (1, 1, 1)
        kw = p.merge({}, lkw, s.rendering_kw)
        l = p._ax.plot(x, y, **kw)
        handle = l

    # add vertical lines at discontinuities
    hvlines = []
    if hasattr(s, "poles_locations"):
        pole_line_kw = p.merge(
            {"color": "k", "linestyle": ":"}, s.poles_rendering_kw)
        hvlines = [
            p._ax.axvline(x_loc, **pole_line_kw) for x_loc in s.poles_locations
        ]

    return [handle, hvlines]


def _update_line2d_helper(renderer, data, handles):
    p, s = renderer.plot, renderer.series
    if s.is_parametric:
        x, y, param = data
    else:
        x, y = data

    line_handles, hvlines = handles

    if s.is_parametric and s.use_cm:
        line, kw, colorbar, cax = line_handles

        if not s.is_scatter:
            segments = p.get_segments(x, y)
            line.set_segments(segments)
        else:
            line.set_offsets(p.np.c_[x, y])

        line.set_array(param)
        line.set_clim(vmin=min(param), vmax=max(param))

        if colorbar:
            colorbar.update_normal(line)

    else:
        line = line_handles[0]
        # TODO: Point2D are updated but not visible.
        line.set_data(x, y)

    # update vertical lines
    if hasattr(s, "poles_locations"):
        if len(hvlines) != len(s.poles_locations):
            pole_line_kw = p.merge(
                {"color": "k", "linestyle": ":"}, s.poles_rendering_kw)

            for hvl in hvlines:
                hvl.remove()
            handles[1] = [
                p._ax.axvline(x_loc, **pole_line_kw)
                for x_loc in s.poles_locations
            ]
        elif len(hvlines) > 0:
            for hvl, x_loc in zip(hvlines, s.poles_locations):
                hvl.set_xdata([x_loc, x_loc])


class Line2DRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_line2d_helper: _update_line2d_helper
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.series.use_cm or hasattr(self.series, "_xlim"):
            self._sal = True

    def draw(self):
        data = self.series.get_data()

        # NOTE: matplotlib's is able to deal with axis limits when
        # LineCollection is present, so no need to compute them here.
        if not self.series.is_2Dline:
            self._set_axis_limits(data)

        if hasattr(self.series, "_xlim") and (self.series._xlim is not None):
            self._set_axis_limits([self.series._xlim, self.series._ylim])

        for draw_method in self.draw_update_map.keys():
            self.handles.append(
                draw_method(self, data))
