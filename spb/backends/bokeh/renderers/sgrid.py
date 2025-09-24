from spb.backends.base_renderer import Renderer
from spb.series import RootLocusSeries
from sympy.external import import_module


def _text_position_limits(r, p, s):
    """Computes the limits for text-annotations.

    Parameters
    ==========
    r : Renderer
    p : BaseBackend
    s : SGridLineSeries
    """
    xlim = p.xlim if p.xlim else s.xlim
    ylim = p.ylim if p.ylim else s.ylim

    if (xlim is None) and (ylim is None):
        xlim = r.default_xlim
        ylim = r.default_ylim
    xtext_pos_lim = xlim[0] + (xlim[1] - xlim[0]) * 0.0 if xlim else -1
    ytext_pos_lim = ylim[1] - (ylim[1] - ylim[0]) * 0.03 if ylim else 1
    return xlim, ylim, xtext_pos_lim, ytext_pos_lim


def _draw_sgrid_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    xi_dict, wn_dict, y_tp, x_ts = data
    lkw = {"line_color": "#aaa", "line_dash": "dotted"}
    kw = p.merge({}, lkw, s.rendering_kw)

    def _add_labels(x, y, labels, text_align="center"):
        source = p.bokeh.models.ColumnDataSource(data={
            "x": x, "y": y, "labels": labels
        })
        handle = p.bokeh.models.LabelSet(
            x="x", y="y", text="labels",
            x_offset="x_offset", y_offset="y_offset", source=source,
            text_baseline="middle", text_align=text_align,
            text_font_size="12px", text_color="#000000"
        )
        p._fig.add_layout(handle)
        return handle

    xlim, ylim, xtext_pos_lim, ytext_pos_lim = _text_position_limits(
        renderer, p, s)

    if s.show_control_axis:
        p._fig.add_layout(
            p.bokeh.models.Span(
                location=0, dimension="width", **kw)
        )
        p._fig.add_layout(
            p.bokeh.models.Span(
                location=0, dimension="height", **kw)
        )

    # damping ratio lines
    xi_handles = []
    xlabels, ylabels, labels = [], [], []
    for (x, a, yp), v in xi_dict.items():
        source = {"x": v["x"], "y": v["y"]}
        h1 = p._fig.line("x", "y", source=source, **kw)
        h2 = None
        if not np.isclose(x, 1):
            source = {"x": v["x"], "y": -v["y"]}
            h2 = p._fig.line("x", "y", source=source, **kw)
        if yp < 0 and not np.isclose(x, 1):
            xtext_pos = 1/yp * ylim[1]
            ytext_pos = yp * xtext_pos_lim
            if np.abs(xtext_pos) > np.abs(xtext_pos_lim):
                xtext_pos = xtext_pos_lim
            else:
                ytext_pos = ytext_pos_lim
            xlabels.append(xtext_pos)
            ylabels.append(ytext_pos)
            labels.append(v["label"])
        xi_handles.append([h1, h2])

    xi_labels = _add_labels(xlabels, ylabels, labels, "left")

    # natural frequency lines
    wn_handles = []
    xlabels, ylabels, labels = [], [], []
    for k, v in wn_dict.items():
        source = {"x": v["x"], "y": v["y"]}
        h1 = p._fig.line("x", "y", source=source, **kw)
        xlabels.append(v["lx"])
        ylabels.append(v["ly"])
        labels.append(v["label"])
        wn_handles.append(h1)

    wn_labels = _add_labels(xlabels, ylabels, labels)

    # peak time lines
    tp_handles = []
    for y in y_tp:
        h = p.bokeh.models.Span(location=y, dimension="width", **kw)
        p._fig.add_layout(h)
        tp_handles.append(h)

    # settling time lines
    ts_handles = []
    for x in x_ts:
        h = p.bokeh.models.Span(location=x, dimension="height", **kw)
        p._fig.add_layout(h)
        ts_handles.append(h)

    return [
        xi_handles, xi_labels,
        wn_handles, wn_labels,
        tp_handles, ts_handles
    ]


def _update_sgrid_helper(renderer, data, handles):
    p, s = renderer.plot, renderer.series
    np = p.np
    xi_dict, wn_dict, y_tp, x_ts = data
    xi_handles, xi_labels, wn_handles, wn_labels = handles[:4]
    tp_handles, ts_handles = handles[4:]

    xlim, ylim, xtext_pos_lim, ytext_pos_lim = _text_position_limits(
        renderer, p, s)

    # damping ratio lines
    xlabels, ylabels, labels = [], [], []
    for (k, v), handles in zip(xi_dict.items(), xi_handles):
        h1, h2 = handles
        s1 = {"x": v["x"], "y": v["y"]}
        h1.data_source.data.update(s1)
        if h2:
            s2 = {"x": v["x"], "y": -v["y"]}
            h2.data_source.data.update(s2)
        x, a, yp = k
        if yp < 0 and not np.isclose(x, 1):
            xtext_pos = 1/yp * ylim[1]
            ytext_pos = yp * xtext_pos_lim
            if np.abs(xtext_pos) > np.abs(xtext_pos_lim):
                xtext_pos = xtext_pos_lim
            else:
                ytext_pos = ytext_pos_lim
            xlabels.append(xtext_pos)
            ylabels.append(ytext_pos)
            labels.append(v["label"])

    xi_labels.source.data.update(
        {"x": xlabels, "y": ylabels, "labels": labels})

    # natural frequency lines
    y_offset = 0 if ylim is None else 0.015 * abs(ylim[1] - ylim[0])
    for v, h1 in zip(wn_dict.values(), wn_handles):
        s1 = {"x": v["x"], "y": v["y"]}
        h1.data_source.data.update(s1)
        xlabels.append(v["lx"])
        ylabels.append(y_offset)
        labels.append(v["label"])

    # peak time lines
    for y, h in zip(y_tp, tp_handles):
        h.location = y

    # settling time lines
    for x, h in zip(x_ts, ts_handles):
        h.location = x


def _find_data_axis_limits(fig, x_key="x", y_key="y"):
    """Loop over the lines in order to find their minimum and
    maximum coordinates.
    """
    np = import_module("numpy")
    bokeh = import_module("bokeh")
    xlims, ylims = [], []
    for line in fig.renderers:
        if isinstance(line.glyph, (bokeh.models.Line, bokeh.models.Scatter)):
            data = line.data_source.data
            if (x_key in data.keys()) and (y_key in data.keys()):
                x = data[x_key]
                y = data[y_key]
                xlims.append([np.nanmin(x), np.nanmax(x)])
                ylims.append([np.nanmin(y), np.nanmax(y)])
    return xlims, ylims


def _modify_axis_limits(lim, margin_factor_lower, margin_factor_upper):
    """Modify the provided axis limits and add appropriate margins.
    """
    np = import_module("numpy")
    min_t, max_t = lim
    # this offset allows to have a little bit of empty space on the
    # LHP of root locus plot
    offset = 0.25
    min_t = min_t - offset if np.isclose(min_t, 0) else min_t
    max_t = max_t + offset if np.isclose(max_t, 0) else max_t
    # provide a little bit of margin
    delta = abs(max_t - min_t)
    lim = [
        min_t - delta * margin_factor_lower,
        max_t + delta * margin_factor_upper
    ]
    if np.isclose(*lim):
        # prevent axis limits to be the same
        lim[0] -= 1
        lim[1] += 1
    return lim


class SGridLineRenderer(Renderer):
    default_xlim = [-11, 1]
    default_ylim = [-5, 5]
    _sal = True
    draw_update_map = {
        _draw_sgrid_helper: _update_sgrid_helper
    }

    def _set_axis_limits_before_compute_data(self):
        np = import_module("numpy")
        # loop over the data already present on the plot and find
        # appropriate axis limits
        root_locus_series = [
            s for s in self.plot.series if isinstance(s, RootLocusSeries)]
        if len(root_locus_series) > 0:
            xlims, ylims = [], []
            for s in root_locus_series:
                xlims.append(s.xlim)
                ylims.append(s.ylim)
        else:
            xlims, ylims = _find_data_axis_limits(self.plot._fig)

        if len(xlims) > 0:
            xlims = np.array(xlims)
            xlim = (np.nanmin(xlims[:, 0]), np.nanmax(xlims[:, 1]))
        else:
            xlim = self.plot.xlim if self.plot.xlim else self.default_xlim

        if len(ylims) > 0:
            ylims = np.array(ylims)
            ylim = (np.nanmin(ylims[:, 0]), np.nanmax(ylims[:, 1]))
        else:
            ylim = self.plot.ylim if self.plot.ylim else self.default_ylim

        xlim = _modify_axis_limits(xlim, 0.15, 0.05)
        ylim = _modify_axis_limits(ylim, 0.05, 0.05)
        self.series.set_axis_limits(xlim, ylim)

    def draw(self):
        if (self.series.xlim is None) and (self.series.ylim is None):
            # if user didn't set xlim/ylim in sgrid() function call
            self._set_axis_limits_before_compute_data()

        data = self.series.get_data()
        self._set_axis_limits([self.series.xlim, self.series.ylim])
        for draw_method in self.draw_update_map.keys():
            self.handles.append(
                draw_method(self, data))

    def update(self, params):
        self.series.params = params
        self._set_axis_limits_before_compute_data()
        data = self.series.get_data()
        self._set_axis_limits([self.series.xlim, self.series.ylim])
        for update_method, handle in zip(
            self.draw_update_map.values(), self.handles
        ):
            update_method(self, data, handle)

    def _set_axis_limits(self, data):
        np = self.plot.np
        # NOTE: axis limits cannot be NaN or Inf
        self.xlims = [[np.nanmin(data[0]), np.nanmax(data[0])]]
        self.ylims = [[np.nanmin(data[1]), np.nanmax(data[1])]]
