from spb.backends.base_renderer import Renderer


def _draw_zgrid_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    xi_dict, wn_dict, tp_dict, ts_dict = data

    lkw = p.grid_line_kw
    kw = p.merge({}, lkw, s.rendering_kw)

    def _add_labels(x, y, labels):
        source = p.bokeh.models.ColumnDataSource(data={
            "x": x, "y": y, "labels": labels
        })
        handle = p.bokeh.models.LabelSet(
            x="x", y="y", text="labels",
            x_offset="x_offset", y_offset="y_offset", source=source,
            text_baseline="middle", text_align="center",
            text_font_size="12px", text_color="#000000"
        )
        p._fig.add_layout(handle)
        return handle

    # damping ratio lines
    xi_handles = []
    xlabels, ylabels, labels = [], [], []
    for k, v in xi_dict.items():
        s1 = {"x": v["x1"], "y": v["y1"]}
        s2 = {"x": v["x2"], "y": v["y2"]}
        h1 = p._fig.line("x", "y", source=s1, **kw)
        h2 = p._fig.line("x", "y", source=s2, **kw)
        xi_handles.append([h1, h2])
        xlabels.append(v["lx"])
        ylabels.append(v["ly"])
        labels.append(v["label"])

    xi_labels = _add_labels(xlabels, ylabels, labels)

    # natural frequency lines
    wn_handles = []
    xlabels, ylabels, labels = [], [], []
    for k, v in wn_dict.items():
        source = {"x": v["x"], "y": v["y"]}
        h1 = p._fig.line("x", "y", source=source, **kw)
        wn_handles.append(h1)
        xlabels.append(v["lx"])
        ylabels.append(v["ly"])
        label = v["label"](p.use_latex)
        labels.append(r"$%s$" % label if p.use_latex else label)
        xi_handles.append(h1)

    wn_labels = _add_labels(xlabels, ylabels, labels)

    # peak time lines
    tp_handles = []
    xlabels, ylabels, labels = [], [], []
    for k, v in tp_dict.items():
        source = {"x": v["x"], "y": v["y"]}
        h1 = p._fig.line("x", "y", source=source, **kw)
        xlabels.append(v["lx"])
        ylabels.append(v["ly"])
        labels.append(v["label"])
        tp_handles.append(h1)

    tp_labels = _add_labels(xlabels, ylabels, labels)

    # settling time lines
    ts_handles = []
    xlabels, ylabels, labels = [], [], []
    for k, v in ts_dict.items():
        source = {"x": v["x"], "y": v["y"]}
        h1 = p._fig.line("x", "y", source=source, **kw)
        xlabels.append(v["lx"])
        ylabels.append(v["ly"])
        labels.append(v["label"])
        ts_handles.append(h1)

    ts_labels = _add_labels(xlabels, ylabels, labels)

    if s.show_control_axis:
        p._fig.add_layout(
            p.bokeh.models.Span(
                location=0, dimension="width", **kw)
        )
        p._fig.add_layout(
            p.bokeh.models.Span(
                location=0, dimension="height", **kw)
        )

    return [
        xi_handles, xi_labels, wn_handles, wn_labels,
        tp_handles, tp_labels, ts_handles, ts_labels
    ]


def _update_zgrid_helper(renderer, data, handles):
    p = renderer.plot
    xi_handles, xi_labels, wn_handles, wn_labels = handles[:4]
    tp_handles, tp_labels, ts_handles, ts_labels = handles[4:]
    xi_dict, wn_dict, tp_dict, ts_dict = data

    xlabels, ylabels, labels = [], [], []
    for v, handles in zip(xi_dict.values(), xi_handles):
        h1, h2 = handles
        s1 = {"x": v["x1"], "y": v["y1"]}
        s2 = {"x": v["x2"], "y": v["y2"]}
        h1.data_source.data.update(s1)
        h2.data_source.data.update(s2)
        xlabels.append(v["lx"])
        ylabels.append(v["ly"])
        labels.append(v["label"])

    xi_labels.source.data.update(
        {"x": xlabels, "y": ylabels, "labels": labels})

    xlabels, ylabels, labels = [], [], []
    for v, h1 in zip(wn_dict.values(), wn_handles):
        source = {"x": v["x"], "y": v["y"]}
        h1.data_source.data.update(source)
        xlabels.append(v["lx"])
        ylabels.append(v["ly"])
        label = v["label"](p.use_latex)
        labels.append(r"$%s$" % label if p.use_latex else label)

    wn_labels.source.data.update(
        {"x": xlabels, "y": ylabels, "labels": labels})

    xlabels, ylabels, labels = [], [], []
    for v, h1 in zip(tp_dict.values(), tp_handles):
        source = {"x": v["x"], "y": v["y"]}
        h1.data_source.data.update(source)
        xlabels.append(v["lx"])
        ylabels.append(v["ly"])
        labels.append(v["label"])

    tp_labels.source.data.update(
        {"x": xlabels, "y": ylabels, "labels": labels})

    xlabels, ylabels, labels = [], [], []
    for v, h1 in zip(ts_dict.values(), ts_handles):
        source = {"x": v["x"], "y": v["y"]}
        h1.data_source.data.update(source)
        xlabels.append(v["lx"])
        ylabels.append(v["ly"])
        labels.append(v["label"])

    ts_labels.source.data.update(
        {"x": xlabels, "y": ylabels, "labels": labels})


class ZGridLineRenderer(Renderer):
    draw_update_map = {
        _draw_zgrid_helper: _update_zgrid_helper
    }
