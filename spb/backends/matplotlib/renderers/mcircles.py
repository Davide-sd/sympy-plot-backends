from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer


def fdac(x):
    """Return the first digit after comma of a number.
    """
    return abs(int((x - int(x)) * 10))


_default_mcircles_line_kw = {
    "color": '0.75', "linestyle": ':', "zorder": 0, "linewidth": 0.9}


def _draw_mcircles_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    rkw = p.merge({}, _default_mcircles_line_kw, s.rendering_kw)

    handles = []
    for (mdb, x, y) in data:
        label = str(int(mdb)) if fdac(mdb) == 0 else str(mdb)
        if len(x) > 1:
            h1 = p._ax.plot(x, y, **rkw)[0]
            xtext_pos = (x.max() + x.min()) / 2
            ytext_pos = y.max()
            h2 = p._ax.text(
                xtext_pos, ytext_pos, label + " dB", fontsize=8,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(
                    facecolor="white", edgecolor="none",
                    boxstyle='square,pad=0'
                )
            )
        else:
            h1 = p._ax.axvline(x[0], **rkw)
            h2 = p._ax.text(
                x[0], 0.95, label + " dB", fontsize=8,
                transform=p._ax.get_xaxis_transform(),
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(
                    facecolor="white", edgecolor="none",
                    boxstyle='square,pad=0'
                )
            )
        handles.append([h1, h2])

    # Mark the -1 point
    p._ax.plot([-1], [0], 'r+')

    return handles


def _update_mcircles_helper(renderer, data, handles):
    p = renderer.plot
    s = renderer.series

    rkw = p.merge({}, _default_mcircles_line_kw, s.rendering_kw)

    for i, ((mdb, x, y), (h1, h2)) in enumerate(zip(data, handles)):
        label = str(int(mdb)) if fdac(mdb) == 0 else str(mdb)
        h2.set_text(label + " dB")

        if len(h1.get_data()[0]) == 2:
            if len(x) > 1:
                # in a previous update, this handle corresponded to a vertical
                # line, which uses some kind of transformation. It is easier
                # to remove it and plot the line again...
                h1.remove()
                handles[i][0] = p._ax.plot(x, y, **rkw)[0]
                xtext_pos = (x.max() + x.min()) / 2
                ytext_pos = y.max()
                h2.set_position([xtext_pos, ytext_pos])
            else:
                h1.set_xdata([x[0], x[0]])
        else:
            if len(x) == 1:
                # in a previous update, this handle corresponded to a normal
                # line. Now, it must represent a vertical line.
                h1.remove()
                handles[i][0] = p._ax.axvline(x[0], **rkw)
            else:
                h1.set_data(x, y)
                xtext_pos = (x.max() + x.min()) / 2
                ytext_pos = y.max()
                h2.set_position([xtext_pos, ytext_pos])


class MCirclesRenderer(MatplotlibRenderer):
    draw_update_map = {
        _draw_mcircles_helper: _update_mcircles_helper
    }
