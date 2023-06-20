from spb.backends.base_renderer import Renderer
from spb.backends.plotly.renderers.surface import _draw_surface_helper, _update_surface_helper
from spb.backends.plotly.renderers.contour import _draw_contour_helper, _update_contour_helper
import warnings


def _draw_domain_coloring_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    x, y, mag, angle, img, colors = data
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    handles = []

    if s.at_infinity:
        mag, angle, img = [np.flip(np.flip(t, axis=0),
            axis=1) for t in [mag, angle, img]]

    h1 = p.go.Image(
        x0=xmin,
        y0=ymin,
        dx=(xmax - xmin) / s.n[0],
        dy=(ymax - ymin) / s.n[1],
        z=img,
        name=s.get_label(p._use_latex),
        customdata=np.dstack([mag, angle]),
        hovertemplate=(
            "x: %{x}<br />y: %{y}<br />RGB: %{z}"
            + "<br />Abs: %{customdata[0]}<br />Arg: %{customdata[1]}"
        ),
    )
    handles.append(h1)
    p._fig.add_trace(h1)

    if (colors is not None) and s.colorbar:
        # chroma/phase-colorbar
        h2 = p.go.Scatter(
            x=[xmin, xmax],
            y=[ymin, ymax],
            showlegend=False,
            mode="markers",
            marker=dict(
                opacity=0,
                colorscale=[
                    "rgb(%s, %s, %s)" % tuple(c) for c in colors
                ],
                color=[-np.pi, np.pi],
                colorbar=dict(
                    tickvals=[
                        -np.pi,
                        -np.pi / 2,
                        0,
                        np.pi / 2,
                        np.pi,
                    ],
                    ticktext=[
                        "-&#x3C0;",
                        "-&#x3C0; / 2",
                        "0",
                        "&#x3C0; / 2",
                        "&#x3C0;",
                    ],
                    x=1 + 0.1 * p._colorbar_counter,
                    title="Argument" if s.get_label(False) == str(s.expr) else s.get_label(p._use_latex),
                    titleside="right",
                ),
                showscale=True,
            ),
        )
        handles.append(h2)
        p._fig.add_trace(h2)
        p._colorbar_counter += 1

    return handles


def _update_domain_coloring_helper(renderer, data, handle):
    # TODO: for some unkown reason, domain_coloring and
    # interactive plot don't like each other...
    raise NotImplementedError


def _draw_analytic_landscape_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np

    xx, yy, mag, angle, colors, colorscale = data
    if s.coloring != "a":
        warnings.warn(
            "The visualization could be wrong becaue Plotly "
            + "doesn't support custom coloring over surfaces. "
            + "The surface color will show the "
            + "argument of the complex function."
        )
    # create a solid color to be used when s.use_cm=False
    col = next(p._cl)
    if s.use_cm:
        if colorscale is None:
            colorscale = "gray"
        else:
            tmp = []
            locations = list(range(0, len(colorscale)))
            locations = [t / (len(colorscale) - 1) for t in locations]
            for loc, c in zip(locations, colorscale):
                tmp.append([loc, "rgb" + str(tuple(c))])
            colorscale = tmp
            # to avoid jumps in the colormap, first and last colors
            # must be the same.
            colorscale[-1][1] = colorscale[0][1]
    else:
        colorscale = [[0, col], [1, col]]
    colormap = next(p._cyccm)
    skw = dict(
        name=s.get_label(p._use_latex),
        showscale=s.colorbar,
        colorbar=dict(
            x=1 + 0.1 * p._colorbar_counter,
            title="Argument",
            titleside="right",
            tickvals=[
                -np.pi,
                -np.pi / 2,
                0,
                np.pi / 2,
                np.pi,
            ],
            ticktext=[
                "-&#x3C0;",
                "-&#x3C0; / 2",
                "0",
                "&#x3C0; / 2",
                "&#x3C0;",
            ]
        ),
        cmin=-np.pi,
        cmax=np.pi,
        colorscale=colorscale,
        surfacecolor=angle,
        customdata=angle,
        hovertemplate="x: %{x}<br />y: %{y}<br />Abs: %{z}<br />Arg: %{customdata}",
    )

    kw = p.merge({}, skw, s.rendering_kw)
    handle = p.go.Surface(x=xx, y=yy, z=mag, **kw)
    p._fig.add_trace(handle)
    p._colorbar_counter += 1
    return len(p._fig.data) - 1


def _update_analytic_landscape_helper(renderer, data, idx):
    p, s = renderer.plot, renderer.series
    np = p.np
    handle = p.fig.data[idx]

    xx, yy, mag, angle, colors, colorscale = data
    handle["z"] = mag
    handle["surfacecolor"] = angle
    handle["customdata"] = angle
    m, M = min(angle.flatten()), max(angle.flatten())
    # show pi symbols on the colorbar if the range is
    # close enough to [-pi, pi]
    if (abs(m + np.pi) < 1e-02) and (abs(M - np.pi) < 1e-02):
        handle["colorbar"]["tickvals"] = [
            m,
            -np.pi / 2,
            0,
            np.pi / 2,
            M,
        ]
        handle["colorbar"]["ticktext"] = [
            "-&#x3C0;",
            "-&#x3C0; / 2",
            "0",
            "&#x3C0; / 2",
            "&#x3C0;",
        ]


class ComplexRenderer(Renderer):
    draw_update_map = {
        _draw_domain_coloring_helper: _update_domain_coloring_helper
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
        else:
            if self.series.is_3Dsurface:
                self.draw_update_map = {
                    _draw_analytic_landscape_helper: _update_analytic_landscape_helper
                }
