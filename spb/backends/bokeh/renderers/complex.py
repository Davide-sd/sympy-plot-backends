from spb.backends.base_renderer import Renderer
from spb.backends.bokeh.renderers.contour import (
    _draw_contour_helper, _update_contour_helper
)


def _draw_domain_coloring_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np

    x, y, mag, angle, img, colors = data
    img = p._get_img(img)

    p._fig.x_range.flipped = s.at_infinity
    if s.at_infinity:
        # NOTE: why img has only one flip while mag-angle two flips? I guess
        # because source.image is mapped to the fig.x_range, but mag-angle are
        # not... might be a bug.
        img = np.flip(img, axis=0)
        mag, angle = [
            np.flip(np.flip(t, axis=0), axis=1) for t in [mag, angle]
        ]
    source = p.bokeh.models.ColumnDataSource(
        {
            "image": [img],
            "abs": [mag],
            "arg": [angle],
        }
    )

    handle = p._fig.image_rgba(
        source=source,
        x=x.min(),
        y=y.min(),
        dw=x.max() - x.min(),
        dh=y.max() - y.min(),
    )
    p._fig.add_tools(p.bokeh.models.HoverTool(
        tooltips=[("x", "$x"), ("y", "$y"), ("Abs", "@abs"), ("Arg", "@arg")],
        renderers=[handle]
    ))

    if (colors is not None) and s.colorbar:
        # chroma/phase-colorbar
        cm1 = p.bokeh.models.LinearColorMapper(
            palette=[tuple(c) for c in colors],
            low=-np.pi, high=np.pi)
        ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        labels = ["-π", "-π / 2", "0", "π / 2", "π"]
        colorbar1 = p.bokeh.models.ColorBar(
            color_mapper=cm1,
            title=(
                "Argument" if s.get_label(False) == str(s.expr)
                else s.get_label(p._use_latex)
            ),
            ticker=p.bokeh.models.tickers.FixedTicker(ticks=ticks),
            major_label_overrides={k: v for k, v in zip(ticks, labels)})
        p._fig.add_layout(colorbar1, "right")

    if s.riemann_mask and s.annotate:
        options = dict(
            line_width=2, color="#000000", size=8, marker="circle",
            level="overlay"
        )
        # points (1, 0), (0, i), (0, -i)
        p._fig.scatter([1, 0, 0], [0, 1, -1], fill_color="white", **options)
        # center point
        p._fig.scatter(
            [0], [0],
            fill_color="white" if s.at_infinity else "black",
            **options
        )
        # annotations
        pixel_offset = 15
        labels = ["0", "i", "-i", "1"]
        sign = 1
        if s.at_infinity:
            labels[0] = "inf"
            sign = -1
        source = p.bokeh.models.ColumnDataSource(data={
            "x": [0, 0, 0, 1], "y": [0, 1, -1, 0], "labels": labels,
            "x_offset": [pixel_offset, 0, 0, sign * pixel_offset],
            "y_offset": [0, pixel_offset, -pixel_offset, 0]
        })
        p._fig.add_layout(
            p.bokeh.models.LabelSet(
                x="x", y="y", text="labels",
                x_offset="x_offset", y_offset="y_offset", source=source,
                text_baseline="middle", text_align="center",
                text_font_style="bold", text_color="#000000"
            ))

    return handle


def _update_domain_coloring_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    np = p.np

    x, y, mag, angle, img, _ = data
    minx, miny = x.min(), y.min()
    maxx, maxy = x.max(), y.max()
    img = p._get_img(img)
    if s.at_infinity:
        # NOTE: why img has only one flip while mag-angle two flips? I guess
        # because source.image is mapped to the fig.x_range, but mag-angle are
        # not... might be a bug.
        img = np.flip(img, axis=0)
        mag, angle = [
            np.flip(np.flip(t, axis=0), axis=1) for t in [mag, angle]
        ]
    source = {
        "image": [img],
        "abs": [mag],
        "arg": [angle],
    }
    handle.data_source.data.update(source)
    handle.glyph.x = minx
    handle.glyph.y = miny
    handle.glyph.dw = abs(maxx - minx)
    handle.glyph.dh = abs(maxy - miny)


def _draw_3d_helper(renderer, data):
    raise NotImplementedError


def _update_3d_helper(renderer, data):
    raise NotImplementedError


class ComplexRenderer(Renderer):
    draw_update_map = {
        _draw_domain_coloring_helper: _update_domain_coloring_helper
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.series.is_3Dsurface:
            self.draw_update_map = {
                _draw_3d_helper: _update_3d_helper
            }
        else:
            if not self.series.is_domain_coloring:
                self.draw_update_map = {
                    _draw_contour_helper: _update_contour_helper
                }
