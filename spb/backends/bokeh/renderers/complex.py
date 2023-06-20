from spb.backends.base_renderer import Renderer
from spb.backends.bokeh.renderers.contour import _draw_contour_helper, _update_contour_helper
import warnings


def _draw_domain_coloring_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np

    x, y, mag, angle, img, colors = data
    img = p._get_img(img)

    p._fig.x_range.flipped = s.at_infinity
    if s.at_infinity:
        mag, angle, img = [np.flip(np.flip(t, axis=0),
            axis=1) for t in [mag, angle, img]]
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

    if (colors is not None) and s.colorbar:
        # chroma/phase-colorbar
        cm1 = p.bokeh.models.LinearColorMapper(
            palette=[tuple(c) for c in colors],
            low=-np.pi, high=np.pi)
        ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        labels = ["-π", "-π / 2", "0", "π / 2", "π"]
        colorbar1 = p.bokeh.models.ColorBar(
            color_mapper=cm1,
            title="Argument" if s.get_label(False) == str(s.expr) else s.get_label(p._use_latex),
            ticker=p.bokeh.models.tickers.FixedTicker(ticks=ticks),
            major_label_overrides={k: v for k, v in zip(ticks, labels)})
        p._fig.add_layout(colorbar1, "right")

    return handle


def _update_domain_coloring_helper(renderer, data, handle):
    p, s = renderer.plot, renderer.series
    x, y, mag, angle, img, _ = data
    minx, miny = x.min(), y.min()
    maxx, maxy = x.max(), y.max()
    img = p._get_img(img)
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
