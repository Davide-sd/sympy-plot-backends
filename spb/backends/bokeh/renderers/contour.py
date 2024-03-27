from spb.backends.base_renderer import Renderer
import warnings


def _draw_contour_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    handle = []

    if s.is_polar:
        raise NotImplementedError()
    x, y, z = data
    x, y, zz = [t.flatten() for t in [x, y, z]]
    minx, miny, minz = min(x), min(y), min(zz)
    maxx, maxy, maxz = max(x), max(y), max(zz)

    cm = next(p._cm)
    ckw = dict(palette=cm)
    kw = p.merge({}, ckw, s.rendering_kw)

    if not s.is_filled:
        warnings.warn("Bokeh does not support line contours.")

    h = p._fig.image(
        image=[z],
        x=minx,
        y=miny,
        dw=abs(maxx - minx),
        dh=abs(maxy - miny),
        **kw
    )
    handle.append(h)
    p._fig.add_tools(p.bokeh.models.HoverTool(
        tooltips=[("x", "$x"), ("y", "$y"), ("z", "@image")],
        renderers=[handle[0]]
    ))

    if s.colorbar:
        colormapper = p.bokeh.models.LinearColorMapper(
            palette=cm, low=minz, high=maxz)
        cbkw = dict(width=8, title=s.get_label(p._use_latex))
        colorbar = p.bokeh.models.ColorBar(
            color_mapper=colormapper, **cbkw)
        p._fig.add_layout(colorbar, "right")
        handle.append(colorbar)

    return handle


def _update_contour_helper(renderer, data, handle):
    s = renderer.series
    x, y, z = data
    minx, miny, minz = x.min(), y.min(), z.min()
    maxx, maxy, maxz = x.max(), y.max(), z.max()
    handle[0].data_source.data.update({"image": [z]})
    handle[0].glyph.x = minx
    handle[0].glyph.y = miny
    handle[0].glyph.dw = abs(maxx - minx)
    handle[0].glyph.dh = abs(maxy - miny)
    if s.colorbar:
        cb = handle[1]
        cb.color_mapper.update(low=minz, high=maxz)


class ContourRenderer(Renderer):
    draw_update_map = {
        _draw_contour_helper: _update_contour_helper
    }
