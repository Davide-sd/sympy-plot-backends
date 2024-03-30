from spb.backends.base_renderer import Renderer


class GenericRenderer(Renderer):
    def draw(self):
        s = self.series
        p = self.plot

        if s.type == "markers":
            kw = p.merge({}, {"color": next(p._cl)}, s.rendering_kw)
            h = p._fig.scatter(*s.args, **kw)
        elif s.type == "annotations":
            h = p.bokeh.models.LabelSet(*s.args, **s.rendering_kw)
            p._fig.add_layout(h)
        elif s.type == "fill":
            kw = p.merge({}, {"fill_color": next(p._cl)}, s.rendering_kw)
            h = p._fig.varea(*s.args, **kw)
        elif s.type == "rectangles":
            kw = p.merge({}, {"fill_color": next(p._cl)}, s.rendering_kw)
            h = p.bokeh.models.Rect(**kw)
            p._fig.add_glyph(*s.args, h)
        else:
            h = None
        return [h]

    def update(self, params):
        raise NotImplementedError
