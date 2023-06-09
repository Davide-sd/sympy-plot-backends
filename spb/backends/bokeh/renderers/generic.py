from spb.backends.base_renderer import Renderer


class GenericRenderer(Renderer):
    def draw(self):
        s = self.series
        p = self.plot
        
        if s.type == "markers":
            kw = p.merge({}, {"color": next(p._cl)}, s.rendering_kw)
            p._fig.scatter(*s.args, **kw)
        elif s.type == "annotations":
            p._fig.add_layout(
                p.bokeh.models.LabelSet(*s.args, **s.rendering_kw))
        elif s.type == "fill":
            kw = p.merge({}, {"fill_color": next(p._cl)}, s.rendering_kw)
            p._fig.varea(*s.args, **kw)
        elif s.type == "rectangles":
            kw = p.merge({}, {"fill_color": next(p._cl)}, s.rendering_kw)
            glyph = p.bokeh.models.Rect(**kw)
            p._fig.add_glyph(*s.args, glyph)

    def update(self, params):
        raise NotImplementedError
