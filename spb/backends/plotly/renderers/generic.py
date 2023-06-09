from spb.backends.base_renderer import Renderer


class GenericRenderer(Renderer):
    def draw(self):
        s = self.series
        p = self.plot
        if s.type == "markers":
            kw = p.merge({}, {"line_color": next(p._cl)}, s.rendering_kw)
            p._fig.add_trace(p.go.Scatter(*s.args, **kw))
        elif s.type == "annotations":
            kw = p.merge({}, {
                "line_color": next(p._cl),
                "mode": "text",
                }, s.rendering_kw)
            p._fig.add_trace(p.go.Scatter(*s.args, **kw))
        elif s.type == "fill":
            kw = p.merge({}, {
                "line_color": next(p._cl),
                "fill": "tozeroy"
                }, s.rendering_kw)
            p._fig.add_trace(p.go.Scatter(*s.args, **kw))
        elif s.type == "rectangles":
            p._fig.add_shape(*s.args, **s.rendering_kw)

    def update(self, params):
        raise NotImplementedError
