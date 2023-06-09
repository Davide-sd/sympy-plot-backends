from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer


class GenericRenderer(MatplotlibRenderer):
    def draw(self):
        s = self.series
        p = self.plot
        if s.type == "markers":
            kw = p.merge({}, {"color": next(p._cl)}, s.rendering_kw)
            p._ax.plot(*s.args, **kw)
        elif s.type == "annotations":
            p._ax.annotate(*s.args, **s.rendering_kw)
        elif s.type == "fill":
            kw = p.merge({}, {"color": next(p._cl)}, s.rendering_kw)
            p._ax.fill_between(*s.args, **kw)
        elif s.type == "rectangles":
            kw = p.merge({}, {"color": next(p._cl)}, s.rendering_kw)
            p._ax.add_patch(
                p.matplotlib.patches.Rectangle(*s.args, **kw))

    def update(self, params):
        raise NotImplementedError
