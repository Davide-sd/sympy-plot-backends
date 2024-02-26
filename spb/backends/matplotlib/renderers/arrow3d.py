from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.backends.matplotlib.renderers.vector2d import (
    _draw_vector2d_helper, _update_vector2d_helper
)


class Arrow3D(FancyArrowPatch):
    """Draws a 3D arrow based on FancyArrowPatch.

    Reference
    =========

    [1] https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    [2] https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html
    """

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _draw_arrow_3d(renderer, data):
    p, s = renderer.plot, renderer.series
    xx, yy, zz, uu, vv, ww = data
    mpatches = p.matplotlib.patches

    arrowstyle = mpatches.ArrowStyle(
        "fancy", head_width=2, head_length=3)
    pkw = dict(
        mutation_scale=2,
        arrowstyle=arrowstyle,
        shrinkA=0, shrinkB=0,
        color=next(p._cl)
    )
    kw = p.merge({}, pkw, s.rendering_kw)
    arrow = Arrow3D(xx, yy, zz, uu, vv, ww, **kw)
    p._ax.add_patch(arrow)

    if s.show_in_legend:
        proxy_artist = p.Line2D(
            [], [],
            color=kw["color"], label=s.get_label(p._use_latex)
        )
        p._legend_handles.append(proxy_artist)
    return [arrow, kw]


def _update_arrow3d(renderer, data, handle):
    """
    NOTE: Altough FancyArrowPatch is able to update an arrow position in a 2D
    space, it doesn't work with Arrow3D. Hence, let's remove the previous
    arrow and add a new one.
    """
    p = renderer.plot
    xx, yy, zz, uu, vv, ww = data
    handle[0].remove()
    kw = handle[1]
    arrow = Arrow3D(xx, yy, zz, uu, vv, ww, **kw)
    p._ax.add_patch(arrow)
    handle[0] = arrow


class Arrow3DRendererFancyArrowPatch(MatplotlibRenderer):
    draw_update_map = {
        _draw_arrow_3d: _update_arrow3d
    }
