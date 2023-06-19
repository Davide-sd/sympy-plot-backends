from spb.backends.base_renderer import Renderer


def _draw_implicit3d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    xx, yy, zz, rr = data
    # create a solid color
    col = next(p._cl)
    colorscale = [[0, col], [1, col]]
    skw = dict(
        isomin=0,
        isomax=0,
        showscale=False,
        colorscale=colorscale
    )
    kw = p.merge({}, skw, s.rendering_kw)
    handle = p.go.Isosurface(
        x=xx.flatten(),
        y=yy.flatten(),
        z=zz.flatten(),
        value=rr.flatten(), **kw
    )
    p._fig.add_trace(handle)
    p._colorbar_counter += 1
    return len(p._fig.data) - 1


def _update_implicit3d_helper(renderer, data, handle):
    raise NotImplementedError


class Implicit3DRenderer(Renderer):
    draw_update_map = {
        _draw_implicit3d_helper: _update_implicit3d_helper
    }
