from spb.backends.base_renderer import Renderer
from spb.backends.utils import get_seeds_points
import warnings


def _draw_vector3d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np
    if s.color_func is not None:
        warnings.warn(
            "PlotlyBackend doesn't support custom "
            "coloring of 2D/3D quivers or streamlines plots. "
            "`color_func` will not be used."
        )

    xx, yy, zz, uu, vv, ww = data
    if s.is_streamlines:
        stream_kw = s.rendering_kw.copy()
        seeds_points = get_seeds_points(
            xx, yy, zz, uu, vv, ww, to_numpy=True, **stream_kw)

        skw = dict(
            colorscale=(
                next(p._cm) if s.use_cm else p._solid_colorscale(s)),
            sizeref=0.3,
            showscale=s.use_cm and s.colorbar,
            colorbar=p._create_colorbar(s.get_label(p.use_latex)),
            starts=dict(
                x=seeds_points[:, 0],
                y=seeds_points[:, 1],
                z=seeds_points[:, 2],
            ),
        )

        # remove rendering-unrelated keywords
        for _k in ["starts", "max_prop", "npoints", "radius"]:
            if _k in stream_kw.keys():
                stream_kw.pop(_k)

        kw = p.merge({}, skw, stream_kw)

        handle = p.go.Streamtube(
                x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
                u=uu.flatten(), v=vv.flatten(), w=ww.flatten(), **kw)
    else:
        mag = np.sqrt(uu**2 + vv**2 + ww**2)
        if s.normalize:
            # NOTE/TODO: as of Plotly 5.9.0, it is impossible
            # to set the color of cones. Hence, by applying the
            # normalization, all cones will have the same
            # color.
            uu, vv, ww = [t / mag for t in [uu, vv, ww]]
        qkw = dict(
            showscale=s.colorbar,
            colorscale=next(p._cm),
            sizemode="absolute",
            sizeref=40,
            colorbar=p._create_colorbar(s.get_label(p.use_latex)),
            cmin=mag.min(),
            cmax=mag.max(),
        )
        kw = p.merge({}, qkw, s.rendering_kw)

        handle = p.go.Cone(
                x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
                u=uu.flatten(), v=vv.flatten(), w=ww.flatten(), **kw)
    p._fig.add_trace(handle)
    return len(p._fig.data) - 1


def _update_vector3d_helper(renderer, data, idx):
    p, s = renderer.plot, renderer.series
    np = p.np
    handle = p.fig.data[idx]

    if s.is_streamlines:
        raise NotImplementedError
    x, y, z, u, v, w = data
    mag = np.sqrt(u**2 + v**2 + w**2)
    if s.normalize:
        # NOTE/TODO: as of Plotly 5.9.0, it is impossible
        # to set the color of cones. Hence, by applying the
        # normalization, all cones will have the same
        # color.
        u, v, w = [t / mag for t in [u, v, w]]
    handle["x"] = x.flatten()
    handle["y"] = y.flatten()
    handle["z"] = z.flatten()
    handle["u"] = u.flatten()
    handle["v"] = v.flatten()
    handle["w"] = w.flatten()
    handle["cmin"] = mag.min()
    handle["cmax"] = mag.max()


class Vector3DRenderer(Renderer):
    draw_update_map = {
        _draw_vector3d_helper: _update_vector3d_helper
    }
