from spb.backends.base_renderer import Renderer


def _draw_implicit3d_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np

    _, _, _, r = data
    xmin, xmax = s.start_x, s.end_x
    ymin, ymax = s.start_y, s.end_y
    zmin, zmax = s.start_z, s.end_z
    a = dict(
        xmin=xmin, xmax=xmax,
        ymin=ymin, ymax=ymax,
        zmin=zmin, zmax=zmax,
        compression_level=9,
        level=0.0, flat_shading=True,
        color=p._convert_to_int(next(p._cl))
    )
    kw = p.merge({}, a, s.rendering_kw)
    plt_iso = p.k3d.marching_cubes(r.T.astype(np.float32), **kw)

    p._fig += plt_iso
    return plt_iso


def _update_implicit3d_helper(renderer, data, handle):
    raise NotImplementedError


class Implicit3DRenderer(Renderer):
    draw_update_map = {
        _draw_implicit3d_helper: _update_implicit3d_helper
    }

def _draw_implicit3dvoxel_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np

    r = data
    xmin, xmax = s.start_x, s.end_x
    ymin, ymax = s.start_y, s.end_y
    zmin, zmax = s.start_z, s.end_z
    a = dict(
        bounds=[xmin, xmax, ymin, ymax, zmin, zmax],
    )
    kw = p.merge({}, a, s.rendering_kw)
    plt_vox = p.k3d.voxels(r, **kw)
    plt_empty = p.k3d.voxels(np.ones((1, 1, 1), dtype=np.uint8),
                                opacity=0,
                                outlines=False,
                                bounds=[xmin, xmax, ymin, ymax, zmin, zmax])

    p._fig += plt_vox
    p._fig += plt_empty
    return plt_vox


def _update_implicit3dvoxel_helper(renderer, data, handle):
    raise NotImplementedError


class Implicit3DRVoxelenderer(Renderer):
    draw_update_map = {
        _draw_implicit3dvoxel_helper: _update_implicit3dvoxel_helper
    }
