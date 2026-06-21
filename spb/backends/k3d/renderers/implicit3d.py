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

    # NOTE: here I'm going to use voxels. K3D also exposes the volume function:
    # https://k3d-jupyter.org/reference/factory.volume.html
    # however, at the time of writing this, it doesn't scale the data to the
    # appropriate boundaries, hence I'll only use voxels.

    mesh_x, mesh_y, mesh_z, f = data
    xmin, xmax = s.start_x, s.end_x
    ymin, ymax = s.start_y, s.end_y
    zmin, zmax = s.start_z, s.end_z
    color = p._convert_to_int(next(p._cl))

    a = dict(
        bounds=[xmin, xmax, ymin, ymax, zmin, zmax],
        color_map=[color],
        outlines=False
    )
    kw = p.merge({}, a, s.rendering_kw)
    plt_vox = p.k3d.voxels(
        f.transpose(2,1,0).astype(np.uint8),
        **kw
    )
    p._fig += plt_vox

    return plt_vox


def _update_implicit3dvoxel_helper(renderer, data, handle):
    raise NotImplementedError


class Implicit3DVoxelRenderer(Renderer):
    draw_update_map = {
        _draw_implicit3dvoxel_helper: _update_implicit3dvoxel_helper
    }
