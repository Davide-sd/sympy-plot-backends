from spb.backends.matplotlib.renderers.renderer import MatplotlibRenderer
from spb.backends.utils import compute_streamtubes
from sympy.external import import_module


def _draw_implicit3d_isosurface_helper(renderer, data):
    skimage = import_module("skimage", import_kwargs={'fromlist': ['measure']})
    if skimage is None:
        raise ModuleNotFoundError(
            "`skimage` is a mandatory dependency for 3D implicit plots,"
            " which is currently missing.")

    p, s = renderer.plot, renderer.series
    mesh_x, mesh_y, mesh_z, f = data
    verts, faces, normals, values = skimage.measure.marching_cubes(f, 0)

    x = mesh_x[:, 0, 0]
    y = mesh_y[0, :, 0]
    z = mesh_z[0, 0, :]
    dx = abs(x[0] - x[1])
    dy = abs(y[0] - y[1])
    dz = abs(z[0] - z[1])
    verts[:, 0] = x[0] + verts[:, 0] * dx
    verts[:, 1] = y[0] + verts[:, 1] * dy
    verts[:, 2] = z[0] + verts[:, 2] * dz

    mkw = dict(linewidths=0.25, edgecolor="k", facecolors=next(p._cl))
    kw = p.merge({}, mkw, s.rendering_kw)

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = p.Poly3DCollection(verts[faces], **kw)
    p.ax.add_collection3d(mesh)

    return [mesh]


def _update_implicit3d_isosurface_helper(renderer, data, handle):
    raise NotImplementedError


class Implicit3DRenderer(MatplotlibRenderer):
    _sal = True
    draw_update_map = {
        _draw_implicit3d_isosurface_helper: _update_implicit3d_isosurface_helper
    }


def _draw_implicit3d_volume_helper(renderer, data):
    skimage = import_module("skimage", import_kwargs={'fromlist': ['measure']})
    if skimage is None:
        raise ModuleNotFoundError(
            "`skimage` is a mandatory dependency for 3D implicit plots,"
            " which is currently missing.")

    p, s = renderer.plot, renderer.series
    mesh_x, mesh_y, mesh_z, f = data
    x_edges = mesh_x[:, 0, 0]
    y_edges = mesh_y[0, :, 0]
    z_edges = mesh_z[0, 0, :]
    dx = abs(x_edges[1] - x_edges[0])
    dy = abs(y_edges[1] - y_edges[0])
    dz = abs(z_edges[1] - z_edges[0])

    x_min, x_max = float(s.range_x[1]), float(s.range_x[2])
    y_min, y_max = float(s.range_y[1]), float(s.range_y[2])
    z_min, z_max = float(s.range_z[1]), float(s.range_z[2])
    x_edges = p.np.linspace(x_min - dx / 2, x_max + dx / 2, s.n1)
    y_edges = p.np.linspace(y_min - dy / 2, y_max + dy / 2, s.n2)
    z_edges = p.np.linspace(z_min - dz / 2, z_max + dz / 2, s.n3)
    x, y, z = p.np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")

    mkw = dict(linewidths=0.25, edgecolor="k", facecolors=next(p._cl))
    kw = p.merge({}, mkw, s.rendering_kw)
    mesh = p.ax.voxels(x, y, z, f, **kw)

    return [mesh]


def _update_implicit3d_volume_helper(renderer, data, handle):
    raise NotImplementedError


class Implicit3DVoxelRenderer(MatplotlibRenderer):
    _sal = True
    draw_update_map = {
        _draw_implicit3d_volume_helper: _update_implicit3d_volume_helper
    }
