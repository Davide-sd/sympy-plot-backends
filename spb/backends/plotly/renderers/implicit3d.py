from sympy.external import import_module
from spb.backends.base_renderer import Renderer
import numpy as np

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


def _draw_implicit3dvoxel_helper(renderer, data):
    p, s = renderer.plot, renderer.series
    np = p.np

    # NOTE: plotly exposes the volume plots:
    # https://plotly.com/python/3d-volume-plots/
    # However, it is very difficult to use them in this context, where people
    # can write (f(x) > 0) | (g(x) < 2). Hence, I'll just use voxels.

    x_edges = np.linspace(float(s.range_x[1]), float(s.range_x[2]), s.n1)
    y_edges = np.linspace(float(s.range_y[1]), float(s.range_y[2]), s.n2)
    z_edges = np.linspace(float(s.range_z[1]), float(s.range_z[2]), s.n3)
    mesh_x, mesh_y, mesh_z, f = data

    col = next(p._cl)
    colorscale = [[0, col], [1, col]]

    mesh = voxel_mesh(
        f, x_edges, y_edges, z_edges, color=col, **s.rendering_kw)
    p._fig.add_trace(mesh)
    p._colorbar_counter += 1
    return len(p._fig.data) - 1



def _update_implicit3dvoxel_helper(renderer, data, handle):
    raise NotImplementedError


class Implicit3DVoxelRenderer(Renderer):
    draw_update_map = {
        _draw_implicit3dvoxel_helper: _update_implicit3dvoxel_helper
    }


def voxel_mesh(
    mask, x_edges, y_edges, z_edges, color="steelblue", opacity=1.0,
    flatshading=True, cull_hidden_faces=True, dedupe_vertices=True,
    **mesh_kwargs
):
    """
    Build a single go.Mesh3d trace rendering a cube for every
    True entry in `mask`.

    Disclaimer: this function was built with Claude Sonnet 4.6 and ChatGPT 5.5.

    Parameters
    ----------
    mask : ndarray of bool, shape (nx, ny, nz)
        Mask[i, j, k] = True means draw a cube occupying the cell between
        x_edges[i]:x_edges[i+1], y_edges[j]:y_edges[j+1], z_edges[k]:z_edges[k+1].
    x_edges, y_edges, z_edges : ndarray
        1D arrays of cell boundaries (length nx+1, ny+1, nz+1).
    color : str
        Single color for all voxels (string, e.g. "steelblue" or "#1f77b4").
    opacity : float
    flatshading : bool
        Keeps each cube face looking flat/blocky rather than smooth-shaded.
        On its own this is not always sufficient -- see the `lighting` note below.
    cull_hidden_faces : bool, default True.
        Skips faces shared by two filled voxels (interior faces nobody will
        ever see). This is the standard voxel-engine trick and massively
        reduces triangle count for solid blobs.
    dedupe_vertices : bool, default True.
        Merges duplicate (x, y, z) points so vertices shared between adjacent
        faces are stored once instead of once per face. Doesn't change triangle
        count or appearance, but typically cuts the serialized figure size
        (and browser memory) by an order of magnitude or more at high
        resolution, since Mesh3d's vertex arrays are the dominant cost.
        Implemented as a single vectorized np.unique pass,
        and -- importantly -- vertices are only ever built for faces that
        survive culling in the first place, so the array being deduplicated
        is already small; this avoids a return-to-Python-loop-speed dedup
        pass on a much larger, mostly-discarded array.

        One side effect: with dedupe on, many faces now share exact vertices
        at voxel edges (by construction -- that's the point of deduping).
        Plotly's WebGL renderer computes per-vertex normals by averaging the
        normals of every triangle touching a vertex, then has a separate
        epsilon-based merge step for near-identical normals before flat
        shading is applied. At a voxel edge, two perpendicular faces meet at
        a shared vertex whose averaged normal points diagonally between
        them, and that merge step can still blend lighting across the edge
        even with flatshading=True, showing up as a faint gradient/halo right
        on the seams instead of a crisp line. Passing
        lighting=dict(vertexnormalsepsilon=0, facenormalsepsilon=0) disables
        that merge step entirely, which is why it's the default below -- it
        forces every triangle to use its own exact face normal with no
        cross-triangle blending. Override by passing your own `lighting=...`
        in mesh_kwargs if you want Plotly's default behavior back.
    **mesh_kwargs : passed through to go.Mesh3d (e.g. lighting, etc.)

    Returns
    -------
    go.Mesh3d trace.
    """
    np = import_module("numpy")
    plotly = import_module(
            'plotly',
            import_kwargs={'fromlist': ['graph_objects', 'figure_factory']},
            warn_not_installed=True,
            min_module_version='5.0.0')
    go = plotly.graph_objects

    # 8 corners of a unit cube, in a fixed order, indexed by the face definitions below
    _UNIT_CORNERS = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ], dtype=float)

    # Outward-facing quads for each of the 6 faces (4 corner indices into
    # _UNIT_CORNERS, in winding order), plus which neighbor cell (delta i,j,k) to
    # check when deciding whether that face is hidden.
    # Corner numbering: 0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0) 4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
    _FACES = [
        ((-1, 0, 0), [0, 4, 7, 3]),  # x- face
        ((1, 0, 0), [1, 2, 6, 5]),  # x+ face
        ((0, -1, 0), [0, 1, 5, 4]),  # y- face
        ((0, 1, 0), [3, 7, 6, 2]),  # y+ face
        ((0, 0, -1), [0, 3, 2, 1]),  # z- face
        ((0, 0, 1), [4, 5, 6, 7]),  # z+ face
    ]
    # Two triangles covering a quad's 4 local corners (0,1,2,3), fan-split from corner 0
    _QUAD_TRIS_LOCAL = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)


    idx = np.argwhere(mask)
    n = len(idx)
    if n == 0:
        return go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], color=color, opacity=opacity)

    nx, ny, nz = mask.shape

    if not cull_hidden_faces:
        face_keep = np.ones((n, 6), dtype=bool)
    else:
        face_keep = np.empty((n, 6), dtype=bool)
        for f, (delta, _) in enumerate(_FACES):
            ni = idx[:, 0] + delta[0]
            nj = idx[:, 1] + delta[1]
            nk = idx[:, 2] + delta[2]
            in_bounds = (
                (ni >= 0) & (ni < nx) &
                (nj >= 0) & (nj < ny) &
                (nk >= 0) & (nk < nz)
            )
            neighbor_filled = np.zeros(n, dtype=bool)
            if in_bounds.any():
                neighbor_filled[in_bounds] = mask[ni[in_bounds], nj[in_bounds], nk[in_bounds]]
            # keep this face only if there's no filled neighbor on that side
            face_keep[:, f] = ~neighbor_filled

    # Lower/upper bounds of each occupied cell
    x0 = x_edges[idx[:, 0]]
    x1 = x_edges[idx[:, 0] + 1]
    y0 = y_edges[idx[:, 1]]
    y1 = y_edges[idx[:, 1] + 1]
    z0 = z_edges[idx[:, 2]]
    z1 = z_edges[idx[:, 2] + 1]

    lo = np.stack([x0, y0, z0], axis=1)  # (n, 3)
    span = np.stack([x1 - x0, y1 - y0, z1 - z0], axis=1)  # (n, 3)

    # Build vertices/triangles ONLY for faces that survived culling. This is
    # the key difference from a naive "build all 8 corners of every cube,
    # then dedupe": a fully-interior cube touching filled neighbors on all 6
    # sides contributes 0 rows here instead of 8 rows that np.unique would
    # later have to sort through and discard. Each kept face gets exactly 4
    # corner vertices (still duplicated across adjacent faces at this stage --
    # the dedupe step below merges those).
    vert_chunks = []
    tri_chunks = []
    running_offset = 0
    for f, (_, quad_corners) in enumerate(_FACES):
        keep = face_keep[:, f]
        if not keep.any():
            continue
        corners = _UNIT_CORNERS[quad_corners]  # (4, 3)
        # (m, 4, 3): one quad of 4 vertices per kept face
        quad_verts = lo[keep][:, None, :] + span[keep][:, None, :] * corners[None, :, :]
        m = quad_verts.shape[0]
        vert_chunks.append(quad_verts.reshape(-1, 3))

        offs = (np.arange(m) * 4 + running_offset)[:, None, None]  # (m, 1, 1)
        tris = _QUAD_TRIS_LOCAL[None, :, :] + offs  # (m, 2, 3)
        tri_chunks.append(tris.reshape(-1, 3))
        running_offset += m * 4

    if vert_chunks:
        verts = np.concatenate(vert_chunks, axis=0)
        all_tris = np.concatenate(tri_chunks, axis=0)
    else:
        verts = np.empty((0, 3))
        all_tris = np.empty((0, 3), dtype=int)

    if dedupe_vertices and len(verts) > 0:
        # Merge identical (x, y, z) rows in one vectorized pass. inverse[i] is
        # the index into the deduped array for original vertex i, so remapping
        # the triangle indices through `inverse` repoints every triangle at
        # the shared vertex without ever touching vertices in a Python loop.
        # Because we only built vertices for kept faces above, every row here
        # is referenced by at least one triangle -- no separate "drop unused
        # vertices" pass is needed.
        verts, inverse = np.unique(verts, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)  # numpy >=2.0 returns a column vector here
        all_tris = inverse[all_tris]

    # Disable Plotly's epsilon-based normal-merging so flatshading actually
    # produces crisp edges at shared voxel-face vertices (see docstring).
    # Respect an explicit lighting= the caller passed in mesh_kwargs instead
    # of silently overriding it.
    mesh_kwargs.setdefault("lighting", dict(vertexnormalsepsilon=0, facenormalsepsilon=0))

    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=all_tris[:, 0], j=all_tris[:, 1], k=all_tris[:, 2],
        color=color,
        opacity=opacity,
        flatshading=flatshading,
        **mesh_kwargs,
    )
