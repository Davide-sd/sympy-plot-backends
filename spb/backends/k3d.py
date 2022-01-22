from sympy.external import import_module
from spb.defaults import cfg
from spb.backends.base_backend import Plot
from spb.backends.utils import compute_streamtubes
from spb.utils import get_vertices_indices
import warnings
import os

k3d = import_module(
    'k3d',
    import_kwargs={'fromlist': ['helpers']},
    min_module_version='2.9.7',
    catch=(RuntimeError,))
cc = import_module(
    'colorcet',
    min_module_version='3.0.0',
    catch=(RuntimeError,))
matplotlib = import_module(
    'matplotlib',
    import_kwargs={'fromlist': ['tri']},
    min_module_version='1.1.0',
    catch=(RuntimeError,))


class K3DBackend(Plot):
    """A backend for plotting SymPy's symbolic expressions using K3D-Jupyter.

    Parameters
    ==========

    line_kw : dict, optional
        A dictionary of keywords/values which is passed to K3D's line
        functions to customize the appearance. Default to:
        `line_kw = dict(width=0.1, shader="mesh")`
        Set `use_cm=False` to switch to a solid color.

    quiver_kw : dict, optional
        A dictionary to customize the apppearance of quivers. Default to:
        `quiver_kw = dict(scale = 1)`.
        Set `use_cm=False` to switch to a solid color.

    show_label : boolean, optional
        Show/hide labels of the expressions. Default to False (labels not
        visible).

    stream_kw : dict, optional
        A dictionary to customize the apppearance of streamlines.
        Default to:
        `stream_kw = dict( width=0.1, shader='mesh' )`
        Refer to k3d.line for more options.
        Set `use_cm=False` to switch to a solid color.

    use_cm : boolean, optional
        If True, apply a color map to the meshes/surface. If False, solid
        colors will be used instead. Default to True.


    See also
    ========

    Plot, MatplotlibBackend, PlotlyBackend, BokehBackend
    """

    _library = "k3d"

    colormaps = [
        k3d.basic_color_maps.CoolWarm,
        k3d.matplotlib_color_maps.Plasma,
        k3d.matplotlib_color_maps.Winter,
        k3d.matplotlib_color_maps.Viridis,
        k3d.paraview_color_maps.Haze,
        k3d.matplotlib_color_maps.Summer,
        k3d.paraview_color_maps.Blue_to_Yellow,
    ]

    cyclic_colormaps = [cc.colorwheel, k3d.paraview_color_maps.Erdc_iceFire_H]

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._get_mode() != 0:
            raise ValueError(
                "Sorry, K3D backend only works within Jupyter Notebook")

        self._show_label = kwargs.get("show_label", False)
        self._bounds = []
        self._clipping = []
        self._handles = dict()

        self._init_cyclers()

        self._fig = k3d.plot(
            grid_visible=self.grid,
            menu_visibility=True,
            background_color=int(cfg["k3d"]["bg_color"]),
            grid_color=int(cfg["k3d"]["grid_color"]),
            label_color=int(cfg["k3d"]["label_color"]),
        )
        if (self.xscale == "log") or (self.yscale == "log"):
            warnings.warn(
                "K3D-Jupyter doesn't support log scales. We will "
                + "continue with linear scales."
            )
        self.plot_shown = False
        self._process_series(self._series)

    @staticmethod
    def _do_sum_kwargs(p1, p2):
        kw = p1._copy_kwargs()
        kw["show_label"] = (p1._show_label or p2._show_label)
        return kw

    @staticmethod
    def _int_to_rgb(RGBint):
        """Convert an integer number to an RGB tuple with components from 0
        to 255.

        https://stackoverflow.com/a/2262152/2329968
        """
        B = RGBint & 255
        G = (RGBint >> 8) & 255
        R = (RGBint >> 16) & 255
        return R, G, B

    @staticmethod
    def _rgb_to_int(RGB):
        """Convert an RGB tuple to an integer number.

        https://stackoverflow.com/q/2262100/2329968
        """
        R, G, B = RGB
        return R * 256 ** 2 + G * 256 + B

    @classmethod
    def _convert_to_int(cls, color):
        """Convert the provided RGB tuple with values from 0 to 1 to an
        integer number.
        """
        color = [int(c * 255) for c in color]
        return cls._rgb_to_int(color)

    def _process_series(self, series):
        np = import_module('numpy', catch=(RuntimeError,))
        mergedeep = import_module('mergedeep', catch=(RuntimeError,))
        merge = mergedeep.merge
        Triangulation = matplotlib.tri.Triangulation
        self._init_cyclers()
        self._fig.auto_rendering = False
        # clear data
        for o in self._fig.objects:
            self._fig.remove_class(o)

        for ii, s in enumerate(series):
            if s.is_3Dline and s.is_point:
                x, y, z, _ = s.get_data()
                positions = np.vstack([x, y, z]).T.astype(np.float32)
                a = dict(point_size=0.2, color=self._convert_to_int(next(self._cl)))
                kw = merge({}, a, s.rendering_kw)
                plt_points = k3d.points(positions=positions, **kw)
                plt_points.shader = "mesh"
                self._fig += plt_points

            elif s.is_3Dline:
                x, y, z, param = s.get_data()
                # K3D doesn't like masked arrays, so filled them with NaN
                x, y, z = [
                    np.ma.filled(t) if isinstance(t, np.ma.core.MaskedArray) else t
                    for t in [x, y, z]
                ]
                vertices = np.vstack([x, y, z]).T.astype(np.float32)
                # keyword arguments for the line object
                a = dict(
                    width=0.1,
                    name=s.label if self._show_label else None,
                    color=self._convert_to_int(next(self._cl)),
                    shader="mesh",
                )
                if self._use_cm:
                    a["attribute"] = (param.astype(np.float32),)
                    a["color_map"] = next(self._cm)
                    a["color_range"] = [s.start, s.end]
                kw = merge({}, a, s.rendering_kw)
                line = k3d.line(vertices, **kw)
                self._fig += line

            elif (s.is_3Dsurface and not s.is_domain_coloring):
                x, y, z = s.get_data()
                # K3D doesn't like masked arrays, so filled them with NaN
                x, y, z = [
                    np.ma.filled(t) if isinstance(t, np.ma.core.MaskedArray) else t
                    for t in [x, y, z]
                ]

                # TODO:
                # Can I use get_vertices_indices also for non parametric surfaces?
                if s.is_parametric:
                    vertices, indices = get_vertices_indices(x, y, z)
                    vertices = vertices.astype(np.float32)
                else:
                    x = x.flatten()
                    y = y.flatten()
                    z = z.flatten()
                    vertices = np.vstack([x, y, z]).T.astype(np.float32)
                    indices = Triangulation(x, y).triangles.astype(np.uint32)

                self._high_aspect_ratio(x, y, z)
                a = dict(
                    name=s.label if self._show_label else None,
                    side="double",
                    flat_shading=False,
                    wireframe=False,
                    color=self._convert_to_int(next(self._cl)),
                )
                if self._use_cm:
                    a["color_map"] = next(self._cm)
                    a["attribute"] = z.astype(np.float32)
                kw = merge({}, a, s.rendering_kw)
                surf = k3d.mesh(vertices, indices, **kw)

                self._fig += surf

            elif s.is_3Dvector and s.is_streamlines:
                xx, yy, zz, uu, vv, ww = s.get_data()
                # K3D doesn't like masked arrays, so filled them with NaN
                xx, yy, zz, uu, vv, ww = [
                    np.ma.filled(t) if isinstance(t, np.ma.core.MaskedArray)
                    else t for t in [xx, yy, zz, uu, vv, ww]]

                vertices, magn = compute_streamtubes(
                    xx, yy, zz, uu, vv, ww, s.rendering_kw)

                stream_kw = s.rendering_kw.copy()
                skw = dict(width=0.1, shader="mesh")
                if self._use_cm and ("color" not in stream_kw.keys()):
                    skw["color_map"] = next(self._cm)
                    skw["color_range"] = [np.nanmin(magn), np.nanmax(magn)]
                    skw["attribute"] = magn
                else:
                    col = stream_kw.pop("color", next(self._cl))
                    if not isinstance(col, int):
                        col = self._convert_to_int(col)
                    stream_kw["color"] = col

                kw = merge({}, skw, stream_kw)
                self._fig += k3d.line(
                    vertices.astype(np.float32), **kw)

            elif s.is_3Dvector:
                xx, yy, zz, uu, vv, ww = s.get_data()
                # K3D doesn't like masked arrays, so filled them with NaN
                xx, yy, zz, uu, vv, ww = [
                    np.ma.filled(t) if isinstance(t, np.ma.core.MaskedArray) else t
                    for t in [xx, yy, zz, uu, vv, ww]
                ]
                xx, yy, zz, uu, vv, ww = [
                    t.flatten().astype(np.float32) for t in [xx, yy, zz, uu, vv, ww]
                ]
                qkw = dict(scale=1)
                qkw = merge(qkw, s.rendering_kw)
                scale = qkw["scale"]
                magnitude = np.sqrt(uu ** 2 + vv ** 2 + ww ** 2)
                vectors = np.array((uu, vv, ww)).T * scale
                origins = np.array((xx, yy, zz)).T
                quiver_kw = s.rendering_kw
                if self._use_cm and ("color" not in quiver_kw.keys()):
                    colormap = next(self._cm)
                    colors = k3d.helpers.map_colors(magnitude, colormap, [])
                    self._handles[ii] = [qkw, colormap]
                else:
                    col = quiver_kw.get("color", next(self._cl))
                    if not isinstance(col, int):
                        col = self._convert_to_int(col)
                    colors = col * np.ones(len(magnitude))
                    self._handles[ii] = [qkw, None]
                vec_colors = np.zeros(2 * len(colors))
                for i, c in enumerate(colors):
                    vec_colors[2 * i] = c
                    vec_colors[2 * i + 1] = c
                vec_colors = vec_colors.astype(np.uint32)
                vec = k3d.vectors(
                    origins=origins - vectors / 2,
                    vectors=vectors,
                    colors=vec_colors,
                )
                self._fig += vec

            elif s.is_complex and s.is_3Dsurface:
                x, y, mag, arg, colors, colorscale = s.get_data()

                x, y, z = [t.flatten() for t in [x, y, mag]]
                vertices = np.vstack([x, y, z]).T.astype(np.float32)
                indices = Triangulation(x, y).triangles.astype(np.uint32)
                self._high_aspect_ratio(x, y, z)

                a = dict(
                    name=s.label if self._show_label else None,
                    side="double",
                    flat_shading=False,
                    wireframe=False,
                    color=self._convert_to_int(next(self._cl)),
                    color_range=[-np.pi, np.pi],
                )
                if self._use_cm:
                    colors = colors.reshape((-1, 3))
                    a["colors"] = [self._rgb_to_int(c) for c in colors]

                    r = []
                    loc = np.linspace(0, 1, colorscale.shape[0])
                    colorscale = colorscale / 255
                    for l, c in zip(loc, colorscale):
                        r.append(l)
                        r += list(c)

                    a["color_map"] = r
                    a["color_range"] = [-np.pi, np.pi]
                kw = merge({}, a, s.rendering_kw)
                surf = k3d.mesh(vertices, indices, **kw)

                self._fig += surf

            else:
                raise NotImplementedError(
                    "{} is not supported by {}\n".format(type(s), type(self).__name__)
                    + "K3D-Jupyter only supports 3D plots."
                )

            if self.update_rendering_kw and (kw is not None):
                s.rendering_kw = kw

        xl = self.xlabel if self.xlabel else "x"
        yl = self.ylabel if self.ylabel else "y"
        zl = self.zlabel if self.zlabel else "z"
        self._fig.axes = [xl, yl, zl]

        if self.title:
            self._fig += k3d.text2d(
                self.title, position=[0.025, 0.015], color=0, size=1, label_box=False
            )
        self._fig.auto_rendering = True

    def _high_aspect_ratio(self, x, y, z):
        """Look for high aspect ratio meshes, where (dz >> dx, dy) and
        eventually set the bounds around the mid point of the mesh in order
        to improve visibility. Bounds will be used to set the camera position.
        """
        mz, Mz, meanz = z.min(), z.max(), z.mean()
        mx, Mx = x.min(), x.max()
        my, My = y.min(), y.max()
        dx, dy, dz = (Mx - mx), (My - my), (Mz - mz)

        # thresholds
        t1, t2 = 10, 3
        if (dz / dx >= t1) and (dz / dy >= t1) and (self.zlim is None):
            if abs(Mz / meanz) > t1:
                Mz = meanz + t2 * max(dx, dy)
            if abs(mz / meanz) > t1:
                mz = meanz - t2 * max(dx, dy)
            self._bounds.append([mx, Mx, my, My, mz, Mz])
        elif self.zlim:
            self._bounds.append([mx, Mx, my, My, self.zlim[0], self.zlim[1]])

    def _update_interactive(self, params):
        np = import_module('numpy', catch=(RuntimeError,))

        # self._fig.auto_rendering = False
        for i, s in enumerate(self.series):
            if s.is_interactive:
                self.series[i].params = params

                if s.is_3Dline and s.is_point:
                    x, y, z, _ = self.series[i].get_data()
                    positions = np.vstack([x, y, z]).T.astype(np.float32)
                    self._fig.objects[i].positions = positions

                elif s.is_3Dline:
                    x, y, z, _ = self.series[i].get_data()
                    vertices = np.vstack([x, y, z]).T.astype(np.float32)
                    self._fig.objects[i].vertices = vertices

                elif s.is_3Dsurface and (not s.is_domain_coloring):
                    x, y, z = self.series[i].get_data()
                    x, y, z = [t.flatten().astype(np.float32) for t in [x, y, z]]
                    vertices = np.vstack([x, y, z]).astype(np.float32)
                    self._fig.objects[i].vertices = vertices.T
                    self._fig.objects[i].attribute = z
                    self._fig.objects[i].color_range = [z.min(), z.max()]

                elif s.is_vector and s.is_3D:
                    if s.is_streamlines:
                        raise NotImplementedError

                    xx, yy, zz, uu, vv, ww = self.series[i].get_data()
                    xx, yy, zz, uu, vv, ww = [
                        t.flatten().astype(np.float32) for t in [xx, yy, zz, uu, vv, ww]
                    ]
                    magnitude = np.sqrt(uu ** 2 + vv ** 2 + ww ** 2)
                    qkw, colormap = self._handles[i]
                    if colormap is not None:
                        colors = k3d.helpers.map_colors(magnitude, colormap, [])
                        vec_colors = np.zeros(2 * len(colors))
                        for j, c in enumerate(colors):
                            vec_colors[2 * j] = c
                            vec_colors[2 * j + 1] = c
                        vec_colors = vec_colors.astype(np.uint32)
                        self.fig.objects[i].colors = vec_colors

                    scale = qkw["scale"]
                    vectors = np.array((uu, vv, ww)).T * scale
                    self.fig.objects[i].vectors = vectors

                elif s.is_complex and s.is_3Dsurface:
                    x, y, mag, _, colors, _ = s.get_data()
                    x, y, z = [t.flatten().astype(np.float32) for t in [x, y, mag]]
                    vertices = np.vstack([x, y, z]).astype(np.float32)
                    self._fig.objects[i].vertices = vertices.T
                    if self._use_cm:
                        colors = colors.reshape((-1, 3))
                        colors = [self._rgb_to_int(c) for c in colors]
                        self._fig.objects[i].colors = colors

        # self._fig.auto_rendering = True

    def show(self):
        """Visualize the plot on the screen."""
        np = import_module('numpy', catch=(RuntimeError,))

        if len(self._fig.objects) != len(self.series):
            self._process_series(self._series)
        self.plot_shown = True

        if len(self._bounds) > 0:
            # when there are very high aspect ratio meshes, or when zlim has
            # been set, we compute a new camera position to improve user
            # experience
            self._fig.camera_auto_fit = False
            bounds = np.array(self._bounds)
            bounds = np.dstack(
                [np.min(bounds[:, 0::2], axis=0), np.max(bounds[:, 1::2], axis=0)]
            ).flatten()
            self._fig.camera = self._fig.get_auto_camera(1.5, 40, 60, bounds)

        self._fig.display()
        if self.zlim:
            self._fig.clipping_planes = [
                [0, 0, 1, self.zlim[0]],
                [0, 0, -1, self.zlim[1]],
            ]

    def save(self, path, **kwargs):
        """Export the plot to a static picture or to an interactive html file.

        Notes
        =====

        K3D-Jupyter is only capable of exporting:

        1. '.png' pictures: refer to [#fn1]_ to visualize the available
           keyword arguments.
        2. '.html' files: when exporting a fully portable html file, by
           default the required Javascript libraries will be loaded with a
           CDN. Set `include_js=True` to include all the javascript code in
           the html file: this will create a bigger file size, but can be
           run without internet connection.

        References
        ==========
        .. [#fn1] https://k3d-jupyter.org/k3d.html#k3d.plot.Plot.fetch_screenshot

        """
        if not self.plot_shown:
            raise ValueError(
                "K3D-Jupyter requires the plot to be shown on the screen "
                + "before saving it."
            )

        ext = os.path.splitext(path)[1]
        if not ext:
            path += ".png"

        if ext in [".html", "htm"]:
            with open(path, 'w') as f:
                include_js = kwargs.pop("include_js", False)
                self.fig.snapshot_include_js = include_js
                f.write(self.fig.get_snapshot(**kwargs))
        elif ext == ".png":
            @self._fig.yield_screenshots
            def _func():
                self._fig.fetch_screenshot(**kwargs)
                screenshot = yield
                with open(path, "wb") as f:
                    f.write(screenshot)

            _func()
        else:
            raise ValueError(
                "K3D-Jupyter can only export '.png' images or " +
                "html files.")


KB = K3DBackend
