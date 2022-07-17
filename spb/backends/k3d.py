import os
from spb.defaults import cfg
from spb.backends.base_backend import Plot
from spb.backends.utils import compute_streamtubes
from spb.utils import get_vertices_indices
from sympy.external import import_module
import warnings


class K3DBackend(Plot):
    """A backend for plotting SymPy's symbolic expressions using K3D-Jupyter.

    Parameters
    ==========

    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to Matplotlib's plot
        functions to customize the appearance of lines, surfaces, images,
        contours, quivers, streamlines...
        To learn more about customization:

        * Default options for line plots:
          ``dict(width=0.1, shader="mesh")``.
          Set ``use_cm=False`` to switch to a solid color.

        * Default options for quiver plots:
          ``dict(scale = 1, pivot = "mid")``. The keys to this
          dictionary are:

          - ``scale``: a float number acting as a scale multiplier.
          - ``pivot``: indicates the part of the arrow that is anchored to the
            X, Y, Z grid. It can be ``"tail", "mid", "middle", "head"``.
          - ``color``: set a solid color by specifying an integer color. If this
            key is not provided, a default color or colormap is used, depenging
            on the value of ``use_cm``.

          Set ``use_cm=False`` to switch to a default solid color.

        * Default options for streamline plots:
          ``dict( width=0.1, shader='mesh' )``.
          Refer to k3d.line for more options.
          Set ``use_cm=False`` to switch to a solid color.

        * To customize surface plots, refers to:

          - k3d.mesh function for 3D surface and parametric surface plots.
          - k3d.marching_cubes function for 3D implicit plots.

    show_label : boolean, optional
        Show/hide labels of the expressions. Default to False (labels not
        visible).

    use_cm : boolean, optional
        If True, apply a color map to the meshes/surface. If False, solid
        colors will be used instead. Default to True.


    Notes
    =====

    After the installation of this plotting module, try one of the examples
    of ``plot3d`` with this backend. If no figure is visible in the output
    cell, follow this procedure:

    1. Save the Notebook.
    2. Close Jupyter server.
    3. Run the following commands, which are going to install the Jupyter
       extension for K3D:

       * jupyter nbextension install --user --py k3d
       * jupyter nbextension enable --user --py k3d

    4. Restart `jupyter notebook`
    5. Open the previous notebook and execute the plot command.


    See also
    ========

    Plot, MatplotlibBackend, PlotlyBackend, BokehBackend, plot3d
    """

    _library = "k3d"

    colormaps = []
    cyclic_colormaps = []

    # quivers-pivot offsets
    _qp_offset = {"tail": 0, "mid": 0.5, "middle": 0.5, "tip": 1}

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.k3d = k3d = import_module(
            'k3d',
            import_kwargs={'fromlist': ['helpers', 'objects']},
            min_module_version='2.9.7')
        cc = import_module(
            'colorcet',
            min_module_version='3.0.0')
        self.matplotlib = import_module(
            'matplotlib',
            import_kwargs={'fromlist': ['tri', 'cm']},
            min_module_version='1.1.0',
            catch=(RuntimeError,))
        cm = self.matplotlib.cm

        self.colorloop = cm.tab10.colors
        self.colormaps = [
            k3d.basic_color_maps.CoolWarm,
            k3d.matplotlib_color_maps.Plasma,
            k3d.matplotlib_color_maps.Winter,
            k3d.matplotlib_color_maps.Viridis,
            k3d.paraview_color_maps.Haze,
            k3d.matplotlib_color_maps.Summer,
            k3d.paraview_color_maps.Blue_to_Yellow,
        ]
        self.cyclic_colormaps = [
            cc.colorwheel, k3d.paraview_color_maps.Erdc_iceFire_H]

        self._init_cyclers()
        super().__init__(*args, **kwargs)
        if self._get_mode() != 0:
            raise ValueError(
                "Sorry, K3D backend only works within Jupyter Notebook")
        self._use_latex = kwargs.get("use_latex", cfg["k3d"]["use_latex"])
        self._set_labels("%s")

        self._show_label = kwargs.get("show_label", False)
        self._bounds = []
        self._clipping = []
        self._handles = dict()
        self.grid = kwargs.get("grid", cfg["k3d"]["grid"])

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

    @property
    def fig(self):
        """Returns the figure."""
        # K3D title is an object in the figure
        n = len(self._fig.objects)
        if self.title is not None:
            n -= 1
        if len(self.series) != n:
            # if the backend was created without showing it
            self.process_series()
        return self._fig

    def process_series(self):
        """ Loop over data series, generates numerical data and add it to the
        figure.
        """
        # this is necessary in order for the series to be added even if
        # show=False
        self._process_series(self._series)

    def _set_piecewise_color(self, s, color):
        """Set the color to the given series"""
        raise NotImplementedError

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
        np = import_module('numpy')
        merge = self.merge
        Triangulation = self.matplotlib.tri.Triangulation
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
                plt_points = self.k3d.points(positions=positions, **kw)
                plt_points.shader = "mesh"
                self._fig += plt_points

            elif s.is_3Dline:
                x, y, z, param = s.get_data()
                vertices = np.vstack([x, y, z]).T.astype(np.float32)
                # keyword arguments for the line object
                a = dict(
                    width=0.1,
                    name=self._get_series_label(s, "%s") if self._show_label else None,
                    color=self._convert_to_int(next(self._cl)) if s.line_color is None else s.line_color,
                    shader="mesh",
                )
                if s.use_cm:
                    a["attribute"] = (param.astype(np.float32),)
                    a["color_map"] = next(self._cm)
                    a["color_range"] = [param.min(), param.max()]
                kw = merge({}, a, s.rendering_kw)
                line = self.k3d.line(vertices, **kw)
                self._fig += line

            elif (s.is_3Dsurface and (not s.is_domain_coloring) and (not s.is_implicit)):
                if s.is_parametric:
                    x, y, z, u, v = s.get_data()
                    vertices, indices = get_vertices_indices(x, y, z)
                    vertices = vertices.astype(np.float32)
                    attribute = s.eval_color_func(vertices[:, 0], vertices[:, 1], vertices[:, 2], u.flatten().astype(np.float32), v.flatten().astype(np.float32))
                else:
                    x, y, z = s.get_data()
                    x = x.flatten()
                    y = y.flatten()
                    z = z.flatten()
                    vertices = np.vstack([x, y, z]).T.astype(np.float32)
                    indices = Triangulation(x, y).triangles.astype(np.uint32)
                    attribute = s.eval_color_func(vertices[:, 0], vertices[:, 1], vertices[:, 2])

                self._high_aspect_ratio(x, y, z)
                a = dict(
                    name=s.get_label(self._use_latex, "%s") if self._show_label else None,
                    side="double",
                    flat_shading=False,
                    wireframe=False,
                    color=self._convert_to_int(next(self._cl)) if s.surface_color is None else s.surface_color,
                )
                if s.use_cm:
                    a["color_map"] = next(self._cm)
                    a["attribute"] = attribute
                    a["color_range"] = [attribute.min(), attribute.max()]

                kw = merge({}, a, s.rendering_kw)
                surf = self.k3d.mesh(vertices, indices, **kw)

                self._fig += surf

            elif s.is_implicit and s.is_3Dsurface:
                _, _, _, r = s.get_data()
                xmin, xmax = s.start_x, s.end_x
                ymin, ymax = s.start_y, s.end_y
                zmin, zmax = s.start_z, s.end_z
                a = dict(
                    xmin=xmin, xmax=xmax,
                    ymin=ymin, ymax=ymax,
                    zmin=zmin, zmax=zmax,
                    compression_level=9,
                    level=0.0, flat_shading=True,
                    color=self._convert_to_int(next(self._cl))
                )
                kw = merge({}, a, s.rendering_kw)
                plt_iso = self.k3d.marching_cubes(r.astype(np.float32), **kw)

                self._fig += plt_iso

            elif s.is_3Dvector and s.is_streamlines:
                xx, yy, zz, uu, vv, ww = s.get_data()
                vertices, magn = compute_streamtubes(
                    xx, yy, zz, uu, vv, ww, s.rendering_kw)

                stream_kw = s.rendering_kw.copy()
                skw = dict(width=0.1, shader="mesh")
                if s.use_cm and ("color" not in stream_kw.keys()):
                    skw["color_map"] = next(self._cm)
                    skw["color_range"] = [np.nanmin(magn), np.nanmax(magn)]
                    skw["attribute"] = magn
                else:
                    col = stream_kw.pop("color", next(self._cl))
                    if not isinstance(col, int):
                        col = self._convert_to_int(col)
                    stream_kw["color"] = col

                kw = merge({}, skw, stream_kw)
                self._fig += self.k3d.line(
                    vertices.astype(np.float32), **kw)

            elif s.is_3Dvector:
                xx, yy, zz, uu, vv, ww = s.get_data()
                qkw = dict(scale=1)
                qkw = merge(qkw, s.rendering_kw)
                quiver_kw = s.rendering_kw
                if s.use_cm and ("color" not in quiver_kw.keys()):
                    colormap = next(self._cm)
                    # store useful info for interactive vector plots
                    self._handles[ii] = [qkw, colormap]
                else:
                    colormap = None
                    col = quiver_kw.get("color", next(self._cl))
                    if not isinstance(col, int):
                        col = self._convert_to_int(col)
                    solid_color = col * np.ones(xx.size)
                    self._handles[ii] = [qkw, colormap]

                origins, vectors, colors = self._build_k3d_vector_data(xx, yy, zz, uu, vv, ww, qkw, colormap)
                if colors is None:
                    colors = solid_color
                vec_colors = self._create_vector_colors(colors)

                pivot = quiver_kw.get("pivot", "mid")
                if pivot not in self._qp_offset.keys():
                    raise ValueError(
                        "`pivot` must be one of the following values: "
                        "{}".format(list(self._qp_offset.keys())))

                vec_kw = qkw.copy()
                kw_to_remove = ["scale", "color", "pivot"]
                for k in kw_to_remove:
                    if k in vec_kw.keys():
                        vec_kw.pop(k)
                vec_kw["origins"] = origins - vectors * self._qp_offset[pivot]
                vec_kw["vectors"] = vectors
                vec_kw["colors"] = vec_colors

                vec = self.k3d.vectors(**vec_kw)
                self._fig += vec

            elif s.is_complex and s.is_3Dsurface:
                x, y, mag, arg, colors, colorscale = s.get_data()

                x, y, z = [t.flatten() for t in [x, y, mag]]
                vertices = np.vstack([x, y, z]).T.astype(np.float32)
                indices = Triangulation(x, y).triangles.astype(np.uint32)
                self._high_aspect_ratio(x, y, z)

                a = dict(
                    name=s.get_label(self._use_latex, "%s") if self._show_label else None,
                    side="double",
                    flat_shading=False,
                    wireframe=False,
                    color=self._convert_to_int(next(self._cl)),
                    color_range=[-np.pi, np.pi],
                )
                if s.use_cm:
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
                surf = self.k3d.mesh(vertices, indices, **kw)

                self._fig += surf

            else:
                raise NotImplementedError(
                    "{} is not supported by {}\n".format(type(s), type(self).__name__)
                    + "K3D-Jupyter only supports 3D plots."
                )

        xl = self.xlabel if self.xlabel else "x"
        yl = self.ylabel if self.ylabel else "y"
        zl = self.zlabel if self.zlabel else "z"
        self._fig.axes = [xl, yl, zl]

        if self.title:
            self._fig += self.k3d.text2d(
                self.title, position=[0.025, 0.015], color=0, size=1, label_box=False
            )
        self._fig.auto_rendering = True

    def _build_k3d_vector_data(self, xx, yy, zz, uu, vv, ww, qkw, colormap):
        """Assemble the origins, vectors and colors (if possible) matrices.
        """
        np = import_module('numpy')

        xx, yy, zz, uu, vv, ww = [
            t.flatten().astype(np.float32) for t in [xx, yy, zz, uu, vv, ww]
        ]
        scale = qkw["scale"]
        magnitude = np.sqrt(uu ** 2 + vv ** 2 + ww ** 2)
        vectors = np.array((uu, vv, ww)).T * scale
        origins = np.array((xx, yy, zz)).T

        colors = None
        if colormap is not None:
            colors = self.k3d.helpers.map_colors(
                magnitude, colormap, [])

        return origins, vectors, colors

    def _create_vector_colors(self, colors):
        """Create a color matrix. Each vector requires one color for the tail
        and one for the head.
        """
        np = import_module('numpy')

        vec_colors = np.zeros(2 * len(colors))
        for i, c in enumerate(colors):
            vec_colors[2 * i] = c
            vec_colors[2 * i + 1] = c
        vec_colors = vec_colors.astype(np.uint32)
        return vec_colors

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
        np = import_module('numpy')

        # K3D title is an object in the figure
        n = len(self._fig.objects)
        if self.title is not None:
            n -= 1
        if len(self.series) != n:
            self._process_series(self.series)

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

                elif s.is_3Dsurface and (not s.is_domain_coloring) and (not s.is_implicit):
                    if s.is_parametric:
                        x, y, z, u, v = s.get_data()
                        x, y, z, u, v = [t.flatten().astype(np.float32) for t in [x, y, z, u, v]]
                        attribute = s.eval_color_func(x, y, z, u, v)
                    else:
                        x, y, z = s.get_data()
                        x, y, z = [t.flatten().astype(np.float32) for t in [x, y, z]]
                        attribute = s.eval_color_func(x, y, z)

                    vertices = np.vstack([x, y, z]).astype(np.float32)
                    self._fig.objects[i].vertices = vertices.T
                    if s.use_cm:
                        self._fig.objects[i].attribute = attribute
                        self._fig.objects[i].color_range = [attribute.min(), attribute.max()]

                elif s.is_vector and s.is_3D:
                    if s.is_streamlines:
                        raise NotImplementedError

                    xx, yy, zz, uu, vv, ww = self.series[i].get_data()
                    qkw, colormap = self._handles[i]
                    origins, vectors, colors = self._build_k3d_vector_data(xx, yy, zz, uu, vv, ww, qkw, colormap)
                    if colors is not None:
                        vec_colors = self._create_vector_colors(colors)
                        self.fig.objects[i].colors = vec_colors

                    pivot = s.rendering_kw.get("pivot", "mid")
                    self.fig.objects[i].origins = origins - vectors * self._qp_offset[pivot]
                    self.fig.objects[i].vectors = vectors

                elif s.is_complex and s.is_3Dsurface:
                    x, y, mag, _, colors, _ = s.get_data()
                    x, y, z = [t.flatten().astype(np.float32) for t in [x, y, mag]]
                    vertices = np.vstack([x, y, z]).astype(np.float32)
                    self._fig.objects[i].vertices = vertices.T
                    if s.use_cm:
                        colors = colors.reshape((-1, 3))
                        colors = [self._rgb_to_int(c) for c in colors]
                        self._fig.objects[i].colors = colors

        # self._fig.auto_rendering = True

    def show(self):
        """Visualize the plot on the screen."""
        np = import_module('numpy')

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
        2. '.html' files: this requires the ``msgpack`` [#fn2]_ python module
           to be installed.

           When exporting a fully portable html file, by default the required
           Javascript libraries will be loaded with a CDN. Set
           ``include_js=True`` to include all the javascript code in the html
           file: this will create a bigger file size, but can be
           run without internet connection.

        References
        ==========
        .. [#fn1] https://k3d-jupyter.org/k3d.html#k3d.plot.Plot.fetch_screenshot

        .. [#fn2] https://github.com/msgpack/msgpack-python

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
