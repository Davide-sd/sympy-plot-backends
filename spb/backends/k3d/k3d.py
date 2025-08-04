import param
import os
from spb.defaults import cfg
from spb.doc_utils.ipython import generate_doc
from spb.backends.base_backend import Plot
from spb.backends.k3d.renderers import (
    Line3DRenderer, Vector3DRenderer,
    ComplexRenderer, SurfaceRenderer, Implicit3DRenderer,
    GeometryRenderer, Arrow3DRenderer
)
from spb.series import (
    Parametric3DLineSeries, ComplexParametric3DLineSeries,
    List3DSeries, Vector3DSeries, SliceVector3DSeries,
    RiemannSphereSeries, Implicit3DSeries,
    ComplexDomainColoringSeries, ComplexSurfaceSeries,
    SurfaceOver2DRangeSeries, ParametricSurfaceSeries,
    PlaneSeries, GeometrySeries, Arrow3DSeries
)
from spb.utils import get_environment
from sympy.external import import_module
import warnings


class K3DBackend(Plot):
    """A backend for plotting SymPy's symbolic expressions using K3D-Jupyter.

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

    wireframe_color = 0x000000
    colormaps = []
    cyclic_colormaps = []
    skip_notebook_check = False

    # quivers-pivot offsets
    _qp_offset = {"tail": 0, "mid": 0.5, "middle": 0.5, "tip": 1}

    _allowed_keys = Plot._allowed_keys + ["show_label", "camera"]

    renderers_map = {
        Parametric3DLineSeries: Line3DRenderer,
        ComplexParametric3DLineSeries: Line3DRenderer,
        List3DSeries: Line3DRenderer,
        Vector3DSeries: Vector3DRenderer,
        SliceVector3DSeries: Vector3DRenderer,
        Implicit3DSeries: Implicit3DRenderer,
        ComplexDomainColoringSeries: ComplexRenderer,
        ComplexSurfaceSeries: ComplexRenderer,
        RiemannSphereSeries: ComplexRenderer,
        SurfaceOver2DRangeSeries: SurfaceRenderer,
        ParametricSurfaceSeries: SurfaceRenderer,
        PlaneSeries: SurfaceRenderer,
        GeometrySeries: GeometryRenderer,
        Arrow3DSeries: Arrow3DRenderer
    }

    _fig = param.Parameter(default=None, doc="""
        The figure in which symbolic expressions will be plotted into.""")
    show_label = param.Boolean(default=False, doc="""
        Show/hide labels of the expressions.""")

    def __init__(self, *args, **kwargs):
        self.np = import_module('numpy')
        self.k3d = k3d = import_module(
            'k3d',
            import_kwargs={'fromlist': ['helpers', 'objects']},
            warn_not_installed=True,
            min_module_version='2.9.7')
        cc = import_module(
            'colorcet',
            warn_not_installed=True,
            min_module_version='3.0.0')
        self.matplotlib = import_module(
            'matplotlib',
            import_kwargs={'fromlist': ['tri', 'cm']},
            min_module_version='1.1.0',
            warn_not_installed=True,
            catch=(RuntimeError,))
        cm = self.matplotlib.cm

        kwargs.setdefault("colorloop", cm.tab10.colors)
        kwargs.setdefault("colormaps", [
            k3d.basic_color_maps.CoolWarm,
            k3d.matplotlib_color_maps.Plasma,
            k3d.matplotlib_color_maps.Winter,
            k3d.matplotlib_color_maps.Viridis,
            k3d.paraview_color_maps.Haze,
            k3d.matplotlib_color_maps.Summer,
            k3d.paraview_color_maps.Blue_to_Yellow,
        ])
        kwargs.setdefault("cyclic_colormaps", [
            cc.colorwheel, k3d.paraview_color_maps.Erdc_iceFire_H])

        kwargs.setdefault("use_latex", cfg["k3d"]["use_latex"])
        kwargs.setdefault("grid", cfg["k3d"]["grid"])

        super().__init__(*args, **kwargs)
        if (not self.skip_notebook_check) and (get_environment() != 0):
            warnings.warn(
                "K3DBackend only works properly within Jupyter Notebook")

        self._set_labels("%s")
        self._set_title("%s")
        self._init_cyclers()

        self._bounds = []
        self._clipping = []
        self._title_handle = None

        kw = dict(
            grid_visible=self.grid,
            menu_visibility=True,
        )
        if cfg["k3d"]["bg_color"]:
            kw["background_color"] = int(cfg["k3d"]["bg_color"])
        if cfg["k3d"]["grid_color"]:
            kw["grid_color"] = int(cfg["k3d"]["grid_color"])
        if cfg["k3d"]["label_color"]:
            kw["label_color"] = int(cfg["k3d"]["label_color"])
        if cfg["k3d"]["camera_mode"]:
            kw["camera_mode"] = cfg["k3d"]["camera_mode"]

        self._use_existing_figure = "fig" in kwargs
        if not self._use_existing_figure:
            self._fig = k3d.plot(**kw)

        if (self.xscale == "log") or (self.yscale == "log"):
            warnings.warn(
                "K3D-Jupyter doesn't support log scales. We will "
                + "continue with linear scales."
            )

        self._create_renderers()

    @property
    def fig(self):
        """Returns the figure."""
        if (
            (len(self.renderers) > 0) and
            (
                (self.renderers[0] and len(self.renderers[0].handles) == 0)
                or (self.renderers[0] is None)
            )
        ):
            # if the backend was created without showing it
            self._process_renderers()
        return self._fig

    def draw(self):
        """ Loop over data renderers, generates numerical data and add it to
        the figure. Note that this method doesn't show the plot.
        """
        # this is necessary in order for the series to be added even if
        # show=False
        self._process_renderers()

    # process_series = draw

    @staticmethod
    def _do_sum_kwargs(p1, p2):
        kw = p1._copy_kwargs()
        sl1 = False if not hasattr(p1, "show_label") else p1.show_label
        sl2 = False if not hasattr(p2, "show_label") else p2.show_label
        kw["show_label"] = sl1 or sl2
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
        # NOTE: starting with NumPy 2.0.0, need to cast to int, otherwise
        # there is a risk that RGB is an array of dtype=uint8, which
        # could raise the error:
        # OverflowError: Python integer 65536 out of bounds for uint8
        R, G, B = [int(t) for t in RGB]
        return R * 256 ** 2 + G * 256 + B

    @classmethod
    def _convert_to_int(cls, color):
        """Convert the provided RGB tuple with values from 0 to 1 to an
        integer number.
        """
        color = [int(c * 255) for c in color]
        return cls._rgb_to_int(color)

    def _process_renderers(self):
        self._init_cyclers()
        self._fig.auto_rendering = False

        if not self._use_existing_figure:
            # clear data
            for o in self._fig.objects:
                self._fig.remove_class(o)

        for r, s in zip(self.renderers, self.series):
            self._check_supported_series(r, s)
            r.draw()

        self._set_axes_texts()
        self._new_camera_position()
        self._add_clipping_planes()
        self._fig.auto_rendering = True

    def _set_axes_texts(self):
        title, xlabel, ylabel, zlabel = self._get_title_and_labels()
        xl = xlabel if xlabel else "x"
        yl = ylabel if ylabel else "y"
        zl = zlabel if zlabel else "z"
        self._fig.axes = [xl, yl, zl]

        if title:
            if not self._title_handle:
                self._fig += self.k3d.text2d(
                    title, position=[0.025, 0.015], color=0, size=1,
                    label_box=False)
                self._title_handle = len(self._fig.objects) - 1
            else:
                self._fig.objects[self._title_handle].text = title

    def _build_k3d_vector_data(
        self, xx, yy, zz, uu, vv, ww, qkw, colormap, normalize, series
    ):
        """Assemble the origins, vectors and colors (if possible) matrices.
        """
        np = import_module('numpy')

        xx, yy, zz, uu, vv, ww = [
            t.flatten().astype(np.float32) for t in [xx, yy, zz, uu, vv, ww]
        ]
        scale = qkw["scale"]
        magnitude = np.sqrt(uu ** 2 + vv ** 2 + ww ** 2)
        if normalize:
            uu, vv, ww = [t / magnitude for t in [uu, vv, ww]]
        vectors = np.array((uu, vv, ww)).T * scale
        origins = np.array((xx, yy, zz)).T

        color_val = magnitude
        if series.color_func is not None:
            color_val = series.eval_color_func(xx, yy, zz, uu, vv, ww)
            # if color_func was a symbolic  expression, than color_val is
            # a matrix. Need to flatten.
            color_val = color_val.flatten()

        colors = None
        if colormap is not None:
            colors = self.k3d.helpers.map_colors(color_val, colormap, [])

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

    def update_interactive(self, params):
        """
        Implement the logic to update the data generated by
        interactive-widget plots.

        Parameters
        ==========

        params : dict
            Map parameter-symbols to numeric values.
        """
        self._bounds = []

        # Because InteractivePlot doesn't call the show method, the following
        # line of code will add the numerical data (if not already present).
        if len(self.renderers) > 0 and len(self.renderers[0].handles) == 0:
            self.draw()

        for r in self.renderers:
            if r.series.is_interactive:
                r.update(params)

        self._set_axes_texts()
        self._new_camera_position()
        self._add_clipping_planes()
        # self._fig.auto_rendering = True

    def _new_camera_position(self):
        if self.camera is None:
            np = import_module('numpy')
            if len(self._bounds) > 0:
                # when there are very high aspect ratio meshes, or when zlim has
                # been set, we compute a new camera position to improve user
                # experience
                self._fig.camera_auto_fit = False
                bounds = np.array(self._bounds)
                bounds = np.dstack([
                    np.min(bounds[:, 0::2], axis=0),
                    np.max(bounds[:, 1::2], axis=0)
                ]).flatten()
                self._fig.camera = self._fig.get_auto_camera(
                    1.5, 40, 60, bounds
                )
        else:
            # TODO: why on Earth doesn't it work?
            self._fig.camera = self.camera

    def show(self):
        """Visualize the plot on the screen."""
        if len(self.renderers) > 0 and len(self.renderers[0].handles) == 0:
            self._process_renderers()
        self._fig.display()

    def _add_clipping_planes(self):
        if len(self._fig.clipping_planes) == 0:
            clipping_planes = []
            if self.zlim:
                clipping_planes += [
                    [0, 0, 1, -self.zlim[0]],
                    [0, 0, -1, self.zlim[1]],
                ]
            if self.xlim:
                clipping_planes += [
                    [1, 0, 0, -self.xlim[0]],
                    [-1, 0, 0, self.xlim[1]],
                ]
            if self.ylim:
                clipping_planes += [
                    [0, 1, 0, -self.ylim[0]],
                    [0, -1, 0, self.ylim[1]],
                ]
            self._fig.clipping_planes = clipping_planes

    def save(self, path, **kwargs):
        """
        Export the plot to a static picture or to an interactive html file.

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

        ext = os.path.splitext(path)[1]
        if not ext:
            path += ".png"

        if ext in [".html", ".htm"]:
            with open(path, 'w') as f:
                include_js = kwargs.pop("include_js", False)
                self.fig.snapshot_include_js = include_js
                f.write(self.fig.get_snapshot(**kwargs))
        elif ext == ".png":
            if get_environment() != 0:
                raise ValueError(
                    "K3D-Jupyter requires the plot to be shown on the screen "
                    + "before saving a png file.")

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


generate_doc(K3DBackend)
