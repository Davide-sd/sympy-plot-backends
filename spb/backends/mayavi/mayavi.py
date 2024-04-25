from spb.defaults import cfg
from spb.backends.base_backend import Plot
from spb.backends.mayavi.renderers import *
from spb.series import *
from spb.utils import get_environment
from sympy.external import import_module
import warnings


class MayaviBackend(Plot):
    """A backend for plotting SymPy's symbolic expressions using Mayavi.

    Parameters
    ==========

    aspect : str, optional
        Set the aspect ratio of the plot. Default to ``"equal"``.
        Possible values:

        - ``"equal"``: sets equal spacing on the axis of a 3D plot.
        - ``"auto"`` adjust the spacing/scaling of objects.

    bg_color : tuple, optional
        A tuple of RGB values from 0 to 1 specifying the background color.
        Default to (0.22, 0.24, 0.29).

    fg_color : tuple, optional
        A tuple of RGB values from 0 to 1 specifying the foreground color,
        that is the color of all text annotation labels (axes, orientation
        axes, scalar bar labels). It should be sufficiently far from
        `bgcolor` to see the annotation texts.
        Default to (1, 1, 1), which represent white color.

    notebook_kw : dict, optional
        A dictionary of options to be passed to ``mlab.init_notebook``.

    rendering_kw : dict, optional
        A dictionary of keywords/values which is passed to Matplotlib's plot
        functions to customize the appearance of lines, surfaces, images,
        contours, quivers, streamlines...
        To learn more about customization:

        * Refer to [#fn1]_ and [#fn2]_ to customize line plots.
        * Refer to [#fn3]_ to customize surface plots. Refers to [#fn3]_ for
          a list of available colormaps.
        * Refer to [#fn4]_ to customize 3D implicit surface plots.
        * Refer to [#fn5]_ and [#fn6]_ to customize quivers and streamlines.

    show_colorbar : bool, optional
        Hide or show the colorbar when a colormap is used. Default to True.

    window : bool, optional
        Launch the plot on a new window. Default to False.
        If the environment is Jupyter Notebook and ``window=False``, then
        the plot will be inserted in the output cell.

    Notes
    =====

    1. Mayavi is difficult to install, hence that step is left to the end user.
       There are no guarantees that this backend works on all systems, as it
       has only been tested on an Ubuntu Machine with Python 3.10 with pip.
    2. Mayavi cannot use colormaps from other plotting libraries. Hence, only
       use the colormap listed in [#fn7]_ .

     References
    ==========
    .. [#fn1] https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#points3d
    .. [#fn2] https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#plot3d
    .. [#fn3] https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#mesh
    .. [#fn4] https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#contour3d
    .. [#fn5] https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#quiver3d
    .. [#fn6] https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#flow
    .. [#fn7] https://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html


    """

    _library = "mayavi"
    _allowed_keys = Plot._allowed_keys + ["window", "notebook_kw"]
    wireframe_color = (0, 0, 0)

    renderers_map = {
        Parametric3DLineSeries: Line3DRenderer,
        ComplexParametric3DLineSeries: Line3DRenderer,
        List3DSeries: Line3DRenderer,
        Vector3DSeries: Vector3DRenderer,
        SliceVector3DSeries: Vector3DRenderer,
        Implicit3DSeries: Implicit3DRenderer,
        SurfaceOver2DRangeSeries: SurfaceRenderer,
        ParametricSurfaceSeries: SurfaceRenderer,
        PlaneSeries: SurfaceRenderer,
    }

    def __init__(self, *args, **kwargs):
        matplotlib = import_module(
            'matplotlib',
            import_kwargs={'fromlist': ['tri', 'cm']},
            min_module_version='1.1.0',
            catch=(RuntimeError,))
        mayavi = import_module(
            'mayavi',
            import_kwargs={'fromlist': ['mlab']},
            min_module_version='4.8.0',
            catch=(RuntimeError,))
        self.mlab = mayavi.mlab
        IPython = import_module(
            'IPython',
            import_kwargs={'fromlist': ['core']},
            min_module_version='8.4.0',
            catch=(RuntimeError,))
        self._display = None
        if IPython:
            # this if statement is required to pass tests with Python 3.7
            self._display = IPython.core.display.display

        self.colorloop = matplotlib.cm.tab10.colors
        self.colormaps = [
            'viridis', 'coolwarm', 'plasma', 'winter', 'summer', 'autumn'
        ]
        self._init_cyclers()
        super().__init__(*args, **kwargs)
        self._use_latex = kwargs.get("use_latex", cfg["mayavi"]["use_latex"])
        self._set_labels()
        window = kwargs.pop("window", False)
        notebook_kw = kwargs.pop("notebook_kw", dict())
        self.grid = kwargs.get("grid", cfg["mayavi"]["grid"])

        if (get_environment() == 0) and (not window):
            self.mlab.init_notebook(**notebook_kw)

        # NOTE/TODO: to adjust the aspect ratio, Mayavi uses the ``extent``
        # keyword argument on its plotting functions. Turns out that the grids
        # sorrounding the objects (giving indications of their dimensions)
        # reads the ``extent`` values. So, if aspect="auto" the grids will
        # show wrong values. Let's use a equal aspect ratio by default.
        self.aspect = kwargs.get("aspect", "equal")
        if self.aspect == "auto":
            warnings.warn("You have set ``aspect='equal'``. Be aware that "
                "if ``grid=True``, then the numbers shown on the grids "
                "are likely to be wrong.")
        self.show_colorbar = kwargs.get("show_colorbar", True)

        size = cfg["mayavi"]["size"]
        if self.size:
            size = self.size
        self._fig = self.mlab.figure(
            size=size,
            bgcolor=cfg["mayavi"]["bg_color"],
            fgcolor=cfg["mayavi"]["fg_color"],
        )
        self._create_renderers()

    def _process_renderers(self):
        mlab = self.mlab
        mlab.clf(self._fig)
        self._init_cyclers()

        for r, s in zip(self.renderers, self.series):
            self._check_supported_series(r, s)
            r.draw()
            if self.grid and s.show_in_legend:
                mlab.axes(
                    xlabel="", ylabel="", zlabel="",
                    x_axis_visibility=True,
                    y_axis_visibility=True,
                    z_axis_visibility=True,
                )
                mlab.outline()

        xl = self.xlabel if self.xlabel else "x"
        yl = self.ylabel if self.ylabel else "y"
        zl = self.zlabel if self.zlabel else "z"
        mlab.orientation_axes(xlabel=xl, ylabel=yl, zlabel=zl)
        if self.title:
            mlab.title(self.title, figure=self._fig, size=0.5)

    def _add_figure_to_kwargs(self, kw):
        if "figure" not in kw.keys():
            # NOTE: strange enough, if figure is a keyword of "a", then
            # the following error would raise:
            # TypeError: Figure not attached to a mayavi engine.
            kw["figure"] = self._fig

    def _get_extent(self, x, y, z):
        """Implement ``aspect='auto'``. """
        np = import_module('numpy')
        if (self.aspect == "auto"):
            return [0, 1, 0, 1, 0, 1]
        return [
            np.nanmin(x), np.nanmax(x),
            np.nanmin(y), np.nanmax(y),
            np.nanmin(z), np.nanmax(z)
        ]

    def _add_colorbar(self, s, obj, colorbar_kw, solid_color):
        merge = self.merge
        if self.show_colorbar and s.use_cm and (solid_color is None):
            colorbar_kw_default = dict(
                title=s.get_label(self._use_latex), orientation="vertical",
                nb_labels=None, nb_colors=None, label_fmt=None
            )
            cbkw = merge({}, colorbar_kw_default, colorbar_kw)
            if "object" not in cbkw:
                # NOTE: strange enough, if object is a keyword of
                # "colorbar_kw_default", then an error is raised
                cbkw["object"] = obj
            self.mlab.colorbar(**cbkw)

    def draw(self):
        self._process_renderers()
    
    # process_series = draw

    def show(self):
        self.draw()
        if self._display:
            self._display(self._fig)

    def save(self, path, **kwargs):
        """Save the current plot.

        Parameters
        ==========

        path : str
            File path with extension.

        kwargs : dict
            Optional backend-specific parameters. Refer to [#fn8]_ to find
            more keyword arguments to control the output file.

        References
        ==========

        .. [#fn8] https://docs.enthought.com/mayavi/mayavi/auto/mlab_figure.html#savefig

        """
        self.mlab.savefig(path, figure=self._fig, **kwargs)

    def close(self):
        self.mlab.close(self._fig)


MAB = MayaviBackend

