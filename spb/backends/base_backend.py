from itertools import cycle, islice
from numbers import Number
import param
from spb.defaults import cfg
from spb.doc_utils.ipython import modify_parameterized_doc
from spb.series import (
    BaseSeries, LineOver1DRangeSeries, ComplexSurfaceBaseSeries
)
from spb.backends.utils import convert_colormap, tick_formatter_multiples_of
from sympy import Symbol, Expr
from sympy.utilities.iterables import is_sequence
from sympy.external import import_module


class _TupleOfRealNumbers(param.Tuple):
    """``xlim, ylim, zlim`` receives 2-elements tuple from the user.
    They must contain numbers. Symbolic numbers (like pi) will be converted to
    floats.

    Note that first `__set__` is called, then `_validate` is called.
    """

    def _validate(self, val):
        super()._validate(val)
        np = import_module("numpy")

        # NOTE: at this stage, each element of val should have been parsed.
        # All symbolic numbers (like S.Half, pi, etc.), should now be float.
        # numerical complex numbers are still the same.
        # symbolic complex numbers are still symbolic expression.
        # symbolic expressions maybe present.

        def check(t):
            is_number = isinstance(t, Number)
            is_real = is_number and (not isinstance(t, (Expr, complex)))
            is_nan, is_finite = False, False
            if is_real:
                is_nan = np.isnan(t)
                is_finite = np.isfinite(t)
            return is_real and (not is_nan) and is_finite

        if val:
            if not all(check(n) for n in val):
                raise ValueError(
                    f"All numbers from {self.name}={val} must be real"
                    " and finite."
                )

    def __set__(self, obj, val):
        parsed_val = val
        if val is not None:
            parsed_val = []
            for v in val:
                try:
                    parsed_val.append(float(v))
                except TypeError:
                    parsed_val.append(v)
            parsed_val = tuple(parsed_val)
        super().__set__(obj, parsed_val)


class _StringOrTupleOrCallable(param.Parameter):
    def __init__(self, default=None, **params):
        super().__init__(default=default, **params)

    def _validate(self, val):
        if not (isinstance(val, (str, tuple, list)) or callable(val)):
            raise ValueError(
                f"Parameter '{self.name}' must be a string or a callable,"
                f" not {type(val).__name__}."
            )
        if isinstance(val, (tuple, list)):
            if len(val) == 0:
                raise ValueError(
                    f"Parameter '{self.name}' value is a tuple of 0-length."
                    " This is not ok. At least one element must be present."
                )
            if not isinstance(val[0], str):
                raise ValueError(
                    f"Parameter '{self.name}' is a tuple. Its first element"
                    " must be a string."
                )
        return val


class PlotAttributes(param.Parameterized):
    colorloop = param.ClassSelector(default=[], class_=(list, tuple), doc="""
        List of colors to be used in line plots or solid color surfaces.""")
    colormaps = param.ClassSelector(default=[], class_=(list, tuple), doc="""
        List of color maps to render surfaces.""")
    cyclic_colormaps = param.ClassSelector(default=[], class_=(list, tuple), doc="""
        List of cyclic color maps to render complex series (the phase/argument
        ranges over [-pi, pi]).""")

    # NOTE: `fig` and `ax` are just placeholder parameters in order for them
    # to appear in the `graphics` docstring. In reality, all backends
    # implement the `_fig` parameter (defined below) where the actual figure
    # is stored (and MatplotlibBackend also implements the `_ax` parameter),
    # as well as read-only `fig` (and `ax`) properties. These properties are
    # mandatory because when a user requests the figure or axis (by executing
    # plot.fig or plot.ax), the backend must first check that `_fig` (or `_ax`)
    # are not None. If they are None, they need to be created first.
    fig = param.Parameter(
        default=None, doc="Get or set the figure where to plot into.")
    ax = param.Parameter(doc="""
        An existing Matplotlib's Axes over which the symbolic
        expressions will be plotted.""")

    theme = param.String(default="", doc="""
        Theme to be used to style the figure. Depending on the backend being
        used, several themes may be available.

        * For Plotly: https://plotly.com/python/templates/
        * For Bokeh: https://docs.bokeh.org/en/latest/docs/reference/themes.html
        * For Matplotlib: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        """)
    title = _StringOrTupleOrCallable(default="", doc="""
        Title of the plot. It can be:

        * a string.
        * a callable receiving a single argument, `use_latex`, which must
          return a string.
        * a tuple of the form `(format_str, symbol 1, symbol 2, etc.)`, which
          creates an output string when parameters `symbol 1, symbol 2, etc.`
          receive numerical values from the widgets. This operation mode only
          works when creating interactive data series (ie, specifying the
          ``params`` dictionary).""")
    xlabel = _StringOrTupleOrCallable(default="", doc="""
        Label of the x-axis. It can be:

        * a string.
        * a callable receiving a single argument, `use_latex`, which must
          return a string.
        * a tuple of the form `(format_str, symbol 1, symbol 2, etc.)`, which
          creates an output string when parameters `symbol 1, symbol 2, etc.`
          receive numerical values from the widgets. This operation mode only
          works when creating interactive data series (ie, specifying the
          ``params`` dictionary).""")
    ylabel = _StringOrTupleOrCallable(default="", doc="""
        Label of the y-axis. It can be:

        * a string.
        * a callable receiving a single argument, `use_latex`, which must
          return a string.
        * a tuple of the form `(format_str, symbol 1, symbol 2, etc.)`, which
          creates an output string when parameters `symbol 1, symbol 2, etc.`
          receive numerical values from the widgets. This operation mode only
          works when creating interactive data series (ie, specifying the
          ``params`` dictionary).""")
    zlabel = _StringOrTupleOrCallable(default="", doc="""
        Label of the z-axis. It can be:

        * a string.
        * a callable receiving a single argument, `use_latex`, which must
          return a string.
        * a tuple of the form `(format_str, symbol 1, symbol 2, etc.)`, which
          creates an output string when parameters `symbol 1, symbol 2, etc.`
          receive numerical values from the widgets. This operation mode only
          works when creating interactive data series (ie, specifying the
          ``params`` dictionary).""")
    size = _TupleOfRealNumbers(default=None, length=2, doc="""
        Set the size of the plot, `(width, height)`.
        For Matplotlib, the size is measured in inches. For Bokeh, Plotly
        and K3D-Jupyter, the size is in pixel.""")
    use_latex = param.Boolean(
        default=True, doc="""
        Turn on/off the rendering of latex labels. If the backend doesn't
        support latex, it will render the string representations instead.""")
    aspect = param.ClassSelector(class_=(str, tuple, list, dict), doc="""
        Set the aspect ratio.

        Possible values for Matplotlib (only works for a 2D plot):

        * ``"auto"``: Matplotlib will fit the plot in the vibile area.
        * ``"equal"``: sets equal spacing.
        * tuple containing 2 float numbers, from which the aspect ratio is
          computed. This only works for 2D plots.

        Possible values for Plotly:

        - ``"equal"``: sets equal spacing on the axis of a 2D plot.
        - For 3D plots:

          * ``"cube"``: fix the ratio to be a cube
          * ``"data"``: draw axes in proportion of their ranges
          * ``"auto"``: automatically produce something that is well
            proportioned using 'data' as the default.
          * manually set the aspect ratio by providing a dictionary.
            For example: ``dict(x=1, y=1, z=2)`` forces the z-axis to appear
            twice as big as the other two.

        Possible values for Bokeh:

        * ``"equal"``: sets equal spacing.
        """)
    axis_center = param.ClassSelector(
        class_=(str, tuple), allow_None=True, doc="""
        Set the location of the intersection between the horizontal and
        vertical axis in a 2D plot. It only works with Matplotlib and it can
        receive the following values:

        * ``None``: traditional layout, with the horizontal axis fixed on the
          bottom and the vertical axis fixed on the left. This is the default
          value.
        * a tuple ``(x, y)`` specifying the exact intersection point.
        * ``'center'``: center of the current plot area.
        * ``'auto'``: the intersection point is automatically computed.
        """)
    camera = param.Parameter(doc="""
        Set the camera position for 3D plots.

        For Matplotlib, it can be a dictionary of keyword arguments that will
        be passed to the ``Axes3D.view_init`` method. Refer to the following
        link for more information:
        https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D.view_init

        For Plotly, it can be a dictionary of keyword arguments that will
        be passed to the layout's ``scene_camera``. Refer to the following
        link for more information:
        https://plotly.com/python/3d-camera-controls/

        For K3D-Jupyter, it is list of 9 numbers, namely:

        * ``x_cam, y_cam, z_cam``:  the position of the camera in the scene
        * ``x_tar, y_tar, z_tar``: the position of the target of the camera
        * ``x_up, y_up, z_up``: components of the up vector
        """)
    axis = param.Boolean(True, doc="Show the axis in the figure.")
    polar_axis = param.Boolean(False, doc="""
        If True, the backend will attempt to use polar axis, otherwise it
        uses cartesian axis. This is only supported for 2D plots.""")
    grid = param.ClassSelector(default=True, class_=(bool, dict), doc="""
        Toggle the visibility of major grid lines. A dictionary of keyword
        arguments can be passed to customized the appearance of the grid
        lines:

        * Matplotlib: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.grid.html
        * Plotly: https://plotly.com/python/axes/#styling-grid-lines
        * Bokeh: https://docs.bokeh.org/en/latest/docs/reference/models/grids.html#module-bokeh.models.grids
        """)
    minor_grid = param.ClassSelector(
        default=False, class_=(bool, dict), doc="""
        Toggle the visibility of minor grid lines. A dictionary of keyword
        arguments can be passed to customized the appearance of the grid lines:

        * Matplotlib: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.grid.html
        * Plotly: https://plotly.com/python/axes/#styling-grid-lines
        * Bokeh: https://docs.bokeh.org/en/latest/docs/reference/models/grids.html#module-bokeh.models.grids
        """)
    hooks = param.List(default=[], doc="""
        List of functions expecting one argument, the current plot object,
        which allows users to further customize the appearance of the plot
        before it is shown on the screen. The hooks are executed:

        1. after the figure has been initialized and populated with
           numerical data.
        2. after the existing renderers update the visualization because
           the user interacted with some widget.

        Note: let ``p`` be the plot object. Then, the user can access the
        figure with ``p.fig``. In case of
        :py:class:`spb.backends.matplotlib.MatplotlibBackend`, the user can
        also retrieve the axes in which data was added with ``p.ax``.
        """)
    legend = param.Boolean(default=None, doc="""
        Toggle the visibility of the legend. If None, the backend will
        automatically determine if it is appropriate to show it.""")
    update_event = param.Boolean(False, allow_None=True, doc="""
        If True and the backend supports such functionality, events like
        drag and zoom will trigger a recompute of the data series within the
        new axis limits.""")
    x_ticks_formatter = param.ClassSelector(
        default=None, class_=tick_formatter_multiples_of, doc="""
        An object of type ``tick_formatter_multiples_of`` which will be used
        to place tick values at each multiple of a specified quantity, along
        the x-axis.""")
    y_ticks_formatter = param.ClassSelector(
        default=None, class_=tick_formatter_multiples_of, doc="""
        An object of type ``tick_formatter_multiples_of`` which will be used
        to place tick values at each multiple of a specified quantity, along
        the y-axis.""")
    xlim = _TupleOfRealNumbers(default=None, length=2, doc="""
        Limit the figure's x-axis to the specified range. The tuple must be in
        the form `(min_val, max_val)`.""")
    ylim = _TupleOfRealNumbers(default=None, length=2, doc="""
        Limit the figure's y-axis to the specified range. The tuple must be in
        the form `(min_val, max_val)`.""")
    zlim = _TupleOfRealNumbers(default=None, length=2, doc="""
        Limit the figure's z-axis to the specified range. The tuple must be in
        the form `(min_val, max_val)`.""")
    xscale = param.Selector(
        default="linear", objects=["linear", "log", None], doc="""
        If the backend supports it, the x-direction will use the specified
        scale. Note that none of the backends support logarithmic scale
        for 3D plots.""")
    yscale = param.Selector(
        default="linear", objects=["linear", "log", None], doc="""
        If the backend supports it, the y-direction will use the specified
        scale. Note that none of the backends support logarithmic scale
        for 3D plots.""")
    zscale = param.Selector(
        default="linear", objects=["linear", "log", None], doc="""
        If the backend supports it, the z-direction will use the specified
        scale. Note that none of the backends support logarithmic scale
        for 3D plots.""")


@modify_parameterized_doc()
class Plot(PlotAttributes):
    """
    Base class for all backends. A backend represents the plotting library,
    which implements the necessary functionalities in order to use SymPy
    plotting functions.

    How the plotting module works:

    1. The user creates the symbolic expressions and calls one of the plotting
       functions.
    2. The plotting functions generate a list of instances of ``BaseSeries``,
       containing the necessary information to generate the appropriate
       numerical data and create the proper visualization
       (eg the expression, ranges, series name, ...).
    3. The plotting functions instantiate the ``Plot`` class, which stores the
       list of series and the main attributes of the plot (eg axis labels,
       title, etc.). Among the keyword arguments, there must be ``backend=``,
       where a subclass of ``Plot`` can be specified in order to use a
       particular plotting library.
    4. Each data series will be associated to a corresponding renderer, which
       receives the numerical data and add it to the figure. A renderer is also
       responsible to keep objects up-to-date when interactive-widgets plots
       are used. The figure is populated with numerical data when the
       ``show()`` method or the ``fig`` attribute are called.

    The backend should check if it supports the data series that it's given.
    Please, explore ``MatplotlibBackend`` source code to understand how a
    backend should be coded.

    Also note that setting attributes to plot objects or to data series after
    they have been instantiated is strongly unrecommended, as it is not
    guaranteed that the figure will be updated.

    Notes
    =====

    In order to be used by SymPy plotting functions, a backend must implement
    the following methods and attributes:

    * ``show(self)``: used to loop over the data series, generate the
      numerical data, plot it and set the axis labels, title, ...
    * ``save(self, path, **kwargs)``: used to save the current plot to the
      specified file path.
    * ``self._fig``: an instance attribute to store the backend-specific plot
      object, which can be retrieved with the ``Plot.fig`` attribute. This
      object can then be used to further customize the resulting plot, using
      backend-specific commands.
    * ``update_interactive(self, params)``: this method receives a dictionary
      mapping parameters to their values from the ``iplot`` function, which
      are going to be used to update the objects of the figure.

    Examples
    ========

    Combine multiple plots together to create a new plot:

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, sin, cos, log, S
       >>> from spb import plot, plot3d
       >>> x, y = symbols("x, y")
       >>> p1 = plot(sin(x), cos(x), show=False)
       >>> p2 = plot(sin(x) * cos(x), log(x), show=False)
       >>> p3 = p1 + p2
       >>> p3.show()

    Use the index notation to access the data series. Let's generate the
    numerical data associated to the first series:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> p1 = plot(sin(x), cos(x), show=False)
       >>> xx, yy = p1[0].get_data()

    Create a new backend with a custom colorloop:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from spb.backends.matplotlib import MB
       >>> class MBchild(MB):
       ...     colorloop = ["r", "g", "b"]
       >>> plot(sin(x) / 3, sin(x) * S(2) / 3, sin(x), backend=MBchild)
       Plot object containing:
       [0]: cartesian line: sin(x)/3 for x over (-10, 10)
       [1]: cartesian line: 2*sin(x)/3 for x over (-10, 10)
       [2]: cartesian line: sin(x) for x over (-10, 10)

    Create a new backend with custom color maps for 3D plots. Note that
    it's possible to use Plotly/Colorcet/Matplotlib colormaps interchangeably.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from spb.backends.matplotlib import MB
       >>> import colorcet as cc
       >>> class MBchild(MB):
       ...     colormaps = ["plotly3", cc.bmy]
       >>> plot3d(
       ...     (cos(x**2 + y**2), (x, -2, 0), (y, -2, 2)),
       ...     (cos(x**2 + y**2), (x, 0, 2), (y, -2, 2)),
       ...     backend=MBchild, n1=25, n2=50, use_cm=True)
       Plot object containing:
       [0]: cartesian surface: cos(x**2 + y**2) for x over (-2.0, 0.0) and y over (-2.0, 2.0)
       [1]: cartesian surface: cos(x**2 + y**2) for x over (0.0, 2.0) and y over (-2.0, 2.0)


    See also
    ========

    MatplotlibBackend, PlotlyBackend, BokehBackend, K3DBackend
    """

    _library = param.String(default="", doc="""
        Set the name of the plotting library being used. This is required in
        order to convert any colormap to the format supported by the
        specified plotting library.""")
    _fig = param.Parameter(default=None, doc="""
        The figure in which symbolic expressions will be plotted into.""")
    _imodule = param.Parameter(default=False, constant=True, doc="""
        Store the interactive module's name that will use this plot instance.
        NOTE: matplotlib is not designed to be interactive, therefore it
        needs a way to detect where its figure is going to be displayed.
        For regular plots, plt.figure can be used. For interactive-parametric
        plots with holoviz panel, matplotlib.figure.Figure must be used.
        Similarly, Plotly must use go.FigureWidgets with holoviz panel,
        instead of the regular go.Figure.""")
    invert_x_axis = param.Boolean(doc="""
        Allow to invert x-axis if the range is given as (symbol, max, min)
        instead of (symbol, min, max).""")

    def _execute_hooks(self):
        for h in self.hooks:
            h(self)

    def _set_labels(self, wrapper="$%s$"):
        """Set the correct axis labels depending on wheter the backend support
        Latex rendering.

        Parameters
        ==========
        use_latex : boolean
            Wheter the backend is customized to show latex labels.
        wrapper : str
            Wrapper string for the latex labels. Default to '$%s$'.
        """
        if not self.use_latex:
            wrapper = "%s"

        if callable(self.xlabel):
            self.xlabel = wrapper % self.xlabel(self.use_latex)
        if callable(self.ylabel):
            self.ylabel = wrapper % self.ylabel(self.use_latex)
        if callable(self.zlabel):
            self.zlabel = wrapper % self.zlabel(self.use_latex)

    def _set_title(self, wrapper="$%s$"):
        """Set the correct title depending on wheter the backend support
        Latex rendering.

        Parameters
        ==========
        use_latex : boolean
            Wheter the backend is customized to show latex labels.
        wrapper : str
            Wrapper string for the latex labels. Default to '$%s$'.
        """
        if not self.use_latex:
            wrapper = "%s"

        if callable(self.title):
            self.title = self.title(wrapper, self.use_latex)

    def _create_parametric_text(self, t, params):
        """Given a tuple of the form `(str, symbol1, symbol2, ...)`
        where `str` is a formatted string, read the values from the parameters
        `symbol1, symbol2, ...`, pass them to the formatted string to create
        an updated text.
        """
        if isinstance(t, (tuple, list)):
            t_symbols = set().union(*[e.free_symbols for e in t[1:]])
            remaining_symbols = set(t_symbols).difference(params.keys())
            # TODO: is it worth creating a `ptext` class so that this check
            # is only run once at the beginning, and not at every update?
            if len(remaining_symbols) > 0:
                raise ValueError(
                    "This parametric text contains symbols that are "
                    "not part of the `params` dictionary:\n"
                    f"{t}.\nWhat are these symbols? {remaining_symbols}"
                )
            values = [
                params[s] if isinstance(s, Symbol) else s.subs(params)
                for s in t[1:]
            ]
            return t[0].format(*values)
        return t

    def _get_title_and_labels(self):
        """Returns the appropriate text to be shown on the title and
        axis labels.
        """
        title = self.title
        xlabel = self.xlabel
        ylabel = self.ylabel
        zlabel = self.zlabel

        params = None
        if len(self.series) > 0:
            # assuming all data series received the same parameters
            params = self.series[0].params
        if params:
            title = self._create_parametric_text(title, params)
            xlabel = self._create_parametric_text(xlabel, params)
            ylabel = self._create_parametric_text(ylabel, params)
            zlabel = self._create_parametric_text(zlabel, params)
        return title, xlabel, ylabel, zlabel

    def __init__(self, *args, **kwargs):
        # the merge function is used by all backends
        self._mergedeep = import_module('mergedeep')
        self.merge = self._mergedeep.merge
        kwargs.setdefault("imodule", cfg["interactive"]["module"])
        process_piecewise = kwargs.pop("process_piecewise", None)

        if "is_polar" in kwargs:
            kwargs.setdefault("polar_axis", kwargs.pop("is_polar"))
        if "fig" in kwargs:
            kwargs["_fig"] = kwargs.pop("fig")

        # remove keyword arguments that are not parameters of this backend
        kwargs = {k: v for k, v in kwargs.items() if k in list(self.param)}

        super().__init__(**kwargs)
        self._init_cyclers()

        # Contains the data objects to be plotted. The backend should be smart
        # enough to iterate over this list.
        grid_series = [s for s in args if s.is_grid]
        non_grid_series = [s for s in args if not s.is_grid]
        # grid series must be the last to be rendered: they need to know
        # the extension of the area to cover with grids, which can be obtained
        # after plotting all other series.
        self._series = non_grid_series + grid_series
        if process_piecewise is not None:
            # if the backend was called by plot_piecewise, each piecewise
            # function must use the same color. Here we preprocess each
            # series to add the correct color
            series = []
            for idx, _series in process_piecewise.items():
                color = next(self._cl)
                for s in _series:
                    self._set_piecewise_color(s, color)
                series.extend(_series)
            self._series = series

        # Automatic legend: if more than 1 data series has been provided
        # and the user has not set legend=False, then show the legend for
        # better clarity.
        if self.legend is None:
            series_to_show = [
                s for s in self._series if (
                    s.show_in_legend and
                    (not s.use_cm) and
                    (not s.is_grid)
                )
            ]
            if len(series_to_show) > 1:
                # don't show the legend if `plot_piecewise` created this
                # backend
                if process_piecewise is None:
                    self.legend = True

        # allow to invert x-axis if the range is given as (symbol, max, min)
        # instead of (symbol, min, max).
        # just check the first series.
        if (
            (len(self._series) > 0) and
            isinstance(self._series[0], LineOver1DRangeSeries)
        ):
            # elements of parametric ranges can't be compared because they
            # are likely going to be symbolic expressions
            if not self._series[0]._parametric_ranges:
                r = self._series[0].ranges[0]
                # Ranges can be real or complex. Cast them to complex and
                # look at the real part.
                if complex(r[1]).real > complex(r[2]).real:
                    self.invert_x_axis = True

    def _copy_kwargs(self):
        """Copy the values of the plot attributes into a dictionary which will
        be later used to create a new `Plot` object having the same attributes.
        """
        params = {}
        # copy all parameters into the dictionary
        for k in list(self.param):
            if k not in ["name", "_fig", "fig", "ax"]:
                params[k] = getattr(self, k)
        return params

    def _init_cyclers(self, start_index_cl=None, start_index_cm=None):
        """Create infinite loop iterators over the provided color maps."""
        tb = type(self)
        colorloop = self.colorloop if not tb.colorloop else tb.colorloop
        colormaps = self.colormaps if not tb.colormaps else tb.colormaps
        cyclic_colormaps = self.cyclic_colormaps
        if tb.cyclic_colormaps:
            cyclic_colormaps = tb.cyclic_colormaps

        if not isinstance(colorloop, (list, tuple)):
            # assume it is a matplotlib's ListedColormap
            self.colorloop = colorloop.colors
        self._cl = cycle(colorloop)

        colormaps = [convert_colormap(cm, self._library) for cm in colormaps]
        self._cm = cycle(colormaps)
        cyclic_colormaps = [
            convert_colormap(cm, self._library) for cm in cyclic_colormaps
        ]
        self._cyccm = cycle(cyclic_colormaps)

        if start_index_cl is not None:
            self._cl = islice(self._cl, start_index_cl, None)
        if start_index_cm is not None:
            self._cm = islice(self._cm, start_index_cm, None)

    def _create_renderers(self):
        """Connect data series to appropriate renderers."""
        self._renderers = []
        for s in self.series:
            t = type(s)
            if t in self.renderers_map.keys():
                self._renderers.append(self.renderers_map[t](self, s))
            else:
                # NOTE: technically, I could just raise an error at this point.
                # However, there are occasions where it might be useful to
                # create data series starting from plotting functions, without
                # showing the plot. Hence, I raise the error later, if needed.
                self._renderers.append(None)

    def _check_supported_series(self, renderer, series):
        if renderer is None:
            raise NotImplementedError(
                f"{type(series).__name__} is not associated to any renderer "
                f"compatible with {type(self).__name__}. Follow these "
                "steps to make it works:\n"
                "1. Code an appropriate rendeder class.\n"
                f"2. Execute {type(self).__name__}.renderers_map.update"
                "({%s})\n" % f"{type(series).__name__}: YourRendererClass"
                + "3. Execute again the plot statement."
            )

    def _use_cyclic_cm(self, param, is_complex):
        """When using complex_plot and `absarg=True`, it might happens that the
        argument is not fully covering the range [-pi, pi]. In such occurences,
        the use of a cyclic colormap would create a misleading plot.
        """
        np = import_module('numpy')

        eps = 0.1
        use_cyclic_cm = False
        if is_complex:
            m, M = np.amin(param), np.amax(param)
            if (
                (m != M) and
                (abs(abs(m) - np.pi) < eps) and
                (abs(abs(M) - np.pi) < eps)
            ):
                use_cyclic_cm = True
        return use_cyclic_cm

    def _set_piecewise_color(self, s, color):
        """Set the color to the given series of a piecewise function."""
        raise NotImplementedError

    def _update_series_ranges(self, *limits):
        """Update the ranges of data series in order to implement pan/zoom
        update events.

        Parameters
        ==========
        limits : iterable
            Each element is a tuple (min, max).

        Returns
        =======
        all_params : dict
            A dictionary containing all the parameters used by the series
        """
        all_params = {}
        # TODO: can ComplexSurfaceBaseSeries be modified such that it has
        # two ranges instead of one? It would allow to simplify this code...
        css = [s for s in self._series
            if isinstance(s, ComplexSurfaceBaseSeries)]
        ncss = [s for s in self._series
            if not isinstance(s, ComplexSurfaceBaseSeries)]

        for s in ncss:
            # skip the ones that don't use `ranges`, like
            # List2D/List3D/Arrow2D/Arrow3D
            # as well as the parametric series
            if (len(s.ranges) > 0) and (not s.is_parametric):
                new_ranges = []
                for r, l in zip(s.ranges, limits):
                    if any(len(t.free_symbols) > 0 for t in r[1:]):
                        # design choice: interactive ranges should not
                        # be modified
                        new_ranges.append(r)
                    else:
                        new_ranges.append((r[0], *l))
                s.ranges = new_ranges
                s._parametric_ranges = True
                with param.edit_constant(s):
                    s._is_interactive = True
            all_params = self.merge({}, all_params, s.params)

        if len(css) > 0:
            xlim, ylim = limits[:2]
            lim = (xlim[0] + 1j * ylim[0], xlim[1] + 1j * ylim[1])
            for s in css:
                if all(len(t.free_symbols) == 0 for t in s.ranges[0][1:]):
                    # design choice: interactive ranges should not
                    # be modified
                    s.ranges = [(s.ranges[0][0], *lim)]
                    s._parametric_ranges = True
                    with param.edit_constant(s):
                        s._is_interactive = True
                all_params = self.merge({}, all_params, s.params)

        return all_params

    @property
    def renderers(self):
        """Returns the renderers associated to each series."""
        return self._renderers

    @property
    def series(self):
        """Returns the series associated to the current plot."""
        return self._series

    def update_interactive(self, params):
        """
        Implement the logic to update the data generated by
        interactive-widget plots.

        Parameters
        ==========

        params : dict
            Map parameter-symbols to numeric values.
        """
        raise NotImplementedError

    def show(self):
        """
        Implement the functionalities to display the plot."""
        raise NotImplementedError

    def save(self, path, **kwargs):
        """
        Implement the functionalities to save the plot.

        Parameters
        ==========

        path : str
            File path with extension.

        kwargs : dict
            Optional backend-specific parameters.
        """
        raise NotImplementedError

    def __str__(self):
        series_strs = [
            ("[%d]: " % i) + str(s) for i, s in enumerate(self._series)
        ]
        return "Plot object containing:\n" + "\n".join(series_strs)

    def __getitem__(self, index):
        return self._series[index]

    def __setitem__(self, index, *args):
        if len(args) == 1 and isinstance(args[0], BaseSeries):
            self._series[index] = args

    def __delitem__(self, index):
        del self._series[index]

    def __add__(self, other):
        return self._do_sum(other)

    def __radd__(self, other):
        if other == 0:
            return self
        return other._do_sum(self)

    def __repr__(self):
        if cfg["use_repr"] is False:
            return object.__repr__(self)
        return super().__repr__()

    def _do_sum(self, other):
        """Differently from Plot.extend, this method creates a new plot
        object, which uses the series of both plots and merges the _kwargs
        dictionary of `self` with the one of `other`.
        """
        if not isinstance(other, Plot):
            raise TypeError(
                "Both sides of the `+` operator must be instances of the Plot "
                + "class.\n Received: {} + {}".format(type(self), type(other))
            )
        series = []
        series.extend(self.series)
        series.extend(other.series)
        kwargs = self._do_sum_kwargs(self, other)
        kwargs.pop("fig", None) # in order to avoid duplicate series
        return type(self)(*series, **kwargs)

    def append(self, series):
        """
        Adds another series to this plot.

        Parameters
        ==========

        arg : BaseSeries
            An instance of `BaseSeries` which will be used to generate the
            numerical data.

        Examples
        ========

        Consider two `Plot` objects, `p1` and `p2`. To add the
        second plot's first series object to the first, use the
        `append` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from spb import plot
           >>> x = symbols('x')
           >>> p1 = plot(x*x, show=False)
           >>> p2 = plot(x, show=False)
           >>> p1.append(p2[0])
           >>> p1
           Plot object containing:
           [0]: cartesian line: x**2 for x over (-10, 10)
           [1]: cartesian line: x for x over (-10, 10)
           >>> p1.show()

        See Also
        ========

        extend

        """
        if isinstance(series, BaseSeries):
            self._series.append(series)
            # auto legend
            if len([not s.is_grid for s in self._series]) > 1:
                self.legend = True
        else:
            raise TypeError("Must specify element of plot to append.")
        # recreate renderers
        self._create_renderers()

    def extend(self, plot_or_series):
        """
        Adds all series from another plot to this plot.

        Parameters
        ==========

        plot_or_series :
            Plot or sequence of BaseSeries

        Examples
        ========

        Consider two `Plot` objects, `p1` and `p2`. To add the
        second plot to the first, use the `extend` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from spb import plot
           >>> x = symbols('x')
           >>> p1 = plot(x**2, show=False)
           >>> p2 = plot(x, -x, show=False)
           >>> p1.extend(p2)
           >>> p1
           Plot object containing:
           [0]: cartesian line: x**2 for x over (-10, 10)
           [1]: cartesian line: x for x over (-10, 10)
           [2]: cartesian line: -x for x over (-10, 10)
           >>> p1.show()

        See Also
        ========

        append

        """
        if isinstance(plot_or_series, Plot):
            self._series.extend(plot_or_series._series)
        elif (
            is_sequence(plot_or_series) and
            all([isinstance(a, BaseSeries) for a in plot_or_series])
        ):
            self._series.extend(plot_or_series)
        else:
            raise TypeError("Expecting Plot or sequence of BaseSeries")
        # auto legend
        if len([not s.is_grid for s in self._series]) > 1:
            self.legend = True
        # recreate renderers
        self._create_renderers()
