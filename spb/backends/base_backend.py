from itertools import cycle
from spb.series import BaseSeries, LineOver1DRangeSeries
from spb.backends.utils import convert_colormap
from sympy import Symbol
from sympy.utilities.iterables import is_sequence
from sympy.external import import_module


class Plot:
    """Base class for all backends. A backend represents the plotting library,
    which implements the necessary functionalities in order to use SymPy
    plotting functions.

    How the plotting module works:

    1. The user creates the symbolic expressions and calls one of the plotting
       functions.
    2. The plotting functions generate a list of instances of the `BaseSeries`
       class, containing the necessary information to plot the expressions
       (eg the expression, ranges, series name, ...). Eventually, these
       objects will generate the numerical data to be plotted.
    3. The plotting functions instantiate the `Plot` class, which stores the
       list of series and the main attributes of the plot (eg axis labels,
       title, etc.). Among the keyword arguments, there must be `backend`,
       a subclass of `Plot` which specify the backend to be used.
    4. The backend will render the numerical data to a plot and (eventually)
       show it on the screen. The figure is populated with numerical data once
       the `show()` method or the `fig` attribute are called.

    The backend should check if it supports the data series that it's given.
    Please, explore the `MatplotlibBackend` source code to understand how a
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
      object, which can be retrieved with the `Plot.fig` attribute. This
      object can then be used to further customize the resulting plot, using
      backend-specific commands.
    * ``update_interactive(self, params)``: this method receives a dictionary
      mapping parameters to their values from the ``iplot`` function, which
      are going to be used to update the objects of the figure.

    Parameters
    ==========

    title : str, optional
        Set the title of the plot. Default to an empty string.

    xlabel, ylabel, zlabel : str, optional
        Set the labels of the plot. Default to an empty string.

    legend : bool, optional
        Show or hide the legend. By default, the backend will automatically
        set it to True if multiple data series are shown.

    xscale, yscale, zscale : str, optional
        Discretization strategy for the provided domain along the specified
        direction. Can be either `'linear'` or `'log'`. Default to
        `'linear'`. If the backend supports it, the specified direction will
        use the user-provided scale. By default, all backends uses linear
        scales for both axis. None of the backends support logarithmic scale
        for 3D plots.

    grid : bool, optional
        Show/Hide the grid. The default value depends on the backend.

    xlim, ylim, zlim : (float, float), optional
        Focus the plot to the specified range. The tuple must be in the form
        `(min_val, max_val)`.

    aspect : (float, float) or str, optional
        Set the aspect ratio of the plot. It only works for 2D plots.
        The values depends on the backend. Read the interested backend's
        documentation to find out the possible values.

    backend : Plot
        The subclass to be used to generate the plot.

    size : (float, float) or None, optional
        Set the size of the plot, `(width, height)`. Default to None.

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
       [0]: cartesian line: sin(x)/3 for x over (-10.0, 10.0)
       [1]: cartesian line: 2*sin(x)/3 for x over (-10.0, 10.0)
       [2]: cartesian line: sin(x) for x over (-10.0, 10.0)

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

    # set the name of the plotting library being used. This is required in
    # order to convert any colormap to the specified plotting library.
    _library = ""

    colorloop = []
    """List of colors to be used in line plots or solid color surfaces."""

    colormaps = []
    """List of color maps to render surfaces."""

    cyclic_colormaps = []
    """List of cyclic color maps to render complex series (the phase/argument
    ranges over [-pi, pi]).
    """

    _allowed_keys = [
        "aspect", "axis", "axis_center", "backend",
        "detect_poles", "grid", "legend", "show", "size", "title", "use_latex",
        "xlabel", "ylabel", "zlabel", "xlim", "ylim", "zlim", "show_axis",
        "xscale", "yscale", "zscale", "process_piecewise", "polar_axis",
        "imodule"
    ]
    """contains a list of public keyword arguments supported by the series.
    It will be used to validate the user-provided keyword arguments.
    """

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
        if not self._use_latex:
            wrapper = "%s"

        if callable(self.xlabel):
            self.xlabel = wrapper % self.xlabel(self._use_latex)
        if callable(self.ylabel):
            self.ylabel = wrapper % self.ylabel(self._use_latex)
        if callable(self.zlabel):
            self.zlabel = wrapper % self.zlabel(self._use_latex)

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
        if not self._use_latex:
            wrapper = "%s"

        if callable(self.title):
            self.title = self.title(wrapper, self._use_latex)

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

        # Options for the graph as a whole.
        # The possible values for each option are described in the docstring
        # of Plot. They are based purely on convention, no checking is done.
        self.title = kwargs.get("title", None)
        self.xlabel = kwargs.get("xlabel", None)
        self.ylabel = kwargs.get("ylabel", None)
        self.zlabel = kwargs.get("zlabel", None)
        self.aspect = kwargs.get("aspect", kwargs.get("aspect_ratio", "auto"))
        self.axis_center = kwargs.get("axis_center", None)
        self.camera = kwargs.get("camera", None)
        self.grid = kwargs.get("grid", True)
        self.xscale = kwargs.get("xscale", None)
        self.yscale = kwargs.get("yscale", None)
        self.zscale = kwargs.get("zscale", None)
        self.polar_axis = kwargs.get("polar_axis", None)
        # NOTE: it would be nice to have detect_poles=True by default.
        # However, the correct detection also depends on the number of points
        # and the value of `eps`. Getting the detection right is likely to
        # be a trial-by-error procedure. Hence, keep this parameter to False.
        self.detect_poles = kwargs.get("detect_poles", False)
        # NOTE: matplotlib is not designed to be interactive, therefore it
        # needs a way to detect where its figure is going to be displayed.
        # For regular plots, plt.figure can be used. For interactive-parametric
        # plots with holoviz panel, matplotlib.figure.Figure must be used.
        self.is_iplot = kwargs.get("is_iplot", False)
        # backend might need to create different types of figure depending on
        # the interactive module being used
        self.imodule = kwargs.get("imodule", None)

        # Contains the data objects to be plotted. The backend should be smart
        # enough to iterate over this list.
        self._series = []
        self._series.extend(args)
        if "process_piecewise" in kwargs.keys():
            # if the backend was called by plot_piecewise, each piecewise
            # function must use the same color. Here we preprocess each
            # series to add the correct color
            series = []
            for idx, _series in kwargs["process_piecewise"].items():
                color = next(self._cl)
                for s in _series:
                    self._set_piecewise_color(s, color)
                series.extend(_series)
            self._series = series

        # Automatic legend: if more than 1 data series has been provided
        # and the user has not set legend=False, then show the legend for
        # better clarity.
        self.legend = kwargs.get("legend", None)
        if self.legend is None:
            series_to_show = [
                s for s in self._series if s.show_in_legend and (not s.use_cm)
            ]
            if len(series_to_show) > 1:
                # don't show the legend if `plot_piecewise` created this
                # backend
                if not ("process_piecewise" in kwargs.keys()):
                    self.legend = True

        # allow to invert x-axis if the range is given as (symbol, max, min)
        # instead of (symbol, min, max).
        # just check the first series.
        self._invert_x_axis = False
        if (
            (len(self._series) > 0) and
            isinstance(self._series[0], LineOver1DRangeSeries)
        ):
            # elements of parametric ranges can't be compared because they
            # are likely going to be symbolic expressions
            if not self._series[0]._interactive_ranges:
                r = self._series[0].ranges[0]
                # Ranges can be real or complex. Cast them to complex and
                # look at the real part.
                if complex(r[1]).real > complex(r[2]).real:
                    self._invert_x_axis = True

        # Objects used to render/display the plots, which depends on the
        # plotting library.
        self._fig = None

        is_real = lambda lim: all(getattr(i, "is_real", True) for i in lim)
        is_finite = lambda lim: all(getattr(i, "is_finite", True) for i in lim)

        # reduce code repetition
        def check_and_set(t_name, t):
            if t:
                if not is_real(t):
                    raise ValueError(
                        f"All numbers from {t_name}={t} must be real")
                if not is_finite(t):
                    raise ValueError(
                        f"All numbers from {t_name}={t} must be finite")
                setattr(self, t_name, (float(t[0]), float(t[1])))

        self.xlim = None
        check_and_set("xlim", kwargs.get("xlim", None))
        self.ylim = None
        check_and_set("ylim", kwargs.get("ylim", None))
        self.zlim = None
        check_and_set("zlim", kwargs.get("zlim", None))
        self.size = None
        check_and_set("size", kwargs.get("size", None))
        self.axis = kwargs.get("show_axis", kwargs.get("axis", True))

    def _copy_kwargs(self):
        """Copy the values of the plot attributes into a dictionary which will
        be later used to create a new `Plot` object having the same attributes.
        """
        return dict(
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            zlabel=self.zlabel,
            aspect=self.aspect,
            axis_center=self.axis_center,
            grid=self.grid,
            xscale=self.xscale,
            yscale=self.yscale,
            zscale=self.zscale,
            detect_poles=self.detect_poles,
            legend=self.legend,
            xlim=self.xlim,
            ylim=self.ylim,
            zlim=self.zlim,
            size=self.size,
            is_iplot=self.is_iplot,
            use_latex=self._use_latex,
            camera=self.camera,
            polar_axis=self.polar_axis,
            axis=self.axis,
        )

    def _init_cyclers(self):
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

    def _get_mode(self):
        """Verify which environment is used to run the code.

        Returns
        =======
            mode : int
                0 - the code is running on Jupyter Notebook or qtconsole
                1 - terminal running IPython
                2 - other type (?)
                3 - probably standard Python interpreter

        # TODO: detect if we are running in Jupyter Lab.
        """
        # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return 0  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return 1  # Terminal running IPython
            else:
                return 2  # Other type (?)
        except NameError:
            return 3  # Probably standard Python interpreter

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

    @property
    def fig(self):
        """Returns the figure used to render/display the plots."""
        return self._fig

    @property
    def renderers(self):
        """Returns the renderers associated to each series."""
        return self._renderers

    @property
    def series(self):
        """Returns the series associated to the current plot."""
        return self._series

    def update_interactive(self, params):
        """Implement the logic to update the data generated by
        interactive-widget plots.

        Parameters
        ==========

        params : dict
            Map parameter-symbols to numeric values.
        """
        raise NotImplementedError

    def show(self):
        """Implement the functionalities to display the plot."""
        raise NotImplementedError

    def save(self, path, **kwargs):
        """Implement the functionalities to save the plot.

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
        return type(self)(*series, **kwargs)

    def append(self, arg):
        """Adds an element from a plot's series to an existing plot.

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
           [0]: cartesian line: x**2 for x over (-10.0, 10.0)
           [1]: cartesian line: x for x over (-10.0, 10.0)
           >>> p1.show()

        See Also
        ========

        extend

        """
        if isinstance(arg, BaseSeries):
            self._series.append(arg)
            # auto legend
            if len(self._series) > 1:
                self.legend = True
        else:
            raise TypeError("Must specify element of plot to append.")
        # recreate renderers
        self._create_renderers()

    def extend(self, arg):
        """Adds all series from another plot.

        Parameters
        ==========

        arg : Plot or sequence of BaseSeries

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
           [0]: cartesian line: x**2 for x over (-10.0, 10.0)
           [1]: cartesian line: x for x over (-10.0, 10.0)
           [2]: cartesian line: -x for x over (-10.0, 10.0)
           >>> p1.show()

        See Also
        ========

        append

        """
        if isinstance(arg, Plot):
            self._series.extend(arg._series)
        elif (
            is_sequence(arg) and
            all([isinstance(a, BaseSeries) for a in arg])
        ):
            self._series.extend(arg)
        else:
            raise TypeError("Expecting Plot or sequence of BaseSeries")
        # auto legend
        if len(self._series) > 1:
            self.legend = True
        # recreate renderers
        self._create_renderers()
