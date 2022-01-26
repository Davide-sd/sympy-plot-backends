
from sympy.utilities.iterables import is_sequence
from sympy.external import import_module
from itertools import cycle
from spb.series import BaseSeries
from spb.backends.utils import convert_colormap
matplotlib = import_module(
    'matplotlib',
    import_kwargs={'fromlist': ['pyplot', 'cm', 'collections', 'colors']},
    min_module_version='1.1.0',
    catch=(RuntimeError,))
cm = matplotlib.cm


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
       show it on the screen.

    It is important to realize that the backend-specific figure object is
    created at instantion of the backend. Thus, all parameters necessary to
    render the plot must be provided at instantion. After instantion, the
    backend is unable to modify the figure.

    The backend should check if it supports the data series that it's given.
    Please, explore the `MatplotlibBackend` source code to understand how a
    backend should be coded.

    Notes
    =====

    In order to be used by SymPy plotting functions, a backend must implement
    the following methods and attributes:

    * `show(self)`: used to loop over the data series, generate the numerical
      data, plot it and set the axis labels, title, ...
    * `save(self, path, **kwargs)`: used to save the current plot to the
      specified file path.
    * `self._fig`: an instance attribute to store the backend-specific plot
      object, which can be retrieved with the `Plot.fig` attribute. This
      object can then be used to further customize the resulting plot, using
      backend-specific commands.
    * `_update_interactive(self, params)`: this

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

        >>> from sympy import *
        >>> from spb import *
        >>> x = symbols("x")
        >>> p1 = plot(sin(x), cos(x), show=False)
        >>> p2 = plot(sin(x) * cos(x), log(x), show=False)
        >>> p3 = p1 + p2
        >>> p3.show()

    Combine multiple plots together to extend an existing plot:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy import *
        >>> from spb import *
        >>> x = symbols("x")
        >>> p1 = plot(sin(x), cos(x), show=False)
        >>> p2 = plot(sin(x) * cos(x), log(x), show=False)
        >>> p1.extend(p2)
        >>> p1.show()

    Use the index notation to access the data series. Let's generate the
    numerical data associated to the first series:

        >>> from sympy import *
        >>> from spb import *
        >>> x = symbols("x")
        >>> p1 = plot(sin(x), cos(x), show=False)
        >>> xx, yy = p1[0].get_data()


    See also
    ========

    MatplotlibBackend, PlotlyBackend, BokehBackend, K3DBackend
    """

    # set the name of the plotting library being used. This is required in
    # order to convert any colormap to the specified plotting library.
    _library = ""

    # list of colors to be used in line plots or solid color surfaces.
    # As a class attribute it must be a list/tuple in order for
    # `plot_piecewise` to work correctly. As an instance attribute it can
    # also be an instance of matplotlib's ListedColormap.
    colorloop = cm.tab10.colors

    # child backends should provide a list of color maps to render surfaces.
    colormaps = []

    # child backends should provide a list of cyclic color maps to render
    # complex series (the phase/argument ranges over [-pi, pi]).
    cyclic_colormaps = []

    def __new__(cls, *args, **kwargs):
        backend = cls._get_backend(kwargs)
        return super().__new__(backend)

    @classmethod
    def _get_backend(cls, kwargs):
        backend = kwargs.get("backend", "matplotlib")
        if not ((type(backend) == type) and issubclass(backend, cls)):
            raise TypeError("backend must be a subclass of Plot")
        return backend

    def __init__(self, *args, **kwargs):
        # Options for the graph as a whole.
        # The possible values for each option are described in the docstring
        # of Plot. They are based purely on convention, no checking is done.
        self.title = kwargs.get("title", None)
        self.xlabel = kwargs.get("xlabel", None)
        self.ylabel = kwargs.get("ylabel", None)
        self.zlabel = kwargs.get("zlabel", None)
        self.aspect = kwargs.get("aspect", "auto")
        self.axis_center = kwargs.get("axis_center", None)
        self.grid = kwargs.get("grid", True)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.zscale = kwargs.get("zscale", "linear")
        # NOTE: the following attributes is strictly related to
        # `plot_piecewise`. If it's True, it means the backend was
        # instantiated by that function.
        self.update_rendering_kw = kwargs.get("update_rendering_kw", False)
        # NOTE: it would be nice to have detect_poles=True by default.
        # However, the correct detection also depends on the number of points
        # and the value of `eps`. Getting the detection right is likely to
        # be a trial-by-error procedure. Hence, keep this parameter to False.
        self.detect_poles = kwargs.get("detect_poles", False)

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

        # The user can choose to use the standard color map loop, or
        # set/provide a solid color loop (for the surface color).
        self._use_cm = kwargs.get("use_cm", True)

        # auto-legend: if more than 1 data series has been provided and the
        # user has not set legend=False, then show the legend for better
        # clarity.
        self.legend = kwargs.get("legend", None)
        if not self.legend:
            self.legend = False
            if (len(self._series) > 1) or (
                any(s.is_parametric for s in self.series) and self._use_cm
            ):
                # don't show the legend if `plot_piecewise` created this
                # backend
                if not self.update_rendering_kw:
                    self.legend = True

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
                        "All numbers from {}={} must be real".format(t_name, t))
                if not is_finite(t):
                    raise ValueError(
                        "All numbers from {}={} must be finite".format(t_name, t))
                setattr(self, t_name, (float(t[0]), float(t[1])))

        self.xlim = None
        check_and_set("xlim", kwargs.get("xlim", None))
        self.ylim = None
        check_and_set("ylim", kwargs.get("ylim", None))
        self.zlim = None
        check_and_set("zlim", kwargs.get("zlim", None))
        self.size = None
        check_and_set("size", kwargs.get("size", None))

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
            update_rendering_kw=self.update_rendering_kw,
            detect_poles=self.detect_poles,
            use_cm=self._use_cm,
            legend=self.legend,
            xlim=self.xlim,
            ylim=self.ylim,
            zlim=self.zlim,
            size=self.size
        )

    def _init_cyclers(self):
        """Create infinite loop iterators over the provided color maps."""

        if not isinstance(self.colorloop, (list, tuple)):
            # assume it is a matplotlib's ListedColormap
            self.colorloop = self.colorloop.colors
        self._cl = cycle(self.colorloop)

        colormaps = [convert_colormap(cm, self._library) for cm in self.colormaps]
        self._cm = cycle(colormaps)
        cyclic_colormaps = [
            convert_colormap(cm, self._library) for cm in self.cyclic_colormaps
        ]
        self._cyccm = cycle(cyclic_colormaps)

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
            if (m != M) and (abs(abs(m) - np.pi) < eps) and (abs(abs(M) - np.pi) < eps):
                use_cyclic_cm = True
        return use_cyclic_cm

    @property
    def fig(self):
        """Returns the figure used to render/display the plots."""
        return self._fig

    @property
    def series(self):
        """Returns the series associated to the current plot."""
        return self._series

    def _update_interactive(self, params):
        """Implement the logic to update the data generated by
        InteractiveSeries.
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
        series_strs = [("[%d]: " % i) + str(s) for i, s in enumerate(self._series)]
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
        elif is_sequence(arg) and all([isinstance(a, BaseSeries) for a in arg]):
            self._series.extend(arg)
        else:
            raise TypeError("Expecting Plot or sequence of BaseSeries")
        # auto legend
        if len(self._series) > 1:
            self.legend = True
