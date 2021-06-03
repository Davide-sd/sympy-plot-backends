"""
-------------------------------------------------------------
|  keyword arg  | Matplolib | Bokeh | Plotly | Mayavi | K3D |
-------------------------------------------------------------
|     xlim      |     Y     |   Y   |    Y   |    N   |  N  |
|     ylim      |     Y     |   Y   |    Y   |    N   |  N  |
|     zlim      |     Y     |   N   |    Y   |    N   |  N  |
|    xscale     |     Y     |   Y   |    Y   |    N   |  N  |
|    yscale     |     Y     |   Y   |    Y   |    N   |  N  |
|    zscale     |     Y     |   N   |    Y   |    N   |  N  |
|     grid      |     Y     |   Y   |    Y   |    Y   |  Y  |
|    aspect     |     Y     |   N   |    N   |    N   |  N  |
|     size      |     Y     |   Y   |    Y   |    Y   |  Y  |
|     title     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    xlabel     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    ylabel     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    zlabel     |     Y     |   N   |    Y   |    Y   |  Y  |
|  line_color   |     Y     |   N   |    N   |    N   |  N  |
| surface_color |     Y     |   N   |    N   |    N   |  N  |
-------------------------------------------------------------
|       2D      |     Y     |   Y   |    Y   |    N   |  N  |
|       3D      |     Y     |   N   |    Y   |    Y   |  Y  |
| Latex Support |     Y     |   N   |    Y   |    N   |  Y  |
| Save Picture  |     Y     |   Y   |    Y   |    Y   |  Y  |
-------------------------------------------------------------
"""

import warnings
import numpy as np
from itertools import cycle
from matplotlib import cm
from sympy.utilities.iterables import is_sequence
from spb.series import BaseSeries

class Plot:
    """Base class for all backends. A backend represents the plotting library,
    which implements the necessary functionalities in order to use SymPy
    plotting functions.

    How the plotting module works:

    1. Whenever a plotting function is called, the provided expressions are
        processed and a list of instances of the `BaseSeries` class is created,
        containing the necessary information to plot the expressions (eg the
        expression, ranges, series name, ...). Eventually, these objects will
        generate the numerical data to be plotted.
    2. A Plot object is instantiated, which stores the list of series and the
        main attributes of the plot (eg axis labels, title, ...).
    3. The backend will then loops through each series object to generate and
        plot the numerical data and set the axis labels, title, ..., according
        to the provided values.

    The backend should check if it supports the data series that it's given.
    Please, explore the `MatplotlibBackend` source code to understand how a 
    backend should be coded.

    Methods
    =======

    In order to be used by SymPy plotting functions, a backend must implement
    the following methods:

    * `show(self)`: used to loop over the data series, generate the numerical
        data, plot it and set the axis labels, title, ...
    * save(self, path): used to save the current plot to the specified file
        path.
    * close(self): used to close the current plot backend (note: some plotting
        library doesn't support this functionality. In that case, just raise a
        warning).
    
    Also, the following attributes are required:

    * self._fig: (instance attribute) it stores the backend-specific plot
        object/s, which can be retrieved with the Plot.fig attribute. These
        objects can then be used to further customize the resulting plot, using
        backend-specific commands. For example, MatplotlibBackend stores a tuple
        (figure, axes).
    
    * support_plotgrid: (class attribute, boolean) if True it means the backend
        is able to generate `PlotGrid` objects. Please, look at 
        `MatplotlibBackend` as an example of a backend supporting `PlotGrid`.

    
    The arguments for the constructor Plot must be subclasses of BaseSeries.

    Any global option can be specified as a keyword argument. The global options
    for a figure are:

    - title : str
    - xlabel : str
    - ylabel : str
    - zlabel : str
    - legend : bool
    - xscale : {'linear', 'log'}
    - yscale : {'linear', 'log'}
    - grid : bool
    - axis_center : tuple of two floats or {'center', 'auto'}
    - xlim : tuple of two floats
    - ylim : tuple of two floats
    - zlim : tuple of two floats
    - aspect : tuple of two floats or {'auto'}
    - backend : a subclass of Plot
    - size : optional tuple of two floats, (width, height); default: None

    Note that a backend migh not use some option!

    Some data series support additional aesthetics or options. However, a 
    backend might not be able to use them. In particular:

    ListSeries, LineOver1DRangeSeries, Parametric2DLineSeries,
    Parametric3DLineSeries support the following:

    - line_color : string, or float, or function, optional
        Specifies the color for the plot, which depends on the backend being
        used.

        For example, if ``MatplotlibBackend`` is being used, then
        Matplotlib string colors are acceptable ("red", "r", "cyan", "c", ...).
        Alternatively, we can use a float number `0 < color < 1` wrapped in a
        string (for example, `line_color="0.5"`) to specify grayscale colors.
        Alternatively, We can specify a function returning a single
        float value: this will be used to apply a color-loop (for example,
        `line_color=lambda x: math.cos(x)`).

        Note that by setting line_color, it would be applied simultaneously
        to all the series.
    
    SurfaceOver2DRangeSeries, ParametricSurfaceSeries support the following:

    - syrface_color : string, or float, or function, optional
        Identical to line_color, but it applied to the surface.
    
    See also
    ========

    MatplotlibBackend, PlotlyBackend, BokehBackend, K3DBackend, MayaviBackend
    """

    # Set it to True in the subclasses if they are able to generate plot grids.
    # Also, clearly states in the docstring of the backend if it supports
    # plotgrids or not
    support_plotgrid = False

    colorloop = cm.tab10

    # child backends can provide a list of color maps to render surfaces.
    colormaps = []

    def __new__(cls, *args, **kwargs):
        backend = cls._get_backend(kwargs)
        return backend(*args, **kwargs)

    @classmethod
    def _get_backend(cls, kwargs):
        backend = kwargs.get("backend", "matplotlib")
        if not ((type(backend) == type) and issubclass(backend, cls)):
            raise TypeError(
                "backend must be a subclass of Plot")
        return backend

    def __init__(self, *args, **kwargs):
        # Options for the graph as a whole.
        # The possible values for each option are described in the docstring of
        # Plot. They are based purely on convention, no checking is done.
        self.title = kwargs.get("title", None)
        self.xlabel = kwargs.get("xlabel", None)
        self.ylabel = kwargs.get("ylabel", None)
        self.zlabel = kwargs.get("zlabel", None)
        self.aspect = kwargs.get("aspect", "auto")
        self.axis_center = kwargs.get("axis_center", "auto")
        self.grid = kwargs.get("grid", True)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.zscale = kwargs.get("zscale", "linear")
        
        # Contains the data objects to be plotted. The backend should be smart
        # enough to iterate over this list.
        self._series = []
        self._series.extend(args)

        # auto-legend: if more than 1 data series has been provided and the user
        # has not set legend=False, then show the legend for better clarity.
        self.legend = kwargs.get("legend", None)
        if self.legend is None:
            if len(self._series) > 1:
                self.legend = True

        # Objects used to render/display the plots, which depends on the 
        # plotting library.
        self._fig = None

        is_real = \
            lambda lim: all(getattr(i, 'is_real', True) for i in lim)
        is_finite = \
            lambda lim: all(getattr(i, 'is_finite', True) for i in lim)

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
    
        # PlotGrid-specific attributes.
        self.subplots = kwargs.get("subplots", None)
        if self.subplots is not None:
            self._series = []
            for p in self.subplots:
                self._series.append(p.series)
        self.nrows = kwargs.get("nrows", 1)
        self.ncols = kwargs.get("ncols", 1)
    
        # make custom keywords available inside self
        self._kwargs = kwargs

        # The user can choose to use the standard color map loop, or set/provide
        # a solid color loop (for the surface color).
        self._use_cm = kwargs.get("use_cm", True)
        # infinite loop iterator over the provided color maps
        self._cm = cycle(self.colormaps)
        # generate a list of RGB tuples (with values from 0 to 1) starting
        # from matplotlib's tab10 color map. This can be used instead of looping
        # through the colormaps
        if not isinstance(self.colorloop, (list, tuple)):
            # assume it is a matplotlib colormap
            self.colorloop = self.colorloop.colors
        self._cl = cycle(self.colorloop)
    
    def set_color_loop(self, cloop):
        """ Set the default color loop to use when use_cm=False. It must
        be a list of tuple (R, G, B) where 0 <= R,G,B <= 1.
        """
        if not isinstance(cloop, (tuple, list)):
            raise TypeError(
                    "cloop must be a list of RGB tuples with values " +
                    "from 0 to 1."
                )
        self._cl = cloop
    
    def _get_mode(self):
        """ Verify which environment is used to run the code.

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
            if shell == 'ZMQInteractiveShell':
                return 0   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return 1  # Terminal running IPython
            else:
                return 2  # Other type (?)
        except NameError:
            return 3      # Probably standard Python interpreter
    
    def _get_pixels(self, s, interval_list):
        """ Create the necessary data to visualize a Bokeh/Plotly Heatmap.
        Heatmap can be thought as an image composed of pixels, each one having
        a different color value.

        Parameters
        ==========
            s : ImplicitSeries
                It will be used to access to attributes of the series.
            interval_list : list
                The already computed interval_list
        
        Returns
        =======
            xarr, yarr : 1D np.ndarrays
                Discretization along the x and y axis
            pixels : 2D np.ndarray (n x n)
                The computed matrix to be used as the heatmap
        """

        # TODO: this approach is incorrect: with adaptive=True, s.n is not the
        # correct discretization parameter. Is it even possible to use Bokeh/
        # Plotly heatmaps with non-uniform discretization steps? 
        # Anyway, when depth>0, the pixels are much more refined than n=300.
        n = s.n
        dx = (s.end_x - s.start_x) / n
        dy = (s.end_y - s.start_y) / n
        xarr = np.linspace(s.start_x, s.end_x, n)
        yarr = np.linspace(s.start_y, s.end_y, n)

        pixels = np.zeros((n, n), dtype=np.dtype('b'))
        if len(interval_list):
            for intervals in interval_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                sx, ex = intervalx.start, intervalx.end
                sy, ey = intervaly.start, intervaly.end
                istart, iend = int((sx - s.start_x) / dx), int((ex - s.start_x) / dx)
                jstart, jend = int((sy - s.start_y) / dy), int((ey - s.start_y) / dy)
                pixels[jstart:jend, istart:iend] = 1
        return xarr, yarr, pixels

    @property
    def fig(self):
        """ Returns the objects used to render/display the plots, which depends
        on the backend (hence the plotting library). For example, 
        MatplotlibBackend will return a tuple: (figure, axes). Other plotting
        libraries may return a single object.
        """
        return self._fig
    
    @property
    def series(self):
        """ Returns the series associated to the current plot.
        """
        return self._series
    
    def _update_interactive(self, params):
        """ Implement the logic to update the data generated by
        InteractiveSeries.
        """
        raise NotImplementedError

    def show(self):
        """ Implement the functionalities to display the plot.
        """
        raise NotImplementedError

    def save(self, path, **kwargs):
        """ Implement the functionalities to save the plot.

        Parameters
        ==========

            path : str
                File path with extension.
            
            kwargs : dict
                Optional backend-specific parameters.
        """
        raise NotImplementedError

    def close(self):
        """ Implement the functionalities to close the plot.
        """
        raise NotImplementedError

    def __str__(self):
        series_strs = [('[%d]: ' % i) + str(s)
                       for i, s in enumerate(self._series)]
        return 'Plot object containing:\n' + '\n'.join(series_strs)

    def __getitem__(self, index):
        return self._series[index]

    def __setitem__(self, index, *args):
        if len(args) == 1 and isinstance(args[0], BaseSeries):
            self._series[index] = args

    def __delitem__(self, index):
        del self._series[index]

    def append(self, arg):
        """Adds an element from a plot's series to an existing plot.

        Examples
        ========

        Consider two ``Plot`` objects, ``p1`` and ``p2``. To add the
        second plot's first series object to the first, use the
        ``append`` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from sympy.plotting import plot
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
        else:
            raise TypeError('Must specify element of plot to append.')

    def extend(self, arg):
        """Adds all series from another plot.

        Examples
        ========

        Consider two ``Plot`` objects, ``p1`` and ``p2``. To add the
        second plot to the first, use the ``extend`` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from sympy.plotting import plot
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

        """
        if isinstance(arg, Plot):
            self._series.extend(arg._series)
        elif is_sequence(arg) and all([isinstance(a, BaseSeries) for a in arg]):
            self._series.extend(arg)
        else:
            raise TypeError('Expecting Plot or sequence of BaseSeries')

class PlotGrid:
    """This class helps to plot subplots from already created sympy plots
    in a single figure. The success of the operation depends on the backend
    being used: for example, MatplotlibBackend is able to generate plotgrids,
    but other backends may not.

    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> from sympy import symbols
        >>> from sympy.plotting import plot, plot3d, PlotGrid
        >>> x, y = symbols('x, y')
        >>> p1 = plot(x, x**2, x**3, (x, -5, 5))
        >>> p2 = plot((x**2, (x, -6, 6)), (x, (x, -5, 5)))
        >>> p3 = plot(x**3, (x, -5, 5))
        >>> p4 = plot3d(x*y, (x, -5, 5), (y, -5, 5))

    Plotting vertically in a single line:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(2, 1 , p1, p2)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x for x over (-5.0, 5.0)
        [1]: cartesian line: x**2 for x over (-5.0, 5.0)
        [2]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)

    Plotting horizontally in a single line:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(1, 3 , p2, p3, p4)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[2]:Plot object containing:
        [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    Plotting in a grid form:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(2, 2, p1, p2 ,p3, p4)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x for x over (-5.0, 5.0)
        [1]: cartesian line: x**2 for x over (-5.0, 5.0)
        [2]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)
        Plot[2]:Plot object containing:
        [0]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[3]:Plot object containing:
        [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    """
    def __init__(self, nrows, ncolumns, *args, **kwargs):
        """
        Parameters
        ==========

        nrows :
            The number of rows that should be in the grid of the
            required subplot.
        ncolumns :
            The number of columns that should be in the grid
            of the required subplot.

        nrows and ncolumns together define the required grid.

        Arguments
        =========

        A list of predefined plot objects entered in a row-wise sequence
        i.e. plot objects which are to be in the top row of the required
        grid are written first, then the second row objects and so on

        Keyword arguments
        =================

        show : Boolean
            The default value is set to ``True``. Set show to ``False`` and
            the function will not display the subplot. The returned instance
            of the ``PlotGrid`` class can then be used to save or display the
            plot by calling the ``save()`` and ``show()`` methods
            respectively.
        size : (float, float), optional
            A tuple in the form (width, height) in inches to specify the size of
            the overall figure. The default value is set to ``None``, meaning
            the size will be set by the default backend.
        """
        self.nrows = nrows
        self.ncols = ncolumns
        self.args = args

        # Since each backend could implement different attributes, if we were
        # to instantiate a PlotGrid starting from plots generated with different
        # backends, chances are an AttributeError would be raised. Hence, all
        # plots must be generated with the same backend, which should also
        # support PlotGrid
        backends = set()
        for a in args:
            backends.add(type(a))
        if len(backends) > 1:
            raise TypeError(
                "PlotGrid requires the provided plots to be generated by " +
                "the same backend.")
        backend = list(backends)[0]
        if not backend.support_plotgrid:
            raise ValueError(
                "{} does not support PlotGrid.".format(backend))
        
        self._backend = backend(nrows=self.nrows, ncols=self.ncols,
                subplots=args, **kwargs)
        
        if kwargs.get("show", True):
            self.show()

    @property
    def fig(self):
        return self._backend.fig

    def show(self):
        """ Display the current plot.
        """
        self._backend.show()

    def save(self, path, **kwargs):
        """ Save the current plot.

        Parameters
        ==========

            path : str
                File path with extension.
            
            kwargs : dict
                Optional backend-specific parameters.
        """
        self._backend.save(path)
    
    def close(self):
        """ Close the current plot.
        """
        self._backend.close()

    def __str__(self):
        plot_strs = [('Plot[%d]:' % i) + str(plot)
                      for i, plot in enumerate(self.args)]

        return 'PlotGrid object containing:\n' + '\n'.join(plot_strs)
