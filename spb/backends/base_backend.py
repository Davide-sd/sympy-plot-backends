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
from mergedeep import merge
from spb.backends.utils import convert_colormap
from cplot import get_srgb1


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

    # set the name of the plotting library being used. This is required in order
    # to convert any colormap to the specified plotting library.
    _library = ""

    # Set it to True in the subclasses if they are able to generate plot grids.
    # Also, clearly states in the docstring of the backend if it supports
    # plotgrids or not
    support_plotgrid = False

    # list of colors to be used in line plots or solid color surfaces.
    # It can also be an instance of matplotlib's ListedColormap.
    colorloop = cm.tab10

    # child backends should provide a list of color maps to render surfaces.
    colormaps = []

    # child backends should provide a list of cyclic color maps to render 
    # complex series (the phase/argument ranges over [-pi, pi]).
    cyclic_colormaps = []

    # pi number is used in all backends to set the ranges for the colorbars in
    # complex plot. It is defined here for commodity, rather than importing 
    # math or numpy on each backend.
    pi = np.pi

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
        self.axis_center = kwargs.get("axis_center", None)
        self.grid = kwargs.get("grid", True)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.zscale = kwargs.get("zscale", "linear")
        # TODO: it would be nice to have detect_poles=True by default. 
        # At this point of development, if that were True there could be times
        # where the algorithm kicks in even when there are no poles. Needs to
        # dig deeper into it...
        self.detect_poles = kwargs.get("detect_poles", False)

        # Contains the data objects to be plotted. The backend should be smart
        # enough to iterate over this list.
        self._series = []
        self._series.extend(args)

        # make custom keywords available inside self
        self._kwargs = kwargs

        # The user can choose to use the standard color map loop, or set/provide
        # a solid color loop (for the surface color).
        self._use_cm = kwargs.get("use_cm", True)

        # auto-legend: if more than 1 data series has been provided and the user
        # has not set legend=False, then show the legend for better clarity.
        self.legend = kwargs.get("legend", None)
        if self.legend is None:
            self.legend = False
            if ((len(self._series) > 1) or 
                (any(s.is_parametric for s in self.series) and self._use_cm)):
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
    
    def _init_cyclers(self):
        """ Create infinite loop iterators over the provided color maps. """

        if not isinstance(self.colorloop, (list, tuple)):
            # assume it is a matplotlib's ListedColormap
            self.colorloop = self.colorloop.colors
        self._cl = cycle(self.colorloop)

        colormaps = [convert_colormap(cm, self._library) for cm
                in self.colormaps]
        self._cm = cycle(colormaps)
        cyclic_colormaps = [convert_colormap(cm, self._library) for cm
                in self.cyclic_colormaps]
        self._cyccm = cycle(cyclic_colormaps)
    
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

    def _get_pixels(self, s, intervals_list):
        """ Create the necessary data to visualize a Bokeh/Plotly Heatmap.
        Heatmap can be thought as an image composed of pixels, each one having
        a different color value. We can think of the adaptively computed data as
        a "non-uniform grid". This function convert the data to a uniform grid
        so that heatmaps can be used.

        The following implementation is just an approximation of the correct
        results, because the intervals might fall in between pixels: those will
        not be rendered.

        Parameters
        ==========
            s : ImplicitSeries
                It will be used to access to attributes of the series.
            intervals_list : list
                The already computed interval_list
        
        Returns
        =======
            xarr, yarr : 1D np.ndarrays
                Discretization along the x and y axis
            pixels : 2D np.ndarray (n x n)
                The computed matrix to be used as the heatmap
        """
        
        warnings.warn(
            "Currently, only MatplotlibBackend is capable of correctly " +
            "visualizing the data generated by the adaptive algorithm. " +
            "You are using {}, hence the visualized ".format(type(self)) +
            "result is just an approximation."
        )

        # look for the minimum delta_x and delta_y
        dx = min([abs(interval[0].start - interval[0].end) for interval in intervals_list])
        dy = min([abs(interval[1].start - interval[1].end) for interval in intervals_list])
        # compute the number of pixels in the horizontal and vertical directions
        n1 = int(np.ceil(abs(s.start_x - s.end_x) / dx))
        n2 = int(np.ceil(abs(s.start_y - s.end_y) / dy))
        # the number of pixels have been ceiled: need to recompute the
        # incremental steps
        dx = (s.end_x - s.start_x) / n1
        dy = (s.end_y - s.start_y) / n2

        xarr = np.linspace(s.start_x, s.end_x, n1)
        yarr = np.linspace(s.start_x, s.end_x, n2)

        pixels = np.zeros((n2, n1), dtype=np.dtype('b'))
        if len(intervals_list):
            for intervals in intervals_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                sx, ex = intervalx.start, intervalx.end
                sy, ey = intervaly.start, intervaly.end
                istart, iend = int((sx - s.start_x) / dx), int((ex - s.start_x) / dx)
                jstart, jend = int((sy - s.start_y) / dy), int((ey - s.start_y) / dy)
                pixels[jstart:jend, istart:iend] = 1
        return xarr, yarr, pixels

    def _detect_poles(self, x, y):
        """ Try to detect for discontinuities by computing the numerical 
        gradient of the provided data.

        NOTE: The data produced by the series module is perfectly fine. It is 
        just the way the data is rendered that connects segments between
        discontinuities, so it makes sense for this function to be placed in
        the backend.

        Returns
        =======
            x, y : np.ndarrays
            modified : boolean
                If the data has been processed and the y-range has changed, then
                `modified=True`. This will be used by Bokeh in order to update 
                the y-range.
        """
        if len(x) < 5:
            # failsafe mechanism: when we plot Piecewise functions, there could
            # be pieces to be evaluated at specific locations, for example x=1.
            # Say we want to plot y=2 at x=0, x=1, x=2, ... These pieces are 
            # going to be combined by Piecewise, thus obtaining a "line" with
            # a few number of points. The number 5 used above is just a 
            # reasonable assumption about the number of those points that are
            # going to be combined. Given the low number of these points, we
            # don't want them to be evaluated by the following algorithm, because 
            # it could fail or raise warnings.
            return x, y, False

        try:
            # TODO: here I used a try-except because there might be times where
            # there are None values inside y. Let's suppose we are plotting
            # log(x), (x, -10, 10). Then for x <= 0 there will be None values,
            # which makes the algorithm fails (can't subtract None values)
            if self.detect_poles:
                # TODO: should eps be a function of the number of discretization 
                # points and the x-range?
                eps = self._kwargs.get("eps", 1e-01)
                yr = np.roll(y, -1)
                # need to set this condition, otherwise there is a change that a
                # "false positive" discontinuity gets inserted at the end of y,
                # then setting ylim when it should not.
                yr[-1] = y[-1]
                b = np.abs((yr - y)) / np.abs(x)
                b = np.arctan(b)
                c = y
                idx = np.abs(b - np.pi / 2) < eps
                c[idx] = np.nan
                yy = c.copy()
                c = np.ma.masked_invalid(c)
                if any(idx) and (self.ylim is None):
                    # auto select a ylim range. At this point, yy contains NaN 
                    # values at the discontinuities. I'm going to combine two 
                    # strategies:
                    # 1. select the minimum positive value and the maximum negative
                    #   value just before a discontinuity.
                    # 2. compute area_rms, a route mean square of the areas of the 
                    #   rectangles (x[i] - x[i-1]) * y[i]
                    #   Then mask away yy where the areas at y[i] are greater than
                    #   area_rms
                    
                    # select indeces just before and just after NaN values
                    idx = np.argwhere(np.isnan(yy)).reshape(-1)
                    idxb, idxp = idx - 1, idx + 1
                    idx = [i for i in list(idxb) + list(idxp) if ((i >= 0) and 
                            (i < len(yy)))]
                    v = yy[idx]
                    vp = [k for k in v if k >= 0]
                    vn = [k for k in v if k < 0]
                    max1 = np.inf if vp == [] else np.min(vp)
                    min1 = -np.inf if vn == [] else np.max(vn)

                    # root mean square approach
                    areas = np.abs(np.roll(x, -1) - x) * yy
                    area_rms = np.sqrt(np.mean([a**2 for a in areas if not np.isnan(a)]))
                    yy[np.abs(areas) > area_rms] = np.nan
                    min2, max2 = np.nanmin(yy), np.nanmax(yy)

                    self.ylim = np.max([min1, min2]), np.min([max1, max2])
                    return x, c, True
                return x, c, False
        except:
            pass
        return x, y, False

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
    
    def __add__(self, other):
        if not isinstance(other, Plot):
            raise TypeError(
            "Both sides of the `+` operator must be instances of the Plot " +
            "class.\n Received: {} + {}".format(type(self), type(other)))
        return self._do_sum(other)
        
    def __radd__(self, other):
        if not isinstance(other, Plot):
            raise TypeError(
            "Both sides of the `+` operator must be instances of the Plot " +
            "class.\n Received: {} + {}".format(type(self), type(other)))
        return other._do_sum(self)
    
    def _do_sum(self, other):
        """ Differently from Plot.extend, this method creates a new plot object,
        which uses the series of both plots and merges the _kwargs dictionary
        of `self` with the one of `other`.
        """
        series = []
        series.extend(self.series)
        series.extend(other.series)
        kwargs = merge({}, self._kwargs, other._kwargs)
        return type(self)(*series, **kwargs)

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
            # auto legend
            if len(self._series) > 1:
                    self.legend = True
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
        # auto legend
        if len(self._series) > 1:
            self.legend = True

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
