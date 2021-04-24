"""Plotting module for Sympy.

A plot is represented by the ``Plot`` class that contains a reference to the
backend and a list of the data series to be plotted. The data series are
instances of classes meant to simplify getting points and meshes from sympy
expressions. ``plot_backends`` is a dictionary with all the backends.

This module gives only the essential. For all the fancy stuff use directly
the backend. You can get the backend wrapper for every plot from the
``_backend`` attribute. Moreover the data series classes have various useful
methods like ``get_points``, ``get_meshes``, etc, that may
be useful if you wish to use another plotting library.

Especially if you need publication ready graphs and this module is not enough
for you - just get the ``_backend`` attribute and add whatever you want
directly to it. In the case of matplotlib (the common way to graph data in
python) just copy ``_backend.fig`` which is the figure and ``_backend.ax``
which is the axis and work on them as you would on any other matplotlib object.

Simplicity of code takes much greater importance than performance. Don't use it
if you care at all about performance. A new backend instance is initialized
every time you call ``show()`` and the old one is left to the garbage collector.
"""


from collections.abc import Callable

from sympy import sympify, Expr, Tuple, Dummy, Symbol
from sympy.external import import_module
from sympy.utilities.iterables import is_sequence

from sympy.utilities.exceptions import SymPyDeprecationWarning

# N.B.
# When changing the minimum module version for matplotlib, please change
# the same in the `SymPyDocTestFinder`` in `sympy/testing/runtests.py`

# Backend specific imports - textplot
from sympy.plotting.textplot import textplot

from sympy.plotting.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries,
    Parametric3DLineSeries, SurfaceOver2DRangeSeries,
    ParametricSurfaceSeries, ContourSeries, BaseSeries,
    _set_discretization_points
)

# Global variable
# Set to False when running tests / doctests so that the plots don't show.
_show = True


def unset_show():
    """
    Disable show(). For use in the tests.
    """
    global _show
    _show = False

##############################################################################
# The public interface
##############################################################################


class Plot:
    """The base class of the plotting module.

    Explanation
    ===========

    This class permits the plotting of sympy expressions using numerous
    backends (matplotlib, textplot, etc).

    The figure can contain an arbitrary number of plots of sympy expressions,
    lists of coordinates of points, etc. Plot exposes the attribute `series`
    that contains all data series to be plotted (expressions for lines or
    surfaces, lists of points, etc (all subclasses of BaseSeries)). Those data
    series are instances of classes not imported by ``from sympy import *``.

    The customization of the figure is on two levels. Global options that
    concern the figure as a whole (eg title, xlabel, scale, etc) and
    per-data series options (eg name) and aesthetics (eg. color, point shape,
    line type, etc.).

    The difference between options and aesthetics is that an aesthetic can be
    a function of the coordinates (or parameters in a parametric plot). The
    supported values for an aesthetic are:
    - None (the backend uses default values)
    - a constant
    - a function of one variable (the first coordinate or parameter)
    - a function of two variables (the first and second coordinate or
    parameters)
    - a function of three variables (only in nonparametric 3D plots)
    Their implementation depends on the backend so they may not work in some
    backends.

    If the plot is parametric and the arity of the aesthetic function permits
    it the aesthetic is calculated over parameters and not over coordinates.
    If the arity does not permit calculation over parameters the calculation is
    done over coordinates.

    Only cartesian coordinates are supported for the moment, but you can use
    the parametric plots to plot in polar, spherical and cylindrical
    coordinates.

    The arguments for the constructor Plot must be subclasses of BaseSeries.

    Any global option can be specified as a keyword argument.

    The global options for a figure are:

    - title : str
    - xlabel : str
    - ylabel : str
    - zlabel : str
    - legend : bool
    - xscale : {'linear', 'log'}
    - yscale : {'linear', 'log'}
    - axis : bool
    - axis_center : tuple of two floats or {'center', 'auto'}
    - xlim : tuple of two floats
    - ylim : tuple of two floats
    - zlim : tuple of two floats
    - aspect_ratio : tuple of two floats or {'auto'}
    - autoscale : bool
    - margin : float in [0, 1]
    - backend : {'default', 'matplotlib', 'text'} or a subclass of BaseBackend
    - size : optional tuple of two floats, (width, height); default: None

    The per data series options and aesthetics are:
    There are none in the base series. See below for options for subclasses.

    Some data series support additional aesthetics or options:

    ListSeries, LineOver1DRangeSeries, Parametric2DLineSeries,
    Parametric3DLineSeries support the following:

    Aesthetics:

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

    options:

    - label : str
    - steps : bool
    - integers_only : bool
    """

    def __new__(cls, *args, **kwargs):
        backend = cls._get_backend(kwargs)
        return backend(*args, **kwargs)

    @classmethod
    def _get_backend(cls, kwargs):
        backend_kw = kwargs.get("backend", "matplotlib")
        if isinstance(backend_kw, str):
            backend = plot_backends[backend_kw]
        elif (type(backend_kw) == type) and issubclass(backend_kw, cls):
            backend = backend_kw
        else:
            raise TypeError(
                "backend must be either a string or a subclass of Plot")
        return backend

    def __init__(self, *args, **kwargs):
        # Options for the graph as a whole.
        # The possible values for each option are described in the docstring of
        # Plot. They are based purely on convention, no checking is done.
        self.title = kwargs.get("title", None)
        self.xlabel = kwargs.get("xlabel", None)
        self.ylabel = kwargs.get("ylabel", None)
        self.zlabel = kwargs.get("zlabel", None)
        self.aspect_ratio = kwargs.get("aspect_ratio", "auto")
        self.axis_center = kwargs.get("axis_center", "auto")
        self.axis = kwargs.get("axis", True)
        self.xscale = kwargs.get("xscale", "linear")
        self.yscale = kwargs.get("yscale", "linear")
        self.zscale = kwargs.get("zscale", "linear")
        self.legend = kwargs.get("legend", False)
        self.autoscale = kwargs.get("autoscale", True)
        self.margin = kwargs.get("margin", 0)

        # Contains the data objects to be plotted. The backend should be smart
        # enough to iterate over this list.
        self._series = []
        self._series.extend(args)

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

##############################################################################
# Backends
##############################################################################

class BaseBackend(Plot):
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
        main attributes of the plot (eg axis labels, title, ...). Note that a
        instance of BaseBackend is also an instance of Plot.
    3. When the "show" command is executed, the backend will loops through each
        series object to generate and plot the numerical data and set the axis
        labels, title, ..., according to the provided values.

    The backend should check if it supports the data series that it's given
    (eg TextBackend supports only LineOver1DRange).

    It's the backend responsibility to know how to use the class of data series
    that it's given. Please, explore the `MatplotlibBackend` source code to
    understand how a backend should be coded.

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
    
    Also, the following attribute is required:

    * self._fig: (instance attribute) it stores the backend-specific plot
        object/s, which can be retrieved with the Plot.fig attribute. These
        objects can then be used to further customize the resulting plot, using
        backend-specific commands. For example, MatplotlibBackend stores a tuple
        (figure, axes).
    
    * support_plotgrid: (class attribute, boolean) if True it means the backend
        is able to generate `PlotGrid` objects. Please, look at 
        `MatplotlibBackend` as an example of a backend supporting `PlotGrid`.

    See also
    ========

    MatplotlibBackend
    """

    # Set it to True in the subclasses if they are able to generate plot grids.
    # Also, clearly states in the docstring of the backend if it supports
    # plotgrids or not
    support_plotgrid = False

    def __new__(cls, *args, **kwargs):
        # This method is needed since the parent class Plot also implements its
        # __new__ method to provide the backend choice.
        # Without BaseBackend.__new__ there would be infinite recursion.
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Don't have to check for the success of importing matplotlib in each case;
# we will only be using this backend if we can successfully import matploblib
class MatplotlibBackend(BaseBackend):
    """ This class implements the functionalities to use Matplotlib with SymPy
    plotting functions.
    """
    # this backend supports PlotGrid
    support_plotgrid = True

    def __init__(self, *args, **kwargs):
        # set global options like title, axis labels, ...
        super().__init__(*args, **kwargs)

        # Backend specific functionalities
        # TODO: we should remove these!!!!!!!
        self.annotations = kwargs.get("annotations", None)
        self.markers = kwargs.get("markers", None)
        self.rectangles = kwargs.get("rectangles", None)
        self.fill = kwargs.get("fill", None)
    
    def _create_figure(self):
        self.matplotlib = import_module('matplotlib',
            import_kwargs={'fromlist': ['pyplot', 'cm', 'collections']},
            min_module_version='1.1.0', catch=(RuntimeError,))
        self.plt = self.matplotlib.pyplot
        self.cm = self.matplotlib.cm
        self.LineCollection = self.matplotlib.collections.LineCollection

        aspect = self.aspect_ratio
        if aspect != 'auto':
            aspect = float(aspect[1]) / aspect[0]

        if (self.nrows == 1) and (self.ncols == 1):
            # ordinary plot
            series_list = [self.series]
        else:
            # PlotGrid
            series_list = self.series

        self.ax = []
        self._fig = self.plt.figure(figsize=self.size)

        for i, series in enumerate(series_list):
            are_3D = [s.is_3D for s in series]

            if any(are_3D) and not all(are_3D):
                raise ValueError('The matplotlib backend can not mix 2D and 3D.')
            elif all(are_3D):
                self.ax.append(self._fig.add_subplot(self.nrows, self.ncols,
                        i + 1, projection='3d', aspect=aspect))

            elif not any(are_3D):
                self.ax.append(self._fig.add_subplot(self.nrows, self.ncols,
                        i + 1, aspect=aspect))
                self.ax[i].spines['left'].set_position('zero')
                self.ax[i].spines['right'].set_color('none')
                self.ax[i].spines['bottom'].set_position('zero')
                self.ax[i].spines['top'].set_color('none')
                self.ax[i].xaxis.set_ticks_position('bottom')
                self.ax[i].yaxis.set_ticks_position('left')

    @property
    def fig(self):
        """ Returns the objects used to render/display the plots
        """
        return self._fig, self.ax

    @staticmethod
    def get_segments(x, y, z=None):
        """ Convert two list of coordinates to a list of segments to be used
        with Matplotlib's LineCollection.

        Parameters
        ==========
            x: list
                List of x-coordinates

            y: list
                List of y-coordinates

            z: list
                List of z-coordinates for a 3D line.
        """
        np = import_module('numpy')
        if z is not None:
            dim = 3
            points = (x, y, z)
        else:
            dim = 2
            points = (x, y)
        points = np.ma.array(points).T.reshape(-1, 1, dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def _process_series(self, series, ax, parent):
        np = import_module('numpy')
        mpl_toolkits = import_module(
            'mpl_toolkits', import_kwargs={'fromlist': ['mplot3d']})

        # XXX Workaround for matplotlib issue
        # https://github.com/matplotlib/matplotlib/issues/17130
        xlims, ylims, zlims = [], [], []

        # TODO: do I need this?
        ax.cla()

        for s in series:
            # Create the collections
            if s.is_2Dline:
                x, y = s.get_data()
                if (isinstance(s.line_color, (int, float)) or
                        callable(s.line_color)):
                    segments = self.get_segments(x, y)
                    collection = self.LineCollection(segments)
                    collection.set_array(s.get_color_array())
                    ax.add_collection(collection)
                else:
                    line, = ax.plot(x, y, label=s.label, color=s.line_color)
            elif s.is_contour:
                ax.contour(*s.get_meshes())
            elif s.is_3Dline:
                x, y, z = s.get_data()
                if (isinstance(s.line_color, (int, float)) or
                        callable(s.line_color)):
                    art3d = mpl_toolkits.mplot3d.art3d
                    segments = self.get_segments(x, y, z)
                    collection = art3d.Line3DCollection(segments)
                    collection.set_array(s.get_color_array())
                    ax.add_collection(collection)
                else:
                    ax.plot(x, y, z, label=s.label,
                        color=s.line_color)

                xlims.append(s._xlim)
                ylims.append(s._ylim)
                zlims.append(s._zlim)
            elif s.is_3Dsurface:
                x, y, z = s.get_meshes()
                collection = ax.plot_surface(x, y, z,
                    cmap=getattr(self.cm, 'viridis', self.cm.jet),
                    rstride=1, cstride=1, linewidth=0.1)
                if isinstance(s.surface_color, (float, int)) or isinstance(s.surface_color, Callable):
                    color_array = s.get_color_array()
                    color_array = color_array.reshape(color_array.size)
                    collection.set_array(color_array)
                else:
                    collection.set_color(s.surface_color)

                xlims.append(s._xlim)
                ylims.append(s._ylim)
                zlims.append(s._zlim)
            elif s.is_implicit:
                points = s.get_raster()
                if len(points) == 2:
                    # interval math plotting
                    x, y = _matplotlib_list(points[0])
                    ax.fill(x, y, facecolor=s.line_color, edgecolor='None')
                else:
                    # use contourf or contour depending on whether it is
                    # an inequality or equality.
                    # XXX: ``contour`` plots multiple lines. Should be fixed.
                    ListedColormap = self.matplotlib.colors.ListedColormap
                    colormap = ListedColormap(["white", s.line_color])
                    xarray, yarray, zarray, plot_type = points
                    if plot_type == 'contour':
                        ax.contour(xarray, yarray, zarray, cmap=colormap)
                    else:
                        ax.contourf(xarray, yarray, zarray, cmap=colormap)
            else:
                raise NotImplementedError(
                    '{} is not supported in the sympy plotting module '
                    'with matplotlib backend. Please report this issue.'
                    .format(ax))

        Axes3D = mpl_toolkits.mplot3d.Axes3D
        if not isinstance(ax, Axes3D):
            ax.autoscale_view(
                scalex=ax.get_autoscalex_on(),
                scaley=ax.get_autoscaley_on())
        else:
            # XXX Workaround for matplotlib issue
            # https://github.com/matplotlib/matplotlib/issues/17130
            if xlims:
                xlims = np.array(xlims)
                xlim = (np.amin(xlims[:, 0]), np.amax(xlims[:, 1]))
                ax.set_xlim(xlim)
            else:
                ax.set_xlim([0, 1])

            if ylims:
                ylims = np.array(ylims)
                ylim = (np.amin(ylims[:, 0]), np.amax(ylims[:, 1]))
                ax.set_ylim(ylim)
            else:
                ax.set_ylim([0, 1])

            if zlims:
                zlims = np.array(zlims)
                zlim = (np.amin(zlims[:, 0]), np.amax(zlims[:, 1]))
                ax.set_zlim(zlim)
            else:
                ax.set_zlim([0, 1])

        # Set global options.
        # TODO The 3D stuff
        # XXX The order of those is important.
        if parent.xscale and not isinstance(ax, Axes3D):
            ax.set_xscale(parent.xscale)
        if parent.yscale and not isinstance(ax, Axes3D):
            ax.set_yscale(parent.yscale)
        if not isinstance(ax, Axes3D) or self.matplotlib.__version__ >= '1.2.0':  # XXX in the distant future remove this check
            ax.set_autoscale_on(parent.autoscale)
        if parent.axis_center:
            val = parent.axis_center
            if isinstance(ax, Axes3D):
                pass
            elif val == 'center':
                ax.spines['left'].set_position('center')
                ax.spines['bottom'].set_position('center')
            elif val == 'auto':
                xl, xh = ax.get_xlim()
                yl, yh = ax.get_ylim()
                pos_left = ('data', 0) if xl*xh <= 0 else 'center'
                pos_bottom = ('data', 0) if yl*yh <= 0 else 'center'
                ax.spines['left'].set_position(pos_left)
                ax.spines['bottom'].set_position(pos_bottom)
            else:
                ax.spines['left'].set_position(('data', val[0]))
                ax.spines['bottom'].set_position(('data', val[1]))
        if not parent.axis:
            ax.set_axis_off()
        if parent.legend:
            if ax.legend():
                ax.legend_.set_visible(parent.legend)
        if parent.margin:
            ax.set_xmargin(parent.margin)
            ax.set_ymargin(parent.margin)
        if parent.title:
            ax.set_title(parent.title)
        if parent.xlabel:
            ax.set_xlabel(parent.xlabel, position=(1, 0))
        if parent.ylabel:
            ax.set_ylabel(parent.ylabel, position=(0, 1))
        if isinstance(ax, Axes3D) and parent.zlabel:
            ax.set_zlabel(parent.zlabel, position=(0, 1))
        if parent.annotations:
            for a in parent.annotations:
                ax.annotate(**a)
        if parent.markers:
            for marker in parent.markers:
                # make a copy of the marker dictionary
                # so that it doesn't get altered
                m = marker.copy()
                args = m.pop('args')
                ax.plot(*args, **m)
        if parent.rectangles:
            for r in parent.rectangles:
                rect = self.matplotlib.patches.Rectangle(**r)
                ax.add_patch(rect)
        if parent.fill:
            ax.fill_between(**parent.fill)

        # xlim and ylim shoulld always be set at last so that plot limits
        # doesn't get altered during the process.
        if parent.xlim:
            ax.set_xlim(parent.xlim)
        if parent.ylim:
            ax.set_ylim(parent.ylim)


    def process_series(self):
        """
        Iterates over every ``Plot`` object and further calls
        _process_series()
        """
        # create the figure from scratch every time, otherwise if the plot was
        # previously shown, it would not be possible to show it again. This
        # behaviour is specific to Matplotlib
        self._create_figure()

        if (self.nrows == 1) and (self.ncols == 1):
            # ordinary plot
            series_list = [self.series]
        else:
            # PlotGrid
            series_list = self.series

        parent = self
        for i, (series, ax) in enumerate(zip(series_list, self.ax)):
            if not ((self.nrows == 1) and (self.ncols == 1)):
                # PlotGrid
                parent = self.subplots[i]
            self._process_series(series, ax, parent)

    def show(self):
        """ Display the current plot.
        """
        self.process_series()
        if _show:
            self._fig.tight_layout()
            self._fig.show()
        else:
            self.close()

    def save(self, path):
        """ Save the current plot at the specified location.

        Parameters
        ==========

            path : str
                The filename of the output image.
            
            kwargs : dict
                Optional backend-specific parameters.
        """
        # the plot must first be created and then saved
        # TODO: do I really need to show the plot in order to save it, or can I
        # just create it and the save it?
        self.show()
        self._fig.savefig(path)

    def close(self):
        """ Close the current plot.
        """
        self.plt.close(self._fig)


class TextBackend(BaseBackend):
    def show(self):
        if not _show:
            return
        if len(self.series) != 1:
            raise ValueError(
                'The TextBackend supports only one graph per Plot.')
        elif not isinstance(self.series[0], LineOver1DRangeSeries):
            raise ValueError(
                'The TextBackend supports only expressions over a 1D range')
        else:
            ser = self.series[0]
            textplot(ser.expr, ser.start, ser.end)


class DefaultBackend(BaseBackend):
    def __new__(cls, *args, **kwargs):
        matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
        if matplotlib:
            return MatplotlibBackend(*args, **kwargs)
        else:
            return TextBackend(*args, **kwargs)


plot_backends = {
    'matplotlib': MatplotlibBackend,
    'text': TextBackend,
    'default': DefaultBackend
}


##############################################################################
# Finding the centers of line segments or mesh faces
##############################################################################

def _matplotlib_list(interval_list):
    """
    Returns lists for matplotlib ``fill`` command from a list of bounding
    rectangular intervals
    """
    xlist = []
    ylist = []
    if len(interval_list):
        for intervals in interval_list:
            intervalx = intervals[0]
            intervaly = intervals[1]
            xlist.extend([intervalx.start, intervalx.start,
                          intervalx.end, intervalx.end, None])
            ylist.extend([intervaly.start, intervaly.end,
                          intervaly.end, intervaly.start, None])
    else:
        #XXX Ugly hack. Matplotlib does not accept empty lists for ``fill``
        xlist.extend([None, None, None, None])
        ylist.extend([None, None, None, None])
    return xlist, ylist


####New API for plotting module ####

# TODO: Add color arrays for plots.
# TODO: Add more plotting options for 3d plots.
# TODO: Adaptive sampling for 3D plots.

def plot(*args, show=True, **kwargs):
    """Plots a function of a single variable as a curve.

    Parameters
    ==========

    args :
        The first argument is the expression representing the function
        of single variable to be plotted.

        The next argument is a 3-tuple denoting the range of the free
        variable. e.g. ``(x, 0, 5)``

        The last optional argument is a string denoting the label of the
        expression to be visualized on the legend. If not provided, the label
        will be the string representation of the expression.

        Typical usage examples are in the followings:

        - Plotting a single expression with a single range.
            ``plot(expr, range, **kwargs)``
        - Plotting a single expression with the default range (-10, 10).
            ``plot(expr, **kwargs)``
        - Plotting multiple expressions with a single range.
            ``plot(expr1, expr2, ..., range, **kwargs)``
        - Plotting multiple expressions with multiple ranges.
            ``plot((expr1, range1), (expr2, range2), ..., **kwargs)``
        - Plotting multiple expressions with multiple ranges and custom labels.
            ``plot((expr1, range1, label1), (expr2, range2, label2), ..., legend=True, **kwargs)``

        It is best practice to specify range explicitly because default
        range may change in the future if a more advanced default range
        detection algorithm is implemented.

    show : bool, optional
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    label : str, optional
        This option is no longer used. Refer to ``args`` to understand
        how to set labels which will be visualized when called with
        ``legend``.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.

    annotations : list, optional
        A list of dictionaries specifying the type of annotation
        required. The keys in the dictionary should be equivalent
        to the arguments of the matplotlib's annotate() function.

    markers : list, optional
        A list of dictionaries specifying the type the markers required.
        The keys in the dictionary should be equivalent to the arguments
        of the matplotlib's plot() function along with the marker
        related keyworded arguments.

    rectangles : list, optional
        A list of dictionaries specifying the dimensions of the
        rectangles to be plotted. The keys in the dictionary should be
        equivalent to the arguments of the matplotlib's
        patches.Rectangle class.

    fill : dict, optional
        A dictionary specifying the type of color filling required in
        the plot. The keys in the dictionary should be equivalent to the
        arguments of the matplotlib's fill_between() function.

    adaptive : bool, optional
        The default value is set to ``True``. Set adaptive to ``False``
        and specify ``n`` if uniform sampling is required.

        The plotting uses an adaptive algorithm which samples
        recursively to accurately plot. The adaptive algorithm uses a
        random point near the midpoint of two points that has to be
        further sampled. Hence the same plots can appear slightly
        different.

    depth : int, optional
        Recursion depth of the adaptive algorithm. A depth of value
        ``n`` samples a maximum of `2^{n}` points.

        If the ``adaptive`` flag is set to ``False``, this will be
        ignored.

    n : int, optional
        Used when the ``adaptive`` is set to ``False``. The function
        is uniformly sampled at ``n`` number of points.

        If the ``adaptive`` flag is set to ``True``, this will be
        ignored.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import symbols
       >>> from sympy.plotting import plot
       >>> x = symbols('x')

    Single Plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x**2, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-5.0, 5.0)

    Multiple plots with single range.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x, x**2, x**3, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x for x over (-5.0, 5.0)
       [1]: cartesian line: x**2 for x over (-5.0, 5.0)
       [2]: cartesian line: x**3 for x over (-5.0, 5.0)

    Multiple plots with different ranges and custom labels.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot((x**2, (x, -6, 6), "$f_{1}$"), (x, (x, -5, 5), "f2"), legend=True)
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-6.0, 6.0)
       [1]: cartesian line: x for x over (-5.0, 5.0)

    No adaptive sampling.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x**2, adaptive=False, n=400)
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-10.0, 10.0)

    See Also
    ========

    Plot, LineOver1DRangeSeries

    """
    args = _plot_sympify(args)
    free = set()
    for a in args:
        if isinstance(a, Expr):
            free |= a.free_symbols
            if len(free) > 1:
                raise ValueError(
                    'The same variable should be used in all '
                    'univariate expressions being plotted.')
    x = free.pop() if free else Symbol('x')
    kwargs.setdefault('xlabel', x.name)
    kwargs.setdefault('ylabel', 'f(%s)' % x.name)
    kwargs = _remove_label_key(kwargs)
    kwargs = _set_discretization_points(kwargs, LineOver1DRangeSeries)
    series = []
    plot_expr = _check_arguments(args, 1, 1)
    series = [LineOver1DRangeSeries(*arg, **kwargs) for arg in plot_expr]

    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot_parametric(*args, show=True, **kwargs):
    """
    Plots a 2D parametric curve.

    Parameters
    ==========

    args
        Common specifications are:

        - Plotting a single parametric curve with a range
            ``plot_parametric((expr_x, expr_y), range)``
        - Plotting multiple parametric curves with the same range
            ``plot_parametric((expr_x, expr_y), ..., range)``
        - Plotting multiple parametric curves with different ranges
            ``plot_parametric((expr_x, expr_y, range), ...)``
        - Plotting multiple parametric curves with different ranges and
            custom labels
            ``plot_parametric((expr_x, expr_y, range, label), ...)``

        ``expr_x`` is the expression representing $x$ component of the
        parametric function.

        ``expr_y`` is the expression representing $y$ component of the
        parametric function.

        ``range`` is a 3-tuple denoting the parameter symbol, start and
        stop. For example, ``(u, 0, 5)``.

        ``label`` is an optional string denoting the label of the expression
        to be visualized on the legend. If not provided, the label will be the
        string representation of the expression.

        If the range is not specified, then a default range of (-10, 10)
        is used.

        However, if the arguments are specified as
        ``(expr_x, expr_y, range), ...``, you must specify the ranges
        for each expressions manually.

        Default range may change in the future if a more advanced
        algorithm is implemented.

    adaptive : bool, optional
        Specifies whether to use the adaptive sampling or not.

        The default value is set to ``True``. Set adaptive to ``False``
        and specify ``n`` if uniform sampling is required.

    depth :  int, optional
        The recursion depth of the adaptive algorithm. A depth of
        value $n$ samples a maximum of $2^n$ points.

    n : int, optional
        Used when the ``adaptive`` flag is set to ``False``.

        Specifies the number of the points used for the uniform
        sampling.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    label : str, optional
        This option is no longer used. Refer to ``args`` to understand
        how to set labels which will be visualized when called with
        ``legend``.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot_parametric
       >>> u = symbols('u')

    A parametric plot with a single expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, -5, 5))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)

    A parametric plot with multiple expressions with the same range:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, cos(u)), (u, -10, 10))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-10.0, 10.0)
       [1]: parametric cartesian line: (u, cos(u)) for u over (-10.0, 10.0)

    A parametric plot with multiple expressions with different ranges and
    custom labels for each curve:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u), (u, -5, 5), "a"),
       ...     (cos(u), u, "b"))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)
       [1]: parametric cartesian line: (cos(u), u) for u over (-10.0, 10.0)

    Notes
    =====

    The plotting uses an adaptive algorithm which samples recursively to
    accurately plot the curve. The adaptive algorithm uses a random point
    near the midpoint of two points that has to be further sampled.
    Hence, repeating the same plot command can give slightly different
    results because of the random sampling.

    If there are multiple plots, then the same optional arguments are
    applied to all the plots drawn in the same canvas. If you want to
    set these options separately, you can index the returned ``Plot``
    object and set it.

    For example, when you specify ``line_color`` once, it would be
    applied simultaneously to both series.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> from sympy import pi
        >>> expr1 = (u, cos(2*pi*u)/2 + 1/2)
        >>> expr2 = (u, sin(2*pi*u)/2 + 1/2)
        >>> p = plot_parametric(expr1, expr2, (u, 0, 1), line_color='blue')

    If you want to specify the line color for the specific series, you
    should index each item and apply the property manually.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> p[0].line_color = 'red'
        >>> p.show()

    See Also
    ========

    Plot, Parametric2DLineSeries
    """
    args = _plot_sympify(args)
    series = []
    kwargs = _remove_label_key(kwargs)
    kwargs = _set_discretization_points(kwargs, Parametric2DLineSeries)
    plot_expr = _check_arguments(args, 2, 1)
    series = [Parametric2DLineSeries(*arg, **kwargs) for arg in plot_expr]
    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot3d_parametric_line(*args, show=True, **kwargs):
    """
    Plots a 3D parametric line plot.

    Usage
    =====

    Single plot:

    ``plot3d_parametric_line(expr_x, expr_y, expr_z, range, label, **kwargs)``

    If the range is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_line((expr_x, expr_y, expr_z, range, label), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    ``expr_x`` : Expression representing the function along x.

    ``expr_y`` : Expression representing the function along y.

    ``expr_z`` : Expression representing the function along z.

    ``range``: ``(u, 0, 5)``, A 3-tuple denoting the range of the parameter
    variable.

    ``label`` : An optional string denoting the label of the expression
        to be visualized on the legend. If not provided, the label will be the
        string representation of the expression.

    Keyword Arguments
    =================

    Arguments for ``Parametric3DLineSeries`` class.

    ``n``: The range is uniformly sampled at ``n`` number of points.

    Aesthetics:

    ``line_color``: string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    ``label``: str
        The label to the plot. It will be used when called with ``legend=True``
        to denote the function with the given label in the plot.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class.

    ``title`` : str. Title of the plot.

    ``size`` : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_line
       >>> u = symbols('u')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line(cos(u), sin(u), u, (u, -5, 5))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)


    Multiple plots with different ranges and custom labels.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line((cos(u), sin(u), u, (u, -5, 5), "a"),
       ...     (sin(u), u**2, u, (u, -3, 3), "b"), legend=True)
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)
       [1]: 3D parametric cartesian line: (sin(u), u**2, u) for u over (-3.0, 3.0)


    See Also
    ========

    Plot, Parametric3DLineSeries

    """
    args = _plot_sympify(args)
    kwargs = _remove_label_key(kwargs)
    kwargs = _set_discretization_points(kwargs, Parametric3DLineSeries)
    series = []
    plot_expr = _check_arguments(args, 3, 1)
    series = [Parametric3DLineSeries(*arg, **kwargs) for arg in plot_expr]
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")
    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot3d(*args, show=True, **kwargs):
    """
    Plots a 3D surface plot.

    Usage
    =====

    Single plot

    ``plot3d(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot3d(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot3d((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    ``expr`` : Expression representing the function along x.

    ``range_x``: (x, 0, 5), A 3-tuple denoting the range of the x
    variable.

    ``range_y``: (y, 0, 5), A 3-tuple denoting the range of the y
     variable.

    Keyword Arguments
    =================

    Arguments for ``SurfaceOver2DRangeSeries`` class:

    ``n1``: int. The x range is sampled uniformly at ``n1`` of points.

    ``n2``: int. The y range is sampled uniformly at ``n2`` of points.

    ``n``: int. The x and y ranges are sampled uniformly at ``n`` of points.
    It overrides ``n1`` and ``n2``.

    Aesthetics:

    ``surface_color``: Function which returns a float. Specifies the color for
    the surface of the plot. See ``sympy.plotting.Plot`` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    ``title`` : str. Title of the plot.
    ``size`` : (float, float), optional
    A tuple in the form (width, height) in inches to specify the size of the
    overall figure. The default value is set to ``None``, meaning the size will
    be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols
       >>> from sympy.plotting import plot3d
       >>> x, y = symbols('x y')

    Single plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(x*y, (x, -5, 5), (y, -5, 5))
       Plot object containing:
       [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)


    Multiple plots with same range

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(x*y, -x*y, (x, -5, 5), (y, -5, 5))
       Plot object containing:
       [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)
       [1]: cartesian surface: -x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)


    Multiple plots with different ranges.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d((x**2 + y**2, (x, -5, 5), (y, -5, 5)),
       ...     (x*y, (x, -3, 3), (y, -3, 3)))
       Plot object containing:
       [0]: cartesian surface: x**2 + y**2 for x over (-5.0, 5.0) and y over (-5.0, 5.0)
       [1]: cartesian surface: x*y for x over (-3.0, 3.0) and y over (-3.0, 3.0)


    See Also
    ========

    Plot, SurfaceOver2DRangeSeries

    """

    args = _plot_sympify(args)
    kwargs = _remove_label_key(kwargs)
    kwargs = _set_discretization_points(kwargs, SurfaceOver2DRangeSeries)
    series = []
    plot_expr = _check_arguments(args, 1, 2)
    series = [SurfaceOver2DRangeSeries(*arg, **kwargs) for arg in plot_expr]
    xlabel = series[0].var_x.name
    ylabel = series[0].var_y.name
    kwargs.setdefault("xlabel", xlabel)
    kwargs.setdefault("ylabel", ylabel)
    kwargs.setdefault("zlabel", "f(%s, %s)" % (xlabel, ylabel))
    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot3d_parametric_surface(*args, show=True, **kwargs):
    """
    Plots a 3D parametric surface plot.

    Explanation
    ===========

    Single plot.

    ``plot3d_parametric_surface(expr_x, expr_y, expr_z, range_u, range_v, **kwargs)``

    If the ranges is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_surface((expr_x, expr_y, expr_z, range_u, range_v), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    ``expr_x``: Expression representing the function along ``x``.

    ``expr_y``: Expression representing the function along ``y``.

    ``expr_z``: Expression representing the function along ``z``.

    ``range_u``: ``(u, 0, 5)``,  A 3-tuple denoting the range of the ``u``
    variable.

    ``range_v``: ``(v, 0, 5)``,  A 3-tuple denoting the range of the v
    variable.

    Keyword Arguments
    =================

    Arguments for ``ParametricSurfaceSeries`` class:

    ``n1``: int. The ``u`` range is sampled uniformly at ``n1`` of points

    ``n2``: int. The ``v`` range is sampled uniformly at ``n2`` of points

    ``n``: int. The u and v ranges are sampled uniformly at ``n`` of points.
    It overrides ``n1`` and ``n2``.

    Aesthetics:

    ``surface_color``: Function which returns a float. Specifies the color for
    the surface of the plot. See ``sympy.plotting.Plot`` for more details.

    If there are multiple plots, then the same series arguments are applied for
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.


    Arguments for ``Plot`` class:

    ``title`` : str. Title of the plot.
    ``size`` : (float, float), optional
    A tuple in the form (width, height) in inches to specify the size of the
    overall figure. The default value is set to ``None``, meaning the size will
    be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_surface
       >>> u, v = symbols('u v')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_surface(cos(u + v), sin(u - v), u - v,
       ...     (u, -5, 5), (v, -5, 5))
       Plot object containing:
       [0]: parametric cartesian surface: (cos(u + v), sin(u - v), u - v) for u over (-5.0, 5.0) and v over (-5.0, 5.0)


    See Also
    ========

    Plot, ParametricSurfaceSeries

    """

    args = _plot_sympify(args)
    kwargs = _remove_label_key(kwargs)
    kwargs = _set_discretization_points(kwargs, ParametricSurfaceSeries)
    plot_expr = _check_arguments(args, 3, 2)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")
    series = [ParametricSurfaceSeries(*arg, **kwargs) for arg in plot_expr]
    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots

def plot_contour(*args, show=True, **kwargs):
    """
    Draws contour plot of a function

    Usage
    =====

    Single plot

    ``plot_contour(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot_contour(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot_contour((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    ``expr`` : Expression representing the function along x.

    ``range_x``: (x, 0, 5), A 3-tuple denoting the range of the x
    variable.

    ``range_y``: (y, 0, 5), A 3-tuple denoting the range of the y
     variable.

    Keyword Arguments
    =================

    Arguments for ``ContourSeries`` class:

    ``n1``: int. The x range is sampled uniformly at ``n1`` of points.

    ``n2``: int. The y range is sampled uniformly at ``n2`` of points.

    ``n``: int. The x and y ranges are sampled uniformly at ``n`` of points.
    It overrides ``n1`` and ``n2``.

    Aesthetics:

    ``surface_color``: Function which returns a float. Specifies the color for
    the surface of the plot. See ``sympy.plotting.Plot`` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    ``title`` : str. Title of the plot.
    ``size`` : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    See Also
    ========

    Plot, ContourSeries

    """

    args = _plot_sympify(args)
    kwargs = _remove_label_key(kwargs)
    kwargs = _set_discretization_points(kwargs, ContourSeries)
    plot_expr = _check_arguments(args, 1, 2)
    series = [ContourSeries(*arg, **kwargs) for arg in plot_expr]
    plot_contours = Plot(*series, **kwargs)
    if show:
        plot_contours.show()
    return plot_contours

def check_arguments(args, expr_len, nb_of_free_symbols):
    """
    Checks the arguments and converts into tuples of the
    form (exprs, ranges).
    """

    SymPyDeprecationWarning(
        feature = "check_arguments",
        useinstead = "_check_arguments",
        issue = 21330,
        deprecated_since_version = "1.9").warn()

    if not args:
        return []
    if expr_len > 1 and isinstance(args[0], Expr):
        # Multiple expressions same range.
        # The arguments are tuples when the expression length is
        # greater than 1.
        if len(args) < expr_len:
            raise ValueError("len(args) should not be less than expr_len")
        for i in range(len(args)):
            if isinstance(args[i], Tuple):
                break
        else:
            i = len(args) + 1

        exprs = Tuple(*args[:i])
        free_symbols = list(set().union(*[e.free_symbols for e in exprs]))
        if len(args) == expr_len + nb_of_free_symbols:
            #Ranges given
            plots = [exprs + Tuple(*args[expr_len:])]
        else:
            default_range = Tuple(-10, 10)
            ranges = []
            for symbol in free_symbols:
                ranges.append(Tuple(symbol) + default_range)

            for i in range(len(free_symbols) - nb_of_free_symbols):
                ranges.append(Tuple(Dummy()) + default_range)
            plots = [exprs + Tuple(*ranges)]
        return plots

    if isinstance(args[0], Expr) or (isinstance(args[0], Tuple) and
                                     len(args[0]) == expr_len and
                                     expr_len != 3):
        # Cannot handle expressions with number of expression = 3. It is
        # not possible to differentiate between expressions and ranges.
        #Series of plots with same range
        for i in range(len(args)):
            if isinstance(args[i], Tuple) and len(args[i]) != expr_len:
                break
            if not isinstance(args[i], Tuple):
                args[i] = Tuple(args[i])
        else:
            i = len(args) + 1

        exprs = args[:i]
        assert all(isinstance(e, Expr) for expr in exprs for e in expr)
        free_symbols = list(set().union(*[e.free_symbols for expr in exprs
                                        for e in expr]))

        if len(free_symbols) > nb_of_free_symbols:
            raise ValueError("The number of free_symbols in the expression "
                             "is greater than %d" % nb_of_free_symbols)
        if len(args) == i + nb_of_free_symbols and isinstance(args[i], Tuple):
            ranges = Tuple(*[range_expr for range_expr in args[
                           i:i + nb_of_free_symbols]])
            plots = [expr + ranges for expr in exprs]
            return plots
        else:
            # Use default ranges.
            default_range = Tuple(-10, 10)
            ranges = []
            for symbol in free_symbols:
                ranges.append(Tuple(symbol) + default_range)

            for i in range(nb_of_free_symbols - len(free_symbols)):
                ranges.append(Tuple(Dummy()) + default_range)
            ranges = Tuple(*ranges)
            plots = [expr + ranges for expr in exprs]
            return plots

    elif isinstance(args[0], Tuple) and len(args[0]) == expr_len + nb_of_free_symbols:
        # Multiple plots with different ranges.
        for arg in args:
            for i in range(expr_len):
                if not isinstance(arg[i], Expr):
                    raise ValueError("Expected an expression, given %s" %
                                     str(arg[i]))
            for i in range(nb_of_free_symbols):
                if not len(arg[i + expr_len]) == 3:
                    raise ValueError("The ranges should be a tuple of "
                                     "length 3, got %s" % str(arg[i + expr_len]))
        return args


def _is_range(r):
    """ A range is defined as (symbol, start, end). start and end should
    be numbers.
    """
    return (isinstance(r, Tuple) and (len(r) == 3) and
                r.args[1].is_number and r.args[2].is_number)

def _create_ranges(free_symbols, ranges, npar):
    """ This function does two things:
    1. Check if the number of free symbols is in agreement with the type of plot
        chosen. For example, plot() requires 1 free symbol; plot3d() requires 2
        free symbols.
    2. Sometime users create plots without providing ranges for the variables.
        Here we create the necessary ranges.

    free_symbols
        The free symbols contained in the expressions to be plotted

    ranges
        The limiting ranges provided by the user

    npar
        The number of free symbols required by the plot functions. For example,
        npar=1 for plot, npar=2 for plot3d, ...

    """

    get_default_range = lambda symbol: Tuple(symbol, -10, 10)

    if len(free_symbols) > npar:
        raise ValueError(
            "Too many free symbols.\n" +
            "Expected {} free symbols.\n".format(npar) +
            "Received {}: {}".format(len(free_symbols), free_symbols)
        )

    if len(ranges) > npar:
        raise ValueError(
            "Too many ranges. Received %s, expected %s" % (len(ranges), npar))

    # free symbols in the ranges provided by the user
    rfs = set().union([r[0] for r in ranges])
    if len(rfs) != len(ranges):
        raise ValueError("Multiple ranges with the same symbol")

    if len(ranges) < npar:
        symbols = free_symbols.difference(rfs)
        if symbols != set():
            # add a range for each missing free symbols
            for s in symbols:
                ranges.append(get_default_range(s))
        # if there is still room, fill them with dummys
        for i in range(npar - len(ranges)):
            ranges.append(get_default_range(Dummy()))

    return ranges

def _check_arguments(args, nexpr, npar):
    """ Checks the arguments and converts into tuples of the
    form (exprs, ranges, name_expr).
    
    Parameters
    ==========

    args
        The arguments provided to the plot functions

    nexpr
        The number of sub-expression forming an expression to be plotted. For
        example:
        nexpr=1 for plot.
        nexpr=2 for plot_parametric: a curve is represented by a tuple of two
            elements.
        nexpr=1 for plot3d.
        nexpr=3 for plot3d_parametric_line: a curve is represented by a tuple
            of three elements.

    npar
        The number of free symbols required by the plot functions. For example,
        npar=1 for plot, npar=2 for plot3d, ...

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import cos, sin, symbols
       >>> from sympy.plotting.plot import _check_arguments
       >>> x = symbols('x')
       >>> _check_arguments([cos(x), sin(x)], 2, 1)
           [(cos(x), sin(x), (x, -10, 10), '(cos(x), sin(x))')]

       >>> _check_arguments([x, x**2], 1, 1)
           [(x, (x, -10, 10), 'x'), (x**2, (x, -10, 10), 'x**2')]
    """
    if not args:
        return []
    output = []

    if all([isinstance(a, Expr) for a in args[:nexpr]]):
        # In this case, with a single plot command, we are plotting either:
        #   1. one expression
        #   2. multiple expressions over the same range

        res = [not (_is_range(a) or isinstance(a, str)) for a in args]
        exprs = [a for a, b in zip(args, res) if b]
        ranges = [r for r in args[nexpr:] if _is_range(r)]
        label = args[-1] if isinstance(args[-1], str) else ""

        if not all([_is_range(r) for r in ranges]):
            raise ValueError(
                        "Expressions must be followed by ranges. Received:\n"
                        "Expressions: %s\n"
                        "Others: %s" % (exprs, ranges))
        free_symbols = set().union(*[e.free_symbols for e in exprs])
        ranges = _create_ranges(free_symbols, ranges, npar)

        if nexpr > 1:
            # in case of plot_parametric or plot3d_parametric_line, there will
            # be 2 or 3 expressions defining a curve. Group them together.
            if len(exprs) == nexpr:
                exprs = (tuple(exprs),)
        for expr in exprs:
            # need this if-else to deal with both plot/plot3d and
            # plot_parametric/plot3d_parametric_line
            e = (expr,) if isinstance(expr, Expr) else expr
            current_label = (label if label else
                    (str(expr) if isinstance(expr, Expr) else str(e)))
            output.append((*e, *ranges, current_label))


    else:
        # In this case, we are plotting multiple expressions, each one with its
        # range. Each "expression" to be plotted has the following form:
        # (expr, range, label) where label is optional

        # look for "global" range and label
        labels = [a for a in args if isinstance(a, str)]
        ranges = [a for a in args if _is_range(a)]
        n = len(ranges) + len(labels)
        new_args = args[:-n] if n > 0 else args

        # Each arg has the form (expr1, expr2, ..., range1 [optional], ...,
        #   label [optional])
        for arg in new_args:
            # look for "local" range and label. If there is not, use "global".
            l = [a for a in arg if isinstance(a, str)]
            if not l:
                l = labels
            r = [a for a in arg if _is_range(a)]
            if not r:
                r = ranges.copy()

            arg = arg[:nexpr]
            free_symbols = set().union(*[a.free_symbols for a in arg])
            if len(r) != npar:
                r = _create_ranges(free_symbols, r, npar)
            label = (str(arg[0]) if nexpr == 1 else str(arg)) if not l else l[0]
            output.append((*arg, *r, label))
        return output

    return output

def _plot_sympify(args):
    """ By allowing the users to set custom labels to the expressions being
    plotted, a critical issue is raised: whenever a special character like $,
    {, }, ... is used in the label (type string), sympify will raise an error.
    This function recursively loop over the arguments passed to the plot
    functions: the sympify function will be applied to all arguments except
    those of type string.
    """
    if isinstance(args, Expr):
        return args

    args = list(args)
    for i, a in enumerate(args):
        if isinstance(a, (list, tuple)):
            args[i] = Tuple(*_plot_sympify(a), sympify=False)
        elif not isinstance(a, str):
            args[i] = sympify(a)
    if isinstance(args, tuple):
        return Tuple(*args, sympify=False)
    return args

def _remove_label_key(d):
    """The "label" keyword is no longer needed in the plot, as it will clash
    with the label provided by the user (or the one automatically generated by
    _check_arguments()). By removing this keyword, the __init__ method of
    the *Series classes will only receive one "label", avoiding the following
    error:
    TypeError: __init__() got multiple values for argument 'label'
    """
    if "label" in d.keys():
        d.pop("label")
    return d
