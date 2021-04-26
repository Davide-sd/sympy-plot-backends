from collections.abc import Callable
from sympy.external import import_module
from spb.backends.plot import BaseBackend

# Global variable
# Set to False when running tests / doctests so that the plots don't show.
_show = True


def unset_show():
    """
    Disable show(). For use in the tests.
    """
    global _show
    _show = False

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
                ax.contourf(*s.get_meshes())
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