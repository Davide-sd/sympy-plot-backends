from collections.abc import Callable
from sympy import latex
from sympy.external import import_module
from spb.backends.base_backend import Plot
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import mpl_toolkits
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from mergedeep import merge
import itertools

"""
TODO:
    1. Besides the axis on the center of the image, there are also a couple of
        axis with ticks on the bottom/right sides. Delete them?
"""

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


class MatplotlibBackend(Plot):
    """ A backend for plotting SymPy's symbolic expressions using Matplotlib.

    Keyword Arguments
    =================

        aspect : str or tuple
            Default to "auto". Possible values:
                * "equal": sets equal spacing on the axis of a 2D plot.
                * tuple containing 2 float numbers
        
        axis_center : str or None
            Set the location of the intersection between the horizontal and
            vertical axis (only works for 2D plots). Possible values:
                "center": center of the current plot area.
                "auto": an algorithm will choose an appropriate value.
                tuple of two float numbers to specify the interested center.
                None: standard Matplotlib layout with vertical axis on the left,
                    horizontal axis on the bottom and a grid.
            Default to "auto".
        
        contour_kw : dict
            A dictionary of keywords/values which is passed to Matplotlib's
            contourf function to customize the appearance.
            Refer to the following web page to learn more about customization:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
        
        line_kw : dict
            A dictionary of keywords/values which is passed to Matplotlib's
            plot functions to customize the appearance of the lines.
            Refer to the following web pages to learn more about customization:
                If the plot is using solid colors:
                https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
                If the plot is using color maps:
                https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection
        
        quiver_kw : dict
            A dictionary of keywords/values which is passed to Matplotlib's
            quivers function to customize the appearance.
            Refer to this documentation page:
            https://matplotlib.org/stable/api/quiver_api.html#module-matplotlib.quiver

        surface_kw : dict
            A dictionary of keywords/values which is passed to Matplotlib's
            surface function to customize the appearance.
            Refer to this documentation page:
            https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface

        stream_kw : dict
            A dictionary of keywords/values which is passed to Matplotlib's
            streamplot function to customize the appearance.
            For 2D vector fields, refer to this documentation page:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.streamplot.html#matplotlib.axes.Axes.streamplot
        
        use_cm : boolean
            If True, apply a color map to the mesh/surface or parametric lines. 
            If False, solid colors will be used instead. Default to True.
    """

    colormaps = [
        cm.viridis, cm.autumn, cm.winter, cm.plasma, cm.jet,
        cm.gnuplot, cm.brg, cm.coolwarm, cm.cool, cm.summer
    ]

    def __new__(cls, *args, **kwargs):
        # Since Plot has its __new__ method, this will prevent infinite
        # recursion
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        # set global options like title, axis labels, ...
        super().__init__(*args, **kwargs)

        # see self._add_handle for more info about the following dictionary
        self._handles = dict()
    
    def _init_cyclers(self):
        if isinstance(self.colorloop, ListedColormap):
            self._cl = itertools.cycle(self.colorloop.colors)
        else:
            self._cl = itertools.cycle(self.colorloop)
        self._cm = itertools.cycle(self.colormaps)
    
    def _create_figure(self):
        # the following import is here in order to avoid a circular import error
        from spb.defaults import cfg
        use_jupyterthemes = cfg["matplotlib"]["use_jupyterthemes"]
        mpl_jupytertheme = cfg["matplotlib"]["jupytertheme"]
        if (self._get_mode() == 0) and use_jupyterthemes:
            # set matplotlib style to match the used Jupyter theme
            try:
                from jupyterthemes import jtplot
                jtplot.style(mpl_jupytertheme)
            except:
                pass

        is_3Dvector = any([s.is_3Dvector for s in self.series])
        aspect = self.aspect
        if aspect != 'auto':
            if aspect == "equal" and is_3Dvector:
                # vector_plot uses an aspect="equal" by default. In that case
                # we would get:
                # NotImplementedError: Axes3D currently only supports the aspect
                # argument 'auto'. You passed in 1.0.
                # This fixes it
                aspect = "auto"
            elif aspect == "equal":
                aspect = 1.0
            else:
                aspect = float(aspect[1]) / aspect[0]

        self._fig = plt.figure(figsize=self.size)
        is_3D = [s.is_3D for s in self.series]
        if any(is_3D) and (not all(is_3D)):
            raise ValueError('The matplotlib backend can not mix 2D and 3D.')

        kwargs = dict(aspect=aspect)
        if all(is_3D):
            kwargs["projection"] = "3d"
        self.ax = self._fig.add_subplot(1, 1, 1, **kwargs)
            
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

    def _add_colorbar(self, c, label, override=False):
        """ Add a colorbar for the specificied collection

        Keyword Aurguments
        ==================
            override : boolean
                For parametric plots the colorbar acts like a legend. Hence,
                when legend=False we don't display the colorbar. However,
                for contour plots the colorbar is essential to understand it.
                Hence, to show it we set override=True.
                Default to False.
        """
        # design choice: instead of showing a legend entry (which
        # would require to work with proxy artists and custom
        # classes in order to create a gradient line), just show a
        # colorbar with the name of the expression on the side.
        if (self.legend and self._use_cm) or override:
            # TODO: colorbar position? used space?
            cb = self._fig.colorbar(c, ax=self.ax)
            cb.set_label(label, rotation=90)
    
    def _add_handle(self, i, h, kw=None):
        """ self._handle is a dictionary where:
            key: integer corresponding to the i-th series.
            value: a list of two elements:
                1. handle of the object created by Matplotlib commands
                2. optionally, keyword arguments used to create the handle. 
                    Some object can't be updated, hence we need to reconstruct
                    it from scratch at every update.
        This dictionary will be used with iplot
        """
        self._handles[i] = [h if not isinstance(h, (list, tuple)) else h[0], kw]

    def _process_series(self, series):
        # XXX Workaround for matplotlib issue
        # https://github.com/matplotlib/matplotlib/issues/17130
        xlims, ylims, zlims = [], [], []

        self.ax.cla()
        self._init_cyclers()

        for i, s in enumerate(series):
            if s.is_2Dline:
                line_kw = self._kwargs.get("line_kw", dict())
                if s.is_parametric and self._use_cm:
                    x, y, param = s.get_data()
                    lkw = dict(array=param, cmap=next(self._cm))
                    segments = self.get_segments(x, y)
                    c = LineCollection(segments,
                            **merge({}, lkw, line_kw))
                    self.ax.add_collection(c)
                    self._add_colorbar(c, s.label)
                    self._add_handle(i, c)
                else:
                    if s.is_parametric:
                        x, y, param = s.get_data()
                    else:
                        x, y = s.get_data()
                    lkw = dict(label=s.label)
                    l = self.ax.plot(x, y, **merge({}, lkw, line_kw))
                    self._add_handle(i, l)
            elif s.is_contour:
                ckw = dict(cmap = next(self._cm))
                contour_kw = self._kwargs.get("contour_kw", dict())
                c = self.ax.contourf(*s.get_meshes(), 
                        **merge({}, ckw, contour_kw))
                self._add_colorbar(c, s.label, True)
                self._add_handle(i, c)
            elif s.is_3Dline:
                x, y, z, param = s.get_data()
                lkw = dict()
                line_kw = self._kwargs.get("line_kw", dict())
                if self._use_cm:
                    segments = self.get_segments(x, y, z)
                    lkw["cmap"] = next(self._cm)
                    lkw["array"] = param
                    c = Line3DCollection(segments, **merge({}, lkw, line_kw))
                    self.ax.add_collection(c)
                    self._add_colorbar(c, s.label)
                    self._add_handle(i, c)
                else:
                    lkw["label"] = s.label
                    l = self.ax.plot(x, y, z, **merge({}, lkw, line_kw))
                    self._add_handle(i, l)

                xlims.append((np.amin(x), np.amax(x)))
                ylims.append((np.amin(y), np.amax(y)))
                zlims.append((np.amin(z), np.amax(z)))
            elif s.is_3Dsurface:
                x, y, z = s.get_data()
                skw = dict(rstride = 1, cstride = 1, linewidth = 0.1)
                if self._use_cm:
                    skw["cmap"] = next(self._cm)
                surface_kw = self._kwargs.get("surface_kw", dict())
                kw = merge({}, skw, surface_kw)
                c = self.ax.plot_surface(x, y, z, **kw)
                self._add_colorbar(c, s.label)
                self._add_handle(i, c, kw)

                xlims.append((np.amin(x), np.amax(x)))
                ylims.append((np.amin(y), np.amax(y)))
                zlims.append((np.amin(z), np.amax(z)))
            elif s.is_implicit:
                points = s.get_data()
                if len(points) == 2:
                    # interval math plotting
                    x, y = _matplotlib_list(points[0])
                    c = self.ax.fill(x, y, edgecolor='None')
                    self._add_handle(i, c)
                else:
                    # use contourf or contour depending on whether it is
                    # an inequality or equality.
                    # XXX: ``contour`` plots multiple lines. Should be fixed.
                    colormap = ListedColormap(["#FFFFFF00", next(self._cl)])
                    xarray, yarray, zarray, plot_type = points
                    ckw = dict(cmap = colormap)
                    contour_kw = self._kwargs.get("contour_kw", dict())
                    if plot_type == 'contour':
                        c = self.ax.contour(xarray, yarray, zarray, 
                            **merge({}, ckw, contour_kw))
                    else:
                        c = self.ax.contourf(xarray, yarray, zarray,
                            **merge({}, ckw, contour_kw))
                    self._add_handle(i, c)
            elif s.is_vector:
                if s.is_2Dvector:
                    xx, yy, uu, vv = s.get_data()
                    magn = np.sqrt(uu**2 + vv**2)
                    streamlines = self._kwargs.get("streamlines", False)
                    if streamlines:
                        skw = dict()
                        stream_kw = self._kwargs.get("stream_kw", dict())
                        if self._use_cm:
                            skw["cmap"] = next(self._cm)
                            skw["color"] = magn
                            kw = merge({}, skw, stream_kw)
                            s = self.ax.streamplot(xx, yy, uu, vv,
                                **kw)
                        else:
                            skw["color"] = next(self._cl)
                            kw = merge({}, skw, stream_kw)
                            s = self.ax.streamplot(xx, yy, uu, vv,
                                **kw)
                        self._add_handle(i, s, kw)
                    else:
                        qkw = dict()
                        quiver_kw = self._kwargs.get("quiver_kw", dict())
                        if self._use_cm:
                            qkw["cmap"] = next(self._cm)
                            q = self.ax.quiver(xx, yy, uu, vv, magn,
                                **merge({}, qkw, quiver_kw))
                            self._add_colorbar(q, s.label)
                        else:
                            qkw["color"] = next(self._cl)
                            q = self.ax.quiver(xx, yy, uu, vv,
                                **merge({}, qkw, quiver_kw))
                        self._add_handle(i, q)
                else:
                    xx, yy, zz, uu, vv, ww = s.get_data()
                    magn = np.sqrt(uu**2 + vv**2 + ww**2)
                    streamlines = self._kwargs.get("streamlines", False)
                    if streamlines:
                        raise NotImplementedError(
                            "Matplotlib currently doesn't expose any function " +
                            "to create streamlines in 3D."
                        )
                    else:
                        qkw = dict()
                        quiver_kw = self._kwargs.get("quiver_kw", dict())
                        if self._use_cm:
                            qkw["cmap"] = next(self._cm)
                            qkw["array"] = magn.flatten()
                            kw = merge({}, qkw, quiver_kw)
                            q = self.ax.quiver(xx, yy, zz, uu, vv, ww,
                                **kw)
                            self._add_colorbar(q, s.label)
                        else:
                            qkw["color"] = next(self._cl)
                            kw = merge({}, qkw, quiver_kw)
                            q = self.ax.quiver(xx, yy, zz, uu, vv, ww,
                                **kw)
                        self._add_handle(i, q, kw)
                    xlims.append((np.amin(xx), np.amax(xx)))
                    ylims.append((np.amin(yy), np.amax(yy)))
                    zlims.append((np.amin(zz), np.amax(zz)))
            else:
                raise NotImplementedError(
                    "{} is not supported by {}\n".format(type(s), type(self).__name__)
                )

        Axes3D = mpl_toolkits.mplot3d.Axes3D

        # Set global options.
        # TODO The 3D stuff
        # XXX The order of those is important.
        if self.xscale and not isinstance(self.ax, Axes3D):
            self.ax.set_xscale(self.xscale)
        if self.yscale and not isinstance(self.ax, Axes3D):
            self.ax.set_yscale(self.yscale)
        if self.axis_center:
            val = self.axis_center
            if isinstance(self.ax, Axes3D):
                pass
            elif val == 'center':
                self.ax.spines['left'].set_position('center')
                self.ax.spines['bottom'].set_position('center')
                self.ax.yaxis.set_ticks_position('left')
                self.ax.xaxis.set_ticks_position('bottom')
                self.ax.spines['right'].set_visible(False)
                self.ax.spines['top'].set_visible(False)
            elif val == 'auto':
                xl, xh = self.ax.get_xlim()
                yl, yh = self.ax.get_ylim()
                pos_left = ('data', 0) if xl*xh <= 0 else 'center'
                pos_bottom = ('data', 0) if yl*yh <= 0 else 'center'
                self.ax.spines['left'].set_position(pos_left)
                self.ax.spines['bottom'].set_position(pos_bottom)
                self.ax.yaxis.set_ticks_position('left')
                self.ax.xaxis.set_ticks_position('bottom')
                self.ax.spines['right'].set_visible(False)
                self.ax.spines['top'].set_visible(False)
            else:
                self.ax.spines['left'].set_position(('data', val[0]))
                self.ax.spines['bottom'].set_position(('data', val[1]))
                self.ax.yaxis.set_ticks_position('left')
                self.ax.xaxis.set_ticks_position('bottom')
                self.ax.spines['right'].set_visible(False)
                self.ax.spines['top'].set_visible(False)
        else:
            self.ax.grid()
        if self.legend:
            handles, _ = self.ax.get_legend_handles_labels()
            # Show the legend only if there are legend entries. For example,
            # if we are plotting only parametric expressions, there will be 
            # only colorbars, no legend entries.
            if len(handles) > 0:
                self.ax.legend(loc="best")
        if self.title:
            self.ax.set_title(self.title)
        if self.xlabel:
            self.ax.set_xlabel(self.xlabel,
                position=(1, 0) if self.axis_center else (0.5, 0))
        if self.ylabel:
            self.ax.set_ylabel(self.ylabel,
                position=(0, 1) if self.axis_center else (0, 0.5))
        if isinstance(self.ax, Axes3D) and self.zlabel:
            self.ax.set_zlabel(self.zlabel, position=(0, 1))

        self._set_lims(xlims, ylims, zlims)
    
    def _set_lims(self, xlims, ylims, zlims):
        Axes3D = mpl_toolkits.mplot3d.Axes3D
        if not isinstance(self.ax, Axes3D):
            self.ax.autoscale_view(
                scalex=self.ax.get_autoscalex_on(),
                scaley=self.ax.get_autoscaley_on())
        else:
            # XXX Workaround for matplotlib issue
            # https://github.com/matplotlib/matplotlib/issues/17130
            if xlims:
                xlims = np.array(xlims)
                xlim = (np.amin(xlims[:, 0]), np.amax(xlims[:, 1]))
                self.ax.set_xlim(xlim)
            else:
                self.ax.set_xlim([0, 1])

            if ylims:
                ylims = np.array(ylims)
                ylim = (np.amin(ylims[:, 0]), np.amax(ylims[:, 1]))
                self.ax.set_ylim(ylim)
            else:
                self.ax.set_ylim([0, 1])

            if zlims:
                zlims = np.array(zlims)
                zlim = (np.amin(zlims[:, 0]), np.amax(zlims[:, 1]))
                self.ax.set_zlim(zlim)
            else:
                self.ax.set_zlim([0, 1])
            
        # xlim and ylim should always be set at last so that plot limits
        # doesn't get altered during the process.
        if self.xlim:
            self.ax.set_xlim(self.xlim)
        if self.ylim:
            self.ax.set_ylim(self.ylim)
        if self.zlim:
            self.ax.set_zlim(self.zlim)

    def _update_interactive(self, params):
        # With this backend, data is only being added once the plot is shown.
        # However, iplot doesn't call the show method. The following line of
        # code will add the numerical data (if not already present).
        if len(self._handles) == 0:
            self.process_series()

        xlims, ylims, zlims = [], [], []
        for i, s in enumerate(self.series):
            if s.is_interactive:
                self.series[i].update_data(params)
                if s.is_2Dline:
                    x, y = self.series[i].get_data()
                    if isinstance(self._handles[i][0], LineCollection):
                        segments = self.get_segments(x, y)
                        self._handles[i][0].set_segments(segments)
                    else:
                        self._handles[i][0].set_data(x, y)
                elif s.is_3Dline:
                    x, y, z = self.series[i].get_data()
                    if isinstance(self._handles[i][0], Line3DCollection):
                        segments = self.get_segments(x, y, z)
                        self._handles[i][0].set_segments(segments)
                    else:
                        self._handles[i][0].set_data_3d(x, y, z)
                    xlims.append((np.amin(x), np.amax(x)))
                    ylims.append((np.amin(y), np.amax(y)))
                    zlims.append((np.amin(z), np.amax(z)))
                elif s.is_3Dsurface:
                    x, y, z = self.series[i].get_data()
                    # NOTE: there isn't a straightforward way of updating data
                    # for a surface plot. Easiest way: delete old, create new.
                    # TODO: by setting the keyword arguments, somehow the update
                    # becomes really really slow.
                    self._handles[i][0].remove()
                    self._handles[i][0] = self.ax.plot_surface(x, y, z,
                            **self._handles[i][1])
                    xlims.append((np.amin(x), np.amax(x)))
                    ylims.append((np.amin(y), np.amax(y)))
                    zlims.append((np.amin(z), np.amax(z)))
                elif s.is_vector and s.is_3D:
                    streamlines = self._kwargs.get("streamlines", False)
                    if streamlines:
                        raise NotImplementedError
                    xx, yy, zz, uu, vv, ww = self.series[i].get_data()
                    # NOTE: there isn't a straightforward way of updating data
                    # for a surface plot. Easiest way: delete old, create new.
                    self._handles[i][0].remove()
                    self._handles[i][0] = self.ax.quiver(xx, yy, zz, uu, vv, ww,
                            **self._handles[i][1])
                    xlims.append((np.amin(xx), np.amax(xx)))
                    ylims.append((np.amin(yy), np.amax(yy)))
                    zlims.append((np.amin(zz), np.amax(zz)))
                elif s.is_vector:
                    xx, yy, uu, vv = self.series[i].get_data()
                    magn = np.sqrt(uu**2 + vv**2)
                    streamlines = self._kwargs.get("streamlines", False)
                    if streamlines:
                        raise NotImplementedError
                        
                        # # TODO: there is no remove() for StreamPlotSet. Is it
                        # # possible to implement a workaround?
                        # kw = self._handles[i][1]
                        # if self._use_cm:
                        #     kw["color"] = magn
                        # self._handles[i][0].remove()
                        # self._handles[i][0] = self.ax.streamplot(xx, yy, uu, vv,
                        #         **kw)
                    else:
                        # TODO: is the colormap scaling as well?
                        self._handles[i][0].set_UVC(uu, vv, magn)
        
        # Update the plot limits according to the new data
        Axes3D = mpl_toolkits.mplot3d.Axes3D
        if not isinstance(self.ax, Axes3D):
            # https://stackoverflow.com/questions/10984085/automatically-rescale-ylim-and-xlim-in-matplotlib
            # recompute the ax.dataLim
            self.ax.relim()
            # update ax.viewLim using the new dataLim
            self.ax.autoscale_view()
        else:
            pass
        
        self._set_lims(xlims, ylims, zlims)

    def process_series(self):
        """
        Iterates over every ``Plot`` object and further calls
        _process_series()
        """
        # create the figure from scratch every time, otherwise if the plot was
        # previously shown, it would not be possible to show it again. This
        # behaviour is specific to Matplotlib
        self._create_figure()
        self._process_series(self.series)

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
        plt.close(self._fig)

MB = MatplotlibBackend