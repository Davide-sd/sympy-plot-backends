from spb.defaults import mayavi_bg_color, mayavi_fg_color
from spb.backends.base_backend import Plot
from mayavi import mlab
from IPython.core.display import display

# TODO
# 1. Implement save feature

class MayaviBackend(Plot):
    """ A backend for plotting SymPy's symbolic expressions using Mayavi.

    Keyword Arguments
    =================

        bg_color : tuple
            A tuple of RGB values from 0 to 1 specifying the background color.
            Default to (0.22, 0.24, 0.29).
        
        fg_color : tuple
            A tuple of RGB values from 0 to 1 specifying the foreground color,
            that is the color of all text annotation labels (axes, orientation
            axes, scalar bar labels). It should be sufficiently far from 
            `bgcolor` to see the annotation texts.
            Default to (1, 1, 1), which represent white color.
        
        use_cm : boolean
            If True, apply a color map to the meshes/surface. If False, solid
            colors will be used instead. Default to True.

    """
    # More colormaps at:
    # https://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html
    colormaps = ['jet', 'autumn', 'Spectral', 'CMRmap', 'YlGnBu',
          'spring', 'summer', 'coolwarm', 'viridis', 'winter']
    
    def __new__(cls, *args, **kwargs):
        # Since Plot has its __new__ method, this will prevent infinite
        # recursion
        return object.__new__(cls)
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._get_mode() == 0:
            mlab.init_notebook()
        
        size = (600, 400)
        if self.size:
            size = self.size
        self._fig = mlab.figure(
            size = size,
            bgcolor = self._kwargs.get("bg_color", mayavi_bg_color),
            fgcolor = self._kwargs.get("fg_color", mayavi_fg_color)
        )
    
    def _process_series(self, series):
        mlab.clf(self._fig)

        cm = iter(self.colormaps)
        for i, s in enumerate(series):
            if s.is_3Dline:
                x, y, z = s.get_data()
                u = s.discretized_var
                mlab.plot3d(x, y, z, u,
                    color = None if self._use_cm else next(self._cl),
                    figure=self._fig,
                    tube_radius=0.05,
                    colormap=next(cm))
            elif s.is_3Dsurface:
                mlab.mesh(*s.get_data(),
                    color = None if self._use_cm else next(self._cl),
                    figure = self._fig,
                    colormap = next(cm),
                    representation = "wireframe" if self._kwargs.get("wireframe", False) else "surface"
                )
            else:
                raise ValueError(
                    "Mayavi only support 3D plots."
                )
            
            if self.axis:
                mlab.axes(xlabel="", ylabel="", zlabel="",
                     x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True)
                mlab.outline()
        
        xl = self.xlabel if self.xlabel else "x"
        yl = self.ylabel if self.ylabel else "y"
        zl = self.zlabel if self.zlabel else "z"
        mlab.orientation_axes(xlabel=xl, ylabel=yl, zlabel=zl)
        if self.title:
            mlab.title(self.title, figure=self._fig, size=0.5)

    def show(self):
        self._process_series(self._series)
        display(self._fig)
    
    def save(self, path, **kwargs):
        """ Save the current plot. Look at the following page to find out more
        keyword arguments to control the output file:
        https://docs.enthought.com/mayavi/mayavi/auto/mlab_figure.html#savefig


        Parameters
        ==========

            path : str
                File path with extension.
            
            kwargs : dict
                Optional backend-specific parameters.
        """
        mlab.savefig(
            path,
            figure = self._fig,
            size = kwargs.get("size", None),
            magnification = kwargs.get("magnification", "auto"),
        )
    
    def close(self):
        mlab.close(self._fig)

MB = MayaviBackend