from base_backend import MyBaseBackend
from mayavi import mlab
from IPython.core.display import display

class MayaviBackend(MyBaseBackend):
    # More colormaps at:
    # https://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html
    colormaps = ['jet', 'autumn', 'Spectral', 'CMRmap', 'YlGnBu',
          'spring', 'summer', 'coolwarm', 'viridis', 'winter']
    
    def __init__(self, parent):
        super().__init__(parent)
        if self._get_mode() == 0:
            mlab.init_notebook()
        
        size = (600, 400)
        if parent.size:
            size = parent.size
        self.fig = mlab.figure(size=size, bgcolor=(0.22, 0.24, 0.29))
    
    def _process_series(self, series):
        mlab.clf(self.fig)

        cm = iter(self.colormaps)
        for i, s in enumerate(series):
            if s.is_3Dline:
                x, y, z = s.get_data()
                length = self._line_length(x, y, z, start=s.start, end=s.end)
                mlab.plot3d(x, y, z, length, figure=self.fig,
                                tube_radius=0.05, colormap=next(cm))
            elif s.is_3Dsurface:
                mlab.mesh(*s.get_data(), figure=self.fig, colormap=next(cm))
            else:
                raise ValueError(
                    "Mayavi only support 3D plots."
                )
            
            if self.parent.axis:
                mlab.axes(xlabel="", ylabel="", zlabel="",
                     x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True)
                mlab.outline()
        
        xl = self.parent.xlabel if self.parent.xlabel else "x"
        yl = self.parent.ylabel if self.parent.ylabel else "y"
        zl = self.parent.zlabel if self.parent.zlabel else "z"
        mlab.orientation_axes(xlabel=xl, ylabel=yl, zlabel=zl)
        if self.parent.title:
            mlab.title(self.parent.title, figure=self.fig, size=0.5)

    def show(self):
        self._process_series(self.parent._series)
        display(self.fig)
    
    def save(self, path):
        pass
    
    def close(self):
        mlab.close(self.fig)