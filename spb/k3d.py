from spb.base_backend import MyBaseBackend
import k3d
import numpy as np
import warnings
from matplotlib.tri import Triangulation

# TODO:
# 1. What does happen when a parametric surface is used in the following lines
#    of code?
#           xmin=s.start_x, xmax=s.end_x,
#           ymin=s.start_y, ymax=s.end_y,

class K3DBackend(MyBaseBackend):
    """ A backend for plotting SymPy's symbolic expressions using K3D-Jupyter.
    """

    colormaps = [
        k3d.basic_color_maps.CoolWarm, k3d.basic_color_maps.Jet,
        k3d.basic_color_maps.BlackBodyRadiation, k3d.matplotlib_color_maps.Plasma,
        k3d.matplotlib_color_maps.Autumn, k3d.matplotlib_color_maps.Winter,
        k3d.paraview_color_maps.Nic_Edge, k3d.paraview_color_maps.Haze
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._get_mode() != 0:
            raise ValueError(
                    "Sorry, K3D backend only works within Jupyter Notebook")
        self._fig = k3d.plot(grid_visible=self.axis)
        if (self.xscale == "log") or (self.yscale == "log"):
            warnings.warn("K3D-Jupyter doesn't support log scales. We will " +
                         "continue with linear scales.")
        self.plot_shown = False

    def _process_series(self, series):
        cm = iter(self.colormaps)

        # def asd(sx, ex, sy, ey):
        #     xlim = self.xlim
        #     ylim = self.ylim
        #     return {
        #         "xmin": sx if (xlim is None) else xlim[0],
        #         "xmax": ex if (xlim is None) else xlim[1],
        #         "ymin": sy if (ylim is None) else ylim[0],
        #         "ymax": ey if (ylim is None) else ylim[1]
        #     }
        
        for i, s in enumerate(series):
            if s.is_3Dline:
                x, y, z = s.get_data()
                vertices = np.dstack([x, y, z]).T.astype(np.float32)
                length = self._line_length(x, y, z, start=s.start, end=s.end)
                line = k3d.line(vertices, attribute=length,
                                width=0.1, color_map=next(cm),
                                color_range=[s.start, s.end], name=s.label)
                self._fig += line
            elif s.is_3Dsurface:
                x, y, z = s.get_data()
                if not s.is_parametric:
                    z = z.astype(np.float32)
                    surf = k3d.surface(z, attribute=z,
                                xmin=np.min(x[:]), xmax=np.max(x[:]),
                                ymin=np.min(y[:]), ymax=np.min(y[:]),
                                color_map=next(cm), name=s.label)
                else:
                    pass
                    # vertices = np.dstack([x, y, z]).astype(np.float32)
                    # indices = Triangulation(x,y).triangles.astype(np.uint32)
                    # surf = k3d.mesh(np.vstack([x,y,z]).T, indices,
                    #             color_map = k3d.colormaps.basic_color_maps.Jet,
                    #             attribute=z,
                    #             color_range = [-1.1,2.01]
                    #         )
                    
                self._fig += surf
            else:
                raise ValueError(
                    "K3D-Jupyter only support 3D plots."
                )
        
        xl = self.xlabel if self.xlabel else "x"
        yl = self.ylabel if self.ylabel else "y"
        zl = self.zlabel if self.zlabel else "z"
        self._fig.axes = [xl, yl, zl]

        if self.title:
            self._fig += k3d.text2d(self.title, 
                 position=[0.025, 0.015], color=0, size=1, label_box=False)
    
    def show(self):
        self._process_series(self._series)
        self.plot_shown = True
        self._fig.display()
    
    def save(self, path):
        if not self.plot_shown:
            raise ValueError(
                "K3D-Jupyter requires the plot to be shown on the screen " + 
                "before saving it."
            )

        self._process_series(self._series)

        @self._fig.yield_screenshots
        def _func():
            self._fig.fetch_screenshot()
            screenshot = yield
            with open(path, 'wb') as f:
                f.write(screenshot)
        _func()