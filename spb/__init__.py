# NOTE: don't load anything, it will slow things down.
# Just let the users know how to import the different backends.
# The slowdown is cause by mayavi.

# from spb.bokeh import BB, BokehBackend
# from spb.plotly import PB, PlotlyBackend
# from spb.k3d import KB, K3DBackend
# from spb.mayavi import MB, MayaviBackend

from spb.functions import (
    plot, plot_parametric, plot_contour,
    plot3d, plot3d_parametric_line, plot3d_parametric_surface
)