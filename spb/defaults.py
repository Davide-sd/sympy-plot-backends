# Find more Plotly themes at the following page:
# https://plotly.com/python/templates/
plotly_theme = "plotly_dark"

# Find more Bokeh themes at the following page:
# https://docs.bokeh.org/en/latest/docs/reference/themes.html
bokeh_theme = "dark_minimal"

# K3D background color
k3d_bg_color = 0xFFFFFF

# Mayavi background and foreground colors
mayavi_bg_color = (0.22, 0.24, 0.29)
mayavi_fg_color = (1, 1, 1)

# backend for 2D plots
from spb.backends.plotly import PlotlyBackend as TWO_D_B
# backend for 3D plots
from spb.backends.k3d import K3DBackend as THREE_D_B