# Sympy Plotting Backends

This module contains a few backends that can be used with [SymPy](github.com/sympy/sympy) as an alternative to the default Matplotlib backend. A backend represents the plotting library being used: it provides the necessary functionalities for quickly and easily plot the most common types of symbolic expressions (line plots, surface plots, parametric plots).

The following plotting libraries are supported: [Plotly](https://plotly.com/), [Bokeh](https://github.com/bokeh/bokeh), [Mayavi](https://github.com/enthought/mayavi), [K3D-Jupyter](https://github.com/K3D-tools/K3D-jupyter)

Each backend has its own advantages and disadvantages, as we can see from the following table:

|               | Matplolib | Bokeh | Plotly | Mayavi | K3D |
|:-------------:|:---------:|:-----:|:------:|:------:|:---:|
|       2D      |     Y     |   Y   |    Y   |    N   |  N  |
|       3D      |     Y     |   N   |    Y   |    Y   |  Y  |
|   PlotGrid    |     Y     |   N   |    N   |    N   |  N  |
| Latex Support |     Y     |   N   |    Y   |    N   |  Y  |
| Save Picture  |     Y     |   Y   |    Y   |    Y   |  Y  |
|  Jupyter NB   |     Y     |   Y   |    Y   |    Y   |  Y  |
| Python Interp |     Y     |   Y   |    Y   |    Y   |  N  |

In particular:
* Matplotlib (default with SymPy) is good but it lacks interactivity (of course, we can use [ipympl](https://github.com/matplotlib/ipympl) but the overall interactive experience is still behind in comparison to the other backends).
* Matplotlib and Plotly are the two most general backend, both supporting 2D and 3D plots.
* Mayavi and K3D only supports 3D plots but, compared to Matplotlib,they are blazingly fast in the user-interaction. Hence, we can increase significantly the number of discretization points obtaining smoother plots. Note that these backends use an aspect ratio of 1 on all axis, meaning that they don't scale the visualization. What you see is the object as you would see it in reality.
* K3D can only be used with Jupyter Notebook, whereas the other backends can also be used with IPython or a simple Python interpreter.
* Plotly and Bokeh require external libraries in order to export plots to png or svg. Read the respective classes' docstring to understand what you need to install.

The following table shows the common keyword arguments implemented in SymPy's `Plot` class. Because each plotting library is unique, some of these options may not be supported by a specific backend:

|  keyword arg  | Matplolib | Bokeh | Plotly | Mayavi | K3D |
|:-------------:|:---------:|:-----:|:------:|:------:|:---:|
|     xlim      |     Y     |   Y   |    Y   |    N   |  N  |
|     ylim      |     Y     |   Y   |    Y   |    N   |  N  |
|     zlim      |     Y     |   N   |    Y   |    N   |  N  |
|    xscale     |     Y     |   Y   |    Y   |    N   |  N  |
|    yscale     |     Y     |   Y   |    Y   |    N   |  N  |
|    zscale     |     Y     |   N   |    Y   |    N   |  N  |
|     axis      |     Y     |   Y   |    Y   |    Y   |  Y  |
|  axis_center  |     Y     |   N   |    N   |    N   |  N  |
| aspect_ratio  |     Y     |   N   |    N   |    N   |  N  |
|   autoscale   |     Y     |   N   |    N   |    N   |  N  |
|    margin     |     Y     |   N   |    N   |    N   |  N  |
|     size      |     Y     |   Y   |    Y   |    Y   |  Y  |
|     title     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    xlabel     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    ylabel     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    zlabel     |     Y     |   N   |    Y   |    Y   |  Y  |

Note: while SymPy's default backend (Matplotlib) is implemented to mimic hand-plotted 2D charts, that is the horizontal and vertical axis are not necessarely fixed to the bottom-side and left-side of the plot, respectively (we can specify their location with `axis_center`), I didn't implement this feature on Bokeh and Plotly because it doesn't add any value to my personal use. If you find that some options could be implemented, please consider contributing with a PR.

Other options are only available to a specific backend, for example:

|  keyword arg  | Matplolib | Bokeh | Plotly | Mayavi | K3D |
|:-------------:|:---------:|:-----:|:------:|:------:|:---:|
|  line_color   |     Y     |   N   |    N   |    N   |  N  |
| surface_color |     Y     |   N   |    N   |    N   |  N  |
|     theme     |     N     |   Y   |    Y   |    N   |  N  |
|   wireframe   |     N     |   N   |    Y   |    N   |  N  |

Please, read the docstring associated to each backend to find out the additional available keyword arguments.

Finally, **these backends come with a memory cost**. Since many of them requires external libraries and/or open a server-process in order to visualize the data, memory usage can quickly rise if we are showing many plots. Keep an eye on you system monitor and act accordingly. 

## Requirements

* `numpy`,
* `sympy>=1.6.1`,
* `matplotlib`,
* `plotly>=4.14.3`,
* `bokeh`,
* `mayavi`,
* `PyQt5`,
* `k3d`

## Installation

1. Open a terminal and move into the module folder, `sympy_plot_backends`.
2. `pip3 install .`

## Usage

Look at the notebooks in the [examples](\examples) folder.
