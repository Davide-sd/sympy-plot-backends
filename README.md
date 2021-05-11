# Sympy Plotting Backends

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Davide-sd/sympy-plot-backends/HEAD)

This module contains a few plotting backends that can be used with [SymPy](github.com/sympy/sympy) as an alternative to the default Matplotlib backend. A backend represents the plotting library: it provides the necessary functionalities to quickly and easily plot the most common types of symbolic expressions (line plots, surface plots, parametric plots).

<div>
<img src="imgs/iplot_bokeh.png" width=250/>
<img src="imgs/plotly-vectors.png" width=250/>
<img src="imgs/k3d-1.png" width=250/>
<img src="imgs/k3d-2.png" width=250/>
<img src="imgs/plotly-2d.png" width=250/>
<img src="imgs/mayavi.png" width=250/>
</div>


## What's new in comparison to SymPy

On top of the usual plotting functions exposed by SymPy (`plot`, `plot_parametric`, `plot3d`, etc.), this module offers:
* capability to use a different backend.
* `iplot` function to create parametric-interactive plots using widgets (sliders, buttons, etc.).
* `get_plot_data` function to easily extract the numerical data from symbolic expressions, which can later be used to create custom plots with our plotting library of choice.


## Backends

The 2 most important reasons for using a different backend are:
1. **Better interactive** experience (explored in the tutorial notebooks).
2. To use the **plotting library we are most comfortable with**. The backend can be used as a starting point to plot symbolic expressions; then, we could use the figure object to add numerical (or experimental) results using the commands associated to the specific plotting library.

The following plotting libraries are supported: [Matplolib](https://matplotlib.org/), [Plotly](https://plotly.com/), [Bokeh](https://github.com/bokeh/bokeh), [Mayavi](https://github.com/enthought/mayavi), [K3D-Jupyter](https://github.com/K3D-tools/K3D-jupyter)

Each backend has its own advantages and disadvantages, as we can see from the following table:

|                        | Matplolib | Bokeh | Plotly | Mayavi | K3D |
|:----------------------:|:---------:|:-----:|:------:|:------:|:---:|
|           2D           |     Y     |   Y   |    Y   |    N   |  N  |
|           3D           |     Y     |   N   |    Y   |    Y   |  Y  |
|        Implicit        |     Y     |   N   |    Y   |    N   |  N  |
|        PlotGrid        |     Y     |   N   |    N   |    N   |  N  |
|      Latex Support     |     Y     |   N   |    Y   |    N   |  Y  |
|      Save Picture      |     Y     |   Y   |    Y   |    Y   |  Y  |
|       Jupyter NB       |     Y     |   Y   |    Y   |    Y   |  Y  |
|   Python Interpreter   |     Y     |   Y   |    Y   |    Y   |  N  |
| Parametric-Interactive |     N     |   Y   |    Y   |    N   |  Y  |

In particular:
* Matplotlib (default with SymPy) is good but it lacks interactivity (of course, we can use [ipympl](https://github.com/matplotlib/ipympl), but it doesn't do miracles especially with 3D plots).
* Matplotlib and Plotly are the two most general backends, both supporting 2D and 3D plots.
* Mayavi and K3D only supports 3D plots but, compared to Matplotlib,they are blazingly fast in the user-interaction. Hence, we can increase significantly the number of discretization points obtaining smoother plots. Note that these backends use an aspect ratio of 1 on all axis: they don't scale the visualization. What you see is the object as you would see it in reality.
* K3D can only be used with Jupyter Notebook, whereas the other backends can also be used with IPython or a simple Python interpreter.
* Plotly and Bokeh require external libraries in order to export plots to png or svg. Read the docstrings of the respective classes to understand what you need to install.
* **Parametric-Interactive**: thanks to [holovis'z panel](https://github.com/holoviz/panel), we can use widgets (sliders, buttons, ...) to visually explore the symbolic expression. This functionality only works within Jupyter Notebook. Currently, `MatplotlibBackend` doesn't support this feature.

The following table shows the common keyword arguments implemented in SymPy's `Plot` class, which is the parent class for all backends. Because each plotting library is unique, some of these options may not be supported by a specific backend (or have not been implemented yet):

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
| aspect_ratio  |     Y     |   Y   |    Y   |    N   |  N  |
|   autoscale   |     Y     |   N   |    N   |    N   |  N  |
|    margin     |     Y     |   N   |    N   |    N   |  N  |
|     size      |     Y     |   Y   |    Y   |    Y   |  Y  |
|     title     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    xlabel     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    ylabel     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    zlabel     |     Y     |   N   |    Y   |    Y   |  Y  |

For example, while SymPy's default backend (Matplotlib) is implemented to mimic hand-plotted 2D charts, that is the horizontal and vertical axis are not necessarely fixed to the bottom-side and left-side of the plot respectively (we can specify their location with `axis_center`), I didn't implement this feature on Bokeh and Plotly because it doesn't add any value to my personal use. If you find that some options could be implemented, please consider contributing with a PR.

Other options are only available to a specific backend, for example:

|  keyword arg  | Matplolib | Bokeh | Plotly | Mayavi | K3D |
|:-------------:|:---------:|:-----:|:------:|:------:|:---:|
|  line_color   |     Y     |   N   |    N   |    N   |  N  |
| surface_color |     Y     |   N   |    N   |    N   |  N  |
|     theme     |     N     |   Y   |    Y   |    N   |  N  |
|   wireframe   |     N     |   N   |    Y   |    Y   |  Y  |
|   bg_color    |     N     |   N   |    N   |    Y   |  Y  |
|   fg_color    |     N     |   N   |    N   |    Y   |  N  |
|  grid_color   |     N     |   N   |    N   |    N   |  Y  |
|    use_cm     |     N     |   N   |    Y   |    Y   |  Y  |
|   show_label  |     N     |   N   |    N   |    N   |  Y  |

Please, read the docstring associated to each backend to find out what they do.


## Warnings

While this module is based on `sympy.plotting`, there are some differences (structural and usability) that make them incompatible. Interchanging between these two modules might lead to some errors! I suggest to use this module instead of `sympy.plotting`. On the usability side, the main differences are:
1. `label` keyword argument has been removed.
2. `nb_of_points_*` keyword arguments have been removed.

Read the tutorials to find out the alternatives to those keyword arguments.

Currently, this module must be considered in beta phase. If you find any bug, please open an issue. If you feel like some feature could be implemented, open an issue or create a PR.

Finally, **some backend comes with a memory cost**. Since they require external libraries and/or open a server-process in order to visualize the data, memory usage can quickly rise if we are showing many plots. Keep an eye on you system monitor and act accordingly (close the kernels, restart the browser, etc.). 


## Requirements

The following list of requirements will automatically be downloaded once you install the module:

`numpy, sympy, matplotlib, plotly>=4.14.3, colorcet, param, panel, holoviews, bokeh, PyQt5, mayavi, k3d`


## Installation

Click the _launch binder_ icon below to interactively explore the notebooks in the tutorial folder.



1. Download this repository: `git clone https://github.com/Davide-sd/sympy-plot-backends`
2. Move into the module folder, `sympy_plot_backends` and install the module with the command `pip install -e .`.
    * Note that we installed it in _editable_ mode. This will allows us to customize it and try different default options.
    * At the moment Mayavi won't be automatically installed. [You need to do that youself by following these steps](https://github.com/enthought/mayavi#installation).
3. Launch Jupyter Notebook and explore the `tutorials` folder to see what this module offers. Alternatively, click the following button to explore it online (in this case Mayavi won't work, therefore comment out the lines of code referring to Mayavi!). [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Davide-sd/sympy-plot-backends/HEAD)
4. **Optional step**: open the file `spb/defaults.py` and change the default backends and related options. [Tutorial 3](tutorials/tutorial-3.customize-the-module.ipynb) helps you with that.


## Usage

```
from sympy import *
from spb import plot3d_parametric_surface as p3ds
var("u, v")

p3ds(v * cos(u), v * sin(u), v + sin(3 * v) / 3 - 4, (u, 0, 2 * pi), (v, 0, 2 * pi))
```
<div align="center">
<img src="imgs/k3d-3.png" width=250/>
</div>

Look at the notebooks in the [tutorials](\tutorials) folder to fully understand what this module offers.
