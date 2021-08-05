# Sympy Plotting Backends

[![PyPI version](https://badge.fury.io/py/sympy-plot-backends.svg)](https://badge.fury.io/py/sympy-plot-backends)
[![Install with conda](https://anaconda.org/davide_sd/sympy_plot_backends/badges/installer/conda.svg)](https://anaconda.org/Davide_sd/sympy_plot_backends)
[![Documentation Status](https://readthedocs.org/projects/sympy-plot-backends/badge/?version=latest)](http://sympy-plot-backends.readthedocs.io/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Davide-sd/sympy-plot-backends/HEAD)


This module contains a few plotting backends that can be used with [SymPy](github.com/sympy/sympy) as an alternative to the default Matplotlib backend. A backend represents the plotting library: it provides the necessary functionalities to quickly and easily plot the most common types of symbolic expressions (line plots, surface plots, parametric plots).

<div>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/iplot_bokeh.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/plotly-vectors.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/plotly_streamlines.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/k3d-1.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/bokeh_domain_coloring.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/k3d_domain_coloring.png" width=250/>
</div>


## What's new in comparison to SymPy

On top of the usual plotting functions exposed by SymPy (`plot`, `plot_parametric`, `plot3d`, etc.), this module offers:
* capability to use a different backend.
* capability to correctly visualize discontinuities on 2D line plots.
* capability to correctlt visualize 2D line plots of piecewise functions.
* `vector_plot` function to quickly visualize 2D/3D vector fields with quivers or streamlines.
* `complex_plot` function to quickly visualize 2D/3D complex functions. In particular, we can visualize the real and imaginary parts, the modulus and argument, 2D/3D domain coloring.
* `polar_plot` function.
* `geometry_plot` to quickly visualize entities from the `sympy.geometry` module.
* `iplot` function to create parametric-interactive plots using widgets (sliders, buttons, etc.).
* `get_plot_data` function to easily extract the numerical data from symbolic expressions, which can later be used to create custom plots with our plotting library of choice.
* `plotgrid` function, which replaces the `PlotGrid` class: it allows to combine multiple plots into a grid-like layout. It works with Matplotlib, Bokeh and Plotly.


## Backends

The 2 most important reasons for using a different backend are:
1. **Better interactive** experience (explored in the tutorial notebooks).
2. To use the **plotting library we are most comfortable with**. The backend can be used as a starting point to plot symbolic expressions; then, we could use the figure object to add numerical (or experimental) results using the commands associated to the specific plotting library.

The following plotting libraries are supported: [Matplolib](https://matplotlib.org/), [Plotly](https://plotly.com/), [Bokeh](https://github.com/bokeh/bokeh), [K3D-Jupyter](https://github.com/K3D-tools/K3D-jupyter)

## Explore the Capabilities

To explore the capabilities before the installation, you can:

1. [Read the documentation](https://sympy-plot-backends.readthedocs.io/) and follow [the tutorials](https://sympy-plot-backends.readthedocs.io/en/latest/tutorials/index.html).
2. Click the following button to run the tutorials with Binder. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Davide-sd/sympy-plot-backends/HEAD)

## Installation

The repository is avaliable on PyPi:

```
pip install sympy_plot_backends
```

And also on Conda:

```
 conda install -c davide_sd sympy_plot_backends 
```


## Warnings

While this module is based on `sympy.plotting`, there are some differences (structural and usability) that make them incompatible. Interchanging between these two modules might lead to some errors! I suggest to use this module instead of `sympy.plotting`. On the usability side, the main differences are:
1. `label` keyword argument has been removed.
2. `nb_of_points_*` keyword arguments have been replaced by `n` or `n1, n2`.
3. `ImplicitSeries` now uses mesh grid algorithm and contour plots by default. It is going to automatically switch to an adaptive algorithm if Boolean expressions are found.

Currently, this module must be considered in beta phase. If you find any bug, please open an issue. If you feel like some feature could be implemented, open an issue or create a PR.

Finally, **some backend comes with a memory cost**. Since they require external libraries and/or open a server-process in order to visualize the data, memory usage can quickly rise if we are showing many plots. Keep an eye on you system monitor and act accordingly (close the kernels, restart the browser, etc.). 


## Known Bugs

To implement this module, a great effort went into the integration of several libraries and maintaining a consistent user experience with the different backends.

However, there are a few known bugs inherited from dependency-libraries that we should be aware of:

* The `iplot` function uses [holovis'z panel](https://github.com/holoviz/panel). However, this introduces a bug: once we decide to use `BokehBackend`, we can't go back to other backends (the plots won't be visible in the output cell). The only way to fix this problem is to close Jupyter server and start again.
* The aforementioned bug also manifests itself while using `plotgrid` with different types of plots (`PlotlyBackend, BokehBackend`, etc.).
* While using the `iplot` function with `BokehBackend`:
  * the user-defined theme won't be applied.
  * rendering of gradient lines is slow.
  * color bars might not update their ranges.
* Plotly:
  * with 2D domain coloring, the vertical axis is reversed, with negative values on the top and positive values on the bottom.
  * with 3D complex plots: when we hover a point, the tooltip will display wrong information for the argument and the phase. Hopefully [this bug](https://github.com/plotly/plotly.js/issues/5003) will be fixed upstream.

