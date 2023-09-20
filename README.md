# Sympy Plotting Backends

[![PyPI version](https://badge.fury.io/py/sympy-plot-backends.svg)](https://badge.fury.io/py/sympy-plot-backends)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/sympy_plot_backends.svg)](https://anaconda.org/conda-forge/sympy_plot_backends)
[![Documentation Status](https://readthedocs.org/projects/sympy-plot-backends/badge/?version=latest)](http://sympy-plot-backends.readthedocs.io/)
![Coverage](https://github.com/Davide-sd/sympy-plot-backends/blob/master/coverage.svg)


This module contains a few plotting backends that can be used with [SymPy](github.com/sympy/sympy) and [Numpy](https://github.com/numpy/numpy). A backend represents the plotting library: it provides the necessary functionalities to quickly and easily plot the most common types of symbolic expressions (line plots, surface plots, parametric plots, vector plots, complex plots, control system plots).

The following plotting libraries are supported: [Matplolib](https://matplotlib.org/), [Plotly](https://plotly.com/), [Bokeh](https://github.com/bokeh/bokeh), [K3D-Jupyter](https://github.com/K3D-tools/K3D-jupyter), [Mayavi](https://github.com/enthought/mayavi).

<div>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/iplot_bokeh.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/mpl-streamplot.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/plotly_streamlines_2.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/K3D-spherical-harmonics.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/bokeh_domain_coloring.png" width=250/>
<img src="https://raw.githubusercontent.com/Davide-sd/sympy-plot-backends/master/imgs/K3D-cone-vectors.png" width=250/>
</div>


## What's new in comparison to SymPy

On top of the usual plotting functions exposed by SymPy (`plot`,
`plot_parametric`, `plot3d`, etc.), this module offers the capabily to:

* use a different plotting library.
* visualize discontinuities on 2D line plots.
* visualize 2D/3D vector fields with quivers or streamlines.
* visualize complex functions with [domain coloring](https://en.wikipedia.org/wiki/Domain_coloring).
* visualize entities from the `sympy.geometry` module.
* visualize control systems' response to input signals, visualize Bode, Nyquist and Nichols plots.
* create parametric-interactive plots using widgets
  (sliders, buttons, etc.) with *ipywidgets* or *Holoviz's Panel*.
* combine multiple plots into a grid-like layout.

Please, read the
[following documentation page](https://sympy-plot-backends.readthedocs.io/en/latest/overview.html#differences-with-sympy-plotting)
to understand the differences between this module and ``sympy.plotting``.

If you feel like some feature could be implemented, open an issue or create
a PR.


## Explore the Capabilities

[Read the documentation](https://sympy-plot-backends.readthedocs.io/) to 
explore the capabilities before the installation, which contains plenty of
examples.


## Installation

*SymPy Plotting Backends* can be installed with `pip` or `conda`. By default,
only basic plotting with Matplotlib will be installed:

```
pip install sympy_plot_backends
```

Or

```
conda install -c conda-forge sympy_plot_backends
```

To install the complete requirements in order to get interactive plots, bokeh,
plotly, k3d, vtk, execute the following command:

```
pip install sympy_plot_backends[all]
```

Or:

```
conda install -c anaconda scipy notebook colorcet
conda install -c conda-forge adaptive
conda install -c conda-forge panel
conda install -c anaconda ipywidgets
conda install -c conda-forge ipympl
conda install -c bokeh ipywidgets_bokeh
conda install -c conda-forge k3d msgpack-python
conda install -c plotly plotly
conda install -c conda-forge vtk
```

Finally, if you are using zshell, the above `pip` command is going to fail.
Use the following instead:

```
pip install "sympy_plot_backends[all]"
```
  

## Warnings

**Some backend comes with a memory cost**. Since they require external libraries and/or open a server-process in order to visualize the data, memory usage can quickly rise if we are showing many plots. Keep an eye on you system monitor and act accordingly (close the kernels, restart the browser, etc.).
