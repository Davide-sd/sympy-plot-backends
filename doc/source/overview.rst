=========
 Overview
=========

In the following overview we are going to briefly introduce the functionalities
exposed by this module.

Plotting functions
==================

The following functions are exposed by this module:

* `plot`: visualize a function of a single variable.

   * capability to correctly visualize discontinuities on 2D line plots.

   * capability to correctlt visualize piecewise functions.

* `plot_parametric`: visualize a 2D parametric curve.
* `plot_polar`: visualize a curve of a given radius as a function of an angle.
* `plot3d`: visualize a function of two variables.
* `plot3d_parametric_line`: visualize a 3D parametric curve.
* `plot3d_parametric_surface`: visualize a 3D parametric surface.
* `vector_plot`: visualize 2D/3D vector fields with quivers or streamlines.
* `complex_plot`: visualize 2D/3D complex functions. In particular, we can
  visualize:

   * list of complex numbers.
   * function of 1 variable over a real range: visualize the real and imaginary
     parts, the modulus and its argument.
   * function of 2 variables over 2 real ranges: visualize the real and imaginary
     parts, the modulus and its argument.
   * complex function over a complex range:
      * domain coloring plot.
      * 3D plot of the real and imaginary part.
      * 3D plot of the modulus colored by the argument.

* `plot_geometry`: visualize entities from the `sympy.geometry` module.
* `iplot`: create parametric-interactive plots using widgets (sliders, buttons, 
  etc.).
* `get_plot_data`: easily extract the numerical data from symbolic expressions,
  which can later be used to create custom plots with any plotting library.
* `plotgrid`: combine multiple plots into a grid-like layout. It works with
  Matplotlib, Bokeh and Plotly.

Backends
========

This module allows the user to chose between 4 different backends (plotting
libraries): `Matplotlib <https://matplotlib.org/>`_, `Plotly <https://plotly.com/>`_,
`Bokeh <https://github.com/bokeh/bokeh>`_, `K3D-Jupyter <https://github.com/K3D-tools/K3D-jupyter>`_.

The 2 most important reasons for using a different backend are:

#. **Better interactive** experience (explored in the tutorial section), which
   translates to better data exploration and visualization (especially when
   working with Jupyter Notebook).

#. To use the **plotting library we are most comfortable with**. The backend
   can be used as a starting point to plot symbolic expressions; then, we could
   use the figure object to add numerical (or experimental) results using the
   commands associated to the specific plotting library.

We will discuss each backend in details in modules documentation.
