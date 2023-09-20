.. Sympy Plotting Backends documentation master file, created by
   sphinx-quickstart on Fri Jul 30 21:36:12 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Sympy Plotting Backends's documentation!
===================================================

This Python module contains a few plotting backends that can be used with
`SymPy <http://github.com/sympy/sympy/>`_ and `Numpy <https://github.com/numpy/numpy>`_.
A backend represents the plotting library:
it provides the necessary functionalities to quickly creates the most common
types of plots, such as line plots, surface plots, parametric plots,
vector plots, complex plots, geometric plots.

Most of the plotting functions can also automatically generates widgets
(sliders, buttons, ...) starting from symbolic expressions. It allows to
better understand the influence of each parameters in a particular expression,
without the knowledge of complicated widget libraries.

.. image:: _static/iplot_bokeh.png
  :width: 220
  :alt: interactive plot with bokeh

.. image:: _static/bokeh_domain_coloring.png
  :width: 220
  :alt: domain coloring

.. image:: _static/trefoil-knot-plotly.png
  :width: 220
  :alt: 3D surface

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview.rst
   install.rst
   modules/index.rst
   tutorials/index.rst
   changelog.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
