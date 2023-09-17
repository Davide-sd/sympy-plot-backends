.. _module-docs:

SPB Modules Reference
=======================

.. module:: spb

This document is automatically generated using spb's docstrings.

**Notes**:

* The examples illustrate the capabilities of the plotting module and the
  backends. If only basic plotting with Numpy and Matplotlib was
  installed, users might need to adapt the code samples to make it
  work with Matplotlib. Refer to :ref:`installation` to install the full
  requirements.

* For technical reasons, all interactive-widgets plots in this documentation
  are created using Holoviz's Panel. However, by default the plotting module
  uses ipywidgets. Often, interactive-widgets examples will ran just fine with
  ipywidgets too. However, if a specific example uses the ``param`` library,
  then users will have to modify the ``params`` dictionary in order to make
  it work with ipywidgets. More information can be found on :ref:`interactive`.

.. toctree::
   :maxdepth: 2

   graphics/index.rst
   plot_functions/index.rst
   plotgrid.rst
   series.rst
   renderers.rst
   interactive.rst
   defaults.rst
   backends/index.rst
   references.rst
