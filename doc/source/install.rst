
.. _installation:

Installation
------------

SymPy Plotting Backends can be installed with `pip` or `conda`. By default,
only basic plotting with Numpy and Matplotlib will be installed::

    pip install sympy_plot_backends

Or::

    conda install -c conda-forge sympy_plot_backends 

To install the complete requirements in order to get adaptive algorithm,
interactive plots, bokeh, plotly, k3d, vtk, execute the following command::

    pip install sympy_plot_backends[all]

If you are using zshell, the above `pip` command is going to fail.
Use the following instead::

    pip install "sympy_plot_backends[all]"

To install the complete requirements with conda::

    conda install -c conda-forge sympy_plot_backends
    # scipy gives more numerical functions, notebook install the
    # Jupyter Notebook (enabling interactivity)
    conda install -c anaconda scipy notebook
    # to install the adaptive algorithm:
    conda install -c conda-forge adaptive
    # to install interactive widgets with holoviz's Panel
    conda install -c conda-forge panel
    # to install interactive widgets with ipywidgets
    conda install -c anaconda ipywidgets
    conda install -c conda-forge ipympl
    conda install -c bokeh ipywidgets_bokeh
    # colorcet gives more colormaps
    conda install -c conda-forge colorcet
    # to install K3D-Jupyter
    conda install -c conda-forge k3d msgpack-python
    # to install Plotly
    conda install -c plotly plotly
    # to enable 3D streamlines plots with matplotlib and K3D-Jupyter
    conda install -c conda-forge vtk



Verify the installation
=======================

After installation, it is best to verify that the freshly-installed SymPy
Plotting Backends work. To do this, start up Python and import the necessary
functionalities, as shown below. To improve typing experience, the actual name
of the module has been set to `spb`::

    $ python
    >>> from sympy import *
    >>> from spb import *

From here, execute some simple statements like the ones below::

    >>> x = Symbol('x')
    >>> plot(sin(x), cos(x))

If the additional requirements have been installed, try the following:

.. code-block:: python

   from sympy import *
   from spb import *
   x, a, b, c = symbols("x, a, b, c")
   plot(
       (cos(a * x + b) * exp(-c * x), "oscillator"),
       (exp(-c * x), "upper limit", {"line_dash": "dotted"}),
       (-exp(-c * x), "lower limit", {"line_dash": "dotted"}),
       (x, 0, 2 * pi),
       params={
           a: (1, 0, 10),     # frequency
           b: (0, 0, 2 * pi), # phase
           c: (0.25, 0, 1)    # damping
       },
       ylim=(-1.25, 1.25),
       backend=BB,
       servable=True
   )

It is also the perfect time to verify that K3D-Jupyter is working:

1. launch ``jupyter notebook``.
2. open a notebook.
3. run the following code:

   .. code-block:: python

      from sympy import *
      from spb import *
      var("x, y")
      plot3d(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2), backend=KB)

   If no figure is visible in the output cell, follow this procedure:

   1. Save the Notebook.
   2. Close Jupyter server.
   3. Run the following commands, which are going to install the Jupyter
      extension for K3D:

       * ``jupyter nbextension install --user --py k3d``
       * ``jupyter nbextension enable --user --py k3d``

   4. Restart ``jupyter notebook``
   5. Open the previous notebook and execute the plot command.

Refer :ref:`functions` to explore visualize the output of some of the
plotting functions, or to the :ref:`tutorials` for a starter guide on using
the plotting backends.


Installing Mayavi
=================

This plotting module comes with ``MayaviBackend``. Mayavi is a 3D plotting
library which can be used on any Python interpreter.
However, it is not the easiest to install.

If you are interested in using it, please follow
`Mayavi's installation instruction <https://docs.enthought.com/mayavi/mayavi/installation.html>`_.


About Matplotlib
================

If you are doing development work on this module, Matplotlib 3.4.2 is required
for tests to pass!