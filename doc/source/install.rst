
.. _installation:

Installation
------------

SymPy Plotting Backends can be installed with `pip` or `conda`::

    pip install sympy_plot_backends

Or::

    conda install sympy_plot_backends


About Matplotlib
================

This module will install Matplotlib v3.4.2. If you want to use a newer version,
just upgrade it with::

    pip install -U matplotlib

Note: if you are doing development work on this module, v3.4.2 is required for
tests to pass!


Verify the installation
=======================

After installation, it is best to verify that the freshly-installed SymPy
Plotting Backends work. To do this, start up Python and import the necessary
functionalities, as shown below. To improve typing experience, the acutal name
of the module has been set to `spb`::

    $ python
    >>> from sympy import *
    >>> from spb import *

From here, execute some simple statements like the ones below::

    >>> x = Symbol('x')
    >>> plot(sin(x), cos(x), backend=BB)

Or:

    >>> plot_complex(sin(x), (x, -3-3j, 3+3j), backend=MB)


It is also the perfect time to verify that K3D-Jupyter is working:

1. launch `jupyter notebook`.
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

   4. Restart `jupyter notebook`
   5. Open the previous notebook and execute the plot command.

Refer to the :ref:`tutorials` for a starter guide on using the plotting backends.
