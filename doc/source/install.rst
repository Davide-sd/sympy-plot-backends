 
.. _installation:

Installation
------------

SymPy Plotting Backends can be installed with `pip` or `conda`::

    pip install sympy_plot_backends

Or::

    conda install sympy_plot_backends

Run SymPy
=========

After installation, it is best to verify that the freshly-installed SymPy
Plotting Backends work. To do this, start up Python and import the necessary
functionalities, as shown below. To improve typing experience, the acutal name
of the module has been set to `spb`::

    $ python
    >>> from sympy import *
    >>> from spb import *
    >>> from spb.backends.matplotlib import MB
    >>> from spb.backends.k3d import KB
    >>> from spb.backends.plotly import PB
    >>> from spb.backends.bokeh import BB

From here, execute some simple SymPy statements like the ones below::

    >>> x = Symbol('x')
    >>> plot(sin(x), cos(x), backend=BB)

Or::

    >>> complex_plot(sin(x), (x, -3-3j, 3+3j), backend=MB)

For a starter guide on using the plotting backends effectively, refer to the
:ref:`tutorial`.
