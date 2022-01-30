Plot
----

About the Implementation
========================

Why different backends inheriting from the ``Plot`` class? Why not using
something like `holoviews <https://holoviews.org/>`_, which allows to plot
numerical data with different plotting libraries using a common interface?
In short:

* Holoviews only support Matplotlib, Bokeh, Plotly. This would make
  impossible to add support for further libraries, such as K3D, ...
* Not all needed features might be implemented on Holoviews. Think for example
  to plotting a gradient-colored line. Matplotlib and Bokeh are able to
  visualize it correctly, Plotly doesn't support this functionality. By not
  using Holoviews, we can more easily implement some work around.


.. module:: spb.backends.base_backend

.. autoclass:: Plot

.. autofunction:: spb.backends.base_backend.Plot.append

.. autofunction:: spb.backends.base_backend.Plot.extend

.. autoattribute:: spb.backends.base_backend.Plot.colorloop

.. autoattribute:: spb.backends.base_backend.Plot.colormaps

.. autoattribute:: spb.backends.base_backend.Plot.cyclic_colormaps

.. autoattribute:: spb.backends.base_backend.Plot.fig
