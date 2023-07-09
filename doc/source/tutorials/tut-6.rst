6 - Extending the module
------------------------

This tutorial explains how to extend the plotting module when the plotting
functions and/or the customization options are not enough to achieve the
desired results.

We are only going to look at extending ``MatplotlibBackend`` (or ``MB``),
as the procedure to modify other backends is the same.

Here is how the plotting module works:

* a plotting function is called, receiving one or more symbolic expression,
  as well as keyword arguments to further customize the output.
* inside the plotting function, one (or more) data series are instantiated for
  each symbolic expression. These will generate the numerical data.
* then, a plot object is instantiated, ``p``, which receives the data series.
  For each data series, an appropriate renderer is created. A renderer
  contains the logic to visualize the numerical data on the chart, and keep it
  updated when a widget-plot is created.
* When ``p.show()`` is executed, a new figure is created and a loop
  is run over the renderers to add the numerical data to the figure.

The structure of a renderer is described in :ref:`renderers`.

Let's say we want to create a new plotting function to fill the area between
2 symbolic expressions. Here is a commented code-block to achieve that goal:

.. plot:: ./tutorials/extended_module.py
   :include-source: True
