Control
-------

This module contains plotting functions for some of the common plots used
in control system. In particular, the following functions:

1. make use of the functions defined in ``spb.graphics.control``.
2. set axis labels to sensible choices.

Refer to :ref:`graphicscontrol` for a general explanation
about the underlying working principles, or if you are interested in a finer
customization of what is shown on the plot.

**NOTEs:**

* All the following examples are generated using Matplotlib. However, Bokeh
  can be used too, which allows for a better data exploration thanks to useful
  tooltips. Set ``backend=BB`` in the function call to use Bokeh.

* For technical reasons, all interactive-widgets plots in this documentation
  are created using Holoviz's Panel. Often, they will ran just fine with
  ipywidgets too. However, if a specific example uses the ``param`` library,
  or widgets from the ``panel`` module, then users will have to modify the
  ``params`` dictionary in order to make it work with ipywidgets.
  Refer to :ref:`interactive` module for more information.


.. module:: spb.plot_functions.control

.. autofunction:: plot_pole_zero
.. autofunction:: plot_step_response
.. autofunction:: plot_impulse_response
.. autofunction:: plot_ramp_response
.. autofunction:: plot_bode_magnitude
.. autofunction:: plot_bode_phase
.. autofunction:: plot_bode
.. autofunction:: plot_nyquist
.. autofunction:: plot_nichols
.. autofunction:: plot_root_locus

