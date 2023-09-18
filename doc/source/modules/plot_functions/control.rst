Control
-------

This module contains plotting functions for some of the common plots used
in control system.

**NOTE:** 
For technical reasons, all interactive-widgets plots in this documentation are
created using Holoviz's Panel. Often, they will ran just fine with ipywidgets
too. However, if a specific example uses the ``param`` library, then users
will have to modify the params dictionary in order to make it work with
ipywidgets. Refer to :ref:`interactive` module for more information.

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

