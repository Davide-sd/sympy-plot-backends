Control
-------

This module contains plotting functions for some of the common plots used
in control system. The main difference between the these functions
and the ones from ``spb.plot_functions.control`` is that the latter set
axis labels to sensible choices.

**NOTE:** 
For technical reasons, all interactive-widgets plots in this documentation are
created using Holoviz's Panel. Often, they will ran just fine with ipywidgets
too. However, if a specific example uses the ``param`` library, then users
will have to modify the params dictionary in order to make it work with
ipywidgets. Refer to :ref:`interactive` module for more information.

.. module:: spb.graphics.control

.. autofunction:: pole_zero
.. autofunction:: impulse_response
.. autofunction:: step_response
.. autofunction:: ramp_response
.. autofunction:: bode_magnitude
.. autofunction:: bode_phase
.. autofunction:: nyquist
.. autofunction:: nichols
.. autofunction:: control_axis
.. autofunction:: sgrid
.. autofunction:: root_locus
