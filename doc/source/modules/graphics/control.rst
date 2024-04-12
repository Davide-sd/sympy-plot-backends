.. _graphicscontrol:

Control
-------

This module contains plotting functions for some of the common plots used
in control system. It works with transfer functions from SymPy, SciPy and the
`control module <https://github.com/python-control/python-control/>`_.
To achieve this seamlessly user experience, the *control module* must be
installed, which solves some critical issues related to
SymPy but also comes with it's own quirks.

In particular, ``impulse_response``, ``step_response`` and ``ramp_response``
provide two modes of operation:

1. ``control=True`` (default value): the symbolic transfer function is
   converted to a transfer function of the *control module*.
   The responses are computed with numerical integration.
2. ``control=False`` the responses are computed with the inverse Laplace
   transform of the symbolic output signal. This step is not trivial:
   sometimes it works fine, other times it produces wrong results,
   other times it consumes too much memory, potentially crashing the machine.

These functions exposes the ``lower_limit=0`` keyword argument, which is the
lower value of the time axis. If the default value (zero) is used, the
responses from the two modes of operation are identical. On the other hand,
if the value is different from zero, then results are different, with the
second mode of operation being correct. The first mode of operation is wrong
because the integration starts with a zero initial condition.

.. plot::
   :context: reset
   :include-source: True

   from sympy import *
   from spb import *
   x = symbols("x")
   s = symbols("s")
   G = (8*s**2 + 18*s + 32) / (s**3 + 6*s**2 + 14*s + 24)
   p1 = graphics(
      step_response(G, upper_limit=10, label="G1 - control", control=True, rendering_kw={"linestyle": "--"}),
      step_response(G, upper_limit=10, label="G1 - sympy", control=False, scatter=True, n=20),
      xlabel="Time [s]", ylabel="Amplitude", xlim=(0, 10), show=False
   )
   p2 = graphics(
      step_response(G, upper_limit=10, label="G1 - control", control=True, rendering_kw={"linestyle": "--"}, lower_limit=2),
      step_response(G, upper_limit=10, label="G1 - sympy", control=False, scatter=True, n=20, lower_limit=2),
      xlabel="Time [s]", ylabel="Amplitude", xlim=(0, 10), show=False
   )
   plotgrid(p1, p2)

The plotting module will warn the user if the first mode of operation is being
used with a ``lower_limit`` different than zero.

**NOTES:**

* All the following examples are generated using Matplotlib. However, Bokeh
  can be used too, which allows for a better data exploration thanks to useful
  tooltips. Set ``backend=BB`` in the function call to use Bokeh.

* For technical reasons, all interactive-widgets plots in this documentation
  are created using Holoviz's Panel. Often, they will ran just fine with
  ipywidgets too. However, if a specific example uses the ``param`` library,
  or widgets from the ``panel`` module, then users will have to modify the
  ``params`` dictionary in order to make it work with ipywidgets.
  Refer to :ref:`interactive` module for more information.


.. module:: spb.graphics.control

.. autofunction:: pole_zero
.. autofunction:: impulse_response
.. autofunction:: step_response
.. autofunction:: ramp_response
.. autofunction:: bode_magnitude
.. autofunction:: bode_phase
.. autofunction:: nyquist
.. autofunction:: ngrid
.. autofunction:: nichols
.. autofunction:: mcircles
.. autofunction:: control_axis
.. autofunction:: sgrid
.. autofunction:: zgrid
.. autofunction:: root_locus
