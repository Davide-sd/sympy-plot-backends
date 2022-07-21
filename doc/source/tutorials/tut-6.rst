6 - Evaluation Algorithm and Singularity Detection
--------------------------------------------------

How does the plotting module works? Conceptually, it is very simple:

1. it converts the symbolic expression to a function by using ``lambdify``,
   which will be used for numerical evaluation. Usually, the default evaluation
   modules are Numpy and Scipy.
2. It will evaluate the function over the specified domain.
3. The numerical data can be post-processed and later plotted.

Regarding numerical evaluation, in the previous tutorials we have seen that 2D
line plots can either use:

* an adaptive algorithm which is going to chose where to evaluate a function in
  order to obtain a smooth plot. The iterative procedure minimizes some loss
  function (``loss_fn``) and will stop when the ``adaptive_goal`` has been
  reached. This is the default algorithm used by the plotting module.
* a uniform meshing algorithm, which divides the specified range into ``n``
  uniformly spaced points over which the function will be evaluated.

In the following tutorial we are going to explore a few examples illustrating
the limitations of the adaptive algorithm. In particular, we will understand
when it is not appropriate to use it. Generally, if a function exhibits
mid-to-high frequencies in relation to the plotting range, then it is better to
switch to the uniform meshing algorithm.

We have also seen that we can play with ``detect_poles`` and ``eps`` in order
to detect singularities. The singularity-dection algorithm is extremely simple,
as it doesn't analyze the symbolic expression in any way. As a matter of fact,
it only relies on the gradient of the numerical data, thus it is a
post-processing step. This means than the user has to detect if a function
contains one or more singularities, eventually activating the detection
algorithm and playing with the parameters in order to get the expected result.
This is a try-and-repeat process until the user is satisfied with the result.

Remember that the documentation associated to the ``plot`` function can be
accessed by executing ``help(plot)``.

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> from sympy import *
   >>> from spb import *
   >>> x = symbols("x")

Example
=======

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> expr = x * sin(20 * x) - Abs(2 * x) + 6
   >>> plot(expr, (x, -1, 1))

Here the plotting module used the adaptive algorithm. In the provided range,
the function has a relatively low frequency, so the adaptive algorithm (using
the default options) was able to create a smooth plot.

Let's try to use a wider plot range:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, (x, -10, 10))

This is a case of mid-to-high frequencies (in relation to the plotting range
used). We can see a few "missed" spikes. If we zoom into the plot, we will also
see a very poor smoothness. The adaptive algorithm worked as expected: it
minimized some loss function (``loss_fn``) until the default goal was reached
(``adaptive_goal=0.01``). To improve the output we can either:

1. decrease the value of ``adaptive_goal``: depending on the value, the
   execution will slow down quite a bit.
2. switch to the uniform meshing algorithm and increase the number of
   discretization points. This techniques will use Numpy arrays for the
   evaluation, so we are going to get relatively good performances.

Let's try to decrease ``adaptive_goal`` by one order of magnitude:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, (x, -10, 10), adaptive_goal=1e-03)


The resulting plot is much better: if we zoom into it we will see a nice smooth
line. However, the evaluation was significantly slower!

For comparison, let's try to use the uniform meshing algorithm. This will
create a nice smooth plot almost instantly:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, (x, -10, 10), adaptive=False, n=1e04)


Example
=======

Depending on the function being plotted, the evaluation with the adaptive
algorithm might produce warning messages. For example:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(floor(x))


.. code-block:: text

   UserWarning: The evaluation with NumPy/SciPy failed.
   TypeError: ufunc 'floor' not supported for the input types, and the inputs
   could not be safely coerced to any supported types according to the casting
   rule ''safe''
   Trying to evaluate the expression with Sympy, but it might be a slow
   operation.

What does that warning message means? For reasons too long to explain, the
numerical arguments passed to the lambdified-function are of type ``complex``.
There are some Numpy/Scipy functions that are not designed for this numerical
data type, for example the ``floor`` and ``ceil`` functions. For example:

>>> import numpy as np
>>> try:
>>>     np.floor(5+0j)
>>> except TypeError as err:
>>>     print(err)


.. code-block:: text

   ufunc 'floor' not supported for the input types, and the inputs could not
   be safely coerced to any supported types according to the casting rule
   ''safe''

The plotting algorithm catches that exception and changes the evaluation module
to SymPy. The evaluation succeds, but it is going to be much much slower!

Here is another rule of thumb: if our symbolic expression contains function
like ``floor`` or ``ceil`` it is better to use the uniform meshing algorithm, in
which the arguments to the function are going to be of type `float`. Also,
since we are dealing with a ``floor`` function, there are discontinuities
between the horizontal segments. Let's activate the singularity-detection
algorithm:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(floor(x), adaptive=False, n=1e04, detect_poles=True)


Example
=======

Let's try another example of a function containing the ``floor`` function:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> expr = tan(floor(30 * x)) + x / 8
   >>> plot(expr, adaptive=False, n=1e04)

There is a wide spread along the y-direction. Let's limit it:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, adaptive=False, n=1e04, ylim=(-10, 10))

Let's remember that we are dealing with a ``floor`` function, so ther should be
distinct segments in the plot:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, adaptive=False, n=1e04, ylim=(-10, 10), detect_poles=True)


Example
=======

The following example will probably take forever to plot (using the adaptive
algorithm):

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> expr = sign(x) * (sin(1 - 1 / cos(x)) + Abs(x) - 6)
   >>> # plot(expr)

We can stop the execution.

Why is it so slow? Let's look at the argument of the ``sin`` function (the
frequency): as ``cos(x)`` approaches 0, the frequency goes to infinity. The
adaptive algorithm is trying to resolve this situation, but it's going to take
a very long time. We have two options:

1. increase the value of ``adaptive_goal``, thus reducing the smoothness of the
   function and potentially loosing important information.
2. use the uniform meshing algorithm.

Let's try the second approach:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, adaptive=False, n=1e04)

Much better, but the plot is still misleading: there is a ``sign`` function in
the expression, so there must be some discontinuities. Let's activate the
singularity detection algorithm:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, adaptive=False, n=1e04, detect_poles=True)

The singularity detection algorithm has done too much: it has also disconnected
the high frequency regions. We can try to get a better visualization by:

* increasing the number of discretization points.
* reducing the ``eps`` parameter. The smaller this parameter, the higher the
  threshold used by the singularity detection algorithm.

This is going to take a few attempts:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, adaptive=False, n=5e04, detect_poles=True, eps=1e-04)


Example
=======

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> expr = sin(20 * x) + sign(sin(19.5 * x)) + x
   >>> plot(expr)

The expression contains a ``sign`` function, so there should be discontinuities.
Also, if we zoom into the plot we see that it is not very "smooth": the
frequency is quite high with respect to the plotting range. So: 

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, adaptive=False, n=1e04, detect_poles=True)


Example
=======

Another function having many singularities:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> expr = 1 / cos(10 * x) + 5 * sin(x)
   >>> plot(expr)

Again, a very big spread along the y-direction. We need to limit it:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, ylim=(-10, 10))

The plot is clearly misleading. We can guess that it has a mid-to-high
frequency with respect to the plotting range. Also, by looking at the
expression there must be singularities:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, ylim=(-10, 10), adaptive=False, n=1e04, detect_poles=True)

We can improve it even further by reducing the ``eps`` parameter:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, ylim=(-10, 10), adaptive=False, n=1e04, detect_poles=True, eps=1e-04)


Example
=======

Let's try to plot the Gamma function:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> expr = gamma(x)
   >>> plot(expr, (x, -5, 5))

A very big spread along the y-direction. We need to limit it:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, (x, -5, 5), ylim=(-5, 5))

Here we can see a few discontinuities. Let's enable the singularity detection
algorithm:

.. plot::
   :context: close-figs
   :format: doctest
   :include-source: True

   >>> plot(expr, (x, -5, 5), ylim=(-5, 5), adaptive=False, n=2e04, detect_poles=True, eps=1e-04)
