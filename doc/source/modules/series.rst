
Series
------

Instances of ``BaseSeries`` performs these tasks:

1. stores the symbolic expressions and ranges.

2. stores rendering-related attributes.

3. create a numerical function of the symbolic expressions using ``lambdify``.

4. when requested, evaluate the numerical function and return the data.

Backends will request numerical data from each data series and will render
them according to the values stored in their attributes.

The best way to instantiate a specific data series is to call one of the
functions from :ref:`graphics`.

The following list shows the public methods/attributes.

.. module:: spb.series

.. autoclass:: BaseSeries

.. autofunction:: spb.series.BaseSeries.get_data

.. autoattribute:: spb.series.BaseSeries.params
