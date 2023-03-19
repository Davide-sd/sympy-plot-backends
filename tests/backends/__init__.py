# NOTE
# Here, let's test that each backend:
#
# 1. receives the correct number of data series.
# 2. raises the necessary errors.
# 3. correctly use the common keyword arguments to customize the plot.
# 4. shows the expected labels.
#
# This should be a good starting point to provide a common user experience
# between different backends.
#
# If your issue is related to the generation of numerical data from a
# particular data series, consider adding tests to test_series.py.
# If your issue is related to the processing and generation of *Series
# objects, consider adding tests to test_functions.py.
