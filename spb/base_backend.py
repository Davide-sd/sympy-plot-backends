import warnings
import numpy as np
from sympy.plotting.plot import BaseBackend

"""
-------------------------------------------------------------
|  keyword arg  | Matplolib | Bokeh | Plotly | Mayavi | K3D |
-------------------------------------------------------------
|     xlim      |     Y     |   Y   |    Y   |    N   |  N  |
|     ylim      |     Y     |   Y   |    Y   |    N   |  N  |
|     zlim      |     Y     |   N   |    Y   |    N   |  N  |
|    xscale     |     Y     |   Y   |    Y   |    N   |  N  |
|    yscale     |     Y     |   Y   |    Y   |    N   |  N  |
|    zscale     |     Y     |   N   |    Y   |    N   |  N  |
|     axis      |     Y     |   Y   |    Y   |    Y   |  Y  |
| aspect_ratio  |     Y     |   N   |    N   |    N   |  N  |
|   autoscale   |     Y     |   N   |    N   |    N   |  N  |
|    margin     |     Y     |   N   |    N   |    N   |  N  |
|     size      |     Y     |   Y   |    Y   |    Y   |  Y  |
|     title     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    xlabel     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    ylabel     |     Y     |   Y   |    Y   |    Y   |  Y  |
|    zlabel     |     Y     |   N   |    Y   |    Y   |  Y  |
|  line_color   |     Y     |   N   |    N   |    N   |  N  |
| surface_color |     Y     |   N   |    N   |    N   |  N  |
-------------------------------------------------------------
|       2D      |     Y     |   Y   |    Y   |    N   |  N  |
|       3D      |     Y     |   N   |    Y   |    Y   |  Y  |
| Latex Support |     Y     |   N   |    Y   |    N   |  Y  |
| Save Picture  |     Y     |   Y   |    Y   |    Y   |  Y  |
-------------------------------------------------------------
"""

class MyBaseBackend(BaseBackend):
    """ This class implements a few methods that could be used by
    the child classes.
    """
    
    def _line_length(self, x, y, z=None, start=None, end=None):
        """ Compute the cumulative length of the line.
        
        Parameters
        ==========
            
            x : numpy array of x-coordinates
            y : numpy array of y-coordinates
            z : numpy array of z-coordinates (optional)
        """
        def diff(x1):
            x2 = np.roll(x1, 1)
            x2[0] = x1[0]
            return x1 - x2
        
        if z is not None:
            z = np.zeros_like(x)
            
        length = np.sqrt(diff(x - x[0])**2 + diff(y - y[0])**2 + 
                    diff(z - z[0])**2)
        length = np.cumsum(length)
        length /= np.max(length)
        length = length * (end - start) + start
        return length
    
    def close(self):
        warnings.warn(str(type(self)) +
            " doesn't implement the concept of closing a figure.")
    
    def _get_mode(self):
        """ Verify which environment is used to run the code.

        Returns
        =======
            mode : int
                0 - the code is running on Jupyter Notebook or qtconsole
                1 - terminal running IPython
                2 - other type (?)
                3 - probably standard Python interpreter

        # TODO: detect if we are running in Jupyter Lab.
        """
        
        # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return 0   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return 1  # Terminal running IPython
            else:
                return 2  # Other type (?)
        except NameError:
            return 3      # Probably standard Python interpreter
