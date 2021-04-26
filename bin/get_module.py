"""Functions to get the correct module version to run tests."""

from __future__ import print_function

import os
import sys


def path_hack():
    """
    Hack sys.path to import correct (local) module.
    """
    this_file = os.path.abspath(__file__)
    module_dir = os.path.join(os.path.dirname(this_file), "..")
    module_dir = os.path.normpath(module_dir)
    print("asd", module_dir)
    sys.path.insert(0, module_dir)
    return module_dir
