from setuptools import setup, find_packages
import os

def readme():
    with open("README.md") as f:
        return f.read()

here = os.path.dirname(os.path.abspath(__file__))
version_ns = {}
with open(os.path.join(here, 'spb', '_version.py')) as f:
    exec (f.read(), {}, version_ns)

_all_deps = [
    "scipy>=1.7.1",  # helps when lambdifying expressions
    "adaptive>=0.13.1",
    "notebook",
    "ipympl>=0.7.0",
    "plotly>=4.14.3",
    "panel>=1.0.0", # this includes param and bokeh
    "ipywidgets_bokeh", # starting from panel v0.13.0, it is not part of panel anymore
    "colorcet",
    "k3d>=2.9.7",
    "vtk",  # needed for streamlines in k3d
    "control>=0.10.0"
    # mayavi-related
    # "mayavi>=4.8.0",
    # "PyQt5>=5.15.7",
]

_dev_deps = _all_deps + [
    "pytest",
    "pytest-mock",
    "sphinx",
    "sphinx-rtd-theme",
    "kaleido",
    "numpydoc"
]

setup(
    name="sympy_plot_backends",
    version=version_ns["__version__"],
    description="Backends for plotting with SymPy",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    keywords="sympy plot plotting backend plotly bokeh mayavi k3d panel",
    url="https://github.com/Davide-sd/sympy-plot-backends",
    author="Davide Sandona",
    author_email="sandona.davide@gmail.com",
    license="BSD License",
    packages=find_packages(exclude=("tests", )),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "packaging",
        "appdirs>=1.4.4",
        "numpy>=1.21.1",
        "sympy>=1.10.1",
        "matplotlib>3.4.2",
        "mergedeep>=1.3.4",
    ],
    extras_require={
        "all": _all_deps,
        "dev": _dev_deps,
    }
)
