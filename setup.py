from setuptools import setup, find_packages
import os

def readme():
    with open("README.md") as f:
        return f.read()

here = os.path.dirname(os.path.abspath(__file__))
version_ns = {}
with open(os.path.join(here, 'spb', '_version.py')) as f:
    exec (f.read(), {}, version_ns)

setup(
    name="sympy_plot_backends",
    version=version_ns["version"],
    description="Backends for plotting with SymPy",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Mathematics",
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
        "appdirs",
        "numpy",
        "scipy",  # helps when lambdifying expressions
        "sympy",
        "matplotlib",
        "mergedeep",
        "ipympl",
        "plotly>=4.14.3",
        "colorcet",
        "param",
        "panel",
        "holoviews",
        "bokeh",
        "ipyevents",
        "k3d",
        "vtk",  # needed for streamlines in k3d
    ],
)
