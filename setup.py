from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name = 'sympy_plot_backends',
    version = '0.1.0',
    description = 'Backends for plotting with SymPy',
    long_description = readme(),
    classifiers=[
        'Development Status :: Beta',
        'License :: GNU GPL v3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Mathematics, Engineering',
    ],
    keywords='sympy plot plotting backend plotly bokeh mayavi k3d',
    url = 'https://github.com/Davide-sd/sympy_plot_backends',
    author = 'Davide Sandona',
    author_email = 'sandona.davide@gmail.com',
    license='GNU GPL v3',
    packages = [
        'spb',
    ],
    include_package_data=True,
    zip_safe = False,
    install_requires = [
        "numpy",
        "sympy>=1.6.1",
        "matplotlib",
        "plotly>=4.14.3",
        "bokeh",
        "mayavi",
        "PyQt5",
        "k3d"
    ]
)