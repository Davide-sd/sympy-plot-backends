# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys
import inspect
import os
import sys
import sphinx_rtd_theme
import ast
from spb.doc_utils import _modify_code, _modify_iplot_code

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'Sympy Plotting Backends'
copyright = '2021, Davide Sandona\''
author = 'Davide Sandona\''

here = os.path.dirname(__file__)
repo = os.path.join(here, '..', '..')
_version_py = os.path.join(repo, 'spb', '_version.py')
version_ns = {}
with open(_version_py) as f:
    exec (f.read(), version_ns)

v = version_ns["__version__"]
# The short X.Y version
version = ".".join(v.split(".")[:-1])
# The full version, including alpha/beta/rc tags
release = v


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.linkcode',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx.ext.viewcode',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_rtd_theme',
    'sphinx_plotly_directive',
    'sphinx_panel_screenshot',
    'sphinx_k3d_screenshot',
]

# nbsphinx_allow_errors = True

# hide the table inside classes autodoc
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [
    '_static',
    '_static/tut-1',
    '_static/tut-2',
    '_static/tut-3',
    '_static/tut-4',
    '_static/tut-5',
]

# html_js_files = [
# ]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'SympyPlottingBackendsdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'SympyPlottingBackends.tex', 'Sympy Plotting Backends Documentation',
     'Davide Sandona\'', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'sympyplottingbackends', 'Sympy Plotting Backends Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'SympyPlottingBackends', 'Sympy Plotting Backends Documentation',
     author, 'SympyPlottingBackends', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}


def linkcode_resolve(domain, info):
    return None


# -- Options for sphinx_plotly_directive --------------------------------------

plotly_include_source = True
plotly_include_directive_source = False
plotly_iframe_height = "375px"
plotly_formats = ["png", "html", "pdf"]
plotly_intercept_code = _modify_code


# # -- Options for sphinx_panel_screenshot --------------------------------------

home_folder = os.path.expanduser("~")
chrome_path = os.path.join(home_folder, "selenium/chrome-linux/chrome")
chrome_driver_path = os.path.join(home_folder, "selenium/drivers/chromedriver")

panel_screenshot_small_size = [800, 550]
panel_screenshot_intercept_code = _modify_iplot_code
panel_screenshot_browser = "chrome"
panel_screenshot_browser_path = chrome_path
panel_screenshot_driver_path = chrome_driver_path

# -- Options for sphinx_k3d_screenshot ----------------------------------------

k3d_screenshot_browser = "chrome"
k3d_screenshot_browser_path = chrome_path
k3d_screenshot_driver_path = chrome_driver_path
k3d_screenshot_intercept_code = _modify_code
