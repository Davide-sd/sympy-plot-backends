"""
Bootstrap template based on the bootstrap.css library.
"""
import pathlib

import param

# from panel.io.resources import CSS_URLS, JS_URLS
# from panel.layout import Card
# from panel.template.base import BasicTemplate
from panel.template.theme import DarkTheme, DefaultTheme
from panel.template import BootstrapTemplate


class SymPyBootstrapTemplate(BootstrapTemplate):
    """
    A BootstrapTemplate designed for SymPy Plotting. Widgets will be placed
    into the sidebar area, which can be positioned on the left,
    on the right, on the top or on the bottom of the plot.
    """

    full_width = param.Boolean(default=True, doc="""
        Use the entire available width of the page. Default to True.""")
    
    header_no_panning = param.Boolean(default=True, doc="""
        Use no padding on the header and also use a smaller font size.
        Default to True.""")

    sidebar_width = param.String("15%", doc="""
        The width of the sidebar, in pixels or %. Default is 350px.""")
    
    sidebar_location = param.String("sbl", doc="""
        The location of the sidebar. Can be one of ['sbl', 'sbr', 'tb', 'bb']
        meaning sidebar-left, sidebar-right, topbar, bottom-bar, respectively.
        Default to 'sbl' (sidebar-left).
        """)
    
    show_header = param.Boolean(default=True, doc="""
        Wheter to show the header. Default to True.""")

    _css = pathlib.Path(__file__).parent / 'bootstrap.css'

    _template = pathlib.Path(__file__).parent / 'bootstrap.html'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_variable("sidebar_location", self.sidebar_location)
        self.add_variable("full_width", self.full_width)
        self.add_variable("header_no_panning", self.header_no_panning)
        self.add_variable("show_header", self.show_header)
    
    def add_variable(self, name, value):
        """
        Add parameters to the template, which may then be referenced
        by the given name in the Jinja2 template.

        Arguments
        ---------
        name : str
          The name to refer to the panel by in the template
        value : object
          Any valid Jinja2 variable type.
        """
        if name in self._render_variables:
            raise ValueError('The name %s has already been used for '
                             'another variable. Ensure each variable '
                             'has a unique name by which it can be '
                             'referenced in the template.' % name)
        self._render_variables[name] = value


class SymPyBootstrapDefaultTheme(DefaultTheme):

    _template = SymPyBootstrapTemplate


class SymPyBootstrapDarkTheme(DarkTheme):

    css = param.Filename(default=pathlib.Path(__file__).parent / 'dark.css')

    _template = SymPyBootstrapTemplate
