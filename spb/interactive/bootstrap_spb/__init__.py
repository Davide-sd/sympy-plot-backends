"""
Bootstrap template based on the bootstrap.css library.
"""
import pathlib
import param
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

    # _css = pathlib.Path(__file__).parent / 'bootstrap.css'

    _template = pathlib.Path(__file__).parent / 'bootstrap.html'

    # TODO: uncomment this and remove the bigger __init__ when this
    # issue gets resolved:
    # https://github.com/holoviz/panel/issues/6275
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.add_variable("sidebar_location", self.sidebar_location)
    #     self.add_variable("full_width", self.full_width)
    #     self.add_variable("header_no_panning", self.header_no_panning)
    #     self.add_variable("show_header", self.show_header)

    def __init__(self, *args, **params):
        from panel.io.notifications import NotificationArea
        from panel.io.resources import _env, parse_template
        from panel.io.state import state
        from pathlib import Path, PurePath
        import jinja2
        from panel.layout import ListLike
        from panel.theme.base import THEMES
        from panel.config import config
        from panel.template.base import BaseTemplate
        from panel.pane import HTML

        tmpl_string = self._template.read_text(encoding='utf-8')
        try:
            template = _env.get_template(str(self._template.relative_to(Path(__file__).parent)))
        except (jinja2.exceptions.TemplateNotFound, ValueError):
            template = parse_template(tmpl_string)

        if 'header' not in params:
            params['header'] = ListLike()
        else:
            params['header'] = self._get_params(params['header'], self.param.header.class_)
        if 'main' not in params:
            params['main'] = ListLike()
        else:
            params['main'] = self._get_params(params['main'], self.param.main.class_)
        if 'sidebar' not in params:
            params['sidebar'] = ListLike()
        else:
            params['sidebar'] = self._get_params(params['sidebar'], self.param.sidebar.class_)
        if 'modal' not in params:
            params['modal'] = ListLike()
        else:
            params['modal'] = self._get_params(params['modal'], self.param.modal.class_)
        if 'theme' in params:
            if isinstance(params['theme'], str):
                params['theme'] = THEMES[params['theme']]
        else:
            params['theme'] = THEMES[config.theme]
        if 'favicon' in params and isinstance(params['favicon'], PurePath):
            params['favicon'] = str(params['favicon'])
        if 'notifications' not in params and config.notifications:
            params['notifications'] = state.notifications if state.curdoc else NotificationArea()

        BaseTemplate.__init__(self, template=template, **params)
        self._js_area = HTML(margin=0, width=0, height=0)
        state_roots = '{% block state_roots %}' in tmpl_string
        if state_roots or 'embed(roots.js_area)' in tmpl_string:
            self._render_items['js_area'] = (self._js_area, [])
        if state_roots or 'embed(roots.actions)' in tmpl_string:
            self._render_items['actions'] = (self._actions, [])
        if (state_roots or 'embed(roots.notifications)' in tmpl_string) and self.notifications:
            self._render_items['notifications'] = (self.notifications, [])
            self._render_variables['notifications'] = True
        if config.browser_info and ('embed(roots.browser_info)' in tmpl_string or state_roots) and state.browser_info:
            self._render_items['browser_info'] = (state.browser_info, [])
            self._render_variables['browser_info'] = True
        self._update_busy()
        self.main.param.watch(self._update_render_items, ['objects'])
        self.modal.param.watch(self._update_render_items, ['objects'])
        self.sidebar.param.watch(self._update_render_items, ['objects'])
        self.header.param.watch(self._update_render_items, ['objects'])
        self.main.param.trigger('objects')
        self.sidebar.param.trigger('objects')
        self.header.param.trigger('objects')
        self.modal.param.trigger('objects')

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
