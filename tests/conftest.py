import pytest


@pytest.fixture
def ipywidgets_options():
    return dict(show=False, imodule="ipywidgets")


@pytest.fixture
def panel_options():
    return dict(show=False, imodule="panel")
