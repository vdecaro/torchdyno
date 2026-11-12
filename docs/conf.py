# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from datetime import datetime
from importlib import import_module

from jinja2.filters import FILTERS

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "TorchDyno"
author = "Valerio De Caro"
copyright = f"{(datetime.now().year)}, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.coverage",
    "sphinxcontrib.icon",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]
viewcode_follow_imported_members = True
autosummary_generate = True
autosectionlabel_prefix_document = True
myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
suppress_warnings = ["toc.not_included"]

coverage_show_missing_items = True
# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

autodoc_default_options = {"undoc-members": True}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_logo = "_static/images/logo.png"

html_static_path = ["_static"]
html_favicon = "_static/favicon/favicon-32x32.png"
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap",
    "css/custom.css",
]

html_context = {
    "github_url": "https://github.com",
    "github_user": "vdecaro",
    "github_repo": "torchdyno",
    "github_version": "main",
    "doc_path": "docs",
}


icon_links = [
    {
        "name": "GitHub",
        "url": "https://github.com/vdecaro/torchdyno",
        "icon": "fa-brands fa-github",
        "type": "fontawesome",
    },
]

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button-field"],
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["last-updated", "copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
    "show_nav_level": 1,
    "navigation_depth": 2,
    "icon_links": icon_links,
    "header_links_before_dropdown": 4,
    "logo": {
        "text": "TorchDyno",
        "alt_text": "TorchDyno - Home",
        "image_dark": "_static/images/logo.png",
        "image_light": "_static/images/logo.png",
    },
}
html_sidebars = {"**": ["sidebar-nav-bs", "sidebar-ethical-ads"]}

html_scaled_image_link = False


def landing_page_template(app, pagename, templatename, context, doctree):
    """Render the root document with the standalone landing template."""
    if pagename == "index":
        return "index.html"
    return None


def setup(app):
    app.connect("html-page-context", landing_page_template)



def filter_out_undoc_class_members(member_name, class_name, module_name):
    module = import_module(module_name)
    cls = getattr(module, class_name)
    if getattr(cls, member_name).__doc__:
        return f"~{class_name}.{member_name}"
    else:
        return ""


def filter_out_parent_class_members(member_name, class_name, module_name):
    module = import_module(module_name)
    cls = getattr(module, class_name)
    if member_name in cls.__dict__:
        return f"~{class_name}.{member_name}"
    else:
        return ""


FILTERS["filter_out_undoc_class_members"] = filter_out_undoc_class_members
FILTERS["filter_out_parent_class_members"] = filter_out_parent_class_members
