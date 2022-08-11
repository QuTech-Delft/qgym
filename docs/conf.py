import os
import sys

package_name = "qgym"
sys.path.insert(
    0, os.path.abspath(f"build/lib/qgym")
)
# -- Project information -----------------------------------------------------

import datetime
import importlib

year = datetime.date.today().year
project = package_name
copyright = f"2022-{year}, TNO; QuTech"
author = "TNO; QuTech"
version = importlib.import_module(package_name).__version__

# -- Extensions --------------------------------------------------------------
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True

mathjax3_config = {
    "tex2jax": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

# -- Options for HTML output -------------------------------------------------

html_title = project
html_theme_options = {
    "analytics_id": "G-7KJ0LH83V6",
    "analytics_anonymize_ip": False,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = ["docs/_templates"]

# NOTE: All the lines are after this are the theme-specific ones. These are
#       written as part of the site generation pipeline for this project.
# !! MARKER !!
html_theme = "sphinx_rtd_theme"

html_css_files = [
    "custom.css",
]
