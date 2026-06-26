import os
import subprocess
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "madspace"
copyright = "2025, Theo Heimel"
author = "Theo Heimel"
release = "0.2.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.napoleon",
    # "sphinx.ext.linkcode",
    "sphinx_autodoc_typehints",
    "breathe",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
}

autoclass_content = "both"
# add_module_names = False
typehints_fully_qualified = False

breathe_projects = {"madspace": "../build/doxygenxml"}
breathe_default_project = "madspace"
breathe_domain_by_extension = {"h": "cpp"}
breathe_default_members = ("members", "undoc-members")


def generate_doxygen_xml(app):
    build_dir = os.path.join(app.confdir, "..", "build")
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)

    try:
        subprocess.call(["doxygen", "--version"])
        retcode = subprocess.call(["doxygen"], cwd=os.path.join(app.confdir, ".."))
        if retcode < 0:
            sys.stderr.write(f"doxygen error code: {-retcode}\n")
    except OSError as e:
        sys.stderr.write(f"doxygen execution failed: {e}\n")


def setup(app):
    app.connect("builder-inited", generate_doxygen_xml)
