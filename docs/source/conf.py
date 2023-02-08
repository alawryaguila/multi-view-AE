# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import warnings
import datetime

sys.path.insert(0, os.path.abspath("../.."))
# -- Project information

project = 'Multi-view-AE'
author = 'Ana Lawry Aguila & Alejandra Jayme'
copyright = f"{datetime.datetime.now().year}, {author}"
release = '0.1'
version = '0.0.2'

# -- General configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel'
]


autosectionlabel_prefix_document = True

# -- sphinx.ext.autosummary
autosummary_generate = True

# -- sphinx.ext.autodoc
autodoc_member_order = "bysource"
autoclass_content = "both"

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
