import os
import sys
import gators

sys.path.insert(0, '..')
project = 'gators'
copyright = '2021, the gators development team.'
author = 'The gators team'
version = gators.__version__


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath',
    'numpydoc',
    'nbsphinx',
    'sphinx.ext.autosummary',
    ]

autoclass_content = "class"
autodoc_member_order = "bysource"
templates_path = ['_templates']
language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
todo_include_todos = True
master_doc = 'index'
numpydoc_show_class_members = False
autosummary_generate = True
panels_add_bootstrap_css = False
html_use_index = False
html_domain_indices = False
html_theme = 'pydata_sphinx_theme'
html_css_files = ['css/gators.css']
html_static_path = ['_static']
html_logo = '../doc_data/GATORS_LOGO.png'
html_favicon = '../doc_data/gators_logo.ico'
# html_theme_options = {
#   "logo_link": "index",
#   "github_url": "https://github.paypal.com/pages/Simility-R/gators/",
# }
man_pages = [
    ('index', 'gators', u'gators Documentation',
     [u'the gators team'], 1)
]
