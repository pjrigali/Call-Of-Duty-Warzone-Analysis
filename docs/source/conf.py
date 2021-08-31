
import os
import sys
sys.path.insert(0, os.path.abspath('..\..'))

project = 'warzone analysis'
copyright = '2021, Peter Rigali'
author = 'Peter Rigali'
release = '2.4.0'
version = '2.4.0'

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
