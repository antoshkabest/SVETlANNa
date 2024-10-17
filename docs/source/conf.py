# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(r'C:\Users\gunne\projects\python\dnn\SVETlANNa\svetlanna'))

project = 'SVETlANNa'
copyright = '2024, Alexey Shcherbakov, Alexey Kokhanovskiy, Vladimir Igoshin, Semen Chugunov, Denis Sakhno'
author = 'Alexey Shcherbakov, Alexey Kokhanovskiy, Vladimir Igoshin, Semen Chugunov, Denis Sakhno'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Автоматическая генерация документации из docstrings
    'sphinx.ext.napoleon',     # Поддержка формата NumPy и Google в docstrings
    'sphinx.ext.viewcode',     # Добавление ссылок на исходный код
    'sphinx.ext.todo',         # Поддержка TODO в документации
    'sphinx_autodoc_typehints',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = [
    r'C:\Users\gunne\projects\python\dnn\SVETlANNa\examples',
    r'C:\Users\gunne\projects\python\dnn\SVETlANNa\tests'
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_logo = '_static/dog.png'


latex_documents = [
    ('index', 'svetlanna.tex', 'SVETLANNa Documentation',
     'Semen Chugunov', 'manual'),
]

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'latex_show_urls': 'footnote'
}
