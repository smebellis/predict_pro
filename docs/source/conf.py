# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os
from pathlib import Path
# Get the absolute path to the src directory
project_root = Path(__file__).resolve().parent.parent.parent  # Adjust based on your structure
print(f"Project Root :{project_root}")
src_dir = str(project_root / 'src')

# Insert the src directory into sys.path
sys.path.insert(0, src_dir)

print(sys.path)  # Optional: for debugging purposes


project = 'ECE5831 Final Project'
copyright = '2024 Ryan Ellis, Ahmad Alamery'
author = 'Ryan Ellis, Ahmad Alamery'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx_simplepdf',]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
