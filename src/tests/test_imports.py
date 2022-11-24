import unittest


class TestImports(unittest.TestCase):
    """
    Basic smoke test to ensure that the installed packages can actually be
    imported (we had a compatibility issue once that was not resolved
    properly by conda).
    """
    def test_infrastructure_packages(self):
        import gdown
        import sphinx
        import click
        import joblib
        import requests

    def test_common_packages(self):
        import numpy
        import scipy.sparse
        import pandas
        import bokeh
        import matplotlib
        import sklearn

    def test_packages(self):
        from IPython.display import display, Markdown, Latex
        from PIL import Image
        from community import community_louvain, modularity
        from partition_igraph import community_ecg as ecg
        import cdlib.algorithms as cd
        import hdbscan
        import igraph
        import imageio
        import imageio.v2
        import leidenalg
        import networkx
        import pynndescent
        import scipy
        import seaborn
        import umap
        import umap.plot
        import zipfile
        import re
