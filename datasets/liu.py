import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from . import base

# This codes loads a labelled subset of the Liu 2019 dataset containing photometric data of
# Be stars and other types of OB stars
# More info at:
# A Catalogue of OB Stars from LAMOST Spectroscopic Survey
#   url: https://arxiv.org/abs/1902.07607
#  Reddening-free Q indices to identify Be star candidates (arXiv):
#  url: https://arxiv.org/abs/2009.06017

coefficients = base.coefficients
systems = base.systems

default_filename = "Liu2019_LAMOST_OBstars-IPHAS-SDSS-2MASS_subset.csv"

def load(filename=default_filename,dropna=True,verbose=False):
    y_columns = ["Halpha"]
    return base.load(filename,base.twomass_x_columns,y_columns,dropna,verbose=verbose)
