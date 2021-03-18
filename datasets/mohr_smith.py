from . import base

# Loads a labelled subset of the Mohr-Smith 2017 dataset containing photometric data of
# Be stars and other types of OB stars
# More info at:
# VizieR Online Data Catalog: New OB star candidates in Carina Arm (Mohr-Smith+, 2015):
#   url: https://ui.adsabs.harvard.edu/abs/2017yCat..74503855M/abstract
#  Reddening-free Q indices to identify Be star candidates (arXiv):
#  url: https://arxiv.org/abs/2009.06017
coefficients=base.coefficients
systems=base.systems

default_filename = "Mohr-Smith_2017.csv"


def load(filename=default_filename,dropna=True,verbose=False):
    y_columns = ["goodOB","EM","SUB","LUM"]

    return base.load(filename,base.twomass_x_columns,y_columns,dropna,verbose=verbose)
