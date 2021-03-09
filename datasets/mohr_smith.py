import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

# This codes loads a labelled subset of the Mohr-Smith 2017 dataset containing photometric data of
# Be stars and other types of OB stars
# More info at:
# VizieR Online Data Catalog: New OB star candidates in Carina Arm (Mohr-Smith+, 2015):
#   url: https://ui.adsabs.harvard.edu/abs/2017yCat..74503855M/abstract
#  Reddening-free Q indices to identify Be star candidates (arXiv):
#  url: https://arxiv.org/abs/2009.06017

coefficients = {'umag': 4.39,
                'gmag':  3.30,
                'rmag':  2.31,
                'imag':  1.71,
                'Hamag': 2.14,  # valor interpolado
                'Jmag':  0.72,
                'Hmag':  0.46,
                'Kmag':  0.306,
                'W1mag': 0.18,
                'W2mag': 0.16}

systems =        {'umag': 'VPHAS',
                'gmag': 'VPHAS',
                'rmag': 'VPHAS',
                'imag': 'VPHAS',
                'Hamag':'VPHAS',
                'Jmag': '2MASS',
                'Hmag': '2MASS',
                'Kmag': '2MASS',
                'W1mag':'WISE',
                'W2mag':'WISE',
                }

default_filename = "old/Mohr-Smith_2017.csv"



def load(filename=None,binary=False,dropna=True):
    # load mohr-smith datsaet
    if filename is None:
        filename = "Mohr-Smith_color.csv"


    folderpath =Path(__file__).parent.absolute()
    filepath = folderpath / filename
    df = pd.read_csv(filepath)
    if dropna:
        n = len(df)
        df.dropna(inplace=True)
        new_n=len(df)
        print(n,new_n)
        if new_n<n:
            print(f"Warning: dropped {new_n-n} rows with incomplete values. ")
            print(f"Rows (original):   {n}")
            print(f"Rows (after drop): {new_n}")

    y_columns = ["goodOB","EM","SUB","LUM"]
    y = df[y_columns]
    x_columns = [ 'umag', 'gmag', 'rmag',
                  'imag', 'Hamag', 'Jmag', 'Hmag', 'Kmag',
                  'W1mag', 'W2mag',
                  ]
    x = df[x_columns]
    metadata_columns = set(df.columns).difference(set(x_columns).union(set(y_columns)))
    # metadata_columns  = ["VPHAS-OB1","RAJ2000","DEJ2000",  'logTeff', 'A0', 'Rv', 'mu',
    #                     'chi2']
    metadata = df[metadata_columns]

    if binary:
        y = df["EM"] ==1
    return x,y,metadata



# magnitude_info_mohr =  {'umag': ['VPHAS', 'u', 3607.7],
#                         'gmag': ['VPHAS', 'g', 4679.5],
#                         'rmag': ['VPHAS', 'r', 6242.1],
#                         'imag': ['VPHAS', 'i', 7508.5],
#                         'Hamag': ['VPHAS', 'Ha', 6590.8],
#                         'Jmag': ['2MASS', 'J', 12350.0],
#                         'Hmag': ['2MASS', 'H', 16620.0],
#                         'Kmag': ['2MASS', 'K', 21590.0],
#                         'W1mag': ['WISE', 'W1', 34000.0, 0.18],
#                         'W2mag': ['WISE', 'W2', 46000.0, 0.16]}