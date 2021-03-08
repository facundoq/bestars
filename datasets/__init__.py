import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd



def common_columns(datasets:[str]):
    def lequal(l1,l2):
        l1=list(sorted((l1)))
        return len(l1) == len(l2) and sorted(l1)==sorted(l2)

    if lequal(datasets,["liu2017","mohr"]):
        return [ 'umag', 'gmag', 'rmag',
                 'imag', 'Hamag', 'Jmag', 'Hmag', 'Kmag',
                 ]
    else:
        raise ValueError(f"No entries  for dataset combination: {datasets}")

#
# def load_data(binary=False,dropna=True):
#     data=np.loadtxt("ob_features.csv",delimiter=",",skiprows=1)
#     id = data[:,0]
#     y = data[:,12]
#     if original:
#         x= data[:,4:12]
#     else:
#         x = data[:,13:]
#
#     if binary:
#         y[y!=1]=0
#         class_names = ["OB","EM"]
#     else:
#         class_names = ["OB","EM","Sub","Over","Multi"]
#
#     return (x,y,id),class_names


def load_liu(filename=None,dropna=True):

    if filename is None:
        filename = "old/Liu2019_Etiquetados_IPHAS-SDSS-2MASS-AllWISE_OK.csv"
        filename ="CANDIDATES-Halfa_COMPLETA_Liu2019_LAMOST_OBstars-IPHAS-SDSS-2MASS_short_2.csv"

    folderpath =Path(__file__).parent.absolute()
    filepath = folderpath / filename
    df = pd.read_csv(filepath)
    if dropna:
        df.dropna(inplace=True)

    y_columns = ["H_alfa"]
    print(df.columns)
    y = df[y_columns]


    x_columns = [ 'umag', 'gmag', 'rmag',
                  'imag', 'Hamag', 'Jmag', 'Hmag', 'Kmag',
                  #'W1mag', 'W2mag', 'W3mag','W4mag',
                  ]
    x = df[x_columns]

    metadata_columns = set(df.columns).difference(set(x_columns).union(set(y_columns)))
    metadata = df[metadata_columns]

    return x,y,metadata

def load_mohr(filename=None,binary=False,dropna=True):
    # load mohr-smith datsaet
    if filename is None:
        filename = "Mohr-Smith_2017.csv"
        #filename = "Mohr-Smith_2017_Completa.csv"

    folderpath =Path(__file__).parent.absolute()
    filepath = folderpath / filename
    df = pd.read_csv(filepath)
    if dropna:
        df.dropna(inplace=True)

    y_columns = ["goodOB","EM","SUB","LUM"]
    y = df[y_columns]
    x_columns = [ 'umag', 'gmag', 'rmag',
                  'imag', 'Hamag', 'Jmag', 'Hmag', 'Kmag',
                  # 'W1mag', 'W2mag', 'W3mag','W4mag',
                  ]
    x = df[x_columns]
    metadata_columns = set(df.columns).difference(set(x_columns).union(set(y_columns)))
    # metadata_columns  = ["VPHAS-OB1","RAJ2000","DEJ2000",  'logTeff', 'A0', 'Rv', 'mu',
    #                     'chi2']
    metadata = df[metadata_columns]

    if binary:
        y = df["EM"] ==1
    return x,y,metadata

datasets = {"liu2017" : load_liu,
            "mohr":load_mohr}

def get_coefficients(dataset:str):

    coefficients = {'umag': ['VPHAS', 'u', 3607.7],
               'gmag': ['VPHAS', 'g', 4679.5],
               'rmag': ['VPHAS', 'r', 6242.1],
               'imag': ['VPHAS', 'i', 7508.5],
               'Hamag': ['VPHAS', 'Ha', 6590.8],
               'Jmag': ['2MASS', 'J', 12350.0],
               'Hmag': ['2MASS', 'H', 16620.0],
               'Kmag': ['2MASS', 'K', 21590.0],
               'W1mag': ['WISE', 'W1', 34000.0, 0.18],
               'W2mag': ['WISE', 'W2', 46000.0, 0.16]}
    if dataset in datasets.keys():
        coefficients = {k:v[2] for k,v in coefficients.items() }
        return coefficients