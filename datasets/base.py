
import numpy as np
from pathlib import Path
import pandas as pd

twomass_x_columns = [ 'umag', 'gmag', 'rmag',
                      'imag', 'Hamag', 'Jmag', 'Hmag', 'Kmag']
allwise_x_columns = twomass_x_columns + ["W1mag","W2mag"]


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


def load(filename:str,x_columns:[str],y_columns:[str],dropna:bool):

    folderpath =Path(__file__).parent.absolute()
    filepath = folderpath / filename
    df = pd.read_csv(filepath)
    y = df[y_columns].copy()
    x = df[x_columns].copy()

    metadata_columns = set(df.columns).difference(set(x_columns).union(set(y_columns)))
    metadata = df[metadata_columns].copy()

    if dropna:
        n = len(x)

        nan_indices_x = np.where(pd.isnull(x).any(axis=1))
        nan_indices_y = np.where(pd.isnull(x).any(axis=1))
        nan_indices = np.union1d(nan_indices_x,nan_indices_y)

        x.drop(nan_indices,inplace=True)
        y.drop(nan_indices,inplace=True)
        metadata.drop(nan_indices,inplace=True)

        new_n=len(x)

        if new_n<n:
            print(f"Warning: dropped {n-new_n} rows with incomplete values. ")
            print(f"Rows (original):   {n}")
            print(f"Rows (after drop): {new_n}")

    return x,y,metadata
