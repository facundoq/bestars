
from typing import List
import numpy as np
from pathlib import Path
import pandas as pd

twomass_x_columns = [ 'u', 'g', 'r',
                      'i', 'Ha', 'J', 'H', 'K']
allwise_x_columns = twomass_x_columns + ["W1","W2"]


coefficients = {'u': 4.39,
                'g':  3.30,
                'r':  2.31,
                'i':  1.71,
                'Ha': 2.14,  # valor interpolado
                'J':  0.72,
                'H':  0.46,
                'K':  0.306,
                'W1': 0.18,
                'W2': 0.16}

systems =        {'u': 'VPHAS',
                  'g': 'VPHAS',
                  'r': 'VPHAS',
                  'i': 'VPHAS',
                  'Ha':'VPHAS',
                  'J': '2MASS',
                  'H': '2MASS',
                  'K': '2MASS',
                  'W1':'WISE',
                  'W2':'WISE',
                  }

rename_columns = {"rmag":"r",
                    "imag":"i",
                    "gmag":"g",
                    "umag":"u",
                    "Hamag":"Ha",
                    "Hmag":"H",
                    "Jmag":"J",
                    "Kmag":"K",
                    "W1mag":"W1",
                    "W2mag":"W2",
                    }
  

def preprocess(df:pd.DataFrame,filename,x_columns:List[str],y_columns:List[str],dropna_x:bool=True,dropna_y:bool=False,verbose=False,dtypes={},fill_values=None,):
    if not fill_values is None:
        for column,value in fill_values.items():
            if verbose:
                print(f"Warning: Filling missing values for {column} with {value}")
            df[column] = df[column].fillna(value)
    df = df.rename(columns=rename_columns)
    y = df[y_columns].copy()
    x = df[x_columns].copy()

    metadata_columns = list(set(df.columns).difference(set(x_columns).union(set(y_columns))))
    metadata = df[metadata_columns].copy()

    if dropna_x or dropna_y:
        n = len(x)
        nan_indices = np.zeros(0)
        if dropna_x:
            nan_indices_x = np.where(pd.isnull(x).any(axis=1))
            nan_indices = np.union1d(nan_indices,nan_indices_x)
        if dropna_y:
            nan_indices_y = np.where(pd.isnull(y).any(axis=1))
            nan_indices = np.union1d(nan_indices,nan_indices_y)
        
        x.drop(nan_indices,inplace=True)
        y.drop(nan_indices,inplace=True)
        metadata.drop(nan_indices,inplace=True)

        new_n=len(x)

        if new_n<n and verbose:
            print(f"Warning loading data from {filename}:")
            print(f"Dropped {n-new_n} rows with missing values. ")
            print(f"Rows (original):   {n}")
            print(f"Rows (after drop): {new_n}")
        x=x.reset_index(drop=True)
        y=y.reset_index(drop=True)
        metadata=metadata.reset_index(drop=True)
    return x,y,metadata

def load(filename:str,x_columns:List[str],y_columns:List[str],dropna:bool,verbose=False,dtypes={},fill_values=None):
    folderpath =Path(__file__).parent.absolute()
    filepath = folderpath / filename
    df = pd.read_csv(filepath,dtype=dtypes)
    return preprocess(df,filename,x_columns,y_columns,dropna,verbose=verbose,dtypes=dtypes,fill_values=fill_values)
    

def map_y_be(y:pd.DataFrame,dataset_name:str):
    n = len(y)
    name="be"
    columns=[name]
    if dataset_name == "aidelman":
        y = pd.DataFrame(y["Be"].to_numpy(),columns=columns)
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")
    assert np.logical_or(y[name].to_numpy()==1,y[name].to_numpy()==0).all()


    return y

def map_y_em(y:pd.DataFrame,dataset_name:str):
    n = len(y)
    name="em"
    columns=[name]
    if dataset_name=="liu":
        y = pd.DataFrame(y["Halpha"].to_numpy(),columns=columns)
    elif dataset_name=="hou":
        y = pd.DataFrame(np.ones(n),columns=columns)
    elif dataset_name=="mohr_smith":
        y = pd.DataFrame(y["EM"].to_numpy(),columns=columns)
    elif dataset_name=="mcswain":
        indices = y["Code"].to_numpy()=="Be"
        indices = indices.astype(int)
        y = pd.DataFrame(indices,columns=columns)
    elif dataset_name == "aidelman":
        y = pd.DataFrame(y["EM"].to_numpy(),columns=columns)
    elif dataset_name == "all_em":
        return y    
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")
    assert np.logical_or(y[name].to_numpy()==1,y[name].to_numpy()==0).all()


    return y
