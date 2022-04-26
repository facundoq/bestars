
from typing import List
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

def preprocess(df:pd.DataFrame,filename,x_columns:List[str],y_columns:List[str],dropna:bool,verbose=False,dtypes={},fill_values=None,):
    if not fill_values is None:
        for column,value in fill_values.items():
            if verbose:
                print(f"Warning: Filling missing values for {column} with {value}")
            df[column] = df[column].fillna(value)

    y = df[y_columns].copy()
    x = df[x_columns].copy()

    metadata_columns = list(set(df.columns).difference(set(x_columns).union(set(y_columns))))
    metadata = df[metadata_columns].copy()

    if dropna:
        n = len(x)

        nan_indices_x = np.where(pd.isnull(x).any(axis=1))
        nan_indices_y = np.where(pd.isnull(y).any(axis=1))
        nan_indices = np.union1d(nan_indices_x,nan_indices_y)
        
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
