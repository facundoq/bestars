import pandas as pd
from . import datasets_by_name
from . import base

coefficients=base.coefficients
systems=base.systems

def load(dropna=True, dataset_names=datasets_by_name.keys(),exclude=[],verbose=False):
    x_all,y_all=pd.DataFrame(),pd.DataFrame()
    m=pd.DataFrame()
    for name in dataset_names:
        if name in exclude:
            continue
        x,y,_ = datasets_by_name[name].load(dropna=dropna,verbose=verbose)
        y = base.map_y_em(y,name)
        x_all=x_all.append(x)
        y_all=y_all.append(y)

    x_all=x_all.reset_index(drop=True)
    y_all=y_all.reset_index(drop=True)
    return x_all,y_all,m