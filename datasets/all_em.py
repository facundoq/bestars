import pandas as pd
from . import dataset_names
from . import base

coefficients=base.coefficients
systems=base.systems

def load(dropna=True):
    x_all,y_all=pd.DataFrame(),pd.DataFrame()
    m=pd.DataFrame()
    for name in dataset_names:
        x,y,_ = dataset_names[name].load(dropna=dropna)
        y = base.map_y_em(y,name)
        x_all=x_all.append(x)
        y_all=y_all.append(y)

    return x_all,y_all,m