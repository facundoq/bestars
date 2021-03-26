import pandas as pd
from . import datasets_by_name
from . import base

coefficients=base.coefficients
systems=base.systems

def load(dropna=True, dataset_names=datasets_by_name.keys(),exclude=[],verbose=False):
    x_all,y_all=pd.DataFrame(),pd.DataFrame()
    metadata_columns = ["RAJ2000","DEJ2000"]
    metadata_all=pd.DataFrame(columns=metadata_columns)
    for name in dataset_names:
        if name in exclude:
            continue
        x,y,metadata = datasets_by_name[name].load(dropna=dropna,verbose=verbose)
        y = base.map_y_em(y,name)
        x_all=x_all.append(x)
        y_all=y_all.append(y)
        metadata_all=metadata_all.append(metadata[metadata_columns])

    x_all=x_all.reset_index(drop=True)
    y_all=y_all.reset_index(drop=True)
    metadata_all = metadata_all.reset_index(drop=True)
    return x_all,y_all,metadata_all