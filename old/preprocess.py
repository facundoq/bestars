import numpy as np
from sklearn.model_selection import train_test_split



foto_sys = {'umag': ['VPHAS', 'u', 3607.7, 4.39],
                'gmag': ['VPHAS', 'g', 4679.5, 3.30],
                'rmag': ['VPHAS', 'r', 6242.1, 2.31],
                'imag': ['VPHAS', 'i', 7508.5, 1.71],
                'Hamag': ['VPHAS', 'Ha', 6590.8, 2.14],  # valor interpolado
                'Jmag': ['2MASS', 'J', 12350.0, 0.72],
                'Hmag': ['2MASS', 'H', 16620.0, 0.46],
                'Kmag': ['2MASS', 'K', 21590.0, 0.306],
                'W1mag': ['WISE', 'W1', 34000.0, 0.18],
                'W2mag': ['WISE', 'W2', 46000.0, 0.16]}


def load_data(binary=False,original=False,split=None):
    data=np.loadtxt("ob_features.csv",delimiter=",",skiprows=1)
    id = data[:,0]
    y = data[:,12]
    if original:
        x= data[:,4:12]
    else:
        x = data[:,13:]

    if binary:
        y[y!=1]=0
        class_names = ["OB","EM"]
    else:
        class_names = ["OB","EM","Sub","Over","Multi"]

    if split is None:
        return (x,y,id),class_names
    else:
        x_train,x_test,y_train,y_test,id_train,id_test= train_test_split(x,y,id,train_size=split,stratify=y,random_state=0)
        return (x_train,y_train,id_train),(x_test,y_test,id_test),class_names
