import numpy as np
from sklearn.model_selection import train_test_split

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
