import numpy as np
import itertools

def q3_coef(v1:np.ndarray,v2:np.ndarray,v3:np.ndarray,coef:float):
    return v1 - v2 - coef * (v2 - v3)

def q4_coef(v1:np.ndarray,v2:np.ndarray,v3:np.ndarray,v4:np.ndarray,coef:float):
    return v1 - v2 - coef * (v3 - v4)

def q3(v1:np.ndarray,v2:np.ndarray,v3:np.ndarray,r1:float,r2:float,r3:float):
    coef = (r1-r2)/r3
    return q3_coef(v1,v2,v3,coef)

def q4(v1:np.ndarray,v2:np.ndarray,v3:np.ndarray,v4:np.ndarray,r1:float,r2:float,r3:float,r4:float):
    coef = (r1-r2)/(r3-r4)
    return q4_coef(v1,v2,v3,v4,coef)

def calculate_q(magnitudes:np.ndarray,coefficients:np.ndarray,names:[str]=None,combinations=3):
    n,n_cols=magnitudes.shape
    assert combinations in [3,4], f"Only q indices with combinations of 3 or 4 magnitudes supported (received combinations={combinations})"
    if combinations==3:
        qfunction=q3
    else:
        qfunction=q4

    column_indices = range(n_cols)
    if names is None:
        names = [f"{i}" for i in column_indices]
    else:
        assert len(names) == n_cols, f"Length of names {len(names)} must match number of columns {n_cols}"

    index_combinations = list(itertools.combinations(column_indices,combinations))
    n_qindices= len(index_combinations)
    qfeatures = np.zeros((n,n_qindices))

    for i,combination in enumerate(index_combinations):
        combination_magnitudes = [magnitudes[:,j] for j in combination]
        combination_coefficients = [coefficients[j] for j in combination]
        qfeatures[:,i] = qfunction(*combination_magnitudes,*combination_coefficients)

    return qfeatures

if __name__ == '__main__':
    x = np.array([[1,2,3,4,5]])
    x = x.repeat(4,0)
    coefficients=np.array([0.1,0.2,0.3,0.4,0.5])
    q=calculate_q(x,coefficients,combinations=4)


