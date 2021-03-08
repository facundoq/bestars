from __future__ import annotations
import numpy as np
import itertools




class Magnitudes:
    @staticmethod
    def merge(ms:[Magnitudes]):
        magnitudes = np.concatenate([m.magnitudes for m in ms],axis=1)
        coefficients = np.concatenate([m.coefficients for m in ms],axis=0)
        column_names = [name for m in ms for name in m.column_names]
        systems = [system  for m in ms for system  in m.systems]
        return Magnitudes(magnitudes,coefficients,column_names,systems)

    @staticmethod
    def check_dims(magnitudes: np.ndarray, coefficients: np.ndarray, column_names: [str], systems: [str]):
        n, n_cols = magnitudes.shape
        assert len(
            column_names) == n_cols, f"Length of names ({len(column_names)}) must match number of columns ({n_cols})"
        assert len(
            coefficients) == n_cols, f"Length of coefficients {len(coefficients)} must match number of columns ({n_cols})"
        assert len(systems) == n_cols, f"Length of systems {len(systems)} must match number of columns ({n_cols})"

    def __init__(self,magnitudes:np.ndarray, coefficients:np.ndarray, column_names:[str], systems:[str]):
        Magnitudes.check_dims(magnitudes,coefficients,column_names,systems)
        self.magnitudes=magnitudes
        self.coefficients=coefficients
        self.column_names=column_names
        self.systems=systems

    def split_by_system(self):
        unique_systems = list(set(systems))
        magnitudes=[]
        for system in unique_systems:
            system_indices = [i for i, v in enumerate(systems) if v == system]
            system_magnitudes = self.magnitudes[:, system_indices]
            system_coefficients = self.coefficients[system_indices]
            system_column_names = [self.column_names[i] for i in system_indices]
            system_systems = [self.systems[i] for i in system_indices]
            magnitudes.append(Magnitudes(system_magnitudes,system_coefficients,system_column_names,system_systems))
        return magnitudes

    def get_combinations(self,combination_size:int):
        column_indices = list(range(self.magnitudes.shape[1]))
        return list(itertools.combinations(column_indices, combination_size))

def q3_coef(v1:np.ndarray,v2:np.ndarray,v3:np.ndarray,coef:float):
    return v1 - v2 - coef * (v2 - v3)

def q4_coef(v1:np.ndarray,v2:np.ndarray,v3:np.ndarray,v4:np.ndarray,coef:float):
    return v1 - v2 - coef * (v3 - v4)

def q3(v1:np.ndarray,v2:np.ndarray,v3:np.ndarray,r1:float,r2:float,r3:float):
    coef = (r1-r2)/r3
    return q3_coef(v1,v2,v3,coef),coef

def q4(v1:np.ndarray,v2:np.ndarray,v3:np.ndarray,v4:np.ndarray,r1:float,r2:float,r3:float,r4:float):
    coef = (r1-r2)/(r3-r4)

    return q4_coef(v1,v2,v3,v4,coef),coef

def get_qfunction(combination_size:int, coefficients:np.ndarray):
    error_message = f"Only q indices with combinations of 3 or 4 magnitudes supported (received combinations={combination_size})"
    assert combination_size in [3, 4], error_message

    if combination_size==3:
        qfunction = q3
    else:
        n_unique = len(np.unique(coefficients))
        repeated_values = n_unique< len(coefficients)
        assert not repeated_values, f"Coefficient values cannot repeat, received: {coefficients}"
        qfunction = q4
    return qfunction


def calculate_q(m:Magnitudes,combination_size:int):
    n,n_cols=m.magnitudes.shape
    index_combinations = m.get_combinations(combination_size)
    qfunction = get_qfunction(combination_size,m.coefficients)
    q_names = [ "_".join([m.column_names[i] for i in combination]) for combination in index_combinations]
    q_systems = ["_".join(set([m.systems[i] for i in combination])) for combination in index_combinations]
    n_qindices= len(index_combinations)
    q_magnitudes = np.zeros((n,n_qindices))
    q_coefficients = np.zeros(n_qindices)
    for i,combination in enumerate(index_combinations):
        combination_magnitudes = [m.magnitudes[:,j] for j in combination]
        combination_coefficients = [m.coefficients[j] for j in combination]
        q_magnitudes[:,i],q_coefficients[i] = qfunction(*combination_magnitudes,*combination_coefficients)

    return Magnitudes(q_magnitudes,q_coefficients,q_names,q_systems)

def calculate(magnitudes:np.ndarray, coefficients:np.ndarray, column_names:[str], systems:[str], combination_size=3,by_system=False):
    m = Magnitudes(magnitudes,coefficients,column_names,systems)
    if by_system:
        m_by_system=m.split_by_system()
    else:
        m_by_system=[m]
    q_by_system = [calculate_q(m, combination_size) for m in m_by_system]
    return Magnitudes.merge(q_by_system)

#
# def calculate_by_system(magnitudes:np.ndarray, coefficients:np.ndarray, column_names:[str], systems:[str],qfunction:Callable,combinations=3):
#     n, n_cols = magnitudes.shape
#     column_indices = range(n_cols)
#     unique_systems = list(set(systems))
#     magnitudes_by_system={}
#     for system in unique_systems:
#         system_indices = [i for i, v in enumerate(systems) if v == system]
#         system_magnitudes = magnitudes[:, system_indices]
#         system_coefficients = coefficients[system_indices]
#         system_column_names = [column_names[i] for i in system_indices]
#         system_column_indices = range(system_magnitudes.shape[1])
#
#         calculate_q(system_magnitudes,system_coefficients,system_column_names,index_combinations,qfunction)
#


if __name__ == '__main__':
    column_names = ["u","v","w","g","h"]
    systems=["VPHAS","VPHAS","ASD","UMASS","UMASS"]
    m = np.array([[1, 2, 3, 4, 5]])
    m = m.repeat(4, 0)
    coefficients = np.array([0.1,0.2,0.3,0.4,0.5])

    q = calculate(m, coefficients, column_names, systems, combination_size=3)
    print(q.magnitudes.shape,"\n",q.column_names,"\n",q.systems)



