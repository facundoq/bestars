from pandas import DataFrame
import numpy as np
from typing import Dict


def q_indices3(df:DataFrame, coefficients:Dict[str, float]):

    column_names=df.columns.name
    i= column_names.index('umag')
    cols= column_names[i:]
    """
    Calculo el parametro Q fotometrico tomando magnitudes de a 3:
    Q= (m1 - m2) - (m2 - m3) * (r1 - r2) / (r2 - r3)
    donde m1, m2, m3 son las magnitudes aparentes en tres filtros
    r1, r2, r3 son los coeficientes A_lambda / Av para cada filtro, respectivamente
    """

    for c in cols[:-2]:
        # busca las columnas del mismo sistema (VPHAS/2MASS/WISE)
        i= np.where(np.array(cols) == c)[0][0]
        r1= coefficients[c]
        for cc in cols[i+1:-1]:
            j= np.where(np.array(cols) == cc)[0][0]
            r2= coefficients[cc]
            aux= r1 - r2
            for ccc in cols[j+1:]:
                r3= coefficients[ccc]
                coef= aux / (r2 - r3)
                df['Q_' + c + '_' + cc + '_' + ccc] = df[c] - df[cc] - coef * (df[cc] - df[ccc])

def q_indices4(df:DataFrame, coefficients:Dict[str,float]):
    #%%
    """
    Calculo el parametro Q fotometrico tomando magnitudes de a 4:
    Q= (m1 - m2) - (m3 - m4) * (r1 - r2) / (r3 - r4)
    donde m1, m2, m3, m4 son las magnitudes aparentes en tres filtros
    r1, r2, r3, r4 son los coeficientes A_lambda / Av para cada filtro,
    respectivamente
    """
    column_names=df.columns.name
    i = column_names.index('umag')
    cols= column_names[i:]

    for c in cols[:-3]:

        i= np.where(np.array(cols) == c)[0][0]
        r1= coefficients[c]
        for cc in cols[i+1:-2]:
            j= np.where(np.array(cols) == cc)[0][0]
            r2= coefficients[cc]
            aux= r1 - r2
            for ccc in cols[j+1:-1]:
                k= np.where(np.array(cols) == ccc)[0][0]
                r3= coefficients[ccc]
                for cccc in cols[k+1:]:
                    r4= coefficients[cccc]
                    coef= aux / (r3 - r4)
                    df['Q_' + c + '_' + cc + '_' + ccc + '_' + cccc] = df[c] - df[cc] - coef * (df[ccc] - df[cccc])




if __name__ == '__main__':
    coefficients = {"u":0.1,"v":0.2,"w":0.3,"g":0.4,"h":0.5}
    column_names = ["u","v","w","g","h"]
    x = np.array([[1,2,3,4,5]])
    x = x.repeat(4,0)
    coefficients = np.array([0.1,0.2,0.3,0.4,0.5])
    q,q_names = calculate_q(x,coefficients,column_names=column_names,combinations=3)
    print(q.shape,q,q_names)
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