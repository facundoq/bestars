
from . import liu,mohr_smith

def common_columns(datasets:[str]):
    def lequal(l1,l2):
        l1=list(sorted((l1)))
        return len(l1) == len(l2) and sorted(l1)==sorted(l2)

    if lequal(datasets,["liu2017","mohr"]):
        return [ 'umag', 'gmag', 'rmag',
                 'imag', 'Hamag', 'Jmag', 'Hmag', 'Kmag',
                 ]
    else:
        raise ValueError(f"No entries  for dataset combination: {datasets}")

#
# def load_data(binary=False,dropna=True):
#     data=np.loadtxt("ob_features.csv",delimiter=",",skiprows=1)
#     id = data[:,0]
#     y = data[:,12]
#     if original:
#         x= data[:,4:12]
#     else:
#         x = data[:,13:]
#
#     if binary:
#         y[y!=1]=0
#         class_names = ["OB","EM"]
#     else:
#         class_names = ["OB","EM","Sub","Over","Multi"]
#
#     return (x,y,id),class_names




# datasets = {"liu2017" : liu.load,
#             "mohr":mohr_smith.load}
#
# def get_coefficients(dataset:str):
#     if dataset in datasets.keys():
#         coefficients = {k:v[2] for k,v in magnitude_info_mohr.items() }
#         return coefficients
#     else:
#         raise ValueError(f"Dataset {dataset} not supported")
#
# def get_systems(dataset:str):
#     if dataset =="mohr":
#         coefficients = {k:v[0] for k,v in magnitude_info_mohr.items() }
#         return coefficients
#     else:
#         raise ValueError(f"Dataset {dataset} not supported")

coefficients = {'umag': 4.39,
                'gmag':  3.30,
                'rmag':  2.31,
                'imag':  1.71,
                'Hamag': 2.14,  # valor interpolado
                'Jmag':  0.72,
                'Hmag':  0.46,
                'Kmag':  0.306,
                'W1mag': 0.18,
                'W2mag': 0.16
                }

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