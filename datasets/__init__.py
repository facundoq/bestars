
from . import liu,mohr_smith,mcswain,hou
from .base import map_y_em

dataset_names =  {"liu":liu,
                 "mohr_smith":mohr_smith,
                 "mcswain": mcswain,
                 "hou":hou}
dataset_names_all = dataset_names.copy()

# add it afterwards to avoid recursion
from . import all_em
dataset_names_all["all_em"] = all_em


