
from . import liu,mohr_smith,mcswain,hou,aidelman
from .base import map_y_em,map_y_be

datasets_by_name_small =  {"liu":liu,
                 "mohr_smith":mohr_smith,
                 "mcswain": mcswain,
                 "hou":hou,
                 }
datasets_by_name = datasets_by_name_small.copy()
datasets_by_name.update({"aidelman":aidelman})

datasets_by_name_all = datasets_by_name.copy()

# add it afterwards to avoid recursion
from . import all_em
datasets_by_name_all["all_em"] = all_em


