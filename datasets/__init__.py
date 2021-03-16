
from . import liu,mohr_smith,mcswain,hou
from .base import map_y_em

datasets_by_name =  {"liu":liu,
                 "mohr_smith":mohr_smith,
                 "mcswain": mcswain,
                 "hou":hou}
datasets_by_name_all = datasets_by_name.copy()

# add it afterwards to avoid recursion
from . import all_em
datasets_by_name_all["all_em"] = all_em


