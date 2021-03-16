
from . import liu,mohr_smith,mcswain,hou
from .base import map_y_em

dataset_names = {"liu":liu,
                 "mohr_smith":mohr_smith,
                 "mcswain": mcswain,
                 "hou":hou}


from . import all_em

# add it afterwards to avoid recursion
dataset_names["all_em"] = all_em

