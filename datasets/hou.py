
from . import base

coefficients=base.coefficients
systems=base.systems

default_filename = "Hou2016_VPHAS-SDSS-IPHAS-2MASS.csv"



def load(filename=default_filename,dropna=True):
    y_columns = ["objtype_Hou"]


    return base.load(filename,base.twomass_x_columns,y_columns,dropna)
