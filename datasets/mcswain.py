from . import base

coefficients=base.coefficients
systems=base.systems

default_filename = "McSwain2005-2009_VPHAS-2MASS.csv"

def load(filename=default_filename,dropna=True,verbose=False):
    y_columns = ["Code"]

    return base.load(filename,base.twomass_x_columns,y_columns,dropna,dropna,verbose=verbose)
