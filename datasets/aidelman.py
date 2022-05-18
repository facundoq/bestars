
from . import base
from pathlib import Path
coefficients=base.coefficients
systems=base.systems
import pandas as pd

default_filename = "data/Concatenadas_sinRepes_dropnan_LAMOST.csv"



def load(filename=default_filename,dropna_x=True,dropna_y=False,fillna_classes=True,verbose=False,fill_values=None):
   y_columns = ["Be","EM"]
   str_cols = ['ID', 'SpT', 'Type',  'obsid', 'objtype', 'class', 'subclass', 
    'B-TS1', 'B-TS2','B-TS', 'EM1', 'GroupID', 'GroupSize']
    # df = pd.read_csv(filename,dtype=str_cols)
    # df["Be"].fillna(0, inplace=True)
   #  str_cols2 = ["EM1",  "EMobj",  "BeC1",  "EM2",  "Be_EM2",  "BeC2",  "BeC"]
   int_cols = []
   str_cols = {k:str for k in str_cols}
   int_cols = {k:int for k in int_cols}
   dtypes = {**str_cols,**int_cols}
   
   # fill_values = {"EM":0,"Be":0}
   folderpath =Path(__file__).parent.parent.absolute()
   filepath = folderpath / filename
   df = pd.read_csv(filepath,dtype=dtypes)
   if fillna_classes:
      for column in y_columns:
         print(f"Warning: replacing nan in {column} with 0.")
         df[column] = df[column].fillna(0)
      df = df.astype({"EM":'int'})
   
   return base.preprocess(df,filename,base.twomass_x_columns,y_columns,dropna_x,dropna_y,verbose=verbose,dtypes=dtypes,fill_values=fill_values)
