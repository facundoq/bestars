import os
from pathlib import Path
from datetime import datetime
from typing import List
import pandas as pd
import abc
import pathlib
import datasets
import sys
from pathlib import Path
from experimenter import Experiment
import matplotlib.pyplot as plt
import numpy as np
import qfeatures

current=Path(__file__)

folderpath = current.parent / "../plots/"

import abc
class Feature(abc.ABC):
    @abc.abstractmethod
    def calculate(self,x:pd.DataFrame,dataset_name:str)->pd.DataFrame:
        pass

class Magnitude(Feature):
    def calculate(self, x: pd.DataFrame, dataset_name: str)->pd.DataFrame:
        return x
    def __repr__(self) -> str:
        return "mag"

class ColorFeatures(Feature):
    def __init__(self,combinations) -> None:
        self.combinations=combinations
    def __repr__(self) -> str:
        colors = ",".join(self.names())
        return f"Colors({colors})"
    def names(self)->List[str]:
        return list([f"{a}-{b}" for a,b in self.combinations])
    def calculate(self, x: pd.DataFrame, dataset_name: str)->pd.DataFrame:
        values = [x[a]-x[b] for a,b in self.combinations]
        values = pd.concat(values,axis=1)
        dict_names = {i:v for i,v in enumerate(self.names())}
        values = values.rename(columns=dict_names)
        return values

class PosterFeature(Feature):
    def __repr__(self) -> str:
        return f"Poster"
    def calculate(self, x: pd.DataFrame, dataset_name: str)->pd.DataFrame:
        standard_colors = ColorFeatures([("u","g"), ("r","i"), ("r", "Ha")])
        q4_poster = QFeature(4,False,["u","g","r","i","Ha"])
        c = UnionFeature([Magnitude(),standard_colors,SubsetFeature(q4_poster,["ugri","ugrHa"])])

class QFeature(Feature):
    def __init__(self,q:int,by_system:str=False,subset=None) -> None:
        assert q in [3,4]
        self.q=q
        self.by_system=by_system
        self.subset=subset

    def __repr__(self) -> str:
        by_system = ['by_system'] if self.by_system else []
        if self.subset is None:
            subset = []
        else:
            subset = ["vars=" + (",".join(self.subset))]
        parameters = by_system+subset
        if len(parameters)>0:
            values = ",".join(parameters)
            parameters_str = f"({values})"
        else:
            parameters_str = ""

        return f"q{self.q}{parameters_str}"

    def calculate(self, x: pd.DataFrame, dataset_name: str)->pd.DataFrame:
        dataset_module = datasets.datasets_by_name_all[dataset_name]
        if not self.subset is None:
            x = x[self.subset]
        coefficients = dataset_module.coefficients
        systems = dataset_module.systems     
        return qfeatures.calculate_df(x, coefficients, systems, combination_size=self.q,by_system=self.by_system)

class SubsetFeature(Feature):
    def __init__(self,feature:Feature,subset:List[str]) -> None:
        self.subset=subset
        self.feature=feature

    def __repr__(self) -> str:
        subset_str = ",".join(self.subset)
        return f"Subset({subset_str})"
    
    def calculate(self, x: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        feature = self.feature.calculate(x,dataset_name)
        return feature[self.subset]
        

class UnionFeature(Feature):
    def __init__(self,features:List[Feature]) -> None:
        self.features=features

    def __repr__(self) -> str:
        feature_str = ",".join(map(str,self.features))
        return f"U({feature_str})"
    
    def calculate(self, x: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        values = [f.calculate(x,dataset_name) for f in self.features]
        return pd.concat(values,axis=1)

class ModelConfig(abc.ABC):
    
    @abc.abstractmethod
    def generate(self,input_shape):
        pass

    @abc.abstractmethod
    def load(self,path:pathlib.Path):
        pass
    @abc.abstractmethod
    def save(self,path:pathlib.Path,model):
        pass

    @abc.abstractmethod
    def id(self)->str:
        pass
    def __repr__(self) -> str:
        return self.id()
    
class DatabaseConfig:
    def __init__(self,id:str,train_size_percentage:float,class_feature:str) -> None:
        self.id=id
        self.train_size_percentage = train_size_percentage
        self.class_feature=class_feature
    def __repr__(self) -> str:
        return f"{self.id}(tp={self.train_size_percentage:.2f},{self.class_feature})"

class TrainConfig:
    def __init__(self,model_config:ModelConfig,database:DatabaseConfig,feature:Feature) -> None:
        self.model_config=model_config
        self.database=database
        self.feature=feature
    def __repr__(self)->str:
        return f"{self.model_config}_{self.database}_{self.feature}"
        

class BeExperiment(Experiment):
    def __init__(self):
        super().__init__(folderpath)

class StarExperiment(BeExperiment):

    def models_path(self):
        return self.base_folderpath / "models" 
    
    def results_path(self):
        return self.base_folderpath / "results" 
    def save_close_fig(self,filename,extra_artists=None):
        plt.tight_layout()
        plt.savefig(self.folderpath / filename,bbox_extra_artists=extra_artists,bbox_inches="tight")
        plt.close()   


