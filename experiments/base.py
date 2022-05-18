import os
from pathlib import Path
from datetime import datetime
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
    def calculate(self,x:pd.DataFrame,dataset_name:str):
        pass
class Magnitude(Feature):
    def calculate(self, x: pd.DataFrame, dataset_name: str):
        return x
    def __repr__(self) -> str:
        return "mag"
class QFeature(Feature):
    def __init__(self,q:int,by_system:str=False) -> None:
        assert q in [3,4]
        self.q=q
        self.by_system=by_system

    def __repr__(self) -> str:
        by_system = '(bysystem)' if self.by_system else ''
        return f"q{self.q}{by_system}"

    def calculate(self, x: pd.DataFrame, dataset_name: str):
        dataset_module = datasets.datasets_by_name_all[dataset_name]
        coefficients = dataset_module.coefficients
        systems = dataset_module.systems     
        return qfeatures.calculate_df(x, coefficients, systems, combination_size=self.q,by_system=self.by_system)


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
    def save_close_fig(self,filename):
        plt.tight_layout()
        plt.savefig(self.folderpath / filename)
        plt.close()   