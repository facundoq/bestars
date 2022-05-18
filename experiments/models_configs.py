def tf_memory_growth():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
tf_memory_growth()

from typing import List
import joblib

from importlib.resources import path
import pathlib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from tensorflow import keras
from tqdm.keras import TqdmCallback
from .base import ModelConfig


class KerasModelConfig(ModelConfig):
    def load(self, path: pathlib.Path):
        return keras.models.load_model(path)
    def save(self, path: pathlib.Path, model):
        return keras.models.save_model(model,path) 

class SKLearnModelConfig(ModelConfig):
    def load(self, path: pathlib.Path):
        return joblib.load(path)
    def save(self, path: pathlib.Path, model):
        return joblib.dump(model, open(path, 'wb'))

class KerasDenseConfig(KerasModelConfig):
    def __init__(self,layer_sizes:List[int],epochs=500) -> None:
        self.layer_sizes=layer_sizes
        self.epochs=epochs
        
    def generate(self,input_shape):
        print(self,input_shape)
        k = len(input_shape)    
        assert (k==1)
        layers = [keras.layers.InputLayer(input_shape=input_shape)] + [keras.layers.Dense(n,activation="relu") for n in self.layer_sizes] + [keras.layers.Dense(1,activation="sigmoid")]    
        model = keras.models.Sequential(layers)
        model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit2=model.fit
        
        def new_fit(x,y,**kwargs):
            model.fit2(x,y,epochs=self.epochs,batch_size=512,verbose=False, callbacks=[TqdmCallback(verbose=0)], **kwargs)
        model.fit = new_fit
        model.id = self.id()
        return model
    def id(self):
        layers = ",".join(map(str,self.layer_sizes))
        return f"DenseNN({layers},e={self.epochs})"


class RandomForestConfig(SKLearnModelConfig):
    def __init__(self,max_depth:int) -> None:
        self.max_depth=max_depth
    def generate(self,input_shape):
        model = RandomForestClassifier(max_depth=self.max_depth, random_state=0)
        model.id  = "RF"
        return model
    def id(self):
        return f"RF(md={self.max_depth})"
    

class GradientBoostingConfig(SKLearnModelConfig):
    def __init__(self,max_depth:int,n_estimators:int,lr:float) -> None:
        self.max_depth=max_depth
        self.n_estimators=n_estimators
        self.lr=lr

    def generate(self,input_shape):
        model =  GradientBoostingClassifier(n_estimators=self.n_estimators, learning_rate=self.lr,max_depth=self.max_depth, random_state=0)
        model.id = self.id()
        return model
    def id(self):
        return f"GB(n_e={self.n_estimators},md={self.max_depth},lr={self.lr})"


