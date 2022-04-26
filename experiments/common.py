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


import datasets
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from .base import Experiment
import qfeatures
from preprocessing import outliers


class StarExperiment(Experiment):


    def save_close_fig(self,filename):
        plt.tight_layout()
        plt.savefig(self.plot_folderpath / filename)
        plt.close()

def calculate_features(x,dataset_name,feature_id=None):
    dataset_module = datasets.datasets_by_name_all[dataset_name]
    coefficients = dataset_module.coefficients

    coefficients_np = np.array([coefficients[k] for k in x.columns])
    systems = [dataset_module.systems[k] for k in x.columns]

    if feature_id is None:
        return x
    elif feature_id == "q3":
        return qfeatures.calculate(x, coefficients_np, x.columns, systems, combination_size=3)
    elif feature_id == "q4": 
        return qfeatures.calculate(x, coefficients_np, x.columns, systems, combination_size=4)       
    else:
        raise ValueError(f"Unsupported feature: {feature_id}")