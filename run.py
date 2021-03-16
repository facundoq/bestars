import abc

import datasets
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from experiments.base import Experiment

import datasets

class DatasetClassDistribution(Experiment):
    def description(self) -> str:
        return "Plot histograms of distribution of classes for each dataset"

    def run(self):
        names = datasets.datasets_by_name_all
        n_datasets = len(names)
        f,axes=plt.subplots(1,n_datasets,sharey=True,sharex=True)
        for i,(name,dataset_module) in enumerate(names.items()):
            x,y,metadata = dataset_module.load()
            y = datasets.map_y_em(y,name)
            ax = axes[i]
            y.hist(ax=ax)
            ax.set_xlabel(name)
            if i==0:
                ax.set_ylabel("Samples")
        plt.savefig(self.plot_folderpath / "histograms.png")



class CrossDatasetAccuracy(Experiment):
    def description(self) -> str:
        return "Evaluate several training/testing dataset combinations."
    def run(self):

        testing_datasets = ["liu"]
        # TODO all but testing
        training_datasets = ["mohr_smith","hou","mcswain"]

        x_train,y_train,metadata_train = datasets.all_em.load(dataset_names=training_datasets)
        x_test,y_test,metadata_train = datasets.all_em.load(dataset_names=testing_datasets)
        print(x_train.shape,x_test.shape)
        # TODO train/test with various classifiers






if __name__ == '__main__':

    experiments = [CrossDatasetAccuracy(),
                   DatasetClassDistribution(),
                   ]
    for e in experiments:
        e.run()