from .common import *

class CrossDatasetAccuracy(StarExperiment):
    def description(self) -> str:
        return "Evaluate several training/testing dataset combinations."
    def run(self):

        testing_datasets = ["liu"]
        # TODO all but testing
        training_datasets = ["mohr_smith","hou","mcswain"]

        x_train,y_train,metadata_train = datasets.all_em.load(dataset_names=training_datasets)
        x_test,y_test,metadata_train = datasets.all_em.load(dataset_names=testing_datasets)
        #print(x_train.shape,x_test.shape)
        # TODO train/test with various classifiers
