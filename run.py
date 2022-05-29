#!/bin/env python3


import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# import tensorflow as tf
# tf.get_logger().setLevel(logging.WARNING)
# logging.getLogger("tensorflow").setLevel(logging.WARNING)

from experiments.cross_dataset import CrossDatasetAccuracy
from experiments.feature_selection import BinaryFeatureSelection
from experiments.statistics import ClassFeaturesDistribution,ClassDistributionComparison, CategoricalFeaturesDistribution, FeatureDistributions,OutlierDetectionTukey,OutlierDetectionNormalConfidenceInterval,MissingValues,FeatureCorrelations,ReducedQ, CorrelationMatrix
from experiments.evaluate import EvaluateClassifiers,DetermineMinimumTrainingSet

# jupyter nbconvert --to pdf --execute "Exploratory analysis.ipynb" --output "all_em.pdf"
from experimenter import Experiment

if __name__ == '__main__':

    experiments = {"all":[
                   # CrossDatasetAccuracy(),
                   ClassDistributionComparison(),
                   ClassFeaturesDistribution(),
                   MissingValues(),
                   FeatureCorrelations(),
                   CategoricalFeaturesDistribution(),
                   FeatureDistributions(),
                   ReducedQ(),
                   CorrelationMatrix(),
                    #BinaryFeatureSelection(),
                #    OutlierDetectionTukey(),
                #    OutlierDetectionNormalConfidenceInterval(),
                   EvaluateClassifiers(),
                   DetermineMinimumTrainingSet(),

                   ]
                   }

    Experiment.main(experiments)