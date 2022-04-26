#!/bin/env python3

from experiments.cross_dataset import CrossDatasetAccuracy
from experiments.feature_selection import BinaryFeatureSelection
from experiments.statistics import DatasetClassDistributionEM,DatasetClassDistribution,OutlierDetectionTukey,OutlierDetectionNormalConfidenceInterval
from experiments.evaluate import EvaluateClassifiers

# jupyter nbconvert --to pdf --execute "Exploratory analysis.ipynb" --output "all_em.pdf"


if __name__ == '__main__':

    experiments = [
                   # CrossDatasetAccuracy(),
                   # DatasetClassDistributionEM(),
                   # DatasetClassDistribution(),
                    #BinaryFeatureSelection(),
                #    OutlierDetectionTukey(),
                #    OutlierDetectionNormalConfidenceInterval(),
                   EvaluateClassifiers()

                   ]

    for e in experiments:
        e.run()