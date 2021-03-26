from experiments.cross_dataset import CrossDatasetAccuracy
from experiments.statistics import DatasetClassDistributionEM,DatasetClassDistribution

if __name__ == '__main__':

    experiments = [CrossDatasetAccuracy(),
                   DatasetClassDistributionEM(),
                   DatasetClassDistribution(),
                   ]
    for e in experiments:
        e.run()