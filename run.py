from experiments.cross_dataset import CrossDatasetAccuracy
from experiments.statistics import DatasetClassDistribution

if __name__ == '__main__':

    experiments = [CrossDatasetAccuracy(),
                   DatasetClassDistribution(),
                   ]
    for e in experiments:
        e.run()