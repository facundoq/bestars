
from experiments.cross_dataset import CrossDatasetAccuracy
from experiments.statistics import DatasetClassDistributionEM,DatasetClassDistribution,OutlierDetectionTukey,OutlierDetectionNormalConfidenceInterval

# jupyter nbconvert --to pdf --execute "Exploratory analysis.ipynb" --output "all_em.pdf"


if __name__ == '__main__':

    experiments = [
                   # CrossDatasetAccuracy(),
                   # DatasetClassDistributionEM(),
                   # DatasetClassDistribution(),
                   OutlierDetectionTukey(),
                   OutlierDetectionNormalConfidenceInterval(),
                   ]

    for e in experiments:
        e.run()