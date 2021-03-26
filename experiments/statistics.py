from .common import *


class DatasetClassDistributionEM(Experiment):
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



class DatasetClassDistribution(Experiment):
    def description(self) -> str:
        return "Plot histograms of distribution of classes for each dataset"

    def run(self):
        names = datasets.datasets_by_name_all
        n_datasets = len(names)
        f,axes=plt.subplots(1,n_datasets)
        for i,(name,dataset_module) in enumerate(names.items()):
            x,y,metadata = dataset_module.load()
            ax = axes[i]
            if len(y.columns)==1:
                print(y)
                y.plot.bar(ax=ax)
                ax.set_xlabel(name)
                if i==0:
                    ax.set_ylabel("Samples")
        plt.savefig(self.plot_folderpath / "histograms.png")
