from .common import *
import abc
from scipy import stats

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
                y.plot.bar(ax=ax)
                ax.set_xlabel(name)
                if i==0:
                    ax.set_ylabel("Samples")
        plt.savefig(self.plot_folderpath / "histograms.png")




class OutlierDetection(Experiment):

    @abc.abstractmethod
    def detect_outliers(self,x:pd.DataFrame):
        pass

    def get_outliers_column(self,x:pd.DataFrame,outliers:pd.DataFrame):
        outlier_columns = []
        for row_i, row in outliers.iterrows():
            column_indices = np.where(row)[0]
            if len(column_indices) > 0:
                column_names = x.columns[column_indices].values
                column_names_str = " | ".join(column_names)
                outlier_columns.append(column_names_str)

        return outlier_columns

    def run_inner(self, x, y, metadata, name:str):
        outliers = self.detect_outliers(x)
        outlier_indices = outliers.any(axis=1)
        outliers_x = x[outlier_indices].copy()

        outlier_columns = self.get_outliers_column(x, outliers)
        outliers_x["outlier_columns"] = outlier_columns

        outliers_metadata = metadata[outlier_indices]
        outliers_x = pd.concat([outliers_x, outliers_metadata], axis=1)
        outliers_y = y[outlier_indices]
        outliers_x = pd.concat([outliers_x, outliers_y], axis=1)
        return outliers_x,outlier_indices

    def run(self):
        names = datasets.datasets_by_name_all
        for i,(name,dataset_module) in enumerate(names.items()):
            x,y,metadata = dataset_module.load(dropna=True)
            outliers,outlier_indices = self.run_inner(x, y, metadata, name)
            outliers.to_csv(self.plot_folderpath / f"{name}.csv")
            
            coefficients = dataset_module.coefficients
            systems = dataset_module.systems
            coefficients_np = np.array([coefficients[k] for k in x.columns])
            systems = [systems[k] for k in x.columns]
            for combination_size in [3,4]:
                q_np = qfeatures.calculate(x.to_numpy(), coefficients_np, x.columns, systems, combination_size=combination_size,by_system=True)
                q = pd.DataFrame(q_np.magnitudes, columns=q_np.column_names)
                outliers,outlier_indices = self.run_inner(q, y, metadata, name)
                outliers = pd.concat([x[outlier_indices],outliers],axis=1)
                outliers.to_csv(self.plot_folderpath / f"{name}_q{combination_size}.csv")






class OutlierDetectionTukey(OutlierDetection):
    def description(self) -> str:
        return "Detect outliers in each dataset using IQR based statistics"
    def detect_outliers(self,x:pd.DataFrame):
        out = outliers.detect_outliers_iqr(x,iqr_factor=3)
        return pd.DataFrame(out)


class OutlierDetectionNormalConfidenceInterval(OutlierDetection):
    def description(self) -> str:
        return "Detect outliers in each dataset using a Confidence interval based statistics"

    def detect_outliers(self,x:pd.DataFrame):
        # m = len(x.columns)  # number of columns = number of hypothesis
        # confidence = 0.99
        # adjusted_confidence = 1 - (1 - confidence) / m  # bonferroni-adjusted confidence
        # max_zscore = stats.norm.ppf(adjusted_confidence)

        # outliers = np.abs(stats.zscore(x - x.mean())) > max_zscore
        # return pd.DataFrame(outliers,columns=x.columns)
        out = outliers.detect_outliers_confidence_interval(x,confidence=0.999)
        return pd.DataFrame(out)