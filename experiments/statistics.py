import itertools
from pathlib import Path


from .common import *
import abc
from scipy import stats

def load_aidelman_raw():
    str_cols = ['ID', 'SpT', 'Type', 'EM', 'Be', 'obsid', 'objtype', 'class', 'subclass', 'B-TS1', 'B-TS2',
       'B-TS', 'EM1', 
       'GroupID', 'GroupSize']
    str_cols2 = ["EM1",  "EMobj",  "BeC1",  "EM2",  "Be_EM2",  "BeC2",  "BeC"]
    types = {k:str for k in str_cols+str_cols2}
    path = Path("data/Concatenadas_sinRepes_dropnan_LAMOST.csv")
    logging.info(f"Loading dataset from {path}...")
    df = pd.read_csv(path,dtype=types)
    n,m=df.shape
    logging.info(f"Done ({n} rows, {m} columns).")
    return df

class ClassDistributionComparison(StarExperiment):
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
            y.hist(ax=ax,density=True)
            ax.set_xlabel(name)
            if i==0:
                ax.set_ylabel("Samples")
        self.save_close_fig("histograms")

class ClassFeaturesDistribution(StarExperiment):
    def description(self) -> str:
        return "Plot histograms of distribution of classes for each dataset"

    def run(self):
        dataset_names = ["aidelman"]
        class_features = {"em":datasets.map_y_em,"be":datasets.map_y_be}
        
        for dataset_name in dataset_names:
            dataset_module = datasets.datasets_by_name[dataset_name]
            n_class_features = len(class_features)
            f,axes=plt.subplots(1,n_class_features,sharey=True,sharex=True)
            x,y,metadata = dataset_module.load()
            for i,(class_feature_name,class_feature_function) in enumerate(class_features.items()):
                y_feature = class_feature_function(y,dataset_name)            
                ax = axes[i]
                # y_feature.hist(ax=ax)
                sn.countplot(x=class_feature_name,ax=ax,data=y_feature)
                for container in ax.containers:
                    ax.bar_label(container)
                ax.set_xlabel(class_feature_name)
                ax.set_ylabel("Samples")
            self.save_close_fig(f"{dataset_name}_distribution")

class CategoricalFeaturesDistribution(StarExperiment):
    def description(self) -> str:
        return "Plot histograms of distribution of classes for each dataset"

    def run(self):
        dataset_name = "aidelman"
        features = ["EM","Be","objtype","class","type"]

        df = load_aidelman_raw()
        df = df.astype("string")
        df = df[features]

        # print(df)
        # print(df["EM"].dtype)
        df.fillna("?",inplace=True)

        # df.plot(kind='pie', subplots=True,
        #  autopct='%1.1f%%', startangle=270,# fontsize=17,figsize=(10,10)
        #   )
        colors = sn.color_palette(palette="pastel")
        n_features = len(features)
        f,axes=plt.subplots(1,n_features,squeeze=False,dpi=250)

        for i,feature_name in enumerate(features):
            ax = axes[0,i]
            counts=df[feature_name].value_counts(ascending=True,dropna=False)

            labels = counts.index.to_list()
            values = counts.to_numpy()
            values = values/values.sum()
            threshold=0.05
            indices = values>threshold
            others = sum(values[values<threshold])
            values = values[indices]
            indices =np.where(indices).tolist()
            print(indices)
            labels = [ labels[i] for i in indices]
            labels.append("Others")
            values = np.append(values,others)
            
            labels = [f"{l}: {v*100:.1f}" for l,v in zip(labels,values)]

            patches,texts = ax.pie(values, colors = colors, startangle=90)
            ax.set_title(feature_name)
            plt.legend(patches,labels,ncol=4,loc="best",fontsize=5)
            # ax.legend(fontsize=7, title_fontsize=10)
            # df.plot.pie(y=feature_name,ax=ax)
            # ax.set_xlabel(class_feature_name)
            # ax.set_ylabel("Samples")
            

        self.save_close_fig(f"{dataset_name}_distribution")



class MissingValues(StarExperiment):
    def description(self) -> str:
        return "Plot missing values in datasets"
    def select_missing(self,df):
        missing_values = df.isnull().sum() / len(df)*100
        missing_values = missing_values[missing_values > 0]
        # missing_values.sort_values(inplace=True)
        missing_values
        missing_values = missing_values.to_frame()
        missing_values.columns = ['count']
        missing_values.index.names = ['Name']
        missing_values['Name'] = missing_values.index
        return missing_values

    def run(self):
        dataset_name = "aidelman"
        df = load_aidelman_raw()
        n,m = df.shape
        missing_values = self.select_missing(df)
        plt.figure(dpi=200)
        ax = sn.barplot(x = 'Name', y = 'count', data=missing_values)
        ax.set(xlabel='Feature', ylabel='Missing values (%)')
        plt.xticks(rotation = 60)
        plt.title(f"Features with missing values. \nDataset with {n} samples and {m} features")
        self.save_close_fig(f"{dataset_name}_missing")

class FeatureDistributions(StarExperiment):
    def description(self) -> str:
        return "Plot histograms of distribution of features, also divided by class, for each dataset"

    def run(self):
        dataset_names = ["aidelman"]
     
        class_features = ["Type","EM","Be"]

        features = [Magnitude(),QFeature(3,True),QFeature(4,True)]
        for dataset_name in dataset_names:
            dataset_module = datasets.datasets_by_name[dataset_name]
            
            x,y,metadata = dataset_module.load(fillna_classes=False)

            y[y==1.0]="Yes"
            y[y==0.0]="No"
            
            y = pd.concat([y,metadata],axis=1)
            y.fillna("Missing",inplace=True)

            for feature in features:
                print(f"{dataset_name}_{feature}")
                x_feature = feature.calculate(x,dataset_name)

                plt.figure(figsize=(10,10))
                sn.boxplot(data=x_feature)
                self.save_close_fig(f"{dataset_name}_distribution")

                self.plot_by_variable(x_feature,y,dataset_name,class_features,feature)
                del x_feature

    def plot_by_variable(self,x,y,dataset_name,class_features,feature_id):
        for i,class_feature_name in enumerate(class_features):
            values = y[class_feature_name].unique()
            
            print(class_feature_name,values)
            values.sort()
            n_values = len(values)
            
            f,axes = plt.subplots(1,n_values,figsize=(n_values*10,1*10),sharey=True,squeeze=False,dpi=50)
            
            for i,value in enumerate(values):
                ax=axes[0,i]
                x_value = x.loc[y[class_feature_name]==value]
                ax.set_title(value)
                ax = sn.boxplot(data=x_value,ax=ax)
                labels = [l.get_text().replace("mag","") for l in ax.get_xticklabels()]
                ax.set_xticklabels(labels)

            plt.suptitle(class_feature_name)
            self.save_close_fig(f"{dataset_name}_{class_feature_name}_{feature_id}.pdf")

class ReducedQ(StarExperiment):
    def description(self) -> str:
        return "Find reduced structure in Q features with blocks"

    def run(self):
        
        def generate_combination_matrix(n_features,n_magnitudes):
            combination_matrix = np.zeros(n_features,n_magnitudes)
            for i,c in enumerate(itertools.combinations(n_features,n_magnitudes)):
                combination_matrix[i,c]=1
            return combination_matrix

        n_features = 5
        q = 3
        features = list(range(n_features))
        full_combinations = list(itertools.combinations(features,3))
        full_combination_matrix = generate_combination_matrix(full_combinations,n_features)
        
    def analyze_matrix(self,matrix:np.ndarray):
        n_features,n = matrix.shape
        epsilon = np.finfo(matrix.dtype).eps
        print(epsilon)
        sn.heatmap(matrix)
        self.save_close_fig(f"q{q}_n{n_features}_full")
        subsets_combinations = list(itertools.combinations(n_features,n_features))
        for i,subset_combinations in enumerate(subsets_combinations):
            subset_combination_matrix = generate_combination_matrix(subset_combinations,n_features)
            condition_number = np.linalg.cond(subset_combination_matrix)
            invertible = condition_number<1/epsilon
            invertible_str = "_sin" if not invertible else "_inv"
            sn.heatmap(subset_combination_matrix)
            subset_combinations_str = ",".join([str(s) for s in subset_combinations])
            plt.title(f"Invertible: {invertible}, condition number: {condition_number}\n{subset_combinations_str}")

            self.save_close_fig(f"q{q}_n{n_features}{invertible_str}_reduced{i:03}")

def plot_corr(df,size=10):
    """Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    """

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)

class CorrelationMatrix(StarExperiment):
    def description(self) -> str:
        return "Plot and store the correlation matrix for all feature types"

    def run(self):
        dataset_names = ["aidelman"]
        
        
        features = [Magnitude(),QFeature(3,True),QFeature(4,True)]
        for dataset_name in dataset_names:
            dataset_module = datasets.datasets_by_name[dataset_name]
            x,_,_ = dataset_module.load(fillna_classes=False)

            for feature in features:
                print(f"{dataset_name}_{feature}")
                x_feature = feature.calculate(x,dataset_name)
                correlation_matrix = x_feature.corr()
                del x_feature
                correlation_matrix.to_pickle(self.folderpath / f"{dataset_name}_{feature}.pkl")
                plot_corr(correlation_matrix)
                self.save_close_fig(f"{dataset_name}_{feature}_correlation")
                

                


class FeatureCorrelations(StarExperiment):
    def description(self) -> str:
        return "Plot histograms of distribution of classes for each dataset"

    def run(self):
        dataset_names = ["aidelman"]
        
        
        for dataset_name in dataset_names:
            dataset_module = datasets.datasets_by_name[dataset_name]
            x,y,metadata = dataset_module.load()
            n_class_features = len(class_features)
            f,axes=plt.subplots(1,n_class_features,sharey=True,sharex=True)
            for i,(class_feature_name,class_feature_function) in enumerate(class_features.items()):
                x,y,metadata = dataset_module.load()
                y = class_feature_function(y,dataset_name)            
                ax = axes[i]
                y.hist(ax=ax)
                ax.set_xlabel(class_feature_name)
                if i==0:
                    ax.set_ylabel("Samples")
            self.save_close_fig(f"{dataset_name}_distribution")

class OutlierDetection(StarExperiment):

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
            outliers.to_csv(self.folderpath / f"{name}.csv")
            
            coefficients = dataset_module.coefficients
            systems = dataset_module.systems
            coefficients_np = np.array([coefficients[k] for k in x.columns])
            systems = [systems[k] for k in x.columns]
            for combination_size in [3,4]:
                q_np = qfeatures.calculate(x.to_numpy(), coefficients_np, x.columns, systems, combination_size=combination_size,by_system=True)
                q = pd.DataFrame(q_np.magnitudes, columns=q_np.column_names)
                outliers,outlier_indices = self.run_inner(q, y, metadata, name)
                outliers = pd.concat([x[outlier_indices],outliers],axis=1)
                outliers.to_csv(self.folderpath / f"{name}_q{combination_size}.csv")






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