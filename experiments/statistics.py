import itertools
from pathlib import Path

from experiments.base import ColorFeatures, SubsetFeature, UnionFeature


from .common import *

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
                ax = axes[0,i]
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
        features = ["EM","Be","objtype","class","Type","EMobj"]

        df = load_aidelman_raw()
        df = df.astype("string")
        df = df[features]
        df.replace( [1.0,1,0.0,0],["Yes","Yes","No","No"],inplace=True)
        df.fillna("Missing",inplace=True)
        colors = sn.color_palette(palette=None)#"pastel")
        min_percentage=0.0001
        max_values = 7
        for i,feature_name in enumerate(features):
            f,axes=plt.subplots(1,1,squeeze=False,dpi=250)
            ax = axes[0,0]
            feature = df[feature_name].copy()
            feature[feature==1]="Yes"
            feature[feature==0]="No"
            feature[feature==1.0]="Yes"
            feature[feature==0.0]="No"
            counts=feature.value_counts(ascending=False,dropna=False,normalize=True)
            i = 0
            for v,c in zip(counts.index,counts.values):
                i+=1
                if c<min_percentage or i>max_values:
                    # print(f"removing {v} ({c},{i})")
                    feature.iloc[feature==v]="Other"
            value_counts = feature.value_counts(normalize=True,dropna=False)
            labels = [f"{l}: {v*100:.2f}%" for l,v in zip(value_counts.index,value_counts.values)]

            patches,texts = ax.pie(value_counts.values, colors = colors, startangle=90)
            ax.set_title(feature_name)
            legend = plt.legend(patches,labels,ncol=4,loc="lower center",fontsize=8)
            # ax.legend(fontsize=7, title_fontsize=10)

            self.save_close_fig(f"{dataset_name}_{feature_name}_distribution",extra_artists=[legend])



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
        return "Plot boxplots of distribution of features, also divided by class, for each dataset"

    def run(self):
        dataset_names = ["aidelman"]
        em_name = "EM"
        b_name = "Tipo espectral B"
        em_old_name = "EM(old)"
        class_features = ["Type","EM","Be",b_name,em_name]
        standard_colors = ColorFeatures([("u","g"), ("r","i"), ("r", "Ha")])
        magnitudes_colors = UnionFeature([Magnitude(),standard_colors])
        q3_system = QFeature(3,True)
        q4 = QFeature(4,True)
        q4_poster = QFeature(4,False,["u","g","r","i","Ha"])
        q4_poster_subset = SubsetFeature(q4_poster,["ugri","ugrHa"])
        poster = UnionFeature([Magnitude(),standard_colors,q4_poster_subset])
        features = [ Magnitude(),#q3,q4,
                     standard_colors,
                     q4_poster_subset,
                     q3_system,
                     poster
                    ]
                    
        for dataset_name in dataset_names:
            dataset_module = datasets.datasets_by_name[dataset_name]
            
            x,y,metadata = dataset_module.load(fillna_classes=False)
            y = pd.concat([y,metadata],axis=1)

            y.rename(columns={"B-TS":b_name},inplace=True)
            y.rename(columns={"EM":em_old_name},inplace=True)
            y.rename(columns={"EMobj":em_name},inplace=True)
            

            yes = "Si"
            no = "No"
            y.replace( [1.0,1,"1",0.0,0,"0"],[yes,yes,yes,no,no,no],inplace=True)
            
            y.fillna("Faltante",inplace=True)
            

            for feature in features:
                print(f"{dataset_name}_{feature}")
                x_feature = feature.calculate(x,dataset_name)
                plt.figure(figsize=(10,10))
                sn.boxplot(data=x_feature)
                self.save_close_fig(f"{dataset_name}_distribution")
                self.plot_by_class_feature(x_feature,y,dataset_name,class_features,feature)
                del x_feature
    def filter_values_by_count(self,class_feature:pd.DataFrame,max_values:int):
        value_counts = class_feature.value_counts(normalize=True,sort=True)
        n_values = len(value_counts)
        if max_values<n_values:
            remaining_values = value_counts.iloc[max_values:]
            for i,v in zip(remaining_values.index,remaining_values.values):
                class_feature[class_feature==i]="Other"
        return class_feature

    def plot_by_class_feature(self,x:pd.DataFrame,y:pd.DataFrame,dataset_name:str,class_features:List[str],feature:Feature,max_values=4):
        for i,class_feature_name in enumerate(class_features):
            class_feature = y[class_feature_name].copy()
            class_feature = self.filter_values_by_count(class_feature,max_values=max_values)
            value_counts = class_feature.value_counts(normalize=True,sort=True)
            n_values = len(value_counts)
            f,axes = plt.subplots(1,n_values,figsize=(n_values*10,1*10),sharey=True,squeeze=False,dpi=50)
            # Sort by name
            names,values = value_counts.index,value_counts.values
            indices = np.argsort(names)
            names = names[indices]
            values = values[indices]
            for i,(name,value) in enumerate(zip(names,values)):
                ax=axes[0,i]
                x_value = x.loc[class_feature==name]
                ax.set_title(f"{name} ({value*100:.1f}%)",fontsize=24)
                ax = sn.boxplot(data=x_value,ax=ax)
                ax.tick_params(axis='both', which='major', labelsize=18)


            plt.suptitle(class_feature_name,fontsize=36)
            self.save_close_fig(f"{dataset_name}_{class_feature_name}_{feature}.png")

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
                

                


# class FeatureCorrelations(StarExperiment):
#     def description(self) -> str:
#         return "Plot histograms of distribution of classes for each dataset"

#     def run(self):
#         dataset_names = ["aidelman"]
        
        
#         for dataset_name in dataset_names:
#             dataset_module = datasets.datasets_by_name[dataset_name]
#             x,y,metadata = dataset_module.load()
#             n_class_features = len(class_features)
#             f,axes=plt.subplots(1,n_class_features,sharey=True,sharex=True)
#             for i,(class_feature_name,class_feature_function) in enumerate(class_features.items()):
#                 x,y,metadata = dataset_module.load()
#                 y = class_feature_function(y,dataset_name)            
#                 ax = axes[i]
#                 y.hist(ax=ax)
#                 ax.set_xlabel(class_feature_name)
#                 if i==0:
#                     ax.set_ylabel("Samples")
#             self.save_close_fig(f"{dataset_name}_distribution")

