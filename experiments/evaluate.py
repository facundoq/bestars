from cgi import test
from pathlib import Path
from typing import Tuple

from experiments.base import DatabaseConfig, TrainConfig
from .common import *
import sklearn
from sklearn.utils.class_weight import compute_class_weight
from itertools import product
import gc
import pickle

def append_all(dict,dict_list):
    for k,v in dict.items():
        dict_list[k].append(v) 

class ClassifierExperiment(StarExperiment):

    def load_dataset(self,dataset_name:str,feature:Feature,class_feature_name:str):
        class_features = { "be":datasets.map_y_be,"em":datasets.map_y_em,}
        class_feature_loader = class_features[class_feature_name]
        dataset_module = datasets.datasets_by_name_all[dataset_name]
        logger.info(f"Loading dataset '{dataset_name}..'")
        x,y,metadata = dataset_module.load()
        
        logger.info(f"Calculating features {feature}..")        
        x_np = feature.calculate(x,dataset_name).to_numpy()
        y = class_feature_loader(y,dataset_name)
        y_np =y[class_feature_name].to_numpy()

        del x,y
        return x_np,y_np,metadata
    def load_split_dataset(self,db:DatabaseConfig,feature:Feature):
        x,y,metadata = self.load_dataset(db.id,feature,db.class_feature)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=1-db.train_size_percentage, random_state=0, stratify=y,shuffle=True)
        del x,y,metadata
        return x_train,x_test,y_train,y_test

    def plot_scores_train_test(self,id,train_scores,test_scores,x_label,x_values):
        subsets = ["train","test"]
        for i,scores in enumerate([train_scores,test_scores]):
            subset = subsets[i]
            results = {x_label:x_values}
            for v in scores.values():
                v[:] = np.round(v,2)
            results.update(scores)
            results = pd.DataFrame(results)
            print(f"{subset}")
            print(results)
            for label in scores.keys():
                f = plt.figure(dpi=200)
                ax = sn.barplot(x=x_label,y=label,data=results)
                ax.tick_params(labelsize=5)
                for container in ax.containers:
                    ax.bar_label(container)
                plt.ylim([0,1])
                plt.xticks(rotation=45)
                self.save_close_fig(f"{id}_{subset}_{label}.png")

    def evaluate(self,model,x, y):
        
        y_test_pred = model.predict(x)
        y_test_pred = y_test_pred>0.5
        f= sklearn.metrics.f1_score(y,y_test_pred)
        p = sklearn.metrics.precision_score(y,y_test_pred)
        r = sklearn.metrics.recall_score(y,y_test_pred)
        return {"fscore":f,"precision":p,"recall":r}

    def experiment(self,model_path:pathlib.Path,tc:TrainConfig):
        result_path = self.results_path() / f"{tc}.pkl"
        if result_path.exists():
            logging.info(f"Found results in {result_path}, loading..")
            with open(result_path,"rb") as f:
                train_scores,test_scores = pickle.load(f)
        else:
            x_train, x_test, y_train, y_test = self.load_split_dataset(tc.database,tc.feature)
            model = self.train(model_path,tc.model_config,x_train,y_train)
            logger.info(f"Evaluating {tc.model_config}")
            train_scores,test_scores = self.evaluate_model(model,x_train,y_train,x_test,y_test)
            del x_train, x_test, y_train, y_test
            del model
            gc.collect()
            keras.backend.clear_session()
            with open(result_path,"wb") as f:
                pickle.dump((train_scores,test_scores),f)
        return train_scores,test_scores

    def experiment_all(self,experiments_parameters:List[Tuple[Path,TrainConfig]]):
        train_scores_models = {"fscore":[],"precision":[],"recall":[]}
        test_scores_models =  {"fscore":[],"precision":[],"recall":[]}
        for model_path,tc in experiments_parameters:
            train_scores,test_scores = self.experiment(model_path,tc)
            append_all(train_scores,train_scores_models)
            append_all(test_scores,test_scores_models)   
        return train_scores_models,test_scores_models
    
    def train(self,model_path:str,model_config:ModelConfig,x_train:np.ndarray,y_train:np.ndarray):        
        if model_path.exists():
            logging.info(f"Trained model found in {model_path}, loading..")
            model = model_config.load(model_path)
        else:
            preprocessor = sklearn.preprocessing.StandardScaler().fit(x_train)
            x_train= preprocessor.transform(x_train)
            logger.info(f"Training {model_config}")
            class_unique = np.array([0,1])
            class_weight = compute_class_weight(class_weight='balanced',
                                                classes=class_unique,
                                                y = y_train)
            class_weight = dict(zip(class_unique, class_weight))
                                                
            logger.info(f"Using class weights: {class_weight}")
            # class_weight = {0: 7/8, 1: 1/8}
            
            model = model_config.generate(x_train.shape[1:])
            model.fit(x_train,y_train,class_weight=class_weight)
            model_config.save(model_path,model)
        return model
    
    def evaluate_model(self,model,x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray,y_test:np.ndarray):
        preprocessor = sklearn.preprocessing.StandardScaler().fit(x_train)
        x_train = preprocessor.transform(x_train)
        x_test = preprocessor.transform(x_test)
        train_scores = self.evaluate(model,x_train,y_train)
        test_scores = self.evaluate(model,x_test,y_test)
        return train_scores,test_scores

    def train_evaluate(self,model_path:str,model_config:ModelConfig,x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray,y_test:np.ndarray):
        model = self.train(model_path,model_config,x_train,y_train)

        train_scores,test_scores= self.evaluate_model(model,x_train, y_train, x_test, y_test)
        return model,train_scores,test_scores


class EvaluateClassifiers(ClassifierExperiment):
    def description(self) -> str:
        return "Compare classifier performance to determine if a sample has EM or is Be based on other features."

    def run(self):
        dataset_names = ["aidelman"] # "all_em",
        model_configs = [SmallNNConfig(), MediumNNConfig(),LargeNNConfig(), #LargeRandomForestConfig(),LargeGradientBoostingConfig()
        ]
        class_features = ["em","be"]
        train_set_percentage = 0.05
        
        for dataset_name, class_feature_name in  product(dataset_names,class_features):
            dataset_config =DatabaseConfig(dataset_name,train_set_percentage,class_feature_name)
            id = f"{dataset_name}_{class_feature_name}"
            logger.info(f"Experiment Case: {id}")
            parameters = []
            model_names = []
            for feature,model_config in product(common_features,model_configs):
                tc = TrainConfig(model_config,dataset_config,feature)
                model_path = self.models_path()/ f"{tc}.h5"
                model_names.append(f"{feature}\n{model_config}")
                parameters.append((model_path,tc))
            train_scores,test_scores = self.experiment_all(parameters)
            self.plot_scores_train_test(id,train_scores,test_scores,"Feature+Model",model_names)
                

# class EvaluateClassifiers2(ClassifierExperiment):
#     def description(self) -> str:
#         return "Compare classifier performance to determine if a sample has EM or is Be based on other features."

#     def run(self):
#         dataset_names = ["aidelman"] # "all_em",
#         model_configs = [SmallNNConfig(), MediumNNConfig(),LargeNNConfig(), #LargeRandomForestConfig(),LargeGradientBoostingConfig()
#         ]
#         class_features = ["em","be"]
#         train_set_percentage = 0.05
#         features_ids = ["mag","q3","q4"]
#         for dataset_name, class_feature_name in  product(dataset_names,class_features):
            
#             train_scores_all = {"fscore":[],"precision":[],"recall":[]}
#             test_scores_all =  {"fscore":[],"precision":[],"recall":[]}
            
#             id = f"{dataset_name}_{class_feature_name}"
#             logger.info(f"Experiment {id}")
#             model_names = []
#             for feature_id in features_ids:
#                 x,y,_ = self.load_dataset(dataset_name,feature_id,class_feature_name)    
#                 dataset_config =DatabaseConfig(dataset_name,train_set_percentage,class_feature_name)
#                 x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=1-train_set_percentage, random_state=0, stratify=y,shuffle=True)
#                 for model_config in model_configs:
#                     train_config = TrainConfig(model_config,dataset_config,feature_id)
#                     model_path = self.models_path() / f"{train_config}.h5"
#                     model_names.append(f"{model_config}+{feature_id}")
#                     model,train_scores,test_scores = self.train_evaluate(model_path,model_config,x_train, y_train,x_test, y_test)
#                     append_all(train_scores,train_scores_all)
#                     append_all(test_scores,test_scores_all)
#                 del x,y,x_train,x_test,y_train,y_test,model
#                 gc.collect()
#                 keras.backend.clear_session()
#             self.plot_scores_train_test(id,train_scores_all,test_scores_all,"Model",model_names)


class DetermineMinimumTrainingSet(ClassifierExperiment):
    def description(self) -> str:
        return "Train classifiers with different training set sizes to determine the minimum number of samples to achieve a decent f-score"

    def run(self):
        dataset_names = ["aidelman"] # "all_em",
        # models = {"mlp":self.mlp,"rf":self.rf,"gbc":self.gbc}
        model_configs = [MediumNNConfig()] #, RandomForest,GradientBoosting,MLP]
        class_features = ["em","be"]
        
        features_ids = ["mag","q3","q4"]
        train_set_percentages = [0.1*i for i in range(1,9)]
        for dataset_name, class_feature_name, feature_id,model_config in  product(dataset_names,class_features,features_ids,model_configs):
            id = f"{dataset_name}_{model_config}_{class_feature_name}_{feature_id}"
            logger.info(f"Experiment Case: {id}")
            parameters = []
            for train_set_percentage in train_set_percentages:
                dataset_config =DatabaseConfig(dataset_name,train_set_percentage,class_feature_name)
                train_config = TrainConfig(model_config,dataset_config,feature_id)
                model_path = self.models_path() / f"{train_config}.h5"
                parameters.append((model_path,train_config))
            train_scores,test_scores = self.experiment_all(parameters)
            self.plot_scores_train_test(id,train_scores,test_scores,"Training set %",train_set_percentages)

# class DetermineMinimumTrainingSet2(ClassifierExperiment):
#     def description(self) -> str:
#         return "Train classifiers with different training set sizes to determine the minimum number of samples to achieve a decent f-score"

#     def run(self):
#         dataset_names = ["aidelman"] # "all_em",
#         # models = {"mlp":self.mlp,"rf":self.rf,"gbc":self.gbc}
#         model_configs = [MediumNNConfig()] #, RandomForest,GradientBoosting,MLP]
#         class_features = ["em","be"]
        
#         features_ids = ["mag","q3","q4"]
#         train_set_percentages = [0.1*i for i in range(1,9)]
#         for dataset_name, class_feature_name, feature_id,model_config in  product(dataset_names,class_features,features_ids,model_configs):
#             id = f"{dataset_name}_{model_config}_{class_feature_name}_{feature_id}"
#             logger.info(f"Experiment {id}")
#             x,y,_ = self.load_dataset(dataset_name,feature_id,class_feature_name)
#             train_scores_all = {"fscore":[],"precision":[],"recall":[]}
#             test_scores_all =  {"fscore":[],"precision":[],"recall":[]}
#             for train_set_percentage in train_set_percentages:
#                 dataset_config =DatabaseConfig(dataset_name,train_set_percentage,class_feature_name)
#                 train_config = TrainConfig(model_config,dataset_config,feature_id)
#                 model_path = self.models_path() / f"{train_config}.h5"
#                 x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=1-train_set_percentage, random_state=0, stratify=y)
#                 model,train_scores,test_scores = self.train_evaluate(model_path,model_config,x_train, y_train,x_test, y_test)
#                 append_all(train_scores,train_scores_all)
#                 append_all(test_scores,test_scores_all)
#                 del x_train,x_test,y_train,y_test,model
#                 gc.collect()
#                 keras.backend.clear_session()
#             self.plot_scores_train_test(id,train_scores_all,test_scores_all,"Training set %",train_set_percentages)