from cgi import test
from .common import *
import abc
from scipy import stats
import itertools
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from itertools import product
import gc

def append_all(dict,dict_list):
    for k,v in dict.items():
        dict_list[k].append(v) 

class ClassifierExperiment(StarExperiment):

    def load_dataset(self,dataset_name,feature_id,class_feature_name):
        class_features = { "be":datasets.map_y_be,"em":datasets.map_y_em,}
        class_feature_loader = class_features[class_feature_name]
        dataset_module = datasets.datasets_by_name_all[dataset_name]
        print(f"Loading dataset '{dataset_name}'...",end="")
        x,y,metadata = dataset_module.load()
        print("done.")

        if not feature_id is  None;
            print("Calculating features..")
            x = calculate_features(x,dataset_name,feature_id)
            print("done.")

        y = class_feature_loader(y,dataset_name)
        y_np =y[class_feature_name].to_numpy()
        x_np = x.to_numpy()
        return x_np,y_np,metadata

    def plot(self,dataset_name,class_feature_name,train_scores,test_scores,x_label,x_values):
        subsets = ["train","test"]
        for i,scores in enumerate([train_scores,test_scores]):
            subset = subsets[i]
            results = {x_label:x_values}
            results.update(scores)
            results = pd.DataFrame(results)
            print("Data Frame:", results)
            for label in scores.keys():
                sn.barplot(x=x_label,y=label,data=results)
                self.save_close_fig(f"{dataset_name}_{class_feature_name}_{subset}_{label}.png")

    def evaluate(self,model,x, y):
        y_test_pred = model.predict(x)
        y_test_pred = y_test_pred>0.5
        f= sklearn.metrics.f1_score(y,y_test_pred)
        p = sklearn.metrics.precision_score(y,y_test_pred)
        r = sklearn.metrics.recall_score(y,y_test_pred)
        return {"fscore":f,"precision":p,"recall":r}

    def experiment(self,model_generator,dataset_name,class_name,feature_id,train_percent):
        x,y,metadata = self.load_dataset(dataset_name,feature_id,class_name)
        model = model_generator(x.shape[1:])
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=1-train_percent, random_state=0, stratify=y)
        train_scores,test_scores = self.train_evaluate(model,x_train, x_test,y_train, y_test)
        del x,y,metadata
        del x_train, x_test, y_train, y_test
        del model
        gc.collect
        return train_scores,test_scores

    def experiment_all(self,experiments_parameters):
        train_scores_models = {"fscore":[],"precision":[],"recall":[]}
        test_scores_models =  {"fscore":[],"precision":[],"recall":[]}
        for ep in experiments_parameters:
            train_scores,test_scores = self.experiment(*ep)
            append_all(train_scores,train_scores_models)
            append_all(test_scores,test_scores_models)   

    def train_evaluate(self,model,x_train, x_test, y_train, y_test):
        
        scaler = sklearn.preprocessing.StandardScaler().fit(x_train)
        x_train=scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        print(f"Training {model.__class__.__name__}...",end="")
        model.fit(x_train,y_train)
        print("done.")
        train_scores = self.evaluate(model,x_train,y_train)
        test_scores = self.evaluate(model,x_test,y_test)
        return train_scores,test_scores


from tensorflow import keras

def NeuralNetwork(input_shape):
    k = len(input_shape)
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Dense(128),
        keras.layers.Dense(64),
        keras.layers.Dense(32),
        keras.layers.Dense(1,activation="sigmoid"),
    ])
    print(model.summary())
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit2=model.fit
    def new_fit(x,y):
        model.fit2(x,y,epochs=3,batch_size=256)
    model.fit = new_fit
    return model

def RandomForest(input_shape):
    return RandomForestClassifier(max_depth=6, random_state=0)

def GradientBoosting(input_shape):
    return GradientBoostingClassifier(n_estimators=300, learning_rate=0.9,max_depth=3, random_state=0)
def MLP(input_shape):
    return MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes = (32,16,8), random_state = 1,max_iter=1000 )


class EvaluateClassifiers(ClassifierExperiment):
    def description(self) -> str:
        return "Compare classifier performance to determine if a sample has EM or is Be based on other features."

    def run(self):
        dataset_names = ["aidelman"] # "all_em",
        # models = {"mlp":self.mlp,"rf":self.rf,"gbc":self.gbc}
        model_generators = [NeuralNetwork, RandomForest,GradientBoosting,MLP]
        class_features = ["em","be"]
        model_names = [model.__class__.__name__ for model in model_generators]
        features_ids = [None,"q3","q4"]
        for dataset_name, class_feature_name, feature_id in  product(dataset_names,class_features,features_ids):
            x,y,metadata = self.load_dataset(dataset_name,feature_id,class_feature_name)
            train_scores_models = {"fscore":[],"precision":[],"recall":[]}
            test_scores_models =  {"fscore":[],"precision":[],"recall":[]}
            for model_generator in model_generators:
                model = model_generator(x.shape[1:])
                x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.20, random_state=0, stratify=y)
                train_scores,test_scores = self.train_evaluate(model,x_train, x_test,y_train, y_test)
                append_all(train_scores,train_scores_models)
                append_all(test_scores,test_scores_models)                
            self.plot(dataset_name,class_feature_name,train_scores_models,test_scores_models,"Model",model_names)
            

        
    
   


class DetermineMinimumTrainingSet(ClassifierExperiment):
    def description(self) -> str:
        return "Train classifiers with different training set sizes to determine the minimum number of samples to achieve a decent f-score"

    def run(self):
        dataset_names = ["aidelman"] # "all_em",
        # models = {"mlp":self.mlp,"rf":self.rf,"gbc":self.gbc}
        model_generators = [NeuralNetwork] #, RandomForest,GradientBoosting,MLP]
        class_features = ["em","be"]
        model_names = [model.__class__.__name__ for model in model_generators]
        features_ids = [None] #,"q3","q4"]
        training_set_sizes = [0.1*i for i in range(1,9)]
        for dataset_name, class_feature_name, feature_id,model_generator in  product(dataset_names,class_features,features_ids,model_generators):
            x,y,metadata = self.load_dataset(dataset_name,feature_id,class_feature_name)
            train_scores_models = {"fscore":[],"precision":[],"recall":[]}
            test_scores_models =  {"fscore":[],"precision":[],"recall":[]}
            for training_set_sizes in training_set_sizes:
                model = model_generator(x.shape[1:])
                x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=1-training_set_sizes, random_state=0, stratify=y)
                train_scores,test_scores = self.train_evaluate(model,x_train, x_test,y_train, y_test)
                append_all(train_scores,train_scores_models)
                append_all(test_scores,test_scores_models)                
            self.plot(dataset_name,class_feature_name,train_scores_models,test_scores_models,"Training set %",training_set_sizes)