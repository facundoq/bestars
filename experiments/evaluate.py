from .common import *
import abc
from scipy import stats
import itertools
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier


class SKLearnClassifiers(StarExperiment):
    def description(self) -> str:
        return "Test simple classifiers to determine if a sample has EM or is Be based on other features."
    def plot_results(self,dataset_name,model_names,scores):
        results = pd.DataFrame({"Model":model_names,"Score":scores})
        print("Data Frame:", results)
        sn.barplot(x="Model",y="Score",data=results)
        self.save_close_fig(f"{dataset_name}_scores.png")

    def run(self):
        dataset_names = ["aidelman"]
        # models = {"mlp":self.mlp,"rf":self.rf,"gbc":self.gbc}
        models = [RandomForestClassifier(max_depth=3, random_state=0)
                  ,MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes = (10, 10), random_state = 1,max_iter=1000 )
                  ,GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0)
                 ]
        model_names = [model.__class__.__name__ for model in models]
        for dataset_name in dataset_names:
            dataset_module = datasets.datasets_by_name_all[dataset_name]
            print(f"Loading dataset '{dataset_name}'...",end="")
            x,y,metadata = dataset_module.load()
            print("done.")
            y = datasets.map_y_em(y,dataset_name)
            y_np =y["em"].to_numpy()
            x_np = x.to_numpy()
            scores = []
            for model in models:
                x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_np, y_np, test_size=0.20, random_state=0, stratify=y)
                score = self.evaluate(model,x_train, x_test,y_train, y_test)
                scores.append(score)    
            self.plot_results(dataset_name,model_names,scores)

    def evaluate(self,model,x_train, x_test, y_train, y_test):
        
        scaler = sklearn.preprocessing.StandardScaler().fit(x_train)
        x_train=scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        print(f"Training {model.__class__.__name__}...",end="")
        model.fit(x_train,y_train)
        print("done.")
        y_test_pred = model.predict(x_test)
        return sklearn.metrics.f1_score(y_test,y_test_pred)
