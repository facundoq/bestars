from .common import *
import abc
from scipy import stats
import itertools
import sklearn
from sklearn.neural_network import MLPClassifier

plt.style.use('ggplot')

class KBestScores:
    def __init__(self,k=10):
        self.k=10
        self.scores=[]
        self.values=[]
    def update(self,score,value):
        if len(self.scores)<self.k:
            self.scores.append(score)
            self.values.append(value)
        else:
            m =min(self.scores)
            if score>m:
                i = self.scores.index(m)
                del self.scores[i]
                del self.values[i]
                self.scores.append(score)
                self.values.append(value)


class BinaryFeatureSelection(Experiment):
    def description(self) -> str:
        return "Plot histograms of distribution of classes for each dataset"


    def run(self):
        name = "all_em"
        feature="q4"
        dataset_module = datasets.datasets_by_name_all[name]
        x,y,metadata = dataset_module.load()
        coefficients = dataset_module.coefficients
        systems = dataset_module.systems
        coefficients_np = np.array([coefficients[k] for k in x.columns])
        systems = [systems[k] for k in x.columns]
        combination_size=4
        q = qfeatures.calculate(x.to_numpy(), coefficients_np, x.columns, systems, combination_size=combination_size)
        x_np = q.magnitudes
        y_np =y["em"].to_numpy()


        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_np, y_np, test_size = 0.20, random_state = 0,stratify=y_np)
        n,d = x_np.shape
        column_indices = list(range(d))
        paired_indices = list(itertools.combinations(column_indices, 2))
        kbest = KBestScores(k=10)
        if self.verbose:
            print(f"Evaluating {len(paired_indices)} binary features combinations:")
        column_names = ["q_a","q_b","f1"]
        output_df = pd.DataFrame(columns=column_names)
        for pair in paired_indices:
            x_train_pair = x_train[:,pair]
            x_test_pair = x_test[:,pair]
            score = self.evaluate_model(x_train_pair, x_test_pair, y_train, y_test)

            kbest.update(score,pair)
            qa_label,qb_label = [q.column_names[c] for c in pair]
            qdf = pd.DataFrame([[qa_label,qb_label,score]],columns=column_names)
            output_df=pd.concat([output_df,qdf],axis=0,ignore_index=True)
        output_df.sort_values(by="f1",ascending=False,inplace=True)
        output_df.to_csv(self.plot_folderpath/ f"accuracies_{name}_{feature}.csv")
        k = len(kbest.scores)

        f, axes = plt.subplots(1, k,figsize=(2*k,2),dpi=150)
        f.tight_layout()
        for i in range(k):
            pair,score = kbest.values[i],kbest.scores[i]
            x_pair=x_np[:,pair]
            axes[i].scatter(x_pair[:,0],x_pair[:,1],c=y_np)
            xlabel,ylabel = [q.column_names[c] for c in pair]
            axes[i].set_xlabel(xlabel,fontsize=5)
            axes[i].set_ylabel(ylabel,fontsize=5)
            axes[i].set_title(f"F = {score:.2f}", fontsize=7)
            axes[i].tick_params(axis='both', which='major', labelsize=5)
        plt.savefig(self.plot_folderpath/ f"best_{name}_{feature}.png")

    def evaluate_model(self,x_train, x_test, y_train, y_test):
        scaler = sklearn.preprocessing.StandardScaler().fit(x_train)
        x_train=scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        clf = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes = (6, 3), random_state = 1,max_iter=1000)
        clf.fit(x_train,y_train)
        y_test_pred = clf.predict(x_test)
        return sklearn.metrics.f1_score(y_test,y_test_pred)

