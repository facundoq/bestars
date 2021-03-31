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
        return "Select pairs of features that are good for classifying EM vs no EM objects."


    def eval_combinations(self, x, y, column_names, dataset_name, feature):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.20,
                                                                                    random_state=0, stratify=y)
        n, d = x.shape
        column_indices = list(range(d))
        paired_indices = list(itertools.combinations(column_indices, 2))
        kbest = KBestScores(k=10)
        if self.verbose:
            print(f"Evaluating {len(paired_indices)} binary features combinations:")
        paired_column_names = ["a", "b", "f1"]
        output_df = pd.DataFrame(columns=paired_column_names)
        for pair in paired_indices:
            x_train_pair = x_train[:, pair]
            x_test_pair = x_test[:, pair]
            score = self.evaluate_model(x_train_pair, x_test_pair, y_train, y_test)

            kbest.update(score, pair)
            qa_label, qb_label = [column_names[c] for c in pair]
            qdf = pd.DataFrame([[qa_label, qb_label, score]], columns=paired_column_names)
            output_df = pd.concat([output_df, qdf], axis=0, ignore_index=True)
        output_df.sort_values(by="f1", ascending=False, inplace=True)
        output_df.to_csv(self.plot_folderpath / f"accuracies_{dataset_name}_{feature}.csv")
        k = len(kbest.scores)

        f, axes = plt.subplots(1, k, figsize=(2 * k, 2), dpi=150)
        f.tight_layout()
        for i in range(k):
            pair, score = kbest.values[i], kbest.scores[i]
            x_pair = x[:, pair]
            scatter=axes[i].scatter(x_pair[:, 0], x_pair[:, 1], c=y)
            legend = axes[i].legend(*scatter.legend_elements(),
                                loc="upper right", title="EM", fontsize=4, title_fontsize=4)
            axes[i].add_artist(legend)

            xlabel, ylabel = [column_names[c] for c in pair]
            axes[i].set_xlabel(xlabel, fontsize=5)
            axes[i].set_ylabel(ylabel, fontsize=5)
            axes[i].set_title(f"F = {score:.2f}", fontsize=7)
            axes[i].tick_params(axis='both', which='major', labelsize=5)

        plt.savefig(self.plot_folderpath / f"best_{dataset_name}_{feature}.png")

    def run(self):
        dataset_name = "all_em"

        dataset_module = datasets.datasets_by_name_all[dataset_name]
        x,y,metadata = dataset_module.load()
        y_np =y["em"].to_numpy()
        x_np = x.to_numpy()
        self.eval_combinations(x_np, y_np, x.columns, dataset_name, "original")

        coefficients = dataset_module.coefficients
        systems = [dataset_module.systems[k] for k in x.columns]
        for combination_size in [3,4]:
            feature = f"q{combination_size}"
            coefficients_np = np.array([coefficients[k] for k in x.columns])
            q = qfeatures.calculate(x_np, coefficients_np, x.columns, systems, combination_size=combination_size)
            self.eval_combinations(q.magnitudes,y_np,q.column_names,dataset_name,feature)



    def evaluate_model(self,x_train, x_test, y_train, y_test):
        scaler = sklearn.preprocessing.StandardScaler().fit(x_train)
        x_train=scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        clf = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes = (6, 3), random_state = 1,max_iter=1000f)
        clf.fit(x_train,y_train)
        y_test_pred = clf.predict(x_test)
        return sklearn.metrics.f1_score(y_test,y_test_pred)

