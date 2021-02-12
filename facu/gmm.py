from sklearn.metrics import classification_report,f1_score,precision_recall_curve,precision_score
from sklearn.mixture import GaussianMixture
from facu import preprocess
import numpy as np
import matplotlib.pyplot as plt

train,test,class_names = preprocess.load_data(binary=True,split=0.4)

x,y,id=train

x_ob = x[y==1,:]

model = GaussianMixture(n_components=2,random_state=0)
model.fit(x_ob)


def calculate_probabilities(model,x):
    #scores = model.score_samples(x)
    component_probabilities= model.predict_proba(x)
    weighted_probabilities = component_probabilities*model.weights_[np.newaxis,:]
    probabilities = np.sum(weighted_probabilities,axis=1)
    return probabilities

probabilities=calculate_probabilities(model,x)
plt.plot(probabilities)
plt.savefig("plots/gmm_probs.png")
plt.close()

# find best threshold
def best_threshold(probabilities,y,num_tresholds=20,metric_function=f1_score):
    thresholds= np.linspace(0,1,num_tresholds)
    best_metric=0
    best_t=-1
    for t in thresholds:
        y_pred=probabilities<t
        metric = metric_function(y,y_pred)
        if best_metric<metric:
            best_metric=metric
            best_t=t
    return best_t,best_metric

def plot_roc(probabilities, y, filepath:str):
    precision, recall, thresholds = precision_recall_curve(y,probabilities)
    plt.plot(precision,recall)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.savefig(filepath)
    plt.close()
    #thresholds= np.linspace(0,1,num_tresholds)
    # p=np.zeros(num_tresholds)
    # r=np.zeros(num_tresholds)
    # for t in thresholds:



best_t, best_metric = best_threshold(probabilities,y,metric_function=precision_score)
print(f"Best threshold: {best_t}, metric: {best_metric}")
y_train_pred = probabilities < best_t
report = classification_report(y,y_train_pred , target_names=class_names)
print("Training set")
print(report)

candidates = np.vstack([id,probabilities]).T

np.savetxt("gmm_candidates.csv",candidates,fmt='%.2f',delimiter=",",header="id,probability")

x_test,y_test,id_test=test
probabilities=calculate_probabilities(model,x_test)

y_test_pred = probabilities < best_t
report = classification_report(y_test,y_test_pred , target_names=class_names)
print("Test set")
print(report)

plot_roc(y_test,y_test_pred,"plots/gmm_pr_curve.png")

