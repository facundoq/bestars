from sklearn import svm
from sklearn.metrics import classification_report
from old import preprocess

x,y,class_names,id= preprocess.load_data(binary=True)

clf = svm.LinearSVC(max_iter=5000)
clf.fit(x,y)

y_pred = clf.predict(x)

report=classification_report(y,y_pred , target_names=class_names)
print(report)