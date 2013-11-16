from ensemble.EnsembleSelection import EnsembleSelection
from ensemble.EnsembleClassifier import EnsembleClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn import datasets


if __name__ == "__main__":
	iris = datasets.load_iris()
	X = iris.data
	Y = iris.target
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y, test_size=0.4, random_state=0)		
	selection = EnsembleSelection()
	clf = EnsembleClassifier()
	models = []
	models = (selection.generate_logistic_regression_classifiers(300))
	ensemble =  selection.form_ensemble(X_train,y_train,models,50,error_metric_name='accuracy',replacement=True)
	y_predicted = clf.predict(X_test,ensemble)
	print accuracy_score(y_predicted,y_test)


