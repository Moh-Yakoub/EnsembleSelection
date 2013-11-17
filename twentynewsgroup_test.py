from ensemble.EnsembleSelection import EnsembleSelection
from ensemble.EnsembleClassifier import EnsembleClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn import cross_validation
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier

if __name__ == "__main__":
	newsgroups_train = fetch_20newsgroups(subset='train')
	vectorizer = TfidfVectorizer()
	train_vectors = vectorizer.fit_transform(newsgroups_train.data)
	X_TRAIN , X_TEST , Y_TRAIN , Y_TEST = cross_validation.train_test_split(train_vectors,newsgroups_train.target, test_size=0.1, random_state=0)
	selection = EnsembleSelection()
	models = [SGDClassifier(loss="hinge", penalty="l2")]+[LinearSVC()]+selection.generate_logistic_regression_classifiers(1)+selection.generate_multionomial_nb_classifiers(1)+selection.generate_bernoulli_nb_classifiers(1)
	ensemble =  selection.form_ensemble(X_TRAIN,Y_TRAIN,models,4,error_metric_name='f1')
	clf = EnsembleClassifier()
	y_predicted = clf.predict(X_TEST,ensemble)
	print f1_score(y_predicted,Y_TEST)
	






