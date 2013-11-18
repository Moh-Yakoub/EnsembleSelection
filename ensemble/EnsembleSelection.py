from EnsembleUtils import auc_error
from sklearn.metrics import auc
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from operator import add
import random
import logging
import itertools
from collections import Counter

class EnsembleSelection:
	'''
	This class implements the EnsembleSelection mechanism
	'''
	'''
	All of the metrics used for classification model selection
	'''
	classification_error_metrics = {
	'auc':auc_error,
	'average_precision':average_precision_score,
	'f1':f1_score,
	'fbeta':fbeta_score,
	'accuracy':accuracy_score,
	'roc_auc':roc_auc_score,
	'roc_curve':roc_curve,
	'jaccard_similarity':jaccard_similarity_score
	}

	'''
	Those are a set of simple classifiers that need zero or trivial params.
	'''
	simple_classifiers = {
	'GaussianNB' : GaussianNB()
	}

	def __init__(self):
		FORMAT = '%(asctime)s  %(message)s'
		logging.basicConfig(format=FORMAT)
		self.logger = logging.getLogger('ensemble-selection-logger')
		self.logger.setLevel(logging.INFO)
	




	'''
	a method that generates a set of logistic regression models based on a different set of parameters
	params:
	n = the number of models generated
	'''

	'''
	TODO @yakoub : See a possible way to generate a 'neat' set of 'seeded' classifiers of any length
	right now we stick to the old trick of randomizing everything
	'''

	def generate_logistic_regression_classifiers(self,n=12):
		self.logger.info('Generating a set of  logistic regression classifiers, n = %d',n)

		models = []
		penalty_vars = ['l1','l2']
		C_vars = [0.1,0.2,1.0]
		fit_vars = [True,False]
		parameters_list = itertools.product(penalty_vars,C_vars,fit_vars)

		for (penalty,C,fit) in parameters_list:
			models.append(LogisticRegression(penalty=penalty,C=C,fit_intercept = fit))

		self.logger.info('Generating a set of  logistic regression classifiers was created successfully!',n)
		return models[0:n]

	def generate_multionomial_nb_classifiers(self,n=2):
		self.logger.info('Generating a set of Multinomial naive bayes classifiers , n = %d',n)
		models = []
		alpha_vars = [0.01,0.1]
		fit_prior_vars = [True,False]
		parameters_list = itertools.product(alpha_vars,fit_prior_vars)
		for (alpha,fit) in parameters_list:
			models.append(MultinomialNB(alpha=alpha,fit_prior=fit))

		self.logger.info('Generating a set of  Multinomial naive bayes classifiers was created successfully!')
		return models[0:n]

	def generate_bernoulli_nb_classifiers(self,n=8):
		self.logger.info('Generating a set of  Bernoulli naive bayes classifiers , n = %d',n)

		models = []
		alpha_vars = [0.01,0.1]
		fit_prior_vars = [True,False]
		binarize_vars = [True,False]
		parameters_list = itertools.product(alpha_vars,fit_prior_vars,binarize_vars)

		for (alpha,fit,binarize) in parameters_list:
			models.append(BernoulliNB(alpha=alpha,fit_prior=fit,binarize = binarize))

		self.logger.info('Generating a set of  Bernoulli naive bayes classifiers was created successfully!',n)
		return models[0:n]

	'''
	The implementation of the ensemble selection,
	params
	models : a list of all models that will for the ensemble
	n : the number of models in the ensemble
	replacement : whether the selection will be with replacement or not
	X: the set of variables
	Y: The target variable
	error_metric: the error metric used: roc-auc by default
	'''
	def form_ensemble(self,X,Y,models,n,replacement=False,error_metric_name='roc_auc'):
		if replacement==False and len(models) <n :
			self.logger.error("the ensemble size must smaller than the models size when not using replacement")
		elif not (error_metric_name) in self.classification_error_metrics:
			self.logger.error("unrecognized error metric:"+error_metric_name)
		else:
			self.logger.info("Forming Ensemble")
			error_metric = self.classification_error_metrics[error_metric_name]
			X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y, test_size=0.4, random_state=0)
			ensemble = []
			## for the model i in ensemble , ensemble_predictions[i] stores the predictions of the model fit
			model_predictions = []
			##stores the last aggregated sum of all of the previous model in the ensemble
			predictions_list = []
			##adding a trivial large number
			min_error = pow(2,10)
			best_index = 0
			for i in range(0,n):
				self.logger.info("Training model  : %d of input models",i)
				model = models[i]
				model.fit(X_train,y_train)
				self.logger.info("Predicting using  model : %d of input models",i)
				predicted = model.predict(X_test)
				model_predictions.append(predicted)
				model_error = error_metric(predicted,y_test)
				if(model_error < min_error):
					min_error = model_error
					predictions_list .append(predicted.tolist())
					best_index = i
			self.logger.info("Training and prediction finished!")
			self.logger.info("Ensemble formation using models")
			##check the best model
			ensemble.append(models[best_index])
			self.logger.info("Adding first model to ensemble")
			min_error = pow(2,10)
			if not replacement:
				model_predictions.pop(best_index)
				models.pop(best_index)
			##using vote aggregation
			for i in range(0,n-1):
				for j in range(0,len(model_predictions)):
					aggregated_predictions = self.vote(predictions_list+[model_predictions[j].tolist()])
					model_error = error_metric(y_test,aggregated_predictions)
					if(model_error < min_error):
						min_error = model_error
						best_index = j

				self.logger.info("Adding the model no. %d to ensemble",i+1)
				predictions_list.append(model_predictions[best_index])
				min_error = pow(2,10)
				ensemble.append(models[best_index])
				if not replacement:
					model_predictions.pop(best_index)
					models.pop(best_index)
			return ensemble

	def vote(self,l):
		n = len(l[0])
		result = []
		for i in range(0,n):
			ithlist = [x[i] for x in l]
			result.append(Counter(ithlist).most_common(1)[0][0])

		return result
